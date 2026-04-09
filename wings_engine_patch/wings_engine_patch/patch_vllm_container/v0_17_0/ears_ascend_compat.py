import functools
import importlib
import inspect
import textwrap
from dataclasses import dataclass


_ASCEND_DRAFT_COMPAT_MODULES = (
    "vllm_ascend.ascend_forward_context",
    "vllm_ascend.compilation.acl_graph",
    "vllm_ascend.attention.mla_v1",
    "vllm_ascend.spec_decode.draft_proposer",
    "vllm_ascend.spec_decode.eagle_proposer",
)
_ASCEND_VLLM_COMPAT_MODULES = (
    "vllm.v1.worker.gpu.spec_decode.eagle",
)
_PATCHED_ATTR = "_wings_ears_ascend_draft_compat_patched"
_SUPPORTED_EARS_METHODS = ("eagle3", "mtp", "suffix")


@dataclass
class _CompatGraphParams:
    events: dict
    workspaces: dict
    handles: dict
    attn_params: dict


def _extend_unique(values, additions):
    items = list(values)
    for addition in additions:
        if addition not in items:
            items.append(addition)
    return type(values)(items) if isinstance(values, tuple) else items


def _align_hidden_states_last_dim(hidden_states, hidden_size):
    if hidden_size is None or not hasattr(hidden_states, "shape") or hidden_states.shape[-1] == hidden_size:
        return hidden_states

    if hidden_states.shape[-1] > hidden_size:
        return hidden_states[..., :hidden_size]

    import torch.nn.functional as functional

    return functional.pad(hidden_states, (0, hidden_size - hidden_states.shape[-1]))


def _get_target_hidden_size(instance):
    hidden_states = getattr(instance, "hidden_states", None)
    if hasattr(hidden_states, "shape"):
        return hidden_states.shape[-1]
    return getattr(instance, "hidden_size", None)


def _patch_ascend_forward_context_module(module) -> None:
    extra_ctx = getattr(module, "_EXTRA_CTX", None)
    extra_attrs = getattr(extra_ctx, "extra_attrs", None)
    if extra_ctx is None or extra_attrs is None:
        return
    attrs_holder = type(extra_ctx) if hasattr(type(extra_ctx), "extra_attrs") else extra_ctx
    current_attrs = getattr(attrs_holder, "extra_attrs", extra_attrs)
    updated_attrs = _extend_unique(current_attrs, ("draft_attn_metadatas",))
    if updated_attrs != current_attrs:
        setattr(attrs_holder, "extra_attrs", updated_attrs)


def _patch_acl_graph_module(module) -> None:
    graph_params_cls = getattr(module, "GraphParams", _CompatGraphParams)
    if not hasattr(module, "_draft_graph_params"):
        module._draft_graph_params = None  # pylint: disable=protected-access

    def set_draft_graph_params(aclgraph_capture_sizes):
        if module._draft_graph_params is not None:  # pylint: disable=protected-access
            raise ValueError("DraftGraph parameters have already been set!")
        module._draft_graph_params = graph_params_cls(  # pylint: disable=protected-access
            {size: [] for size in aclgraph_capture_sizes},
            {size: None for size in aclgraph_capture_sizes},
            {size: [] for size in aclgraph_capture_sizes},
            {size: [] for size in aclgraph_capture_sizes},
        )

    def update_draft_graph_params_workspaces(num_tokens, workspace):
        if module._draft_graph_params is not None:  # pylint: disable=protected-access
            module._draft_graph_params.workspaces[num_tokens] = workspace  # pylint: disable=protected-access

    def get_draft_graph_params():
        return module._draft_graph_params  # pylint: disable=protected-access

    module.__dict__.setdefault("set_draft_graph_params", set_draft_graph_params)
    module.__dict__.setdefault("update_draft_graph_params_workspaces", update_draft_graph_params_workspaces)
    module.__dict__.setdefault("get_draft_graph_params", get_draft_graph_params)


def _patch_mla_v1_module(module) -> None:
    for attr_name in ("SUPPORTED_SPECULATIVE_METHODS", "SUPPORTED_DRAFT_METHODS"):
        if not hasattr(module, attr_name):
            continue
        current_value = getattr(module, attr_name)
        updated_value = _extend_unique(current_value, _SUPPORTED_EARS_METHODS)
        if updated_value != current_value:
            setattr(module, attr_name, updated_value)
        return
    module.SUPPORTED_SPECULATIVE_METHODS = _SUPPORTED_EARS_METHODS


def _patch_vllm_worker_eagle_module(module) -> None:
    try:
        speculator_module = importlib.import_module(f"{module.__name__}.speculator")
    except ModuleNotFoundError:
        speculator_module = None
    if speculator_module is not None:
        for attr_name in ("EagleSpeculator", "EagleCudaGraphManager"):
            attr_value = getattr(speculator_module, attr_name, None)
            if attr_value is not None:
                module.__dict__.setdefault(attr_name, attr_value)

    def prepare_eagle_inputs(
        input_buffers,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
    ):
        import torch

        num_reqs = input_batch.num_reqs
        last_token_indices = torch.empty(num_reqs, dtype=torch.int64, device=num_sampled.device)
        for batch_idx in range(num_reqs):
            req_state_idx = int(input_batch.idx_mapping[batch_idx])
            query_start = int(input_batch.query_start_loc[batch_idx])
            query_end = int(input_batch.query_start_loc[batch_idx + 1])
            query_len = query_end - query_start - int(num_rejected[batch_idx])
            if query_len <= 0:
                raise ValueError("Query length must stay positive after rejected tokens are removed.")

            if int(num_sampled[batch_idx]) > 0:
                next_token = last_sampled[req_state_idx]
            else:
                next_token = next_prefill_tokens[req_state_idx]

            if query_len > 1:
                input_buffers.input_ids[query_start : query_start + query_len - 1] = input_batch.input_ids[
                    query_start + 1 : query_start + query_len
                ]

            last_token_index = query_start + query_len - 1
            last_token_indices[batch_idx] = last_token_index
            input_buffers.input_ids[last_token_index] = next_token.to(dtype=input_buffers.input_ids.dtype)
            input_buffers.positions[query_start : query_start + query_len] = input_batch.positions[
                query_start : query_start + query_len
            ]
        return last_token_indices

    def prepare_eagle_decode(
        draft_tokens,
        output_hidden_states,
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_buffers,
        input_hidden_states,
        max_model_len,
        max_num_reqs,
    ):
        import torch

        num_reqs = int(draft_tokens.shape[0])
        device = input_buffers.query_start_loc.device
        input_buffers.query_start_loc[: num_reqs + 1] = torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
        if max_num_reqs + 1 > num_reqs + 1:
            input_buffers.query_start_loc[num_reqs + 1 : max_num_reqs + 1] = num_reqs
        if max_num_reqs > num_reqs:
            input_buffers.seq_lens[num_reqs:max_num_reqs] = 0

        for req_idx in range(num_reqs):
            src_idx = int(last_token_indices[req_idx])
            input_buffers.input_ids[req_idx] = draft_tokens[req_idx].to(dtype=input_buffers.input_ids.dtype)
            input_hidden_states[req_idx] = output_hidden_states[src_idx]

            next_position = min(int(input_buffers.positions[src_idx]) + 1, max_model_len - 1)
            input_buffers.positions[req_idx] = next_position

            seq_len = int(target_seq_lens[req_idx]) - int(num_rejected[req_idx]) + 1
            input_buffers.seq_lens[req_idx] = min(seq_len, max_model_len)

    module.__dict__.setdefault("prepare_eagle_inputs", prepare_eagle_inputs)
    module.__dict__.setdefault("prepare_eagle_decode", prepare_eagle_decode)


def _patch_draft_proposer_module(module) -> None:
    cls = getattr(module, "AscendDraftModelProposer", None)
    if cls is None:
        return

    original_init = cls.__init__
    if getattr(original_init, "_wings_parallel_spec_patched", False):
        return

    def patched_init(self, vllm_config, device, runner=None):
        original_init(self, vllm_config, device, runner=runner)
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        if spec_cfg is not None:
            draft_cfg = getattr(spec_cfg, "draft_model_config", None)
            if draft_cfg is not None:
                draft_max = getattr(draft_cfg, "max_model_len", None)
                if draft_max is not None and draft_max < self.max_model_len:
                    self.max_model_len = draft_max

    patched_init._wings_parallel_spec_patched = True  # pylint: disable=protected-access
    cls.__init__ = patched_init


def _patch_eagle_proposer_module(module) -> None:
    cls = getattr(module, "SpecDecodeBaseProposer", None)
    if cls is None:
        return

    original_set_inputs_first_pass = getattr(cls, "set_inputs_first_pass", None)
    if (
        original_set_inputs_first_pass is not None
        and not getattr(original_set_inputs_first_pass, "_wings_parallel_spec_patched", False)
    ):
        set_inputs_signature = inspect.signature(original_set_inputs_first_pass)

        def patched_set_inputs_first_pass(self, *args, **kwargs):
            bound = set_inputs_signature.bind(self, *args, **kwargs)
            if "target_hidden_states" in bound.arguments:
                bound.arguments["target_hidden_states"] = _align_hidden_states_last_dim(
                    bound.arguments["target_hidden_states"],
                    _get_target_hidden_size(self),
                )
            return original_set_inputs_first_pass(*bound.args, **bound.kwargs)

        patched_set_inputs_first_pass._wings_parallel_spec_patched = True  # pylint: disable=protected-access
        cls.set_inputs_first_pass = patched_set_inputs_first_pass

    original_method = getattr(cls, "_run_merged_draft", None)
    if original_method is None:
        return
    if getattr(original_method, "_wings_parallel_spec_patched", False):
        return

    try:
        source = inspect.getsource(original_method)
    except (OSError, TypeError):
        return
    if "self.vllm_config.model_config.max_model_len" not in source:
        return

    patched_source = textwrap.dedent(source).replace(
        "self.vllm_config.model_config.max_model_len",
        "self.max_model_len",
    )
    if patched_source == textwrap.dedent(source):
        return

    local_namespace = {}
    exec(patched_source, original_method.__globals__, local_namespace)  # pylint: disable=exec-used
    patched_run_merged_draft = local_namespace.get(original_method.__name__)
    if patched_run_merged_draft is None:
        return

    patched_run_merged_draft = functools.update_wrapper(patched_run_merged_draft, original_method)
    patched_run_merged_draft._wings_parallel_spec_patched = True  # pylint: disable=protected-access
    cls._run_merged_draft = patched_run_merged_draft


_MODULE_PATCHERS = {
    "vllm_ascend.ascend_forward_context": _patch_ascend_forward_context_module,
    "vllm_ascend.compilation.acl_graph": _patch_acl_graph_module,
    "vllm_ascend.attention.mla_v1": _patch_mla_v1_module,
    "vllm.v1.worker.gpu.spec_decode.eagle": _patch_vllm_worker_eagle_module,
    "vllm_ascend.spec_decode.draft_proposer": _patch_draft_proposer_module,
    "vllm_ascend.spec_decode.eagle_proposer": _patch_eagle_proposer_module,
}


def patch_vllm_ascend_draft_compat(module) -> None:
    module_name = getattr(module, "__name__", None)
    patcher = _MODULE_PATCHERS.get(module_name)
    if patcher is None or getattr(module, _PATCHED_ATTR, False):
        return
    patcher(module)
    setattr(module, _PATCHED_ATTR, True)
