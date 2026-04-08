from dataclasses import dataclass


_ASCEND_DRAFT_COMPAT_MODULES = (
    "vllm_ascend.ascend_forward_context",
    "vllm_ascend.compilation.acl_graph",
    "vllm_ascend.attention.mla_v1",
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


def _patch_ascend_forward_context_module(module) -> None:
    extra_ctx = getattr(module, "_EXTRA_CTX", None)
    extra_attrs = getattr(extra_ctx, "extra_attrs", None)
    if extra_ctx is None or extra_attrs is None:
        return
    updated_attrs = _extend_unique(extra_attrs, ("draft_attn_metadatas",))
    if updated_attrs != extra_attrs:
        extra_ctx.extra_attrs = updated_attrs


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


_MODULE_PATCHERS = {
    "vllm_ascend.ascend_forward_context": _patch_ascend_forward_context_module,
    "vllm_ascend.compilation.acl_graph": _patch_acl_graph_module,
    "vllm_ascend.attention.mla_v1": _patch_mla_v1_module,
}


def patch_vllm_ascend_draft_compat(module) -> None:
    module_name = getattr(module, "__name__", None)
    patcher = _MODULE_PATCHERS.get(module_name)
    if patcher is None or getattr(module, _PATCHED_ATTR, False):
        return
    patcher(module)
    setattr(module, _PATCHED_ATTR, True)
