import copy
import inspect
import logging
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResolvedSpeculativeTokenSettings:
    num_speculative_tokens: int | None
    speculative_token_range: list[int] | None
    draft_confidence_threshold: float


_ADAPTIVE_DRAFT_SUPPORTED_METHODS = {"draft_model", "eagle3"}


class AdaptiveDraftLengthController:
    def __init__(
        self,
        allowed_lengths: list[int],
        initial_length: int,
        *,
        acceptance_ewma: float = 0.6,
        alpha: float = 0.2,
    ):
        if not allowed_lengths:
            raise ValueError("allowed_lengths must not be empty.")
        if not _is_strictly_increasing(allowed_lengths):
            raise ValueError("allowed_lengths must be strictly increasing.")
        if initial_length not in allowed_lengths:
            raise ValueError("initial_length must appear in allowed_lengths.")
        self.allowed_lengths = allowed_lengths
        self.current_length = initial_length
        self.acceptance_ewma = acceptance_ewma
        self.alpha = alpha

    def observe_iteration(
        self,
        *,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> int:
        if num_draft_tokens < 0 or num_accepted_tokens < 0:
            raise ValueError("draft and accepted tokens must be non-negative.")
        if num_accepted_tokens > num_draft_tokens:
            raise ValueError("accepted tokens cannot exceed drafted tokens.")
        if num_draft_tokens == 0:
            return self.current_length

        batch_acceptance_rate = num_accepted_tokens / num_draft_tokens
        self.acceptance_ewma = (
            (1.0 - self.alpha) * self.acceptance_ewma
            + self.alpha * batch_acceptance_rate
        )
        current_index = self.allowed_lengths.index(self.current_length)
        if self.acceptance_ewma >= 0.75 and current_index < len(self.allowed_lengths) - 1:
            self.current_length = self.allowed_lengths[current_index + 1]
        elif self.acceptance_ewma <= 0.45 and current_index > 0:
            self.current_length = self.allowed_lengths[current_index - 1]
        return self.current_length


class _StderrProxy:
    @staticmethod
    def write(message: str) -> int:
        return sys.stderr.write(message)  # pylint: disable=logging-not-lazy

    @staticmethod
    def flush() -> None:
        sys.stderr.flush()


LOGGER = logging.getLogger("wings_accel.adaptive_draft_model")


def _configure_logger() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(_StderrProxy())
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _is_strictly_increasing(values: list[int]) -> bool:
    for left, right in zip(values, values[1:]):
        if left >= right:
            return False
    return True


def _supports_adaptive_draft_length(method: str | None) -> bool:
    return method in _ADAPTIVE_DRAFT_SUPPORTED_METHODS


_configure_logger()


def log_runtime_state(event: str, /, **fields) -> None:
    parts = [f"{key}={value}" for key, value in sorted(fields.items())]
    suffix = f" {' '.join(parts)}" if parts else ""
    LOGGER.info("[wins-accel] %s%s", event, suffix)


def resolve_speculative_token_settings(
    *,
    method: str | None,
    num_speculative_tokens: int | None,
    speculative_token_range: list[int] | None,
    draft_confidence_threshold: float | None = None,
) -> ResolvedSpeculativeTokenSettings:
    if draft_confidence_threshold is None:
        draft_confidence_threshold = 0.0
    if not 0.0 <= float(draft_confidence_threshold) <= 1.0:
        raise ValueError("draft_confidence_threshold must be between 0.0 and 1.0.")
    if (
        float(draft_confidence_threshold) > 0.0
        and method is not None
        and method != "draft_model"
    ):
        raise ValueError(
            "draft_confidence_threshold is only supported for draft_model."
        )

    if speculative_token_range is None:
        return ResolvedSpeculativeTokenSettings(
            num_speculative_tokens=num_speculative_tokens,
            speculative_token_range=None,
            draft_confidence_threshold=float(draft_confidence_threshold),
        )

    if method is None:
        method = "draft_model"
    if not _supports_adaptive_draft_length(method):
        supported_methods = ", ".join(sorted(_ADAPTIVE_DRAFT_SUPPORTED_METHODS))
        raise ValueError(
            "speculative_token_range is only supported for "
            f"{supported_methods}."
        )
    if num_speculative_tokens is None:
        raise ValueError(
            f"{method} requires num_speculative_tokens when "
            "speculative_token_range is provided."
        )
    if not speculative_token_range:
        raise ValueError("speculative_token_range must not be empty.")
    if any(
        not isinstance(value, int) or value <= 0 for value in speculative_token_range
    ):
        raise ValueError(
            "speculative_token_range must contain only positive integers."
        )
    if len(set(speculative_token_range)) != len(speculative_token_range):
        raise ValueError("speculative_token_range must not contain duplicates.")
    if not _is_strictly_increasing(speculative_token_range):
        raise ValueError(
            "speculative_token_range must be strictly increasing as supplied."
        )
    if speculative_token_range[-1] > num_speculative_tokens:
        raise ValueError(
            "speculative_token_range must not contain values greater than "
            "num_speculative_tokens."
        )
    if num_speculative_tokens not in speculative_token_range:
        raise ValueError(
            "num_speculative_tokens must appear in speculative_token_range."
        )
    return ResolvedSpeculativeTokenSettings(
        num_speculative_tokens=num_speculative_tokens,
        speculative_token_range=speculative_token_range,
        draft_confidence_threshold=float(draft_confidence_threshold),
    )


def apply_confidence_threshold_to_draft_tokens(
    *,
    draft_token_ids,
    draft_token_confidences,
    draft_confidence_threshold: float,
):
    if draft_confidence_threshold <= 0.0:
        return draft_token_ids

    if draft_token_ids.ndim != 2 or draft_token_confidences.ndim != 2:
        raise ValueError("draft_token_ids and draft_token_confidences must be rank-2.")
    if draft_token_ids.shape != draft_token_confidences.shape:
        raise ValueError(
            "draft_token_ids and draft_token_confidences must have the same shape."
        )

    low_confidence_mask = draft_token_confidences < draft_confidence_threshold
    seen_low_confidence = low_confidence_mask.cumsum(dim=1) > 0
    pad_mask = torch.cat(
        [
            torch.zeros(
                (draft_token_ids.shape[0], 1),
                dtype=torch.bool,
                device=draft_token_ids.device,
            ),
            seen_low_confidence[:, :-1],
        ],
        dim=1,
    )
    return draft_token_ids.masked_fill(pad_mask, -1)


def trim_trailing_invalid_draft_tokens(
    draft_token_ids: list[list[int]],
) -> list[list[int]]:
    trimmed_rows: list[list[int]] = []
    for row in draft_token_ids:
        last_valid_index = -1
        for index, token_id in enumerate(row):
            if token_id >= 0:
                last_valid_index = index
        trimmed_rows.append(row[: last_valid_index + 1] if last_valid_index >= 0 else [])
    return trimmed_rows


def _resolve_num_draft_tokens_for_controller(
    instance,
    valid_sampled_token_count: list[int],
) -> int:
    get_draft_token_ids_cpu = getattr(instance, "_get_draft_token_ids_cpu", None)
    if callable(get_draft_token_ids_cpu):
        draft_token_ids, _ = get_draft_token_ids_cpu()
        if draft_token_ids:
            return sum(
                sum(1 for token_id in row if token_id >= 0) for row in draft_token_ids
            )
    return (getattr(instance, "draft_length", 0) or 0) * len(valid_sampled_token_count)


def _call_member(target, member_name: str, *args, **kwargs):
    return getattr(target, member_name)(*args, **kwargs)


def _increment_optional_tensor_attribute(target, attribute_name: str, value) -> None:
    current_value = getattr(target, attribute_name, None)
    if current_value is not None:
        current_value += value.to(current_value.dtype)


def _clear_attribute(target, attribute_name: str) -> None:
    setattr(target, attribute_name, None)


def _call_original_class_method(raw_method, original_method, instance, *args, **kwargs):
    if isinstance(raw_method, staticmethod):
        return original_method(*args, **kwargs)
    return original_method(instance, *args, **kwargs)


def _call_with_optional_draft_length(propose_method, *, draft_length: int | None, **kwargs):
    signature = inspect.signature(propose_method)
    supports_draft_length = "draft_length" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if supports_draft_length and draft_length is not None:
        kwargs["draft_length"] = draft_length
    return propose_method(**kwargs)


def _should_preserve_uniform_decode_query_len(instance) -> bool:
    compilation_config = getattr(instance, "compilation_config", None)
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    if cudagraph_mode is None:
        return False

    decode_mode = getattr(cudagraph_mode, "decode_mode", None)
    if not callable(decode_mode):
        return False

    decoded_mode = decode_mode()
    none_mode = getattr(type(cudagraph_mode), "NONE", None)
    return decoded_mode != none_mode


def _resolve_uniform_decode_query_len(instance, spec_config) -> int:
    num_spec_tokens = getattr(instance, "num_spec_tokens", None)
    if num_spec_tokens is None:
        num_spec_tokens = getattr(spec_config, "num_speculative_tokens", 0) or 0
    if _should_preserve_uniform_decode_query_len(instance):
        return 1 + num_spec_tokens
    return 1


@dataclass
class ProposeInputs:
    target_token_ids: object
    target_positions: object
    target_hidden_states: object
    next_token_ids: object
    token_indices_to_sample: object
    common_attn_metadata: object
    sampling_metadata: object
    mm_embed_inputs: object = None
    num_rejected_tokens_gpu: object = None
    slot_mappings: object = None
    draft_length: int | None = None


@dataclass
class DraftIterationState:
    batch_size: int
    input_batch_size: int
    batch_size_across_dp: int
    common_attn_metadata: object
    per_layer_attn_metadata: dict[str, object]
    positions: object
    hidden_states: object
    continue_mask: object
    draft_token_ids_list: list[object]
    pad_token_id: int
    cudagraph_runtime_mode: object


@dataclass
class GPUModelRunnerProposeInputs:
    scheduler_output: object
    sampled_token_ids: object
    sampling_metadata: object
    hidden_states: object
    sample_hidden_states: object
    aux_hidden_states: object
    spec_decode_metadata: object
    common_attn_metadata: object
    slot_mappings: object


def _bind_propose_inputs(original_propose, instance, args, kwargs) -> ProposeInputs:
    signature = inspect.signature(original_propose)
    bind_kwargs = dict(kwargs)
    explicit_draft_length = bind_kwargs.get("draft_length")
    if "draft_length" not in signature.parameters:
        bind_kwargs.pop("draft_length", None)
    bound_arguments = signature.bind(instance, *args, **bind_kwargs)
    bound_arguments.apply_defaults()
    arguments = bound_arguments.arguments
    return ProposeInputs(
        target_token_ids=arguments["target_token_ids"],
        target_positions=arguments["target_positions"],
        target_hidden_states=arguments["target_hidden_states"],
        next_token_ids=arguments["next_token_ids"],
        token_indices_to_sample=arguments["token_indices_to_sample"],
        common_attn_metadata=arguments["common_attn_metadata"],
        sampling_metadata=arguments["sampling_metadata"],
        mm_embed_inputs=arguments.get("mm_embed_inputs"),
        num_rejected_tokens_gpu=arguments.get("num_rejected_tokens_gpu"),
        slot_mappings=arguments.get("slot_mappings"),
        draft_length=arguments.get("draft_length", explicit_draft_length),
    )


def _bind_gpu_model_runner_propose_inputs(
    raw_method,
    original_method,
    instance,
    args,
    kwargs,
) -> GPUModelRunnerProposeInputs:
    signature = inspect.signature(original_method)
    if isinstance(raw_method, staticmethod):
        bound_arguments = signature.bind(*args, **kwargs)
    else:
        bound_arguments = signature.bind(instance, *args, **kwargs)
    bound_arguments.apply_defaults()
    arguments = bound_arguments.arguments
    return GPUModelRunnerProposeInputs(
        scheduler_output=arguments["scheduler_output"],
        sampled_token_ids=arguments["sampled_token_ids"],
        sampling_metadata=arguments["sampling_metadata"],
        hidden_states=arguments["hidden_states"],
        sample_hidden_states=arguments["sample_hidden_states"],
        aux_hidden_states=arguments["aux_hidden_states"],
        spec_decode_metadata=arguments["spec_decode_metadata"],
        common_attn_metadata=arguments["common_attn_metadata"],
        slot_mappings=arguments["slot_mappings"],
    )


def _call_original_propose(
    original_propose,
    *,
    supports_draft_length: bool,
    instance,
    inputs: ProposeInputs,
):
    kwargs = {
        "mm_embed_inputs": inputs.mm_embed_inputs,
        "num_rejected_tokens_gpu": inputs.num_rejected_tokens_gpu,
        "slot_mappings": inputs.slot_mappings,
    }
    if supports_draft_length:
        kwargs["draft_length"] = inputs.draft_length
    original_num_speculative_tokens = None
    if not supports_draft_length and inputs.draft_length is not None:
        original_num_speculative_tokens = getattr(instance, "num_speculative_tokens", None)
        instance.num_speculative_tokens = inputs.draft_length
    try:
        return original_propose(
            instance,
            inputs.target_token_ids,
            inputs.target_positions,
            inputs.target_hidden_states,
            inputs.next_token_ids,
            inputs.token_indices_to_sample,
            inputs.common_attn_metadata,
            inputs.sampling_metadata,
            **kwargs,
        )
    finally:
        if original_num_speculative_tokens is not None:
            instance.num_speculative_tokens = original_num_speculative_tokens


def _should_use_confidence_filter(instance, confidence_threshold: float) -> bool:
    return (
        getattr(instance, "method", None) == "draft_model"
        and confidence_threshold > 0.0
        and not getattr(instance, "parallel_drafting", False)
        and not getattr(instance, "use_local_argmax_reduction", False)
    )


def _should_use_padded_adaptive_draft(
    instance,
    *,
    draft_length: int | None,
    confidence_threshold: float,
) -> bool:
    if draft_length is None:
        return False
    return (
        _supports_adaptive_draft_length(getattr(instance, "method", None))
        and draft_length < getattr(instance, "num_speculative_tokens", 0)
        and not getattr(instance, "parallel_drafting", False)
        and not getattr(instance, "use_local_argmax_reduction", False)
        and confidence_threshold >= 0.0
    )


def _require_runner(instance):
    runner = getattr(instance, "runner", None)
    if runner is None:
        raise RuntimeError(
            "SpecDecodeBaseProposer.runner must be initialized before propose()."
        )
    return runner


def _build_per_layer_attn_metadata(instance, common_attn_metadata):
    per_layer_attn_metadata: dict[str, object] = {}
    attn_metadata = None
    for attn_group in instance.draft_attn_groups:
        attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
            common_attn_metadata=common_attn_metadata,
            draft_index=0,
        )
        for layer_name in attn_group.layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
    return per_layer_attn_metadata, attn_metadata


def _build_model_inputs(instance, num_input_tokens: int, mm_embed_inputs):
    if instance.supports_mm_inputs:
        mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)
        instance.inputs_embeds[:num_input_tokens] = instance.model.embed_input_ids(
            instance.input_ids[:num_input_tokens],
            multimodal_embeddings=mm_embeds,
            is_multimodal=is_mm_embed,
        )
        return None, instance.inputs_embeds[:num_input_tokens]
    return instance.input_ids[:num_input_tokens], None


def _run_model_forward(
    instance,
    module,
    *,
    per_layer_attn_metadata: dict[str, object],
    common_attn_metadata,
    num_input_tokens: int,
    num_tokens_across_dp,
    cudagraph_runtime_mode,
    input_ids,
    inputs_embeds,
):
    model_kwargs = {
        "input_ids": input_ids,
        "positions": _call_member(instance, "_get_positions", num_input_tokens),
        "inputs_embeds": inputs_embeds,
    }
    if instance.pass_hidden_states_to_model:
        model_kwargs["hidden_states"] = instance.hidden_states[:num_input_tokens]

    with module.set_forward_context(
        per_layer_attn_metadata,
        instance.vllm_config,
        num_tokens=num_input_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        slot_mapping=_call_member(
            instance,
            "_get_slot_mapping",
            num_input_tokens,
            common_attn_metadata.slot_mapping,
        ),
    ):
        ret_hidden_states = instance.model(**model_kwargs)
        if not instance.model_returns_tuple():
            return ret_hidden_states, ret_hidden_states
        return ret_hidden_states


def _extract_sampling_inputs(
    instance,
    *,
    last_hidden_states,
    hidden_states,
    token_indices_to_sample,
):
    sample_hidden_states = last_hidden_states[token_indices_to_sample]
    if instance.uses_mrope:
        positions = instance.mrope_positions[:, token_indices_to_sample]
    else:
        positions = instance.positions[token_indices_to_sample]
    return sample_hidden_states, positions, hidden_states[token_indices_to_sample]


def _resolve_pad_token_id(speculative_config) -> int:
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    pad_token_id = getattr(hf_config, "pad_token_id", None)
    if pad_token_id is None:
        return 0
    return pad_token_id


def _prepare_first_draft_iteration_state(
    instance,
    module,
    *,
    common_attn_metadata,
    per_layer_attn_metadata,
    positions,
    hidden_states,
    draft_token_ids,
    num_rejected_tokens_gpu,
    pad_token_id: int,
) -> DraftIterationState:
    cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = _call_member(
        instance,
        "_determine_batch_execution_and_padding",
        common_attn_metadata.batch_size(),
    )
    batch_size = common_attn_metadata.batch_size()
    common_attn_metadata.num_actual_tokens = batch_size
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.query_start_loc = instance.arange[: batch_size + 1]
    common_attn_metadata.query_start_loc_cpu = module.torch.from_numpy(
        instance.token_arange_np[: batch_size + 1]
    ).clone()

    if instance.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
        common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
        _clear_attribute(common_attn_metadata, "_seq_lens_cpu")
        _clear_attribute(common_attn_metadata, "_num_computed_tokens_cpu")

    continue_mask = module.torch.ones(
        instance.vllm_config.scheduler_config.max_num_batched_tokens,
        dtype=module.torch.bool,
        device=instance.device,
    )
    continue_mask[:batch_size] = True
    return DraftIterationState(
        batch_size=batch_size,
        input_batch_size=input_batch_size,
        batch_size_across_dp=batch_size_across_dp,
        common_attn_metadata=common_attn_metadata,
        per_layer_attn_metadata=per_layer_attn_metadata,
        positions=positions,
        hidden_states=hidden_states,
        continue_mask=continue_mask,
        draft_token_ids_list=[draft_token_ids],
        pad_token_id=pad_token_id,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
    )


def _advance_positions(instance, module, positions):
    positions += 1
    if instance.uses_mrope:
        exceeds_max_model_len = positions[0] >= instance.max_model_len
        clamped_positions = module.torch.where(
            exceeds_max_model_len.unsqueeze(0),
            module.torch.zeros_like(positions),
            positions,
        )
    else:
        exceeds_max_model_len = positions >= instance.max_model_len
        clamped_positions = module.torch.where(
            exceeds_max_model_len,
            0,
            positions,
        )
    return positions, exceeds_max_model_len, clamped_positions


def _update_common_attn_seq_lens(
    common_attn_metadata,
    *,
    active_mask,
    exceeds_max_model_len,
    max_model_len: int,
) -> None:
    length_increments = active_mask.to(common_attn_metadata.seq_lens.dtype)
    common_attn_metadata.seq_lens += length_increments
    common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
    common_attn_metadata.max_seq_len = min(
        common_attn_metadata.max_seq_len + 1,
        max_model_len,
    )

    active_mask_cpu = active_mask.cpu()
    _increment_optional_tensor_attribute(
        common_attn_metadata,
        "_seq_lens_cpu",
        active_mask_cpu,
    )
    _increment_optional_tensor_attribute(
        common_attn_metadata,
        "_num_computed_tokens_cpu",
        active_mask_cpu,
    )


def _update_common_attn_slot_mapping(
    instance,
    module,
    *,
    common_attn_metadata,
    clamped_positions,
    exceeds_max_model_len,
    active_mask,
    token_index: int,
) -> None:
    block_size = instance.draft_attn_groups[0].kv_cache_spec.block_size
    if instance.uses_mrope:
        block_numbers = clamped_positions[0] // block_size
    else:
        block_numbers = clamped_positions // block_size
    block_ids = common_attn_metadata.block_table_tensor.gather(
        dim=1,
        index=block_numbers.view(-1, 1),
    ).view(-1)
    if instance.uses_mrope:
        common_attn_metadata.slot_mapping = (
            block_ids * block_size + clamped_positions[0] % block_size
        )
    else:
        common_attn_metadata.slot_mapping = (
            block_ids * block_size + clamped_positions % block_size
        )
    common_attn_metadata.slot_mapping.masked_fill_(
        exceeds_max_model_len,
        module.PADDING_SLOT_ID,
    )
    if token_index > 0:
        common_attn_metadata.slot_mapping[~active_mask] = module.PADDING_SLOT_ID


def _refresh_per_layer_attn_metadata(
    instance,
    *,
    common_attn_metadata,
    per_layer_attn_metadata: dict[str, object],
    draft_index: int,
):
    attn_metadata = None
    for attn_group in instance.draft_attn_groups:
        attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
            common_attn_metadata=common_attn_metadata,
            draft_index=draft_index,
        )
        for layer_name in attn_group.layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
    return attn_metadata


def _run_iteration_forward_pass(
    instance,
    module,
    *,
    state: DraftIterationState,
    input_ids,
    clamped_positions,
):
    instance.input_ids[: state.batch_size] = input_ids
    _call_member(instance, "_set_positions", state.batch_size, clamped_positions)  # pylint: disable=function-ret
    instance.hidden_states[: state.batch_size] = state.hidden_states
    if instance.supports_mm_inputs:
        instance.inputs_embeds[: state.batch_size] = instance.model.embed_input_ids(
            input_ids
        )
        step_input_ids = None
        inputs_embeds = instance.inputs_embeds[: state.input_batch_size]
    else:
        step_input_ids = instance.input_ids[: state.input_batch_size]
        inputs_embeds = None
    return _run_model_forward(
        instance,
        module,
        per_layer_attn_metadata=state.per_layer_attn_metadata,
        common_attn_metadata=state.common_attn_metadata,
        num_input_tokens=state.input_batch_size,
        num_tokens_across_dp=state.batch_size_across_dp,
        cudagraph_runtime_mode=state.cudagraph_runtime_mode,
        input_ids=step_input_ids,
        inputs_embeds=inputs_embeds,
    )


def _advance_confidence_filtered_step(
    instance,
    module,
    *,
    state: DraftIterationState,
    token_index: int,
    confidence_threshold: float,
) -> bool:
    active_mask = state.continue_mask[: state.batch_size]
    input_ids = state.draft_token_ids_list[-1].int()
    if token_index > 0:
        input_ids = module.torch.where(active_mask, input_ids, state.pad_token_id)

    (
        state.positions,
        exceeds_max_model_len,
        clamped_positions,
    ) = _advance_positions(instance, module, state.positions)
    _update_common_attn_seq_lens(
        state.common_attn_metadata,
        active_mask=active_mask,
        exceeds_max_model_len=exceeds_max_model_len,
        max_model_len=instance.max_model_len,
    )
    _update_common_attn_slot_mapping(
        instance,
        module,
        common_attn_metadata=state.common_attn_metadata,
        clamped_positions=clamped_positions,
        exceeds_max_model_len=exceeds_max_model_len,
        active_mask=active_mask,
        token_index=token_index,
    )
    _ = _refresh_per_layer_attn_metadata(
        instance,
        common_attn_metadata=state.common_attn_metadata,
        per_layer_attn_metadata=state.per_layer_attn_metadata,
        draft_index=token_index + 1,
    )

    last_hidden_states, hidden_states = _run_iteration_forward_pass(
        instance,
        module,
        state=state,
        input_ids=input_ids,
        clamped_positions=clamped_positions,
    )

    state.hidden_states = hidden_states[: state.batch_size]
    logits = instance.model.compute_logits(last_hidden_states[: state.batch_size])
    draft_token_ids = logits.argmax(dim=-1)
    state.draft_token_ids_list.append(draft_token_ids)

    confidence = logits.softmax(dim=-1, dtype=module.torch.float32).max(dim=-1).values
    if (confidence < confidence_threshold).all():
        return False
    state.continue_mask[: state.batch_size] &= confidence >= confidence_threshold
    return True


def _finalize_confidence_filtered_draft(
    instance,
    module,
    *,
    state: DraftIterationState,
    effective_draft_length: int,
    confidence_threshold: float,
):
    draft_token_ids = module.torch.stack(state.draft_token_ids_list, dim=1)
    generated_length = draft_token_ids.shape[1]
    runner = getattr(instance, "runner", None)
    if runner is not None and generated_length < effective_draft_length:
        runner.draft_length = generated_length
        log_runtime_state(
            "confidence-early-stop",
            requested_draft_length=effective_draft_length,
            generated_length=generated_length,
            confidence_threshold=confidence_threshold,
        )
    if draft_token_ids.shape[1] < instance.num_speculative_tokens:
        draft_padding = module.torch.full(
            (state.batch_size, instance.num_speculative_tokens - draft_token_ids.shape[1]),
            -1,
            dtype=draft_token_ids.dtype,
            device=draft_token_ids.device,
        )
        draft_token_ids = module.torch.cat([draft_token_ids, draft_padding], dim=1)
    return draft_token_ids


def _register_or_apply_post_import_hook(module_name, patcher) -> None:
    import wrapt

    if module_name in sys.modules:
        patcher(sys.modules[module_name])
    wrapt.register_post_import_hook(patcher, module_name)


def _patch_metrics_module(module) -> None:
    module.AdaptiveDraftLengthController = AdaptiveDraftLengthController


def _patch_arg_utils_module(module) -> None:
    model_fields = getattr(module.SpeculativeConfig, "model_fields", {})
    if {
        "speculative_token_range",
        "draft_confidence_threshold",
    }.issubset(model_fields):
        return

    original_create_speculative_config = module.EngineArgs.create_speculative_config
    if getattr(original_create_speculative_config, "_wings_adaptive_draft_patched", False):
        return

    def patched_create_speculative_config(
        self,
        target_model_config,
        target_parallel_config,
    ):
        if self.speculative_config is None:
            return None

        original_user_config = self.speculative_config
        stripped_config = copy.deepcopy(original_user_config)
        speculative_token_range = stripped_config.pop("speculative_token_range", None)
        draft_confidence_threshold = stripped_config.pop(
            "draft_confidence_threshold",
            None,
        )

        self.speculative_config = stripped_config
        try:
            spec_config = original_create_speculative_config(
                self,
                target_model_config,
                target_parallel_config,
            )
        finally:
            self.speculative_config = original_user_config

        if spec_config is None:
            return None

        resolved = resolve_speculative_token_settings(
            method=spec_config.method,
            num_speculative_tokens=spec_config.num_speculative_tokens,
            speculative_token_range=speculative_token_range,
            draft_confidence_threshold=draft_confidence_threshold,
        )
        spec_config.num_speculative_tokens = resolved.num_speculative_tokens
        spec_config.speculative_token_range = resolved.speculative_token_range
        spec_config.draft_confidence_threshold = resolved.draft_confidence_threshold
        if (
            resolved.speculative_token_range is not None
            or resolved.draft_confidence_threshold > 0.0
        ):
            log_runtime_state(
                "resolved-speculative-config",
                method=spec_config.method,
                num_speculative_tokens=resolved.num_speculative_tokens,
                speculative_token_range=resolved.speculative_token_range,
                draft_confidence_threshold=resolved.draft_confidence_threshold,
            )
        return spec_config

    patched_create_speculative_config._wings_adaptive_draft_patched = True  # pylint: disable=protected-access
    module.EngineArgs.create_speculative_config = patched_create_speculative_config


def _patch_gpu_model_runner_module(module) -> None:
    runner_cls = module.GPUModelRunner
    source_init = inspect.getsource(runner_cls.__init__)
    if "draft_length_controller" not in source_init:
        original_init = runner_cls.__init__
        if not getattr(original_init, "_wings_adaptive_draft_patched", False):

            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.draft_length_controller = None
                spec_config = getattr(self, "speculative_config", None)
                token_range = getattr(spec_config, "speculative_token_range", None)
                if (
                    spec_config is not None
                    and _supports_adaptive_draft_length(
                        getattr(spec_config, "method", None)
                    )
                    and token_range is not None
                ):
                    self.draft_length_controller = AdaptiveDraftLengthController(
                        token_range,
                        initial_length=spec_config.num_speculative_tokens,
                    )
                    self.draft_length = self.draft_length_controller.current_length
                    self.uniform_decode_query_len = _resolve_uniform_decode_query_len(
                        self,
                        spec_config,
                    )

            patched_init._wings_adaptive_draft_patched = True  # pylint: disable=protected-access
            runner_cls.__init__ = patched_init

    module.AdaptiveDraftLengthController = AdaptiveDraftLengthController

    source_update = inspect.getsource(
        getattr(runner_cls, "_update_states_after_model_execute")
    )
    if "draft_length_controller" in source_update:
        return

    original_update = getattr(runner_cls, "_update_states_after_model_execute")
    if getattr(original_update, "_wings_adaptive_draft_patched", False):
        return

    def patched_update_states_after_model_execute(self, *args, **kwargs):
        result = original_update(self, *args, **kwargs)
        controller = getattr(self, "draft_length_controller", None)
        if controller is not None:
            valid_sampled_token_count = _call_member(
                self,
                "_get_valid_sampled_token_count",
            )
            if valid_sampled_token_count:
                previous_draft_length = self.draft_length
                num_draft_tokens = _resolve_num_draft_tokens_for_controller(
                    self,
                    valid_sampled_token_count,
                )
                num_accepted_tokens = sum(
                    max(0, count - 1) for count in valid_sampled_token_count
                )
                if num_accepted_tokens > num_draft_tokens:
                    log_runtime_state(
                        "adaptive-draft-length-accounting-clamped",
                        num_draft_tokens=num_draft_tokens,
                        num_accepted_tokens=num_accepted_tokens,
                        previous_draft_length=previous_draft_length,
                    )
                    num_accepted_tokens = num_draft_tokens
                self.draft_length = controller.observe_iteration(
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted_tokens,
                )
                if self.draft_length != previous_draft_length:
                    log_runtime_state(
                        "adaptive-draft-length-updated",
                        previous_draft_length=previous_draft_length,
                        next_draft_length=self.draft_length,
                        num_draft_tokens=num_draft_tokens,
                        num_accepted_tokens=num_accepted_tokens,
                    )
        return result

    patched_update_states_after_model_execute._wings_adaptive_draft_patched = True  # pylint: disable=protected-access
    setattr(
        runner_cls,
        "_update_states_after_model_execute",
        patched_update_states_after_model_execute,
    )

    original_propose_draft_token_ids = getattr(runner_cls, "propose_draft_token_ids", None)
    if original_propose_draft_token_ids is not None and not getattr(
        original_propose_draft_token_ids,
        "_wings_adaptive_draft_patched",
        False,
    ):
        raw_propose_draft_token_ids = inspect.getattr_static(
            runner_cls,
            "propose_draft_token_ids",
        )

        def patched_propose_draft_token_ids(
            self,
            *args,
            **kwargs,
        ):
            inputs = _bind_gpu_model_runner_propose_inputs(
                raw_propose_draft_token_ids,
                original_propose_draft_token_ids,
                self,
                args,
                kwargs,
            )
            scheduler_output = inputs.scheduler_output
            sampled_token_ids = inputs.sampled_token_ids
            sampling_metadata = inputs.sampling_metadata
            hidden_states = inputs.hidden_states
            aux_hidden_states = inputs.aux_hidden_states
            spec_decode_metadata = inputs.spec_decode_metadata
            common_attn_metadata = inputs.common_attn_metadata
            slot_mappings = inputs.slot_mappings
            spec_config = getattr(self, "speculative_config", None)
            if not (
                spec_config is not None
                and _supports_adaptive_draft_length(getattr(spec_config, "method", None))
            ):
                return _call_original_class_method(
                    raw_propose_draft_token_ids,
                    original_propose_draft_token_ids,
                    self,
                    inputs.scheduler_output,
                    inputs.sampled_token_ids,
                    inputs.sampling_metadata,
                    inputs.hidden_states,
                    inputs.sample_hidden_states,
                    inputs.aux_hidden_states,
                    inputs.spec_decode_metadata,
                    inputs.common_attn_metadata,
                    inputs.slot_mappings,
                )

            num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
            if spec_config.disable_padded_drafter_batch:
                if not isinstance(sampled_token_ids, list):
                    raise ValueError(
                        "sampled_token_ids should be a python list when"
                        "padded-batch is disabled."
                    )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                if not isinstance(sampled_token_ids, torch.Tensor):
                    raise ValueError(
                        "sampled_token_ids should be a torch.Tensor when"
                        "padded-batch is enabled."
                    )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                )
                _call_member(
                    self,
                    "_copy_valid_sampled_token_count",
                    next_token_ids, valid_sampled_tokens_count
                )

            num_rejected_tokens_gpu = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = _call_member(
                    self,
                    "_get_positions",
                    num_scheduled_tokens,
                )
                if self.use_aux_hidden_state_outputs:
                    if aux_hidden_states is None:
                        raise ValueError(
                            "aux_hidden_states cannot be None when "
                            "use_aux_hidden_state_outputs is True"
                        )
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                    target_token_ids = self.input_ids.gpu[token_indices]
                    target_positions = _call_member(
                        self,
                        "_get_positions",
                        token_indices,
                    )
                    if self.use_aux_hidden_state_outputs:
                        if aux_hidden_states is None:
                            raise ValueError(
                                "aux_hidden_states cannot be None when "
                                "use_aux_hidden_state_outputs is True"
                            )
                        target_hidden_states = torch.cat(
                            [h[token_indices] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[token_indices]
                else:
                    (
                        common_attn_metadata,
                        token_indices_to_sample,
                        num_rejected_tokens_gpu,
                    ) = self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    )
                    total_num_tokens = common_attn_metadata.num_actual_tokens
                    target_token_ids = self.input_ids.gpu[:total_num_tokens]
                    target_positions = _call_member(
                        self,
                        "_get_positions",
                        total_num_tokens,
                    )
                    if self.use_aux_hidden_state_outputs:
                        if aux_hidden_states is None:
                            raise ValueError(
                                "aux_hidden_states cannot be None when "
                                "use_aux_hidden_state_outputs is True"
                            )
                        target_hidden_states = torch.cat(
                            [h[:total_num_tokens] for h in aux_hidden_states], dim=-1
                        )
                    else:
                        target_hidden_states = hidden_states[:total_num_tokens]

            if self.supports_mm_inputs and self.drafter.supports_mm_inputs:
                mm_embed_inputs = _call_member(
                    self,
                    "_gather_mm_embeddings",
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            return _call_with_optional_draft_length(
                self.drafter.propose,
                draft_length=getattr(self, "draft_length", None),
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )

        patched_propose_draft_token_ids._wings_adaptive_draft_patched = True  # pylint: disable=protected-access
        setattr(runner_cls, "propose_draft_token_ids", patched_propose_draft_token_ids)

    raw_get_draft_token_ids_cpu = inspect.getattr_static(
        runner_cls,
        "_get_draft_token_ids_cpu",
    )
    original_get_draft_token_ids_cpu = getattr(runner_cls, "_get_draft_token_ids_cpu")
    if getattr(original_get_draft_token_ids_cpu, "_wings_adaptive_draft_patched", False):
        return

    def patched_get_draft_token_ids_cpu(self, *args, **kwargs):
        draft_token_ids, req_ids = _call_original_class_method(
            raw_get_draft_token_ids_cpu,
            original_get_draft_token_ids_cpu,
            self,
            *args,
            **kwargs,
        )
        spec_config = getattr(self, "speculative_config", None)
        confidence_threshold = getattr(spec_config, "draft_confidence_threshold", 0.0)
        if (
            getattr(spec_config, "method", None) == "draft_model"
            and confidence_threshold > 0.0
            and draft_token_ids
        ):
            trimmed_draft_token_ids = trim_trailing_invalid_draft_tokens(draft_token_ids)
            if trimmed_draft_token_ids != draft_token_ids:
                log_runtime_state(
                    "trimmed-draft-token-export",
                    confidence_threshold=confidence_threshold,
                    request_count=len(req_ids),
                )
            draft_token_ids = trimmed_draft_token_ids
        return draft_token_ids, req_ids

    patched_get_draft_token_ids_cpu._wings_adaptive_draft_patched = True  # pylint: disable=protected-access
    setattr(runner_cls, "_get_draft_token_ids_cpu", patched_get_draft_token_ids_cpu)


def _patch_spec_decode_eagle_module(module) -> None:
    original_propose = module.SpecDecodeBaseProposer.propose
    if getattr(original_propose, "_wings_adaptive_draft_patched", False):
        return

    source_propose = inspect.getsource(module.SpecDecodeBaseProposer.propose)
    if "draft_confidence_threshold" in source_propose:
        return
    original_propose_supports_draft_length = (
        "draft_length" in inspect.signature(original_propose).parameters
    )

    def patched_propose(self, *args, **kwargs):
        inputs = _bind_propose_inputs(original_propose, self, args, kwargs)
        if inputs.draft_length is None and getattr(self, "runner", None) is not None:
            inputs.draft_length = getattr(self.runner, "draft_length", None)
        effective_draft_length = inputs.draft_length or self.num_speculative_tokens
        confidence_threshold = getattr(
            getattr(self, "speculative_config", None),
            "draft_confidence_threshold",
            0.0,
        )
        use_confidence_filter = _should_use_confidence_filter(
            self,
            confidence_threshold,
        )
        use_padded_adaptive_draft = _should_use_padded_adaptive_draft(
            self,
            draft_length=effective_draft_length,
            confidence_threshold=confidence_threshold,
        ) and hasattr(self, "set_inputs_first_pass")
        if not (use_confidence_filter or use_padded_adaptive_draft):
            return _call_original_propose(
                original_propose,
                supports_draft_length=original_propose_supports_draft_length,
                instance=self,
                inputs=inputs,
            )

        if effective_draft_length <= 1 and not use_padded_adaptive_draft:
            return _call_original_propose(
                original_propose,
                supports_draft_length=original_propose_supports_draft_length,
                instance=self,
                inputs=inputs,
            )

        target_hidden_states = inputs.target_hidden_states
        if getattr(self, "method", None) == "eagle3":
            eagle3_model_cls = getattr(module, "Eagle3LlamaForCausalLM", None)
            if eagle3_model_cls is not None and isinstance(self.model, eagle3_model_cls):
                target_hidden_states = self.model.combine_hidden_states(
                    target_hidden_states
                )
                if target_hidden_states.shape[-1] != self.hidden_size:
                    raise ValueError(f"target_hidden_states.shape[-1] {target_hidden_states.shape[-1]} != self.hidden_size {self.hidden_size}")

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=inputs.target_token_ids,
                next_token_ids=inputs.next_token_ids,
                target_positions=inputs.target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=inputs.token_indices_to_sample,
                cad=inputs.common_attn_metadata,
                num_rejected_tokens_gpu=inputs.num_rejected_tokens_gpu,
            )
        )

        _ = _require_runner(self)
        per_layer_attn_metadata, attn_metadata = _build_per_layer_attn_metadata(
            self,
            common_attn_metadata,
        )
        (
            cudagraph_runtime_mode,
            num_input_tokens,
            num_tokens_across_dp,
        ) = _call_member(self, "_determine_batch_execution_and_padding", num_tokens)
        input_ids, inputs_embeds = _build_model_inputs(
            self,
            num_input_tokens,
            inputs.mm_embed_inputs,
        )
        last_hidden_states, hidden_states = _run_model_forward(
            self,
            module,
            per_layer_attn_metadata=per_layer_attn_metadata,
            common_attn_metadata=common_attn_metadata,
            num_input_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        sample_hidden_states, positions, hidden_states = _extract_sampling_inputs(
            self,
            last_hidden_states=last_hidden_states,
            hidden_states=hidden_states,
            token_indices_to_sample=token_indices_to_sample,
        )

        if isinstance(attn_metadata, module.TreeAttentionMetadata):
            return _call_original_propose(
                original_propose,
                supports_draft_length=original_propose_supports_draft_length,
                instance=self,
                inputs=inputs,
            )

        draft_token_ids = _call_member(self, "_greedy_sample", sample_hidden_states)

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata,
            self.allowed_attn_types,
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        state = _prepare_first_draft_iteration_state(
            self,
            module,
            common_attn_metadata=common_attn_metadata,
            per_layer_attn_metadata=per_layer_attn_metadata,
            positions=positions,
            hidden_states=hidden_states,
            draft_token_ids=draft_token_ids,
            num_rejected_tokens_gpu=inputs.num_rejected_tokens_gpu,
            pad_token_id=_resolve_pad_token_id(self.speculative_config),
        )

        for token_index in range(effective_draft_length - 1):
            should_continue = _advance_confidence_filtered_step(
                self,
                module,
                state=state,
                token_index=token_index,
                confidence_threshold=confidence_threshold if use_confidence_filter else 0.0,
            )
            if not should_continue:
                break

        return _finalize_confidence_filtered_draft(
            self,
            module,
            state=state,
            effective_draft_length=effective_draft_length,
            confidence_threshold=confidence_threshold,
        )

    patched_propose._wings_adaptive_draft_patched = True  # pylint: disable=protected-access
    module.SpecDecodeBaseProposer.propose = patched_propose


def patch_vllm_adaptive_draft_model():
    log_runtime_state("adaptive_draft_model patch enabled")
    _register_or_apply_post_import_hook("vllm.v1.spec_decode.metrics", _patch_metrics_module)
    _register_or_apply_post_import_hook("vllm.engine.arg_utils", _patch_arg_utils_module)
    _register_or_apply_post_import_hook(
        "vllm.v1.worker.gpu_model_runner",
        _patch_gpu_model_runner_module,
    )
    _register_or_apply_post_import_hook(
        "vllm.v1.spec_decode.eagle",
        _patch_spec_decode_eagle_module,
    )
