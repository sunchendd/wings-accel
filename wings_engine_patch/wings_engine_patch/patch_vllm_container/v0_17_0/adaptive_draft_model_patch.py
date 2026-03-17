import copy
import inspect
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedSpeculativeTokenSettings:
    num_speculative_tokens: int | None
    speculative_token_range: list[int] | None
    draft_confidence_threshold: float


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
        if any(
            left >= right for left, right in zip(allowed_lengths, allowed_lengths[1:])
        ):
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


def log_runtime_state(event: str, /, **fields) -> None:
    parts = [f"{key}={value}" for key, value in sorted(fields.items())]
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[wins-accel] {event}{suffix}", file=sys.stderr)


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

    if speculative_token_range is None:
        return ResolvedSpeculativeTokenSettings(
            num_speculative_tokens=num_speculative_tokens,
            speculative_token_range=None,
            draft_confidence_threshold=float(draft_confidence_threshold),
        )

    if method is None:
        method = "draft_model"
    if method != "draft_model":
        raise ValueError("speculative_token_range is only supported for draft_model.")
    if num_speculative_tokens is None:
        raise ValueError(
            "draft_model requires num_speculative_tokens when "
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
    if any(
        left >= right
        for left, right in zip(
            speculative_token_range,
            speculative_token_range[1:],
        )
    ):
        raise ValueError(
            "speculative_token_range must be strictly increasing as supplied."
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

    import torch

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

    patched_create_speculative_config._wings_adaptive_draft_patched = True
    module.EngineArgs.create_speculative_config = patched_create_speculative_config


def _patch_gpu_model_runner_module(module) -> None:
    source_init = inspect.getsource(module.GPUModelRunner.__init__)
    if "draft_length_controller" not in source_init:
        original_init = module.GPUModelRunner.__init__
        if not getattr(original_init, "_wings_adaptive_draft_patched", False):

            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.draft_length_controller = None
                spec_config = getattr(self, "speculative_config", None)
                token_range = getattr(spec_config, "speculative_token_range", None)
                if (
                    spec_config is not None
                    and getattr(spec_config, "method", None) == "draft_model"
                    and token_range is not None
                ):
                    self.draft_length_controller = AdaptiveDraftLengthController(
                        token_range,
                        initial_length=spec_config.num_speculative_tokens,
                    )
                    self.draft_length = self.draft_length_controller.current_length
                    self.uniform_decode_query_len = 1

            patched_init._wings_adaptive_draft_patched = True
            module.GPUModelRunner.__init__ = patched_init

    module.AdaptiveDraftLengthController = AdaptiveDraftLengthController

    source_update = inspect.getsource(module.GPUModelRunner._update_states_after_model_execute)
    if "draft_length_controller" in source_update:
        return

    original_update = module.GPUModelRunner._update_states_after_model_execute
    if getattr(original_update, "_wings_adaptive_draft_patched", False):
        return

    def patched_update_states_after_model_execute(self, *args, **kwargs):
        result = original_update(self, *args, **kwargs)
        controller = getattr(self, "draft_length_controller", None)
        if controller is not None:
            valid_sampled_token_count = self._get_valid_sampled_token_count()
            if valid_sampled_token_count:
                previous_draft_length = self.draft_length
                num_draft_tokens = self.draft_length * len(valid_sampled_token_count)
                num_accepted_tokens = sum(
                    max(0, count - 1) for count in valid_sampled_token_count
                )
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

    patched_update_states_after_model_execute._wings_adaptive_draft_patched = True
    module.GPUModelRunner._update_states_after_model_execute = (
        patched_update_states_after_model_execute
    )

    original_get_draft_token_ids_cpu = module.GPUModelRunner._get_draft_token_ids_cpu
    if getattr(original_get_draft_token_ids_cpu, "_wings_adaptive_draft_patched", False):
        return

    def patched_get_draft_token_ids_cpu(self, *args, **kwargs):
        draft_token_ids, req_ids = original_get_draft_token_ids_cpu(self, *args, **kwargs)
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

    patched_get_draft_token_ids_cpu._wings_adaptive_draft_patched = True
    module.GPUModelRunner._get_draft_token_ids_cpu = patched_get_draft_token_ids_cpu


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

    def call_original_propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        *,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
        draft_length=None,
    ):
        kwargs = {
            "mm_embed_inputs": mm_embed_inputs,
            "num_rejected_tokens_gpu": num_rejected_tokens_gpu,
            "slot_mappings": slot_mappings,
        }
        if original_propose_supports_draft_length:
            kwargs["draft_length"] = draft_length
        return original_propose(
            self,
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            token_indices_to_sample,
            common_attn_metadata,
            sampling_metadata,
            **kwargs,
        )

    def patched_propose(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        token_indices_to_sample,
        common_attn_metadata,
        sampling_metadata,
        mm_embed_inputs=None,
        num_rejected_tokens_gpu=None,
        slot_mappings=None,
        draft_length=None,
    ):
        if draft_length is None and getattr(self, "runner", None) is not None:
            draft_length = getattr(self.runner, "draft_length", None)

        confidence_threshold = getattr(
            getattr(self, "speculative_config", None),
            "draft_confidence_threshold",
            0.0,
        )
        use_confidence_filter = (
            getattr(self, "method", None) == "draft_model"
            and confidence_threshold > 0.0
            and not getattr(self, "parallel_drafting", False)
            and not getattr(self, "use_local_argmax_reduction", False)
        )
        if not use_confidence_filter:
            return call_original_propose(
                self,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
                draft_length=draft_length,
            )

        effective_draft_length = draft_length or self.num_speculative_tokens
        batch_size = common_attn_metadata.batch_size()
        if effective_draft_length <= 1:
            return call_original_propose(
                self,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
                draft_length=draft_length,
            )

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        assert self.runner is not None

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata,
                draft_index=0,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)
            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        with module.set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                num_input_tokens,
                common_attn_metadata.slot_mapping,
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if isinstance(attn_metadata, module.TreeAttentionMetadata):
            return call_original_propose(
                self,
                target_token_ids,
                target_positions,
                target_hidden_states,
                next_token_ids,
                token_indices_to_sample,
                common_attn_metadata,
                sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
                draft_length=draft_length,
            )

        draft_token_ids = self._greedy_sample(sample_hidden_states)

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

        draft_token_ids_list = [draft_token_ids]
        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = module.torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        hf_config = getattr(self.speculative_config.draft_model_config, "hf_config", None)
        pad_token_id = getattr(hf_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        continue_mask = module.torch.ones(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=module.torch.bool,
            device=self.device,
        )
        continue_mask[:batch_size] = True

        for token_index in range(effective_draft_length - 1):
            active_mask = continue_mask[:batch_size]
            input_ids = draft_token_ids_list[-1].int()
            if token_index > 0:
                input_ids = module.torch.where(active_mask, input_ids, pad_token_id)

            if self.uses_mrope:
                positions += 1
                exceeds_max_model_len = positions[0] >= self.max_model_len
                clamped_positions = module.torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    module.torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = module.torch.where(
                    exceeds_max_model_len,
                    0,
                    positions,
                )

            length_increments = active_mask.to(common_attn_metadata.seq_lens.dtype)
            common_attn_metadata.seq_lens += length_increments
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
            common_attn_metadata.max_seq_len = min(
                common_attn_metadata.max_seq_len + 1,
                self.max_model_len,
            )

            active_mask_cpu = active_mask.cpu()
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += active_mask_cpu.to(
                    common_attn_metadata._seq_lens_cpu.dtype
                )
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += active_mask_cpu.to(
                    common_attn_metadata._num_computed_tokens_cpu.dtype
                )

            block_size = self.draft_attn_groups[0].kv_cache_spec.block_size
            if self.uses_mrope:
                block_numbers = clamped_positions[0] // block_size
            else:
                block_numbers = clamped_positions // block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1,
                index=block_numbers.view(-1, 1),
            )
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
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

            for attn_group in self.draft_attn_groups:
                attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=token_index + 1,
                )
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)
                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with module.set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    input_batch_size,
                    common_attn_metadata.slot_mapping,
                ),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

            probs = logits.softmax(dim=-1, dtype=module.torch.float32)
            confidence = probs.max(dim=-1).values
            if (confidence < confidence_threshold).all():
                break
            continue_mask[:batch_size] &= confidence >= confidence_threshold

        draft_token_ids = module.torch.stack(draft_token_ids_list, dim=1)
        generated_length = draft_token_ids.shape[1]
        if getattr(self, "runner", None) is not None and generated_length < effective_draft_length:
            self.runner.draft_length = generated_length
            log_runtime_state(
                "confidence-early-stop",
                requested_draft_length=effective_draft_length,
                generated_length=generated_length,
                confidence_threshold=confidence_threshold,
            )
        if draft_token_ids.shape[1] < self.num_speculative_tokens:
            draft_padding = module.torch.full(
                (batch_size, self.num_speculative_tokens - draft_token_ids.shape[1]),
                -1,
                dtype=draft_token_ids.dtype,
                device=draft_token_ids.device,
            )
            draft_token_ids = module.torch.cat([draft_token_ids, draft_padding], dim=1)
        return draft_token_ids

    patched_propose._wings_adaptive_draft_patched = True
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
