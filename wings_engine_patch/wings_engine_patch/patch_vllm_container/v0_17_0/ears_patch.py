import logging
import os
import sys

from .ears_ascend_compat import _ASCEND_DRAFT_COMPAT_MODULES
from .ears_ascend_compat import _ASCEND_VLLM_COMPAT_MODULES
from .ears_ascend_compat import patch_vllm_ascend_draft_compat as _patch_vllm_ascend_draft_compat


LOGGER = logging.getLogger("wings_accel.ears")
_SUPPORTED_EARS_METHODS = {"mtp", "eagle3", "suffix"}
_EARS_REJECTION_SAMPLER_CLASS = None


class _StderrProxy:
    @staticmethod
    def write(message: str) -> int:
        return sys.stderr.write(message)

    @staticmethod
    def flush() -> None:
        sys.stderr.flush()


def _configure_logger() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(_StderrProxy())
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def log_runtime_state(event: str, /, **fields) -> None:
    parts = [f"{key}={value}" for key, value in sorted(fields.items())]
    suffix = f" {' '.join(parts)}" if parts else ""
    LOGGER.info("[wins-accel] %s%s", event, suffix)


def _torch():
    import torch

    return torch


def rejection_greedy_sample_spec_len_1_pytorch(
    output_token_ids,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
):
    torch = _torch()
    accept_req_mask = draft_token_ids == target_argmax
    output_token_ids[:, 0] = target_argmax
    output_token_ids[:, 1] = torch.where(
        accept_req_mask,
        bonus_token_ids.squeeze(1),
        output_token_ids[:, 1],
    )


def rejection_greedy_sample_pytorch(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    draft_tokens_per_req,
    max_spec_len,
    is_greedy=None,
):
    torch = _torch()
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req, device=device)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(num_tokens, device=device) - start_indices[token_req_ids]

    mismatch_global = draft_token_ids != target_argmax
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        pos_matrix = torch.full((batch_size, max_spec_len), -1, dtype=torch.long, device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len), False, dtype=torch.bool, device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix, max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = first_mismatch_pos_per_req == max_spec_len * 2
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[no_mismatch_mask]

    copy_len = torch.minimum(first_mismatch_pos_per_req + 1, draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    final_copy_mask = copy_mask & is_greedy.unsqueeze(1)
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[global_idx[final_copy_mask]].to(output_token_ids.dtype)

    needs_bonus = is_greedy & (first_mismatch_pos_per_req >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids.squeeze(1)[bonus_rows]


def rejection_random_sample_ears_pytorch(
    output_token_ids,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    bonus_token_ids,
    recovered_token_ids,
    uniform_probs,
    is_greedy,
    max_spec_len,
    vocab_size,
    is_ngram=False,
    base_tolerance=0.0,
):
    torch = _torch()
    del vocab_size
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_device = torch.tensor([0], device=device)
    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start

    pos_indices = torch.arange(max_spec_len, device=device)[None, :]
    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]

    if is_ngram:
        draft_token_probs = torch.ones_like(draft_tokens, dtype=torch.float32)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_spec_len)

    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()
    flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
    target_token_probs = flat_target_probs.view(batch_size, max_spec_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    max_target_probs = target_probs.max(dim=-1).values
    token_uncertainties = (1.0 - max_target_probs)[global_token_indices]
    tolerance = base_tolerance * token_uncertainties
    adjusted_uniform = uniform_token_probs - tolerance

    acceptance_condition = (draft_token_probs > 0.0) & (
        target_token_probs / draft_token_probs >= adjusted_uniform
    )
    first_rejection = (~acceptance_condition) & valid_mask

    default_pos = torch.full((batch_size, 1), max_spec_len, device=device)
    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True),
        first_rejection.float().argmax(dim=1, keepdim=True),
        default_pos,
    )
    should_skip = (pos_indices >= first_reject_pos) & valid_mask

    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)
    first_reject_mask = (
        (pos_indices == first_reject_pos)
        & valid_mask
        & non_greedy_mask[:, None]
    )
    final_update_mask = update_mask | first_reject_mask
    final_tokens = torch.where(
        first_reject_mask,
        recovered_tokens,
        torch.where(
            final_acceptance,
            draft_tokens,
            output_token_ids[:, :max_spec_len],
        ),
    )
    output_token_ids[:, :max_spec_len] = torch.where(
        final_update_mask,
        final_tokens,
        output_token_ids[:, :max_spec_len],
    )

    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    final_bonus_mask = non_greedy_mask & no_rejection & (num_draft_per_batch < (max_spec_len + 1))
    bonus_pos_mask = torch.arange(output_token_ids.shape[1], device=device)[None, :] == num_draft_per_batch[:, None]
    bonus_pos_mask = bonus_pos_mask & final_bonus_mask[:, None]
    output_token_ids[:] = torch.where(
        bonus_pos_mask,
        bonus_token_ids.view(-1, 1).expand_as(output_token_ids),
        output_token_ids,
    )


def _get_entropy_adaptive_rejection_sampler_class():
    global _EARS_REJECTION_SAMPLER_CLASS
    if _EARS_REJECTION_SAMPLER_CLASS is not None:
        return _EARS_REJECTION_SAMPLER_CLASS
    torch = _torch()

    from dataclasses import replace

    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.rejection_sampler import (
        GREEDY_TEMPERATURE,
        MAX_SPEC_LEN,
        PLACEHOLDER_TOKEN_ID,
        RejectionSampler,
        apply_sampling_constraints,
        generate_uniform_probs,
        sample_recovered_tokens,
    )

    def ears_rejection_sample(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_probs,
        bonus_token_ids,
        sampling_metadata,
        base_tolerance=0.0,
    ):
        batch_size = len(num_draft_tokens)
        num_tokens = draft_token_ids.shape[0]
        vocab_size = target_probs.shape[-1]
        device = target_probs.device

        output_token_ids = torch.empty(
            (batch_size, max_spec_len + 1),
            dtype=torch.int32,
            device=device,
        )
        output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

        if sampling_metadata.all_greedy:
            is_greedy = None
        else:
            is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

        if not sampling_metadata.all_random:
            target_argmax = target_probs.argmax(dim=-1)
            if (
                min(num_draft_tokens) == 1
                and max(num_draft_tokens) == 1
                and sampling_metadata.all_greedy
            ):
                rejection_greedy_sample_spec_len_1_pytorch(
                    output_token_ids,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                )
            else:
                rejection_greedy_sample_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    num_draft_tokens,
                    max_spec_len,
                    is_greedy,
                )
            if sampling_metadata.all_greedy:
                return output_token_ids

        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )
        recovered_token_ids = sample_recovered_tokens(
            max_spec_len,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            sampling_metadata,
            device,
        )
        rejection_random_sample_ears_pytorch(
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            is_ngram=draft_probs is None,
            base_tolerance=base_tolerance,
        )
        return output_token_ids

    class EntropyAdaptiveRejectionSampler(RejectionSampler):
        def __init__(self, sampler, base_tolerance=0.1):
            super().__init__(sampler)
            self.base_tolerance = base_tolerance

        def forward(
            self,
            metadata,
            draft_probs,
            logits,
            sampling_metadata,
        ):
            assert metadata.max_spec_len <= MAX_SPEC_LEN

            bonus_logits_indices = metadata.bonus_logits_indices
            target_logits_indices = metadata.target_logits_indices

            bonus_logits = logits[bonus_logits_indices]
            bonus_sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
                predict_bonus_token=True,
                logprobs_mode_override=(
                    "processed_logits"
                    if self.is_processed_logprobs_mode
                    else "raw_logits"
                ),
            )
            bonus_token_ids = bonus_sampler_output.sampled_token_ids

            raw_target_logits = logits[target_logits_indices].to(torch.float32)
            target_logits = raw_target_logits
            if not self.is_processed_logprobs_mode:
                target_logits = target_logits.clone()
            target_logits = self.apply_logits_processors(
                target_logits,
                sampling_metadata,
                metadata,
            )
            target_logits = apply_sampling_constraints(
                target_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
            )
            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
            output_token_ids = ears_rejection_sample(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                draft_probs,
                target_probs,
                bonus_token_ids,
                sampling_metadata,
                base_tolerance=self.base_tolerance,
            )

            logprobs_tensors = None
            if sampling_metadata.max_num_logprobs is not None:
                logprobs_tensors = self._get_logprobs_tensors(
                    sampling_metadata.max_num_logprobs,
                    metadata,
                    logits,
                    target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                    bonus_sampler_output.logprobs_tensors.logprobs,
                    output_token_ids,
                )

            return SamplerOutput(
                sampled_token_ids=output_token_ids,
                logprobs_tensors=logprobs_tensors,
            )

    _EARS_REJECTION_SAMPLER_CLASS = EntropyAdaptiveRejectionSampler
    return _EARS_REJECTION_SAMPLER_CLASS


def _register_or_apply_post_import_hook(module_name, patcher) -> None:
    import wrapt

    if module_name in sys.modules:
        patcher(sys.modules[module_name])
    wrapt.register_post_import_hook(patcher, module_name)


def _read_ears_tolerance() -> float:
    envs_module = sys.modules.get("vllm_ascend.envs")
    if envs_module is not None and hasattr(envs_module, "VLLM_EARS_TOLERANCE"):
        return float(getattr(envs_module, "VLLM_EARS_TOLERANCE"))
    return float(os.getenv("VLLM_EARS_TOLERANCE", "0.0"))


def _maybe_enable_ears_sampler(runner) -> None:
    spec_config = getattr(runner, "speculative_config", None)
    method = getattr(spec_config, "method", None)
    tolerance = _read_ears_tolerance()

    if tolerance <= 0.0:
        return
    if method not in _SUPPORTED_EARS_METHODS:
        return
    if getattr(runner, "rejection_sampler", None) is None:
        return
    if getattr(runner, "sampler", None) is None:
        return

    sampler_cls = _get_entropy_adaptive_rejection_sampler_class()
    runner.rejection_sampler = sampler_cls(
        runner.sampler,
        base_tolerance=tolerance,
    )
    log_runtime_state("ears sampler enabled", method=method, base_tolerance=tolerance)


def patch_vllm_ascend_draft_compat(module) -> None:
    return _patch_vllm_ascend_draft_compat(module)


def patch_vllm_ears():
    from .ears_ascend_runtime_hooks import register_ascend_runtime_hooks
    from .ears_nvidia_runtime_hooks import register_nvidia_runtime_hooks

    log_runtime_state("ears patch enabled")
    for module_name in _ASCEND_DRAFT_COMPAT_MODULES + _ASCEND_VLLM_COMPAT_MODULES:
        _register_or_apply_post_import_hook(module_name, patch_vllm_ascend_draft_compat)
    register_ascend_runtime_hooks(_register_or_apply_post_import_hook)
    register_nvidia_runtime_hooks(_register_or_apply_post_import_hook)


_configure_logger()
