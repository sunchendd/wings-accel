"""
EARS (Entropy-Adaptive Rejection Sampling) patch for vllm-ascend.

Improves speculative decoding acceptance rates by dynamically adjusting
the rejection threshold based on prediction uncertainty:
    tolerance = base_tolerance * (1 - max(target_probs))
    accept if target_prob / draft_prob >= (uniform_prob - tolerance)

Controlled by VLLM_EARS_TOLERANCE environment variable (default 0.0 = disabled).
Applies when spec decode method is "mtp", "eagle3", or "suffix".

This patch:
1. Injects VLLM_EARS_TOLERANCE into vllm_ascend.envs
2. Provides EntropyAdaptiveRejectionSampler (subclasses RejectionSampler)
3. Patches NPUModelRunner._set_up_drafter to use EARS sampler when enabled

Source: vllm-ascend deepseek-ears branch
"""

import os
import sys


def _register_or_apply_post_import_hook(module_name: str, hook) -> None:
    import wrapt
    if module_name in sys.modules:
        hook(sys.modules[module_name])
    else:
        wrapt.register_post_import_hook(hook, module_name)


def rejection_random_sample_ears_pytorch(
    output_token_ids,   # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,    # [num_tokens]
    draft_probs,        # [num_tokens, vocab_size] or None
    target_probs,       # [num_tokens, vocab_size]
    bonus_token_ids,    # [batch_size, 1]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,      # [num_tokens]
    is_greedy,          # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
    base_tolerance=0.0,
):
    """EARS-enhanced rejection sampling with uncertainty-based tolerance.

    Identical to standard rejection sampling except the acceptance condition
    is relaxed by a tolerance proportional to model uncertainty:
        tolerance = base_tolerance * (1 - max(target_probs))
        accept if target_prob / draft_prob >= (uniform_prob - tolerance)
    """
    import torch
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start

    max_draft_len = max_spec_len
    pos_indices_cpu = torch.arange(max_draft_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]

    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]

    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens.float())
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_draft_len)

    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = draft_tokens.flatten()
    flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
    target_token_probs = flat_target_probs.view(batch_size, max_draft_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    zero_threshold_cpu = torch.tensor([0.0], pin_memory=True, dtype=torch.float32)
    zero_threshold = zero_threshold_cpu.to(device, non_blocking=True)

    # EARS: compute per-token uncertainty and adjust acceptance threshold
    max_target_probs = target_probs.max(dim=-1).values   # [num_tokens]
    uncertainties = 1.0 - max_target_probs
    token_uncertainties = uncertainties[global_token_indices]  # [batch_size, max_draft_len]
    tolerance = base_tolerance * token_uncertainties

    adjusted_uniform = uniform_token_probs - tolerance
    acceptance_condition = (draft_token_probs > zero_threshold) & (
        target_token_probs / draft_token_probs.clamp(min=1e-8) >= adjusted_uniform
    )

    first_rejection = (~acceptance_condition) & valid_mask

    default_pos_cpu = torch.full([batch_size, 1], max_draft_len, pin_memory=True)
    default_pos = default_pos_cpu.to(device, non_blocking=True)

    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True),
        first_rejection.float().argmax(dim=1, keepdim=True),
        default_pos,
    )
    pos_mask = pos_indices >= first_reject_pos
    should_skip = pos_mask & valid_mask

    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)

    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask  # noqa: F841
    final_tokens = torch.where(
        first_reject_mask,
        recovered_tokens,
        torch.where(
            final_acceptance,
            draft_tokens,
            output_token_ids[:, :max_draft_len],
        ),
    )

    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_mask,
        final_tokens,
        output_token_ids[:, :max_draft_len],
    )

    # Fill bonus token for fully accepted sequences
    fully_accepted = ~first_rejection.any(dim=1)
    output_token_ids[fully_accepted, num_draft_per_batch[fully_accepted].long()] = (
        bonus_token_ids[fully_accepted, 0] if bonus_token_ids.dim() == 2
        else bonus_token_ids[fully_accepted]
    )


def _get_entropy_adaptive_rejection_sampler_class():
    """Return EntropyAdaptiveRejectionSampler, importing lazily to avoid eager deps."""
    from dataclasses import replace  # noqa: F401

    import torch
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.rejection_sampler import (
        GREEDY_TEMPERATURE,
        MAX_SPEC_LEN,
        PLACEHOLDER_TOKEN_ID,
        RejectionSampler,
        generate_uniform_probs,
    )
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

    from vllm_ascend.sample.rejection_sampler import apply_sampling_constraints

    class EntropyAdaptiveRejectionSampler(RejectionSampler):
        """EARS (Entropy-Adaptive Rejection Sampling) implementation.

        Subclasses RejectionSampler to introduce dynamic tolerance based on
        prediction uncertainty, improving speculative decoding acceptance rates.
        """

        def __init__(self, sampler: Sampler, base_tolerance: float = 0.1):
            super().__init__(sampler)
            self.base_tolerance = base_tolerance

        def forward(
            self,
            metadata: SpecDecodeMetadata,
            draft_probs: torch.Tensor | None,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
        ) -> SamplerOutput:
            assert metadata.max_spec_len <= MAX_SPEC_LEN

            bonus_logits = logits[metadata.bonus_logits_indices]
            bonus_sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=replace(
                    sampling_metadata,
                    max_num_logprobs=-1,
                ),
                predict_bonus_token=True,
                logprobs_mode_override=(
                    "processed_logits" if self.is_processed_logprobs_mode else "raw_logits"
                ),
            )
            bonus_token_ids = bonus_sampler_output.sampled_token_ids

            raw_target_logits = logits[metadata.target_logits_indices].to(torch.float32)
            target_logits = raw_target_logits
            if not self.is_processed_logprobs_mode:
                target_logits = target_logits.clone()
            target_logits = self.apply_logits_processors(
                target_logits, sampling_metadata, metadata
            )
            target_logits = apply_sampling_constraints(
                target_logits,
                metadata.cu_num_draft_tokens,
                sampling_metadata,
            )
            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)

            batch_size = len(metadata.num_draft_tokens)
            output_token_ids = torch.empty(
                (batch_size, metadata.max_spec_len + 1),
                dtype=torch.int32,
                device=target_probs.device,
            )
            output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

            if sampling_metadata.all_greedy:
                is_greedy = None
            else:
                is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

            if not sampling_metadata.all_random:
                # Greedy or mixed: use argmax for greedy portion
                recovered_token_ids = target_probs.argmax(dim=-1).to(torch.int32)
            else:
                recovered_token_ids = target_probs.argmax(dim=-1).to(torch.int32)

            uniform_probs = generate_uniform_probs(
                metadata.cu_num_draft_tokens, target_probs.device
            )

            if is_greedy is None or sampling_metadata.all_greedy:
                # Pure greedy: standard argmax comparison
                self._greedy_rejection_sample(
                    output_token_ids,
                    metadata,
                    target_probs,
                    bonus_token_ids,
                )
            else:
                rejection_random_sample_ears_pytorch(
                    output_token_ids,
                    metadata.cu_num_draft_tokens,
                    metadata.draft_token_ids,
                    draft_probs,
                    target_probs,
                    bonus_token_ids,
                    recovered_token_ids,
                    uniform_probs,
                    is_greedy,
                    metadata.max_spec_len,
                    target_probs.shape[-1],
                    IS_NGRAM=(draft_probs is None),
                    base_tolerance=self.base_tolerance,
                )

            return self._build_sampler_output(
                output_token_ids, bonus_sampler_output, sampling_metadata, metadata
            )

    return EntropyAdaptiveRejectionSampler


def _patch_vllm_ascend_envs_module(module) -> None:
    env_vars = getattr(module, "env_variables", None)
    if env_vars is None:
        return
    if "VLLM_EARS_TOLERANCE" in env_vars:
        return
    env_vars["VLLM_EARS_TOLERANCE"] = lambda: float(
        os.getenv("VLLM_EARS_TOLERANCE", "0.0")
    )


def _patch_vllm_ascend_model_runner_module(module) -> None:
    runner_cls = getattr(module, "NPUModelRunner", None)
    if runner_cls is None:
        return

    original_set_up_drafter = runner_cls._set_up_drafter
    if getattr(original_set_up_drafter, "_wings_ears_patched", False):
        return

    def patched_set_up_drafter(self):
        original_set_up_drafter(self)
        # Replace the rejection_sampler with EARS if tolerance > 0
        try:
            import vllm_ascend.envs as envs_ascend
        except ImportError:
            return

        ears_tolerance = getattr(envs_ascend, "VLLM_EARS_TOLERANCE", 0.0)
        if ears_tolerance <= 0:
            return

        spec_method = getattr(getattr(self, "speculative_config", None), "method", None)
        if spec_method not in ("mtp", "eagle3", "suffix"):
            return

        EarsSampler = _get_entropy_adaptive_rejection_sampler_class()
        current_sampler = getattr(self, "sampler", None)
        self.rejection_sampler = EarsSampler(
            current_sampler,
            base_tolerance=ears_tolerance,
        )
        print(
            f"[Wings Engine Patch] EARS enabled: base_tolerance={ears_tolerance}",
            file=sys.stderr,
        )

    patched_set_up_drafter._wings_ears_patched = True  # pylint: disable=protected-access
    runner_cls._set_up_drafter = patched_set_up_drafter


def patch_vllm_ears():
    print(
        "[wins-accel] ears patch enabled",
        file=sys.stderr,
    )
    _register_or_apply_post_import_hook(
        "vllm_ascend.envs",
        _patch_vllm_ascend_envs_module,
    )
    _register_or_apply_post_import_hook(
        "vllm_ascend.worker.model_runner_v1",
        _patch_vllm_ascend_model_runner_module,
    )
