import copy
import inspect
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedSpeculativeTokenSettings:
    num_speculative_tokens: int | None
    speculative_token_range: list[int] | None


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


def resolve_speculative_token_settings(
    *,
    method: str | None,
    num_speculative_tokens: int | None,
    speculative_token_range: list[int] | None,
) -> ResolvedSpeculativeTokenSettings:
    if speculative_token_range is None:
        return ResolvedSpeculativeTokenSettings(
            num_speculative_tokens=num_speculative_tokens,
            speculative_token_range=None,
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
    )


def _register_or_apply_post_import_hook(module_name, patcher) -> None:
    import wrapt

    if module_name in sys.modules:
        patcher(sys.modules[module_name])
    wrapt.register_post_import_hook(patcher, module_name)


def _patch_metrics_module(module) -> None:
    module.AdaptiveDraftLengthController = AdaptiveDraftLengthController


def _patch_arg_utils_module(module) -> None:
    model_fields = getattr(module.SpeculativeConfig, "model_fields", {})
    if "speculative_token_range" in model_fields:
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
        )
        spec_config.num_speculative_tokens = resolved.num_speculative_tokens
        spec_config.speculative_token_range = resolved.speculative_token_range
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
                num_draft_tokens = self.draft_length * len(valid_sampled_token_count)
                num_accepted_tokens = sum(
                    max(0, count - 1) for count in valid_sampled_token_count
                )
                self.draft_length = controller.observe_iteration(
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted_tokens,
                )
        return result

    patched_update_states_after_model_execute._wings_adaptive_draft_patched = True
    module.GPUModelRunner._update_states_after_model_execute = (
        patched_update_states_after_model_execute
    )


def _patch_spec_decode_eagle_module(module) -> None:
    signature = inspect.signature(module.SpecDecodeBaseProposer.propose)
    if "draft_length" in signature.parameters:
        return

    original_propose = module.SpecDecodeBaseProposer.propose
    if getattr(original_propose, "_wings_adaptive_draft_patched", False):
        return

    def patched_propose(self, *args, draft_length=None, **kwargs):
        if getattr(self, "method", None) != "draft_model":
            return original_propose(self, *args, **kwargs)

        if draft_length is None and getattr(self, "runner", None) is not None:
            draft_length = getattr(self.runner, "draft_length", None)

        full_length = self.num_speculative_tokens
        if draft_length is None or draft_length >= full_length:
            return original_propose(self, *args, **kwargs)

        self.num_speculative_tokens = draft_length
        try:
            draft_token_ids = original_propose(self, *args, **kwargs)
        finally:
            self.num_speculative_tokens = full_length

        if hasattr(draft_token_ids, "shape"):
            import torch

            draft_padding = torch.full(
                (draft_token_ids.shape[0], full_length - draft_token_ids.shape[1]),
                -1,
                dtype=draft_token_ids.dtype,
                device=draft_token_ids.device,
            )
            return torch.cat([draft_token_ids, draft_padding], dim=1)

        return [
            list(token_ids) + [-1] * (full_length - len(token_ids))
            for token_ids in draft_token_ids
        ]

    patched_propose._wings_adaptive_draft_patched = True
    module.SpecDecodeBaseProposer.propose = patched_propose


def patch_vllm_adaptive_draft_model():
    print("[Vllm Patch] adaptive_draft_model patch enabled", file=sys.stderr)
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
