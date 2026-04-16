import logging
import sys

from wings_engine_patch.patch_common.ears_core import (
    SUPPORTED_EARS_METHODS,
    get_entropy_adaptive_rejection_sampler_class as _get_entropy_adaptive_rejection_sampler_class,
    maybe_enable_sampler as _maybe_enable_ears_sampler_core,
    parse_ears_tolerance as _read_ears_tolerance,
)

from .ears_ascend_compat import _ASCEND_DRAFT_COMPAT_MODULES
from .ears_ascend_compat import _ASCEND_VLLM_COMPAT_MODULES
from .ears_ascend_compat import patch_vllm_ascend_draft_compat as _patch_vllm_ascend_draft_compat


LOGGER = logging.getLogger("wings_accel.ears")
_SUPPORTED_EARS_METHODS = SUPPORTED_EARS_METHODS  # Re-export for backward compatibility


def _torch():
    """Re-export for backward compatibility with existing tests."""
    from wings_engine_patch.patch_common.ears_core import _torch as core_torch

    return core_torch()


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


# Keep _sample_recovered_tokens_pytorch for existing tests
def _sample_recovered_tokens_pytorch(
    *,
    num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    sampling_metadata,
):
    from wings_engine_patch.patch_common.ears_core import _torch

    torch = _torch()
    if draft_token_ids.numel() == 0:
        return torch.empty_like(draft_token_ids)

    device = target_probs.device
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()
    for req_idx, generator in sampling_metadata.generators.items():
        if num_draft_tokens[req_idx] > 0:
            q[req_idx].exponential_(generator=generator)
    inv_q = q.reciprocal()

    req_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        torch.tensor(num_draft_tokens, device=device),
    )

    if draft_probs is None:
        probs = target_probs.clone()
        probs[torch.arange(draft_token_ids.shape[0], device=device), draft_token_ids] = 0.0
    else:
        probs = torch.clamp(target_probs - draft_probs, min=0.0)

    scores = probs * inv_q[req_ids]
    return scores.argmax(dim=-1).to(draft_token_ids.dtype)


def _register_or_apply_post_import_hook(module_name, patcher) -> None:
    import wrapt

    if module_name in sys.modules:
        patcher(sys.modules[module_name])
    wrapt.register_post_import_hook(patcher, module_name)


def _maybe_enable_ears_sampler(runner) -> None:
    tolerance = _read_ears_tolerance({})
    if _maybe_enable_ears_sampler_core(runner, base_tolerance=tolerance):
        spec_config = getattr(runner, "speculative_config", None)
        method = getattr(spec_config, "method", None)
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
