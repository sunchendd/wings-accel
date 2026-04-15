import logging
import sys

from wings_engine_patch.patch_common.ears_core import (
    get_entropy_adaptive_rejection_sampler_class as _get_entropy_adaptive_rejection_sampler_class,
    maybe_enable_sampler as _maybe_enable_ears_sampler_core,
    parse_ears_tolerance as _read_ears_tolerance,
)

LOGGER = logging.getLogger("wings_accel.ears")
_SUPPORTED_EARS_METHODS = {"mtp", "suffix"}


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


def _register_or_apply_post_import_hook(module_name, patcher) -> None:
    import wrapt

    if module_name in sys.modules:
        patcher(sys.modules[module_name])
    wrapt.register_post_import_hook(patcher, module_name)


def _maybe_enable_ears_sampler(runner) -> None:
    tolerance = _read_ears_tolerance({})
    spec_config = getattr(runner, "speculative_config", None)
    method = getattr(spec_config, "method", None)
    if method not in _SUPPORTED_EARS_METHODS:
        return
    if _maybe_enable_ears_sampler_core(runner, base_tolerance=tolerance):
        log_runtime_state("ears sampler enabled", method=method, base_tolerance=tolerance)


def patch_vllm_ears():
    from .ears_nvidia_runtime_hooks import register_nvidia_runtime_hooks

    log_runtime_state("ears patch enabled")
    register_nvidia_runtime_hooks(_register_or_apply_post_import_hook)


_configure_logger()
