"""Ascend-specific EARS (Entropy Adaptive Rejection Sampling) patches for vLLM."""

import logging
import os
import sys

from wings_engine_patch.patch_vllm_container.v0_17_0.ears_patch import (
    _get_entropy_adaptive_rejection_sampler_class,
    _read_ears_tolerance,
    log_runtime_state,
)

from .ears_ascend_compat import _ASCEND_DRAFT_COMPAT_MODULES
from .ears_ascend_compat import _ASCEND_VLLM_COMPAT_MODULES
from .ears_ascend_compat import patch_vllm_ascend_draft_compat as _patch_vllm_ascend_draft_compat


LOGGER = logging.getLogger("wings_accel.ears_ascend")
_SUPPORTED_EARS_METHODS = {"mtp", "eagle3", "suffix"}


def _register_or_apply_post_import_hook(module_name, patcher) -> None:
    import wrapt

    if module_name in sys.modules:
        patcher(sys.modules[module_name])
    wrapt.register_post_import_hook(patcher, module_name)


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
    log_runtime_state("ears sampler enabled (ascend)", method=method, base_tolerance=tolerance)


def patch_vllm_ascend_draft_compat(module) -> None:
    return _patch_vllm_ascend_draft_compat(module)


def patch_vllm_ears():
    """Enable EARS patches for vllm-ascend."""
    from .ears_ascend_runtime_hooks import register_ascend_runtime_hooks

    log_runtime_state("ears patch enabled (ascend)")
    for module_name in _ASCEND_DRAFT_COMPAT_MODULES + _ASCEND_VLLM_COMPAT_MODULES:
        _register_or_apply_post_import_hook(module_name, patch_vllm_ascend_draft_compat)
    register_ascend_runtime_hooks(_register_or_apply_post_import_hook)
