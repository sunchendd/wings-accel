"""
Compatibility wrapper for ears_ascend_runtime_hooks.

This module has been moved to wings_engine_patch.patch_vllm_ascend_container.v0_17_0rc1.
This wrapper provides backward compatibility.
"""

from wings_engine_patch.patch_vllm_ascend_container.v0_17_0rc1.ears_ascend_runtime_hooks import (
    _maybe_enable_ears_sampler,
    _patch_vllm_ascend_envs_module,
    _patch_vllm_ascend_model_runner_module,
    register_ascend_runtime_hooks,
)

__all__ = [
    "_maybe_enable_ears_sampler",
    "_patch_vllm_ascend_envs_module",
    "_patch_vllm_ascend_model_runner_module",
    "register_ascend_runtime_hooks",
]
