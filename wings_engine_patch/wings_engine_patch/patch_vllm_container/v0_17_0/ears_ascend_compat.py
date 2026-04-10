"""
Compatibility wrapper for ears_ascend_compat.

This module has been moved to wings_engine_patch.patch_vllm_ascend_container.v0_17_0rc1.
This wrapper provides backward compatibility.
"""

from wings_engine_patch.patch_vllm_ascend_container.v0_17_0rc1.ears_ascend_compat import (
    _ASCEND_DRAFT_COMPAT_MODULES,
    _ASCEND_VLLM_COMPAT_MODULES,
    _PATCHED_ATTR,
    _SUPPORTED_EARS_METHODS,
    patch_vllm_ascend_draft_compat,
)

__all__ = [
    "_ASCEND_DRAFT_COMPAT_MODULES",
    "_ASCEND_VLLM_COMPAT_MODULES",
    "_PATCHED_ATTR",
    "_SUPPORTED_EARS_METHODS",
    "patch_vllm_ascend_draft_compat",
]
