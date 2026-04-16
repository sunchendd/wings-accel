# SPDX-License-Identifier: Apache-2.0

"""Ascend runtime helpers used by the first NPU integration pass."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import os

import torch

from lmcache.logging import init_logger

logger = init_logger(__name__)

_DEVICE_ALIASES = {
    "ascend": "npu",
    "cuda": "cuda",
    "cpu": "cpu",
    "npu": "npu",
    "xpu": "xpu",
}


def _normalize_device_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return _DEVICE_ALIASES.get(text, text)


def _call_platform_flag(current_platform: Any, method_name: str) -> bool:
    method = getattr(current_platform, method_name, None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            logger.debug("platform.%s() raised during device detection", method_name)
    return False


def _torch_device_api(device_name: str):
    normalized = _normalize_device_name(device_name)
    if normalized == "cuda":
        return torch.cuda
    if normalized == "xpu":
        return getattr(torch, "xpu", None)
    if normalized == "npu":
        return getattr(torch, "npu", None)
    return None


def _is_device_available(device_name: str) -> bool:
    api = _torch_device_api(device_name)
    if api is None or not hasattr(api, "is_available"):
        return False
    try:
        return bool(api.is_available())
    except Exception:
        return False


def detect_runtime_accelerator(current_platform: Any | None = None) -> str:
    forced = _normalize_device_name(os.getenv("WINGS_FORCE_DEVICE_PLATFORM"))
    if forced is not None:
        return forced

    if current_platform is not None:
        if _call_platform_flag(current_platform, "is_cuda_alike"):
            return "cuda"
        if _call_platform_flag(current_platform, "is_xpu"):
            return "xpu"
        if _call_platform_flag(current_platform, "is_npu") or _call_platform_flag(
            current_platform, "is_ascend"
        ):
            return "npu"

        for attr_name in ("device_type", "platform_type", "device_name"):
            normalized = _normalize_device_name(getattr(current_platform, attr_name, None))
            if normalized in {"cuda", "xpu", "npu"}:
                return normalized

        platform_name = type(current_platform).__name__.lower()
        if "ascend" in platform_name or "npu" in platform_name:
            return "npu"

    for candidate in ("npu", "cuda", "xpu"):
        if _is_device_available(candidate):
            return candidate

    return "cpu"


def get_runtime_torch_device(current_platform: Any | None = None):
    device_name = detect_runtime_accelerator(current_platform)
    if device_name == "cpu":
        raise RuntimeError("No accelerator device is available for LMCache engine.")

    api = _torch_device_api(device_name)
    if api is None:
        raise RuntimeError(f"Unsupported accelerator device: {device_name}")
    return api, device_name


def is_storage_accelerator_available(dst_device: str) -> bool:
    normalized = _normalize_device_name(str(dst_device).split(":", 1)[0])
    if normalized not in {"cuda", "xpu", "npu"}:
        return False
    return _is_device_available(normalized)


def resolve_backend_dst_device(
    metadata: Any | None,
    dst_device: str = "cuda",
    current_platform: Any | None = None,
) -> str:
    runtime_accelerator = detect_runtime_accelerator(current_platform)
    if metadata is not None and getattr(metadata, "role", None) != "scheduler":
        if runtime_accelerator in {"cuda", "xpu", "npu"}:
            api = _torch_device_api(runtime_accelerator)
            assert api is not None
            return f"{runtime_accelerator}:{api.current_device()}"

    requested = _normalize_device_name(str(dst_device).split(":", 1)[0])
    if requested == "cuda" and runtime_accelerator == "npu":
        requested = "npu"
    if requested in {"cuda", "xpu", "npu"} and _is_device_available(requested):
        api = _torch_device_api(requested)
        assert api is not None
        return f"{requested}:{api.current_device()}"

    return "cpu"


def create_runtime_stream(current_platform: Any | None = None):
    accelerator = detect_runtime_accelerator(current_platform)
    api = _torch_device_api(accelerator)
    if api is None or not hasattr(api, "Stream"):
        return None
    return api.Stream()


def get_manual_numa_mapping(extra_config: Mapping[str, Any] | None) -> dict[int, int] | None:
    if not isinstance(extra_config, Mapping):
        return None

    for key in (
        "gpu_to_numa_mapping",
        "npu_to_numa_mapping",
        "device_to_numa_mapping",
    ):
        value = extra_config.get(key)
        if isinstance(value, Mapping):
            return {int(k): int(v) for k, v in value.items()}

    wings_config = extra_config.get("wings")
    if isinstance(wings_config, Mapping):
        ascend_config = wings_config.get("ascend")
        if isinstance(ascend_config, Mapping):
            value = ascend_config.get("device_to_numa_mapping")
            if isinstance(value, Mapping):
                return {int(k): int(v) for k, v in value.items()}

    return None


def resolve_physical_device_id(
    current_platform: Any | None,
    device_index: int,
) -> int:
    if current_platform is None:
        return device_index

    method = getattr(current_platform, "device_id_to_physical_device_id", None)
    if callable(method):
        try:
            return int(method(device_index))
        except Exception:
            logger.warning(
                "Failed to resolve physical device id for accelerator device %s",
                device_index,
            )
    return device_index
