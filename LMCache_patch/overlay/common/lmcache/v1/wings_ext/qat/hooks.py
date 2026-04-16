# SPDX-License-Identifier: Apache-2.0

"""QAT environment detection hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import Any
import importlib.util
import importlib
import os
import subprocess
import sys

from lmcache.logging import init_logger
from lmcache.v1.wings_ext.qat.device_probe import (
    KNOWN_QAT_DEVICE_IDS,
    probe_qat_devices,
)
from lmcache.v1.wings_ext.qat.manager import QATCompressionStats, QATKVCacheManager
from lmcache.v1.wings_ext.config import get_wings_feature_config

logger = init_logger(__name__)


def _detect_runtime_accelerator() -> str:
    try:
        runtime_module = importlib.import_module(
            "lmcache.v1.wings_ext.ascend.runtime"
        )
        return str(runtime_module.detect_runtime_accelerator())
    except Exception:
        return "cpu"


@dataclass
class QATProbeResult:
    enabled: bool
    reason: str
    available_devices: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def _default_vendored_kv_agent_root() -> Path:
    return Path(__file__).resolve().parents[4] / "third_party" / "kv-agent"


def _resolve_vendored_kv_agent_root(feature_config: dict[str, Any]) -> Path | None:
    configured_root = (
        feature_config.get("kv_agent_root")
        or os.getenv("WINGS_QAT_KV_AGENT_ROOT")
        or str(_default_vendored_kv_agent_root())
    )
    root = Path(str(configured_root)).resolve()
    return root if root.exists() else None


def _inspect_vendored_kv_agent(root: Path | None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "vendored_source_available": False,
        "vendored_extension_built": False,
        "vendored_runtime_libraries_present": False,
    }
    if root is None:
        return details

    package_dir = root / "kv_agent"
    lib_dir = root / "lib"
    built_extensions = sorted(package_dir.glob("_C*.so"))
    runtime_libraries = sorted(lib_dir.glob("*.so*")) if lib_dir.is_dir() else []

    details.update(
        {
            "vendored_root": str(root),
            "vendored_source_available": (
                root.is_dir()
                and (root / "setup.py").is_file()
                and package_dir.is_dir()
                and (package_dir / "__init__.py").is_file()
                and (root / "kv_agent.cpp").is_file()
            ),
            "vendored_extension_built": bool(built_extensions),
            "vendored_runtime_libraries_present": bool(runtime_libraries),
            "vendored_extension_files": [str(path) for path in built_extensions],
            "vendored_runtime_libraries": [str(path) for path in runtime_libraries],
        }
    )
    return details


def _resolve_kv_agent_runtime(module_name: str, feature_config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {
        "module_name": module_name,
        "module_spec_found": False,
        "runtime_source": "none",
    }

    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        details["module_spec_found"] = True
        details["runtime_source"] = "system"
        details["module_origin"] = spec.origin
        return True, details

    vendored_root = _resolve_vendored_kv_agent_root(feature_config)
    details.update(_inspect_vendored_kv_agent(vendored_root))
    if vendored_root is None:
        return False, details

    if details["vendored_source_available"] and details["vendored_extension_built"]:
        vendored_root_str = str(vendored_root)
        if vendored_root_str not in sys.path:
            sys.path.insert(0, vendored_root_str)
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            details["module_spec_found"] = True
            details["runtime_source"] = "vendored"
            details["module_origin"] = spec.origin
            return True, details

    return False, details


def _probe_device_count() -> int:
    env_value = os.getenv("WINGS_QAT_AVAILABLE_DEVICES")
    if env_value:
        try:
            return max(0, int(env_value))
        except ValueError:
            logger.warning(
                "Ignoring invalid WINGS_QAT_AVAILABLE_DEVICES=%s", env_value
            )

    if which("lspci") is None:
        return 0

    try:
        completed = subprocess.run(
            ["lspci", "-n"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return 0

    if completed.returncode != 0:
        return 0

    count = 0
    for line in completed.stdout.splitlines():
        if any(device_id in line for device_id in KNOWN_QAT_DEVICE_IDS):
            count += 1
    return count


def _initialize_qat_compression_state(
    backend: Any, feature_config: dict[str, Any]
) -> QATCompressionStats:
    stats = QATCompressionStats(
        ratio=float(feature_config.get("initial_compression_ratio", 1.0) or 1.0),
        alpha=float(feature_config.get("compression_ratio_alpha", 0.5) or 0.5),
        min_ratio=float(feature_config.get("min_compression_ratio", 0.3) or 0.3),
        max_ratio=float(feature_config.get("max_compression_ratio", 1.0) or 1.0),
    )
    backend.wings_qat_compression = stats
    return stats


def configure_local_disk_backend_for_qat(backend: Any) -> QATProbeResult:
    feature_config = get_wings_feature_config(backend.engine_config, "qat")
    _initialize_qat_compression_state(backend, feature_config)
    if not bool(feature_config.get("enabled", False)):
        result = QATProbeResult(enabled=False, reason="feature disabled")
        backend.wings_qat_probe = result
        return result

    if _detect_runtime_accelerator() == "npu":
        raise RuntimeError(
            "Wings QAT is not supported on Ascend/NPU. "
            "Disable extra_config.wings.qat.enabled for platform=ascend."
        )

    module_name = str(feature_config.get("module_name", "kv_agent"))
    module_available, module_details = _resolve_kv_agent_runtime(
        module_name, feature_config
    )
    device_probe = probe_qat_devices(
        timeout=int(feature_config.get("device_probe_timeout_s", 30) or 30)
    )
    env_override_device_count = _probe_device_count()
    available_devices = (
        env_override_device_count if env_override_device_count > 0 else device_probe.available_devices
    )
    module_details["device_probe"] = {
        "total_devices": device_probe.total_devices,
        "available_devices": device_probe.available_devices,
        "any_qat_detected": device_probe.any_qat_detected,
        "used_adf_ctl": device_probe.used_adf_ctl,
        "node_stats": device_probe.node_stats,
        "device_details": device_probe.device_details,
        "lspci_matches": device_probe.lspci_matches,
        "message": device_probe.message,
    }
    if env_override_device_count > 0:
        module_details["device_probe"]["env_override_device_count"] = env_override_device_count

    if not module_available:
        reason = f"missing python runtime for module: {module_name}"
        if module_details.get("vendored_source_available") and not module_details.get(
            "vendored_extension_built"
        ):
            reason = (
                "vendored kv_agent source is present, but the extension is not built yet"
            )
        result = QATProbeResult(
            enabled=False,
            reason=reason,
            available_devices=available_devices,
            details=module_details,
        )
    elif available_devices <= 0:
        result = QATProbeResult(
            enabled=False,
            reason="no QAT devices detected",
            available_devices=0,
            details=module_details,
        )
    else:
        try:
            backend.wings_qat_manager = QATKVCacheManager.from_backend(
                backend, module_name=module_name
            )
            result = QATProbeResult(
                enabled=True,
                reason="QAT acceleration available",
                available_devices=available_devices,
                details=module_details,
            )
        except Exception as exc:
            module_details["manager_init_error"] = str(exc)
            result = QATProbeResult(
                enabled=False,
                reason=f"failed to initialize QAT runtime: {exc}",
                available_devices=available_devices,
                details=module_details,
            )

    backend.wings_qat_probe = result
    if result.enabled:
        logger.info(
            "Wings QAT hook enabled with %d available device(s)",
            result.available_devices,
        )
    else:
        logger.warning("Wings QAT hook falling back to default path: %s", result.reason)
    return result


def estimate_qat_physical_size(backend: Any, raw_size: int) -> int:
    probe = getattr(backend, "wings_qat_probe", None)
    stats = getattr(backend, "wings_qat_compression", None)
    if probe is None or not probe.enabled or not isinstance(stats, QATCompressionStats):
        return raw_size
    return stats.estimate_size(raw_size)


def record_qat_persisted_file(
    backend: Any,
    path: str,
    raw_size: int,
    reserved_size: int | None = None,
) -> int:
    manager = getattr(backend, "wings_qat_manager", None)
    stats = getattr(backend, "wings_qat_compression", None)
    if manager is None or not isinstance(stats, QATCompressionStats):
        return raw_size

    actual_size = manager.get_persisted_size(path)
    if actual_size > 0:
        stats.update(raw_size, actual_size)
    else:
        actual_size = raw_size

    if reserved_size is not None and hasattr(backend, "disk_lock"):
        with backend.disk_lock:
            backend.current_cache_size += actual_size - reserved_size

    return actual_size


def save_memory_obj_with_qat(
    backend: Any,
    memory_obj: Any,
    path: str,
) -> bool:
    manager = getattr(backend, "wings_qat_manager", None)
    probe = getattr(backend, "wings_qat_probe", None)
    if manager is None or probe is None or not probe.enabled:
        return False
    if not manager.supports_memory_obj(memory_obj):
        return False
    manager.save_memory_obj(memory_obj, path)
    return True


def load_memory_obj_with_qat(
    backend: Any,
    memory_obj: Any,
    path: str,
) -> bool:
    manager = getattr(backend, "wings_qat_manager", None)
    probe = getattr(backend, "wings_qat_probe", None)
    if manager is None or probe is None or not probe.enabled:
        return False
    if not manager.supports_memory_obj(memory_obj):
        return False
    manager.load_memory_obj(memory_obj, path)
    return True
