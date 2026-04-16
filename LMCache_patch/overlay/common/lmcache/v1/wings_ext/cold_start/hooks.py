# SPDX-License-Identifier: Apache-2.0

"""Manifest-based local-disk cold-start hooks for Wings LMCache integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import json
import os
import tempfile

import torch

from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.wings_ext.config import get_wings_feature_config

logger = init_logger(__name__)

MANIFEST_VERSION = 1
DEFAULT_MANIFEST_NAME = ".wings_cold_start_manifest.json"


@dataclass
class ColdStartSummary:
    enabled: bool
    manifest_path: Optional[str] = None
    scanned_manifests: int = 0
    restored_entries: int = 0
    skipped_entries: int = 0
    errors: int = 0


@dataclass
class ColdStartManifestState:
    enabled: bool
    manifest_path: Optional[str]
    write_interval: int = 100
    writes_since_save: int = 0
    dirty: bool = False


def _tensor_to_list(value: Optional[torch.Tensor]) -> Optional[list[int]]:
    if value is None:
        return None
    return value.detach().cpu().tolist()


def _dtype_from_string(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.replace("torch.", "")
    return getattr(torch, normalized)


def _manifest_state(backend: Any) -> ColdStartManifestState:
    state = getattr(backend, "_wings_cold_start_state", None)
    if state is None:
        state = ColdStartManifestState(enabled=False, manifest_path=None)
        backend._wings_cold_start_state = state
    return state


def _default_manifest_path(backend: Any) -> str:
    return os.path.join(backend.path, DEFAULT_MANIFEST_NAME)


def _worker_scoped_manifest_path(backend: Any, manifest_path: str) -> str:
    metadata = getattr(backend, "metadata", None)
    worker_id = getattr(metadata, "worker_id", None)
    world_size = int(getattr(metadata, "world_size", 1) or 1)

    if worker_id is None or world_size <= 1:
        return manifest_path

    manifest = Path(manifest_path)
    if manifest.suffix:
        worker_name = f"{manifest.stem}.worker{worker_id}{manifest.suffix}"
    else:
        worker_name = f"{manifest.name}.worker{worker_id}"
    return str(manifest.with_name(worker_name))


def _manifest_entry_from_metadata(key: CacheEngineKey, metadata: DiskCacheMetadata) -> dict[str, object]:
    return {
        "payload_path": metadata.path,
        "size": metadata.size,
        "shape": list(metadata.shape) if metadata.shape is not None else None,
        "dtype": str(metadata.dtype) if metadata.dtype is not None else None,
        "fmt": metadata.fmt.value if metadata.fmt is not None else None,
        "cached_positions": _tensor_to_list(metadata.cached_positions),
    }


def _build_manifest_snapshot(backend: Any) -> dict[str, object]:
    current_cache_size = sum(metadata.size for metadata in backend.dict.values())
    return {
        "version": MANIFEST_VERSION,
        "hash_algorithm": getattr(backend.engine_config, "pre_caching_hash_algorithm", None),
        "storage_path": os.path.abspath(backend.path),
        "max_cache_size": int(backend.max_cache_size),
        "current_cache_size": int(current_cache_size),
        "lru_keys": [key.to_string() for key in backend.dict.keys()],
        "entries": {
            key.to_string(): _manifest_entry_from_metadata(key, metadata)
            for key, metadata in backend.dict.items()
        },
    }


def _manifest_uses_bytes_chunk_hash(
    backend: Any,
    payload: Optional[dict[str, object]] = None,
) -> bool:
    hash_algorithm = None
    if payload is not None:
        hash_algorithm = payload.get("hash_algorithm")
    if hash_algorithm is None:
        hash_algorithm = getattr(backend.engine_config, "pre_caching_hash_algorithm", None)
    return hash_algorithm in {"sha256", "sha256_cbor", "xxhash", "xxhash_cbor"}


def _cache_engine_key_from_manifest(
    backend: Any,
    key_str: str,
    payload: Optional[dict[str, object]] = None,
) -> CacheEngineKey:
    key = CacheEngineKey.from_string(key_str)
    if _manifest_uses_bytes_chunk_hash(backend, payload):
        chunk_hash_str = key_str.split("@")[3]
        if not chunk_hash_str.startswith("-"):
            key.chunk_hash = bytes.fromhex(chunk_hash_str)
    return key


def save_manifest(backend: Any, force: bool = False) -> None:
    state = _manifest_state(backend)
    if not state.enabled or not state.manifest_path:
        return
    if not force and not state.dirty:
        return

    manifest_path = Path(state.manifest_path)
    tmp_path: Optional[Path] = None
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_manifest_snapshot(backend)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f"{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        os.replace(tmp_path, manifest_path)
        state.writes_since_save = 0
        state.dirty = False
    except Exception:
        logger.exception("Failed to save Wings cold-start manifest to %s", state.manifest_path)
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _maybe_save_manifest(backend: Any) -> None:
    state = _manifest_state(backend)
    if not state.enabled:
        return
    state.writes_since_save += 1
    state.dirty = True
    if state.write_interval > 0 and state.writes_since_save >= state.write_interval:
        save_manifest(backend, force=True)


def note_manifest_write(backend: Any, key: CacheEngineKey) -> None:
    if key not in backend.dict:
        return
    _maybe_save_manifest(backend)


def note_manifest_remove(backend: Any, key: CacheEngineKey) -> None:
    del key
    _maybe_save_manifest(backend)


def initialize_manifest_state(backend: Any) -> ColdStartSummary:
    feature_config = get_wings_feature_config(backend.engine_config, "cold_start")
    enabled = bool(feature_config.get("enabled", False))
    manifest_path = str(feature_config.get("manifest_path") or _default_manifest_path(backend))
    manifest_path = _worker_scoped_manifest_path(backend, manifest_path)
    write_interval = int(feature_config.get("manifest_write_interval", 100) or 0)
    state = ColdStartManifestState(
        enabled=enabled,
        manifest_path=manifest_path if enabled else None,
        write_interval=write_interval,
    )
    backend._wings_cold_start_state = state

    summary = ColdStartSummary(enabled=enabled, manifest_path=state.manifest_path)
    if not enabled:
        return summary

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        logger.info("No Wings cold-start manifest found at %s", manifest_path)
        return summary

    try:
        payload = json.loads(manifest_file.read_text())
    except Exception:
        logger.exception("Failed to load Wings cold-start manifest from %s", manifest_path)
        summary.errors += 1
        return summary

    summary.scanned_manifests = 1
    if not isinstance(payload, dict) or payload.get("version") != MANIFEST_VERSION:
        logger.warning("Ignoring incompatible Wings cold-start manifest at %s", manifest_path)
        summary.skipped_entries += 1
        return summary

    manifest_storage_path = payload.get("storage_path")
    if manifest_storage_path and os.path.abspath(str(manifest_storage_path)) != os.path.abspath(backend.path):
        logger.warning(
            "Ignoring cold-start manifest because storage_path mismatched: %s != %s",
            manifest_storage_path,
            backend.path,
        )
        summary.skipped_entries += 1
        return summary

    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        logger.warning("Ignoring cold-start manifest with invalid entries: %s", manifest_path)
        summary.skipped_entries += 1
        return summary

    ordered_keys = payload.get("lru_keys", [])
    if not isinstance(ordered_keys, list) or not ordered_keys:
        ordered_keys = list(entries.keys())

    for key_str in ordered_keys:
        try:
            if key_str not in entries:
                continue
            manifest = entries[key_str]
            if not isinstance(manifest, dict):
                summary.skipped_entries += 1
                continue

            payload_path = str(manifest["payload_path"])
            if not Path(payload_path).is_file():
                logger.warning(
                    "Skipping cold-start entry because payload is missing: %s",
                    payload_path,
                )
                summary.skipped_entries += 1
                continue

            key = _cache_engine_key_from_manifest(backend, key_str, payload)
            metadata = DiskCacheMetadata(
                path=payload_path,
                size=int(manifest["size"]),
                shape=torch.Size(manifest["shape"]) if manifest.get("shape") is not None else None,
                dtype=_dtype_from_string(manifest["dtype"]) if manifest.get("dtype") is not None else None,
                cached_positions=(
                    torch.tensor(manifest["cached_positions"], dtype=torch.int64)
                    if manifest.get("cached_positions") is not None
                    else None
                ),
                fmt=MemoryFormat(int(manifest["fmt"])) if manifest.get("fmt") is not None else MemoryFormat.UNDEFINED,
                pin_count=0,
            )

            if key in backend.dict:
                summary.skipped_entries += 1
                continue

            backend.dict[key] = metadata
            backend.cache_policy.update_on_put(key)
            backend.current_cache_size += metadata.size
            summary.restored_entries += 1
        except Exception:
            logger.exception("Failed to rebuild cold-start metadata from key %s", key_str)
            summary.errors += 1

    backend.usage = backend.current_cache_size
    backend.stats_monitor.update_local_storage_usage(backend.usage)
    logger.info(
        "Cold-start manifest restore finished: restored=%d scanned=%d skipped=%d errors=%d path=%s",
        summary.restored_entries,
        summary.scanned_manifests,
        summary.skipped_entries,
        summary.errors,
        summary.manifest_path,
    )
    return summary
