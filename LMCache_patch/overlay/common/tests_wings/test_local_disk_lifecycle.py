# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
import os

import torch

from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.wings_ext.cold_start import (
    initialize_manifest_state,
    note_manifest_write,
    save_manifest,
)


class _DummyPolicy:
    def update_on_put(self, key):
        del key


class _DummyStats:
    def __init__(self):
        self.usage = 0

    def update_local_storage_usage(self, usage):
        self.usage = usage


def _make_backend(storage_dir, *, manifest_path=None, backend_path=None):
    return SimpleNamespace(
        path=str(backend_path or storage_dir),
        max_cache_size=1024 * 1024,
        current_cache_size=0,
        usage=0,
        dict=OrderedDict(),
        cache_policy=_DummyPolicy(),
        stats_monitor=_DummyStats(),
        engine_config=SimpleNamespace(
            extra_config={
                "wings": {
                    "cold_start": {
                        "enabled": True,
                        "manifest_path": str(
                            manifest_path
                            or os.path.join(str(storage_dir), "cold-start-manifest.json")
                        ),
                        "manifest_write_interval": 1,
                    }
                }
            }
        ),
    )


def _write_disk_entry(backend, payload_path, *, chunk_hash=1234, size=7):
    key = CacheEngineKey("model", 1, 0, chunk_hash, torch.float16)
    backend.dict[key] = DiskCacheMetadata(
        path=str(payload_path),
        size=size,
        shape=torch.Size([2, 4, 8]),
        dtype=torch.float16,
        cached_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        fmt=MemoryFormat.KV_T2D,
        pin_count=0,
    )
    backend.current_cache_size = size
    backend.usage = size
    note_manifest_write(backend, key)
    save_manifest(backend, force=True)
    return key


def test_local_disk_lifecycle_restore_rebuilds_usage_accounting(tmp_path):
    backend = _make_backend(tmp_path)
    initialize_manifest_state(backend)

    payload_path = tmp_path / "chunk-restore.pt"
    payload_path.write_bytes(b"payload")
    key = _write_disk_entry(backend, payload_path)

    restored_backend = _make_backend(tmp_path)
    summary = initialize_manifest_state(restored_backend)

    assert summary.restored_entries == 1
    assert restored_backend.current_cache_size == 7
    assert restored_backend.usage == 7
    assert restored_backend.stats_monitor.usage == 7
    assert restored_backend.dict[key].path == str(payload_path)


def test_local_disk_lifecycle_restore_skips_missing_payload(tmp_path):
    backend = _make_backend(tmp_path)
    initialize_manifest_state(backend)

    payload_path = tmp_path / "chunk-missing.pt"
    payload_path.write_bytes(b"payload")
    _write_disk_entry(backend, payload_path, chunk_hash=5678)
    payload_path.unlink()

    restored_backend = _make_backend(tmp_path)
    summary = initialize_manifest_state(restored_backend)

    assert summary.restored_entries == 0
    assert summary.skipped_entries == 1
    assert restored_backend.current_cache_size == 0
    assert restored_backend.usage == 0


def test_local_disk_lifecycle_manifest_rejects_storage_path_mismatch(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest_path = source_dir / "cold-start-manifest.json"

    backend = _make_backend(source_dir, manifest_path=manifest_path)
    initialize_manifest_state(backend)

    payload_path = source_dir / "chunk-path-mismatch.pt"
    payload_path.write_bytes(b"payload")
    _write_disk_entry(backend, payload_path, chunk_hash=9012)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    restored_backend = _make_backend(
        other_dir,
        manifest_path=manifest_path,
        backend_path=other_dir,
    )
    summary = initialize_manifest_state(restored_backend)

    assert summary.restored_entries == 0
    assert summary.skipped_entries == 1
    assert len(restored_backend.dict) == 0
