# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
import os
import threading

import torch

from lmcache.utils import CacheEngineKey, DiskCacheMetadata
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.wings_ext.cold_start import hooks as cold_start_hooks
from lmcache.v1.wings_ext.cold_start import (
    initialize_manifest_state,
    note_manifest_remove,
    note_manifest_write,
    save_manifest,
)


class _DummyPolicy:
    def update_on_put(self, key):
        del key


class _DummyStats:
    def update_local_storage_usage(self, usage):
        self.usage = usage


def _make_backend(storage_dir, *, worker_id=0, world_size=1):
    return SimpleNamespace(
        path=str(storage_dir),
        max_cache_size=1024 * 1024,
        current_cache_size=0,
        usage=0,
        dict=OrderedDict(),
        metadata=SimpleNamespace(worker_id=worker_id, world_size=world_size),
        cache_policy=_DummyPolicy(),
        stats_monitor=_DummyStats(),
        engine_config=SimpleNamespace(
            extra_config={
                "wings": {
                    "cold_start": {
                        "enabled": True,
                        "manifest_path": os.path.join(
                            str(storage_dir), "cold-start-manifest.json"
                        ),
                        "manifest_write_interval": 1,
                    }
                }
            }
        ),
    )


def test_cold_start_manifest_roundtrip(tmp_path):
    backend = _make_backend(tmp_path)
    summary = initialize_manifest_state(backend)
    assert summary.enabled is True
    assert summary.restored_entries == 0

    payload_path = tmp_path / "chunk-1.pt"
    payload_path.write_bytes(b"payload")

    key = CacheEngineKey("model", 1, 0, 1234, torch.float16)
    backend.dict[key] = DiskCacheMetadata(
        path=str(payload_path),
        size=7,
        shape=torch.Size([2, 4, 8]),
        dtype=torch.float16,
        cached_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        fmt=MemoryFormat.KV_T2D,
        pin_count=0,
    )
    backend.current_cache_size = 7

    note_manifest_write(backend, key)
    save_manifest(backend, force=True)

    restored_backend = _make_backend(tmp_path)
    restore_summary = initialize_manifest_state(restored_backend)
    assert restore_summary.restored_entries == 1
    restored_meta = restored_backend.dict[key]
    assert restored_meta.path == str(payload_path)
    assert restored_meta.size == 7
    assert restored_meta.shape == torch.Size([2, 4, 8])
    assert restored_meta.dtype == torch.float16
    assert restored_meta.fmt == MemoryFormat.KV_T2D
    assert restored_meta.cached_positions.tolist() == [0, 1, 2]


def test_cold_start_manifest_remove_updates_snapshot(tmp_path):
    backend = _make_backend(tmp_path)
    initialize_manifest_state(backend)

    payload_path = tmp_path / "chunk-2.pt"
    payload_path.write_bytes(b"payload")

    key = CacheEngineKey("model", 1, 0, 5678, torch.float16)
    backend.dict[key] = DiskCacheMetadata(
        path=str(payload_path),
        size=7,
        shape=torch.Size([2, 4, 8]),
        dtype=torch.float16,
        cached_positions=None,
        fmt=MemoryFormat.KV_T2D,
        pin_count=0,
    )
    backend.current_cache_size = 7

    note_manifest_write(backend, key)
    backend.dict.pop(key)
    backend.current_cache_size = 0
    note_manifest_remove(backend, key)
    save_manifest(backend, force=True)

    restored_backend = _make_backend(tmp_path)
    restore_summary = initialize_manifest_state(restored_backend)
    assert restore_summary.restored_entries == 0
    assert key not in restored_backend.dict


def test_cold_start_manifest_concurrent_saves_use_unique_temp_files(tmp_path, monkeypatch):
    backend_a = _make_backend(tmp_path)
    backend_b = _make_backend(tmp_path)
    initialize_manifest_state(backend_a)
    initialize_manifest_state(backend_b)

    replace_sources = []
    replace_lock = threading.Lock()
    replace_barrier = threading.Barrier(2)

    def _fake_replace(src, dst):
        del dst
        with replace_lock:
            replace_sources.append(str(src))
        replace_barrier.wait(timeout=5)
        src_path = os.fspath(src)
        if os.path.exists(src_path):
            os.unlink(src_path)

    monkeypatch.setattr(cold_start_hooks.os, "replace", _fake_replace)

    threads = [
        threading.Thread(target=save_manifest, args=(backend, True))
        for backend in (backend_a, backend_b)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(replace_sources) == 2
    assert len(set(replace_sources)) == 2


def test_cold_start_manifest_is_worker_scoped_for_multi_worker(tmp_path):
    backend_rank0 = _make_backend(tmp_path, worker_id=0, world_size=2)
    backend_rank1 = _make_backend(tmp_path, worker_id=1, world_size=2)

    summary_rank0 = initialize_manifest_state(backend_rank0)
    summary_rank1 = initialize_manifest_state(backend_rank1)

    assert summary_rank0.manifest_path.endswith("cold-start-manifest.worker0.json")
    assert summary_rank1.manifest_path.endswith("cold-start-manifest.worker1.json")
    assert summary_rank0.manifest_path != summary_rank1.manifest_path


def test_cold_start_manifest_restore_isolated_per_worker(tmp_path):
    rank0_backend = _make_backend(tmp_path, worker_id=0, world_size=2)
    rank1_backend = _make_backend(tmp_path, worker_id=1, world_size=2)
    initialize_manifest_state(rank0_backend)
    initialize_manifest_state(rank1_backend)

    rank0_payload = tmp_path / "chunk-rank0.pt"
    rank0_payload.write_bytes(b"payload-0")
    rank0_key = CacheEngineKey("model", 2, 0, 1111, torch.float16)
    rank0_backend.dict[rank0_key] = DiskCacheMetadata(
        path=str(rank0_payload),
        size=9,
        shape=torch.Size([2, 4, 8]),
        dtype=torch.float16,
        cached_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        fmt=MemoryFormat.KV_T2D,
        pin_count=0,
    )
    rank0_backend.current_cache_size = 9
    note_manifest_write(rank0_backend, rank0_key)
    save_manifest(rank0_backend, force=True)

    rank1_payload = tmp_path / "chunk-rank1.pt"
    rank1_payload.write_bytes(b"payload-1")
    rank1_key = CacheEngineKey("model", 2, 1, 2222, torch.float16)
    rank1_backend.dict[rank1_key] = DiskCacheMetadata(
        path=str(rank1_payload),
        size=9,
        shape=torch.Size([2, 4, 8]),
        dtype=torch.float16,
        cached_positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        fmt=MemoryFormat.KV_T2D,
        pin_count=0,
    )
    rank1_backend.current_cache_size = 9
    note_manifest_write(rank1_backend, rank1_key)
    save_manifest(rank1_backend, force=True)

    restored_rank0 = _make_backend(tmp_path, worker_id=0, world_size=2)
    restored_rank1 = _make_backend(tmp_path, worker_id=1, world_size=2)
    summary_rank0 = initialize_manifest_state(restored_rank0)
    summary_rank1 = initialize_manifest_state(restored_rank1)

    assert summary_rank0.restored_entries == 1
    assert summary_rank1.restored_entries == 1
    assert rank0_key in restored_rank0.dict
    assert rank1_key not in restored_rank0.dict
    assert rank1_key in restored_rank1.dict
    assert rank0_key not in restored_rank1.dict
