"""
Connector-level tests for LocalStoreKVStore eviction semantics.

These tests validate that LocalStore behaves like a CPU-side analogue of vLLM's
KVCacheManager/BlockPool:

- Running requests "pin" blocks (refcount > 0), making them non-evictable.
- Finished requests release pins, making blocks eligible for LRU eviction.
- Eviction updates both:
  - the worker-side C++ LocalStore (real KV data), and
  - the scheduler-side shared index (lookup view used for prefix hits).
"""

import os
import uuid
from multiprocessing import shared_memory

import pytest
import torch


def _mk_ids(n: int) -> list[bytes]:
    return [bytes([i & 0xFF]) * 16 for i in range(n)]


@pytest.fixture
def unique_id():
    return f"pytest_localstore_eviction_{uuid.uuid4().hex}"


@pytest.fixture
def cleanup_shared_index(unique_id):
    yield
    from vsparse.store.localstore.localstore_connector import _sanitize_shm_name

    shm_name = f"vllm_sparse_localstore_idx_{_sanitize_shm_name(unique_id)}"
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
    except FileNotFoundError:
        shm = None
    if shm is not None:
        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    lock_path = f"/tmp/vllm_sparse_localstore_{_sanitize_shm_name(shm_name)}.lock"
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _make_worker_store(unique_id: str, *, capacity: int):
    from vsparse.store.localstore.localstore_connector import LocalStoreKVStore

    cfg = {
        "role": "worker",
        "device": 0,
        "capacity": int(capacity),
        "io_size": 4096,
        "tensor_size": 1,
        "unique_id": unique_id,
        "shared_index_enable": True,
        "index_capacity": 4096,
    }
    return LocalStoreKVStore(cfg)


def _make_scheduler_store(unique_id: str):
    from vsparse.store.localstore.localstore_connector import LocalStoreKVStore

    cfg = {
        "role": "scheduler",
        "tensor_size": 1,
        "is_mla": False,
        "unique_id": unique_id,
        "shared_index_enable": True,
        "index_capacity": 4096,
    }
    return LocalStoreKVStore(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for LocalStore tests.")
def test_lru_eviction_updates_shared_index(cleanup_shared_index, unique_id):
    worker_store = _make_worker_store(unique_id, capacity=2)
    scheduler_store = _make_scheduler_store(unique_id)

    a, b, c = _mk_ids(3)

    worker_store.create([a])
    # Simulate an offload pipeline:
    # 1) allocate storage slot
    # 2) commit success to make the block "ready" and visible to lookup()
    worker_store.commit([a], True)
    worker_store.create([b])
    worker_store.commit([b], True)

    assert worker_store.lookup([a])[0] is True
    assert scheduler_store.lookup([a])[0] is True

    # Store is at capacity=2. Reserving a new block must evict the LRU block.
    worker_store.before_dump("r0", [c])
    worker_store.create([c])
    worker_store.commit([c], True)

    assert worker_store.lookup([a])[0] is False
    assert scheduler_store.lookup([a])[0] is False
    assert worker_store.lookup([b])[0] is True
    assert scheduler_store.lookup([b])[0] is True
    assert worker_store.lookup([c])[0] is True
    assert scheduler_store.lookup([c])[0] is True

    worker_store.request_finished("r0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for LocalStore tests.")
def test_pinned_blocks_are_not_evicted(cleanup_shared_index, unique_id):
    worker_store = _make_worker_store(unique_id, capacity=2)
    scheduler_store = _make_scheduler_store(unique_id)

    a, b, c = _mk_ids(3)
    worker_store.create([a])
    worker_store.commit([a], True)
    worker_store.create([b])
    worker_store.commit([b], True)

    # Pin A for a running request. A should never be selected as an eviction victim.
    worker_store.pin_blocks("r1", [a])
    worker_store.before_dump("r2", [c])
    worker_store.create([c])
    worker_store.commit([c], True)

    assert worker_store.lookup([a])[0] is True
    assert scheduler_store.lookup([a])[0] is True
    assert worker_store.lookup([b])[0] is False
    assert scheduler_store.lookup([b])[0] is False

    worker_store.request_finished("r2")
    worker_store.request_finished("r1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for LocalStore tests.")
def test_request_finished_unpins_and_allows_eviction(cleanup_shared_index, unique_id):
    worker_store = _make_worker_store(unique_id, capacity=2)

    a, b, c = _mk_ids(3)
    worker_store.create([a])
    worker_store.commit([a], True)
    worker_store.create([b])
    worker_store.commit([b], True)

    # Pin both blocks, then release them on request finish. After that, one of them
    # must be evicted to make room for C.
    worker_store.pin_blocks("r1", [a, b])
    worker_store.request_finished("r1")

    worker_store.before_dump("r2", [c])
    worker_store.create([c])
    worker_store.commit([c], True)

    remaining = sum(bool(x) for x in worker_store.lookup([a, b]))
    assert remaining == 1
    assert worker_store.lookup([c])[0] is True

    worker_store.request_finished("r2")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for LocalStore tests.")
def test_reservation_prevents_oversubscription(cleanup_shared_index, unique_id):
    worker_store = _make_worker_store(unique_id, capacity=1)
    a, b = _mk_ids(2)

    # Capacity=1: once A is reserved by r1, reserving another new block should fail
    # because there is no evictable block.
    worker_store.before_dump("r1", [a])
    with pytest.raises(RuntimeError):
        worker_store.before_dump("r2", [b])

    worker_store.after_dump_fail("r1", [a])
    worker_store.commit([a], is_success=False)
