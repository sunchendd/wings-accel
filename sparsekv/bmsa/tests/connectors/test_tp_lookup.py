import os
import time
import uuid
from multiprocessing import shared_memory

import pytest
import torch


class CountingLookupStore:
    def __init__(self, store):
        self._store = store
        self.calls: list[list[bytes]] = []

    def lookup(self, block_ids: list[bytes]) -> list[bool]:
        self.calls.append(list(block_ids))
        return self._store.lookup(block_ids)


class CountingHasher:
    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def __call__(self, x: bytes) -> bytes:
        self.calls += 1
        return self._fn(x)


def _identity(x: bytes) -> bytes:
    return x


def _suffix(suffix: bytes):
    def _h(x: bytes) -> bytes:
        if not x:
            return x
        b0 = bytes([x[0] ^ suffix[0]])
        return b0 + x[1:]

    return _h


@pytest.fixture
def unique_id():
    return f"pytest_tp_lookup_{uuid.uuid4().hex}"


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


@pytest.fixture
def worker_store(unique_id, cleanup_shared_index):
    from vsparse.store.localstore.localstore_connector import LocalStoreKVStore

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for LocalStoreKVStore tests.")

    cfg = {
        "role": "worker",
        "device": 0,
        "capacity": 8,
        "io_size": 4096,
        "tensor_size": 1,
        "unique_id": unique_id,
        "shared_index_enable": True,
        "index_capacity": 4096,
    }
    return LocalStoreKVStore(cfg)


@pytest.fixture
def scheduler_store(unique_id, cleanup_shared_index):
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


def _mk_ids(n: int) -> list[bytes]:
    return [bytes([i & 0xFF]) * 16 for i in range(n)]


def test_tp1_single_lookup_contiguous_prefix(worker_store, scheduler_store):
    from vsparse.connector.sparse_connector import (
        _count_contiguous_external_hits_tp_aware,
    )

    ids = _mk_ids(4)
    worker_store.commit(ids[:2], True)
    store = CountingLookupStore(scheduler_store)
    hits = _count_contiguous_external_hits_tp_aware(
        store=store, logical_block_ids=ids, tp_rank_hashers=[_identity]
    )
    assert hits == 2
    assert len(store.calls) == 1
    assert store.calls[0] == ids


def test_tp2_all_ranks_and_hits(worker_store, scheduler_store):
    from vsparse.connector.sparse_connector import (
        _count_contiguous_external_hits_tp_aware,
    )

    ids = _mk_ids(3)
    h1 = _suffix(b"\x01")
    worker_store.commit([ids[0], ids[1]], True)
    worker_store.commit([h1(ids[0]), h1(ids[1])], True)

    store = CountingLookupStore(scheduler_store)
    hits = _count_contiguous_external_hits_tp_aware(
        store=store, logical_block_ids=ids, tp_rank_hashers=[_identity, h1]
    )
    assert hits == 2
    assert len(store.calls) == 1
    assert store.calls[0] == [
        ids[0],
        h1(ids[0]),
        ids[1],
        h1(ids[1]),
        ids[2],
        h1(ids[2]),
    ]


def test_tp2_missing_any_rank_breaks(worker_store, scheduler_store):
    from vsparse.connector.sparse_connector import (
        _count_contiguous_external_hits_tp_aware,
    )

    ids = _mk_ids(2)
    h1 = _suffix(b"\x01")
    worker_store.commit([ids[0], ids[1]], True)
    worker_store.commit([h1(ids[0])], True)

    hits = _count_contiguous_external_hits_tp_aware(
        store=scheduler_store, logical_block_ids=ids, tp_rank_hashers=[_identity, h1]
    )
    assert hits == 1


def test_tp2_non_contiguous_hits_do_not_count(worker_store, scheduler_store):
    from vsparse.connector.sparse_connector import (
        _count_contiguous_external_hits_tp_aware,
    )

    ids = _mk_ids(3)
    h1 = _suffix(b"\x01")
    worker_store.commit([ids[0], ids[2]], True)
    worker_store.commit([h1(ids[0]), h1(ids[2])], True)

    hits = _count_contiguous_external_hits_tp_aware(
        store=scheduler_store, logical_block_ids=ids, tp_rank_hashers=[_identity, h1]
    )
    assert hits == 1


def test_tp2_overhead_counts_are_bounded(worker_store, scheduler_store):
    from vsparse.connector.sparse_connector import (
        _count_contiguous_external_hits_tp_aware,
    )

    n = 128
    ids = _mk_ids(n)
    h1 = CountingHasher(_suffix(b"\x01"))
    worker_store.commit(ids, True)
    worker_store.commit([h1(x) for x in ids], True)
    h1.calls = 0

    store_tp1 = CountingLookupStore(scheduler_store)
    hits_tp1 = _count_contiguous_external_hits_tp_aware(
        store=store_tp1, logical_block_ids=ids, tp_rank_hashers=[_identity]
    )
    assert hits_tp1 == n
    assert len(store_tp1.calls) == 1
    assert h1.calls == 0

    store_tp2 = CountingLookupStore(scheduler_store)
    hits_tp2 = _count_contiguous_external_hits_tp_aware(
        store=store_tp2, logical_block_ids=ids, tp_rank_hashers=[_identity, h1]
    )
    assert hits_tp2 == n
    assert len(store_tp2.calls) == 1
    assert h1.calls == n


@pytest.mark.skipif(
    os.getenv("VLLM_TP_LOOKUP_BENCH") != "1",
    reason="Set VLLM_TP_LOOKUP_BENCH=1 to run micro-benchmark.",
)
def test_tp2_micro_benchmark(worker_store, scheduler_store):
    from vsparse.connector.sparse_connector import (
        _count_contiguous_external_hits_tp_aware,
    )

    n = 512
    ids = _mk_ids(n)
    h1 = _suffix(b"\x01")
    worker_store.commit(ids, True)
    worker_store.commit([h1(x) for x in ids], True)

    t0 = time.perf_counter()
    _count_contiguous_external_hits_tp_aware(
        store=scheduler_store, logical_block_ids=ids, tp_rank_hashers=[_identity]
    )
    t1 = time.perf_counter()
    _count_contiguous_external_hits_tp_aware(
        store=scheduler_store, logical_block_ids=ids, tp_rank_hashers=[_identity, h1]
    )
    t2 = time.perf_counter()

    print(f"tp1_lookup_s={t1 - t0:.6f}, tp2_lookup_s={t2 - t1:.6f}, n={n}")
