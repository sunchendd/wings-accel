"""
LocalStore KVStore implementation used by BMSA KV offload + prefetch pipelines.

LocalStore is a minimal, process-local block store that supports:

- Alloc / Lookup by block_id (string)
- Load / Dump by (block_id, offset, length, address)

Key implementation characteristics:

- `io_size` defines the *virtual address space size* of each block in bytes.
  All reads/writes
  are (offset, length) slices within this per-block space. Any overflow
  (offset+length > io_size) is a hard error.
- `capacity` defines how many blocks can be allocated concurrently.
- `device` is used by the underlying backend to determine which device
  (e.g., CUDA device id)
  the store should operate on.
- When used with the BMSA prefetch C++ extension, `cc_store()` exposes a `void*` pointer
  compatible with the store ABI expected by that extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import threading
import time
from typing import Any

import os

import numpy as np
import torch

from vllm.logger import init_logger
from vsparse.kvstore.kvstore import KVStoreBase, Task
from vsparse.shared_index import SharedBlockIndex as _SharedBlockIndex


logger = init_logger(__name__)


@dataclass
class LocalStoreTask(Task):
    """A thin Task wrapper holding the integer task id returned by the C++ store."""

    task_id: int


def _infer_localstore_io_size(config: dict[str, Any]) -> int:
    """
    Infer the per-block io_size (bytes) for LocalStore.

    LocalStore enforces that (offset + length) does not exceed io_size.
    In sparse KV transfer, the connector typically computes a `kv_block_size`
    large enough to cover all layers and K/V shards, and LocalStore uses
    max(io_size, kv_block_size) to be safe.
    """

    io_size = int(config.get("io_size") or 0)
    kv_block_size = int(config.get("kv_block_size") or 0)
    return max(io_size, kv_block_size)


def _infer_localstore_capacity(config: dict[str, Any], io_size: int) -> int:
    """
    Infer how many blocks LocalStore can hold concurrently.

    Supported knobs:
    - capacity: explicit number of blocks
    - max_cache_size: total bytes; capacity computed as floor(max_cache_size / io_size)
    """

    capacity = config.get("capacity")
    if capacity is not None:
        return max(int(capacity), 1)

    max_cache_size = config.get("max_cache_size")
    if max_cache_size is not None and io_size > 0:
        return max(int(max_cache_size) // int(io_size), 1)

    return 1


class LocalStoreKVStore(KVStoreBase):
    """
    KVStoreBase adapter around the pybind `local_kvstore.LocalStore`.

    This adapter converts vLLM-level tensor lists to (data_ptr, size_bytes)
    arrays expected by the C++ backend, and provides
    `create/lookup/load/dump/wait/commit/check` in a connector-friendly style.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        role = str(config.get("role") or "")
        self._role = role
        self._shared_index: _SharedBlockIndex | None = None
        self._lock = threading.RLock()
        self._store_api_lock = threading.Lock()
        self._async_dump_enabled = bool(config.get("async_dump_enabled", True))
        self._async_dump_poll_interval_s = float(
            config.get("async_dump_poll_interval_s", 0.002)
        )
        self._async_dump_budget_s = float(config.get("async_dump_budget_s", 0.01))
        self._async_dump_stop_event = threading.Event()
        self._async_dump_thread: threading.Thread | None = None
        self._pending_dump_tasks: dict[int, tuple[str, list[bytes | str], int]] = {}
        self._capacity_blocks: int | None = None
        self._block_state: dict[bytes | str, str] = {}
        self._block_refcnt: dict[bytes | str, int] = {}
        self._req_to_blocks: dict[str, dict[bytes | str, None]] = {}
        self._lru: "OrderedDict[bytes | str, None]" = OrderedDict()
        if bool(config.get("shared_index_enable", True)):
            unique_id = str(config.get("unique_id") or "default")
            index_capacity = int(config.get("index_capacity") or 0)
            if index_capacity <= 0:
                cap_hint = int(config.get("capacity") or 0)
                tp_size = int(config.get("tp_size") or 1)
                if cap_hint > 0:
                    index_capacity = max(1024, cap_hint * max(tp_size, 1) * 8)
                else:
                    index_capacity = 131072
            self._shared_index = _SharedBlockIndex.open_or_create(unique_id, index_capacity)

        if role == "scheduler":
            self._store = None
            self._tensor_size = int(config.get("tensor_size") or 0)
            self._is_mla = bool(config.get("is_mla") or False)
            return

        from vsparse.native import _local_kvstore

        self._store = _local_kvstore.LocalStore()
        io_size = _infer_localstore_io_size(config)
        if io_size <= 0:
            raise ValueError(
                "LocalStoreKVStore requires a positive io_size or kv_block_size."
            )
        capacity = _infer_localstore_capacity(config, io_size)
        self._capacity_blocks = int(capacity)
        device_id = int(config.get("device", -1))

        store_config = _local_kvstore.LocalStore.Config(io_size, capacity)
        store_config.deviceId = device_id
        if "backend" in config:
            # Optional: use an existing CCStore<> backend instead of allocating memory
            # here.
            # This only works within the same process.
            store_config.backend = config["backend"]

        ret = int(self._store.Setup(store_config))
        if ret != 0:
            raise RuntimeError(
                f"LocalStore.Setup failed with ret={ret}, io_size={io_size}, "
                f"capacity={capacity}, device_id={device_id}"
            )
        self._tensor_size = int(config.get("tensor_size") or 0)
        self._is_mla = bool(config.get("is_mla") or False)
        if self._async_dump_enabled:
            self._start_async_dump_thread()

    def __del__(self) -> None:
        shared_index = getattr(self, "_shared_index", None)
        try:
            self._async_dump_stop_event.set()
            t = getattr(self, "_async_dump_thread", None)
            if t is not None and t.is_alive():
                t.join(timeout=0.2)
        except Exception:
            pass
        if shared_index is None:
            return
        try:
            unlink_on_del = self.config.get("shared_index_unlink_on_del", None)
            if unlink_on_del is None:
                env = os.getenv("VLLM_SPARSE_LOCALSTORE_UNLINK_ON_DEL")
                unlink_on_del = env == "1" if env is not None else False
            unlink_on_del = bool(unlink_on_del) and str(getattr(self, "_role", "")) == "scheduler"
            shared_index.close(unlink=unlink_on_del)
        except Exception:
            return

    def close_shared_index(self, *, unlink: bool = False) -> None:
        shared_index = getattr(self, "_shared_index", None)
        if shared_index is None:
            return
        self._shared_index = None
        shared_index.close(unlink=bool(unlink))

    def cc_store(self):
        """
        Return the underlying C++ store pointer.

        This is primarily used by the BMSA prefetch C++ extension, which expects a
        store-ABI-compatible pointer.
        """

        if self._store is None:
            return 0
        return self._store.CCStoreImpl()

    @staticmethod
    def _normalize_block_id(block_id: bytes | str) -> bytes | str:
        """
        Normalize a block id for internal bookkeeping.

        Important constraints:
        - The underlying C++ LocalStore treats the block id as an opaque string key.
          We must NOT change its value (e.g. padding/truncation), otherwise commits
          or lookups could mismatch the originally allocated key.
        - BMSA uses 16-byte md5 digests (bytes) as block ids; those are passed
          through unchanged.
        - Some unit tests use small string keys (e.g. "b2"); those should also
          be passed through unchanged.
        """
        if isinstance(block_id, (bytes, str)):
            return block_id
        raise TypeError(f"Unsupported block_id type: {type(block_id)}")

    @classmethod
    def _normalize_block_ids(cls, block_ids: list[bytes | str]) -> list[bytes | str]:
        if not block_ids:
            return []
        out: list[bytes | str] = []
        seen: set[bytes | str] = set()
        for bid in block_ids:
            nbid = cls._normalize_block_id(bid)
            if nbid in seen:
                continue
            seen.add(nbid)
            out.append(nbid)
        return out

    def _lru_touch_mru(self, block_id: bytes | str) -> None:
        if block_id in self._lru:
            self._lru.move_to_end(block_id, last=True)
        else:
            self._lru[block_id] = None

    def _lru_remove(self, block_id: bytes | str) -> None:
        self._lru.pop(block_id, None)

    def _ensure_space_for_new_blocks(self, num_new_blocks: int) -> None:
        """
        Ensure the LocalStore has enough free capacity for allocating `num_new_blocks`
        *new* blocks.

        This method implements an external-storage analogue of vLLM's BlockPool:

        - Blocks currently referenced by any running request are "pinned" and are
          not evictable.
        - Ready-but-unpinned blocks are tracked in an LRU list and can be evicted
          when capacity pressure happens.
        - Eviction is implemented by committing `success=False`, which:
            1) frees the underlying C++ buffer and removes the block entry
            2) removes the block id from the shared index (scheduler lookup view)

        If the store is full and there are no evictable blocks, this method raises
        RuntimeError to force the caller to fall back (e.g., disable offload for
        the affected blocks).
        """
        if self._store is None:
            return
        if num_new_blocks <= 0:
            return
        if self._capacity_blocks is None:
            raise RuntimeError("LocalStoreKVStore capacity is not initialized.")

        need_evict = len(self._block_state) + int(num_new_blocks) - int(self._capacity_blocks)
        if need_evict <= 0:
            return

        victims: list[bytes] = []
        while need_evict > 0:
            if not self._lru:
                raise RuntimeError(
                    "LocalStoreKVStore is full and no evictable blocks are available. "
                    "All blocks appear to be pinned by active requests."
                )
            victim, _ = self._lru.popitem(last=False)
            victims.append(victim)
            need_evict -= 1

        self.commit(victims, is_success=False)

    def pin_blocks(self, request_id: str, block_ids: list[bytes | str]) -> None:
        """
        Pin blocks for a request.

        Pinned blocks are protected from eviction. This matches KVCacheManager's
        semantics where blocks in use by running requests must not be evicted.
        """
        if self._store is None:
            return
        req_id = str(request_id)
        norm = self._normalize_block_ids(block_ids)
        if not norm:
            return
        with self._lock:
            pinned = self._req_to_blocks.setdefault(req_id, {})
            for bid in norm:
                if bid in pinned:
                    continue
                pinned[bid] = None
                prev = int(self._block_refcnt.get(bid, 0))
                self._block_refcnt[bid] = prev + 1
                if prev == 0:
                    self._lru_remove(bid)

    def unpin_blocks(self, request_id: str, block_ids: list[bytes | str]) -> None:
        """
        Unpin a subset of blocks for a request.

        When a block's refcount drops to 0 and it is "ready", it becomes an
        eviction candidate and is appended to the LRU as MRU.
        """
        if self._store is None:
            return
        req_id = str(request_id)
        norm = self._normalize_block_ids(block_ids)
        if not norm:
            return
        with self._lock:
            pinned = self._req_to_blocks.get(req_id)
            if not pinned:
                return
            for bid in norm:
                if bid not in pinned:
                    continue
                pinned.pop(bid, None)
                prev = int(self._block_refcnt.get(bid, 0))
                new = prev - 1
                if new < 0:
                    raise RuntimeError(f"LocalStoreKVStore refcnt underflow for block {bid!r}")
                if new == 0:
                    self._block_refcnt.pop(bid, None)
                    if self._block_state.get(bid) == "ready":
                        self._lru_touch_mru(bid)
                else:
                    self._block_refcnt[bid] = new
            if not pinned:
                self._req_to_blocks.pop(req_id, None)

    def request_finished(self, request_id: str) -> None:
        """
        Release all pins held by a request.

        This should be called when the inference request finishes (normal finish
        or abort). It mirrors KVCacheManager.free(): blocks are no longer pinned
        by the request, but may remain in the store as prefix-cache candidates
        (LRU-managed) until evicted.
        """
        if self._store is None:
            return
        req_id = str(request_id)
        with self._lock:
            pinned = self._req_to_blocks.get(req_id)
            if not pinned:
                return
            to_unpin = list(pinned.keys())
        self.unpin_blocks(req_id, to_unpin)

    def before_dump(self, request_id: str, block_ids: list[bytes | str]) -> None:
        """
        Called by the connector before dumping new KV blocks into the store.

        Responsibilities:
        - Ensure capacity for newly allocated blocks via LRU eviction.
        - Reserve capacity for blocks that are about to be dumped to avoid
          over-subscription within the same step.
        - Pin these blocks for the owning request to prevent eviction while the
          request is still running.
        """
        if self._store is None:
            return
        req_id = str(request_id)
        norm = self._normalize_block_ids(block_ids)
        if not norm:
            return
        with self._lock:
            new_blocks = [bid for bid in norm if bid not in self._block_state]
            self._ensure_space_for_new_blocks(len(new_blocks))
            for bid in new_blocks:
                self._block_state[bid] = "reserved"
        self.pin_blocks(req_id, norm)

    def after_dump_success(self, request_id: str, block_ids: list[bytes | str]) -> None:
        """
        Called after a successful dump + commit(success=True).

        At this point, blocks are ready for lookup/load. We keep them pinned
        since they belong to a running request; they will be released at
        request_finished().
        """
        if self._store is None:
            return None
        return None

    def after_dump_fail(self, request_id: str, block_ids: list[bytes | str]) -> None:
        """
        Called when dumping KV blocks fails (either submission failure or async
        execution failure).

        This releases any reservations and pins created by before_dump().
        The caller is expected to follow up with commit(success=False) to
        reclaim underlying store buffers if they were allocated.
        """
        if self._store is None:
            return
        req_id = str(request_id)
        norm = self._normalize_block_ids(block_ids)
        if not norm:
            return
        self.unpin_blocks(req_id, norm)
        with self._lock:
            for bid in norm:
                if self._block_state.get(bid) == "reserved":
                    self._block_state.pop(bid, None)

    def create(self, block_ids: list[bytes]) -> list[int]:
        if self._store is None:
            return [-1 for _ in block_ids]
        return [int(x) for x in self._store.AllocBatch(block_ids)]

    def lookup(self, block_ids: list[bytes]) -> list[bool]:
        if self._store is None:
            if self._shared_index is None:
                return [False for _ in block_ids]
            return self._shared_index.lookup_many(block_ids)
        return [bool(x) for x in self._store.LookupBatch(block_ids)]

    def prefetch(self, block_ids: list[bytes]) -> None:
        return None

    @staticmethod
    def _to_addr_and_size(tensors: list[torch.Tensor]) -> tuple[list[int], list[int]]:
        """
        Convert tensors to raw device addresses and sizes in bytes.

        For CUDA backend, the address must be a device pointer (CUDA tensor).
        Passing CPU tensors will typically fail in the device memcpy path.
        """

        addrs: list[int] = []
        sizes: list[int] = []
        for t in tensors:
            addrs.append(int(t.data_ptr()))
            sizes.append(int(t.numel() * t.element_size()))
        return addrs, sizes

    def load(
            self, block_ids: list[bytes], offset: list[int], dst_tensor: list[torch.Tensor]
    ) -> Task:
        addrs, sizes = self._to_addr_and_size(dst_tensor)
        task_id = int(self._store.Load(block_ids, offset, addrs, sizes))
        return LocalStoreTask(task_id=task_id)

    def dump(
            self, block_ids: list[bytes], offset: list[int], src_tensor: list[torch.Tensor]
    ) -> Task:
        addrs, sizes = self._to_addr_and_size(src_tensor)
        task_id = int(self._store.Dump(block_ids, offset, addrs, sizes))
        return LocalStoreTask(task_id=task_id)

    def _load_segments(
            self,
            block_ids: list[bytes],
            offset: list[int],
            dst_addr: list[int],
            size: list[int],
    ) -> Task:
        task_id = int(self._store.Load(block_ids, offset, dst_addr, size))
        return LocalStoreTask(task_id=task_id)

    def _dump_segments(
            self,
            block_ids: list[bytes],
            offset: list[int],
            src_addr: list[int],
            size: list[int],
    ) -> Task:
        task_id = int(self._store.Dump(block_ids, offset, src_addr, size))
        return LocalStoreTask(task_id=task_id)

    def _matrix_to_segments(
            self, block_ids: list[bytes], addrs: list[list[int]] | np.ndarray
    ) -> tuple[list[bytes], list[int], list[int], list[int]]:
        if self._tensor_size <= 0:
            raise ValueError("LocalStoreKVStore requires config['tensor_size'] > 0.")
        addrs_np = np.asarray(addrs, dtype=np.uint64)
        if addrs_np.ndim != 2:
            raise ValueError("Expected 2D addrs matrix.")
        if addrs_np.shape[0] != len(block_ids):
            raise ValueError(
                f"Row count mismatch: addrs.shape[0]={addrs_np.shape[0]} "
                f"but len(block_ids)={len(block_ids)}"
            )

        num_blocks, num_cols = addrs_np.shape
        if self._is_mla:
            num_layers = num_cols
            has_v = False
        else:
            if num_cols % 2 != 0:
                raise ValueError(
                    "Non-MLA LocalStore expects K/V address matrix with even columns."
                )
            num_layers = num_cols // 2
            has_v = True

        if has_v:
            cols = np.arange(num_cols, dtype=np.int64)
            layer_id = cols % int(num_layers)
            is_v = (cols >= int(num_layers)).astype(np.int64)
            col_offsets = (layer_id * 2 + is_v) * int(self._tensor_size)
        else:
            col_offsets = np.arange(num_cols, dtype=np.int64) * int(self._tensor_size)

        seg_block_ids = [
            bid
            for bid in block_ids
            for _ in range(num_cols)
        ]
        seg_offsets = np.tile(col_offsets, int(num_blocks)).astype(np.int64).tolist()
        seg_addrs = addrs_np.reshape(-1).astype(np.uint64).tolist()
        seg_sizes = [int(self._tensor_size)] * (int(num_blocks) * int(num_cols))
        return seg_block_ids, seg_offsets, seg_addrs, seg_sizes

    def load_data(
            self,
            block_ids: list[bytes],
            shard_index: list[int],
            dst_addr: list[list[int]] | np.ndarray,
    ) -> Task:
        seg_block_ids, seg_offsets, seg_addrs, seg_sizes = self._matrix_to_segments(
            block_ids, dst_addr
        )
        return self._load_segments(seg_block_ids, seg_offsets, seg_addrs, seg_sizes)

    def dump_data(
            self,
            block_ids: list[bytes],
            shard_index: list[int],
            src_addr: list[list[int]] | np.ndarray,
    ) -> Task:
        seg_block_ids, seg_offsets, seg_addrs, seg_sizes = self._matrix_to_segments(
            block_ids, src_addr
        )
        return self._dump_segments(seg_block_ids, seg_offsets, seg_addrs, seg_sizes)

    def dump_data_async(
            self,
            request_id: str,
            block_ids: list[bytes],
            shard_index: list[int],
            src_addr: list[list[int]] | np.ndarray,
    ) -> Task:
        if self._store is None:
            return LocalStoreTask(task_id=-1)
        rid = str(request_id)
        norm = self._normalize_block_ids(block_ids)
        self.before_dump(rid, norm)
        try:
            with self._store_api_lock:
                task = self.dump_data(list(norm), shard_index, src_addr)
        except Exception:
            self.after_dump_fail(rid, norm)
            self.commit(norm, is_success=False)
            raise
        task_id = int(getattr(task, "task_id", -1))
        if task_id >= 0:
            with self._lock:
                self._pending_dump_tasks[task_id] = (rid, list(norm), len(norm))
        return task

    def dump_data_sync(
            self,
            request_id: str,
            block_ids: list[bytes],
            shard_index: list[int],
            src_addr: list[list[int]] | np.ndarray,
    ) -> Task:
        """
        Dump KV data and do not return until the blocks are committed.

        vLLM's worker-side `wait_for_save()` contract is synchronous: by the time
        the forward step exits, saved blocks must be visible to both:
        - the worker-side LocalStore (prefetch/offload path), and
        - the scheduler-side shared index (prefix/shared-block lookup path).

        BMSA's first decode step depends on that visibility. If we only enqueue an
        async dump here, the scheduler may still see a dense prompt layout for a few
        decode steps and the sparse block-table / slot-mapping state diverges.
        """
        if self._store is None:
            return LocalStoreTask(task_id=-1)
        rid = str(request_id)
        norm = self._normalize_block_ids(block_ids)
        self.before_dump(rid, norm)
        try:
            with self._store_api_lock:
                task = self.dump_data(list(norm), shard_index, src_addr)
            ret = self.wait(task)
        except Exception:
            self.after_dump_fail(rid, norm)
            self.commit(norm, is_success=False)
            raise

        if int(ret) == 0:
            self.commit(norm, is_success=True)
            self.after_dump_success(rid, norm)
            return task

        self.after_dump_fail(rid, norm)
        self.commit(norm, is_success=False)
        raise RuntimeError(
            f"LocalStoreKVStore synchronous dump failed for request {rid!r} "
            f"with ret={int(ret)}"
        )

    def wait(self, task: LocalStoreTask) -> int:
        with self._store_api_lock:
            return int(self._store.Wait(int(task.task_id)))

    def commit(self, block_ids: list[bytes | str], is_success: bool = True) -> None:
        norm = self._normalize_block_ids(block_ids)
        if self._store is not None:
            self._store.CommitBatch(norm, bool(is_success))
        if self._shared_index is not None:
            bytes_ids: list[bytes] = []
            for bid in norm:
                if not isinstance(bid, (bytes, bytearray, memoryview)):
                    bytes_ids = []
                    break
                bytes_ids.append(bytes(bid))
            if bytes_ids:
                if bool(is_success):
                    self._shared_index.add_many(bytes_ids)
                else:
                    self._shared_index.remove_many(bytes_ids)

        if self._store is None:
            return

        with self._lock:
            if bool(is_success):
                for bid in norm:
                    self._block_state[bid] = "ready"
                    if int(self._block_refcnt.get(bid, 0)) == 0:
                        self._lru_touch_mru(bid)
            else:
                for bid in norm:
                    self._block_state.pop(bid, None)
                    self._lru_remove(bid)
                    self._block_refcnt.pop(bid, None)
                    for req_id, pinned in list(self._req_to_blocks.items()):
                        if bid in pinned:
                            pinned.pop(bid, None)
                            if not pinned:
                                self._req_to_blocks.pop(req_id, None)

    def check(self, task: LocalStoreTask) -> tuple[int, bool]:
        with self._store_api_lock:
            ret, finished = self._store.CheckPy(int(task.task_id))
            return int(ret), bool(finished)

    def _start_async_dump_thread(self) -> None:
        if self._async_dump_thread is not None and self._async_dump_thread.is_alive():
            return
        t = threading.Thread(target=self._async_dump_loop, daemon=True)
        self._async_dump_thread = t
        t.start()

    def _async_dump_loop(self) -> None:
        while not self._async_dump_stop_event.is_set():
            try:
                self._drain_pending_dumps(self._async_dump_budget_s)
            except Exception:
                pass
            self._async_dump_stop_event.wait(self._async_dump_poll_interval_s)

    def _drain_pending_dumps(self, budget_s: float) -> None:
        if self._store is None:
            return
        with self._lock:
            task_ids = list(self._pending_dump_tasks.keys())
        if not task_ids:
            return
        start = None if budget_s <= 0 else time.monotonic()
        for task_id in task_ids:
            if start is not None and (time.monotonic() - start) >= budget_s:
                break
            with self._lock:
                meta = self._pending_dump_tasks.get(int(task_id))
            if meta is None:
                continue
            rid, block_ids, _n = meta
            ret, finished = self.check(LocalStoreTask(task_id=int(task_id)))
            if not finished:
                continue

            try:
                ret = self.wait(LocalStoreTask(task_id=int(task_id)))
            except Exception:
                ret = 1

            with self._lock:
                self._pending_dump_tasks.pop(int(task_id), None)

            if int(ret) == 0:
                self.commit(block_ids, is_success=True)
                self.after_dump_success(rid, block_ids)
            else:
                self.after_dump_fail(rid, block_ids)
                self.commit(block_ids, is_success=False)
