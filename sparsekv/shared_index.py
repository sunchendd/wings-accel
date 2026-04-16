from __future__ import annotations

import os
import re
import struct
import fcntl
from multiprocessing import resource_tracker
from multiprocessing import shared_memory

_BLOCK_ID_SIZE_BYTES = 16


def _sanitize_shm_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^0-9a-zA-Z_\\-\\.]+", "_", s)
    s = s.strip("._-")
    if not s:
        s = "default"
    return s[:200]


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x - 1).bit_length())


def _unlink_shared_memory_no_resource_tracker(name: str) -> None:
    try:
        from multiprocessing.shared_memory import _posixshmem

        _posixshmem.shm_unlink(name)
        return
    except Exception:
        pass

    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    except FileNotFoundError:
        return
    finally:
        shm.close()


class SharedBlockIndex:
    _MAGIC = b"LSIX"
    _VERSION = 1
    _HEADER_STRUCT = struct.Struct("<4sIIIQ")
    _STATE_EMPTY = 0
    _STATE_OCCUPIED = 1
    _STATE_TOMBSTONE = 2

    def __init__(self, *, name: str, capacity: int, create: bool) -> None:
        self._name = name
        self._key_len = _BLOCK_ID_SIZE_BYTES
        self._capacity = int(_next_power_of_two(int(capacity)))
        self._lock_path = f"/tmp/vllm_sparse_localstore_{_sanitize_shm_name(name)}.lock"
        os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
        self._lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        self._created = bool(create)

        total_size = self._total_size(self._capacity, self._key_len)
        if create:
            shm = shared_memory.SharedMemory(name=self._name, create=True, size=total_size)
        else:
            shm = shared_memory.SharedMemory(name=self._name, create=False)
            if shm.size != total_size:
                shm.close()
                raise RuntimeError(
                    f"SharedBlockIndex size mismatch for {self._name}: "
                    f"expected {total_size}, got {shm.size}"
                )
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        self._shm = shm
        self._buf = shm.buf

        if create:
            self._write_header()
            tail = self._buf[self._keys_offset():]
            tail[:] = b"\x00" * len(tail)
        else:
            self._validate_header()

    @classmethod
    def _shm_name(cls, unique_id: str) -> str:
        return f"vllm_sparse_localstore_idx_{_sanitize_shm_name(unique_id)}"

    @classmethod
    def open_or_create(cls, unique_id: str, capacity: int) -> "SharedBlockIndex":
        shm_name = cls._shm_name(unique_id)
        try:
            return cls(name=shm_name, capacity=capacity, create=False)
        except FileNotFoundError:
            return cls(name=shm_name, capacity=capacity, create=True)
        except RuntimeError as e:
            msg = str(e)
            should_recreate = (
                    "size mismatch" in msg
                    or "invalid magic" in msg
                    or "incompatible version" in msg
                    or "header mismatch" in msg
            )
            if not should_recreate:
                raise
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
            except FileNotFoundError:
                return cls(name=shm_name, capacity=capacity, create=True)
            try:
                try:
                    resource_tracker.unregister(shm._name, "shared_memory")  # pylint: disable=protected-access
                except Exception:
                    pass
                try:
                    _unlink_shared_memory_no_resource_tracker(shm._name)  # pylint: disable=protected-access
                except FileNotFoundError:
                    pass
            finally:
                shm.close()
            return cls(name=shm_name, capacity=capacity, create=True)

    @classmethod
    def open_existing(cls, unique_id: str) -> "SharedBlockIndex | None":
        shm_name = cls._shm_name(unique_id)
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
        except FileNotFoundError:
            return None

        try:
            try:
                resource_tracker.unregister(shm._name, "shared_memory")  # pylint: disable=protected-access
            except Exception:
                pass
            buf = shm.buf
            magic, ver, cap, key_len, _count = cls._HEADER_STRUCT.unpack(
                buf[: cls._HEADER_STRUCT.size]
            )
            if magic != cls._MAGIC or int(ver) != int(cls._VERSION) or int(key_len) != _BLOCK_ID_SIZE_BYTES:
                shm.close()
                return None
            shm.close()
            return cls(name=shm_name, capacity=int(cap), create=False)
        except Exception:
            try:
                shm.close()
            except Exception:
                pass
            return None

    @staticmethod
    def _total_size(capacity: int, key_len: int) -> int:
        header = SharedBlockIndex._HEADER_STRUCT.size
        keys = int(capacity) * int(key_len)
        states = int(capacity)
        return header + keys + states

    def _write_header(self) -> None:
        self._buf[: self._HEADER_STRUCT.size] = self._HEADER_STRUCT.pack(
            self._MAGIC,
            int(self._VERSION),
            int(self._capacity),
            int(self._key_len),
            0,
        )

    def _get_counts(self) -> tuple[int, int]:
        _magic, _ver, _cap, _key_len, packed = self._HEADER_STRUCT.unpack(
            self._buf[: self._HEADER_STRUCT.size]
        )
        occ = (int(packed) >> 32) & 0xFFFFFFFF
        tomb = int(packed) & 0xFFFFFFFF
        return int(occ), int(tomb)

    def _set_counts(self, occ: int, tomb: int) -> None:
        occ32 = int(occ) & 0xFFFFFFFF
        tomb32 = int(tomb) & 0xFFFFFFFF
        packed = (occ32 << 32) | tomb32
        magic, ver, cap, key_len, _ = self._HEADER_STRUCT.unpack(
            self._buf[: self._HEADER_STRUCT.size]
        )
        self._buf[: self._HEADER_STRUCT.size] = self._HEADER_STRUCT.pack(
            magic,
            int(ver),
            int(cap),
            int(key_len),
            int(packed),
        )

    def _validate_header(self) -> None:
        magic, ver, cap, key_len, _count = self._HEADER_STRUCT.unpack(
            self._buf[: self._HEADER_STRUCT.size]
        )
        if magic != self._MAGIC:
            raise RuntimeError(f"SharedBlockIndex {self._name} has invalid magic={magic!r}")
        if int(ver) != int(self._VERSION):
            raise RuntimeError(
                f"SharedBlockIndex {self._name} has incompatible version={ver}"
            )
        if int(cap) != int(self._capacity) or int(key_len) != int(self._key_len):
            raise RuntimeError(
                f"SharedBlockIndex {self._name} header mismatch: "
                f"cap={cap} key_len={key_len} expected cap={self._capacity} key_len={self._key_len}"
            )

    def _keys_offset(self) -> int:
        return self._HEADER_STRUCT.size

    def _states_offset(self) -> int:
        return self._keys_offset() + self._capacity * self._key_len

    def _hash(self, key: bytes) -> int:
        if len(key) != self._key_len:
            key = bytes(key)[: self._key_len].ljust(self._key_len, b"\0")
        x = int.from_bytes(key[:8], "little", signed=False) ^ int.from_bytes(
            key[8:16], "little", signed=False
        )
        x ^= x >> 33
        x *= 0xFF51AFD7ED558CCD & ((1 << 64) - 1)
        x ^= x >> 33
        x *= 0xC4CEB9FE1A85EC53 & ((1 << 64) - 1)
        x ^= x >> 33
        return int(x) & (self._capacity - 1)

    def _slot_key_view(self, slot: int) -> memoryview:
        start = self._keys_offset() + int(slot) * self._key_len
        return self._buf[start: start + self._key_len]

    def _slot_state(self, slot: int) -> int:
        return int(self._buf[self._states_offset() + int(slot)])

    def _set_slot_state(self, slot: int, state: int) -> None:
        self._buf[self._states_offset() + int(slot)] = int(state)

    def lookup_many(self, block_ids: list[bytes]) -> list[bool]:
        out: list[bool] = []
        for bid in block_ids:
            out.append(self._lookup_one(bid))
        return out

    def _lookup_one(self, block_id: bytes) -> bool:
        if len(block_id) != self._key_len:
            block_id = bytes(block_id)[: self._key_len].ljust(self._key_len, b"\0")
        slot = self._hash(block_id)
        for _ in range(self._capacity):
            state = self._slot_state(slot)
            if state == self._STATE_EMPTY:
                return False
            if state == self._STATE_OCCUPIED:
                if self._slot_key_view(slot).tobytes() == block_id:
                    return True
            slot = (slot + 1) & (self._capacity - 1)
        return False

    def add_many(self, block_ids: list[bytes]) -> None:
        if not block_ids:
            return
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            occ, tomb = self._get_counts()
            if tomb > max(occ, self._capacity // 4):
                self._compact_inplace()
                occ, tomb = self._get_counts()
            for bid in block_ids:
                ok, occ, tomb = self._add_one(bid, occ, tomb)
                if not ok:
                    self._compact_inplace()
                    occ, tomb = self._get_counts()
                    ok, occ, tomb = self._add_one(bid, occ, tomb)
                    if not ok:
                        return
            self._set_counts(occ, tomb)
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _add_one(self, block_id: bytes, occ: int, tomb: int) -> tuple[bool, int, int]:
        if len(block_id) != self._key_len:
            block_id = bytes(block_id)[: self._key_len].ljust(self._key_len, b"\0")
        slot = self._hash(block_id)
        for _ in range(self._capacity):
            state = self._slot_state(slot)
            if state == self._STATE_EMPTY:
                self._slot_key_view(slot)[:] = block_id
                self._set_slot_state(slot, self._STATE_OCCUPIED)
                return True, occ + 1, tomb
            if state == self._STATE_TOMBSTONE:
                self._slot_key_view(slot)[:] = block_id
                self._set_slot_state(slot, self._STATE_OCCUPIED)
                return True, occ + 1, max(0, tomb - 1)
            if state == self._STATE_OCCUPIED and self._slot_key_view(slot).tobytes() == block_id:
                return True, occ, tomb
            slot = (slot + 1) & (self._capacity - 1)
        return False, occ, tomb

    def remove_many(self, block_ids: list[bytes]) -> None:
        if not block_ids:
            return
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            occ, tomb = self._get_counts()
            for bid in block_ids:
                removed, occ, tomb = self._remove_one(bid, occ, tomb)
            self._set_counts(occ, tomb)
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _remove_one(self, block_id: bytes, occ: int, tomb: int) -> tuple[bool, int, int]:
        if len(block_id) != self._key_len:
            block_id = bytes(block_id)[: self._key_len].ljust(self._key_len, b"\0")
        slot = self._hash(block_id)
        for _ in range(self._capacity):
            state = self._slot_state(slot)
            if state == self._STATE_EMPTY:
                return False, occ, tomb
            if state == self._STATE_OCCUPIED and self._slot_key_view(slot).tobytes() == block_id:
                self._set_slot_state(slot, self._STATE_TOMBSTONE)
                return True, max(0, occ - 1), tomb + 1
            slot = (slot + 1) & (self._capacity - 1)
        return False, occ, tomb

    def _compact_inplace(self) -> None:
        keys: list[bytes] = []
        for slot in range(self._capacity):
            if self._slot_state(slot) == self._STATE_OCCUPIED:
                keys.append(self._slot_key_view(slot).tobytes())
        states_off = self._states_offset()
        keys_off = self._keys_offset()
        self._buf[states_off: states_off + self._capacity] = b"\x00" * self._capacity
        self._buf[keys_off: keys_off + self._capacity * self._key_len] = b"\x00" * (
                self._capacity * self._key_len
        )
        occ, tomb = 0, 0
        for k in keys:
            ok, occ, tomb = self._add_one(k, occ, tomb)
            if not ok:
                break
        self._set_counts(occ, tomb)

    def close(self, *, unlink: bool = False) -> None:
        try:
            self._shm.close()
            if unlink:
                try:
                    _unlink_shared_memory_no_resource_tracker(self._shm._name)  # pylint: disable=protected-access
                except FileNotFoundError:
                    pass
        finally:
            try:
                os.close(self._lock_fd)
            except Exception:
                pass
