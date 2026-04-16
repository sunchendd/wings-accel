# paged_kmeans/utils.py
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


@contextmanager
def timer(enabled: bool, name: str):
    """
    一个极简的 wall-clock 计时器，用于本地调试与快速对比。

    注意：
    - 这是 Python 侧计时，默认不会做 `torch.cuda.synchronize()`。
      因此在 CUDA 上测 kernel 性能时，建议在进入/退出计时区间前后手动同步，
      或在测试用例中使用 `torch.cuda.Event` 来统计更可信的 kernel 时间。
    - 这里保留 print 是为了在 prototype 阶段快速观察，不做 logging 体系集成。
    """
    if not enabled:
        yield
        return
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[TIMER] {name}: {(t1 - t0) * 1000:.3f} ms")


def normalize_(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对 `x` 的最后一维做 in-place 归一化（L2 norm）。

    - 输入输出同一个张量（原地修改）
    - 归一化在 float32 中计算 norm，再 cast 回原 dtype，尽量减少数值误差
    - 主要用于对齐参考实现中“部分迭代会 normalize centroid”的语义
    """
    denom = torch.linalg.norm(x.float(), dim=-1, keepdim=True).clamp_min(eps)
    x.div_(denom.to(dtype=x.dtype))
    return x


_REQUEST_SLOT_BITS = 32
_REQUEST_SLOT_MASK = (1 << _REQUEST_SLOT_BITS) - 1


def pack_request_handle(*, slot: int, generation: int) -> int:
    if slot < 0:
        raise ValueError("slot must be >= 0")
    if generation < 0:
        raise ValueError("generation must be >= 0")
    if slot > _REQUEST_SLOT_MASK:
        raise ValueError("slot overflow for packed handle")
    if generation > _REQUEST_SLOT_MASK:
        raise ValueError("generation overflow for packed handle")
    return (int(generation) << _REQUEST_SLOT_BITS) | int(slot)


def unpack_request_slot(handle: int) -> int:
    return int(int(handle) & _REQUEST_SLOT_MASK)


def unpack_request_generation(handle: int) -> int:
    return int(int(handle) >> _REQUEST_SLOT_BITS)


def request_slots_from_handles(handles: torch.Tensor) -> torch.Tensor:
    if handles.dtype not in (torch.int64, torch.int32):
        handles = handles.to(torch.int64)
    if handles.ndim != 1:
        raise ValueError("handles must be shape [B]")
    return (handles.to(torch.int64) & _REQUEST_SLOT_MASK).to(torch.int64)


class RequestHandleAllocator:
    def __init__(self, *, max_requests: int) -> None:
        self.max_requests = int(max_requests)
        if self.max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        self._free_slots: list[int] = list(range(self.max_requests - 1, -1, -1))
        self._generation: list[int] = [0 for _ in range(self.max_requests)]
        self._in_use: list[bool] = [False for _ in range(self.max_requests)]

    def allocate(self) -> int:
        if not self._free_slots:
            raise RuntimeError("RequestHandleAllocator: no free slots")
        slot = int(self._free_slots.pop())
        self._in_use[slot] = True
        gen = int(self._generation[slot])
        return pack_request_handle(slot=slot, generation=gen)

    def is_alive(self, handle: int) -> bool:
        slot = unpack_request_slot(handle)
        if slot < 0 or slot >= self.max_requests:
            return False
        if not self._in_use[slot]:
            return False
        return unpack_request_generation(handle) == int(self._generation[slot])

    def free(self, handle: int) -> None:
        """
        释放一个 request_handle（并推进对应 slot 的 generation）。

        释放顺序纪律（非常重要）：
        - 正确顺序必须是：先对各组件执行 `remove_handle(handle)` 清理状态，再调用 `free(handle)` 归还 handle。
          例如在 `PagedKMeansClusterManager.free_handle()` 里就是这个顺序。
        - 不要先 free 再 remove：
          因为 `remove_handle()` 通常会通过 `is_alive(handle)` 来判断是否需要清理；
          若 handle 已被 free，则 is_alive=False，remove_handle 可能直接 no-op，
          导致上一代 slot 的残留状态没有被清掉，后续 slot 复用时出现跨请求数据污染。
        """
        slot = unpack_request_slot(handle)
        if slot < 0 or slot >= self.max_requests:
            raise ValueError("invalid handle slot")
        if not self.is_alive(handle):
            raise RuntimeError("free called on non-alive handle")
        self._in_use[slot] = False
        self._generation[slot] = int(self._generation[slot]) + 1
        self._free_slots.append(int(slot))
