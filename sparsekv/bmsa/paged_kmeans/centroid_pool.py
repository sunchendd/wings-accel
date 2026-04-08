from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from vsparse.native import _paged_kmeans
from .utils import (
    RequestHandleAllocator,
    request_slots_from_handles,
    unpack_request_generation,
    unpack_request_slot,
)

KSpec = int | torch.Tensor
ReturnFormat = Literal["packed", "dict"]


@dataclass(frozen=True)
class CentroidWriteInfo:
    """
    一次 `append_centroids()` 写入操作的元信息。

    这些字段用于把“centroid 池的 dense 存储布局”映射回“逻辑 cluster id 空间”：
    - `request_slot`：该 request 在池中的槽位（0..max_requests-1）
    - `kv_head`：kv head 维度（0..num_kv_heads-1）
    - `(request_slot, kv_head)` 共同决定了 dense row：
      row = request_slot * num_kv_heads + kv_head
    - `start/length`：本次追加写入在 dense row 内的区间 [start, start+length)
    - `base_cluster_id`：本次追加段在“逻辑 cluster id 空间”里的起始值

    设计动机：
    - 增量聚类会不断追加新的簇，如果只用 local index（0..K-1），会和历史簇冲突
    - `base_cluster_id + local_cluster_index` 形成单调递增的逻辑 cluster id
    """

    request_handle: int
    request_slot: int
    layer: int
    kv_head: int
    base_cluster_id: int
    start: int
    length: int


@dataclass(frozen=True)
class TopKPackedResult:
    """
    Packed topk output to avoid Python dict overhead.

    这是 `batch_topk(..., return_format="packed")` 的返回结构。

    目标：
    - 避免 Python dict/loop 带来的开销（decode 阶段每步都会调用 TopK）
    - 保留足够信息，让调用方可以在 Python 侧进行轻量的后处理（比如按 request 做裁剪）

    字段语义：
    - `logical_cluster_ids`: shape [B, Hkv, max_k], dtype int64
      - 每个 batch request、每个 kv_head 输出一个长度为 max_k 的 cluster id 列表
      - 真实有效长度为 `min(k_per_req[b], valid_k[b, h])`
      - 如果 request 未注册（例如 request 已结束），对应的 request_slot 为 -1，
        调用方应将其视为无效输出
    - `k_per_req`: shape [B], dtype int64
      - 每个 request 的 TopK 需求（允许是 int 或 [B] 张量输入）
    - `valid_k`: shape [B, Hkv], dtype int32
      - 每个 request、每个 kv_head 当前实际可用的 centroid 数（增量聚类会不断增长）
    - `request_slots`: shape [B], dtype int64
      - request_id 在 centroid pool 中对应的槽位；-1 表示该 request_id 未注册
      - 这是 CentroidPool 的“并发槽位”（0..max_requests-1），用于定位 dense layout 的行：
        row = request_slot * num_kv_heads + kv_head
      - 它不是 vLLM 的 `index_in_batch`，也不是 vLLM 的 `slot_mapping`；
        它只在自己的链路内有意义。
    """

    logical_cluster_ids: torch.Tensor
    k_per_req: torch.Tensor
    valid_k: torch.Tensor
    request_slots: torch.Tensor


@dataclass
class _CutlassWorkspace:
    cap_bg: int
    D_out: torch.Tensor
    softmax_o: torch.Tensor
    norm: torch.Tensor
    summ: torch.Tensor


class CentroidPool:
    """
    以 dense layout 存储每个 request 的 centroids，并提供与参考实现一致的 TopK 接口。

    核心数据结构（per layer）：
    - `centroids[layer]`: shape [R*Hkv, Kmax, D]
      - R = max_requests
      - Hkv = num_kv_heads
      - dense row = request_slot * Hkv + kv_head
      - Kmax = max_num_centroids（单请求允许累计的最大簇数，包含增量追加）
    - `cluster_size[layer]`: shape [R*Hkv, Kmax], int32
      - cluster_size==0 表示空簇/无效簇，需要在 TopK 前 mask 掉
    - `valid_k[layer]`: shape [R*Hkv], int32
      - 表示每行当前写入的 centroid 数（<=Kmax）
    - `next_cluster_id[layer]`: shape [R*Hkv], int64
      - 逻辑 cluster id 空间的“下一次可分配 id”（单调递增）
      - `append_centroids()` 会返回 base_cluster_id=old_next_cluster_id，
        并更新 next_cluster_id += K

    生命周期管理（packed request_handle）：
    - request_handle 是一个 packed int64：低 32bit 为 slot，高 32bit 为 generation
    - slot 用于定位 dense layout 的 row；generation 用于在异步场景下避免 slot 复用导致的 ABA
    - `append_centroids(request_handle, ...)` 会在首次看到该 (slot,generation) 时重置该 slot 的元数据
    - 请求结束时，上层必须调用 `remove_handle(request_handle)` 清理该 slot 的元数据

    TopK 语义（对齐 `reference_retroinfer`）：
    - 输入 query shape: [B, num_heads, D]，其中 num_heads = num_kv_heads * group_size
    - 对每个 kv_head，把其对应的 group queries 取出来：shape [B, group_size, D]
    - 计算 logits = (Q @ C^T) * (1/sqrt(D))，对 Kmax 维做 softmax
    - 对 group_size 维求和得到 dist: [B, Kmax]，再对 dist 做 TopK (largest=True)
    - 对无效簇（cluster_size==0 或 index>=valid_k）做 mask，确保不会被 TopK 选中

    request_handle controls lifetime.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        num_heads: int,
        head_dim: int,
        max_requests: int,
        handle_allocator: RequestHandleAllocator,
        max_num_centroids: int,
        dtype: torch.dtype,
        device: torch.device,
        zero_centroids_on_register: bool = True,
    ) -> None:
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads({num_heads}) must be divisible by "
                f"num_kv_heads({num_kv_heads})"
            )

        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.num_heads = int(num_heads)
        self.group_size = int(num_heads // num_kv_heads)
        self.head_dim = int(head_dim)

        self.max_requests = int(max_requests)
        self.max_num_centroids = int(max_num_centroids)
        self._handles = handle_allocator
        if int(self._handles.max_requests) != int(self.max_requests):
            raise ValueError("handle_allocator.max_requests must match max_requests")

        if device.type == "cuda" and device.index is None and torch.cuda.is_available():
            device = torch.device("cuda", index=torch.cuda.current_device())
        self.device = device
        self.dtype = dtype
        self.zero_centroids_on_register = bool(zero_centroids_on_register)

        RH = self.max_requests * self.num_kv_heads

        self.centroids = [
            torch.empty(
                (RH, self.max_num_centroids, self.head_dim), device=device, dtype=dtype
            )
            for _ in range(self.num_layers)
        ]
        self.cluster_size = [
            torch.zeros((RH, self.max_num_centroids), device=device, dtype=torch.int32)
            for _ in range(self.num_layers)
        ]
        self.valid_k = [
            torch.zeros((RH,), device=device, dtype=torch.int32)
            for _ in range(self.num_layers)
        ]
        self.next_cluster_id = [
            torch.zeros((RH,), device=device, dtype=torch.int64)
            for _ in range(self.num_layers)
        ]
        self._active_generation_cpu: list[int] = [-1 for _ in range(self.max_requests)]
        self._active_generation = torch.full(
            (self.max_requests,),
            -1,
            device=self.device,
            dtype=torch.int32,
        )

        self._rsqrt_dim = 1.0 / math.sqrt(self.head_dim)
        self._dist_dtype_min = torch.finfo(torch.float32).min
        self._k_idx_int32 = torch.arange(
            self.max_num_centroids, device=self.device, dtype=torch.int32
        )
        self._cutlass_ws: dict[torch.dtype, _CutlassWorkspace] = {}

    def _get_cutlass_ws(
        self, *, BG: int, G: int, Kmax: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ws = self._cutlass_ws.get(dtype)
        blk = (Kmax + 256 - 1) // 256
        if ws is None or int(ws.cap_bg) < int(BG):
            cap_bg = int(BG)
            D_out = torch.empty((cap_bg, G, Kmax), device=self.device, dtype=dtype)
            softmax_o = torch.empty((cap_bg, G, Kmax), device=self.device, dtype=dtype)
            norm = torch.empty(
                (cap_bg, G, blk), device=self.device, dtype=torch.float32
            )
            summ = torch.empty(
                (cap_bg, G, blk), device=self.device, dtype=torch.float32
            )
            ws = _CutlassWorkspace(
                cap_bg=cap_bg, D_out=D_out, softmax_o=softmax_o, norm=norm, summ=summ
            )
            self._cutlass_ws[dtype] = ws
        return (
            ws.D_out[:BG],
            ws.softmax_o[:BG],
            ws.norm[:BG],
            ws.summ[:BG],
        )

    def _reset_slot(self, slot: int) -> None:
        h0 = slot * self.num_kv_heads
        h1 = h0 + self.num_kv_heads
        for layer_idx in range(self.num_layers):
            self.valid_k[layer_idx][h0:h1].zero_()
            self.next_cluster_id[layer_idx][h0:h1].zero_()
            self.cluster_size[layer_idx][h0:h1].zero_()
            if self.zero_centroids_on_register:
                self.centroids[layer_idx][h0:h1].zero_()

    def _ensure_active_handle(self, request_handle: int) -> int:
        if not self._handles.is_alive(request_handle):
            raise KeyError("request_handle not alive in centroid pool")
        slot = unpack_request_slot(request_handle)
        gen = unpack_request_generation(request_handle)
        if self._active_generation_cpu[slot] != gen:
            self._reset_slot(int(slot))
            self._active_generation_cpu[slot] = int(gen)
            self._active_generation[int(slot)] = int(gen)
        return int(slot)

    @torch.no_grad()
    def remove_handle(self, request_handle: int) -> None:
        if not self._handles.is_alive(request_handle):
            return
        slot = unpack_request_slot(request_handle)
        self._reset_slot(int(slot))
        self._active_generation_cpu[slot] = -1
        self._active_generation[int(slot)] = -1

    # ---------------- write centroids ----------------

    @torch.no_grad()
    def append_centroids(
        self,
        request_handle: int,
        layer: int,
        kv_head: int,
        centroids: torch.Tensor,  # [K, D]
        cluster_size: torch.Tensor,  # [K]
        *,
        allow_async_copy: bool = False,
    ) -> CentroidWriteInfo:
        if centroids.ndim != 2:
            raise ValueError(f"centroids must be [K,D], got {tuple(centroids.shape)}")
        if cluster_size.ndim != 1:
            raise ValueError(
                f"cluster_size must be [K], got {tuple(cluster_size.shape)}"
            )
        K, D = centroids.shape
        if self.head_dim != D:
            raise ValueError(f"head_dim mismatch: {D} vs {self.head_dim}")
        if cluster_size.numel() != K:
            raise ValueError("cluster_size length mismatch")
        if not (0 <= layer < self.num_layers):
            raise IndexError("layer out of range")
        if not (0 <= kv_head < self.num_kv_heads):
            raise IndexError("kv_head out of range")
        if K <= 0:
            raise ValueError("K must be > 0")

        if centroids.dtype != self.dtype:
            centroids = centroids.to(self.dtype)
        if cluster_size.dtype != torch.int32:
            cluster_size = cluster_size.to(torch.int32)

        if centroids.device != self.device:
            if not allow_async_copy:
                raise ValueError(
                    "centroids must be on pool device (or set allow_async_copy=True)"
                )
            centroids = centroids.to(self.device, non_blocking=True)
        if cluster_size.device != self.device:
            cluster_size = cluster_size.to(self.device, non_blocking=True)

        slot = self._ensure_active_handle(request_handle)
        row = slot * self.num_kv_heads + int(kv_head)

        start = int(self.valid_k[layer][row].item())
        end = start + int(K)
        if end > self.max_num_centroids:
            raise RuntimeError(
                "Centroid overflow: "
                f"request_handle={request_handle}, layer={layer}, kv_head={kv_head}, "
                f"need end={end} > Kmax={self.max_num_centroids}."
            )

        base = int(self.next_cluster_id[layer][row].item())

        self.centroids[layer][row, start:end, :].copy_(centroids, non_blocking=True)
        self.cluster_size[layer][row, start:end].copy_(cluster_size, non_blocking=True)

        self.valid_k[layer][row] = end
        self.next_cluster_id[layer][row] = base + int(K)

        return CentroidWriteInfo(
            request_handle=int(request_handle),
            request_slot=int(slot),
            layer=int(layer),
            kv_head=int(kv_head),
            base_cluster_id=base,
            start=start,
            length=int(K),
        )

    # ---------------- k normalization ----------------

    @staticmethod
    def _normalize_k_spec(k: KSpec, B: int, device: torch.device) -> torch.Tensor:
        if isinstance(k, int):
            if k <= 0:
                raise ValueError("k must be > 0")
            return torch.full((B,), int(k), device=device, dtype=torch.int64)
        if not torch.is_tensor(k):
            raise TypeError("k must be int or tensor")
        if k.ndim != 1 or k.numel() != B:
            raise ValueError(f"k tensor must be [B], got {tuple(k.shape)}")
        if k.device != device:
            k = k.to(device)
        k = k.to(torch.int64)
        if torch.any(k <= 0):
            raise ValueError("k must be all > 0")
        return k

    # ---------------- fast path: by slots ----------------

    @torch.no_grad()
    def batch_topk_packed_by_slots_torch(
        self,
        request_slots: torch.Tensor,  # [B] int64, -1 means invalid
        layer: int,
        queries: torch.Tensor,  # [B, H, D]
        k: KSpec,
        *,
        sort_results: bool = True,
    ) -> TopKPackedResult:
        """
        Same semantics as RetroInfer demo, but takes request_slots directly
        (fastest, no Python dict).

        输入：
        - `request_slots`: shape [B]，每个 request 的 pool slot；
          -1 表示该 request 无效/未注册
        - `queries`: shape [B, num_heads, head_dim]，必须在 pool device 上
        - `k`: 可以是 int 或 shape=[B] 的张量（每个 request 允许不同的 TopK 需求）

        输出：
        - `TopKPackedResult.logical_cluster_ids`: shape [B, num_kv_heads, max_k]
          其中 `max_k = min(max(k), max_num_centroids)`。
          每个 request 的有效长度为 `min(k_per_req[b], valid_k[b,h])`。

        语义对齐：
        - logits = (Q @ C^T) * (1/sqrt(D))
        - 在 centroid 维做 softmax，得到每个 head 对每个 centroid 的概率分布
        - 对 GQA 的 group head 维求和，得到 dist
        - dist 做 TopK(largest=True)
        """
        if request_slots.ndim != 1:
            raise ValueError("request_slots must be [B]")
        if queries.ndim != 3:
            raise ValueError("queries must be [B,H,D]")
        B, H, D = queries.shape
        if H % int(self.group_size) != 0 or self.head_dim != D:
            raise ValueError("queries shape mismatch")
        if queries.device != self.device:
            raise ValueError("queries must be on pool device")
        if request_slots.device != self.device:
            request_slots = request_slots.to(self.device)
        if not (0 <= layer < self.num_layers):
            raise IndexError("layer out of range")

        device = queries.device
        k_per_req = self._normalize_k_spec(k, B, device)

        Hkv = int(H // int(self.group_size))
        Kmax = self.max_num_centroids
        G = self.group_size

        # rows: [B,Hkv] -> [BG]
        kv_heads = torch.arange(Hkv, device=device)
        rows = (
            request_slots[:, None] * int(self.num_kv_heads) + kv_heads[None, :]
        ).reshape(-1)  # [BG]
        invalid_bg = (request_slots[:, None] < 0).expand(B, Hkv).reshape(-1)  # [BG]

        safe_rows = rows.clamp_min(0)

        C = self.centroids[layer][safe_rows]  # [BG,Kmax,D]
        CS = self.cluster_size[layer][safe_rows]  # [BG,Kmax]
        VK = self.valid_k[layer][safe_rows]  # [BG]

        # queries -> [BG,G,D]
        A = queries.view(B, Hkv, G, D).reshape(-1, G, D)

        logits = torch.einsum(
            "bgd,bkd->bgk", A.to(torch.float32), C.to(torch.float32)
        ) * float(self._rsqrt_dim)
        probs = F.softmax(logits, dim=-1)
        dist = probs.sum(dim=1)  # [BG,Kmax]

        k_idx = self._k_idx_int32[None, :Kmax]
        dtype_min = self._dist_dtype_min
        dist.masked_fill_(CS == 0, dtype_min)
        dist.masked_fill_(k_idx >= VK[:, None], dtype_min)
        dist.masked_fill_(invalid_bg[:, None], dtype_min)

        max_k = int(k_per_req.max().item())
        max_k = min(max_k, Kmax)
        topi = torch.topk(
            dist, k=max_k, dim=-1, largest=True, sorted=sort_results
        ).indices  # [BG,max_k]

        base_cluster = self.next_cluster_id[layer][safe_rows] - VK.to(
            torch.int64
        )  # [BG]
        logical = base_cluster[:, None] + topi.to(torch.int64)  # [BG,max_k]
        logical = logical.view(B, Hkv, max_k)

        VK2 = VK.view(B, Hkv)

        return TopKPackedResult(
            logical_cluster_ids=logical.contiguous(),
            k_per_req=k_per_req.contiguous(),
            valid_k=VK2.contiguous(),
            request_slots=request_slots.to(torch.int64).contiguous(),
        )

    @torch.no_grad()
    def batch_topk_packed_by_handles_torch(
        self,
        request_handles: torch.Tensor,  # [B] int64
        layer: int,
        queries: torch.Tensor,  # [B,H,D]
        k: KSpec,
        *,
        sort_results: bool = True,
    ) -> TopKPackedResult:
        handles = request_handles.to(torch.int64).to(queries.device)
        request_slots_raw = request_slots_from_handles(handles).to(queries.device)
        request_gens = (handles >> 32).to(torch.int64)

        valid_slot = (request_slots_raw >= 0) & (request_slots_raw < int(self.max_requests))
        slots_safe = torch.where(valid_slot, request_slots_raw, torch.zeros_like(request_slots_raw))
        active_gen = self._active_generation.index_select(0, slots_safe.to(torch.int64)).to(torch.int64)
        alive = valid_slot & (active_gen == request_gens)
        request_slots = torch.where(alive, request_slots_raw, torch.full_like(request_slots_raw, -1))
        return self.batch_topk_packed_by_slots_torch(
            request_slots=request_slots,
            layer=layer,
            queries=queries,
            k=k,
            sorted=sort_results,
        )

    # ---------------- topk (cutlass) ----------------

    @torch.no_grad()
    def batch_topk_packed_by_slots_cutlass(
        self,
        request_slots: torch.Tensor,  # [B] int64, -1 means invalid
        layer: int,
        queries: torch.Tensor,  # [B, H, D]
        k: KSpec,
        *,
        sort_results: bool = True,
    ) -> TopKPackedResult:
        """
        cutlass backend 的 packed TopK（输入直接是 request_slots）。

        与 torch 版本保持同样的 dist 定义与 mask 规则，只是把
        `(Q @ C^T) + softmax` 的计算交给 `_paged_kmeans.batch_gemm_softmax`。
        """
        if request_slots.ndim != 1:
            raise ValueError("request_slots must be [B]")
        if queries.ndim != 3:
            raise ValueError("queries must be [B,H,D]")
        B, H, D = queries.shape
        if H % int(self.group_size) != 0 or self.head_dim != D:
            raise ValueError("queries shape mismatch")
        if queries.device != self.device:
            raise ValueError("queries must be on pool device")
        if queries.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("cutlass path requires fp16/bf16")
        if request_slots.device != self.device:
            request_slots = request_slots.to(self.device)
        if not (0 <= layer < self.num_layers):
            raise IndexError("layer out of range")

        device = queries.device
        k_per_req = self._normalize_k_spec(k, B, device)

        Hkv = int(H // int(self.group_size))
        Kmax = self.max_num_centroids
        G = self.group_size

        rows = (
            request_slots[:, None] * int(self.num_kv_heads) + torch.arange(Hkv, device=device)[None, :]
        ).reshape(-1)  # [BG]
        invalid_bg = (request_slots[:, None] < 0).expand(B, Hkv).reshape(-1)  # [BG]
        safe_rows = rows.clamp_min(0)

        C = self.centroids[layer][safe_rows]  # [BG,Kmax,D]
        CS = self.cluster_size[layer][safe_rows]  # [BG,Kmax]
        VK = self.valid_k[layer][safe_rows]  # [BG]

        A = queries.view(B, Hkv, G, D).reshape(-1, G, D).contiguous()  # [BG,G,D]

        D_out, softmax_o, norm, summ = self._get_cutlass_ws(
            BG=B * Hkv, G=G, Kmax=Kmax, dtype=queries.dtype
        )
        n_eff = int(torch.clamp(VK.max(), min=1, max=Kmax).item())

        rsqrt_dim = float(self._rsqrt_dim)
        _paged_kmeans.batch_gemm_softmax(
            A,
            C,
            D_out,
            norm,
            summ,
            softmax_o,
            B * Hkv,
            G,
            n_eff,
            self.head_dim,
            rsqrt_dim,
            0.0,
        )

        dtype_min = self._dist_dtype_min
        dist = softmax_o[:, :, :n_eff].sum(dim=1, dtype=torch.float32)  # [BG,n_eff]
        CS_eff = CS[:, :n_eff]
        k_idx = self._k_idx_int32[None, :n_eff]
        dist.masked_fill_(CS_eff == 0, dtype_min)
        dist.masked_fill_(k_idx >= VK[:, None], dtype_min)
        dist.masked_fill_(invalid_bg[:, None], dtype_min)

        max_k = int(k_per_req.max().item())
        max_k = min(max_k, n_eff)
        top = torch.topk(
            dist, k=max_k, dim=-1, largest=True, sorted=sort_results
        ).indices  # [BG,max_k]

        base_cluster = self.next_cluster_id[layer][safe_rows] - VK.to(
            torch.int64
        )  # [BG]
        logical = (base_cluster[:, None] + top.to(torch.int64)).view(B, Hkv, max_k)

        return TopKPackedResult(
            logical_cluster_ids=logical.contiguous(),
            k_per_req=k_per_req.contiguous(),
            valid_k=VK.view(B, Hkv).contiguous(),
            request_slots=request_slots.to(torch.int64).contiguous(),
        )

    @torch.no_grad()
    def batch_topk_packed_by_handles_cutlass(
        self,
        request_handles: torch.Tensor,  # [B] int64
        layer: int,
        queries: torch.Tensor,  # [B,H,D]
        k: KSpec,
        *,
        sort_results: bool = True,
    ) -> TopKPackedResult:
        handles = request_handles.to(torch.int64).to(queries.device)
        request_slots_raw = request_slots_from_handles(handles).to(queries.device)
        request_gens = (handles >> 32).to(torch.int64)

        valid_slot = (request_slots_raw >= 0) & (request_slots_raw < int(self.max_requests))
        slots_safe = torch.where(valid_slot, request_slots_raw, torch.zeros_like(request_slots_raw))
        active_gen = self._active_generation.index_select(0, slots_safe.to(torch.int64)).to(torch.int64)
        alive = valid_slot & (active_gen == request_gens)
        request_slots = torch.where(alive, request_slots_raw, torch.full_like(request_slots_raw, -1))
        return self.batch_topk_packed_by_slots_cutlass(
            request_slots=request_slots,
            layer=layer,
            queries=queries,
            k=k,
            sorted=sort_results,
        )

    @torch.no_grad()
    def batch_topk_by_handles(
        self,
        request_handles: torch.Tensor,  # [B] int64
        layer: int,
        queries: torch.Tensor,  # [B,H,D]
        k: KSpec,
        *,
        backend: str = "torch",
        sort_results: bool = True,
        return_format: ReturnFormat = "packed",
        request_ids: Sequence[str] | None = None,
    ):
        handles = request_handles.to(torch.int64).to(queries.device)
        request_slots_raw = request_slots_from_handles(handles).to(queries.device)
        request_gens = (handles >> 32).to(torch.int64)

        valid_slot = (request_slots_raw >= 0) & (request_slots_raw < int(self.max_requests))
        slots_safe = torch.where(valid_slot, request_slots_raw, torch.zeros_like(request_slots_raw))
        active_gen = self._active_generation.index_select(0, slots_safe.to(torch.int64)).to(torch.int64)
        alive = valid_slot & (active_gen == request_gens)
        request_slots = torch.where(alive, request_slots_raw, torch.full_like(request_slots_raw, -1))
        return self.batch_topk_by_slots(
            request_slots=request_slots,
            layer=layer,
            queries=queries,
            k=k,
            backend=backend,
            sorted=sort_results,
            return_format=return_format,
            request_ids=request_ids,
        )

    @torch.no_grad()
    def batch_topk_by_slots(
        self,
        request_slots: torch.Tensor,
        layer: int,
        queries: torch.Tensor,
        k: KSpec,
        *,
        backend: str = "torch",
        sort_results: bool = True,
        return_format: ReturnFormat = "packed",
        request_ids: Sequence[str] | None = None,
    ):
        """
        slot 版本的统一 TopK API（避免 request_id -> slot 的 Python dict 映射）。

        使用约定：
        - decode 主路径建议：`return_format="packed"`，并在 CUDA 上使用 `backend="cutlass"`
        - 若需要兼容 dict 输出（return_format="dict"），必须同时提供 `request_ids`，
          用于构造返回的 `{request_id: ...}` key。

        vLLM 视角的入参解释：
        - `request_slots` 需要与 `queries` 的 batch 维度对齐：`queries[b]` 属于哪个 request，
          `request_slots[b]` 就填该 request 在 CentroidPool 的 slot（由 `register_request()` 分配，
          或在首次 `append_centroids()` 时隐式分配）。
        - `request_slots` 的生命周期是“slot 级”的：只用于定位 dense layout 行。
          若采用 packed request_handle（推荐），则需要额外依赖 generation 来区分 slot 的不同生命周期。
          请求结束时应由上层调用 `remove_handle(request_handle)` 进行清理，然后再释放 handle。
        - 若某个 request 在当前 step 不需要 TopK（例如仍处于 prefill），调用方不应把它塞进 B 维度；
          或者把它对应的 `request_slots[b]` 设为 -1 以显式屏蔽输出。
        """
        if request_slots.ndim != 1:
            raise ValueError("request_slots must be [B]")
        if backend == "torch":
            packed = self.batch_topk_packed_by_slots_torch(
                request_slots=request_slots,
                layer=layer,
                queries=queries,
                k=k,
                sorted=sort_results,
            )
        elif backend == "cutlass":
            packed = self.batch_topk_packed_by_slots_cutlass(
                request_slots=request_slots,
                layer=layer,
                queries=queries,
                k=k,
                sorted=sort_results,
            )
        else:
            raise ValueError(f"unknown backend: {backend}")

        if return_format == "packed":
            return packed

        if request_ids is None:
            raise ValueError("return_format='dict' requires request_ids")

        Hkv = self.num_kv_heads
        max_k = packed.logical_cluster_ids.shape[-1]

        rid_list = list(request_ids)
        if len(rid_list) != int(packed.request_slots.shape[0]):
            raise ValueError("request_ids length mismatch with request_slots batch size")

        out: dict[str, dict[int, torch.Tensor]] = {}
        for b, rid in enumerate(rid_list):
            kk = int(packed.k_per_req[b].item())
            per_h: dict[int, torch.Tensor] = {}
            for h in range(Hkv):
                valid = int(packed.valid_k[b, h].item())
                k_eff = min(kk, valid, max_k)
                if packed.request_slots[b].item() < 0 or k_eff <= 0:
                    per_h[h] = torch.empty((0,), device=self.device, dtype=torch.int64)
                else:
                    per_h[h] = packed.logical_cluster_ids[b, h, :k_eff].contiguous()
            out[rid] = per_h
        return out
