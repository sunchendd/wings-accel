from __future__ import annotations

import torch

from vsparse.bmsa.paged_kmeans import PagedKMeansConfig


def make_cfg(
    *,
    num_layers: int = 4,
    num_kv_heads: int = 2,
    num_heads: int = 8,
    head_dim: int = 16,
    pool_size_per_layer: int = 4096,
    max_requests: int = 16,
    max_num_centroids: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """
    构造一个尽可能“最小化依赖”的 `PagedKMeansConfig`，用于单元测试。

    设计原则：
    - 测试 `CentroidPool/ClusterIndex/PagedKMeansClusterManager` 时，
      不引入 Triton/CUDA 依赖
    - `block_size/num_heads/num_kv_heads/head_dim` 等字段只用于 shape/语义校验
    - `dtype/device` 允许在 CPU 上跑通大部分测试，CUDA 相关测试会自行 skip
    """
    if max_num_centroids is None:
        max_num_centroids = int(pool_size_per_layer)
    return PagedKMeansConfig(
        block_size=16,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        num_heads=num_heads,
        max_requests=max_requests,
        max_num_centroids=max_num_centroids,
    )


@torch.no_grad()
def build_csr_from_assign(
    assign_1d: torch.Tensor, K: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    由 `assign[L]`（每个 token 属于哪个 cluster）构造 CSR。

    输入：
    - `assign_1d`: shape [L], 值域 [0, K-1]，dtype 允许 int32/int64
    - `K`: cluster 数

    输出（与 `PagedKMeansClusterer.cluster()` 的 CSR 语义对齐）：
    - `cnt`: shape [K], int32，每个 cluster 的 token 数
    - `offsets`: shape [K+1], int32，prefix-sum；cluster c 的 token 区间为
      [offsets[c], offsets[c+1])
    - `perm`: shape [L], int32，按 cluster 分组后的 token_index 列表
      （token_index 是窗口内的 0..L-1）

    失败含义：
    - 若该函数的 CSR 与被测实现 CSR 语义不一致，会导致 `ClusterIndex.lookup_token_pos`
      等测试在“应当通过”时失败，从而提示 CSR 约定发生了漂移。
    """
    assert assign_1d.ndim == 1
    L = assign_1d.numel()
    device = assign_1d.device

    cnt = torch.bincount(assign_1d.to(torch.int64), minlength=K).to(torch.int32)
    offsets = torch.zeros((K + 1,), device=device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(cnt, dim=0)

    # perm：稳定分组（cluster 0..K-1，各自内部按 token idx 升序）
    perm = torch.empty((L,), device=device, dtype=torch.int32)
    write = offsets[:-1].clone()
    token_idx = torch.arange(L, device=device, dtype=torch.int32)
    # 稳定：按 token 顺序写入各 cluster
    for t in range(L):
        c = int(assign_1d[t].item())
        pos = int(write[c].item())
        perm[pos] = token_idx[t]
        write[c] += 1

    return cnt, offsets, perm


class FakePagedKMeansClusterer:
    """
    用于 `PagedKMeansClusterManager` 集成测试的“可控 clusterer”。

    目的：
    - 让 manager 的测试不依赖 Triton/CUDA 环境（可在纯 CPU 下跑通）
    - 输出的 shape/CSR 语义与真实 `PagedKMeansClusterer.cluster()` 对齐

    行为：
    - `assign` 固定为 `token_idx % K`，因此每个 cluster 的 token 分布可预期且
      deterministic
    - `centroids/sum_v` 置零即可（manager 测试只关心索引拼接与 cluster_id 规划）
    """

    def __init__(self, cfg, *, H: int, D: int):
        self.cfg = cfg
        self.H = H
        self.D = D

    @torch.no_grad()
    def cluster(
        self,
        kv_cache,
        block_table_1d,
        start_pos: torch.Tensor,
        L: int,
        num_segments: int = 1,
        num_centroids: int = 64,
    ):
        device = (
            kv_cache.device
            if isinstance(kv_cache, torch.Tensor)
            else torch.device(self.cfg.device)
        )
        H, D = self.H, self.D
        K = num_centroids

        # 伪造输出：centroids / sum_v 只是随机但固定 seed 不做（避免测试依赖随机）
        centroids = torch.zeros((H, K, D), device=device, dtype=torch.float32)
        sum_v = torch.zeros((H, K, D), device=device, dtype=torch.float32)

        assign = torch.empty((H, L), device=device, dtype=torch.int32)
        cnt = torch.empty((H, K), device=device, dtype=torch.int32)
        offsets = torch.empty((H, K + 1), device=device, dtype=torch.int32)
        perm = torch.empty((H, L), device=device, dtype=torch.int32)

        base_assign = torch.arange(L, device=device, dtype=torch.int32) % K

        for h in range(H):
            a = base_assign.clone()
            assign[h] = a
            c, off, p = build_csr_from_assign(a, K)
            cnt[h] = c
            offsets[h] = off
            perm[h] = p

        return centroids, sum_v, assign, cnt, offsets, perm
