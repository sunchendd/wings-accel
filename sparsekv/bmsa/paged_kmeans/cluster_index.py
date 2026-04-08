from __future__ import annotations

from dataclasses import dataclass

import torch

from .utils import RequestHandleAllocator


@dataclass
class IndexChunk:
    """
    一个“增量聚类段”的 CSR 索引切片（per request / layer / kv_head）。

    这里的 CSR 语义与 `PagedKMeansClusterer.cluster()` 的输出对齐：
    - `offsets`: shape [K+1]，offsets[c]..offsets[c+1] 是 cluster c 的 token 列表区间
    - `perm`: shape [L]，按 cluster 分组后的 token_index 列表
      （token_index 是窗口内的 0..L-1）
    - `token_pos`: shape [L]，窗口内 token_index -> 全局 position id 的映射

    `base_cluster_id` 用于把不同 chunk 的 cluster id 拼接成全局“逻辑 cluster id”：
    - 例如第一次 prefill 产生 K 个 cluster，则其 base_cluster_id=0
    - 下一次增量窗口再产生 K 个 cluster，则 base_cluster_id=K
    - 对外暴露的 cluster_id 就是 base_cluster_id + local_cluster_id
    """

    base_cluster_id: int
    offsets: torch.Tensor     # [K+1]
    perm: torch.Tensor        # [L]
    token_pos: torch.Tensor   # [L]
    token_pos_perm: torch.Tensor  # [L], token_pos[perm]


@dataclass(frozen=True)
class ClusterKVBatched:
    """
    ClusterIndex 从 paged KV cache 中“按簇连续”提取出来的一批 KV 数据（per kv_head）。

    字段语义：
    - `cluster_ids`: shape [C]，int64，本批次包含的全局逻辑 cluster id（可不连续）
    - `offsets`: shape [C+1]，int32，prefix-sum，cluster i 的 token 区间为
      [offsets[i], offsets[i+1])
    - `kv`: shape [2, T, head_dim]，与 vLLM kv_cache 对齐的 (K,V) 数据
      - T = offsets[-1]，为所有 cluster token 的总数
      - token 维度按 cluster 分组连续存放（cluster-major）
    - `token_pos`: shape [T]，int32，与 `kv` 的 token 维一一对应的 position id
      - 便于后续调试/校验，也可用于 CPUKVStore 的回写排序策略
    """

    cluster_ids: torch.Tensor
    offsets: torch.Tensor
    kv: torch.Tensor
    token_pos: torch.Tensor


class ClusterIndex:
    """
    基于 CSR 的 cluster -> token position 反查索引

    设计目的：
    - 聚类发生在 GPU 上，输出 cluster->tokens 的分组信息（CSR）
    - Decode 阶段的 TopK 输出是 cluster_id
    - 需要通过本索引把 cluster_id 快速还原成一组topk blocks
      （后续用于 PrefetchEngine 召回）
    """

    def __init__(self, *, handle_allocator: RequestHandleAllocator):
        self._handles = handle_allocator
        self.chunks: dict[tuple[int, int, int], list[IndexChunk]] = {}

    @staticmethod
    def _gather_token_pos_from_chunk(
        ch: IndexChunk, cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        从单个 chunk 中批量提取 cluster_ids 对应的 token position。

        这里的实现避免了 “for cid in cluster_ids: slice offsets” 的 Python 循环，
        而是把一批 clusters 的 CSR 区间拼接成一个扁平化 gather：

        输入：
        - ch.offsets: [K+1]
        - ch.perm: [L]
        - ch.token_pos: [L]
        - cluster_ids: [C]（逻辑 cluster id，保证落在该 chunk 覆盖范围内）

        输出：
        - positions: [T]，T 为这些 clusters 的 token 总数
          顺序不保证全局有序；调用方如需稳定顺序可自行 sort/unique。
        """
        device = ch.token_pos.device
        if cluster_ids.numel() == 0:
            return torch.empty((0,), device=device, dtype=torch.int32)

        base = int(ch.base_cluster_id)
        local = (cluster_ids.to(torch.int64) - int(base)).to(torch.int64)  # [C]

        # starts/ends: [C]
        L = int(ch.perm.numel())
        starts = ch.offsets.index_select(0, local.to(torch.int64)).to(torch.int64)
        ends = ch.offsets.index_select(0, (local + 1).to(torch.int64)).to(torch.int64)
        starts = starts.clamp_min(0).clamp_max(L)
        ends = ends.clamp_min(0).clamp_max(L)
        lens = (ends - starts).clamp_min(0)
        nz = lens > 0
        if not bool(nz.any()):
            return torch.empty((0,), device=device, dtype=torch.int32)

        starts = starts[nz]
        lens = lens[nz]

        total = int(lens.sum().item())
        if total <= 0:
            return torch.empty((0,), device=device, dtype=torch.int32)

        # 将可变长 segments 拼接成扁平索引：
        # seg_ends: 每个 segment 的结束位置（exclusive）
        # seg_id: flat_idx 属于第几个 segment
        seg_ends = torch.cumsum(lens, dim=0)  # [C]
        flat_idx = torch.arange(total, device=device, dtype=torch.int64)  # [T]
        seg_id = torch.bucketize(flat_idx, seg_ends, right=True)  # [T] in [0,C)

        seg_ends_prev = torch.cat(
            [torch.zeros((1,), device=device, dtype=torch.int64), seg_ends[:-1]],
            dim=0,
        )  # [C]
        in_seg = flat_idx - seg_ends_prev.index_select(0, seg_id)  # [T]
        perm_idx = starts.index_select(0, seg_id) + in_seg  # [T]

        return ch.token_pos_perm.index_select(0, perm_idx.to(torch.int64)).to(torch.int32)

    def add_chunk(
        self,
        request_handle: int,
        layer: int,
        kv_head: int,
        base_cluster_id: int,
        offsets: torch.Tensor,
        perm: torch.Tensor,
        token_pos: torch.Tensor,
    ):
        """
        追加一个聚类 chunk 的 CSR 索引（对应一次 prefill 或一次增量窗口聚类）

        约定：
        - `offsets/perm/token_pos` 的语义见 `IndexChunk` docstring
        - `base_cluster_id` 必须与 `CentroidPool.append_centroids()` 返回的
          `base_cluster_id` 对齐
          以保证 decode TopK 返回的“全局 cluster id”能够路由到正确 chunk
        """
        if not self._handles.is_alive(int(request_handle)):
            raise KeyError("request_handle not alive in cluster index")
        assert token_pos.ndim == 1
        token_pos_perm = token_pos.index_select(0, perm.to(torch.int64)).to(torch.int32)
        key = (int(request_handle), int(layer), int(kv_head))
        self.chunks.setdefault(key, []).append(
            IndexChunk(
                base_cluster_id=base_cluster_id,
                offsets=offsets,
                perm=perm,
                token_pos=token_pos,
                token_pos_perm=token_pos_perm,
            )
        )

    @torch.no_grad()
    def lookup_token_pos(
        self,
        request_handle: int,
        layer: int,
        kv_head: int,
        cluster_ids: list[int],
    ) -> torch.Tensor:
        if not self._handles.is_alive(int(request_handle)):
            return torch.empty(0, dtype=torch.int32)
        key = (int(request_handle), int(layer), int(kv_head))
        if key not in self.chunks:
            return torch.empty(0, dtype=torch.int32)
        if not cluster_ids:
            device = self.chunks[key][0].token_pos.device
            return torch.empty(0, device=device, dtype=torch.int32)

        out = []
        for cid in cluster_ids:
            for ch in self.chunks[key]:
                lo = ch.base_cluster_id
                hi = lo + ch.offsets.numel() - 1
                if lo <= cid < hi:
                    local = cid - lo
                    s = ch.offsets[local]
                    e = ch.offsets[local + 1]
                    out.append(ch.token_pos_perm[s:e])

        if not out:
            device = self.chunks[key][0].token_pos.device
            return torch.empty(0, device=device, dtype=torch.int32)

        return torch.sort(torch.cat(out)).values

    @torch.no_grad()
    def lookup_token_pos_tensor(
        self,
        request_handle: int,
        layer: int,
        kv_head: int,
        cluster_ids: torch.Tensor,  # [C] int64/int32
        *,
        sorted_unique: bool = True,
    ) -> torch.Tensor:
        """
        `lookup_token_pos()` 的 tensor 版本（高性能，避免 per-cluster Python 循环）。

        输入：
        - cluster_ids: 1D 张量，逻辑 cluster id（可以不排序、可以重复）

        输出：
        - positions: 1D int32 张量
          - 若 sorted_unique=True：返回去重并升序排序后的 positions（便于后续 block 统计）
          - 否则：返回拼接结果（可能重复、无序）
        """
        if not self._handles.is_alive(int(request_handle)):
            return torch.empty((0,), dtype=torch.int32, device=cluster_ids.device)
        key = (int(request_handle), int(layer), int(kv_head))
        if key not in self.chunks:
            return torch.empty((0,), dtype=torch.int32, device=cluster_ids.device)
        if cluster_ids.ndim != 1:
            raise ValueError("cluster_ids must be 1D tensor")
        if cluster_ids.numel() == 0:
            device = self.chunks[key][0].token_pos.device
            return torch.empty((0,), device=device, dtype=torch.int32)

        cids = cluster_ids.to(torch.int64)
        out: list[torch.Tensor] = []
        for ch in self.chunks[key]:
            lo = int(ch.base_cluster_id)
            hi = lo + int(ch.offsets.numel()) - 1
            mask = (cids >= lo) & (cids < hi)
            if not bool(mask.any()):
                continue
            out.append(self._gather_token_pos_from_chunk(ch, cids[mask]))

        if not out:
            device = self.chunks[key][0].token_pos.device
            return torch.empty((0,), device=device, dtype=torch.int32)

        pos = torch.cat(out, dim=0).to(torch.int32)
        if not sorted_unique:
            return pos
        # 对 decode 的 block 覆盖统计来说，“unique 后再 histogram”更接近 max-coverage under budget 的目标：
        # 我们关心的是“覆盖了多少不同 token 位置”，而不是 per-kvhead 对同一 token 的重复计数。
        return torch.unique(pos, sorted=True).to(torch.int32)


    def remove_handle(self, request_handle: int):
        """
        清理某个 request 的所有索引。

        该方法必须与 `CentroidPool.remove_handle()` 同步调用，否则会出现：
        - 内存随请求增长而泄漏
        - request_handle 复用或 slot 复用时发生跨请求污染
        """
        for k in list(self.chunks.keys()):
            if k[0] == int(request_handle):
                del self.chunks[k]

    @torch.no_grad()
    def select_blocks_from_topk_clusters(
        self,
        *,
        request_handle: int,
        layer: int,
        per_kvhead_cluster_ids: list[torch.Tensor],
        block_size: int,
        num_prompt_blocks: int,
        budget_blocks: int,
        mandatory_block_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """
        将 “cluster-level TopK” 转换为 “block-level selection under budget”。

        背景（与 BMSA 的对接语义一致）：
        - BMSA 下游（PrefetchEngine + block_table 替换）只认识 block 粒度的索引；
        - paged-kmeans TopK 的直接输出是 cluster id（token-level 聚类空间）；
        - 因此需要做近似覆盖：在 block budget 约束下，尽量让选出的 blocks 覆盖更多
          TopK clusters 对应的 token。

        本实现选择了一个性能/语义折中但非常高效的近似：
        1) 对每个 kv_head，把 TopK cluster_ids 还原成 token positions；
        2) 对所有 kv_head 做 union（positions 去重）；
        3) 将 positions 映射到 block_index = pos // block_size，并对每个 block 做计数；
        4) 在 budget_blocks 内选取计数最大的 blocks（并注入 mandatory blocks）。

        说明：
        - 这是 “max-coverage under uniform cost + disjoint blocks” 的近似：
          tokens 天然按 block 分区，因此在 union 语义下，选 Top-count blocks
          等价于最大化覆盖 token 数。
        - mandatory_block_indices 用于兼容 BMSA 的一些稳定性策略（例如固定保留前几个 blocks）。

        返回：
        - 1D int64 张量，长度为 budget_blocks（若可选 blocks 不足，会用 0 填充）
        - 每个元素是 “prompt block 的逻辑 index”（0..num_prompt_blocks-1），用于进一步映射到物理 blockID
        """
        device = None
        for c in per_kvhead_cluster_ids:
            if c.numel() > 0:
                device = c.device
                break
        if device is None:
            # 无 cluster 输入：仅返回 mandatory（或空）并 pad
            out = []
            if mandatory_block_indices:
                out = [int(x) for x in mandatory_block_indices if 0 <= int(x) < int(num_prompt_blocks)]
            out = out[: max(0, int(budget_blocks))]
            if len(out) < int(budget_blocks):
                out = out + [0] * (int(budget_blocks) - len(out))
            return torch.tensor(out, device="cpu", dtype=torch.int64)

        if budget_blocks <= 0 or num_prompt_blocks <= 0:
            return torch.empty((0,), device=device, dtype=torch.int64)

        if mandatory_block_indices is None:
            mandatory_block_indices = []
        mandatory = [int(x) for x in mandatory_block_indices if 0 <= int(x) < int(num_prompt_blocks)]
        # 去重且保持顺序
        seen = set()
        mandatory = [x for x in mandatory if not (x in seen or seen.add(x))]
        mandatory = mandatory[: int(budget_blocks)]

        mask_len = int(num_prompt_blocks) * int(block_size)
        token_mask = torch.zeros((mask_len,), device=device, dtype=torch.bool)
        for kv_head, cids in enumerate(per_kvhead_cluster_ids):
            if cids.numel() == 0:
                continue
            pos = self.lookup_token_pos_tensor(
                request_handle=request_handle,
                layer=layer,
                kv_head=int(kv_head),
                cluster_ids=cids,
                sorted_unique=False,
            )
            if pos.numel() <= 0:
                continue
            pos64 = pos.to(torch.int64)
            pos64 = pos64.clamp_min(0).clamp_max(mask_len - 1)
            token_mask[pos64] = True

        if not bool(token_mask.any()):
            out = mandatory
            if len(out) < int(budget_blocks):
                out = out + [0] * (int(budget_blocks) - len(out))
            return torch.tensor(out, device=device, dtype=torch.int64)

        scores = token_mask.view(int(num_prompt_blocks), int(block_size)).sum(dim=1).to(
            torch.float32
        )

        if mandatory:
            scores[torch.tensor(mandatory, device=scores.device, dtype=torch.int64)] = float("inf")

        # 4) top-k blocks
        k_need = int(budget_blocks) - len(mandatory)
        if k_need <= 0:
            return torch.tensor(mandatory[: int(budget_blocks)], device=device, dtype=torch.int64)

        top = torch.topk(scores, k=min(k_need, int(num_prompt_blocks)), largest=True, sorted=True).indices
        top_list = top.to(torch.int64).tolist()
        picked = mandatory + [int(x) for x in top_list if int(x) not in set(mandatory)]
        picked = picked[: int(budget_blocks)]
        if len(picked) < int(budget_blocks):
            picked = picked + [0] * (int(budget_blocks) - len(picked))
        return torch.tensor(picked, device=device, dtype=torch.int64)
