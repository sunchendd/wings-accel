import os
import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F


def _seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ReqSpec:
    """单个请求的规格：逻辑长度、起始位置（在它自己的序列内）"""
    req_id: int
    seq_len_total: int


@dataclass
class BatchSchedule:
    """
    模拟“连续批处理”：把多个请求的 token 在当前 step 拼成一个大列表。
    scheduled_req_ids: 本 step 被调度的 req 列表（子集）
    per_req_new_tokens: 每个被调度 req 本 step 新增 token 数（prefill 可等于 seq_len_total；decode 通常=1~n）
    """
    scheduled_req_ids: List[int]
    per_req_new_tokens: Dict[int, int]


@dataclass
class PagedCacheState:
    """
    vLLM 风格 paged cache 状态（简化版，但满足测试需要）
    kv_cache: [2, num_blocks, block_size, H, D]
    block_table: [num_reqs, max_blocks_per_req] 逻辑block->物理block，未用填 0
    cur_lens: [num_reqs] 当前已写入 token 数（decode/prefill 更新它）
    """
    kv_cache: torch.Tensor
    block_table: torch.Tensor
    cur_lens: torch.Tensor
    block_size: int
    num_blocks: int
    num_kv_heads: int
    head_dim: int


def build_random_noncontiguous_block_table(
    num_reqs: int,
    blocks_per_req: List[int],
    num_blocks: int,
    device: str,
    ensure_noncontig: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """
    构造非连续物理块映射 block_table（逻辑 block id -> 物理 block id）
    - 确保每个 req 的 blocks 尽量不连续（随机抽取）
    """
    _seed_all(seed)
    max_bpr = max(blocks_per_req)
    block_table = torch.zeros((num_reqs, max_bpr), dtype=torch.int32, device=device)

    all_blocks = list(range(num_blocks))
    random.shuffle(all_blocks)
    cursor = 0

    for r in range(num_reqs):
        n = blocks_per_req[r]
        assert cursor + n <= len(all_blocks), "num_blocks 不足以分配给所有请求"
        chosen = all_blocks[cursor:cursor + n]
        cursor += n

        if ensure_noncontig and n >= 3:
            # 尝试避免完全连续（不是严格保证，但概率很低）
            chosen_sorted = sorted(chosen)
            is_contig = all((chosen_sorted[i] + 1 == chosen_sorted[i + 1]) for i in range(n - 1))
            if is_contig:
                # 简单扰动：交换中间两个
                chosen[1], chosen[-2] = chosen[-2], chosen[1]

        block_table[r, :n] = torch.tensor(chosen, dtype=torch.int32, device=device)

    return block_table


def init_paged_cache_state(
    reqs: List[ReqSpec],
    block_size: int,
    num_blocks: int,
    num_kv_heads: int,
    head_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
) -> PagedCacheState:
    """
    初始化 paged KVCache + block_table（非连续）
    kv_cache 初始为 0，后续由 write_tokens_to_paged_cache 写入。
    """
    _seed_all(seed)
    num_reqs = len(reqs)
    blocks_per_req = [math.ceil(r.seq_len_total / block_size) for r in reqs]

    kv_cache = torch.zeros((2, num_blocks, block_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    block_table = build_random_noncontiguous_block_table(
        num_reqs=num_reqs,
        blocks_per_req=blocks_per_req,
        num_blocks=num_blocks,
        device=device,
        ensure_noncontig=True,
        seed=seed + 17,
    )
    cur_lens = torch.zeros((num_reqs,), dtype=torch.int32, device=device)

    return PagedCacheState(
        kv_cache=kv_cache,
        block_table=block_table,
        cur_lens=cur_lens,
        block_size=block_size,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def _write_one_token(
    state: PagedCacheState,
    req_id: int,
    token_idx_in_req: int,
    k: torch.Tensor,  # [H, D]
    v: torch.Tensor,  # [H, D]
):
    """
    将单个 token 的 K/V 写入 paged kv_cache 的正确 slot
    token_idx_in_req: 该 token 在该请求序列内的绝对位置（从 0 开始）
    """
    bs = state.block_size
    logical_block = token_idx_in_req // bs
    offset = token_idx_in_req % bs
    phys = int(state.block_table[req_id, logical_block].item())
    state.kv_cache[0, phys, offset].copy_(k)
    state.kv_cache[1, phys, offset].copy_(v)


def simulate_step_and_write(
    state: PagedCacheState,
    schedule: BatchSchedule,
    *,
    dtype: Optional[torch.dtype] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, Tuple[int, int]]]:
    """
    模拟一次 vLLM step 的“连续批处理”：
    - 对 schedule.scheduled_req_ids 里的请求，生成 per_req_new_tokens[req] 个新 token
    - 把这些 token 的 K/V 写入 paged kv_cache（位置=cur_len..cur_len+new_tokens-1）
    - 返回：
        keys_contig: [T, H, D] 本 step 的 token 列表（跨请求拼接）
        values_contig: [T, H, D]
        slot_mapping: [T] 每个 token 对应写入到哪个物理 slot（phys_block*block_size + offset）
        req_ranges: {req_id: (start_index_in_T, length)} 方便做子集提取
    注：
    - slot_mapping 是为了模拟 vLLM 真实输入；本测试中主要用于验证映射正确性/抽取子集。
    """
    _seed_all(seed)
    device = state.kv_cache.device
    H, D = state.num_kv_heads, state.head_dim
    use_dtype = dtype if dtype is not None else state.kv_cache.dtype

    tokens_k = []
    tokens_v = []
    slot_mapping = []
    req_ranges: Dict[int, Tuple[int, int]] = {}

    t_cursor = 0
    for rid in schedule.scheduled_req_ids:
        new_toks = schedule.per_req_new_tokens[rid]
        start = t_cursor
        for i in range(new_toks):
            abs_pos = int(state.cur_lens[rid].item()) + i

            # 随机生成 K/V（可复现）
            k = torch.randn((H, D), device=device, dtype=use_dtype)
            v = torch.randn((H, D), device=device, dtype=use_dtype)

            _write_one_token(state, rid, abs_pos, k, v)

            # 生成 slot_mapping
            bs = state.block_size
            logical_block = abs_pos // bs
            offset = abs_pos % bs
            phys = int(state.block_table[rid, logical_block].item())
            slot_mapping.append(phys * bs + offset)

            tokens_k.append(k)
            tokens_v.append(v)
            t_cursor += 1

        req_ranges[rid] = (start, new_toks)
        state.cur_lens[rid] += new_toks

    keys_contig = torch.stack(tokens_k, dim=0) if tokens_k else torch.empty((0, H, D), device=device, dtype=use_dtype)
    values_contig = torch.stack(tokens_v, dim=0) if tokens_v else torch.empty((0, H, D), device=device, dtype=use_dtype)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    return keys_contig, values_contig, slot_mapping, req_ranges


def extract_req_kv_contiguous_from_paged(
    state: PagedCacheState,
    req_id: int,
    start_pos: int,
    L: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 paged KVCache 抽取某个 req 的一段连续 KV（用于对齐 reference 连续实现）
    输出 key/value: [H, L, D]
    """
    device = state.kv_cache.device
    H, D = state.num_kv_heads, state.head_dim
    key = torch.empty((H, L, D), device=device, dtype=state.kv_cache.dtype)
    val = torch.empty_like(key)
    bs = state.block_size
    bt = state.block_table[req_id]

    for t in range(L):
        pos = start_pos + t
        logical_block = pos // bs
        offset = pos % bs
        phys = int(bt[logical_block].item())
        key[:, t].copy_(state.kv_cache[0, phys, offset])
        val[:, t].copy_(state.kv_cache[1, phys, offset])
    return key, val


def build_contiguous_kv_from_req(
    state: PagedCacheState,
    req_id: int,
    upto_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    把某个 req 的 [0:upto_len] 全部 KV 抽成连续张量供 reference 用：
      key/value: [H, upto_len, D]
    """
    return extract_req_kv_contiguous_from_paged(state, req_id, 0, upto_len)


@torch.no_grad()
def centroid_cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    a,b: [H, K, D] 或 reshape 后 [-1, D]
    """
    a2 = a.reshape(-1, a.shape[-1]).to(torch.float32)
    b2 = b.reshape(-1, b.shape[-1]).to(torch.float32)
    return float(F.cosine_similarity(a2, b2, dim=-1).mean().item())


@torch.no_grad()
def mean_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean().item())


@torch.no_grad()
def assignment_match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    a,b: [H, L] int32
    """
    assert a.shape == b.shape
    return float((a == b).to(torch.float32).mean().item())


@torch.no_grad()
def cnt_l1_err(cnt_a: torch.Tensor, cnt_b: torch.Tensor) -> float:
    return float((cnt_a.to(torch.int64) - cnt_b.to(torch.int64)).abs().sum().item())


def bench_cuda_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    """
    CUDA 基准：返回平均毫秒
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / iters

