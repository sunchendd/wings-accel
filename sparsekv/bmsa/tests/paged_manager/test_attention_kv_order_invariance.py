import importlib.util
import math

import pytest
import torch


def _apply_simple_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    A minimal RoPE implementation for testing invariance of attention to KV order.

    Args:
        x: [T, H, D] tensor (float16/bfloat16/float32).
        positions: [T] int64 positions.
    Returns:
        [T, H, D] tensor with RoPE applied on the last dimension.
    """
    if x.ndim != 3:
        raise ValueError("x must be [T, H, D]")
    T, _, D = x.shape
    if D % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if positions.shape != (T,):
        raise ValueError("positions must be [T]")

    half = D // 2
    dtype = x.dtype
    device = x.device

    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    theta = positions.to(device=device, dtype=torch.float32).view(T, 1) * inv_freq.view(1, half)
    cos = torch.cos(theta).to(dtype=dtype)
    sin = torch.sin(theta).to(dtype=dtype)

    x1 = x[..., :half]
    x2 = x[..., half:]

    out1 = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
    out2 = x1 * sin.unsqueeze(1) + x2 * cos.unsqueeze(1)
    return torch.cat([out1, out2], dim=-1)


def _attention_single_query(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Reference attention for a single query position (decode-like).

    Args:
        q: [H, D]
        k: [N, H, D]
        v: [N, H, D]
    Returns:
        out: [H, D]
    """
    if q.ndim != 2 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q=[H,D], k=v=[N,H,D]")
    if k.shape != v.shape:
        raise ValueError("k/v shape mismatch")
    if (k.shape[1], k.shape[2]) != q.shape:
        raise ValueError("q and k/v head dims mismatch")

    D = int(q.shape[1])
    scores = torch.einsum("hd,nhd->hn", q, k) / math.sqrt(D)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hn,nhd->hd", probs, v)
    return out


@torch.no_grad()
def test_decode_attention_invariant_to_kv_token_permutation_cpu() -> None:
    """
    对 decode-like attention（单 query）而言，只要 K/V 成对一致重排，token 顺序不影响输出。
    这里同时对 K 应用 RoPE，以贴近“KV 已融合 RoPE 信息”的场景。
    """
    torch.manual_seed(0)
    N, H, D = 257, 4, 64

    k = torch.randn((N, H, D), dtype=torch.float32)
    v = torch.randn((N, H, D), dtype=torch.float32)
    q = torch.randn((H, D), dtype=torch.float32)

    pos = torch.arange(N, dtype=torch.int64)
    k = _apply_simple_rope(k, pos)

    out0 = _attention_single_query(q, k, v)

    perm = torch.randperm(N)
    out1 = _attention_single_query(q, k.index_select(0, perm), v.index_select(0, perm))

    torch.testing.assert_close(out0, out1, rtol=1e-5, atol=1e-5)


HAS_VLLM_FLASH_ATTN = importlib.util.find_spec("vllm.vllm_flash_attn") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
@pytest.mark.skipif(not HAS_VLLM_FLASH_ATTN, reason="vllm_flash_attn is not available.")
@torch.no_grad()
def test_vllm_flash_attn_decode_invariant_to_paged_kv_permutation_cuda() -> None:
    """
    用 vLLM 的 flash_attn_varlen_func（paged KV + block_table）做一次最小实验：
    仅 decode（每序列 1 个 query），把 KV 在 paged cache 内做 token 维度的乱序，
    验证输出保持不变。
    """
    from vllm.attention.utils.fa_utils import get_flash_attn_version
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    device = torch.device("cuda")

    block_size = 16
    blocks_used = 4
    num_blocks = 8
    H = 4
    D = 64
    N = blocks_used * block_size

    q = torch.randn((1, H, D), device=device, dtype=torch.float16)

    k_seq = torch.randn((N, H, D), device=device, dtype=torch.float16)
    v_seq = torch.randn((N, H, D), device=device, dtype=torch.float16)
    pos = torch.arange(N, device=device, dtype=torch.int64)
    k_seq = _apply_simple_rope(k_seq, pos)

    key_cache = torch.zeros((num_blocks, block_size, H, D), device=device, dtype=torch.float16)
    value_cache = torch.zeros((num_blocks, block_size, H, D), device=device, dtype=torch.float16)
    key_cache[:blocks_used].view(N, H, D).copy_(k_seq)
    value_cache[:blocks_used].view(N, H, D).copy_(v_seq)

    perm = torch.randperm(N, device=device)
    key_cache_perm = key_cache.clone()
    value_cache_perm = value_cache.clone()
    key_cache_perm[:blocks_used].view(N, H, D).copy_(k_seq.index_select(0, perm))
    value_cache_perm[:blocks_used].view(N, H, D).copy_(v_seq.index_select(0, perm))

    cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([N], device=device, dtype=torch.int32)
    block_table = torch.arange(blocks_used, device=device, dtype=torch.int32).view(1, blocks_used)

    out0 = torch.empty((1, H, D), device=device, dtype=torch.float16)
    out1 = torch.empty((1, H, D), device=device, dtype=torch.float16)

    fa_version = get_flash_attn_version()
    flash_attn_varlen_func(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out0,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=N,
        softmax_scale=1.0 / math.sqrt(D),
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        scheduler_metadata=None,
        fa_version=fa_version,
        num_splits=0,
        s_aux=None,
    )
    flash_attn_varlen_func(
        q=q,
        k=key_cache_perm,
        v=value_cache_perm,
        out=out1,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=N,
        softmax_scale=1.0 / math.sqrt(D),
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        scheduler_metadata=None,
        fa_version=fa_version,
        num_splits=0,
        s_aux=None,
    )

    torch.testing.assert_close(out0, out1, rtol=2e-3, atol=2e-3)

