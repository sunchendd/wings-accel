from __future__ import annotations

import pytest
import torch

from vsparse.bmsa.paged_kmeans import CentroidPool, RequestHandleAllocator


def _centroids_and_segments_for_seq_len(seq_len: int) -> tuple[int, int]:
    if seq_len % 16384 != 0:
        raise ValueError("seq_len must be one of 16K/32K/64K/128K")
    k_centroids = seq_len // 16
    num_segments = seq_len // 8192
    return int(k_centroids), int(num_segments)


def _topk_for_seq_len(seq_len: int) -> int:
    return int(min(int(seq_len * 0.2), 8192))


def _make_mixed_seq_lens(B: int) -> list[int]:
    base = [16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024]
    return [base[i % len(base)] for i in range(B)]


def _fill_pool_for_mixed_batch(
    *,
    device: torch.device,
    dtype: torch.dtype,
    B: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    seq_lens: list[int],
) -> tuple[CentroidPool, RequestHandleAllocator, torch.Tensor, torch.Tensor]:
    if len(seq_lens) != B:
        raise ValueError("seq_lens length mismatch")

    K_by_req = [(_centroids_and_segments_for_seq_len(L)[0]) for L in seq_lens]
    Kmax = int(max(K_by_req))

    handles = RequestHandleAllocator(max_requests=B)
    pool = CentroidPool(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        num_heads=num_kv_heads * group_size,
        head_dim=head_dim,
        max_requests=B,
        handle_allocator=handles,
        max_num_centroids=Kmax,
        dtype=dtype,
        device=device,
    )

    layer = 0
    req_handles = [handles.allocate() for _ in range(B)]
    req_handles_t = torch.tensor(req_handles, device=device, dtype=torch.int64)

    for b in range(B):
        seq_len = int(seq_lens[b])
        K_req, num_segments = _centroids_and_segments_for_seq_len(seq_len)
        seg_sizes = [K_req // num_segments] * num_segments
        seg_sizes[-1] += K_req - sum(seg_sizes)

        for kvh in range(num_kv_heads):
            for seg_k in seg_sizes:
                c = torch.randn((seg_k, head_dim), device=device, dtype=dtype)
                cs = torch.ones((seg_k,), device=device, dtype=torch.int32)
                pool.append_centroids(req_handles[b], layer, kvh, c, cs)

    q = torch.randn((B, num_kv_heads * group_size, head_dim), device=device, dtype=dtype)
    k_per_req = torch.tensor([_topk_for_seq_len(L) for L in seq_lens], device=device)
    return pool, handles, req_handles_t, q, k_per_req


@pytest.mark.slow_test
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_batch_topk_production_grid_torch_vs_cutlass_cuda_print() -> None:
    device = torch.device("cuda")
    dtype = torch.float16

    kv_heads_list = [2, 4]
    head_dim = 128
    group_size = 8

    warmup = 20
    iters = 200
    layer = 0

    print(
        "[perf] topk_production_grid "
        "cols=B,Hkv,D,seq_lens,Kmax,num_segments,k_per_req_max,k_eff_max,torch_ms,cutlass_ms"
    )

    base_lens = [16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024]
    batch_cases: list[list[int]] = [
        [base_lens[0]],
        [base_lens[1]],
        [base_lens[2]],
        [base_lens[3]],
        _make_mixed_seq_lens(4),
        _make_mixed_seq_lens(8),
        _make_mixed_seq_lens(16),
    ]

    torch.manual_seed(0)
    for seq_lens in batch_cases:
        B = int(len(seq_lens))

        K_by_req = []
        segs_by_req = []
        for L in seq_lens:
            K_req, nseg = _centroids_and_segments_for_seq_len(int(L))
            K_by_req.append(K_req)
            segs_by_req.append(nseg)

        Kmax = int(max(K_by_req))
        k_per_req = [int(_topk_for_seq_len(int(L))) for L in seq_lens]
        k_per_req_max = int(max(k_per_req))
        k_eff_max = int(max(min(k_per_req[i], K_by_req[i]) for i in range(B)))

        for Hkv in kv_heads_list:
            pool, _, req_handles_t, q, k_per_req_t = _fill_pool_for_mixed_batch(
                device=device,
                dtype=dtype,
                B=B,
                num_kv_heads=Hkv,
                group_size=group_size,
                head_dim=head_dim,
                seq_lens=seq_lens,
            )

            for _ in range(warmup):
                _ = pool.batch_topk_by_handles(
                    req_handles_t,
                    layer,
                    q,
                    k=k_per_req_t,
                    backend="torch",
                    return_format="packed",
                )
                _ = pool.batch_topk_by_handles(
                    req_handles_t,
                    layer,
                    q,
                    k=k_per_req_t,
                    backend="cutlass",
                    return_format="packed",
                )
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iters):
                _ = pool.batch_topk_by_handles(
                    req_handles_t,
                    layer,
                    q,
                    k=k_per_req_t,
                    backend="torch",
                    return_format="packed",
                )
            end.record()
            torch.cuda.synchronize()
            t_torch = start.elapsed_time(end) / iters

            start.record()
            for _ in range(iters):
                _ = pool.batch_topk_by_handles(
                    req_handles_t,
                    layer,
                    q,
                    k=k_per_req_t,
                    backend="cutlass",
                    return_format="packed",
                )
            end.record()
            torch.cuda.synchronize()
            t_cutlass = start.elapsed_time(end) / iters

            print(
                "[perf] topk_production "
                f"B={B} Hkv={Hkv} D={head_dim} "
                f"seq_lens={seq_lens} Kmax={Kmax} "
                f"num_segments={segs_by_req} "
                f"k_per_req_max={k_per_req_max} k_eff_max={k_eff_max} "
                f"torch={t_torch:.3f}ms cutlass={t_cutlass:.3f}ms"
            )
