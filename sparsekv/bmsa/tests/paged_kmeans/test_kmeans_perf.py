import time

import torch

from vsparse.bmsa.paged_kmeans import PagedKMeansClusterer
from vsparse.bmsa.paged_kmeans import PagedKMeansConfig
from .utils import *
from .reference_kmeans import segment_k_means


def _require_cuda():
    assert torch.cuda.is_available(), "需要 CUDA 才能运行 Triton 测试"



def test_perf_paged_prefill_only():
    """
    用例：只测 paged 实现的 prefill 性能
    """
    _require_cuda()

    device = "cuda"
    H, D = 8, 128
    L = 4096
    reqs = [ReqSpec(0, L)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=2048,
                                   num_kv_heads=H,
                                   head_dim=D,
                                   device=device,
                                   seed=81)
    simulate_step_and_write(state, BatchSchedule([0], {0: L}), seed=82)

    cfg = PagedKMeansConfig(head_dim=D, block_size=16, num_kv_heads=H, num_heads=28, iters=3, num_layers=28)
    clusterer = PagedKMeansClusterer(cfg)

    def fn():
        clusterer.cluster(state.kv_cache, state.block_table[0], torch.tensor(0, device=device), L=L)

    ms = bench_cuda_ms(fn, warmup=10, iters=50)
    print(f"\npaged prefill: L={L}, H={H}, D={D} => {ms:.3f} ms")
    print("PASS test_perf_paged_prefill_only")


def test_perf_vs_reference_prefill():
    """
    用例：paged vs reference 性能对比（prefill）
    注意：
    - reference 输入需连续 KV，所以这里要先抽取连续 KV（这一步会有额外开销）
    - 为公平比较“算法核心”，我们分别测：
        A) reference 仅 kmeans（不含抽取）
        B) paged 直接 kmeans
      抽取开销单独报告
    """
    _require_cuda()

    device = "cuda"
    H, D = 8, 128
    L = 2048
    reqs = [ReqSpec(0, L)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=1024,
                                   num_kv_heads=H,
                                   head_dim=D,
                                   device=device,
                                   seed=91)
    simulate_step_and_write(state, BatchSchedule([0], {0: L}), seed=92)

    cfg = PagedKMeansConfig(head_dim=D, block_size=16, num_kv_heads=H, num_heads=28, iters=3, num_layers=28)
    clusterer = PagedKMeansClusterer(cfg)

    def fn_paged():
        clusterer.cluster(state.kv_cache, state.block_table[0], torch.tensor(0, device=device), L=L)

    # 预先抽一次，避免把抽取包含进 reference 的 kmeans 计时
    key, value = build_contiguous_kv_from_req(state, 0, L)

    def fn_ref_core():
        segment_k_means(key, value, num_centroids=64, num_iters=3, num_segments=1)

    def fn_extract_only():
        build_contiguous_kv_from_req(state, 0, L)

    ms_paged = bench_cuda_ms(fn_paged, warmup=10, iters=50)
    ms_ref = bench_cuda_ms(fn_ref_core, warmup=10, iters=50)
    ms_extract = bench_cuda_ms(fn_extract_only, warmup=10, iters=50)

    print(f"\npaged core    : {ms_paged:.3f} ms")
    print(f"ref core      : {ms_ref:.3f} ms")
    print(f"extract only  : {ms_extract:.3f} ms (paged->contig cost)")
    print("PASS test_perf_vs_reference_prefill")


def test_perf_decode_window_vs_reference():
    """
    用例：decode 小窗口 L=128 的 paged vs reference 性能对比
    """
    _require_cuda()

    device = "cuda"
    H, D = 8, 128
    prefill = 2048
    Lw = 128
    total = prefill + Lw

    reqs = [ReqSpec(0, total)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=1024,
                                   num_kv_heads=H,
                                   head_dim=D,
                                   device=device,
                                   seed=101)
    simulate_step_and_write(state, BatchSchedule([0], {0: prefill}), seed=102)
    simulate_step_and_write(state, BatchSchedule([0], {0: Lw}), seed=103)

    start_pos = prefill
    cfg = PagedKMeansConfig(head_dim=D, block_size=16, num_kv_heads=H, num_heads=28, iters=3, num_layers=28)
    clusterer = PagedKMeansClusterer(cfg)

    def fn_paged():
        clusterer.cluster(state.kv_cache, state.block_table[0], torch.tensor(start_pos, device=device), L=Lw)

    # reference 预抽取窗口
    key_w, val_w = extract_req_kv_contiguous_from_paged(state, 0, start_pos, Lw)

    def fn_ref():
        segment_k_means(key_w, val_w, num_centroids=64, num_iters=3, num_segments=1)

    def fn_extract_only():
        extract_req_kv_contiguous_from_paged(state, 0, start_pos, Lw)

    ms_paged = bench_cuda_ms(fn_paged, warmup=20, iters=100)
    ms_ref = bench_cuda_ms(fn_ref, warmup=20, iters=100)
    ms_extract = bench_cuda_ms(fn_extract_only, warmup=20, iters=100)

    print(f"\npaged decode window core : {ms_paged:.3f} ms")
    print(f"ref   decode window core : {ms_ref:.3f} ms")
    print(f"extract window only      : {ms_extract:.3f} ms")
    print("PASS test_perf_decode_window_vs_reference")


if __name__ == "__main__":
    test_perf_paged_prefill_only()
    test_perf_vs_reference_prefill()
    test_perf_decode_window_vs_reference()
