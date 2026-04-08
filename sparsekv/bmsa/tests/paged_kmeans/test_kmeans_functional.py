import torch

from vsparse.bmsa.paged_kmeans import PagedKMeansConfig
from vsparse.bmsa.paged_kmeans import PagedKMeansClusterer

from .utils import *


def _require_cuda():
    assert torch.cuda.is_available(), "需要 CUDA 才能运行 Triton 测试"


def test_paged_prefill_single_req_full_cluster():
    """
    用例：单请求 Prefill 全量聚类（L=seq_len）
    覆盖：paged block_table 非连续、init centroids、full assign/update、csr build
    """
    _require_cuda()

    device = "cuda"
    reqs = [ReqSpec(0, 2048)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=512,
                                   num_kv_heads=8,
                                   head_dim=128,
                                   device=device,
                                   seed=1)

    # 模拟 prefill：一次写满
    schedule = BatchSchedule([0], {0: 2048})
    simulate_step_and_write(state, schedule, seed=2)

    cfg = PagedKMeansConfig(
        head_dim=128, block_size=16, num_kv_heads=8, num_layers=28,
        iters=3, num_heads=28,
    )
    clusterer = PagedKMeansClusterer(cfg)

    centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
        kv_cache=state.kv_cache,
        block_table_1d=state.block_table[0],
        start_pos=torch.tensor(0, device=device),
        L=2048,
    )

    assert centroids.shape == (8, 64, 128)
    assert sum_v.shape == (8, 64, 128)
    assert assign.shape == (8, 2048)
    assert cnt.shape == (8, 64)
    assert offsets.shape == (8, 65)
    assert perm.shape == (8, 2048)
    assert int(cnt.sum().item()) == 8 * 2048
    assert int(offsets[:, -1].min().item()) == 2048
    print("PASS test_paged_prefill_single_req_full_cluster")


def test_paged_prefill_multi_req_individual_cluster():
    """
    用例：多请求 Prefill，各自独立做聚类（分别调用 cluster）
    覆盖：不同 req 的 block_table 不同且物理块非连续
    """
    _require_cuda()

    device = "cuda"
    reqs = [ReqSpec(0, 1024), ReqSpec(1, 1536), ReqSpec(2, 768)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=512,
                                   num_kv_heads=8,
                                   head_dim=128,
                                   device=device,
                                   seed=11)

    schedule = BatchSchedule([0, 1, 2], {0: 1024, 1: 1536, 2: 768})
    simulate_step_and_write(state, schedule, seed=12)

    cfg = PagedKMeansConfig(head_dim=128, block_size=16, num_kv_heads=8, num_heads=28, iters=3, num_layers=28)
    clusterer = PagedKMeansClusterer(cfg)

    for rid, L in [(0, 1024), (1, 1536), (2, 768)]:
        centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
            state.kv_cache, state.block_table[rid], torch.tensor(0, device=device), L=L
        )
        assert assign.shape == (8, L)
        assert int(offsets[:, -1].min().item()) == L
    print("PASS test_paged_prefill_multi_req_individual_cluster")


def test_paged_prefill_non_power_of_two_num_centroids():
    """
    回归用例：允许 num_centroids(K) 不是 2 的幂（但通常仍对齐到 32）。

    背景：
    - CRSA 集成时，`get_num_segments_and_centroids()` 会保证：
        1) K 是 32 的倍数（兼容 assign kernel 的 tile 约束，如 assign_block_k=32）
        2) K % num_segments == 0（分段训练/expanded pool 的语义需要）
      但不会保证 K 是 2 的幂。
    - Triton 的 `tl.arange(0, X)` 对 X 有 “必须是 2 的幂” 的约束；如果某个 kernel
      直接写 `tl.arange(0, K)`，则在 K=96/1536 这类场景会在 JIT 编译时报错。

    这个用例用 K=96（32 对齐但非 2 的幂）验证聚类与 CSR 构建路径可正常运行。
    """
    _require_cuda()

    device = "cuda"
    reqs = [ReqSpec(0, 512)]
    state = init_paged_cache_state(
        reqs, block_size=16, num_blocks=128, num_kv_heads=8, head_dim=128,
        device=device, seed=111,
    )
    simulate_step_and_write(state, BatchSchedule([0], {0: 512}), seed=112)

    cfg = PagedKMeansConfig(head_dim=128, block_size=16, num_kv_heads=8, num_heads=28, iters=3, num_layers=28)
    clusterer = PagedKMeansClusterer(cfg)

    K = 96  # 32 的倍数，但不是 2 的幂
    centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
        state.kv_cache, state.block_table[0], torch.tensor(0, device=device), L=512, num_centroids=K
    )
    assert centroids.shape == (8, K, 128)
    assert sum_v.shape == (8, K, 128)
    assert assign.shape == (8, 512)
    assert cnt.shape == (8, K)
    assert offsets.shape == (8, K + 1)
    assert perm.shape == (8, 512)
    assert int(offsets[:, -1].min().item()) == 512
    print("PASS test_paged_prefill_non_power_of_two_num_centroids")


def test_paged_decode_incremental_cluster_only_new_tokens():
    """
    用例：Decode 增量聚类：prefill 之后，每 step 新增 1 token，只聚类新增窗口 L=1
    覆盖：start_pos 非 0、L 很小、block 边界跨越
    """
    _require_cuda()

    device = "cuda"
    reqs = [ReqSpec(0, 300)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=128,
                                   num_kv_heads=4,
                                   head_dim=64,
                                   device=device,
                                   seed=21)

    # prefill 先写 255（接近 block 边界）
    simulate_step_and_write(state, BatchSchedule([0], {0: 255}), seed=22)

    cfg = PagedKMeansConfig(head_dim=64, block_size=16, num_kv_heads=4, num_heads=28, iters=3, num_layers=28)
    clusterer = PagedKMeansClusterer(cfg)

    # 对新增 token 连续做 10 次增量聚类
    for _ in range(10):
        prev = int(state.cur_lens[0].item())
        simulate_step_and_write(state, BatchSchedule([0], {0: 1}), seed=100 + prev)
        centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
            state.kv_cache, state.block_table[0], torch.tensor(prev, device=device), L=1
        )
        assert assign.shape == (4, 1)
        assert int(offsets[:, -1].min().item()) == 1
    print("PASS test_paged_decode_incremental_cluster_only_new_tokens")


def test_paged_segmented_training_then_full_final_iter():
    """
    用例：num_segments>1 的分段训练 + 最后一轮 full
    覆盖：segmented kernels 路径、L_seg=L//S 的截断逻辑
    """
    _require_cuda()

    device = "cuda"
    reqs = [ReqSpec(0, 2049)]  # 故意让 L 不能被 num_segments 整除
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=512,
                                   num_kv_heads=8,
                                   head_dim=128,
                                   device=device,
                                   seed=31)
    simulate_step_and_write(state, BatchSchedule([0], {0: 2049}), seed=32)

    cfg = PagedKMeansConfig(
        head_dim=128, block_size=16, num_kv_heads=8,
        num_heads=28, iters=3, num_layers=28
    )
    clusterer = PagedKMeansClusterer(cfg)
    centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
        state.kv_cache, state.block_table[0], torch.tensor(0, device=device),
        L=2049, num_segments=4, num_centroids=128,
    )
    assert assign.shape == (8, 2049)
    assert int(offsets[:, -1].min().item()) == 2049
    print("PASS test_paged_segmented_training_then_full_final_iter")


def test_paged_dot_metric_smoke():
    """
    用例：metric=DOT 冒烟测试
    """
    _require_cuda()

    device = "cuda"
    reqs = [ReqSpec(0, 512)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=128,
                                   num_kv_heads=8,
                                   head_dim=128,
                                   device=device,
                                   seed=41)
    simulate_step_and_write(state, BatchSchedule([0], {0: 512}), seed=42)

    cfg = PagedKMeansConfig(
        head_dim=128, block_size=16, num_kv_heads=8, num_heads=28, iters=2, num_layers=28
    )
    clusterer = PagedKMeansClusterer(cfg)
    centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
        state.kv_cache, state.block_table[0], torch.tensor(0, device=device), L=512
    )
    assert assign.shape == (8, 512)
    print("PASS test_paged_dot_metric_smoke")


def test_continuous_batch_and_subset_extraction_correctness():
    """
    用例：模拟连续批处理（多请求 token 拼接），本 step 只调度子集请求；
          验证：
          - slot_mapping 正确（写到的物理 slot 与 block_table 一致）
          - 能从 paged cache 抽取“被调度子集请求”的连续 KV 段进行聚类
    """
    _require_cuda()

    device = "cuda"
    reqs = [ReqSpec(0, 800), ReqSpec(1, 900), ReqSpec(2, 700), ReqSpec(3, 1000)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=512,
                                   num_kv_heads=8,
                                   head_dim=128,
                                   device=device,
                                   seed=51)

    # step0：prefill 全部写一部分
    simulate_step_and_write(state, BatchSchedule([0, 1, 2, 3], {0: 400, 1: 450, 2: 350, 3: 500}), seed=52)

    # step1：只调度 req 1 和 req 3，模拟连续批处理 token 列表
    keys, values, slot_mapping, req_ranges = simulate_step_and_write(
        state, BatchSchedule([1, 3], {1: 40, 3: 10}), seed=53
    )

    # 校验 slot_mapping 与实际写入位置一致：抽样检查
    bs = state.block_size
    for _ in range(20):
        i = random.randrange(0, slot_mapping.numel())
        slot = int(slot_mapping[i].item())
        phys = slot // bs
        off = slot % bs
        # 对比写入的 kv_cache 与 keys/values
        assert torch.allclose(state.kv_cache[0, phys, off], keys[i], atol=0, rtol=0)
        assert torch.allclose(state.kv_cache[1, phys, off], values[i], atol=0, rtol=0)

    # 对 req1 的新增段做聚类（仅聚类新增段）
    rid = 1
    start_in_step, new_len = req_ranges[rid]
    # 注意：新增段在该 req 内的起点 = step1 开始前的 cur_len - new_len
    start_pos = int(state.cur_lens[rid].item()) - new_len

    cfg = PagedKMeansConfig(head_dim=128, block_size=16, num_kv_heads=8, num_heads=28, iters=3, num_layers=28)
    clusterer = PagedKMeansClusterer(cfg)
    centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
        state.kv_cache, state.block_table[rid], torch.tensor(start_pos, device=device), L=new_len
    )
    assert assign.shape == (8, new_len)
    assert int(offsets[:, -1].min().item()) == new_len
    print("PASS test_continuous_batch_and_subset_extraction_correctness")


if __name__ == "__main__":
    test_paged_prefill_single_req_full_cluster()
    test_paged_prefill_multi_req_individual_cluster()
    test_paged_prefill_non_power_of_two_num_centroids()
    test_paged_decode_incremental_cluster_only_new_tokens()
    test_paged_segmented_training_then_full_final_iter()
    test_paged_dot_metric_smoke()
    test_continuous_batch_and_subset_extraction_correctness()
