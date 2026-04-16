import torch
import torch.nn.functional as F

from vsparse.bmsa.paged_kmeans import PagedKMeansClusterer
from vsparse.bmsa.paged_kmeans import PagedKMeansConfig
from .utils import *
from .reference_kmeans import segment_k_means


def _require_cuda():
    assert torch.cuda.is_available(), "需要 CUDA 才能运行 Triton 测试"



def test_quality_prefill_vs_reference_contiguous():
    """
    用例：Prefill 全量聚类质量对齐（paged vs reference 连续）
    要点：
    - 数据只写一份到 paged KVCache
    - 再从 paged 抽取连续 KV 给 reference（保证数据一致）
    - 对比：centroid cosine、value_sum mae、assign match
    """
    _require_cuda()

    device = "cuda"
    H, D = 8, 128
    L = 2048
    reqs = [ReqSpec(0, L)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=512,
                                   num_kv_heads=H,
                                   head_dim=D,
                                   device=device,
                                   seed=61)
    simulate_step_and_write(state, BatchSchedule([0], {0: L}), seed=62)

    # reference 连续输入
    key, value = build_contiguous_kv_from_req(state, 0, L)

    K = 64

    # reference 期望输入：[num_groups, num_tokens, head_dim]，其中 num_groups = batch_size(=1)*num_heads
    ref_centroids, ref_vsum, ref_clusters, ref_cnt = segment_k_means(
        key, value, num_centroids=K, num_iters=3, num_segments=1
    )

    cfg = PagedKMeansConfig(
        head_dim=D, block_size=16, num_kv_heads=H, num_heads=28, iters=3, num_layers=28
    )

    clusterer = PagedKMeansClusterer(cfg)
    centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
        state.kv_cache, state.block_table[0], torch.tensor(0, device=device), L=L
    )

    # reference 返回 clusters/cluster_size，不直接给 assign；因此这里用一个对齐方式：
    # - 用 reference 的 max_idx 需要从 reference 代码返回；你给的 segment_k_means 里 max_idx 是中间变量
    # 为了严格“不改参考代码”，这里不强行对比 assign；
    # 我们主要对比 centroids 与 value_sum（这两者能稳定反映训练结果），以及 cnt 与 ref_cnt(=cluster_size)
    cos = centroid_cosine_mean(centroids, ref_centroids)
    vs_mae = mean_abs_err(sum_v, ref_vsum)
    # cnt: paged cnt=[H,K]，reference cluster_size=[H,K]（名字不同，但含义一致）
    cnt_mae = mean_abs_err(cnt, ref_cnt)

    print(f"\ncentroid cosine mean = {cos:.6f}")
    print(f"value_sum MAE      = {vs_mae:.6e}")
    print(f"cluster_size MAE   = {cnt_mae:.6e}")

    # 阈值：按你的目标“效果一致水平线”，这里设得较严格；若你环境/数值路径略不同可调小
    assert cos > 0.95, "centroids 余弦相似度过低"
    assert vs_mae < 2e-2, "value_sum 误差过大"
    assert cnt_mae < 1e-3, "cluster_size 不一致"
    print("PASS test_quality_prefill_vs_reference_contiguous")


def test_quality_decode_new_tokens_vs_reference_window():
    """
    用例：Decode 增量聚类质量对齐
    - 先 prefill 到某长度
    - 再新增 decode_len token
    - 只对新增窗口做聚类（paged vs reference 对同一窗口抽取的连续 KV）
    """
    _require_cuda()

    device = "cuda"
    H, D = 8, 128
    prefill = 1024
    decode_len = 128
    total = prefill + decode_len

    reqs = [ReqSpec(0, total)]
    state = init_paged_cache_state(reqs,
                                   block_size=16,
                                   num_blocks=512,
                                   num_kv_heads=H,
                                   head_dim=D,
                                   device=device,
                                   seed=71)

    simulate_step_and_write(state, BatchSchedule([0], {0: prefill}), seed=72)
    simulate_step_and_write(state, BatchSchedule([0], {0: decode_len}), seed=73)

    start_pos = prefill
    # reference：抽窗口
    key_w, val_w = extract_req_kv_contiguous_from_paged(state, 0, start_pos, decode_len)
    ref_centroids, ref_vsum, _, ref_cnt = segment_k_means(
        key_w, val_w, num_centroids=64, num_iters=3, num_segments=1
    )

    cfg = PagedKMeansConfig(head_dim=D, block_size=16, num_kv_heads=H, num_heads=28, iters=3, num_layers=28)
    clusterer = PagedKMeansClusterer(cfg)
    centroids, sum_v, assign, cnt, offsets, perm = clusterer.cluster(
        state.kv_cache, state.block_table[0], torch.tensor(start_pos, device=device), L=decode_len
    )

    cos = centroid_cosine_mean(centroids, ref_centroids)
    vs_mae = mean_abs_err(sum_v, ref_vsum)
    cnt_mae = mean_abs_err(cnt, ref_cnt)

    print(f"\ncentroid cosine mean = {cos:.6f}")
    print(f"value_sum MAE      = {vs_mae:.6e}")
    print(f"cluster_size MAE   = {cnt_mae:.6e}")

    assert cos > 0.999
    assert vs_mae < 2e-2
    assert cnt_mae < 1e-3
    print("PASS test_quality_decode_new_tokens_vs_reference_window")


if __name__ == "__main__":
    test_quality_prefill_vs_reference_contiguous()
    test_quality_decode_new_tokens_vs_reference_window()
