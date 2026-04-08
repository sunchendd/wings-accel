import torch

from vsparse.bmsa.paged_kmeans import (
    CentroidPool,
    RequestHandleAllocator,
    pack_request_handle,
)


def _make_pool(
    *,
    num_layers=1,
    num_kv_heads=2,
    num_heads=8,
    D=8,
    max_requests=8,
    max_num_centroids=16,
    device="cpu",
):
    handles = RequestHandleAllocator(max_requests=max_requests)
    pool = CentroidPool(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        head_dim=D,
        max_requests=max_requests,
        handle_allocator=handles,
        max_num_centroids=max_num_centroids,
        dtype=torch.float32,
        device=torch.device(device),
    )
    return pool, handles


def test_batch_topk_gqa_group_mapping_and_empty_cluster_mask():
    """
    验证 GQA 下 query->kv_head 的 group 映射语义，以及“空簇 mask”语义。

    配置：
    - num_heads=8, num_kv_heads=2 => group_size=4
    - kv_head0 使用 q[0:4]，kv_head1 使用 q[4:8]

    预期：
    - kv_head0 的 group0 指向某个维度时，应选到对应 centroid
    - cluster_size==0 的 centroid 必须永远不会被 TopK 选中

    若失败：
    - 说明 GQA 分组或 mask 语义与参考实现不一致，会导致 decode TopK 召回簇错误。
    """
    pool, handles = _make_pool(
        num_kv_heads=2, num_heads=8, D=8, max_requests=4, max_num_centroids=8
    )
    layer = 0
    req = handles.allocate()

    # kv_head0 的 centroids: c0 强匹配 [1,0,0..]
    c0 = torch.zeros((4, 8))
    c0[0, 0] = 10.0  # cluster 0
    c0[1, 1] = 10.0  # cluster 1
    c0[2, 2] = 10.0  # cluster 2
    c0[3, 3] = 10.0  # cluster 3
    cs0 = torch.tensor([1, 1, 0, 1], dtype=torch.int32)  # cluster2 空簇应被 mask

    # kv_head1 的 centroids: 匹配另外维度
    c1 = torch.zeros((4, 8))
    c1[0, 4] = 10.0
    c1[1, 5] = 10.0
    c1[2, 6] = 10.0
    c1[3, 7] = 10.0
    cs1 = torch.tensor([1, 0, 1, 1], dtype=torch.int32)  # cluster1 空簇

    pool.append_centroids(req, layer, 0, c0, cs0)
    pool.append_centroids(req, layer, 1, c1, cs1)

    # 构造 queries: [B=1, num_heads=8, D=8]
    # group0(0:4) 指向维度0 => 应选 cluster0（且不会选空簇 cluster2）
    # group1(4:8) 指向维度6 => 应选 cluster2（kv_head1 的 cluster2 非空）
    q = torch.zeros((1, 8, 8))
    q[0, 0:4, 0] = 1.0
    q[0, 4:8, 6] = 1.0

    out = pool.batch_topk_by_handles(
        request_handles=torch.tensor([req], dtype=torch.int64),
        layer=layer,
        queries=q,
        k=2,
        backend="torch",
        return_format="packed",
    )
    got = out.logical_cluster_ids[0]
    assert got.shape == (2, 2)
    assert int(got[0, 0].item()) == 0
    assert int(got[1, 0].item()) == 2


@torch.no_grad()
def test_append_centroids_zeroes_unwritten_tail_by_default():
    pool, handles = _make_pool(
        num_kv_heads=2, num_heads=8, D=8, max_requests=4, max_num_centroids=8
    )
    layer = 0
    req = handles.allocate()

    c0 = torch.zeros((4, 8))
    c0[0, 0] = 10.0
    c0[1, 1] = 10.0
    c0[2, 2] = 10.0
    c0[3, 3] = 10.0
    cs0 = torch.tensor([1, 1, 0, 1], dtype=torch.int32)

    c1 = torch.zeros((4, 8))
    c1[0, 4] = 10.0
    c1[1, 5] = 10.0
    c1[2, 6] = 10.0
    c1[3, 7] = 10.0
    cs1 = torch.tensor([1, 0, 1, 1], dtype=torch.int32)

    info0 = pool.append_centroids(req, layer, 0, c0, cs0)
    info1 = pool.append_centroids(req, layer, 1, c1, cs1)

    row0 = info0.request_slot * pool.num_kv_heads + info0.kv_head
    row1 = info1.request_slot * pool.num_kv_heads + info1.kv_head
    tail0 = pool.centroids[layer][row0, 4:, :]
    tail1 = pool.centroids[layer][row1, 4:, :]
    assert torch.equal(tail0, torch.zeros_like(tail0))
    assert torch.equal(tail1, torch.zeros_like(tail1))


def test_batch_topk_across_multiple_segments_global_cluster_id():
    """
    验证“增量追加簇”时的逻辑 cluster id 拼接语义。

    预期：
    - 第二次 append 的 `base_cluster_id` 紧接第一次 append 的长度
    - TopK 返回的 `logical_cluster_id` 必须是“全局 id”，而不是 local 0..K-1

    若失败：
    - 下游 ClusterIndex 无法路由到正确 chunk，会召回错误 token。
    """
    pool, handles = _make_pool(
        num_kv_heads=1, num_heads=4, D=4, max_requests=4, max_num_centroids=16
    )
    layer = 0
    req = handles.allocate()

    # 追加两段 segment，验证 global_cluster_id 拼接正确
    cA = torch.eye(4)  # K=4
    csA = torch.ones((4,), dtype=torch.int32)
    segA = pool.append_centroids(req, layer, 0, cA, csA)
    assert segA.base_cluster_id == 0

    cB = torch.eye(4) * 2.0
    csB = torch.ones((4,), dtype=torch.int32)
    segB = pool.append_centroids(req, layer, 0, cB, csB)
    assert segB.base_cluster_id == 4  # 全局 id 紧接前一段

    # query 指向第二段的第 1 个 centroid（local=1 => global=5）
    q = torch.zeros((1, 4, 4))
    q[0, :, 1] = 10.0  # group_size=4 -> mean 不影响
    out = pool.batch_topk_by_handles(
        torch.tensor([req], dtype=torch.int64),
        layer,
        q,
        k=1,
        backend="torch",
        return_format="packed",
    )
    got = int(out.logical_cluster_ids[0, 0, 0].item())
    assert got == 5


@torch.no_grad()
def test_cutlass_matches_torch_packed_on_cuda() -> None:
    """
    质量/一致性测试：
    - cutlass backend 的 packed 输出必须与 torch backend 完全一致

    若失败：
    - 说明 cutlass kernel 输出/后处理与参考语义不一致，
      在实际 decode 中会出现不确定的检索差异。
    """
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    dtype = torch.float16

    handles = RequestHandleAllocator(max_requests=4)
    pool = CentroidPool(
        num_layers=1,
        num_kv_heads=2,
        num_heads=8,
        head_dim=8,
        max_requests=4,
        handle_allocator=handles,
        max_num_centroids=8,
        dtype=dtype,
        device=device,
    )

    req = handles.allocate()
    layer = 0

    c0 = torch.zeros((4, 8), device=device, dtype=dtype)
    c0[0, 0] = 50.0
    c0[1, 1] = 40.0
    c0[2, 2] = 30.0
    c0[3, 3] = 20.0
    cs0 = torch.tensor([1, 1, 0, 1], device=device, dtype=torch.int32)

    c1 = torch.zeros((4, 8), device=device, dtype=dtype)
    c1[0, 4] = 10.0
    c1[1, 5] = 20.0
    c1[2, 6] = 60.0
    c1[3, 7] = 30.0
    cs1 = torch.tensor([1, 0, 1, 1], device=device, dtype=torch.int32)

    pool.append_centroids(req, layer, 0, c0, cs0)
    pool.append_centroids(req, layer, 1, c1, cs1)

    q = torch.zeros((1, 8, 8), device=device, dtype=dtype)
    q[0, 0:4, 0] = 1.0
    q[0, 4:8, 6] = 1.0

    req_handles = torch.tensor([req], device=device, dtype=torch.int64)
    out_torch = pool.batch_topk_by_handles(
        req_handles, layer, q, k=2, backend="torch", return_format="packed"
    )
    out_cutlass = pool.batch_topk_by_handles(
        req_handles, layer, q, k=2, backend="cutlass", return_format="packed"
    )

    assert torch.equal(out_torch.logical_cluster_ids, out_cutlass.logical_cluster_ids)
    assert torch.equal(out_torch.k_per_req, out_cutlass.k_per_req)
    assert torch.equal(out_torch.valid_k, out_cutlass.valid_k)


if __name__ == "__main__":
    test_batch_topk_gqa_group_mapping_and_empty_cluster_mask()
    test_batch_topk_across_multiple_segments_global_cluster_id()
    test_cutlass_matches_torch_packed_on_cuda()


def test_batch_topk_k_tensor_per_request_and_unregistered_request_dict_format():
    """
    验证两种边界语义：
    1) `k` 允许是 shape=[B] 的张量，每个 request 有不同的 TopK 需求
    2) 对未注册的 request_id，dict 格式输出应该返回空 tensor（而不是报错）

    若失败：
    - 说明 batch API 的边界约定不稳定，容易在 scheduler 的动态 active set 下踩坑。
    """
    pool, handles = _make_pool(
        num_kv_heads=1, num_heads=4, D=4, max_requests=4, max_num_centroids=8
    )
    layer = 0

    req0 = handles.allocate()
    _ = handles.allocate()
    c0 = torch.eye(4)
    cs0 = torch.ones((4,), dtype=torch.int32)
    pool.append_centroids(req0, layer, 0, c0, cs0)

    q = torch.zeros((2, 4, 4))
    q[:, :, 0] = 10.0
    k = torch.tensor([1, 3], dtype=torch.int64)

    unknown_handle = pack_request_handle(slot=3, generation=0)
    out = pool.batch_topk_by_handles(
        request_handles=torch.tensor([req0, unknown_handle], dtype=torch.int64),
        layer=layer,
        queries=q,
        k=k,
        backend="torch",
        return_format="dict",
        request_ids=[str(req0), "9999"],
    )
    assert int(out[str(req0)][0].numel()) == 1
    assert int(out["9999"][0].numel()) == 0
