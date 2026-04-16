import pytest
import torch

from vsparse.bmsa.paged_kmeans import CentroidPool, RequestHandleAllocator, unpack_request_slot


def _make_pool(
    *, max_requests: int = 4, max_num_centroids: int = 32, device: str = "cpu"
) -> tuple[CentroidPool, RequestHandleAllocator]:
    handles = RequestHandleAllocator(max_requests=max_requests)
    pool = CentroidPool(
        num_layers=2,
        num_kv_heads=2,
        num_heads=8,
        head_dim=16,
        max_requests=max_requests,
        handle_allocator=handles,
        max_num_centroids=max_num_centroids,
        dtype=torch.float32,
        device=torch.device(device),
    )
    return pool, handles


def test_append_centroids_writes_correct_row_and_layer():
    """
    验证 `append_centroids()` 的写入位置是正确的：
    - 不同 request 需要落在不同 request_slot
    - 不同 layer/kv_head 需要落在不同的 dense row

    若失败：
    - 说明 centroid pool 的 dense layout 映射（slot/head/layer）出现错位，
      后续 TopK 会在错误的 centroids 上计算，结果会完全不可用。
    """
    pool, handles = _make_pool(max_requests=4, max_num_centroids=32)
    D = pool.head_dim

    req0 = handles.allocate()
    req1 = handles.allocate()

    c0 = torch.full((6, D), 1.0)
    s0 = torch.ones((6,), dtype=torch.int32)
    info0 = pool.append_centroids(
        req0, layer=0, kv_head=0, centroids=c0, cluster_size=s0
    )

    c1 = torch.full((5, D), 2.0)
    s1 = torch.ones((5,), dtype=torch.int32)
    info1 = pool.append_centroids(
        req1, layer=1, kv_head=1, centroids=c1, cluster_size=s1
    )

    row0 = info0.request_slot * pool.num_kv_heads + info0.kv_head
    row1 = info1.request_slot * pool.num_kv_heads + info1.kv_head

    got0 = pool.centroids[0][row0, info0.start:info0.start + info0.length, :]
    got1 = pool.centroids[1][row1, info1.start:info1.start + info1.length, :]

    assert torch.allclose(got0, c0)
    assert torch.allclose(got1, c1)


def test_cluster_id_monotonic_per_request_layer_kvhead():
    """
    验证逻辑 cluster id 的“追加式分配”语义：
    - 同一 (request, layer, kv_head) 上多次 append
      必须得到单调递增的 `base_cluster_id`

    若失败：
    - 增量聚类会产生重复 cluster_id，下游 `ClusterIndex/CPUKVStore` 无法区分 chunk，
      会导致召回错误或覆盖错误。
    """
    pool, handles = _make_pool(max_requests=4, max_num_centroids=64)
    D = pool.head_dim
    req, layer, kvh = handles.allocate(), 0, 1

    a = pool.append_centroids(
        req, layer, kvh, torch.zeros((5, D)), torch.ones((5,), dtype=torch.int32)
    )
    b = pool.append_centroids(
        req, layer, kvh, torch.zeros((3, D)), torch.ones((3,), dtype=torch.int32)
    )
    c = pool.append_centroids(
        req, layer, kvh, torch.zeros((8, D)), torch.ones((8,), dtype=torch.int32)
    )

    assert a.base_cluster_id == 0
    assert b.base_cluster_id == 5
    assert c.base_cluster_id == 8


def test_overflow_raises():
    """
    验证单请求累计簇数超过 `max_num_centroids` 时会显式报错。

    若失败：
    - 可能出现 silent memory overwrite，导致后续请求/kv_head 的 centroids 被污染，
      难以排查，因此必须保持“溢出即失败”的行为。
    """
    pool, handles = _make_pool(max_requests=2, max_num_centroids=10)
    D = pool.head_dim
    h = handles.allocate()
    pool.append_centroids(
        h, 0, 0, torch.zeros((10, D)), torch.ones((10,), dtype=torch.int32)
    )

    with pytest.raises(RuntimeError):
        pool.append_centroids(
            h, 0, 0, torch.zeros((1, D)), torch.ones((1,), dtype=torch.int32)
        )


def test_remove_request_frees_slot_without_corrupting_others():
    """
    验证 request 生命周期管理：
    - remove_request 会释放 slot
    - 新 request 复用该 slot 不会破坏仍在 active 的其它 request 数据

    若失败：
    - 高并发/连续批处理场景下，request 结束与新 request 到来频繁交错，
      一旦 slot 复用时发生数据污染，会导致跨请求错误（致命）。
    """
    pool, handles = _make_pool(max_requests=3, max_num_centroids=16)
    D = pool.head_dim

    ha = handles.allocate()
    hb = handles.allocate()
    info_a = pool.append_centroids(
        ha,
        0,
        0,
        torch.full((4, D), 11.0),
        torch.ones((4,), dtype=torch.int32),
    )
    info_b = pool.append_centroids(
        hb,
        0,
        0,
        torch.full((4, D), 22.0),
        torch.ones((4,), dtype=torch.int32),
    )

    pool.remove_handle(hb)
    handles.free(hb)

    hc = handles.allocate()
    info_c = pool.append_centroids(
        hc,
        0,
        0,
        torch.full((4, D), 33.0),
        torch.ones((4,), dtype=torch.int32),
    )

    row_a = info_a.request_slot * pool.num_kv_heads + info_a.kv_head
    got_a = pool.centroids[0][row_a, info_a.start:info_a.start + info_a.length, :]
    assert torch.allclose(got_a, torch.full((4, D), 11.0))

    assert handles.is_alive(ha)
    assert not handles.is_alive(hb)
    assert handles.is_alive(hc)
    assert unpack_request_slot(hb) == unpack_request_slot(hc)


def test_stale_handle_is_masked_in_topk():
    pool, handles = _make_pool(max_requests=1, max_num_centroids=8)
    D = pool.head_dim
    layer = 0

    h1 = handles.allocate()
    pool.append_centroids(
        h1,
        layer,
        0,
        torch.randn((4, D)),
        torch.ones((4,), dtype=torch.int32),
    )

    pool.remove_handle(h1)
    handles.free(h1)

    h2 = handles.allocate()
    pool.append_centroids(
        h2,
        layer,
        0,
        torch.randn((4, D)),
        torch.ones((4,), dtype=torch.int32),
    )

    q = torch.randn((1, pool.num_heads, D))
    out = pool.batch_topk_by_handles(
        torch.tensor([h1], dtype=torch.int64),
        layer,
        q,
        k=2,
        backend="torch",
        return_format="packed",
    )
    assert int(out.request_slots[0].item()) == -1
