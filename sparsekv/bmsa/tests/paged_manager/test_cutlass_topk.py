from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vsparse.bmsa.paged_kmeans import (
    CentroidPool,
    RequestHandleAllocator,
    unpack_request_slot,
)


@torch.no_grad()
def _reference_retroinfer_topk_packed(
    pool: CentroidPool,
    request_handles: list[int],
    layer: int,
    queries: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    一个“参考语义”的 TopK 实现，用来对齐 `reference_retroinfer` 的思路：
    - logits = (Q @ C^T) * (1/sqrt(D))
    - 在 centroid 维做 softmax
    - 对 GQA 的 group 维度求和得到 dist
    - dist 上做 TopK(largest=True)

    注意：这里直接读取 `CentroidPool` 内部存储，并按其 dense row layout 取数，
    用于语义一致性验证，而不是性能基准。
    """
    slots = torch.tensor(
        [unpack_request_slot(h) for h in request_handles],
        device=queries.device,
        dtype=torch.int64,
    )

    B, H, D = queries.shape
    Hkv = pool.num_kv_heads
    G = pool.group_size
    Kmax = pool.max_num_centroids

    rows = (
        slots[:, None] * Hkv + torch.arange(Hkv, device=queries.device)[None, :]
    ).reshape(-1)
    invalid_bg = (slots[:, None] < 0).expand(B, Hkv).reshape(-1)
    safe_rows = rows.clamp_min(0)

    C = pool.centroids[layer][safe_rows]  # [BG,Kmax,D]
    CS = pool.cluster_size[layer][safe_rows]  # [BG,Kmax]
    VK = pool.valid_k[layer][safe_rows]  # [BG]

    A = queries.view(B, Hkv, G, D).reshape(-1, G, D)
    logits = torch.einsum(
        "bgd,bkd->bgk", A.to(torch.float32), C.to(torch.float32)
    ) * float(pool._rsqrt_dim)
    probs = F.softmax(logits, dim=-1)
    dist = probs.sum(dim=1)  # [BG,Kmax]

    k_idx = torch.arange(Kmax, device=queries.device, dtype=torch.int32)[None, :]
    dtype_min = torch.finfo(dist.dtype).min
    dist.masked_fill_(CS == 0, dtype_min)
    dist.masked_fill_(k_idx >= VK[:, None], dtype_min)
    dist.masked_fill_(invalid_bg[:, None], dtype_min)

    topi = torch.topk(
        dist, k=min(int(k), Kmax), dim=-1, largest=True, sorted=True
    ).indices  # [BG,k]
    base_cluster = pool.next_cluster_id[layer][safe_rows] - VK.to(torch.int64)
    logical = (base_cluster[:, None] + topi.to(torch.int64)).view(B, Hkv, -1)
    return logical


@torch.no_grad()
def test_torch_topk_matches_reference_semantics_cpu() -> None:
    """
    质量/语义一致性测试（CPU）：
    - torch backend 的输出应与参考语义完全一致

    若失败：
    - 说明当前 TopK 的核心语义（softmax+sum+topk）发生漂移，
      decode 阶段检索簇的含义会和原型不同。
    """
    handles = RequestHandleAllocator(max_requests=4)
    pool = CentroidPool(
        num_layers=1,
        num_kv_heads=2,
        num_heads=8,
        head_dim=8,
        max_requests=4,
        handle_allocator=handles,
        max_num_centroids=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    req = handles.allocate()
    layer = 0

    c0 = torch.zeros((4, 8))
    c0[0, 0] = 50.0
    c0[1, 1] = 40.0
    c0[2, 2] = 30.0
    c0[3, 3] = 20.0
    cs0 = torch.tensor([1, 1, 0, 1], dtype=torch.int32)

    c1 = torch.zeros((4, 8))
    c1[0, 4] = 10.0
    c1[1, 5] = 20.0
    c1[2, 6] = 60.0
    c1[3, 7] = 30.0
    cs1 = torch.tensor([1, 0, 1, 1], dtype=torch.int32)

    pool.append_centroids(req, layer, 0, c0, cs0)
    pool.append_centroids(req, layer, 1, c1, cs1)

    q = torch.zeros((1, 8, 8))
    q[0, 0:4, 0] = 1.0
    q[0, 4:8, 6] = 1.0

    req_handles = torch.tensor([req], dtype=torch.int64)
    out = pool.batch_topk_by_handles(req_handles, layer, q, k=2, backend="torch", return_format="packed")
    ref = _reference_retroinfer_topk_packed(pool, [req], layer, q, k=2)
    assert torch.equal(out.logical_cluster_ids, ref)


@torch.no_grad()
def test_torch_topk_matches_reference_randomized_cpu() -> None:
    """
    随机化质量测试（CPU）：
    - 使用固定 seed 构造随机 centroids/queries
    - torch backend 输出必须与参考语义一致

    该测试覆盖：
    - `valid_k` 截断（只 append 部分簇）
    - `cluster_size==0` 的 mask
    - k 小于/大于 valid_k 的裁剪行为（通过 k 取小值触发）
    """
    torch.manual_seed(0)
    handles = RequestHandleAllocator(max_requests=4)
    pool = CentroidPool(
        num_layers=1,
        num_kv_heads=2,
        num_heads=8,
        head_dim=16,
        max_requests=4,
        handle_allocator=handles,
        max_num_centroids=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    req = handles.allocate()
    layer = 0
    K_append = 20
    D = pool.head_dim

    for h in range(pool.num_kv_heads):
        c = torch.randn((K_append, D), dtype=torch.float32)
        cs = torch.ones((K_append,), dtype=torch.int32)
        cs[::7] = 0
        pool.append_centroids(req, layer, h, c, cs)

    q = torch.randn((1, pool.num_heads, D), dtype=torch.float32)
    req_handles = torch.tensor([req], dtype=torch.int64)
    out = pool.batch_topk_by_handles(req_handles, layer, q, k=8, backend="torch", return_format="packed")
    ref = _reference_retroinfer_topk_packed(pool, [req], layer, q, k=8)
    assert torch.equal(out.logical_cluster_ids, ref)


@torch.no_grad()
def test_cutlass_topk_matches_torch_cuda() -> None:
    """
    质量/一致性测试（CUDA）：
    - cutlass backend 的输出必须与 torch backend 完全一致

    若失败：
    - 说明 cutlass kernel 或其后处理逻辑与 torch/reference 不一致。
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
    out_torch = pool.batch_topk_by_handles(req_handles, layer, q, k=2, backend="torch", return_format="packed")
    out_cutlass = pool.batch_topk_by_handles(req_handles, layer, q, k=2, backend="cutlass", return_format="packed")
    assert torch.equal(out_torch.logical_cluster_ids, out_cutlass.logical_cluster_ids)


@torch.no_grad()
def test_cutlass_topk_by_slots_matches_by_request_ids_cuda() -> None:
    """
    测试点：`batch_topk_by_slots(..., backend='cutlass')` 语义与兼容模式一致。

    背景：
    - 兼容模式 `batch_topk(..., backend='cutlass')` 内部需要 request_id -> slot 的
      Python dict 映射（decode 热路径会产生不必要 CPU 开销）
    - 优化路径由上层直接传 `request_slots`，应当产出完全一致的 packed 输出

    预期：
    - 两种入口的 packed 输出逐元素一致
    """
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    dtype = torch.float16

    handles = RequestHandleAllocator(max_requests=8)
    pool = CentroidPool(
        num_layers=1,
        num_kv_heads=2,
        num_heads=8,
        head_dim=16,
        max_requests=8,
        handle_allocator=handles,
        max_num_centroids=64,
        dtype=dtype,
        device=device,
    )

    req = handles.allocate()
    layer = 0
    K = 32
    D = pool.head_dim

    for h in range(pool.num_kv_heads):
        c = torch.randn((K, D), device=device, dtype=dtype)
        cs = torch.ones((K,), device=device, dtype=torch.int32)
        pool.append_centroids(req, layer, h, c, cs)

    q = torch.randn((1, pool.num_heads, D), device=device, dtype=dtype)
    request_handles = torch.tensor([req], device=device, dtype=torch.int64)
    request_slots = torch.tensor([unpack_request_slot(req)], device=device, dtype=torch.int64)

    out_by_handles = pool.batch_topk_by_handles(
        request_handles, layer, q, k=8, backend="cutlass", return_format="packed"
    )
    out_by_slots = pool.batch_topk_by_slots(
        request_slots, layer, q, k=8, backend="cutlass", return_format="packed"
    )
    assert torch.equal(out_by_handles.logical_cluster_ids, out_by_slots.logical_cluster_ids)


def test_topk_semantics_softmax_sum_caps_single_head_dominance():
    """
    验证 TopK 的“softmax + group sum”语义的关键特性：
    - 每个 head 在 softmax 后对某个 centroid 的贡献最多为 1
    - 因此“单个 head 极强偏好某个 centroid”不会像 dot-sum 那样无限放大，
      多数 head 的偏好可以覆盖单 head 的极端值

    该特性是 `reference_retroinfer` 里 dist 定义（sum(softmax_o, dim=group)）带来的。

    若失败：
    - 说明实现不再是 reference 语义（可能变成 dot-product 直接 topk），
      会影响检索质量与稀疏预算的可控性。
    """
    handles = RequestHandleAllocator(max_requests=2)
    pool = CentroidPool(
        num_layers=1,
        num_kv_heads=1,
        num_heads=4,
        head_dim=1,
        max_requests=2,
        handle_allocator=handles,
        max_num_centroids=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    req = handles.allocate()
    layer = 0

    centroids = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    cs = torch.ones((2,), dtype=torch.int32)
    pool.append_centroids(req, layer, 0, centroids, cs)

    q = torch.empty((1, 4, 1), dtype=torch.float32)
    q[0, 0, 0] = 100.0
    q[0, 1:, 0] = -1.0

    out = pool.batch_topk_by_handles(
        request_handles=torch.tensor([req], dtype=torch.int64),
        layer=layer,
        queries=q,
        k=1,
        backend="torch",
        return_format="packed",
    )
    assert int(out.logical_cluster_ids[0, 0, 0].item()) == 1


@pytest.mark.slow_test
@torch.no_grad()
def test_topk_perf_torch_vs_cutlass_cuda_smoke() -> None:
    """
    性能 smoke test（仅统计并打印，不做速度断言）：
    - 对比 torch backend 与 cutlass backend 的 batch_topk 用时

    该测试主要用于：
    - 在本机/目标 GPU 上快速获得量级判断，便于写测试报告
    - 确认 cutlass 路径能正确运行并复用内部 workspace（避免每次分配）
    """
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    dtype = torch.float16

    B = 8
    num_kv_heads = 4
    group_size = 8
    num_heads = num_kv_heads * group_size
    D = 64
    Kmax = 2048

    handles = RequestHandleAllocator(max_requests=16)
    pool = CentroidPool(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        head_dim=D,
        max_requests=16,
        handle_allocator=handles,
        max_num_centroids=Kmax,
        dtype=dtype,
        device=device,
    )

    layer = 0
    req_handles = [handles.allocate() for _ in range(B)]
    for rid in req_handles:
        for h in range(num_kv_heads):
            c = torch.randn((Kmax, D), device=device, dtype=dtype)
            cs = torch.ones((Kmax,), device=device, dtype=torch.int32)
            pool.append_centroids(rid, layer, h, c, cs)

    q = torch.randn((B, num_heads, D), device=device, dtype=dtype)

    warmup = 10
    iters = 50

    rh = torch.tensor(req_handles, device=device, dtype=torch.int64)
    for _ in range(warmup):
        _ = pool.batch_topk_by_handles(rh, layer, q, k=64, backend="torch", return_format="packed")
        _ = pool.batch_topk_by_handles(rh, layer, q, k=64, backend="cutlass", return_format="packed")

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = pool.batch_topk_by_handles(rh, layer, q, k=64, backend="torch", return_format="packed")
    end.record()
    torch.cuda.synchronize()
    t_torch = start.elapsed_time(end) / iters

    start.record()
    for _ in range(iters):
        _ = pool.batch_topk_by_handles(rh, layer, q, k=64, backend="cutlass", return_format="packed")
    end.record()
    torch.cuda.synchronize()
    t_cutlass = start.elapsed_time(end) / iters

    print(
        f"[perf] batch_topk(B={B},H={num_heads},D={D},K={Kmax},k=64) "
        f"torch={t_torch:.3f}ms cutlass={t_cutlass:.3f}ms"
    )
