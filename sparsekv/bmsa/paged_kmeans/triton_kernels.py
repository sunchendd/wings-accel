import triton
import triton.language as tl

# =========================
# Triton kernels (pagedKV)
# =========================


@triton.jit
def _init_centroids_kernel(
    KV, BLOCK_TABLE, START_POS, OUT,
    stride_kv0, stride_kv1, stride_kv2, stride_kv3, stride_kv4,
    stride_bt0,
    stride_out0, stride_out1, stride_out2,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    k = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # --- FIX (3): match reference init exactly ---
    # integer equivalent: sample = floor((2*k*L + L) / (2*K)) = (2*k*L + L) // (2*K)
    sample = (2 * k * L + L) // (2 * K)

    start_pos = tl.load(START_POS).to(tl.int32)
    pos = start_pos + sample

    logical_block = pos // block_size
    offset = pos - logical_block * block_size
    phys = tl.load(BLOCK_TABLE + logical_block * stride_bt0).to(tl.int32)

    ptr = (
        KV
        + 0 * stride_kv0
        + phys * stride_kv1
        + offset * stride_kv2
        + h * stride_kv3
        + offs_d * stride_kv4
    )
    v = tl.load(ptr, mask=d_mask, other=0.0)

    out_ptr = OUT + h * stride_out0 + k * stride_out1 + offs_d * stride_out2
    tl.store(out_ptr, v, mask=d_mask)


@triton.jit
def _assign_sumkv_segmented_kernel(
    KV, BLOCK_TABLE, START_POS,
    CENTROIDS, SUM_K, SUM_V, CNT, ASSIGN,
    stride_kv0, stride_kv1, stride_kv2, stride_kv3, stride_kv4,
    stride_bt0,
    stride_c0, stride_c1, stride_c2,
    stride_sk0, stride_sk1, stride_sk2,
    stride_sv0, stride_sv1, stride_sv2,
    stride_cnt0, stride_cnt1,
    stride_a0, stride_a1,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    L_seg: tl.constexpr,
    K_seg: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_z = tl.program_id(1)  # z = h * NUM_SEGMENTS + seg
    h = pid_z // NUM_SEGMENTS
    seg = pid_z - h * NUM_SEGMENTS

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < L_seg

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    start_pos0 = tl.load(START_POS).to(tl.int32)
    pos = start_pos0 + seg * L_seg + offs_n

    logical_block = pos // block_size
    offset = pos - logical_block * block_size
    phys = tl.load(
        BLOCK_TABLE + logical_block * stride_bt0,
        mask=n_mask,
        other=0,
    ).to(tl.int32)

    k_ptr = (
        KV
        + 0 * stride_kv0
        + phys[:, None] * stride_kv1
        + offset[:, None] * stride_kv2
        + h * stride_kv3
        + offs_d[None, :] * stride_kv4
    )
    v_ptr = (
        KV
        + 1 * stride_kv0
        + phys[:, None] * stride_kv1
        + offset[:, None] * stride_kv2
        + h * stride_kv3
        + offs_d[None, :] * stride_kv4
    )

    # load in original dtype (fp16/bf16), accumulate in fp32
    kvec = tl.load(k_ptr, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    vvec = tl.load(v_ptr, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    # --- FIX (1)(2): strict match reference semantics ---
    # Reference uses dot(k, centroid) with NO key normalization.
    k_for_dot = kvec

    c_base_k = seg * K_seg

    max_val = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    max_idx_local = tl.zeros([BLOCK_N], dtype=tl.int32)

    for ck in tl.range(0, K_seg, BLOCK_K):
        offs_k = ck + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_seg

        c_ptr = (
            CENTROIDS
            + h * stride_c0
            + (c_base_k + offs_k)[None, :] * stride_c1
            + offs_d[:, None] * stride_c2
        )
        c = tl.load(c_ptr, mask=d_mask[:, None] & k_mask[None, :], other=0.0)

        ip = tl.dot(k_for_dot, c).to(tl.float32)
        tmax, tidx = tl.max(ip, axis=1, return_indices=True)
        tidx = tidx + ck

        upd = tmax > max_val
        max_val = tl.maximum(max_val, tmax)
        max_idx_local = tl.where(upd, tidx, max_idx_local)

    max_idx = max_idx_local + c_base_k

    tl.store(
        ASSIGN + h * stride_a0 + (seg * L_seg + offs_n) * stride_a1,
        max_idx,
        mask=n_mask,
    )

    # sums in fp32
    tl.atomic_add(
        SUM_K + h * stride_sk0 + max_idx[:, None] * stride_sk1 + offs_d[None, :] * stride_sk2,
        kvec.to(tl.float32),
        mask=n_mask[:, None] & d_mask[None, :],
        sem="relaxed",
    )
    tl.atomic_add(
        SUM_V + h * stride_sv0 + max_idx[:, None] * stride_sv1 + offs_d[None, :] * stride_sv2,
        vvec.to(tl.float32),
        mask=n_mask[:, None] & d_mask[None, :],
        sem="relaxed",
    )
    tl.atomic_add(
        CNT + h * stride_cnt0 + max_idx * stride_cnt1,
        tl.full([BLOCK_N], 1, dtype=tl.int32),
        mask=n_mask,
        sem="relaxed",
    )


@triton.jit
def _update_centroids_segmented_kernel(
    CENTROIDS, SUM_K, CNT,
    stride_c0, stride_c1, stride_c2,
    stride_s0, stride_s1, stride_s2,
    stride_cnt0, stride_cnt1,
    head_dim: tl.constexpr,
    K_seg: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_z = tl.program_id(1)
    h = pid_z // NUM_SEGMENTS
    seg = pid_z - h * NUM_SEGMENTS

    start_k = pid_k * BLOCK_K
    offs_k_local = start_k + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    k_mask = offs_k_local < K_seg
    d_mask = offs_d < head_dim

    k_global = seg * K_seg + offs_k_local

    cnt = tl.load(
        CNT + h * stride_cnt0 + k_global * stride_cnt1,
        mask=k_mask,
        other=0,
    ).to(tl.float32)

    valid = cnt > 0
    cnt_safe = tl.maximum(cnt, 1.0)  # avoid inf/nan

    s = tl.load(
        SUM_K + h * stride_s0 + k_global[:, None] * stride_s1 + offs_d[None, :] * stride_s2,
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    x = s / cnt_safe[:, None]
    if NORMALIZE:
        norm = tl.sqrt(tl.sum(x * x, axis=1) + 1e-8)
        x = x / norm[:, None]

    tl.store(
        CENTROIDS + h * stride_c0 + k_global[:, None] * stride_c1 + offs_d[None, :] * stride_c2,
        x.to(CENTROIDS.type.element_ty),
        mask=k_mask[:, None] & d_mask[None, :] & valid[:, None],
    )


@triton.jit
def _assign_sumkv_full_kernel(
    KV, BLOCK_TABLE, START_POS,
    CENTROIDS, SUM_K, SUM_V, CNT, ASSIGN,
    stride_kv0, stride_kv1, stride_kv2, stride_kv3, stride_kv4,
    stride_bt0,
    stride_c0, stride_c1, stride_c2,
    stride_sk0, stride_sk1, stride_sk2,
    stride_sv0, stride_sv1, stride_sv2,
    stride_cnt0, stride_cnt1,
    stride_a0, stride_a1,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    h = tl.program_id(1)

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < L

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    start_pos = tl.load(START_POS).to(tl.int32)
    pos = start_pos + offs_n

    logical_block = pos // block_size
    offset = pos - logical_block * block_size
    phys = tl.load(BLOCK_TABLE + logical_block * stride_bt0, mask=n_mask, other=0).to(tl.int32)

    k_ptr = (
        KV
        + 0 * stride_kv0
        + phys[:, None] * stride_kv1
        + offset[:, None] * stride_kv2
        + h * stride_kv3
        + offs_d[None, :] * stride_kv4
    )
    v_ptr = (
        KV
        + 1 * stride_kv0
        + phys[:, None] * stride_kv1
        + offset[:, None] * stride_kv2
        + h * stride_kv3
        + offs_d[None, :] * stride_kv4
    )

    kvec = tl.load(k_ptr, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    vvec = tl.load(v_ptr, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    # --- FIX (1)(2): strict match reference semantics: dot only, no key normalization ---
    k_for_dot = kvec

    max_val = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    max_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    for ck in tl.range(0, K, BLOCK_K):
        offs_k = ck + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        c_ptr = (
            CENTROIDS
            + h * stride_c0
            + offs_k[None, :] * stride_c1
            + offs_d[:, None] * stride_c2
        )
        c = tl.load(c_ptr, mask=d_mask[:, None] & k_mask[None, :], other=0.0)

        ip = tl.dot(k_for_dot, c).to(tl.float32)
        tmax, tidx = tl.max(ip, axis=1, return_indices=True)
        tidx += ck

        upd = tmax > max_val
        max_val = tl.maximum(max_val, tmax)
        max_idx = tl.where(upd, tidx, max_idx)

    tl.store(ASSIGN + h * stride_a0 + offs_n * stride_a1, max_idx, mask=n_mask)

    tl.atomic_add(
        SUM_K + h * stride_sk0 + max_idx[:, None] * stride_sk1 + offs_d[None, :] * stride_sk2,
        kvec.to(tl.float32),
        mask=n_mask[:, None] & d_mask[None, :],
        sem="relaxed",
    )
    tl.atomic_add(
        SUM_V + h * stride_sv0 + max_idx[:, None] * stride_sv1 + offs_d[None, :] * stride_sv2,
        vvec.to(tl.float32),
        mask=n_mask[:, None] & d_mask[None, :],
        sem="relaxed",
    )
    tl.atomic_add(
        CNT + h * stride_cnt0 + max_idx * stride_cnt1,
        tl.full([BLOCK_N], 1, dtype=tl.int32),
        mask=n_mask,
        sem="relaxed",
    )


@triton.jit
def _update_centroids_full_kernel(
    CENTROIDS, SUM_K, CNT,
    stride_c0, stride_c1, stride_c2,
    stride_s0, stride_s1, stride_s2,
    stride_cnt0, stride_cnt1,
    head_dim: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    h = tl.program_id(0)
    start_k = tl.program_id(1) * BLOCK_K

    offs_k = start_k + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    k_mask = offs_k < K
    d_mask = offs_d < head_dim

    cnt = tl.load(
        CNT + h * stride_cnt0 + offs_k * stride_cnt1,
        mask=k_mask,
        other=0,
    ).to(tl.float32)

    valid = cnt > 0
    cnt_safe = tl.maximum(cnt, 1.0)

    s = tl.load(
        SUM_K + h * stride_s0 + offs_k[:, None] * stride_s1 + offs_d[None, :] * stride_s2,
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    x = s / cnt_safe[:, None]
    if NORMALIZE:
        norm = tl.sqrt(tl.sum(x * x, axis=1) + 1e-8)
        x = x / norm[:, None]

    tl.store(
        CENTROIDS + h * stride_c0 + offs_k[:, None] * stride_c1 + offs_d[None, :] * stride_c2,
        x.to(CENTROIDS.type.element_ty),
        mask=k_mask[:, None] & d_mask[None, :] & valid[:, None],
    )


@triton.jit
def _cnt_to_offsets_kernel(
    CNT, OFFSETS,
    stride_cnt0, stride_cnt1,
    stride_off0, stride_off1,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)
    # 说明：这里的 `tl.arange(0, ...)` 的 range 必须是 2 的幂（Triton 约束），否则会在
    # kernel 编译阶段直接抛异常：ValueError: arange's range must be a power of 2。
    #
    # 但在 CRSA 集成到 vLLM 的长上下文场景里，num_centroids(K) 的生成策略是：
    # - 对齐到 32（满足 assign kernel 的 tile 约束），并满足 K % num_segments == 0；
    # - K 不保证是 2 的幂（例如 seq_len=24K 且 block_size=16 时，常见 K=1536）。
    #
    # 因此本 kernel 采用 “pad-to-power-of-two + mask” 的标准 Triton 写法：
    # - 用 BLOCK_K=next_pow2(K) 作为 arange 的上界；
    # - 用 k_mask=offs_k<K 屏蔽尾部 padding 元素；
    # 从而保持 offsets 的语义完全不变，同时消除 “K 必须是 2 的幂” 的隐含要求。
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < K

    cnt = tl.load(
        CNT + h * stride_cnt0 + offs_k * stride_cnt1,
        mask=k_mask,
        other=0,
    )
    cumsum = tl.cumsum(cnt, axis=0)

    if tl.program_id(1) == 0:
        tl.store(OFFSETS + h * stride_off0, 0)

    tl.store(
        OFFSETS + h * stride_off0 + (offs_k + 1) * stride_off1,
        cumsum,
        mask=k_mask,
    )


@triton.jit
def _build_perm_kernel(
    ASSIGN, OFFSETS, WRITE_PTR, PERM,
    stride_a0, stride_a1,
    stride_off0, stride_off1,
    stride_wp0, stride_wp1,
    stride_p0, stride_p1,
    L: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    h = tl.program_id(1)

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < L

    assign = tl.load(
        ASSIGN + h * stride_a0 + offs_n * stride_a1,
        mask=n_mask,
        other=0,
    )

    wp = tl.atomic_add(
        WRITE_PTR + h * stride_wp0 + assign * stride_wp1,
        1,
        mask=n_mask,
    )

    tl.store(
        PERM + h * stride_p0 + wp * stride_p1,
        offs_n,
        mask=n_mask,
    )
