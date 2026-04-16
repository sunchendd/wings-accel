import torch
import triton

from .config import PagedKMeansConfig
from .triton_kernels import (
    _assign_sumkv_full_kernel,
    _assign_sumkv_segmented_kernel,
    _build_perm_kernel,
    _cnt_to_offsets_kernel,
    _init_centroids_kernel,
    _update_centroids_full_kernel,
    _update_centroids_segmented_kernel,
)


class PagedKMeansWorkspace:
    def __init__(self, H, K, D, L, device):
        self.sum_k = torch.zeros((H, K, D), device=device, dtype=torch.float32)
        self.sum_v = torch.zeros((H, K, D), device=device, dtype=torch.float32)
        self.cnt = torch.zeros((H, K), device=device, dtype=torch.int32)
        self.assign = torch.empty((H, L), device=device, dtype=torch.int32)

    def reset(self):
        self.sum_k.zero_()
        self.sum_v.zero_()
        self.cnt.zero_()


class PagedKMeansClusterer:
    def __init__(self, cfg: PagedKMeansConfig):
        cfg.validate()
        self.cfg = cfg

    @torch.no_grad()
    def cluster(
        self,
        kv_cache: torch.Tensor,        # [2, NB, BS, H, D]
        block_table_1d: torch.Tensor,  # [M]
        start_pos: torch.Tensor,       # scalar tensor on device
        L: int,
        num_segments: int = 1,
        num_centroids: int = 64,
    ):
        cfg = self.cfg
        device = kv_cache.device
        _, _, _, H, D = kv_cache.shape

        assert num_segments >= 1
        assert num_centroids % num_segments == 0

        assert num_centroids % cfg.assign_block_k == 0
        if num_segments > 1:
            assert (num_centroids // num_segments) % cfg.assign_block_k == 0

        K = num_centroids
        S = num_segments
        K_seg = K // S

        centroids_dtype = kv_cache.dtype
        centroids = torch.empty((H, K, D), device=device, dtype=centroids_dtype)

        # init (fixed to reference sampling)
        _init_centroids_kernel[(H, K)](kv_cache, block_table_1d, start_pos, centroids, *kv_cache.stride(),
            block_table_1d.stride(0),
            *centroids.stride(),
            block_size=cfg.block_size,
            head_dim=D,
            L=L,
            K=K,
            BLOCK_D=D,
            num_warps=cfg.num_warps,
            num_stages=cfg.assign_num_stages,
        )

        ws = PagedKMeansWorkspace(H, K, D, L, device)

        L_seg = L // S  # drop remainder during segmented iters like reference

        for it in range(cfg.iters):
            ws.reset()

            # reference: early iters normalize_centroids=True; last iter False
            normalize_centroids = (it < cfg.iters - 1)

            if it < cfg.iters - 1 and S > 1:
                grid_n = triton.cdiv(L_seg, cfg.block_n_tokens)
                grid_z = H * S

                _assign_sumkv_segmented_kernel[(grid_n, grid_z)](kv_cache, block_table_1d, start_pos, centroids,
                    ws.sum_k, ws.sum_v, ws.cnt, ws.assign,
                    *kv_cache.stride(),
                    block_table_1d.stride(0),
                    *centroids.stride(),
                    *ws.sum_k.stride(),
                    *ws.sum_v.stride(),
                    *ws.cnt.stride(),
                    *ws.assign.stride(),
                    block_size=cfg.block_size,
                    head_dim=D,
                    L_seg=L_seg,
                    K_seg=K_seg,
                    NUM_SEGMENTS=S,
                    BLOCK_N=cfg.block_n_tokens,
                    BLOCK_K=cfg.assign_block_k,
                    BLOCK_D=D,
                    num_warps=cfg.num_warps,
                    num_stages=cfg.assign_num_stages,
                    )

                _update_centroids_segmented_kernel[(triton.cdiv(K_seg, cfg.update_block_k), grid_z)](centroids,
                    ws.sum_k, ws.cnt,
                    *centroids.stride(),
                    *ws.sum_k.stride(),
                    *ws.cnt.stride(),
                    head_dim=D,
                    K_seg=K_seg,
                    NUM_SEGMENTS=S,
                    BLOCK_K=cfg.update_block_k,
                    BLOCK_D=D,
                    NORMALIZE=normalize_centroids,
                    num_warps=cfg.num_warps,
                    num_stages=cfg.update_num_stages,
                    )
            else:
                _assign_sumkv_full_kernel[(triton.cdiv(L, cfg.block_n_tokens), H)](kv_cache, block_table_1d, start_pos,
                    centroids, ws.sum_k, ws.sum_v,
                    ws.cnt, ws.assign,
                    *kv_cache.stride(),
                    block_table_1d.stride(0),
                    *centroids.stride(),
                    *ws.sum_k.stride(),
                    *ws.sum_v.stride(),
                    *ws.cnt.stride(),
                    *ws.assign.stride(),
                    block_size=cfg.block_size,
                    head_dim=D,
                    L=L,
                    K=K,
                    BLOCK_N=cfg.block_n_tokens,
                    BLOCK_K=cfg.assign_block_k,
                    BLOCK_D=D,
                    num_warps=cfg.num_warps,
                    num_stages=cfg.assign_num_stages,
                    )

                _update_centroids_full_kernel[(H, triton.cdiv(K, cfg.update_block_k))](centroids, ws.sum_k, ws.cnt,
                    *centroids.stride(),
                    *ws.sum_k.stride(),
                    *ws.cnt.stride(),
                    head_dim=D,
                    K=K,
                    BLOCK_K=cfg.update_block_k,
                    BLOCK_D=D,
                    NORMALIZE=normalize_centroids,
                    # final full iteration: no normalize (match reference)
                    num_warps=cfg.num_warps,
                    num_stages=cfg.update_num_stages,
                    )

        offsets = torch.empty((H, K + 1), device=device, dtype=torch.int32)
        # Triton 的 tl.arange 要求 range 为 2 的幂；而我们上层的 num_centroids(K)
        # 只保证对齐到 32（并满足 K % num_segments==0），不保证为 2 的幂。
        # 这里把 K 向上 pad 到 next_pow2(K) 传给 kernel，kernel 内部会用 mask 保持语义不变。
        block_k = 1 << (int(K) - 1).bit_length()
        _cnt_to_offsets_kernel[(H, 1)](ws.cnt, offsets, *ws.cnt.stride(),
                                       *offsets.stride(),
                                       K=K,
                                       BLOCK_K=block_k,
                                       num_warps=cfg.num_warps,
                                       num_stages=cfg.csr_num_stages,
                                       )

        write_ptr = offsets[:, :-1].clone()
        perm = torch.empty((H, L), device=device, dtype=torch.int32)
        _build_perm_kernel[(triton.cdiv(L, cfg.block_n_tokens), H)](ws.assign, offsets, write_ptr, perm,
                    *ws.assign.stride(),
                    *offsets.stride(),
                    *write_ptr.stride(),
                    *perm.stride(),
                    L=L,
                    BLOCK_N=cfg.block_n_tokens,
                    num_warps=cfg.num_warps,
                    num_stages=cfg.csr_num_stages,
                    )

        sum_v_out = ws.sum_v.to(centroids_dtype)
        return centroids, sum_v_out, ws.assign, ws.cnt, offsets, perm
