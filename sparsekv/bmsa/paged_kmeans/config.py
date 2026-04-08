from dataclasses import dataclass
from enum import Enum
from     # 防御性断言

    return num_segments, num_centroids import List, Tuple

import math
import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype


K = 1024


def _ceil_div(a: int, b: int) -> int:
    """向上取整除法：ceil(a / b)"""
    return (a + b - 1) // b


def _round_up(x: int, base: int) -> int:
    """将 x 向上对齐到 base 的倍数"""
    return _ceil_div(x, base) * base


def get_num_segments_and_centroids(seq_len: int, block_size: int) -> Tuple[int, int]:
    """
    返回 (num_segments, num_centroids)，满足：
    - num_centroids % num_segments == 0
    - num_centroids 是 32 的倍数（同时也是 16 的倍数）
    - (num_centroids // num_segments) 是 32 的倍数（同时也是 16 的倍数）

    策略：
    1) num_segments 初始值：
       - seq_len > 64K：按 8K 粒度估算，否则按 4K 粒度估算
       - 取 ceil(seq_len / unit)（避免过小为 0），再保证 >= 1
    2) num_centroids 初始值：
       - 先取 ceil(seq_len / block_size)，再向上对齐到 32 的倍数
    3) 为满足 ratio 也为 32 倍数：令 ratio 为 32 的倍数
       - ratio = ceil(centroids_init / num_segments) 后再向上对齐到 32
       - 最终 num_centroids = num_segments * ratio
       - 这样 num_centroids 与 ratio 都是 32 的倍数；且 num_centroids % num_segments == 0

    重要说明：
    - PagedKMeansClusterer 的 assign kernel 需要 `num_centroids % assign_block_k == 0`。
      默认配置 assign_block_k=32，因此这里直接对齐到 32，避免集成时断言失败。
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # (1) num_segments：动态 4K/8K
    unit = 8 * K if seq_len > 64 * K else 4 * K
    num_segments = max(1, _ceil_div(seq_len, unit))

    # (2) num_centroids 初始值：基于 seq_len / block_size，并对齐到 32（兼容默认 assign_block_k=32）
    centroids_init = max(32, _round_up(_ceil_div(seq_len, block_size), 32))

    # (3) ratio 对齐到 32，确保 (num_centroids // num_segments) 是 32 的倍数
    ratio = _round_up(_ceil_div(centroids_init, num_segments), 32)
    num_centroids = num_segments * ratio

    # 防御性断言
    # assert num_centroids % num_segments == 0
    # assert num_centroids % 16 == 0  # => “16 或 32 的倍数”
    # assert (num_centroids // num_segments) % 16 == 0

    return num_segments, num_centroids


class MetricType(str, Enum):
    """
    相似度度量类型。

    目前实现固定对齐 `reference_retroinfer` 的 TopK 语义：
    - 使用 scaled dot-product 得到 logits
    - 在 centroid 维度做 softmax
    - 对 GQA 的 group 维度求和，得到每个 kv_head 的 centroid 分数 dist
    - 对 dist 做 TopK (largest=True)

    因此这里保留枚举主要是为了配置兼容性，便于后续扩展。
    """

    DOT = "dot"


@dataclass
class PagedKMeansConfig:
    """
    PagedKMeansConfig：执行KMeans聚类时的配置参数

    这组参数既服务于 Prefill 的全量聚类，也服务于 Decode 的增量聚类（滑动窗口）。
    需要注意：增量聚类采用“追加簇”的思路（expanded centroid pool），不会把新 token
    重新分配到历史簇，也不会触发全量重聚类；因此 `num_centroids` 对于增量窗口一般较小。

    常见取值示例：
        Prefill -> L: 32K, num_segments: 4, num_centroids: 32K/16=2048
        Prefill -> L: 128K, num_segments: 16, num_centroids: 128K/16=8192
        Decode -> L: 1024, num_segments: 1, num_centroids: 1024/16=64

    Triton参数取值建议：
        block_k_centroids：经验优先级：32 > 16 > 64
        block_n_tokens：128，Decode时 L=1024 比较小，如果发现 launch 不够饱满，可试 256
        num_warps： 4，当 BLOCK_N=256 或 BLOCK_K=64 时可试 8
        num_stages：必要Kernel取2，非必要取1

    与 vLLM 的关系：
    - `block_size` 来自 vLLM 的 KV block size（PagedKV 的组织单位）
    - `num_layers/num_heads/num_kv_heads/head_dim` 与模型结构一致
    - `max_requests/max_num_centroids` 用于 centroid 池化管理，避免按最大长度直接预分配
    """

    head_dim: int
    block_size: int
    num_kv_heads: int
    num_layers: int
    num_heads: int

    # clustering
    iters: int = 3
    metric: MetricType = MetricType.DOT

    # tiling (assign)
    block_n_tokens: int = 128
    assign_block_k: int = 32

    # tiling (update) - follow reference
    update_block_k: int = 128

    # launch tuning
    num_warps: int = 4
    assign_num_stages: int = 2
    update_num_stages: int = 1
    csr_num_stages: int = 1

    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    max_requests: int = 128
    max_num_centroids: int = 8192

    def validate(self):
        assert self.block_n_tokens in (64, 128, 256)
        assert self.assign_block_k in (16, 32, 64)
        assert self.update_block_k in (64, 128, 256)
        assert self.iters >= 1
        assert self.max_requests >= 1
        assert self.max_num_centroids is not None and self.max_num_centroids >= 1


def build_default_kmeans_config(vllm_config: VllmConfig) -> PagedKMeansConfig:
    block_size = vllm_config.cache_config.block_size
    num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    head_dim = vllm_config.model_config.get_head_size()
    max_requests = vllm_config.sparse_config.max_num_seqs
    kv_cache_dtype = kv_cache_dtype_str_to_dtype(
        vllm_config.cache_config.cache_dtype,
        vllm_config.model_config,
    )
    max_model_len = int(vllm_config.model_config.max_model_len)
    _, k_max = get_num_segments_and_centroids(max_model_len, block_size)

    return PagedKMeansConfig(
        block_size=block_size,  # vLLM block size
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_requests=int(max_requests),
        max_num_centroids=k_max,
        dtype=kv_cache_dtype,
    )
