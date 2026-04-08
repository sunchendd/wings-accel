"""
SparseConfig and BMSAConfig dataclasses for sparse KV cache attention.

These classes are injected into the vllm.config namespace at runtime
by the sparse_kv monkey-patch. They mirror the interface original
0005-xfusion patch provided but only include BMSA algorithm support.
"""
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Optional


SparseAlgoType = Literal["BMSA"]
HeadBudgetPolicy = Literal["uniform", "dynamic"]
LayerPolicyType = Literal["mix", "all_sparse"]


@dataclass
class BMSAConfig:
    """Configuration for Block-Mean Sparse Attention algorithm."""

    min_topk_len: int = 64
    max_topk_len: int = 512
    num_prefetch_blocks: int = 8
    init_windows_size: int = 8
    ptopk_prefetch_enable: bool = True
    topk_update_interval: int = 3
    align_kv_cache: bool = False
    enable_cuda_topk: bool = False
    topk_type: Literal["block-mean", "paged-kmeans"] = "block-mean"
    kmeans_centroid_topk_ratio: float = 0.20
    kmeans_topk_backend: Literal["cutlass", "torch"] = "cutlass"

    def get_blocks_budget(
        self,
        prompt_len: int,
        block_size: int,
        total_budget: float,
    ) -> int:
        if prompt_len <= 0:
            return 0
        if total_budget is None or total_budget <= 0:
            return 0

        blocks_len = int(math.ceil(prompt_len / block_size))
        if total_budget < 1:
            topk_len = int(math.ceil(blocks_len * float(total_budget)))
        else:
            topk_len = int(math.ceil(total_budget / block_size))

        if topk_len < self.min_topk_len:
            topk_len = min(self.min_topk_len, blocks_len)
        elif topk_len > self.max_topk_len:
            topk_len = min(self.max_topk_len, blocks_len)
        return topk_len

    @staticmethod
    def finalize(vllm_config: Any) -> None:
        return


@dataclass
class LayerSparsePolicy:
    """Configuration for layer-wise sparse policy."""

    policy_type: LayerPolicyType = "all_sparse"
    full_attn_layers: Optional[list[int]] = None
    sparse_layers: Optional[list[int]] = None

    def __post_init__(self) -> None:
        if self.policy_type == "mix":
            if self.full_attn_layers is None and self.sparse_layers is None:
                raise ValueError(
                    "When policy_type is 'mix', either full_attn_layers or "
                    "sparse_layers must be specified."
                )
            if self.full_attn_layers is not None and self.sparse_layers is not None:
                raise ValueError(
                    "Cannot specify both full_attn_layers and sparse_layers."
                )
            layers_to_check = self.full_attn_layers or self.sparse_layers
            if layers_to_check and any(idx < 0 for idx in layers_to_check):
                raise ValueError("Layer indices must be non-negative.")


@dataclass
class SparseConfig:
    """Configuration for Sparse KV cache and Sparse Attention."""

    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 32

    enable_sparse: bool = False
    lc_sparse_threshold: Optional[int] = None
    sparse_algo_type: SparseAlgoType = "BMSA"
    sparse_algo_config: Optional[BMSAConfig] = None
    enable_static_sparse: bool = False
    max_num_seqs: int = 32
    head_budget_policy: HeadBudgetPolicy = "uniform"
    total_budget: float = 0.3
    topk_oversample: float = 1.25
    enable_dynamic_budgets: bool = True
    layer_policy: Optional[LayerSparsePolicy] = None

    def __post_init__(self) -> None:
        if not self.enable_sparse:
            return

        if self.total_budget is not None and self.total_budget <= 0:
            self.enable_sparse = False
            return

        if self.enable_sparse and self.sparse_algo_config is None:
            self.sparse_algo_config = BMSAConfig()

        if self.layer_policy is None:
            self.layer_policy = LayerSparsePolicy()

        if self.lc_sparse_threshold is None:
            self.lc_sparse_threshold = 0

        if self.lc_sparse_threshold < 0:
            raise ValueError(
                f"lc_sparse_threshold must be non-negative, "
                f"got {self.lc_sparse_threshold}"
            )

    def finalize(self, vllm_config: Any) -> None:
        """Called after VllmConfig.__post_init__ to finalize sparse config."""
        if self.sparse_algo_config is not None:
            self.sparse_algo_config.finalize(vllm_config)

    def get_blocks_budget(self, prompt_len: int, block_size: int) -> int:
        if prompt_len <= 0:
            return 0
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        assert self.sparse_algo_config is not None
        return self.sparse_algo_config.get_blocks_budget(
            prompt_len, block_size, self.total_budget
        )

    def compute_hash(self) -> str:
        if not self.enable_sparse:
            return hashlib.md5(
                b"sparse_disabled", usedforsecurity=False
            ).hexdigest()

        factors: list[Any] = [
            self.enable_sparse,
            self.lc_sparse_threshold,
            self.sparse_algo_type,
            self.enable_static_sparse,
            self.head_budget_policy,
            self.total_budget,
        ]
        if self.layer_policy is not None:
            factors.append(self.layer_policy.policy_type)
            factors.append(str(self.layer_policy.full_attn_layers))
            factors.append(str(self.layer_policy.sparse_layers))
        if self.sparse_algo_config is not None:
            factors.append(str(self.sparse_algo_config))

        return hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()
