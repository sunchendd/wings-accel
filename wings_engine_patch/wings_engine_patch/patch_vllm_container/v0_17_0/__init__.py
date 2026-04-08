__all__ = ["patch_vllm_adaptive_draft_model", "patch_vllm_sparse_kv"]

from .adaptive_draft_model_patch import patch_vllm_adaptive_draft_model
from .sparse_kv_patch import patch_vllm_sparse_kv
