from importlib import import_module


__all__ = [
    "adaptive_draft_model_patch",
    "ears_ascend_runtime_hooks",
    "ears_patch",
    "patch_vllm_adaptive_draft_model",
    "patch_vllm_ascend_draft_compat",
    "patch_vllm_ears",
    "patch_vllm_sparse_kv",
    "sparse_kv_patch",
]


def __getattr__(name):
    if name == "adaptive_draft_model_patch":
        return import_module(f"{__name__}.adaptive_draft_model_patch")
    if name == "ears_ascend_runtime_hooks":
        return import_module(f"{__name__}.ears_ascend_runtime_hooks")
    if name == "ears_patch":
        return import_module(f"{__name__}.ears_patch")
    if name == "ears_nvidia_runtime_hooks":
        return import_module(f"{__name__}.ears_nvidia_runtime_hooks")
    if name == "sparse_kv_patch":
        return import_module(f"{__name__}.sparse_kv_patch")
    if name == "patch_vllm_adaptive_draft_model":
        from .adaptive_draft_model_patch import patch_vllm_adaptive_draft_model

        return patch_vllm_adaptive_draft_model
    if name == "patch_vllm_ascend_draft_compat":
        from .ears_patch import patch_vllm_ascend_draft_compat

        return patch_vllm_ascend_draft_compat
    if name == "patch_vllm_ears":
        from .ears_patch import patch_vllm_ears

        return patch_vllm_ears
    if name == "patch_vllm_sparse_kv":
        from .sparse_kv_patch import patch_vllm_sparse_kv

        return patch_vllm_sparse_kv
    raise AttributeError(name)
