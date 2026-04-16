__all__ = [
    "ears_patch",
    "patch_vllm_ears",
    "patch_vllm_sparse_kv",
    "sparse_kv_patch",
]

from importlib import import_module


def __getattr__(name):
    if name == "ears_patch":
        return import_module(f"{__name__}.ears_patch")
    if name == "ears_nvidia_runtime_hooks":
        return import_module(f"{__name__}.ears_nvidia_runtime_hooks")
    if name == "sparse_kv_patch":
        return import_module(f"{__name__}.sparse_kv_patch")
    if name == "patch_vllm_ears":
        from .ears_patch import patch_vllm_ears

        return patch_vllm_ears
    if name == "patch_vllm_sparse_kv":
        from .sparse_kv_patch import patch_vllm_sparse_kv

        return patch_vllm_sparse_kv
    raise AttributeError(name)
