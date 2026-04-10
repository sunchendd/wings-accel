"""vllm-ascend 0.17.0rc1 runtime patches."""

from importlib import import_module

__all__ = [
    "draft_model_patch",
    "ears_patch",
    "patch_vllm_draft_model",
]


def __getattr__(name):
    if name == "draft_model_patch":
        return import_module(f"{__name__}.draft_model_patch")
    if name == "ears_patch":
        return import_module(f"{__name__}.ears_patch")
    if name == "patch_vllm_draft_model":
        from .draft_model_patch import patch_vllm_draft_model

        return patch_vllm_draft_model
    raise AttributeError(name)
