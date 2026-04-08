import os

from .ears_patch import _maybe_enable_ears_sampler


def _patch_vllm_ascend_envs_module(module) -> None:
    env_variables = getattr(module, "env_variables", None)
    if not isinstance(env_variables, dict):
        return
    env_variables.setdefault(
        "VLLM_EARS_TOLERANCE",
        lambda: float(os.getenv("VLLM_EARS_TOLERANCE", "0.0")),
    )


def _patch_vllm_ascend_model_runner_module(module) -> None:
    runner_cls = getattr(module, "NPUModelRunner", None)
    if runner_cls is None:
        return

    original_set_up_drafter = runner_cls._set_up_drafter
    if getattr(original_set_up_drafter, "_wings_ears_patched", False):
        return

    def patched_set_up_drafter(self, *args, **kwargs):
        result = original_set_up_drafter(self, *args, **kwargs)
        _maybe_enable_ears_sampler(self)
        return result

    patched_set_up_drafter._wings_ears_patched = True  # pylint: disable=protected-access
    runner_cls._set_up_drafter = patched_set_up_drafter


def register_ascend_runtime_hooks(register_hook) -> None:
    register_hook("vllm_ascend.envs", _patch_vllm_ascend_envs_module)
    register_hook(
        "vllm_ascend.worker.model_runner_v1",
        _patch_vllm_ascend_model_runner_module,
    )
