from .ears_patch import _maybe_enable_ears_sampler


def _patch_vllm_gpu_model_runner_module(module) -> None:
    runner_cls = getattr(module, "GPUModelRunner", None)
    if runner_cls is None:
        return

    original_init = runner_cls.__init__
    if getattr(original_init, "_wings_ears_patched", False):
        return

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _maybe_enable_ears_sampler(self)

    patched_init._wings_ears_patched = True  # pylint: disable=protected-access
    runner_cls.__init__ = patched_init


def register_nvidia_runtime_hooks(register_hook) -> None:
    register_hook(
        "vllm.v1.worker.gpu_model_runner",
        _patch_vllm_gpu_model_runner_module,
    )
