import inspect
import logging
import sys


LOGGER = logging.getLogger("wings_accel.draft_model")


class _StderrProxy:
    @staticmethod
    def write(message: str) -> int:
        return sys.stderr.write(message)

    @staticmethod
    def flush() -> None:
        sys.stderr.flush()



def _configure_logger() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(_StderrProxy())
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False



def log_runtime_state(event: str, /, **fields) -> None:
    parts = [f"{key}={value}" for key, value in sorted(fields.items())]
    suffix = f" {' '.join(parts)}" if parts else ""
    LOGGER.info("[wins-accel] %s%s", event, suffix)



def _register_or_apply_post_import_hook(module_name: str, patcher) -> None:
    import wrapt

    if module_name in sys.modules:
        patcher(sys.modules[module_name])
    wrapt.register_post_import_hook(patcher, module_name)



def _align_hidden_states_last_dim(hidden_states, hidden_size):
    if (
        hidden_size is None
        or not hasattr(hidden_states, "shape")
        or hidden_states.shape[-1] == hidden_size
    ):
        return hidden_states

    if hidden_states.shape[-1] > hidden_size:
        return hidden_states[..., :hidden_size].contiguous()

    import torch.nn.functional as functional

    return functional.pad(hidden_states, (0, hidden_size - hidden_states.shape[-1]))



def _get_target_hidden_size(instance):
    hidden_states = getattr(instance, "hidden_states", None)
    if hasattr(hidden_states, "shape"):
        return hidden_states.shape[-1]
    return getattr(instance, "hidden_size", None)



def _patch_eagle_proposer_module(module) -> None:
    cls = getattr(module, "SpecDecodeBaseProposer", None)
    if cls is None:
        return

    original_set_inputs_first_pass = getattr(cls, "set_inputs_first_pass", None)
    if original_set_inputs_first_pass is None:
        return
    if getattr(original_set_inputs_first_pass, "_wings_draft_model_patched", False):
        return

    set_inputs_signature = inspect.signature(original_set_inputs_first_pass)

    def patched_set_inputs_first_pass(self, *args, **kwargs):
        bound = set_inputs_signature.bind(self, *args, **kwargs)
        if getattr(self, "method", None) == "draft_model":
            target_hidden_states = bound.arguments.get("target_hidden_states")
            aligned_hidden_states = _align_hidden_states_last_dim(
                target_hidden_states,
                _get_target_hidden_size(self),
            )
            if aligned_hidden_states is not target_hidden_states:
                log_runtime_state(
                    "aligned-draft-hidden-states",
                    from_hidden_size=target_hidden_states.shape[-1],
                    to_hidden_size=aligned_hidden_states.shape[-1],
                )
                bound.arguments["target_hidden_states"] = aligned_hidden_states
        return original_set_inputs_first_pass(*bound.args, **bound.kwargs)

    patched_set_inputs_first_pass._wings_draft_model_patched = True  # pylint: disable=protected-access
    cls.set_inputs_first_pass = patched_set_inputs_first_pass



def patch_vllm_draft_model():
    _register_or_apply_post_import_hook(
        "vllm_ascend.spec_decode.eagle_proposer",
        _patch_eagle_proposer_module,
    )
    log_runtime_state("draft_model patch enabled")


_configure_logger()
