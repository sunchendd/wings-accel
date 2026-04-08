__all__ = ["patch_vllm_ascend_parallel_spec_decode", "patch_vllm_ears"]

from .parallel_spec_decode_patch import patch_vllm_ascend_parallel_spec_decode
from wings_engine_patch.patch_vllm_container.v0_17_0.ears_patch import patch_vllm_ears
