import wrapt


def patch_ASCEND_QUANTIZATION_METHOD_MAP():
    def update_map(m):
        from vllm_ascend.quantization.utils import ASCEND_QUANTIZATION_METHOD_MAP
        from .w8a16fp8 import (AscendW8A16FP8LinearMethod,
                               AscendW8A16FP8FusedMoEMethod,
                               AscendW8A16FP8KVCacheMethod)
        from .w4a16nvfp4 import AscendW4A16NVFP4LinearMethod
        
        ASCEND_QUANTIZATION_METHOD_MAP['W8A16FP8'] = {
            "linear": AscendW8A16FP8LinearMethod,
            "moe": AscendW8A16FP8FusedMoEMethod,
            "attention": AscendW8A16FP8KVCacheMethod,
        }
        ASCEND_QUANTIZATION_METHOD_MAP['W4A16NVFP4'] = {
            "linear": AscendW4A16NVFP4LinearMethod,
        }

    wrapt.register_post_import_hook(
        update_map,
        'vllm_ascend.quantization.utils'
    )
