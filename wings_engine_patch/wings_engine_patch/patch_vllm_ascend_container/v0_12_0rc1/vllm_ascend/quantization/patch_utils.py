import wrapt


def patch_ASCEND_QUANTIZATION_METHOD_MAP():
    # `vllm_ascend.quantization.utils` has a circular import in 0.12.0rc1 that
    # prevents it from ever finishing loading, so a hook registered directly on
    # that module would never fire.  Instead we hook on
    # `vllm_ascend.quantization.quant_config`, which imports utils *after* the
    # circular dependency is resolved, guaranteeing the module is stable by the
    # time our callback runs.
    def update_map(m):
        import sys
        utils = sys.modules.get('vllm_ascend.quantization.utils')
        if utils is None:
            # utils hasn't been imported yet — import it now through quant_config
            from vllm_ascend.quantization import utils  # noqa: F401
            utils = sys.modules['vllm_ascend.quantization.utils']

        from .w8a16fp8 import (AscendW8A16FP8LinearMethod,
                               AscendW8A16FP8FusedMoEMethod,
                               AscendW8A16FP8KVCacheMethod)
        from .w4a16nvfp4 import AscendW4A16NVFP4LinearMethod

        utils.ASCEND_QUANTIZATION_METHOD_MAP['W8A16FP8'] = {
            "linear": AscendW8A16FP8LinearMethod,
            "moe": AscendW8A16FP8FusedMoEMethod,
            "attention": AscendW8A16FP8KVCacheMethod,
        }
        utils.ASCEND_QUANTIZATION_METHOD_MAP['W4A16NVFP4'] = {
            "linear": AscendW4A16NVFP4LinearMethod,
        }

    wrapt.register_post_import_hook(
        update_map,
        'vllm_ascend.quantization.quant_config'
    )
