import inspect
import wrapt


def _make_create_weights_wrapper():
    """Return a wrapper that delegates to self.quant_method.create_weights when present."""
    def _wrapper(wrapped, instance, args, kwargs):
        self = instance
        if hasattr(self.quant_method, 'create_weights'):
            self.quant_method.create_weights(*args, **kwargs)
            return
        return wrapped(*args, **kwargs)
    return _wrapper


def _make_is_layer_skipped_wrapper():
    def _wrapper(wrapped, instance, args, kwargs):
        sig = inspect.signature(wrapped)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments

        prefix = arguments['prefix']
        fused_mapping = arguments['fused_mapping']
        self = instance

        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = self.quant_description[shard_prefix +
                                                            '.weight'] in ["FLOAT", "BFLOAT16", "FLOAT16"]

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")
        else:
            is_skipped = self.quant_description[prefix + '.weight'] in ["FLOAT", "BFLOAT16", "FLOAT16"]

        assert is_skipped is not None
        return is_skipped
    return _wrapper


def _make_get_quant_method_wrapper():
    def _wrapper(wrapped, instance, args, kwargs):
        from vllm_ascend.quantization.quant_config import AscendKVCacheMethod
        from vllm.attention.layer import Attention

        self = instance
        sig = inspect.signature(wrapped)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments

        layer = arguments['layer']
        prefix = arguments['prefix']

        if isinstance(layer, Attention) and self.quant_description.get(
                'kv_quant_type') in ('C8', 'W8A16FP8'):
            return AscendKVCacheMethod(self, prefix)

        return wrapped(*args, **kwargs)
    return _wrapper


def _apply_all_quant_config_patches(m):
    """Single combined hook: wraps all four AscendQuantConfig methods at once."""
    wrapt.wrap_function_wrapper(
        'vllm_ascend.quantization.quant_config',
        'AscendQuantConfig.is_layer_skipped_ascend',
        _make_is_layer_skipped_wrapper(),
    )
    wrapt.wrap_function_wrapper(
        'vllm_ascend.quantization.quant_config',
        'AscendQuantConfig.get_quant_method',
        _make_get_quant_method_wrapper(),
    )
    wrapt.wrap_function_wrapper(
        'vllm_ascend.quantization.quant_config',
        'AscendLinearMethod.create_weights',
        _make_create_weights_wrapper(),
    )
    wrapt.wrap_function_wrapper(
        'vllm_ascend.quantization.quant_config',
        'AscendFusedMoEMethod.create_weights',
        _make_create_weights_wrapper(),
    )


def patch_AscendQuantConfig_is_layer_skipped_ascend():
    wrapt.register_post_import_hook(
        _apply_all_quant_config_patches,
        'vllm_ascend.quantization.quant_config'
    )


def patch_AscendQuantConfig_get_quant_method():
    # All four patches are applied together by _apply_all_quant_config_patches.
    # Registering the hook here is a no-op when is_layer_skipped_ascend already
    # registered it; wrapt deduplicates hooks registered on the same module.
    wrapt.register_post_import_hook(
        _apply_all_quant_config_patches,
        'vllm_ascend.quantization.quant_config'
    )


def patch_AscendLinearMethod_create_weights():
    wrapt.register_post_import_hook(
        _apply_all_quant_config_patches,
        'vllm_ascend.quantization.quant_config'
    )


def patch_AscendFusedMoEMethod_create_weights():
    wrapt.register_post_import_hook(
        _apply_all_quant_config_patches,
        'vllm_ascend.quantization.quant_config'
    )
