import wrapt


def patch_AscendQuantConfig_is_layer_skipped_ascend():
    def _wrapper(wrapped, instance, args, kwargs):
        prefix, fused_mapping = args
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

    wrapt.register_post_import_hook(
        lambda m: wrapt.wrap_function_wrapper(
            'vllm_ascend.quantization.quant_config',
            'AscendQuantConfig.is_layer_skipped_ascend',
            _wrapper
        ),
        'vllm_ascend.quantization.quant_config'
    )


def patch_AscendQuantConfig_get_quant_method():
    def _wrapper(wrapped, instance, args, kwargs):
        # Lazy import to avoid top-level dependency issues
        from vllm_ascend.quantization.quant_config import AscendKVCacheMethod
        from vllm.attention.layer import Attention
        
        self = instance
        if len(args) == 2:
            layer, prefix = args
        elif len(args) == 1 and 'prefix' in kwargs:
            layer = args[0]
            prefix = kwargs['prefix']
        elif len(args) == 0 and 'layer' in kwargs and 'prefix' in kwargs:
            layer = kwargs['layer']
            prefix = kwargs['prefix']
        else:
            raise ValueError("Invalid arguments")

        if isinstance(layer, Attention) and self.quant_description.get(
                'kv_quant_type') in ('C8', 'W8A16FP8'):
            return AscendKVCacheMethod(self, prefix)
            
        return wrapped(*args, **kwargs)

    wrapt.register_post_import_hook(
        lambda m: wrapt.wrap_function_wrapper(
            'vllm_ascend.quantization.quant_config',
            'AscendQuantConfig.get_quant_method',
            _wrapper
        ),
        'vllm_ascend.quantization.quant_config'
    )


def patch_AscendLinearMethod_create_weights():
    def _wrapper(wrapped, instance, args, kwargs):
        self = instance
        if hasattr(self.quant_method, 'create_weights'):
            self.quant_method.create_weights(*args, **kwargs)
            return
        return wrapped(*args, **kwargs)

    wrapt.register_post_import_hook(
        lambda m: wrapt.wrap_function_wrapper(
            'vllm_ascend.quantization.quant_config',
            'AscendLinearMethod.create_weights',
            _wrapper
        ),
        'vllm_ascend.quantization.quant_config'
    )


def patch_AscendFusedMoEMethod_create_weights():
    def _wrapper(wrapped, instance, args, kwargs):
        self = instance
        if hasattr(self.quant_method, 'create_weights'):
            self.quant_method.create_weights(*args, **kwargs)
            return
        return wrapped(*args, **kwargs)

    wrapt.register_post_import_hook(
        lambda m: wrapt.wrap_function_wrapper(
            'vllm_ascend.quantization.quant_config',
            'AscendFusedMoEMethod.create_weights',
            _wrapper
        ),
        'vllm_ascend.quantization.quant_config'
    )
