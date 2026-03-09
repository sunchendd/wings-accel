import sys
import wrapt

def patch_vllm_report_kv_cache_config():
    def _report_kv_cache_config_wrapper(wrapped, instance, args, kwargs):
        print('[Vllm Patch] vllm.v1.core.kv_cache_utils._report_kv_cache_config called', file=sys.stderr)
        return wrapped(*args, **kwargs)

    def _apply_patch(module):
        try:
            wrapt.wrap_function_wrapper(
                module,
                '_report_kv_cache_config',
                _report_kv_cache_config_wrapper
            )
        except Exception as e:
            print(f"[Wings Engine Patch] Failed to hook vllm.v1.core.kv_cache_utils._report_kv_cache_config: {e}", file=sys.stderr)

    wrapt.register_post_import_hook(
        _apply_patch,
        'vllm.v1.core.kv_cache_utils'
    )
