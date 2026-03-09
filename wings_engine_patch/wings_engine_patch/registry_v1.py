import sys
from typing import List

# Structure: 
# {
#   'engine_name': {
#       'version_string': {
#           'is_default': bool, 
#           'builder': function that returns {'features': ..., 'non_propagating_patches': ...},
#           # OR pre-hydrated:
#           'features': { ... },
#           'non_propagating_patches': { ... }
#       }
#   }
# }

# --- Feature Builders ---
def _build_vllm_ascend_v0_12_0rc1_features():
    """Builder for vllm_ascend 0.12.0rc1 features."""
    
    # Import specific sub-modules directly
    from wings_engine_patch.patch_vllm_ascend_container.v0_12_0rc1.vllm_ascend.quantization import patch_utils
    from wings_engine_patch.patch_vllm_ascend_container.v0_12_0rc1.vllm_ascend.quantization import patch_quant_config
    from wings_engine_patch.patch_vllm_ascend_container.v0_12_0rc1.vllm_ascend.ops.fused_moe import patch_moe_comm_method
    from wings_engine_patch.patch_vllm_ascend_container.v0_12_0rc1.vllm_ascend.ops.fused_moe import patch_moe_mlp

    SOFT_FP8_SPECIFIC = [
        patch_utils.patch_ASCEND_QUANTIZATION_METHOD_MAP,
        patch_quant_config.patch_AscendQuantConfig_is_layer_skipped_ascend,
        patch_quant_config.patch_AscendQuantConfig_get_quant_method,
        patch_quant_config.patch_AscendLinearMethod_create_weights,
        patch_quant_config.patch_AscendFusedMoEMethod_create_weights,
        patch_moe_comm_method.patch_moe_comm_method_fused_experts,
        patch_moe_mlp.patch_moe_mlp_functions
    ]

    SOFT_FP4_SPECIFIC = [
         patch_utils.patch_ASCEND_QUANTIZATION_METHOD_MAP,
         patch_quant_config.patch_AscendQuantConfig_is_layer_skipped_ascend,
         patch_quant_config.patch_AscendQuantConfig_get_quant_method,
         patch_quant_config.patch_AscendLinearMethod_create_weights,
    ]
    
    # Properties scoped to THIS specific engine version
    non_propagating_patches = set()

    return {
        'features': {
            'soft_fp8': SOFT_FP8_SPECIFIC,
            'soft_fp4': SOFT_FP4_SPECIFIC,
        },
        'non_propagating_patches': non_propagating_patches
    }

def _build_vllm_v0_12_0_empty_features():
    from wings_engine_patch.patch_vllm_container.v0_12_0_empty import test_patch
    
    return {
        'features': {
            'test_patch': [test_patch.patch_vllm_report_kv_cache_config]
        }
    }

_registered_patches = {
    'vllm_ascend': {
        "0.12.0rc1": {
            'is_default': True,
            'builder': _build_vllm_ascend_v0_12_0rc1_features
        }
    },
    'vllm': {
        "0.12.0+empty": {
            'is_default': True,
            'builder': _build_vllm_v0_12_0_empty_features
        }
    }
}

def _ensure_features_loaded(ver_specs):
    """Executes the builder if features are not yet loaded."""
    if 'features' not in ver_specs and 'builder' in ver_specs:
        ver_specs.update(ver_specs['builder']())

def _expand_features_by_shared_patches(ver_specs, selected_features):
    """
    Automatically enable features that share patches with the selected features.
    Only considers patches NOT in 'non_propagating_patches' set.
    """
    if not ver_specs:
        return set(selected_features)

    feature_map = ver_specs.get('features', {})
    non_propagating = ver_specs.get('non_propagating_patches', set())
    
    def should_propagate(patch_func):
        return patch_func not in non_propagating

    def get_propagating_patches_for_feature(feat):
        patches = []
        for p in feature_map.get(feat, []):
            if should_propagate(p):
                patches.append(p)
        return set(patches)

    patch_to_features = {}
    for feat, patches in feature_map.items():
        for p in patches:
            if should_propagate(p):
                # p is now the function object itself, which is hashable
                if p not in patch_to_features:
                    patch_to_features[p] = set()
                patch_to_features[p].add(feat)

    current_features = set(selected_features)
    
    while True:
        original_size = len(current_features)
        
        active_patches = set()
        for f in current_features:
            active_patches.update(get_propagating_patches_for_feature(f))
            
        for patch in active_patches:
            linked_features = patch_to_features.get(patch, set())
            current_features.update(linked_features)
            
        if len(current_features) == original_size:
            break
            
    return current_features


def enable(inference_engine: str, features: List[str], version: str):
    engine_specs = _registered_patches.get(inference_engine)
    if not engine_specs:
         print(f"[Wings Engine Patch] Warning: Engine '{inference_engine}' is not registered.", file=sys.stderr)
         return

    # Resolve Version and Hydrate Specs
    ver_specs = engine_specs.get(version)
    used_version = version 
    
    if not ver_specs:
        for ver, specs in engine_specs.items():
            if specs.get('is_default', False):
                ver_specs = specs
                used_version = ver
                print(f"[Wings Engine Patch] Info: Version mismatch ({version} requested). Using default version '{used_version}' (Engine: {inference_engine}).", file=sys.stderr)
                break
    
    if not ver_specs:
         print(f"[Wings Engine Patch] Warning: Version '{version}' (and no default) not found in registry for {inference_engine}.", file=sys.stderr)
         return

    # Load features/imports lazily
    try:
        _ensure_features_loaded(ver_specs)
    except ImportError as e:
        print(f"[Wings Engine Patch] Error loading patches for {inference_engine}@{used_version}: {e}", file=sys.stderr)
        return

    # 1. Expand feature set based on shared patches
    expanded_features = _expand_features_by_shared_patches(ver_specs, features)
    
    if len(expanded_features) > len(set(features)):
        print(f"[Wings Engine Patch] Feature Set Expanded: {features} -> {list(expanded_features)} due to shared patches.", file=sys.stderr)
    
    all_selected_patches = set()
    
    feature_map = ver_specs.get('features', {})
    for feat in expanded_features:
        if feat in feature_map:
            # Add all patch functions directly
            for patch_func in feature_map[feat]:
                all_selected_patches.add(patch_func)
        else:
             print(f"[Wings Engine Patch] Warning: Feature '{feat}' not found in registry for {inference_engine}@{used_version}.", file=sys.stderr)

    # 3. Apply Patches
    # Convert to list for execution
    # Since function objects usually don't have a stable sort order by default that persists across runs without name,
    # we can sort by module+name for deterministic execution order.
    
    sorted_patches = sorted(list(all_selected_patches), key=lambda f: (f.__module__, f.__name__))

    for patch_func in sorted_patches:
        try:
            # print(f"Applying patch: {patch_func.__module__}.{patch_func.__name__}")
            patch_func()
        except Exception as e:
             print(f"[Wings Engine Patch] Error executing patch {patch_func.__name__}: {e}", file=sys.stderr)
