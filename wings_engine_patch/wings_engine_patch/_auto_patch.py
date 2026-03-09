import os
import json
import sys
import traceback


try:
    env_options = os.environ.get("WINGS_ENGINE_PATCH_OPTIONS")

    if env_options:
        from .registry import enable
        
        try:
            # Expected format (versioned dict):
            # {"vllm_ascend": {"version": "0.12.0", "features": ["soft_fp8"]}, ...}
            options = json.loads(env_options)
            
            if isinstance(options, dict):
                for engine_key, config in options.items():
                    # Only support Dictionary with version info
                    if isinstance(config, dict):
                        version = config.get("version")
                        features = config.get("features", [])
                        
                        if not version:
                            print(f"[Wings Engine Patch] Warning: Configuration for engine '{engine_key}' is missing 'version'. Ignoring.", file=sys.stderr)
                            continue

                        if isinstance(features, list) and features:
                            enable(engine_key, features, version=version)
                    else:
                        print(f"[Wings Engine Patch] Warning: Configuration for engine '{engine_key}' must be a dictionary with 'version' and 'features'. Ignoring.", file=sys.stderr)
            else:
                print("[Wings Engine Patch] Warning: WINGS_ENGINE_PATCH_OPTIONS must be a JSON dictionary.", file=sys.stderr)
                
        except json.JSONDecodeError as e:
            print(f"[Wings Engine Patch] Warning: Failed to parse WINGS_ENGINE_PATCH_OPTIONS: {e}", file=sys.stderr)
except Exception:
    print("[Wings Engine Patch] Critical Error during auto-patching:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
