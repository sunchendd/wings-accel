import os
import json
import sys
import traceback


try:
    env_options = os.environ.get("WINGS_ENGINE_PATCH_OPTIONS")

    if env_options:
        from .registry import enable
        from .registry_v1 import normalize_engine_name

        def _find_duplicate_engine_aliases(engine_configs):
            grouped_keys = {}
            for engine_name in engine_configs:
                canonical_engine_name = normalize_engine_name(engine_name)
                grouped_keys.setdefault(canonical_engine_name, []).append(engine_name)
            return {
                canonical_engine_name: alias_keys
                for canonical_engine_name, alias_keys in grouped_keys.items()
                if len(alias_keys) > 1
            }

        try:
            # Expected format (versioned dict):
            # {"vllm_ascend": {"version": "0.12.0", "features": ["soft_fp8"]}, ...}
            options = json.loads(env_options)

            if isinstance(options, dict):
                duplicate_aliases = _find_duplicate_engine_aliases(options)
                if duplicate_aliases:
                    duplicates = ", ".join(
                        f"{canonical_engine_name}: {sorted(alias_keys)}"
                        for canonical_engine_name, alias_keys in sorted(duplicate_aliases.items())
                    )
                    print(
                        "[Wings Engine Patch] Error: duplicate engine alias keys are not allowed in "
                        f"WINGS_ENGINE_PATCH_OPTIONS: {duplicates}",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                for engine_key, config in options.items():
                    # Only support Dictionary with version info
                    if isinstance(config, dict):
                        version = config.get("version")
                        features = config.get("features", [])

                        if not version:
                            print(f"[Wings Engine Patch] Warning: Configuration for engine \'{engine_key}\' is missing \'version\'. Ignoring.", file=sys.stderr)
                            continue

                        if isinstance(features, list) and features:
                            canonical_engine_key = normalize_engine_name(engine_key)
                            failures = enable(canonical_engine_key, features, version=version)
                            for patch_name, exc in failures:
                                print(f"[Wings Engine Patch] Patch failed — {patch_name}: {exc}", file=sys.stderr)
                    else:
                        print(f"[Wings Engine Patch] Warning: Configuration for engine \'{engine_key}\' must be a dictionary with \'version\' and \'features\'. Ignoring.", file=sys.stderr)
            else:
                print("[Wings Engine Patch] Warning: WINGS_ENGINE_PATCH_OPTIONS must be a JSON dictionary.", file=sys.stderr)

        except json.JSONDecodeError as e:
            print(f"[Wings Engine Patch] Warning: Failed to parse WINGS_ENGINE_PATCH_OPTIONS: {e}", file=sys.stderr)
except Exception:
    print("[Wings Engine Patch] Critical Error during auto-patching:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    # pylint: disable=avoid-using-exit
    raise SystemExit(1) from None
    # pylint: enable=avoid-using-exit
