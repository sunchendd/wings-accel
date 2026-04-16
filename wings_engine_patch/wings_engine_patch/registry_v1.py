import sys
import threading
from dataclasses import dataclass
from typing import List, Tuple

from packaging.version import InvalidVersion, Version

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

def _build_vllm_v0_17_0_features():
    from wings_engine_patch.patch_vllm_container.v0_17_0 import (
        ears_patch,
        sparse_kv_patch,
    )

    return {
        "features": {
            "ears": [
                ears_patch.patch_vllm_ears,
            ],
            "sparse_kv": [
                sparse_kv_patch.patch_vllm_sparse_kv,
            ],
        }
    }


def _build_vllm_v0_19_0_features():
    from wings_engine_patch.patch_vllm_container.v0_19_0 import (
        ears_patch,
    )

    return {
        "features": {
            "ears": [
                ears_patch.patch_vllm_ears,
            ],
        }
    }


def _build_vllm_ascend_v0_17_0_features():
    from wings_engine_patch.patch_vllm_ascend_container.v0_17_0rc1 import (
        draft_model_patch,
        ears_patch,
    )

    return {
        "features": {
            "ears": [
                ears_patch.patch_vllm_ears,
            ],
            "draft_model": [
                draft_model_patch.patch_vllm_draft_model,
            ],
        }
    }


def _build_vllm_ascend_v0_18_0_features():
    from wings_engine_patch.patch_vllm_ascend_container.v0_18_0rc1 import (
        draft_model_patch,
        ears_patch,
    )

    return {
        "features": {
            "ears": [
                ears_patch.patch_vllm_ears,
            ],
            "draft_model": [
                draft_model_patch.patch_vllm_draft_model,
            ],
        }
    }


_ENGINE_ALIASES = {
    "vllm": "vllm",
    "vllm-ascend": "vllm-ascend",
    "vllm_ascend": "vllm-ascend",
}


def normalize_engine_name(inference_engine: str) -> str:
    return _ENGINE_ALIASES.get(inference_engine, inference_engine)

_registered_patches = {
    'vllm': {
        "0.17.0": {
            'is_default': False,
            'builder': _build_vllm_v0_17_0_features
        },
        "0.19.0": {
            'is_default': True,
            'builder': _build_vllm_v0_19_0_features
        }
    },
    'vllm-ascend': {
        "0.17.0rc1": {
            'is_default': False,
            'builder': _build_vllm_ascend_v0_17_0_features
        },
        "0.18.0rc1": {
            'is_default': True,
            'builder': _build_vllm_ascend_v0_18_0_features
        }
    }
}

_builder_lock = threading.Lock()


class PatchVersionError(RuntimeError):
    """Base class for runtime version-policy failures."""


class UnsupportedVersionError(PatchVersionError):
    """Requested a version that is explicitly unsupported."""


class ForwardCompatibilityPatchError(PatchVersionError):
    """Fallback patching for a newer, unvalidated version did not succeed."""


@dataclass(frozen=True)
class VersionSelection:
    requested_version: str
    resolved_version: str
    ver_specs: dict
    resolution_kind: str


def _get_default_version_spec(inference_engine: str, engine_specs: dict) -> tuple[str, dict]:
    for version_str, specs in engine_specs.items():
        if specs.get("is_default", False):
            return version_str, specs
    raise UnsupportedVersionError(
        f"Engine '{inference_engine}' has no default patch version configured."
    )


def _parse_registered_versions(
    inference_engine: str,
    engine_specs: dict,
) -> list[tuple[Version, str, dict]]:
    parsed_versions: list[tuple[Version, str, dict]] = []
    for version_str, specs in engine_specs.items():
        try:
            parsed_versions.append((Version(version_str), version_str, specs))
        except InvalidVersion as exc:
            raise UnsupportedVersionError(
                f"Registered patch version '{version_str}' for engine '{inference_engine}' "
                "is not PEP 440 compatible."
            ) from exc
    parsed_versions.sort(key=lambda item: item[0])
    return parsed_versions


def _select_version(inference_engine: str, requested_version: str, engine_specs: dict) -> VersionSelection:
    if requested_version in engine_specs:
        return VersionSelection(
            requested_version=requested_version,
            resolved_version=requested_version,
            ver_specs=engine_specs[requested_version],
            resolution_kind="exact",
        )

    try:
        requested = Version(requested_version)
    except InvalidVersion as exc:
        raise UnsupportedVersionError(
            f"Requested version '{requested_version}' for engine '{inference_engine}' is not a "
            "valid PEP 440 version."
        ) from exc

    parsed_versions = _parse_registered_versions(inference_engine, engine_specs)
    if not parsed_versions:
        raise UnsupportedVersionError(
            f"Engine '{inference_engine}' has no registered patch versions."
        )

    min_supported, min_version_str, _ = parsed_versions[0]
    max_supported, max_version_str, _ = parsed_versions[-1]

    if (
        not requested.is_prerelease
        and max_supported.is_prerelease
        and requested.release == max_supported.release
    ):
        raise UnsupportedVersionError(
            f"Requested version '{requested_version}' for engine '{inference_engine}' is not a "
            f"validated patched version. Supported versions: {sorted(engine_specs.keys())}."
        )

    if requested < min_supported:
        raise UnsupportedVersionError(
            f"Requested version '{requested_version}' for engine '{inference_engine}' is older "
            f"than the minimum supported patched version '{min_version_str}'. Historical versions "
            "are not supported."
        )

    if requested > max_supported:
        resolved_version, ver_specs = _get_default_version_spec(inference_engine, engine_specs)
        print(
            f"[Wings Engine Patch] Warning: Requested version '{requested_version}' is newer than "
            f"highest validated version '{max_version_str}' for engine '{inference_engine}'. "
            f"Trying default patch set '{resolved_version}'.",
            file=sys.stderr,
        )
        return VersionSelection(
            requested_version=requested_version,
            resolved_version=resolved_version,
            ver_specs=ver_specs,
            resolution_kind="future_fallback",
        )

    raise UnsupportedVersionError(
        f"Requested version '{requested_version}' for engine '{inference_engine}' is not a "
        f"validated patched version. Supported versions: {sorted(engine_specs.keys())}."
    )


def _ensure_features_loaded(ver_specs):
    """Executes the builder if features are not yet loaded. Thread-safe."""
    if 'features' in ver_specs:
        return
    with _builder_lock:
        # Double-checked locking: another thread may have built while we waited
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


def enable(inference_engine: str, features: List[str], version: str) -> List[Tuple[str, Exception]]:
    """Enable patches for the given engine/version/features.

    Returns a list of ``(patch_name, exception)`` tuples for every patch that
    failed to apply, so callers can surface structured warnings.  An empty list
    means all patches applied successfully.
    """
    failures: List[Tuple[str, Exception]] = []

    canonical_engine_name = normalize_engine_name(inference_engine)
    engine_specs = _registered_patches.get(canonical_engine_name)
    if not engine_specs:
        print(f"[Wings Engine Patch] Warning: Engine '{inference_engine}' is not registered.", file=sys.stderr)
        return failures

    selection = _select_version(canonical_engine_name, version, engine_specs)
    ver_specs = selection.ver_specs
    used_version = selection.resolved_version

    # Load features/imports lazily
    try:
        _ensure_features_loaded(ver_specs)
    except ImportError as e:
        if selection.resolution_kind == "future_fallback":
            raise ForwardCompatibilityPatchError(
                f"Requested version '{version}' is newer than the validated patch set. "
                f"Tried default patch set '{used_version}', but loading patches failed: {e}"
            ) from e
        print(f"[Wings Engine Patch] Error loading patches for {inference_engine}@{used_version}: {e}", file=sys.stderr)
        return failures

    # 1. Expand feature set based on shared patches
    expanded_features = _expand_features_by_shared_patches(ver_specs, features)
    
    if len(expanded_features) > len(set(features)):
        print(f"[Wings Engine Patch] Feature Set Expanded: {features} -> {list(expanded_features)} due to shared patches.", file=sys.stderr)
    
    all_selected_patches = set()
    
    feature_map = ver_specs.get('features', {})
    for feat in expanded_features:
        if feat in feature_map:
            for patch_func in feature_map[feat]:
                all_selected_patches.add(patch_func)
        else:
             print(f"[Wings Engine Patch] Warning: Feature '{feat}' not found in registry for {inference_engine}@{used_version}.", file=sys.stderr)

    # Apply patches in deterministic order
    sorted_patches = sorted(list(all_selected_patches), key=lambda f: (f.__module__, f.__name__))

    for patch_func in sorted_patches:
        try:
            patch_func()
        except Exception as e:
            patch_name = f"{patch_func.__module__}.{patch_func.__name__}"
            print(f"[Wings Engine Patch] Error executing patch {patch_name}: {e}", file=sys.stderr)
            failures.append((patch_name, e))

    if selection.resolution_kind == "future_fallback" and failures:
        failed_patch_names = ", ".join(patch_name for patch_name, _ in failures)
        raise ForwardCompatibilityPatchError(
            f"Requested version '{version}' is newer than the validated patch set. "
            f"Tried default patch set '{used_version}', but patching failed: {failed_patch_names}"
        ) from failures[0][1]

    return failures
