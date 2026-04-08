#!/usr/bin/env python3
"""
wings-accel Feature Installation CLI

Implements the CLI contract defined in §3.3.6.2.1 of the wings-infer design document.

Usage:
    python3 install.py --list
    python3 install.py --features '{"vllm":{"version":"0.17.0","features":["ears"]}}'
    python3 install.py --features '{"vllm":{"version":"0.17.0","features":["ears"]}}' --dry-run
    python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode","ears"]}}'

The --features JSON format:
    {
        "<engine_name>": {
            "version": "<version_string>",
            "features": ["<feature_name>", ...]
        }
    }

Example:
    python3 install.py --features '{"vllm":{"version":"0.17.0","features":["ears"]}}'
    export WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.17.0","features":["ears"]}}'

Deployment layout (all files at the same level as install.py):
    install.py
    supported_features.json
    wings_engine_patch-*.whl
    wrapt-*-linux_x86_64.whl
    wrapt-*-linux_aarch64.whl
    arctic_inference-*.whl      (pre-built wheel, installed offline)
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import logging
import platform
import re
import subprocess
import sys
from pathlib import Path


class _StreamProxy:
    def __init__(self, stream_name: str):
        self._stream_name = stream_name

    def write(self, message: str) -> int:
        return getattr(sys, self._stream_name).write(message)

    def flush(self) -> None:
        getattr(sys, self._stream_name).flush()


def _build_logger(name: str, stream_name: str) -> logging.Logger:
    configured_logger = logging.getLogger(name)
    if configured_logger.handlers:
        return configured_logger
    handler = logging.StreamHandler(_StreamProxy(stream_name))
    handler.setFormatter(logging.Formatter("%(message)s"))
    configured_logger.addHandler(handler)
    configured_logger.setLevel(logging.INFO)
    configured_logger.propagate = False
    return configured_logger


logger = _build_logger(__name__, "stdout")
stderr_logger = _build_logger("stderr", "stderr")

# Root-level supported_features.json is the CLI/MaaS-facing source of truth.
_BASE_DIR = Path(__file__).resolve().parent
_SUPPORTED_FEATURES_PATH = _BASE_DIR / "supported_features.json"
# Deployment layout: all files (whl, source tarballs) sit next to install.py.
# Source-tree fallback: build/output/ (used during local development).
_LOCAL_WHEEL_DIR = _BASE_DIR if _BASE_DIR.name == "output" else _BASE_DIR / "build" / "output"

# Map engine names to pyproject.toml [optional-dependencies] extras keys.
_ENGINE_TO_EXTRAS = {
    "vllm": "vllm",
}


# ---------------------------------------------------------------------------
# supported_features.json helpers
# ---------------------------------------------------------------------------

def load_supported_features() -> dict:
    if not _SUPPORTED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"supported_features.json not found at {_SUPPORTED_FEATURES_PATH}")
    with open(_SUPPORTED_FEATURES_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_schema(data: dict) -> None:
    """Validate supported_features.json per §3.3.6.3.3.

    Rules:
    1. schema_version, updated_at, engines are required.
    2. Each engine must have at least one version.
    3. Each engine must have exactly one is_default:true version.
    """
    required_top = {"schema_version", "updated_at", "engines"}
    missing = required_top - data.keys()
    if missing:
        raise ValueError(f"supported_features.json missing required top-level fields: {missing}")

    for engine_name, engine_def in data["engines"].items():
        versions = engine_def.get("versions", {})
        if not versions:
            raise ValueError(f"Engine '{engine_name}' has no versions defined.")

        defaults = [v for v, spec in versions.items() if spec.get("is_default", False)]
        if len(defaults) == 0:
            raise ValueError(
                f"Engine '{engine_name}' has no default version (is_default: true). "
                "Exactly one version must be marked as default."
            )
        if len(defaults) > 1:
            raise ValueError(
                f"Engine '{engine_name}' has {len(defaults)} default versions: {defaults}. "
                "Exactly one version must be marked as default."
            )


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------

def _get_default_version_spec(engine_name: str, versions: dict) -> tuple[str, dict]:
    for ver, spec in versions.items():
        if spec.get("is_default", False):
            return ver, spec
    raise ValueError(
        f"Engine '{engine_name}' has no default version (is_default: true). "
        "Exactly one version must be marked as default."
    )


class _FallbackInvalidVersion(ValueError):
    pass


@functools.total_ordering
class _FallbackVersion:
    def __init__(self, version: str):
        if not re.fullmatch(r"\d+(?:\.\d+)*", version):
            raise _FallbackInvalidVersion(version)
        parts = [int(part) for part in version.split(".")]
        while parts and parts[-1] == 0:
            parts.pop()
        self._parts = tuple(parts)

    def __lt__(self, other):
        if not isinstance(other, _FallbackVersion):
            return NotImplemented
        return self._parts < other._parts

    def __eq__(self, other):
        if not isinstance(other, _FallbackVersion):
            return NotImplemented
        return self._parts == other._parts


def _get_packaging_version_types():
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return _FallbackVersion, _FallbackInvalidVersion
    return Version, InvalidVersion


def _parse_supported_versions(engine_name: str, versions: dict) -> list[tuple[object, str, dict]]:
    Version, InvalidVersion = _get_packaging_version_types()
    parsed_versions: list[tuple[object, str, dict]] = []
    for version_str, spec in versions.items():
        try:
            parsed_versions.append((Version(version_str), version_str, spec))
        except InvalidVersion as exc:
            raise ValueError(
                f"Engine '{engine_name}' declares unsupported version string '{version_str}' "
                "that is not PEP 440 compatible."
            ) from exc
    parsed_versions.sort(key=lambda item: item[0])
    return parsed_versions


def _classify_requested_version(
    engine_name: str,
    requested_version: str,
    versions: dict,
) -> tuple[str, str, dict]:
    Version, InvalidVersion = _get_packaging_version_types()
    if requested_version in versions:
        return "exact", requested_version, versions[requested_version]

    try:
        requested = Version(requested_version)
    except InvalidVersion as exc:
        raise ValueError(
            f"Version '{requested_version}' for engine '{engine_name}' is not a valid PEP 440 version."
        ) from exc

    parsed_versions = _parse_supported_versions(engine_name, versions)
    if not parsed_versions:
        raise ValueError(f"Engine '{engine_name}' has no versions defined.")

    min_supported, min_version_str, _ = parsed_versions[0]
    max_supported, max_version_str, _ = parsed_versions[-1]

    if requested < min_supported:
        raise ValueError(
            f"Version '{requested_version}' for engine '{engine_name}' is older than the minimum "
            f"supported patched version '{min_version_str}'. Historical versions are not supported."
        )

    if requested > max_supported:
        default_version, default_spec = _get_default_version_spec(engine_name, versions)
        return "future_fallback", default_version, default_spec

    raise ValueError(
        f"Version '{requested_version}' for engine '{engine_name}' is not a validated patched version. "
        f"Supported versions: {sorted(versions.keys())}."
    )


def resolve_version(engine_name: str, requested_version: str, engine_spec: dict):
    """Resolve version with explicit old-version rejection and future fallback.

    Returns (resolved_version_str, version_spec_dict).
    """
    versions = engine_spec.get("versions", {})
    resolution_kind, resolved_version, version_spec = _classify_requested_version(
        engine_name,
        requested_version,
        versions,
    )

    if resolution_kind == "future_fallback":
        supported_versions = _parse_supported_versions(engine_name, versions)
        highest_validated_version = supported_versions[-1][1]
        stderr_logger.warning(
            f"[wings-accel] Warning: version '{requested_version}' is newer than the highest "
            f"validated version '{highest_validated_version}' for engine '{engine_name}'. "
            f"Trying default version '{resolved_version}'."
        )

    return resolved_version, version_spec


def parse_requested_install(raw_features_json: str, manifest: dict) -> tuple[str, str, list[str]]:
    """Parse and validate a public installer request.

    Returns:
        tuple[str, str, list[str]]: (engine_name, resolved_version, requested_features)
    """
    try:
        features_config = json.loads(raw_features_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--features is not valid JSON: {exc}") from exc

    if not isinstance(features_config, dict):
        raise ValueError("--features must be a JSON object.")

    if len(features_config) != 1:
        raise ValueError("--features must contain exactly one top-level engine.")

    engine_name, config = next(iter(features_config.items()))
    if not isinstance(config, dict):
        raise ValueError(
            f"config for '{engine_name}' must be a JSON object with 'version' and 'features' keys."
        )

    engines_spec = manifest["engines"]
    if engine_name not in engines_spec:
        raise ValueError(
            f"engine '{engine_name}' is not listed in supported_features.json. "
            f"Available engines: {list(engines_spec.keys())}"
        )

    known_engine_config_keys = {"version", "features"}
    unknown_keys = set(config.keys()) - known_engine_config_keys
    if unknown_keys:
        raise ValueError(
            f"unknown keys {sorted(unknown_keys)} in config for '{engine_name}'. "
            f"Expected keys: {sorted(known_engine_config_keys)}."
        )

    if "version" not in config:
        raise ValueError(f"'version' is required for engine '{engine_name}'.")
    requested_version = config["version"]
    if not isinstance(requested_version, str):
        raise ValueError(f"'version' for engine '{engine_name}' must be a string.")
    if not requested_version:
        raise ValueError(f"'version' is required for engine '{engine_name}'.")

    if "features" not in config:
        raise ValueError(f"'features' is required for engine '{engine_name}'.")

    requested_features = config["features"]
    if not isinstance(requested_features, list):
        raise ValueError(f"'features' for engine '{engine_name}' must be a list.")
    if not requested_features:
        raise ValueError(f"'features' for engine '{engine_name}' must be a non-empty list.")
    if any(not isinstance(feature_name, str) for feature_name in requested_features):
        raise ValueError(f"'features' for engine '{engine_name}' must only contain strings.")

    resolved_version, version_spec = resolve_version(engine_name, requested_version, engines_spec[engine_name])
    available_features = set(version_spec.get("features", {}).keys())
    unknown_features = sorted(set(requested_features) - available_features)
    if unknown_features:
        raise ValueError(
            f"features {unknown_features} are not publicly supported for "
            f"{engine_name}@{resolved_version}. Available public features: {sorted(available_features)}."
        )

    return engine_name, resolved_version, requested_features


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _find_local_whl() -> Path | None:
    """Return the wings_engine_patch .whl in _BASE_DIR (deployment layout).

    Falls back to build/output/ for local development (source-tree layout).
    """
    for search_dir in (_BASE_DIR, _LOCAL_WHEEL_DIR):
        if search_dir.exists():
            whls = [p for p in search_dir.glob("*.whl") if "wings_engine_patch" in p.name]
            if whls:
                return max(whls, key=lambda p: p.stat().st_mtime)
    return None


def _find_local_wheel_by_prefix(prefix: str) -> Path | None:
    for search_dir in (_LOCAL_WHEEL_DIR, _BASE_DIR):
        if not search_dir.exists():
            continue

        matches = list(search_dir.glob(f"{prefix}-*.whl"))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def _install_local_feature_wheels(features: list[str], dry_run: bool = False) -> None:
    feature_wheels = {}

    for feature_name in features:
        package_name, wheel_path = feature_wheels.get(feature_name, (None, None))
        if package_name is None:
            continue
        if wheel_path is None:
            raise FileNotFoundError(
                f"Feature '{feature_name}' requires local wheel '{package_name}-*.whl', "
                f"but none was found in {_LOCAL_WHEEL_DIR}."
            )

        cmd = [sys.executable, "-m", "pip", "install", str(wheel_path), "--force-reinstall"]
        if dry_run:
            logger.info("[dry-run] Would run: %s", " ".join(str(c) for c in cmd))
            continue

        logger.info(
            "[wings-accel] Installing feature dependency '%s' from %s ...",
            package_name,
            wheel_path.name,
        )
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            stderr_logger.error(
                "[wings-accel] Error: failed to install feature dependency '%s' (exit %s).",
                package_name,
                e.returncode,
            )
            raise


def _find_arctic_inference_whl() -> Path | None:
    for search_dir in (_LOCAL_WHEEL_DIR, _BASE_DIR):
        if not search_dir.exists():
            continue
        for pattern in (
            "arctic_inference-*.whl",
            "arctic-inference-*.whl",
        ):
            matches = list(search_dir.glob(pattern))
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def _is_arctic_inference_installed() -> bool:
    return _is_module_available("arctic_inference")


def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _has_local_runtime_deps() -> bool:
    return _is_module_available("wrapt") and _is_module_available("packaging")


def _install_local_dependency(
    package_name: str,
    module_name: str,
    wheel_path: Path | None,
    dry_run: bool = False,
    *,
    no_deps: bool = False,
    missing_ok: bool = False,
) -> None:
    if _is_module_available(module_name):
        logger.info(f"[wings-accel] {package_name} already installed, skipping.")
        return

    if wheel_path is None:
        message = (
            f"[wings-accel] {package_name} wheel not found in {_LOCAL_WHEEL_DIR} or {_BASE_DIR}, "
            "skipping."
        )
        if dry_run:
            logger.info(f"[dry-run] {message}")
            return
        if missing_ok:
            logger.info(message)
            return
        raise FileNotFoundError(message)

    cmd = [sys.executable, "-m", "pip", "install", str(wheel_path)]
    if no_deps:
        cmd.append("--no-deps")

    if dry_run:
        logger.info(f"[dry-run] Would run: {' '.join(str(c) for c in cmd)}")
        return

    logger.info(f"[wings-accel] Installing {package_name} from {wheel_path.name} ...")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        stderr_logger.error(
            f"[wings-accel] Error: failed to install {package_name} (exit {exc.returncode})."
        )
        raise


def install_runtime_dependencies(dry_run: bool = False) -> None:
    runtime_dependencies = (
        ("wrapt", "wrapt", _find_local_wheel_by_prefix("wrapt"), False),
        ("packaging", "packaging", _find_local_wheel_by_prefix("packaging"), False),
    )

    for package_name, module_name, wheel_path, missing_ok in runtime_dependencies:
        _install_local_dependency(
            package_name,
            module_name,
            wheel_path,
            dry_run=dry_run,
            missing_ok=missing_ok,
        )

    _install_arctic_inference(dry_run=dry_run)


def _install_arctic_inference(dry_run: bool = False) -> None:
    """Install arctic-inference from the local pre-built wheel (offline, no compilation).

    Skipped when:
      - arctic-inference is already installed (e.g. vllm-ascend containers ship it), or
      - the host architecture is not x86_64 (the pre-built wheel only targets x86_64).
    """
    arch = platform.machine()
    if arch != "x86_64":
        logger.info(
            f"[wings-accel] Skipping arctic-inference install: architecture is {arch} (x86_64 only)."
        )
        return

    _install_local_dependency(
        "arctic-inference",
        "arctic_inference",
        _find_arctic_inference_whl(),
        dry_run=dry_run,
        no_deps=True,
        missing_ok=True,
    )


def install_engine(
    engine_name: str,
    version: str,
    features: list,
    dry_run: bool = False,
) -> None:
    """Run pip install for the given engine's extras group.

    Supports fully offline installation:
      - When a local .whl is found, uses --find-links + --no-index so pip
        resolves ALL dependencies (e.g. wrapt) from the same local directory.
      - Falls back to --no-deps when all runtime deps are already importable.
      - Falls back to PyPI when no local wheel is available.
    """
    _install_local_feature_wheels(features, dry_run=dry_run)

    extras = _ENGINE_TO_EXTRAS.get(engine_name, engine_name)
    local_whl = _find_local_wheel_by_prefix("wings_engine_patch") or _find_local_whl()

    if local_whl:
        pkg = f"{local_whl}[{extras}]"
        local_dir = str(local_whl.parent)
        if _has_local_runtime_deps():
            cmd = [sys.executable, "-m", "pip", "install", pkg, "--no-deps"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", pkg, "--no-index", "--find-links", local_dir]
    else:
        pkg = f"wings_engine_patch[{extras}]"
        cmd = [sys.executable, "-m", "pip", "install", pkg]

    if dry_run:
        logger.info(f"[dry-run] Would run: {' '.join(str(c) for c in cmd)}")
        _print_env_hint(engine_name, version, features, dry_run=True)
        return

    logger.info(f"[wings-accel] Installing for engine '{engine_name}' (extras: [{extras}]) ...")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        if local_whl and "--no-index" in cmd:
            stderr_logger.warning(
                "[wings-accel] Offline install failed (missing dependency wheels?), "
                "retrying with --no-deps ..."
            )
            cmd_fallback = [sys.executable, "-m", "pip", "install", pkg, "--no-deps"]
            try:
                subprocess.check_call(cmd_fallback)
            except subprocess.CalledProcessError as e2:
                stderr_logger.error(
                    f"[wings-accel] Error: pip install failed (exit {e2.returncode})."
                )
                raise
        else:
            stderr_logger.error(f"[wings-accel] Error: pip install failed (exit {e.returncode}).")
            raise

    _print_env_hint(engine_name, version, features)


def _print_env_hint(engine_name: str, version: str, features: list, dry_run: bool = False) -> None:
    patch_options = {engine_name: {"version": version, "features": features}}
    prefix = "[dry-run] " if dry_run else ""
    logger.info(
        f"\n{prefix}[wings-accel] ✅ Done. To enable patches at runtime, set:\n"
        f"  export WINGS_ENGINE_PATCH_OPTIONS='{json.dumps(patch_options)}'"
    )


# ---------------------------------------------------------------------------
# Developer self-validation (--check)
# ---------------------------------------------------------------------------

def check_installed(
    engine_name: str,
    version: str,
    features: list
) -> bool:
    """Verify that wings_engine_patch is installed and the requested engine/features
    are registered in the patch registry.

    Returns:
        bool: True if all checks pass, False otherwise
    """
    logger.info(f"[wings-accel] Checking {engine_name}@{version} features: {features}")

    # 1. package installed?
    try:
        import wings_engine_patch  # noqa: F401
        import wings_engine_patch.registry_v1 as reg_v1

        logger.info("  ✅ wings_engine_patch installed")
    except ImportError as e:
        stderr_logger.error(f"  ❌ wings_engine_patch not installed: {e}")
        return False

    # 2. engine registered?
    registry = getattr(reg_v1, "_registered_patches", {})
    if engine_name not in registry:
        stderr_logger.error(f"  ❌ Engine '{engine_name}' not found in patch registry.")
        return False
    logger.info(f"  ✅ Engine '{engine_name}' registered in patch registry")

    # 3. version policy
    engine_versions = registry[engine_name]
    try:
        resolution_kind, resolved_version, ver_spec = _classify_requested_version(
            engine_name,
            version,
            engine_versions,
        )
    except ValueError as exc:
        stderr_logger.error(f"  ❌ {exc}")
        return False

    if resolution_kind == "exact":
        logger.info(f"  ✅ Version '{version}' found")
    else:
        supported_versions = _parse_supported_versions(engine_name, engine_versions)
        highest_validated_version = supported_versions[-1][1]
        logger.warning(
            f"  ⚠️  Version '{version}' is newer than highest validated version "
            f"'{highest_validated_version}'; trying default '{resolved_version}'"
        )

    # 4. features declared / builder present?
    has_spec = "builder" in ver_spec or "features" in ver_spec
    if not has_spec:
        stderr_logger.error(
            f"  ❌ No patch spec (builder or features) for {engine_name}@{resolved_version}."
        )
        return False
    logger.info("  ✅ Patch spec available (lazy builder or pre-loaded features)")

    # 5. individual features
    declared_features = set()
    if "features" in ver_spec:
        declared_features = set(ver_spec["features"].keys())
    elif "builder" in ver_spec:
        # builder is lazy; we trust declaration in supported_features.json
        logger.info("  ℹ️  Patch uses lazy builder; feature declarations taken from supported_features.json")

    for feat in features:
        if not declared_features or feat in declared_features:
            logger.info(f"  ✅ Feature '{feat}' declared")
        else:
            logger.warning(f"  ⚠️  Feature '{feat}' not in loaded spec (may be loaded lazily)")

    return True


# ---------------------------------------------------------------------------
# List features
# ---------------------------------------------------------------------------

def list_features(data: dict) -> None:
    logger.info(
        f"wings-accel supported features "
        f"(schema v{data['schema_version']}, updated {data['updated_at']})"
    )
    desc = data.get("description", "")
    if desc:
        logger.info(f"{desc}\n")

    for engine_name, engine_def in data["engines"].items():
        logger.info(f"  Engine: {engine_name}")
        edesc = engine_def.get("description", "")
        if edesc:
            logger.info(f"    {edesc}")
        for ver, ver_spec in engine_def.get("versions", {}).items():
            default_marker = "  [default]" if ver_spec.get("is_default") else ""
            logger.info(f"    Version: {ver}{default_marker}")
            features = ver_spec.get("features", {})
            if features:
                for feat, feat_def in features.items():
                    fdesc = feat_def.get("description", "")
                    logger.info(f"      - {feat}: {fdesc}")
            else:
                logger.info("      (no features declared)")
    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="install.py",
        description="wings-accel feature installation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 install.py --list
  python3 install.py --features '{"vllm":{"version":"0.17.0","features":["ears"]}}'
  python3 install.py --features '{"vllm":{"version":"0.17.0","features":["ears"]}}' --dry-run
  python3 install.py --check --features '{"vllm_ascend":{"version":"0.17.0","features":["parallel_spec_decode","ears"]}}'
  export WINGS_ENGINE_PATCH_OPTIONS='{"vllm":{"version":"0.17.0","features":["ears"]}}'
""",
    )
    parser.add_argument(
        "--features",
        type=str,
        metavar="JSON",
        help='JSON string: {"<engine>": {"version": "<ver>", "features": [...]}}',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without executing pip install",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that installed patches are registered and callable (developer self-validation)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported engines, versions, and features",
    )
    parser.add_argument(
        "--install-runtime-deps",
        action="store_true",
        help="Install fixed runtime dependencies only (wrapt, packaging, arctic-inference)",
    )
    args = parser.parse_args()

    # Load and validate supported_features.json
    try:
        data = load_supported_features()
        validate_schema(data)
    except (FileNotFoundError, ValueError) as e:
        stderr_logger.error(f"[wings-accel] Error: {e}")
        sys.exit(1)

    # --list
    if args.list:
        list_features(data)
        return

    if args.install_runtime_deps:
        install_runtime_dependencies(dry_run=args.dry_run)
        return

    # --features is required for all other modes
    if not args.features:
        parser.print_help()
        sys.exit(0)

    try:
        engine_name, resolved_version, requested_features = parse_requested_install(args.features, data)
    except ValueError as e:
        stderr_logger.error(f"[wings-accel] Error: {e}")
        sys.exit(1)

    if not args.check:
        install_runtime_dependencies(dry_run=args.dry_run)

    if args.check:
        ok = check_installed(engine_name, resolved_version, requested_features)
        sys.exit(0 if ok else 1)

    install_engine(
        engine_name,
        resolved_version,
        requested_features,
        dry_run=args.dry_run,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
