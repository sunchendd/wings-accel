#!/usr/bin/env python3
"""
wings-accel Feature Installation CLI

Implements the CLI contract defined in §3.3.6.2.1 of the wings-infer design document.

Usage:
    python install.py --list
    python install.py --features '<JSON>'
    python install.py --features '<JSON>' --dry-run
    python install.py --check --features '<JSON>'

The --features JSON format:
    {
        "<engine_name>": {
            "version": "<version_string>",
            "features": ["<feature_name>", ...]
        }
    }

Example:
    python install.py --features '{"vllm": {"version": "0.17.0", "features": ["adaptive_draft_model"]}}'

Deployment layout (all files at the same level as install.py):
    install.py
    supported_features.json
    wings_engine_patch-*.whl
    wrapt-*-linux_x86_64.whl
    wrapt-*-linux_aarch64.whl
    arctic_inference-*.whl      (pre-built wheel, installed offline)
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from packaging.version import InvalidVersion, Version


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
_LOCAL_WHEEL_DIR = _BASE_DIR

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


def _parse_supported_versions(engine_name: str, versions: dict) -> list[tuple[Version, str, dict]]:
    parsed_versions: list[tuple[Version, str, dict]] = []
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


# ---------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------

def validate_features(
    engine_name: str,
    version: str,
    requested_features: list,
    version_spec: dict,
) -> None:
    """Warn (not error) when a requested feature is not in the version spec."""
    available = set(version_spec.get("features", {}).keys())
    unknown = set(requested_features) - available
    if unknown:
        stderr_logger.warning(
            f"[wings-accel] Warning: features {sorted(unknown)} are not listed in "
            f"{engine_name}@{version}. Available: {sorted(available)}. "
            "Proceeding with installation anyway."
        )


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _find_local_whl() -> Path | None:
    """Return the wings_engine_patch .whl in _BASE_DIR (deployment layout).

    Falls back to build/output/ for local development (source-tree layout).
    """
    for search_dir in (_LOCAL_WHEEL_DIR, _BASE_DIR / "build" / "output"):
        if search_dir.exists():
            whls = [p for p in search_dir.glob("*.whl") if "wings_engine_patch" in p.name]
            if whls:
                return max(whls, key=lambda p: p.stat().st_mtime)
    return None


def _has_local_runtime_deps() -> bool:
    try:
        import wrapt  # noqa: F401
        import packaging  # noqa: F401
    except ImportError:
        return False
    return True


def _find_arctic_inference_whl() -> Path | None:
    """Return the arctic-inference pre-built wheel sitting next to install.py."""
    for pattern in (
        "arctic_inference-*.whl",
        "arctic-inference-*.whl",
    ):
        matches = list(_BASE_DIR.glob(pattern))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def _install_arctic_inference(dry_run: bool = False) -> None:
    """Install arctic-inference from the local pre-built wheel (offline, no compilation)."""
    whl = _find_arctic_inference_whl()
    if whl is None:
        logger.info("[wings-accel] arctic-inference wheel not found, skipping.")
        return

    cmd = [sys.executable, "-m", "pip", "install", str(whl), "--no-deps"]
    if dry_run:
        logger.info(f"[dry-run] Would run: {' '.join(str(c) for c in cmd)}")
        return

    logger.info(f"[wings-accel] Installing arctic-inference from {whl.name} ...")
    try:
        subprocess.check_call(cmd)
        logger.info("[wings-accel] ✅ arctic-inference installed.")
    except subprocess.CalledProcessError as e:
        stderr_logger.error(
            f"[wings-accel] Error: arctic-inference install failed (exit {e.returncode})."
        )
        raise


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
    extras = _ENGINE_TO_EXTRAS.get(engine_name, engine_name)
    local_whl = _find_local_whl()

    if local_whl:
        pkg = f"{local_whl}[{extras}]"
        local_dir = str(local_whl.parent)
        if _has_local_runtime_deps():
            cmd = [sys.executable, "-m", "pip", "install", pkg,
                   "--force-reinstall", "--no-deps"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", pkg,
                   "--force-reinstall", "--no-index", "--find-links", local_dir]
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
            cmd_fallback = [sys.executable, "-m", "pip", "install", pkg,
                            "--force-reinstall", "--no-deps"]
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
  python install.py --list
  python install.py --features '{"vllm": {"version": "0.17.0", "features": ["adaptive_draft_model"]}}'
  python install.py --features '{"vllm": {"version": "0.17.0", "features": ["adaptive_draft_model"]}}' --dry-run
  python install.py --check --features '{"vllm": {"version": "0.17.0", "features": ["adaptive_draft_model"]}}'
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

    # --features is required for all other modes
    if not args.features:
        parser.print_help()
        sys.exit(0)

    # Parse --features JSON
    try:
        features_config = json.loads(args.features)
    except json.JSONDecodeError as e:
        stderr_logger.error(f"[wings-accel] Error: --features is not valid JSON: {e}")
        sys.exit(1)

    if not isinstance(features_config, dict):
        stderr_logger.error("[wings-accel] Error: --features must be a JSON object.")
        sys.exit(1)

    engines_spec = data["engines"]
    exit_code = 0

    # Install arctic-inference once (before engine loop) when not in check mode.
    # vllm containers do not ship arctic-inference by default.
    if not args.check and args.features:
        _install_arctic_inference(dry_run=args.dry_run)

    for engine_name, config in features_config.items():
        if not isinstance(config, dict):
            stderr_logger.error(
                f"[wings-accel] Error: config for '{engine_name}' must be a JSON object "
                "with 'version' and optional 'features' keys."
            )
            sys.exit(1)

        if engine_name not in engines_spec:
            stderr_logger.error(
                f"[wings-accel] Error: engine '{engine_name}' is not listed in supported_features.json. "
                f"Available engines: {list(engines_spec.keys())}"
            )
            sys.exit(1)

        requested_version = config.get("version")
        requested_features = config.get("features", [])

        known_engine_config_keys = {"version", "features"}
        unknown_keys = set(config.keys()) - known_engine_config_keys
        if unknown_keys:
            stderr_logger.warning(
                f"[wings-accel] Warning: unknown keys {sorted(unknown_keys)} in config for "
                f"'{engine_name}'. Expected keys: {sorted(known_engine_config_keys)}."
            )

        if not requested_version:
            stderr_logger.error(f"[wings-accel] Error: 'version' is required for engine '{engine_name}'.")
            sys.exit(1)

        if not isinstance(requested_features, list):
            stderr_logger.error(f"[wings-accel] Error: 'features' for engine '{engine_name}' must be a list.")
            sys.exit(1)

        # Resolve version with fallback
        try:
            resolved_version, version_spec = resolve_version(
                engine_name, requested_version, engines_spec[engine_name]
            )
        except ValueError as e:
            stderr_logger.error(f"[wings-accel] Error: {e}")
            sys.exit(1)

        # Validate features (warn only, don't abort)
        validate_features(engine_name, resolved_version, requested_features, version_spec)

        if args.check:
            ok = check_installed(engine_name, resolved_version, requested_features)
            if not ok:
                exit_code = 1
        else:
            install_engine(
                engine_name,
                resolved_version,
                requested_features,
                dry_run=args.dry_run,
            )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
