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
"""

import argparse
import json
import logging
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

def resolve_version(engine_name: str, requested_version: str, engine_spec: dict):
    """Resolve version with default fallback per §3.3.6.4.5.2.

    Returns (resolved_version_str, version_spec_dict).
    """
    versions = engine_spec.get("versions", {})

    if requested_version in versions:
        return requested_version, versions[requested_version]

# Fallback: find the default version
    for ver, spec in versions.items():
        if spec.get("is_default", False):
            stderr_logger.warning(
                f"[wings-accel] Warning: version '{requested_version}' not found for engine "
                f"'{engine_name}'. Falling back to default version '{ver}'."
            )
            return ver, spec

    raise ValueError(
        f"Version '{requested_version}' not found for engine '{engine_name}' "
        "and no default version is configured."
    )


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
    """Return the most recently built .whl in build/output/, if present."""
    if _LOCAL_WHEEL_DIR.exists():
        whls = list(_LOCAL_WHEEL_DIR.glob("*.whl"))
        if whls:
            return max(whls, key=lambda p: p.stat().st_mtime)
    return None


def _has_local_runtime_deps() -> bool:
    try:
        import wrapt  # noqa: F401
    except ImportError:
        return False
    return True


def install_engine(
    engine_name: str,
    version: str,
    features: list,
    dry_run: bool = False,
) -> None:
    """Run pip install for the given engine's extras group."""
    extras = _ENGINE_TO_EXTRAS.get(engine_name, engine_name)
    local_whl = _find_local_whl()

    if local_whl:
        pkg = f"{local_whl}[{extras}]"
        cmd = [sys.executable, "-m", "pip", "install", pkg, "--force-reinstall"]
        if _has_local_runtime_deps():
            cmd.append("--no-deps")
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

    # 3. version (exact or default)?
    engine_versions = registry[engine_name]
    if version in engine_versions:
        ver_spec = engine_versions[version]
        logger.info(f"  ✅ Version '{version}' found")
    else:
        defaults = {v: s for v, s in engine_versions.items() if s.get("is_default", False)}
        if defaults:
            default_ver = next(iter(defaults))
            logger.warning(f"  ⚠️  Version '{version}' not in registry; default is '{default_ver}'")
            ver_spec = defaults[default_ver]
        else:
            stderr_logger.error(f"  ❌ Version '{version}' not found and no default configured.")
            return False

    # 4. features declared / builder present?
    has_spec = "builder" in ver_spec or "features" in ver_spec
    if not has_spec:
        stderr_logger.error(f"  ❌ No patch spec (builder or features) for {engine_name}@{version}.")
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
