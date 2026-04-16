#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


TARGET_METADATA = {
    "nvidia-x86": {
        "accelerator": "nvidia",
        "architecture": "x86_64",
        "platform": "common",
    },
    "ascend-arm": {
        "accelerator": "ascend",
        "architecture": "aarch64",
        "platform": "ascend",
    },
}


def build_manifest(output_root: Path, manifest_path: Path) -> dict:
    base_dir = manifest_path.parent
    targets: dict[str, dict] = {}

    for target, metadata in TARGET_METADATA.items():
        target_dir = output_root / target
        if not target_dir.is_dir():
            continue

        wheels = sorted(path for path in target_dir.glob("*.whl") if path.is_file())
        if not wheels:
            continue

        lmcache_wheels = [path for path in wheels if path.name.startswith("lmcache-")]
        if len(lmcache_wheels) != 1:
            raise SystemExit(
                f"Expected exactly one lmcache wheel in {target_dir}, found {len(lmcache_wheels)}"
            )

        primary = lmcache_wheels[0]
        companions = [path for path in wheels if path != primary]
        dependency_dir = target_dir / "deps"
        dependency_wheels = []
        if dependency_dir.is_dir():
            dependency_wheels = sorted(
                path for path in dependency_dir.glob("*.whl") if path.is_file()
            )

        targets[target] = {
            **metadata,
            "directory": str(target_dir.relative_to(base_dir)),
            "primary_wheel": str(primary.relative_to(base_dir)),
            "companion_wheels": [str(path.relative_to(base_dir)) for path in companions],
            "dependency_wheels": [str(path.relative_to(base_dir)) for path in dependency_wheels],
        }

    return {
        "schema_version": "1.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the LMCache build manifest.")
    parser.add_argument("--output-root", required=True, help="LMCache target output root directory")
    parser.add_argument("--manifest-path", required=True, help="Path to write the manifest JSON")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(output_root, manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())