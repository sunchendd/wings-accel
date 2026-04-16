#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PATCH_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PATCH_ROOT / "scripts"


COMMAND_SPECS = {
    "prepare-common-sources": ("prepare_common_sources.sh", ["--platform", "common"]),
    "unpack-upstream": ("unpack_upstream.sh", []),
    "materialize-workspace": ("materialize_workspace.sh", []),
    "apply-patchset": ("apply_patchset.sh", []),
    "build-kv-agent": ("build_kv_agent.sh", []),
    "build-wheel": ("build_wheel.sh", []),
    "prepare-ascend-sources": ("prepare_ascend_sources.sh", ["--platform", "ascend"]),
    "regen-patchset": ("regen_patchset.sh", []),
    "verify-upgrade": ("verify_upgrade.sh", []),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone entrypoint for the LMCache patch-first workflow. "
            "Pass platform flags such as '--platform ascend' after the command "
            "when needed."
        )
    )
    parser.add_argument(
        "command",
        choices=sorted(COMMAND_SPECS.keys()),
        help="Workflow step to run inside LMCache_patch.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed through to the underlying shell script.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    script_name, default_args = COMMAND_SPECS[args.command]
    script_path = SCRIPTS_DIR / script_name
    if not script_path.is_file():
        parser.error(f"Missing workflow script: {script_path}")

    passthrough = list(args.script_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    cmd = ["/bin/bash", str(script_path), *default_args, *passthrough]
    completed = subprocess.run(cmd, cwd=PATCH_ROOT)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
