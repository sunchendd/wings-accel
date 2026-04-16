#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


def _status(label: str, state: str, detail: str) -> None:
    print(f"[{state}] {label}: {detail}")


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _torch_lib_dir() -> Path | None:
    try:
        import torch
    except Exception:
        return None
    return Path(torch.__file__).resolve().parent / "lib"


def _split_ld_library_path() -> list[str]:
    value = os.environ.get("LD_LIBRARY_PATH", "")
    return [entry for entry in value.split(":") if entry]


def _command_exists(command: str) -> bool:
    return subprocess.run(
        ["/usr/bin/env", "bash", "-lc", f"command -v {command}"],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0


def _import_module(name: str):
    try:
        module = importlib.import_module(name)
        return module, None
    except Exception as exc:
        return None, exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight checker for Wings LMCache QAT runtime readiness."
    )
    parser.add_argument(
        "--require-devices",
        action="store_true",
        help="Return non-zero when no QAT devices are visible.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    _status("python", "INFO", sys.executable)

    torch_lib_dir = _torch_lib_dir()
    if torch_lib_dir is None:
        failures.append("torch is not importable in the current Python environment")
        _status("torch", "FAIL", "module import failed")
    else:
        _status("torch", "PASS", f"lib dir at {torch_lib_dir}")
        if str(torch_lib_dir) in _split_ld_library_path():
            _status("LD_LIBRARY_PATH", "PASS", f"contains {torch_lib_dir}")
        else:
            warnings.append(
                f"LD_LIBRARY_PATH is missing {torch_lib_dir}; kv_agent may fail with missing libc10.so"
            )
            _status(
                "LD_LIBRARY_PATH",
                "WARN",
                f"missing {torch_lib_dir}; export LD_LIBRARY_PATH={torch_lib_dir}:${{LD_LIBRARY_PATH}}",
            )

    lmcache_module, lmcache_error = _import_module("lmcache")
    if lmcache_module is None:
        failures.append(f"lmcache import failed: {lmcache_error}")
        _status("lmcache", "FAIL", repr(lmcache_error))
    else:
        _status("lmcache", "PASS", str(Path(lmcache_module.__file__).resolve()))

    kv_agent_module, kv_agent_error = _import_module("kv_agent")
    if kv_agent_module is None:
        failures.append(f"kv_agent import failed: {kv_agent_error}")
        _status("kv_agent", "FAIL", repr(kv_agent_error))
    else:
        _status("kv_agent", "PASS", str(Path(kv_agent_module.__file__).resolve()))

    candidate_lib_roots = []
    if kv_agent_module is not None:
        candidate_lib_roots.append(Path(kv_agent_module.__file__).resolve().parent / "lib")
    candidate_lib_roots.extend(
        [
            Path("/opt/wings-qat/lib"),
            Path("/usr/local/lib"),
            Path("/usr/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
        ]
    )

    required_runtime_libs = [
        "libqatzip.so",
        "libqat_s.so",
        "libusdm_drv_s.so",
    ]
    for lib_name in required_runtime_libs:
        resolved = _find_first_existing([root / lib_name for root in candidate_lib_roots])
        if resolved is None:
            warnings.append(f"runtime library not found: {lib_name}")
            _status(lib_name, "WARN", "not found in common library roots")
        else:
            _status(lib_name, "PASS", str(resolved))

    for command in ["adf_ctl", "lspci"]:
        if _command_exists(command):
            _status(command, "PASS", "available")
        else:
            warnings.append(f"tool not available: {command}")
            _status(command, "WARN", "not available")

    if lmcache_module is not None:
        try:
            from lmcache.v1.wings_ext.qat.device_probe import probe_qat_devices

            summary = probe_qat_devices(timeout=5)
            detail = (
                f"available={summary.available_devices}, total={summary.total_devices}, "
                f"used_adf_ctl={summary.used_adf_ctl}, message={summary.message or 'n/a'}"
            )
            if summary.available_devices > 0:
                _status("qat_devices", "PASS", detail)
            else:
                message = "no QAT devices visible to runtime"
                if args.require_devices:
                    failures.append(message)
                    _status("qat_devices", "FAIL", detail)
                else:
                    warnings.append(message)
                    _status("qat_devices", "WARN", detail)
        except Exception as exc:
            failures.append(f"QAT probe failed: {exc}")
            _status("qat_devices", "FAIL", repr(exc))

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item}")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nQAT runtime precheck passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())