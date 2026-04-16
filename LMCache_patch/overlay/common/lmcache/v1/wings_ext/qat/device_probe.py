# SPDX-License-Identifier: Apache-2.0

"""QAT device probing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from shutil import which
from typing import Any
import re
import subprocess

from lmcache.logging import init_logger

logger = init_logger(__name__)

KNOWN_QAT_DEVICE_IDS = {
    "8086:37c8",
    "8086:19e2",
    "8086:0435",
    "8086:6f54",
    "8086:4940",
    "8086:4942",
}

DEVICE_PATTERN = re.compile(
    r"qat_dev(?P<id>\d+)\s*-\s*type:\s*(?P<type>\w+),\s*"
    r"inst_id:\s*(?P<inst_id>\d+),\s*node_id:\s*(?P<node_id>\d+),\s*"
    r"bsf:\s*(?P<bsf>[\w:.]+),\s*#accel:\s*(?P<accel>\d+)\s*"
    r"#engines:\s*(?P<engines>\d+)\s*state:\s*(?P<state>\w+)"
)


@dataclass
class QATDeviceProbeSummary:
    total_devices: int = 0
    available_devices: int = 0
    any_qat_detected: bool = False
    used_adf_ctl: bool = False
    node_stats: dict[int, dict[str, Any]] = field(default_factory=dict)
    device_details: list[dict[str, Any]] = field(default_factory=list)
    lspci_matches: list[str] = field(default_factory=list)
    message: str = ""


def _run_lspci_check() -> QATDeviceProbeSummary:
    summary = QATDeviceProbeSummary(message="lspci fallback did not detect any QAT devices")
    if which("lspci") is None:
        summary.message = "lspci command not available"
        return summary

    try:
        completed = subprocess.run(
            ["lspci", "-nn"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        summary.message = f"failed to execute lspci: {exc}"
        return summary

    if completed.returncode != 0:
        summary.message = completed.stderr.strip() or "lspci returned non-zero status"
        return summary

    matches = [
        line
        for line in completed.stdout.splitlines()
        if any(device_id in line for device_id in KNOWN_QAT_DEVICE_IDS)
    ]
    summary.lspci_matches = matches
    summary.total_devices = len(matches)
    summary.available_devices = len(matches)
    summary.any_qat_detected = bool(matches)
    if matches:
        summary.message = f"detected {len(matches)} QAT devices via lspci fallback"
    return summary


def _group_devices_by_node(devices: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    node_stats: dict[int, dict[str, Any]] = {}
    for device in devices:
        if device["state"].lower() != "up":
            continue
        node_id = int(device["node_id"])
        bucket = node_stats.setdefault(node_id, {"available_count": 0, "devices": []})
        bucket["available_count"] += 1
        bucket["devices"].append(
            {
                "device_id": device["id"],
                "type": device["type"],
                "bsf": device["bsf"],
            }
        )
    return node_stats


def probe_qat_devices(timeout: int = 30) -> QATDeviceProbeSummary:
    if which("adf_ctl") is None:
        return _run_lspci_check()

    summary = QATDeviceProbeSummary(used_adf_ctl=True)
    try:
        completed = subprocess.run(
            ["adf_ctl", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        summary.message = f"adf_ctl status timed out after {timeout}s"
        return summary
    except OSError as exc:
        summary.message = f"failed to execute adf_ctl: {exc}"
        return summary

    if completed.returncode != 0:
        summary.message = completed.stderr.strip() or "adf_ctl returned non-zero status"
        return summary

    devices: list[dict[str, Any]] = []
    for match in DEVICE_PATTERN.finditer(completed.stdout):
        device = match.groupdict()
        device["id"] = int(device["id"])
        device["inst_id"] = int(device["inst_id"])
        device["node_id"] = int(device["node_id"])
        device["accel"] = int(device["accel"])
        device["engines"] = int(device["engines"])
        devices.append(device)

    summary.device_details = devices
    summary.total_devices = len(devices)
    summary.available_devices = sum(
        1 for device in devices if str(device["state"]).lower() == "up"
    )
    summary.any_qat_detected = bool(devices)
    summary.node_stats = _group_devices_by_node(devices)
    if not devices:
        summary.message = "no QAT devices parsed from adf_ctl status"
    elif summary.available_devices <= 0:
        summary.message = "all detected QAT devices are down"
    else:
        summary.message = (
            f"detected {summary.available_devices}/{summary.total_devices} QAT devices up"
        )
    return summary
