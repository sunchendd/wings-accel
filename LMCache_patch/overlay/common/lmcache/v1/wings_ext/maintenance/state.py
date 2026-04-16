# SPDX-License-Identifier: Apache-2.0

"""In-process maintenance-mode state for Wings LMCache integration."""

from __future__ import annotations

from threading import Lock


_lock = Lock()
_maintenance_mode = False
_maintenance_message = "Service is under maintenance"


def _snapshot() -> dict[str, object]:
    return {
        "maintenance_mode": _maintenance_mode,
        "message": _maintenance_message,
    }


def set_maintenance_mode(enabled: bool, message: str | None = None) -> dict[str, object]:
    global _maintenance_mode, _maintenance_message
    with _lock:
        _maintenance_mode = bool(enabled)
        if message:
            _maintenance_message = message
        return _snapshot()


def get_maintenance_state() -> dict[str, object]:
    with _lock:
        return _snapshot()


def is_maintenance_mode_enabled() -> bool:
    with _lock:
        return _maintenance_mode


def get_maintenance_message() -> str:
    with _lock:
        return _maintenance_message
