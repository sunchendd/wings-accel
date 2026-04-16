# SPDX-License-Identifier: Apache-2.0

"""State helpers for Wings full-sync policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import time


@dataclass
class FullSyncStartDecision:
    reason: Optional[str]
    skip: bool = False
    message: str = ""


@dataclass
class FullSyncHistoryEntry:
    reason: Optional[str]
    success: bool
    started_at: float
    finished_at: float
    duration_s: Optional[float]
    key_count: Optional[int]
    sync_id: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullSyncState:
    attempt_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_reason: Optional[str] = None
    last_success_at: Optional[float] = None
    last_failure_at: Optional[float] = None
    last_duration_s: Optional[float] = None
    last_key_count: Optional[int] = None
    history: list[FullSyncHistoryEntry] = field(default_factory=list)


def get_full_sync_state(sender: Any) -> FullSyncState:
    state = getattr(sender, "_wings_full_sync_state", None)
    if isinstance(state, FullSyncState):
        return state
    state = FullSyncState()
    sender._wings_full_sync_state = state
    return state


def count_hot_cache_keys(sender: Any) -> Optional[int]:
    local_cpu_backend = getattr(sender, "local_cpu_backend", None)
    if local_cpu_backend is None or not hasattr(local_cpu_backend, "get_keys"):
        return None
    try:
        return len(local_cpu_backend.get_keys())
    except Exception:
        return None


def remember_full_sync_attempt(sender: Any, reason: Optional[str]) -> dict[str, Any]:
    context = {
        "reason": reason,
        "started_at": time.time(),
        "key_count": count_hot_cache_keys(sender),
        "sync_id": getattr(sender, "_current_sync_id", None),
    }
    sender._wings_full_sync_context = context
    state = get_full_sync_state(sender)
    state.attempt_count += 1
    state.last_reason = reason
    state.last_key_count = context["key_count"]
    return context


def finalize_full_sync_attempt(
    sender: Any,
    reason: Optional[str],
    success: bool,
    history_limit: int,
    metadata: Optional[dict[str, Any]] = None,
) -> FullSyncHistoryEntry:
    state = get_full_sync_state(sender)
    context = getattr(sender, "_wings_full_sync_context", {}) or {}
    finished_at = time.time()
    started_at = context.get("started_at", finished_at)
    duration_s = finished_at - started_at if started_at is not None else None
    key_count = context.get("key_count")
    sync_id = getattr(sender, "_current_sync_id", None) or context.get("sync_id")

    entry = FullSyncHistoryEntry(
        reason=reason,
        success=success,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s,
        key_count=key_count,
        sync_id=sync_id,
        metadata=dict(metadata or {}),
    )
    state.history.append(entry)
    if history_limit > 0 and len(state.history) > history_limit:
        state.history = state.history[-history_limit:]

    state.last_duration_s = duration_s
    if success:
        state.success_count += 1
        state.last_success_at = finished_at
    else:
        state.failure_count += 1
        state.last_failure_at = finished_at

    sender._wings_full_sync_context = None
    return entry
