# SPDX-License-Identifier: Apache-2.0

"""Full-sync lifecycle hooks for Wings LMCache integration."""

from __future__ import annotations

from typing import Any, Optional
import time

from lmcache.logging import init_logger
from lmcache.v1.wings_ext.config import get_wings_feature_config
from lmcache.v1.wings_ext.full_sync.state import (
    FullSyncStartDecision,
    finalize_full_sync_attempt,
    get_full_sync_state,
    remember_full_sync_attempt,
)

logger = init_logger(__name__)


def before_full_sync_start(
    sender: Any, reason: Optional[str]
) -> FullSyncStartDecision:
    feature_config = get_wings_feature_config(sender.config, "full_sync")
    if not bool(feature_config.get("enabled", False)):
        return FullSyncStartDecision(reason=reason)

    normalized_reason = reason or str(
        feature_config.get("default_reason", "wings_manual_full_sync")
    )
    min_interval_s = float(feature_config.get("min_interval_s", 0) or 0)
    state = get_full_sync_state(sender)
    if (
        min_interval_s > 0
        and state.last_success_at is not None
        and (time.time() - state.last_success_at) < min_interval_s
    ):
        remaining_s = max(0.0, min_interval_s - (time.time() - state.last_success_at))
        return FullSyncStartDecision(
            reason=normalized_reason,
            skip=True,
            message=(
                "cooldown active for reason="
                f"{normalized_reason}, remaining_s={remaining_s:.3f}"
            ),
        )

    remember_full_sync_attempt(sender, normalized_reason)
    sender._wings_full_sync_started_at = time.time()
    logger.info("Wings full-sync hook armed with reason=%s", normalized_reason)
    return FullSyncStartDecision(reason=normalized_reason)


def after_full_sync_finish(sender: Any, reason: Optional[str], success: bool) -> None:
    feature_config = get_wings_feature_config(sender.config, "full_sync")
    if not bool(feature_config.get("enabled", False)):
        return

    started_at = getattr(sender, "_wings_full_sync_started_at", None)
    duration_s = time.time() - started_at if started_at is not None else None
    history_limit = int(feature_config.get("history_limit", 16) or 16)
    entry = finalize_full_sync_attempt(
        sender,
        reason=reason,
        success=success,
        history_limit=history_limit,
        metadata={"duration_s": duration_s},
    )
    if duration_s is None:
        logger.info("Wings full-sync hook finished: reason=%s success=%s", reason, success)
        return

    logger.info(
        "Wings full-sync hook finished: reason=%s success=%s duration_s=%.3f key_count=%s attempts=%s history_size=%s",
        entry.reason,
        entry.success,
        duration_s,
        entry.key_count,
        get_full_sync_state(sender).attempt_count,
        len(get_full_sync_state(sender).history),
    )
