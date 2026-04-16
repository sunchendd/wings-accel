# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
import time

from lmcache.v1.wings_ext.full_sync import (
    after_full_sync_finish,
    before_full_sync_start,
    get_full_sync_state,
)


def _make_sender(**feature_overrides):
    full_sync_config = {
        "enabled": True,
        "default_reason": "wings_manual_full_sync",
        "history_limit": 2,
        "min_interval_s": 0,
    }
    full_sync_config.update(feature_overrides)
    return SimpleNamespace(
        config=SimpleNamespace(extra_config={"wings": {"full_sync": full_sync_config}}),
        local_cpu_backend=SimpleNamespace(get_keys=lambda: [1, 2, 3]),
        _current_sync_id="sync-1",
    )


def test_before_full_sync_start_skips_when_cooldown_active():
    sender = _make_sender(min_interval_s=60)
    state = get_full_sync_state(sender)
    state.last_success_at = time.time()

    decision = before_full_sync_start(sender, None)

    assert decision.skip is True
    assert decision.reason == "wings_manual_full_sync"
    assert "cooldown active" in decision.message


def test_full_sync_history_tracks_success_and_failure():
    sender = _make_sender(history_limit=2)

    first = before_full_sync_start(sender, "manual")
    assert first.skip is False
    after_full_sync_finish(sender, first.reason, True)

    second = before_full_sync_start(sender, "retry")
    assert second.skip is False
    after_full_sync_finish(sender, second.reason, False)

    third = before_full_sync_start(sender, None)
    assert third.skip is False
    after_full_sync_finish(sender, third.reason, True)

    state = get_full_sync_state(sender)
    assert state.attempt_count == 3
    assert state.success_count == 2
    assert state.failure_count == 1
    assert state.last_reason == "wings_manual_full_sync"
    assert state.last_key_count == 3
    assert state.last_success_at is not None
    assert state.last_failure_at is not None
    assert len(state.history) == 2
    assert state.history[-1].reason == "wings_manual_full_sync"
    assert state.history[-1].success is True
