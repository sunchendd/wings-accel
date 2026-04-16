# SPDX-License-Identifier: Apache-2.0

"""Helpers for reading Wings-specific LMCache configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


WINGS_EXTRA_CONFIG_KEY = "wings"


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def get_wings_extra_config(config: Any) -> dict[str, Any]:
    extra_config = getattr(config, "extra_config", None)
    if not isinstance(extra_config, Mapping):
        return {}
    return _as_dict(extra_config.get(WINGS_EXTRA_CONFIG_KEY))


def get_wings_feature_config(config: Any, feature_name: str) -> dict[str, Any]:
    return _as_dict(get_wings_extra_config(config).get(feature_name))


def is_wings_feature_enabled(
    config: Any,
    feature_name: str,
    default: bool = False,
) -> bool:
    feature_config = get_wings_feature_config(config, feature_name)
    enabled = feature_config.get("enabled", default)
    return bool(enabled)

