# SPDX-License-Identifier: Apache-2.0

"""Maintenance-mode API for Wings LMCache integration."""

from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import PlainTextResponse

from lmcache.v1.wings_ext.maintenance.state import (
    get_maintenance_state,
    set_maintenance_mode,
)

router = APIRouter()


class MaintenanceModeRequest(BaseModel):
    enabled: bool
    message: str | None = None


@router.get("/maintenance/mode")
async def get_maintenance_mode() -> PlainTextResponse:
    return PlainTextResponse(
        content=json.dumps(get_maintenance_state(), indent=2),
        media_type="application/json",
    )


@router.post("/maintenance/mode")
async def post_maintenance_mode(
    mode_request: MaintenanceModeRequest,
) -> PlainTextResponse:
    state = set_maintenance_mode(mode_request.enabled, mode_request.message)
    return PlainTextResponse(
        content=json.dumps(
            {
                "status": "success",
                **state,
            },
            indent=2,
        ),
        media_type="application/json",
    )
