# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account-scoped registration API for custom RVC checkpoint files."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette import status

from auth.authentication import get_current_subject
from core.inference import audio_rvc_models
from models.inference import AudioRVCModelListResponse, AudioRVCModelRecord

router = APIRouter()


def _limit_declared_size(request: Request, maximum: int) -> None:
    declared = request.headers.get("content-length")
    if declared is None:
        return
    try:
        size = int(declared)
    except ValueError:
        return
    if size > maximum:
        raise HTTPException(status_code = 413, detail = "RVC model file is too large.")


def _registry_error(exc: audio_rvc_models.RVCModelError) -> HTTPException:
    return HTTPException(status_code = exc.status, detail = exc.detail)


@router.get("/audio/rvc-models", response_model = AudioRVCModelListResponse)
async def list_rvc_models(current_subject: str = Depends(get_current_subject)):
    return {"models": await asyncio.to_thread(audio_rvc_models.list_models)}


@router.post(
    "/audio/rvc-models",
    status_code = status.HTTP_201_CREATED,
    response_model = AudioRVCModelRecord,
)
async def upload_rvc_checkpoint(
    request: Request,
    filename: str = Query(..., min_length = 1, max_length = 255),
    name: str | None = Query(None, max_length = 255),
    current_subject: str = Depends(get_current_subject),
):
    _limit_declared_size(request, audio_rvc_models.RVC_CHECKPOINT_MAX_BYTES)
    try:
        return await audio_rvc_models.register_checkpoint(
            request.stream(), filename = filename, name = name
        )
    except audio_rvc_models.RVCModelError as exc:
        raise _registry_error(exc) from None


@router.post(
    "/audio/rvc-models/{model_id}/index",
    status_code = status.HTTP_201_CREATED,
    response_model = AudioRVCModelRecord,
)
async def upload_rvc_index(
    model_id: str,
    request: Request,
    filename: str = Query(..., min_length = 1, max_length = 255),
    current_subject: str = Depends(get_current_subject),
):
    _limit_declared_size(request, audio_rvc_models.RVC_INDEX_MAX_BYTES)
    try:
        return await audio_rvc_models.register_index(
            model_id, request.stream(), filename = filename
        )
    except audio_rvc_models.RVCModelError as exc:
        raise _registry_error(exc) from None


@router.get("/audio/rvc-models/{model_id}", response_model = AudioRVCModelRecord)
async def get_rvc_model(model_id: str, current_subject: str = Depends(get_current_subject)):
    model = await asyncio.to_thread(audio_rvc_models.get, model_id)
    if model is None:
        raise HTTPException(status_code = 404, detail = "RVC model not found.")
    return model
