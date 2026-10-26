# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from auth.authentication import get_current_subject
from core.inference import audio_gallery
from models.inference import AudioConvertParams, AudioRunRequest
from routes.audio_rvc import router as audio_rvc_router
from utils.account_context import AccountContext, bind_account, reset_account


def _client(monkeypatch, tmp_path, active):
    monkeypatch.setattr(
        audio_gallery,
        "gallery_dir",
        lambda: tmp_path / active[0].account_id / "audio",
    )
    app = FastAPI()

    async def subject():
        token = bind_account(active[0])
        try:
            yield active[0].username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(audio_rvc_router, prefix = "/api/inference")
    app.include_router(audio_rvc_router, prefix = "/v1")
    return TestClient(app)


def test_registration_endpoints_use_opaque_ids_and_enforce_account_scope(monkeypatch, tmp_path):
    active = [AccountContext("alice", "alice")]
    with _client(monkeypatch, tmp_path, active) as client:
        uploaded = client.post(
            "/api/inference/audio/rvc-models",
            params = {"filename": "Dafeiyu.pth", "name": "大肥魚"},
            content = b"checkpoint",
            headers = {"content-type": "application/octet-stream"},
        )
        assert uploaded.status_code == 201, uploaded.text
        record = uploaded.json()
        assert record["name"] == "大肥魚"
        assert len(record["id"]) == 32
        assert "path" not in uploaded.text.lower()
        assert client.get("/v1/audio/rvc-models").json()["models"] == [record]

        indexed = client.post(
            f"/v1/audio/rvc-models/{record['id']}/index",
            params = {"filename": "Dafeiyu.index"},
            content = b"index",
            headers = {"content-type": "application/octet-stream"},
        )
        assert indexed.status_code == 201, indexed.text
        assert indexed.json()["has_index"] is True

        active[0] = AccountContext("bob", "bob")
        assert client.get(f"/api/inference/audio/rvc-models/{record['id']}").status_code == 404
        active[0] = AccountContext("alice", "alice")
        assert client.get(f"/v1/audio/rvc-models/{record['id']}").status_code == 200


def test_registration_rejects_unsupported_extensions_and_oversized_body(monkeypatch, tmp_path):
    from core.inference import audio_rvc_models

    active = [AccountContext("alice", "alice")]
    with _client(monkeypatch, tmp_path, active) as client:
        rejected = client.post(
            "/api/inference/audio/rvc-models",
            params = {"filename": "voice.exe"},
            content = b"x",
        )
        assert rejected.status_code == 400

        monkeypatch.setattr(audio_rvc_models, "RVC_CHECKPOINT_MAX_BYTES", 3)
        oversized = client.post(
            "/api/inference/audio/rvc-models",
            params = {"filename": "voice.pth"},
            content = b"four",
        )
        assert oversized.status_code == 413


def test_custom_model_id_schema_rejects_paths_and_target_refs():
    model_id = "a" * 32
    assert "rvc_model_id" not in AudioConvertParams().model_dump()
    assert AudioConvertParams(rvc_model_id = model_id).model_dump()["rvc_model_id"] == model_id

    with pytest.raises(ValidationError):
        AudioRunRequest.model_validate(
            {
                "workflow": "convert",
                "convert": {"rvc_model_id": model_id},
                "inputs": {"target": {"input_id": "b" * 32}},
            }
        )
    with pytest.raises(ValidationError):
        AudioRunRequest.model_validate(
            {
                "workflow": "convert",
                "convert": {"rvc_model_id": model_id, "voice": "manthos"},
                "inputs": {"source": {"input_id": "b" * 32}},
            }
        )
    with pytest.raises(ValidationError):
        AudioRunRequest.model_validate(
            {
                "workflow": "convert",
                "inputs": {"source": {"input_id": "b" * 32}},
                "options": {"voice_model_path": "C:/outside/model.pth"},
            }
        )
