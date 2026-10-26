# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio

import pytest

from core.inference import audio_gallery, audio_rvc_models


async def _chunks(*parts: bytes):
    for part in parts:
        yield part


@pytest.fixture
def account_audio(tmp_path, monkeypatch):
    active = ["alice"]
    monkeypatch.setattr(
        audio_gallery,
        "gallery_dir",
        lambda: tmp_path / active[0] / "audio",
    )
    return active


def _register(filename: str, *parts: bytes, name: str | None = None):
    return asyncio.run(
        audio_rvc_models.register_checkpoint(
            _chunks(*parts), filename = filename, name = name
        )
    )


def test_checkpoint_and_index_are_account_scoped_and_paths_stay_private(account_audio):
    record = _register(r"C:\\voices\\dafeiyu.pth", b"weights", name = "大肥魚")
    assert record["name"] == "大肥魚"
    assert record["format"] == "pth"
    assert record["bytes"] == len(b"weights")
    assert set(record) == {
        "id", "name", "format", "bytes", "has_index", "index_bytes", "created_at"
    }
    assert str(audio_rvc_models.models_dir()) not in str(record)

    checkpoint, index = audio_rvc_models.resolve_paths(record["id"], retrieval_blend = 0)
    assert checkpoint.name == f"{record['id']}.pth"
    assert index is None
    with pytest.raises(audio_rvc_models.RVCModelError, match = "Register an"):
        audio_rvc_models.resolve_paths(record["id"], retrieval_blend = 0.5)

    indexed = asyncio.run(
        audio_rvc_models.register_index(
            record["id"], _chunks(b"faiss"), filename = "dafeiyu.index"
        )
    )
    assert indexed["has_index"] is True
    assert indexed["index_bytes"] == len(b"faiss")
    checkpoint, index = audio_rvc_models.resolve_paths(record["id"], retrieval_blend = 0.5)
    assert checkpoint.is_file() and index is not None and index.read_bytes() == b"faiss"

    account_audio[0] = "bob"
    assert audio_rvc_models.get(record["id"]) is None
    with pytest.raises(audio_rvc_models.RVCModelError, match = "not found"):
        audio_rvc_models.resolve_paths(record["id"])
    account_audio[0] = "alice"
    assert audio_rvc_models.get(record["id"])["name"] == "大肥魚"


@pytest.mark.parametrize("filename", ["voice.bin", "../voice.exe", ""])
def test_checkpoint_extension_is_allowlisted(account_audio, filename):
    with pytest.raises(audio_rvc_models.RVCModelError, match = "extension|filename"):
        _register(filename, b"x")


def test_stream_limit_removes_partial_upload(account_audio, monkeypatch):
    monkeypatch.setattr(audio_rvc_models, "RVC_CHECKPOINT_MAX_BYTES", 4)
    with pytest.raises(audio_rvc_models.RVCModelError, match = "too large"):
        _register("voice.pth", b"123", b"45")
    directory = audio_rvc_models.models_dir()
    assert list(directory.glob("*.pth")) == []
    assert list(directory.glob("*.tmp")) == []
