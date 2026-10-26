# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account-scoped registry for user-supplied RVC checkpoints and retrieval indexes.

The API accepts only opaque IDs. Runtime paths are derived from the authenticated
account's audio directory and are never returned to the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterable, Optional

from core.inference import audio_gallery
from utils.upload_limits import RVC_CHECKPOINT_MAX_BYTES, RVC_INDEX_MAX_BYTES

_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_CHECKPOINT_EXTENSIONS = frozenset((".pth", ".pt"))
MAX_MODELS = 16


class RVCModelError(Exception):
    """A safe, user-facing registry refusal."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def models_dir() -> Path:
    """Return the current account's private RVC directory."""
    account_audio = audio_gallery.gallery_dir()
    directory = account_audio / "rvc_models"
    try:
        directory.mkdir(parents = True, exist_ok = True)
        directory.resolve().relative_to(account_audio.resolve())
    except (OSError, ValueError):
        raise RVCModelError(404, "RVC model storage is unavailable.") from None
    return directory


def _safe_id(model_id: str) -> bool:
    return isinstance(model_id, str) and bool(_ID_RE.fullmatch(model_id))


def _metadata_path(model_id: str) -> Optional[Path]:
    if not _safe_id(model_id):
        return None
    directory = models_dir()
    path = directory / f"{model_id}.json"
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return None
    return path


def _read(model_id: str) -> Optional[dict[str, Any]]:
    path = _metadata_path(model_id)
    if path is None or path.is_symlink() or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("id") != model_id:
        return None
    if record.get("format") not in {ext[1:] for ext in _CHECKPOINT_EXTENSIONS}:
        return None
    return record


def _write_json(path: Path, record: dict[str, Any]) -> None:
    directory = models_dir()
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        raise RVCModelError(404, "RVC model storage is unavailable.") from None
    tmp = directory / f".{uuid.uuid4().hex}.metadata.tmp"
    try:
        with open(tmp, "x", encoding = "utf-8") as stream:
            json.dump(record, stream, ensure_ascii = False, separators = (",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok = True)
        raise RVCModelError(500, "Could not save RVC model metadata.") from None


def _clean_filename(filename: str, allowed: frozenset[str]) -> tuple[str, str]:
    basename = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not basename or len(basename) > 255 or basename in {".", ".."}:
        raise RVCModelError(400, "Provide a valid upload filename.")
    suffix = Path(basename).suffix.lower()
    if suffix not in allowed:
        choices = ", ".join(sorted(allowed))
        raise RVCModelError(400, f"Upload a file with one of these extensions: {choices}.")
    return basename, suffix


def _clean_label(value: Optional[str], fallback: str) -> str:
    label = (value or "").replace("\\", " ").replace("/", " ").strip()
    label = "".join(char for char in label if char >= " " and char != "\x7f")
    if not label:
        label = Path(fallback).stem
    if not label or len(label) > 128:
        raise RVCModelError(400, "RVC model name must be 1 to 128 characters.")
    return label


async def _write_upload(
    chunks: AsyncIterable[bytes], *, destination: Path, cap: int
) -> tuple[int, str]:
    directory = models_dir()
    try:
        destination.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        raise RVCModelError(404, "RVC model storage is unavailable.") from None
    tmp = directory / f".{uuid.uuid4().hex}.upload.tmp"
    digest = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "xb") as stream:
            async for chunk in chunks:
                if not isinstance(chunk, (bytes, bytearray)):
                    raise RVCModelError(400, "Invalid upload body.")
                size += len(chunk)
                if size > cap:
                    raise RVCModelError(413, "RVC model file is too large.")
                if chunk:
                    digest.update(chunk)
                    stream.write(chunk)
            if size == 0:
                raise RVCModelError(400, "RVC model file is empty.")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, destination)
    except RVCModelError:
        tmp.unlink(missing_ok = True)
        raise
    except OSError:
        tmp.unlink(missing_ok = True)
        raise RVCModelError(500, "Could not store the RVC model file.") from None
    except BaseException:
        tmp.unlink(missing_ok = True)
        raise
    return size, digest.hexdigest()


def _record(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": meta["id"],
        "name": meta["name"],
        "format": meta["format"],
        "bytes": int(meta.get("bytes") or 0),
        "has_index": bool(meta.get("index_bytes")),
        "index_bytes": int(meta.get("index_bytes") or 0),
        "created_at": meta["created_at"],
    }


async def register_checkpoint(
    chunks: AsyncIterable[bytes], *, filename: str, name: Optional[str] = None
) -> dict[str, Any]:
    basename, suffix = _clean_filename(filename, _CHECKPOINT_EXTENSIONS)
    if len(list_models()) >= MAX_MODELS:
        raise RVCModelError(400, f"You have {MAX_MODELS} registered RVC models.")
    directory = models_dir()
    model_id = uuid.uuid4().hex
    destination = directory / f"{model_id}{suffix}"
    size, digest = await _write_upload(
        chunks, destination = destination, cap = RVC_CHECKPOINT_MAX_BYTES
    )
    now = datetime.now(timezone.utc).isoformat(timespec = "seconds").replace("+00:00", "Z")
    meta = {
        "id": model_id,
        "name": _clean_label(name, basename),
        "format": suffix[1:],
        "bytes": size,
        "sha256": digest,
        "index_bytes": 0,
        "created_at": now,
    }
    try:
        _write_json(directory / f"{model_id}.json", meta)
    except BaseException:
        destination.unlink(missing_ok = True)
        raise
    return _record(meta)


async def register_index(
    model_id: str, chunks: AsyncIterable[bytes], *, filename: str
) -> dict[str, Any]:
    meta = _read(model_id)
    if meta is None:
        raise RVCModelError(404, "RVC model not found.")
    _basename, suffix = _clean_filename(filename, frozenset((".index",)))
    del _basename, suffix
    directory = models_dir()
    destination = directory / f"{model_id}.index"
    try:
        # Exclusive creation reserves the one index slot while the request streams the upload.
        with open(destination, "xb"):
            pass
    except FileExistsError:
        raise RVCModelError(409, "An index is already registered for this RVC model.") from None
    except OSError:
        raise RVCModelError(500, "Could not reserve the RVC index slot.") from None
    try:
        size, digest = await _write_upload(
            chunks, destination = destination, cap = RVC_INDEX_MAX_BYTES
        )
        meta["index_bytes"] = size
        meta["index_sha256"] = digest
        _write_json(directory / f"{model_id}.json", meta)
    except BaseException:
        destination.unlink(missing_ok = True)
        raise
    return _record(meta)


def get(model_id: str) -> Optional[dict[str, Any]]:
    meta = _read(model_id)
    if meta is None:
        return None
    checkpoint = _owned_file(model_id, f"{model_id}.{meta['format']}")
    if checkpoint is None:
        return None
    if meta.get("index_bytes") and _owned_file(model_id, f"{model_id}.index") is None:
        return None
    return _record(meta)


def list_models() -> list[dict[str, Any]]:
    directory = models_dir()
    records = []
    try:
        for path in directory.glob("*.json"):
            model_id = path.stem
            meta = _read(model_id)
            if meta is not None and get(model_id) is not None:
                records.append(meta)
    except OSError:
        return []
    records.sort(key = lambda meta: meta.get("created_at", ""), reverse = True)
    return [_record(meta) for meta in records]


def _owned_file(model_id: str, filename: str) -> Optional[Path]:
    if not _safe_id(model_id) or Path(filename).name != filename:
        return None
    directory = models_dir()
    path = directory / filename
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return None
    return path if path.is_file() and not path.is_symlink() else None


def resolve_paths(
    model_id: str, retrieval_blend: Any = 0.0
) -> tuple[Path, Optional[Path]]:
    meta = _read(model_id)
    if meta is None:
        raise RVCModelError(404, "RVC model not found.")
    checkpoint = _owned_file(model_id, f"{model_id}.{meta['format']}")
    if checkpoint is None:
        raise RVCModelError(404, "RVC model file is unavailable.")
    try:
        blend = float(retrieval_blend or 0.0)
    except (TypeError, ValueError):
        raise RVCModelError(400, "RVC retrieval blend must be between 0 and 1.") from None
    if not math.isfinite(blend) or blend < 0 or blend > 1:
        raise RVCModelError(400, "RVC retrieval blend must be between 0 and 1.")
    index = _owned_file(model_id, f"{model_id}.index")
    if blend > 0 and index is None:
        raise RVCModelError(400, "Register an .index file or set retrieval_blend to 0.")
    return checkpoint, index if blend > 0 else None
