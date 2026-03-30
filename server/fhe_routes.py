"""FHE API routes — encrypted versions of enroll/search/identities."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from fhe.crypto import (
    StaleCiphertextError,
    decrypt_result,
    deserialize_ciphertext,
    encrypt_embedding,
    encrypt_query,
    eval_dot_product,
)
from fhe.db import delete_fhe_identities, get_fhe_connection, load_fhe_identities, upsert_fhe_identity
from fhe.runtime import claim_native_fhe

logger = logging.getLogger("fhe")

router = APIRouter(prefix="/fhe", tags=["fhe"])


class FHEEnrollRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)
    embedding: list[float] = Field(..., min_length=128, max_length=128)
    metadata: dict[str, Any] | None = None

    @field_validator("label")
    @classmethod
    def strip_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("label cannot be empty")
        return cleaned


class FHESearchRequest(BaseModel):
    embedding: list[float] = Field(..., min_length=128, max_length=128)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.35, ge=-1.0, le=1.0)


class FHEMatch(BaseModel):
    label: str
    similarity: float
    metadata: dict[str, Any] | None


class FHESearchResponse(BaseModel):
    count: int
    best_match: FHEMatch | None
    matches: list[FHEMatch]


def _normalize(embedding: list[float]) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Embedding norm cannot be zero")
    return arr / norm


def _require_native_fhe() -> None:
    reason = claim_native_fhe()
    if reason is not None:
        raise HTTPException(status_code=503, detail=reason)


def _prune_stale_identities(conn, identities: list[dict[str, Any]], *, stream: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    active: list[dict[str, Any]] = []
    stale_labels: list[str] = []

    for identity in identities:
        try:
            deserialize_ciphertext(identity["ciphertext"])
        except StaleCiphertextError:
            stale_labels.append(identity["label"])
        else:
            active.append(identity)

    if stale_labels:
        removed = delete_fhe_identities(conn, stale_labels)
        logger.warning(
            "[search%s] Removed %d stale FHE identities whose ciphertext tokens no longer exist in memory: %s",
            "/stream" if stream else "",
            removed,
            ", ".join(stale_labels),
        )

    return active, stale_labels


@router.post("/enroll")
def fhe_enroll(payload: FHEEnrollRequest) -> dict[str, str]:
    _require_native_fhe()
    t_start = time.time()
    try:
        unit_vec = _normalize(payload.embedding)
        logger.info("[enroll] Encrypting embedding for '%s'", payload.label)
        ct_bytes = encrypt_embedding(unit_vec.tolist())

        with get_fhe_connection() as conn:
            upsert_fhe_identity(conn, payload.label, ct_bytes, payload.metadata)

        elapsed = time.time() - t_start
        logger.info("[enroll] '%s' enrolled in %.3fs (%d bytes ciphertext)",
                     payload.label, elapsed, len(ct_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "saved", "label": payload.label}


@router.post("/search", response_model=FHESearchResponse)
def fhe_search(payload: FHESearchRequest) -> FHESearchResponse:
    _require_native_fhe()
    t_start = time.time()
    try:
        query_vec = _normalize(payload.embedding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("[search] Encrypting query embedding")
    enc_query = deserialize_ciphertext(encrypt_query(query_vec.tolist()))

    with get_fhe_connection() as conn:
        identities = load_fhe_identities(conn)
        identities, stale_labels = _prune_stale_identities(conn, identities)

    if not identities:
        if stale_labels:
            logger.info("[search] All available FHE identities were stale and have been removed")
        else:
            logger.info("[search] No FHE identities enrolled")
        return FHESearchResponse(count=0, best_match=None, matches=[])

    logger.info("[search] Comparing against %d encrypted identities", len(identities))
    ranked: list[FHEMatch] = []
    for identity in identities:
        enc_stored = deserialize_ciphertext(identity["ciphertext"])
        enc_result = eval_dot_product(enc_query, enc_stored)
        score = decrypt_result(enc_result)
        logger.info("[search] %s → similarity %.6f", identity["label"], score)

        if score >= payload.threshold:
            ranked.append(FHEMatch(
                label=identity["label"],
                similarity=round(score, 4),
                metadata=identity["metadata"],
            ))

    ranked.sort(key=lambda m: m.similarity, reverse=True)
    top = ranked[: payload.top_k]

    elapsed = time.time() - t_start
    logger.info("[search] Complete in %.3fs — %d match(es)", elapsed, len(top))

    return FHESearchResponse(
        count=len(top),
        best_match=top[0] if top else None,
        matches=top,
    )


@router.post("/search/stream")
def fhe_search_stream(payload: FHESearchRequest) -> StreamingResponse:
    """SSE endpoint that streams progress during FHE search."""
    _require_native_fhe()
    try:
        query_vec = _normalize(payload.embedding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def generate():
        t_start = time.time()

        yield f"data: {json.dumps({'type': 'status', 'message': 'Encrypting query...'})}\n\n"
        enc_query = deserialize_ciphertext(encrypt_query(query_vec.tolist()))

        with get_fhe_connection() as conn:
            identities = load_fhe_identities(conn)
            identities, stale_labels = _prune_stale_identities(conn, identities, stream=True)

        total = len(identities)
        if total == 0:
            if stale_labels:
                yield f"data: {json.dumps({'type': 'status', 'message': f'Removed {len(stale_labels)} stale encrypted identities. Re-enroll them to search again.'})}\n\n"
            yield f"data: {json.dumps({'type': 'result', 'count': 0, 'best_match': None, 'matches': []})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'message': f'Comparing against {total} encrypted identities...', 'total': total, 'current': 0})}\n\n"

        ranked = []
        for i, identity in enumerate(identities):
            enc_stored = deserialize_ciphertext(identity["ciphertext"])
            enc_result = eval_dot_product(enc_query, enc_stored)
            score = decrypt_result(enc_result)
            logger.info("[search/stream] %s → %.6f", identity["label"], score)

            if score >= payload.threshold:
                ranked.append({
                    "label": identity["label"],
                    "similarity": round(score, 4),
                    "metadata": identity["metadata"],
                })

            yield f"data: {json.dumps({'type': 'progress', 'current': i + 1, 'total': total, 'label': identity['label'], 'similarity': round(score, 4)})}\n\n"

        ranked.sort(key=lambda m: m["similarity"], reverse=True)
        top = ranked[: payload.top_k]
        elapsed = time.time() - t_start

        yield f"data: {json.dumps({'type': 'result', 'count': len(top), 'best_match': top[0] if top else None, 'matches': top, 'elapsed': round(elapsed, 2)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/enroll/bulk")
def fhe_enroll_bulk(payload: dict) -> dict:
    """Bulk enroll — same as /enroll/bulk but encrypts each embedding."""
    _require_native_fhe()
    entries = payload.get("entries", [])
    if not entries or len(entries) > 50:
        raise HTTPException(status_code=422, detail="entries must be 1-50 items")

    results = []
    enrolled = 0
    errors = 0

    with get_fhe_connection() as conn:
        for entry in entries:
            try:
                label = entry["label"].strip()
                embedding = entry["embedding"]
                metadata = entry.get("metadata")
                if len(embedding) != 128:
                    raise ValueError("embedding must be 128 floats")

                unit_vec = _normalize(embedding)
                ct_bytes = encrypt_embedding(unit_vec.tolist())
                upsert_fhe_identity(conn, label, ct_bytes, metadata, commit=False)
                results.append({"label": label, "status": "saved"})
                enrolled += 1
            except Exception as exc:
                results.append({"label": entry.get("label", "?"), "status": "error", "detail": str(exc)})
                errors += 1
        conn.commit()

    logger.info("[enroll/bulk] %d enrolled, %d errors", enrolled, errors)
    return {"enrolled": enrolled, "errors": errors, "results": results}


@router.get("/identities")
def fhe_identities() -> dict[str, Any]:
    _require_native_fhe()
    with get_fhe_connection() as conn:
        identities = load_fhe_identities(conn)

    items = [
        {
            "label": i["label"],
            "metadata": i["metadata"],
            "created_at": i["created_at"],
        }
        for i in identities
    ]
    return {"count": len(items), "items": items}
