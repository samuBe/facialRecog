"""FHE API routes — encrypted versions of enroll/search/identities."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from fhe.crypto import (
    decrypt_result,
    deserialize_ciphertext,
    encrypt_embedding,
    encrypt_query,
    eval_dot_product,
)
from fhe.db import get_fhe_connection, load_fhe_identities, upsert_fhe_identity

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


@router.post("/enroll")
def fhe_enroll(payload: FHEEnrollRequest) -> dict[str, str]:
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
    t_start = time.time()
    try:
        query_vec = _normalize(payload.embedding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("[search] Encrypting query embedding")
    enc_query = deserialize_ciphertext(encrypt_query(query_vec.tolist()))

    with get_fhe_connection() as conn:
        identities = load_fhe_identities(conn)

    if not identities:
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


@router.post("/enroll/bulk")
def fhe_enroll_bulk(payload: dict) -> dict:
    """Bulk enroll — same as /enroll/bulk but encrypts each embedding."""
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
