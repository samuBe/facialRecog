"""Experimental OpenFHE routes that do not disturb the HEIR-backed API."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fhe.openfhe_backend import (
    create_client_session,
    encrypted_dot_product,
    get_client_session,
    openfhe_available_reason,
)

logger = logging.getLogger("fhe.openfhe")

router = APIRouter(prefix="/fhe-openfhe", tags=["fhe-openfhe"])


class OpenFHESearchRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    encrypted_query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.35, ge=-1.0, le=1.0)


class OpenFHEDotProductRequest(BaseModel):
    lhs: list[float] = Field(..., min_length=128, max_length=128)
    rhs: list[float] = Field(..., min_length=128, max_length=128)


class OpenFHEKeyUploadRequest(BaseModel):
    crypto_context: str = Field(..., min_length=1)
    public_key: str = Field(..., min_length=1)
    eval_automorphism_key: str | None = None
    eval_mult_key: str | None = None


def _require_openfhe() -> None:
    reason = openfhe_available_reason()
    if reason is not None:
        raise HTTPException(status_code=503, detail=reason)


def _decode_b64(value: str) -> bytes:
    try:
        return base64.b64decode(value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc


def _load_plaintext_identities() -> list[dict[str, Any]]:
    from main import get_connection, load_identities

    with get_connection() as conn:
        rows = load_identities(conn)

    return [
        {
            "label": label,
            "embedding": embedding.tolist(),
            "metadata": metadata,
        }
        for label, embedding, metadata in rows
    ]


@router.get("/health")
def openfhe_health() -> dict[str, str | None]:
    reason = openfhe_available_reason()
    return {
        "status": "ok" if reason is None else "disabled",
        "reason": reason,
    }


@router.post("/session")
def openfhe_create_session() -> dict[str, str]:
    _require_openfhe()
    session = create_client_session()
    return {"session_id": session.session_id, "status": "created"}


@router.post("/session/{session_id}/keys")
def openfhe_upload_keys(session_id: str, payload: OpenFHEKeyUploadRequest) -> dict[str, str]:
    _require_openfhe()
    session = get_client_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="OpenFHE session not found")

    try:
        session.load_client_keys(
            crypto_context_bytes=_decode_b64(payload.crypto_context),
            public_key_bytes=_decode_b64(payload.public_key),
            eval_automorphism_key_bytes=(
                _decode_b64(payload.eval_automorphism_key)
                if payload.eval_automorphism_key
                else None
            ),
            eval_mult_key_bytes=(
                _decode_b64(payload.eval_mult_key)
                if payload.eval_mult_key
                else None
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load OpenFHE client key material")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "ok", "session_id": session_id}


@router.post("/dot-product")
def openfhe_dot_product(payload: OpenFHEDotProductRequest) -> dict[str, float | str]:
    _require_openfhe()
    try:
        result = encrypted_dot_product(payload.lhs, payload.rhs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Experimental OpenFHE dot product failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result["backend"] = "openfhe"
    return result


@router.post("/search")
def openfhe_search(payload: OpenFHESearchRequest) -> dict[str, Any]:
    _require_openfhe()
    session = get_client_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="OpenFHE session not found")

    try:
        identities = _load_plaintext_identities()
    except Exception as exc:
        logger.exception("Failed to load plaintext identities for OpenFHE search")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not identities:
        return {
            "count": 0,
            "candidates": [],
            "backend": "openfhe-client",
            "elapsed_ms": 0.0,
        }

    started = time.time()
    try:
        encrypted_query = session.deserialize_ciphertext(_decode_b64(payload.encrypted_query))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to deserialize encrypted query")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    candidates: list[dict[str, Any]] = []

    for identity in identities:
        try:
            encrypted_score = session.eval_dot_product_with_plaintext(
                encrypted_query,
                identity["embedding"],
            )
            serialized_score = session.serialize_ciphertext(encrypted_score)
        except Exception as exc:
            logger.exception("Experimental OpenFHE search failed for %s", identity["label"])
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        candidates.append(
            {
                "label": identity["label"],
                "encrypted_score": base64.b64encode(serialized_score).decode(),
                "metadata": identity["metadata"],
            }
        )

    return {
        "count": len(candidates),
        "candidates": candidates,
        "backend": "openfhe-client",
        "elapsed_ms": round((time.time() - started) * 1000, 2),
        "candidate_count": len(identities),
    }
