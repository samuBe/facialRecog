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


class OpenFHEEnrollRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1, max_length=80)
    encrypted_embedding: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class OpenFHEBulkEnrollRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    entries: list[dict[str, Any]] = Field(..., min_length=1, max_length=50)


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


def _get_openfhe_db_conn():
    import sqlite3
    from main import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_openfhe_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS openfhe_identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            ciphertext BLOB NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _upsert_openfhe_identity(conn, label: str, ciphertext: bytes, metadata: dict | None, *, commit: bool = True) -> None:
    import json
    conn.execute(
        """INSERT INTO openfhe_identities(label, ciphertext, metadata)
           VALUES (?, ?, ?)
           ON CONFLICT(label)
           DO UPDATE SET ciphertext=excluded.ciphertext, metadata=excluded.metadata""",
        (label, ciphertext, json.dumps(metadata) if metadata else None),
    )
    if commit:
        conn.commit()


def _load_openfhe_identities(conn) -> list[dict[str, Any]]:
    import json
    rows = conn.execute(
        "SELECT label, ciphertext, metadata, created_at FROM openfhe_identities"
    ).fetchall()
    return [
        {
            "label": row["label"],
            "ciphertext": row["ciphertext"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "created_at": row["created_at"],
        }
        for row in rows
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


@router.post("/enroll")
def openfhe_enroll(payload: OpenFHEEnrollRequest) -> dict[str, str]:
    """Enroll a client-encrypted embedding. Server never sees plaintext."""
    _require_openfhe()
    session = get_client_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="OpenFHE session not found")

    ct_bytes = _decode_b64(payload.encrypted_embedding)
    # Validate it deserializes (catches corrupt data early)
    try:
        session.deserialize_ciphertext(ct_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ciphertext: {exc}") from exc

    label = payload.label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="label cannot be empty")

    with _get_openfhe_db_conn() as conn:
        _init_openfhe_table(conn)
        _upsert_openfhe_identity(conn, label, ct_bytes, payload.metadata)

    logger.info("[enroll] '%s' enrolled (%d bytes ciphertext)", label, len(ct_bytes))
    return {"status": "saved", "label": label}


@router.post("/enroll/bulk")
def openfhe_enroll_bulk(payload: OpenFHEBulkEnrollRequest) -> dict[str, Any]:
    """Bulk enroll client-encrypted embeddings. Server never sees plaintext."""
    _require_openfhe()
    session = get_client_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="OpenFHE session not found")

    results = []
    enrolled = 0
    errors = 0

    with _get_openfhe_db_conn() as conn:
        _init_openfhe_table(conn)
        for entry in payload.entries:
            try:
                label = entry["label"].strip()
                ct_bytes = _decode_b64(entry["encrypted_embedding"])
                metadata = entry.get("metadata")

                session.deserialize_ciphertext(ct_bytes)
                _upsert_openfhe_identity(conn, label, ct_bytes, metadata, commit=False)
                results.append({"label": label, "status": "saved"})
                enrolled += 1
            except Exception as exc:
                results.append({"label": entry.get("label", "?"), "status": "error", "detail": str(exc)})
                errors += 1
        conn.commit()

    logger.info("[enroll/bulk] %d enrolled, %d errors", enrolled, errors)
    return {"enrolled": enrolled, "errors": errors, "results": results}


@router.get("/identities")
def openfhe_identities() -> dict[str, Any]:
    """List enrolled OpenFHE identities (no ciphertexts exposed)."""
    with _get_openfhe_db_conn() as conn:
        _init_openfhe_table(conn)
        identities = _load_openfhe_identities(conn)

    items = [
        {"label": i["label"], "metadata": i["metadata"], "created_at": i["created_at"]}
        for i in identities
    ]
    return {"count": len(items), "items": items}


@router.post("/search")
def openfhe_search(payload: OpenFHESearchRequest) -> dict[str, Any]:
    """Search with client-encrypted query against stored encrypted embeddings.
    Server computes ciphertext-ciphertext inner product. Never sees plaintext."""
    _require_openfhe()
    session = get_client_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="OpenFHE session not found")

    with _get_openfhe_db_conn() as conn:
        _init_openfhe_table(conn)
        identities = _load_openfhe_identities(conn)

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
            enc_stored = session.deserialize_ciphertext(identity["ciphertext"])
            encrypted_score = session.eval_inner_product(encrypted_query, enc_stored)
            serialized_score = session.serialize_ciphertext(encrypted_score)
        except Exception as exc:
            logger.exception("OpenFHE search failed for %s", identity["label"])
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
