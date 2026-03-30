"""Toy FHE endpoints for validating client-side encryption round-trip."""

from __future__ import annotations

import base64
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fhe.runtime import claim_toy_fhe

logger = logging.getLogger("fhe.toy")

router = APIRouter(prefix="/fhe", tags=["fhe-toy"])

# In-memory store for client keys (single-client toy)
_client_state: dict[str, Any] = {}


def _openfhe():
    import openfhe

    return openfhe


class KeyUploadRequest(BaseModel):
    crypto_context: str  # base64-encoded binary-serialized CryptoContext
    public_key: str  # base64-encoded binary-serialized PublicKey
    eval_keys: str | None = None  # optional, needed for EvalMult (Phase B)


class ToyAddRequest(BaseModel):
    ct_a: str  # base64-encoded binary-serialized Ciphertext
    ct_b: str  # base64-encoded binary-serialized Ciphertext


@router.post("/keys")
def upload_keys(payload: KeyUploadRequest) -> dict[str, str]:
    """Receive and cache client-generated CKKS keys."""
    reason = claim_toy_fhe()
    if reason is not None:
        raise HTTPException(status_code=503, detail=reason)

    try:
        openfhe = _openfhe()
        cc_bytes = base64.b64decode(payload.crypto_context)
        pk_bytes = base64.b64decode(payload.public_key)

        # Deserialize -- context must be deserialized first
        cc = openfhe.DeserializeCryptoContextString(cc_bytes, openfhe.BINARY)
        pk = openfhe.DeserializePublicKeyString(pk_bytes, openfhe.BINARY)

        _client_state["cc"] = cc
        _client_state["pk"] = pk

        if payload.eval_keys:
            ek_bytes = base64.b64decode(payload.eval_keys)
            _client_state["has_eval_keys"] = True

        logger.info("[keys] Client keys cached (context + public key)")
        return {"status": "ok"}

    except Exception as exc:
        logger.exception("[keys] Failed to deserialize client keys")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/toy-add")
def toy_add(payload: ToyAddRequest) -> dict[str, str]:
    """Perform homomorphic addition on two client-encrypted ciphertexts."""
    reason = claim_toy_fhe()
    if reason is not None:
        raise HTTPException(status_code=503, detail=reason)

    if "cc" not in _client_state:
        raise HTTPException(
            status_code=400,
            detail="No client keys uploaded. POST /fhe/keys first.",
        )

    try:
        openfhe = _openfhe()
        cc = _client_state["cc"]

        ct_a_bytes = base64.b64decode(payload.ct_a)
        ct_b_bytes = base64.b64decode(payload.ct_b)

        # Deserialize ciphertexts (context must be loaded already)
        ct_a = openfhe.DeserializeCiphertextString(ct_a_bytes, openfhe.BINARY)
        ct_b = openfhe.DeserializeCiphertextString(ct_b_bytes, openfhe.BINARY)

        # Homomorphic addition -- server never sees plaintext
        ct_sum = cc.EvalAdd(ct_a, ct_b)

        # Serialize result
        result_bytes = openfhe.Serialize(ct_sum, openfhe.BINARY)
        result_b64 = base64.b64encode(result_bytes).decode()

        logger.info(
            "[toy-add] Homomorphic addition complete, result: %d bytes",
            len(result_bytes),
        )
        return {"result": result_b64}

    except Exception as exc:
        logger.exception("[toy-add] Failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
