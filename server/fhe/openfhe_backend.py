"""Experimental OpenFHE-backed encrypted dot products for sidecar routes."""

from __future__ import annotations

import importlib
import logging
import time
from functools import lru_cache
from typing import Any
from uuid import uuid4

import numpy as np

EMBEDDING_DIM = 128
OPENFHE_RING_DIM = 16384

logger = logging.getLogger("fhe.openfhe")
ROTATION_STEPS = [1, 2, 4, 8, 16, 32, 64]


def openfhe_available_reason() -> str | None:
    if importlib.util.find_spec("openfhe") is None:
        return "openfhe not installed"
    return None


def _openfhe():
    return importlib.import_module("openfhe")


def _enable_ckks_features(cc: Any, openfhe: Any) -> None:
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)


def _decrypt_plaintext(cc: Any, ciphertext: Any, secret_key: Any) -> Any:
    try:
        return cc.Decrypt(ciphertext, secret_key)
    except TypeError:
        return cc.Decrypt(secret_key, ciphertext)


def _real_values(plaintext: Any, *, expected_length: int) -> list[float]:
    if hasattr(plaintext, "SetLength"):
        plaintext.SetLength(expected_length)
    if hasattr(plaintext, "GetRealPackedValue"):
        return [float(v) for v in plaintext.GetRealPackedValue()]
    return [float(plaintext[i].real) for i in range(expected_length)]


class OpenFHEDotProductContext:
    def __init__(self) -> None:
        openfhe = _openfhe()

        params = openfhe.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(2)
        params.SetScalingModSize(50)
        if hasattr(params, "SetFirstModSize"):
            params.SetFirstModSize(60)
        params.SetRingDim(OPENFHE_RING_DIM)
        params.SetBatchSize(EMBEDDING_DIM)
        params.SetSecurityLevel(openfhe.SecurityLevel.HEStd_128_classic)
        params.SetScalingTechnique(openfhe.ScalingTechnique.FLEXIBLEAUTO)

        cc = openfhe.GenCryptoContext(params)
        _enable_ckks_features(cc, openfhe)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)

        self._openfhe = openfhe
        self.cc = cc
        self.keys = keys

        self._rotation_steps = ROTATION_STEPS
        self._rotation_keys_ready = False
        for keygen_name in ("EvalRotateKeyGen", "EvalAtIndexKeyGen"):
            if hasattr(cc, keygen_name):
                try:
                    getattr(cc, keygen_name)(keys.secretKey, self._rotation_steps)
                    self._rotation_keys_ready = True
                    break
                except Exception:
                    logger.debug("OpenFHE %s unavailable", keygen_name, exc_info=True)

    def encrypt_vector(self, embedding: list[float]) -> Any:
        plaintext = self.cc.MakeCKKSPackedPlaintext(embedding)
        return self.cc.Encrypt(self.keys.publicKey, plaintext)

    def _rotate(self, ciphertext: Any, step: int) -> Any:
        if hasattr(self.cc, "EvalRotate"):
            return self.cc.EvalRotate(ciphertext, step)
        if hasattr(self.cc, "EvalAtIndex"):
            return self.cc.EvalAtIndex(ciphertext, step)
        raise AttributeError("No ciphertext rotation API available")

    def eval_dot_product(self, lhs: Any, rhs: Any) -> tuple[Any, str]:
        product = self.cc.EvalMult(lhs, rhs)

        if hasattr(self.cc, "EvalSum"):
            try:
                return self.cc.EvalSum(product, EMBEDDING_DIM), "ciphertext-eval-sum"
            except Exception:
                logger.debug("OpenFHE EvalSum failed, falling back", exc_info=True)

        if self._rotation_keys_ready:
            try:
                acc = product
                for step in self._rotation_steps:
                    acc = self.cc.EvalAdd(acc, self._rotate(acc, step))
                return acc, "ciphertext-rotate-add"
            except Exception:
                logger.debug("OpenFHE rotate-add reduction failed, falling back", exc_info=True)

        return product, "decrypt-reduce"

    def decrypt_score(self, ciphertext: Any, reduction_mode: str) -> float:
        plaintext = _decrypt_plaintext(self.cc, ciphertext, self.keys.secretKey)
        values = _real_values(
            plaintext,
            expected_length=1 if reduction_mode != "decrypt-reduce" else EMBEDDING_DIM,
        )
        if reduction_mode == "decrypt-reduce":
            return float(sum(values[:EMBEDDING_DIM]))
        return float(values[0])


class OpenFHEClientSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._openfhe = _openfhe()
        self.cc: Any | None = None
        self.public_key: Any | None = None
        self.ready = False

    def load_client_keys(
        self,
        *,
        crypto_context_bytes: bytes,
        public_key_bytes: bytes,
        eval_automorphism_key_bytes: bytes | None = None,
        eval_mult_key_bytes: bytes | None = None,
    ) -> None:
        openfhe = self._openfhe

        self.cc = openfhe.DeserializeCryptoContextString(
            crypto_context_bytes,
            openfhe.BINARY,
        )
        self.public_key = openfhe.DeserializePublicKeyString(
            public_key_bytes,
            openfhe.BINARY,
        )

        if eval_mult_key_bytes:
            openfhe.DeserializeEvalMultKeyString(eval_mult_key_bytes, openfhe.BINARY)
        if eval_automorphism_key_bytes:
            openfhe.DeserializeEvalAutomorphismKeyString(
                eval_automorphism_key_bytes,
                openfhe.BINARY,
            )

        self.ready = True

    def ensure_ready(self) -> None:
        if not self.ready or self.cc is None or self.public_key is None:
            raise RuntimeError("OpenFHE client session keys not uploaded")

    def deserialize_ciphertext(self, ciphertext_bytes: bytes) -> Any:
        self.ensure_ready()
        return self._openfhe.DeserializeCiphertextString(
            ciphertext_bytes,
            self._openfhe.BINARY,
        )

    def serialize_ciphertext(self, ciphertext: Any) -> bytes:
        self.ensure_ready()
        return self._openfhe.Serialize(ciphertext, self._openfhe.BINARY)

    def eval_dot_product_with_plaintext(self, encrypted_query: Any, embedding: list[float]) -> Any:
        self.ensure_ready()
        plaintext = self.cc.MakeCKKSPackedPlaintext(embedding)
        acc = self.cc.EvalMult(encrypted_query, plaintext)

        for step in ROTATION_STEPS:
            acc = self.cc.EvalAdd(acc, self.cc.EvalAtIndex(acc, step))

        return acc


_CLIENT_SESSIONS: dict[str, OpenFHEClientSession] = {}


def create_client_session() -> OpenFHEClientSession:
    session_id = uuid4().hex
    session = OpenFHEClientSession(session_id)
    _CLIENT_SESSIONS[session_id] = session
    return session


def get_client_session(session_id: str) -> OpenFHEClientSession | None:
    return _CLIENT_SESSIONS.get(session_id)


@lru_cache(maxsize=1)
def openfhe_dot_product_ctx() -> OpenFHEDotProductContext:
    logger.info("Initializing experimental OpenFHE CKKS context...")
    started = time.time()
    ctx = OpenFHEDotProductContext()
    logger.info("Experimental OpenFHE ready in %.3fs", time.time() - started)
    return ctx


def normalize_embedding(embedding: list[float]) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Embedding norm cannot be zero")
    return arr / norm


def encrypted_dot_product(lhs: list[float], rhs: list[float]) -> dict[str, float | str]:
    ctx = openfhe_dot_product_ctx()
    lhs_unit = normalize_embedding(lhs).tolist()
    rhs_unit = normalize_embedding(rhs).tolist()

    started = time.time()
    enc_lhs = ctx.encrypt_vector(lhs_unit)
    enc_rhs = ctx.encrypt_vector(rhs_unit)
    enc_result, reduction_mode = ctx.eval_dot_product(enc_lhs, enc_rhs)
    score = ctx.decrypt_score(enc_result, reduction_mode)
    elapsed_ms = round((time.time() - started) * 1000, 2)

    return {
        "similarity": score,
        "reduction_mode": reduction_mode,
        "elapsed_ms": elapsed_ms,
    }
