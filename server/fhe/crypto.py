"""High-level FHE encrypt/decrypt wrappers around heir_py compiled context."""

from __future__ import annotations

import logging
import pickle
import time
import uuid
from typing import Any

import numpy as np

from fhe.ops import fhe_dot_product_ctx

logger = logging.getLogger("fhe")

# In-process ciphertext store.
# heir_py's pybind11 Ciphertext objects cannot be pickled, so we keep them in
# memory and hand callers an opaque token (UUID bytes).  For a real deployment
# this would be replaced by OpenFHE's native serialization (SerializeToFile /
# DeserializeFromFile via the C++ API).
_CT_STORE: dict[str, Any] = {}


class StaleCiphertextError(KeyError):
    """Raised when a stored ciphertext token no longer exists in process memory."""


def _store_ct(enc_value: Any) -> bytes:
    """Persist an EncValue in the in-process store; return an opaque token."""
    key = str(uuid.uuid4())
    _CT_STORE[key] = enc_value
    token = key.encode()
    # NOTE: pickle is used for demo only. For production, use OpenFHE native serialization.
    return pickle.dumps(token)


def encrypt_embedding(embedding: list[float]) -> bytes:
    """Encrypt an embedding for storage (arg_0 slot). Returns serialized ciphertext."""
    ctx = fhe_dot_product_ctx()
    arr = np.asarray(embedding, dtype=np.float32)

    t0 = time.time()
    encrypted = ctx.encrypt_arg_0(arr)
    elapsed = time.time() - t0
    logger.info("Encrypted embedding (arg_0) in %.3fs", elapsed)

    return _store_ct(encrypted)


def encrypt_query(embedding: list[float]) -> bytes:
    """Encrypt a query embedding (arg_1 slot). Returns serialized ciphertext."""
    ctx = fhe_dot_product_ctx()
    arr = np.asarray(embedding, dtype=np.float32)

    t0 = time.time()
    encrypted = ctx.encrypt_arg_1(arr)
    elapsed = time.time() - t0
    logger.info("Encrypted query (arg_1) in %.3fs", elapsed)

    return _store_ct(encrypted)


def deserialize_ciphertext(ct_bytes: bytes) -> Any:
    """Deserialize ciphertext from stored bytes."""
    # NOTE: pickle.loads on untrusted data is a security risk. Demo only.
    token = pickle.loads(ct_bytes)
    key = token.decode()
    if key not in _CT_STORE:
        raise StaleCiphertextError(f"Ciphertext token not found in store: {key}")
    return _CT_STORE[key]


def eval_dot_product(enc_query: Any, enc_stored: Any) -> Any:
    """Compute homomorphic dot product of two encrypted embeddings."""
    ctx = fhe_dot_product_ctx()

    t0 = time.time()
    result = ctx.eval(enc_stored, enc_query)
    elapsed = time.time() - t0
    logger.info("Homomorphic dot product in %.3fs", elapsed)

    return result


def decrypt_result(enc_result: Any) -> float:
    """Decrypt a single encrypted dot product result."""
    ctx = fhe_dot_product_ctx()
    return float(ctx.decrypt_result(enc_result))
