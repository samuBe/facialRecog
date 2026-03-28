import numpy as np
import pytest


def test_dot_product_roundtrip():
    """Encrypt two unit vectors, compute dot product on ciphertext, verify result."""
    from fhe.ops import fhe_dot_product_ctx

    ctx = fhe_dot_product_ctx()

    np.random.seed(42)
    a = np.random.randn(128).astype(np.float32)
    b = np.random.randn(128).astype(np.float32)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    enc_a = ctx.encrypt_arg_0(a)
    enc_b = ctx.encrypt_arg_1(b)
    result_enc = ctx.eval(enc_a, enc_b)
    result = ctx.decrypt_result(result_enc)

    expected = float(np.dot(a, b))
    assert abs(result - expected) < 1e-3, f"FHE result {result} too far from expected {expected}"


def test_dot_product_identical_vectors():
    """Dot product of a unit vector with itself should be ~1.0."""
    from fhe.ops import fhe_dot_product_ctx

    ctx = fhe_dot_product_ctx()

    a = np.ones(128, dtype=np.float32)
    a = a / np.linalg.norm(a)

    enc_a = ctx.encrypt_arg_0(a)
    enc_b = ctx.encrypt_arg_1(a)
    result_enc = ctx.eval(enc_a, enc_b)
    result = ctx.decrypt_result(result_enc)

    assert abs(result - 1.0) < 1e-3, f"Self-similarity should be ~1.0, got {result}"


def test_dot_product_orthogonal():
    """Dot product of orthogonal vectors should be ~0.0."""
    from fhe.ops import fhe_dot_product_ctx

    ctx = fhe_dot_product_ctx()

    a = np.zeros(128, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(128, dtype=np.float32)
    b[1] = 1.0

    enc_a = ctx.encrypt_arg_0(a)
    enc_b = ctx.encrypt_arg_1(b)
    result_enc = ctx.eval(enc_a, enc_b)
    result = ctx.decrypt_result(result_enc)

    assert abs(result) < 1e-3, f"Orthogonal dot product should be ~0.0, got {result}"
