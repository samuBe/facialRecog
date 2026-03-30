"""Tests for client-side FHE toy endpoints."""
import base64
import pytest
import openfhe
from fastapi.testclient import TestClient
from main import app


def make_test_context():
    """Create CKKS context matching the toy parameters."""
    params = openfhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(1)
    params.SetScalingModSize(50)
    params.SetRingDim(8192)
    params.SetBatchSize(4096)
    params.SetSecurityLevel(openfhe.SecurityLevel.HEStd_128_classic)
    params.SetScalingTechnique(openfhe.ScalingTechnique.FLEXIBLEAUTO)

    cc = openfhe.GenCryptoContext(params)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    return cc


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def ckks_setup():
    """Generate CKKS keys and encrypted values."""
    cc = make_test_context()
    keys = cc.KeyGen()

    ptxt1 = cc.MakeCKKSPackedPlaintext([1.0])
    ptxt2 = cc.MakeCKKSPackedPlaintext([2.0])
    ct1 = cc.Encrypt(keys.publicKey, ptxt1)
    ct2 = cc.Encrypt(keys.publicKey, ptxt2)

    return {"cc": cc, "keys": keys, "ct1": ct1, "ct2": ct2}


def _b64(obj):
    """Serialize an OpenFHE object to base64 string."""
    return base64.b64encode(openfhe.Serialize(obj, openfhe.BINARY)).decode()


def test_upload_keys(client, ckks_setup):
    """POST /fhe/keys accepts serialized public key and crypto context."""
    resp = client.post("/fhe/keys", json={
        "crypto_context": _b64(ckks_setup["cc"]),
        "public_key": _b64(ckks_setup["keys"].publicKey),
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_toy_add_no_keys(client):
    """POST /fhe/toy-add fails if no keys uploaded."""
    # Clear client state first
    from fhe_toy_routes import _client_state
    _client_state.clear()

    resp = client.post("/fhe/toy-add", json={
        "ct_a": base64.b64encode(b"fake").decode(),
        "ct_b": base64.b64encode(b"fake").decode(),
    })
    assert resp.status_code == 400
    assert "No client keys" in resp.json()["detail"]


def test_toy_add_full_roundtrip(client, ckks_setup):
    """Full: upload keys -> send CTs -> get encrypted sum -> decrypt locally ~ 3.0"""
    cc = ckks_setup["cc"]
    keys = ckks_setup["keys"]

    # 1. Upload keys
    resp = client.post("/fhe/keys", json={
        "crypto_context": _b64(cc),
        "public_key": _b64(keys.publicKey),
    })
    assert resp.status_code == 200

    # 2. Send ciphertexts for addition
    resp = client.post("/fhe/toy-add", json={
        "ct_a": _b64(ckks_setup["ct1"]),
        "ct_b": _b64(ckks_setup["ct2"]),
    })
    assert resp.status_code == 200
    result_b64 = resp.json()["result"]
    assert len(result_b64) > 0

    # 3. Deserialize and decrypt locally (simulating browser)
    result_bytes = base64.b64decode(result_b64)
    ct_sum = openfhe.DeserializeCiphertextString(result_bytes, openfhe.BINARY)

    plaintext = cc.Decrypt(ct_sum, keys.secretKey)
    plaintext.SetLength(1)
    value = plaintext.GetRealPackedValue()[0]

    assert abs(value - 3.0) < 1e-3, f"Expected ~3.0, got {value}"
