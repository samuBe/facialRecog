"""Test cross-platform OpenFHE serialization compatibility."""
import openfhe
from pathlib import Path

PARAMS = {
    "mult_depth": 1,
    "scaling_mod_size": 50,
    "first_mod_size": 60,
    "ring_dim": 8192,
    "batch_size": 4096,
}


def make_ckks_context():
    params = openfhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(PARAMS["mult_depth"])
    params.SetScalingModSize(PARAMS["scaling_mod_size"])
    params.SetFirstModSize(PARAMS["first_mod_size"])
    params.SetRingDim(PARAMS["ring_dim"])
    params.SetBatchSize(PARAMS["batch_size"])
    params.SetSecurityLevel(openfhe.SecurityLevel.HEStd_128_classic)
    params.SetScalingTechnique(openfhe.ScalingTechnique.FLEXIBLEAUTO)

    cc = openfhe.GenCryptoContext(params)
    cc.Enable(openfhe.PKESchemeFeature.PKE)
    cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)
    return cc


def test_python_roundtrip():
    """Verify Python can serialize and deserialize."""
    cc = make_ckks_context()
    keys = cc.KeyGen()

    ptxt = cc.MakeCKKSPackedPlaintext([1.0, 2.0, 3.0])
    ct = cc.Encrypt(keys.publicKey, ptxt)

    # Serialize
    ct_bytes = openfhe.Serialize(ct, openfhe.BINARY)
    assert len(ct_bytes) > 0, "Serialized ciphertext is empty"

    # Deserialize
    ct2 = openfhe.DeserializeCiphertextString(ct_bytes, openfhe.BINARY)

    # Decrypt and verify
    result = cc.Decrypt(ct2, keys.secretKey)
    result.SetLength(3)
    vals = result.GetRealPackedValue()
    assert abs(vals[0] - 1.0) < 1e-3, f"Expected ~1.0, got {vals[0]}"
    assert abs(vals[1] - 2.0) < 1e-3, f"Expected ~2.0, got {vals[1]}"
    assert abs(vals[2] - 3.0) < 1e-3, f"Expected ~3.0, got {vals[2]}"
    print(f"Python roundtrip: PASS  (decrypted: {vals})")


def export_for_wasm():
    """Export serialized artifacts for WASM testing."""
    cc = make_ckks_context()
    keys = cc.KeyGen()

    ptxt = cc.MakeCKKSPackedPlaintext([1.0])
    ct = cc.Encrypt(keys.publicKey, ptxt)

    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(exist_ok=True)

    # Serialize to binary
    cc_bytes = openfhe.Serialize(cc, openfhe.BINARY)
    pk_bytes = openfhe.Serialize(keys.publicKey, openfhe.BINARY)
    sk_bytes = openfhe.Serialize(keys.secretKey, openfhe.BINARY)
    ct_bytes = openfhe.Serialize(ct, openfhe.BINARY)

    (out_dir / "cc.bin").write_bytes(cc_bytes)
    (out_dir / "pk.bin").write_bytes(pk_bytes)
    (out_dir / "sk.bin").write_bytes(sk_bytes)
    (out_dir / "ct.bin").write_bytes(ct_bytes)

    print(f"Exported to {out_dir}")
    print(f"  cc.bin: {len(cc_bytes)} bytes")
    print(f"  pk.bin: {len(pk_bytes)} bytes")
    print(f"  sk.bin: {len(sk_bytes)} bytes")
    print(f"  ct.bin: {len(ct_bytes)} bytes")


if __name__ == "__main__":
    test_python_roundtrip()
    export_for_wasm()
