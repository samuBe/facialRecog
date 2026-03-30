/**
 * Test cross-platform OpenFHE serialization: WASM deserializes Python-exported artifacts.
 *
 * Reads cc.bin, pk.bin, ct.bin from fixtures/, deserializes them,
 * encrypts 2.0, computes EvalAdd(ct_from_python, ct_of_2),
 * serializes the result to fixtures/ct_sum_from_wasm.bin.
 */
import { createRequire } from "module";
import { readFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const wasmPath = join(__dirname, "../../client/wasm/openfhe_pke.js");

async function main() {
  const initOpenFHE = require(wasmPath);
  const openfhe = await initOpenFHE();

  const fixtureDir = join(__dirname, "fixtures");

  // Read Python-exported files
  const ccBuf = readFileSync(join(fixtureDir, "cc.bin"));
  const pkBuf = readFileSync(join(fixtureDir, "pk.bin"));
  const ctBuf = readFileSync(join(fixtureDir, "ct.bin"));

  console.log(`Read fixtures: cc=${ccBuf.length}B, pk=${pkBuf.length}B, ct=${ctBuf.length}B`);

  // Deserialize
  const cc = openfhe.DeserializeCryptoContextFromBuffer(
    new Uint8Array(ccBuf),
    openfhe.SerType.BINARY
  );
  console.log("Deserialized CryptoContext:", cc ? "OK" : "FAIL");

  const pk = openfhe.DeserializePublicKeyFromBuffer(
    new Uint8Array(pkBuf),
    openfhe.SerType.BINARY
  );
  console.log("Deserialized PublicKey:", pk ? "OK" : "FAIL");

  const ctFromPython = openfhe.DeserializeCiphertextFromBuffer(
    new Uint8Array(ctBuf),
    openfhe.SerType.BINARY
  );
  console.log("Deserialized Ciphertext:", ctFromPython ? "OK" : "FAIL");

  // Encrypt the value 2.0
  const vec = new openfhe.VectorDouble();
  vec.push_back(2.0);
  const ptxt = cc.MakeCKKSPackedPlaintext(vec);
  const ctTwo = cc.Encrypt(pk, ptxt);
  console.log("Encrypted 2.0: OK");

  // EvalAdd (WASM uses EvalAddCipherCipher)
  const ctSum = cc.EvalAddCipherCipher(ctFromPython, ctTwo);
  console.log("EvalAdd(1.0, 2.0): OK");

  // Serialize result
  const sumBuf = openfhe.SerializeCiphertextToBuffer(ctSum, openfhe.SerType.BINARY);
  writeFileSync(join(fixtureDir, "ct_sum_from_wasm.bin"), Buffer.from(sumBuf));
  console.log(`Wrote ct_sum_from_wasm.bin: ${sumBuf.length} bytes`);

  // Also test WASM-originated encryption (Step 4 prep)
  console.log("\n--- Step 4: WASM encrypts, export for Python ---");

  // Create fresh context with same params
  const params = new openfhe.CCParamsCryptoContextCKKSRNS();
  params.SetMultiplicativeDepth(1);
  params.SetScalingModSize(50);
  params.SetRingDim(8192);
  params.SetBatchSize(4096);
  params.SetSecurityLevel(openfhe.SecurityLevel.HEStd_128_classic);
  params.SetScalingTechnique(openfhe.ScalingTechnique.FLEXIBLEAUTO);

  const cc2 = openfhe.GenCryptoContextCKKS(params);
  cc2.Enable(openfhe.PKESchemeFeature.PKE);
  cc2.Enable(openfhe.PKESchemeFeature.KEYSWITCH);
  cc2.Enable(openfhe.PKESchemeFeature.LEVELEDSHE);

  const keys2 = cc2.KeyGen();

  // Encrypt 1.0 and 2.0
  const vec1 = new openfhe.VectorDouble();
  vec1.push_back(1.0);
  const vec2 = new openfhe.VectorDouble();
  vec2.push_back(2.0);

  const pt1 = cc2.MakeCKKSPackedPlaintext(vec1);
  const pt2 = cc2.MakeCKKSPackedPlaintext(vec2);
  const ct1 = cc2.Encrypt(keys2.publicKey, pt1);
  const ct2 = cc2.Encrypt(keys2.publicKey, pt2);

  // Serialize all
  const wasmDir = join(fixtureDir, "wasm_originated");
  const { mkdirSync } = await import("fs");
  mkdirSync(wasmDir, { recursive: true });

  const cc2Buf = openfhe.SerializeCryptoContextToBuffer(cc2, openfhe.SerType.BINARY);
  const pk2Buf = openfhe.SerializePublicKeyToBuffer(keys2.publicKey, openfhe.SerType.BINARY);
  const sk2Buf = openfhe.SerializePrivateKeyToBuffer(keys2.secretKey, openfhe.SerType.BINARY);
  const ct1Buf = openfhe.SerializeCiphertextToBuffer(ct1, openfhe.SerType.BINARY);
  const ct2Buf = openfhe.SerializeCiphertextToBuffer(ct2, openfhe.SerType.BINARY);

  writeFileSync(join(wasmDir, "cc.bin"), Buffer.from(cc2Buf));
  writeFileSync(join(wasmDir, "pk.bin"), Buffer.from(pk2Buf));
  writeFileSync(join(wasmDir, "sk.bin"), Buffer.from(sk2Buf));
  writeFileSync(join(wasmDir, "ct1.bin"), Buffer.from(ct1Buf));
  writeFileSync(join(wasmDir, "ct2.bin"), Buffer.from(ct2Buf));

  console.log(`Exported WASM-originated files to ${wasmDir}`);
  console.log(`  cc.bin: ${cc2Buf.length}B, pk.bin: ${pk2Buf.length}B, sk.bin: ${sk2Buf.length}B`);
  console.log(`  ct1.bin: ${ct1Buf.length}B, ct2.bin: ${ct2Buf.length}B`);

  console.log("\nAll WASM tests: PASS");
}

main().catch((e) => {
  console.error("WASM test FAILED:", e);
  process.exit(1);
});
