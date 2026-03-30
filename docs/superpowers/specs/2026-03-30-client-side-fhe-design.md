# Client-Side FHE Encryption — Phase A: Toy Proof-of-Concept

## Goal

Validate that a browser can generate CKKS keys, encrypt values, send ciphertexts to the server, have the server compute homomorphically, and decrypt the result client-side — proving the full round-trip before integrating into the facial recognition app.

## Context

The current FHE implementation (Phase 1) has the server encrypting and decrypting all data. The client sends plaintext 128-float embeddings, and the server uses heir_py (wrapping OpenFHE/CKKS) for encryption, homomorphic dot product, and decryption. This design moves encryption and decryption to the browser so the server never sees plaintext.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Browser FHE library | openfhe-wasm | Same underlying C++ library as server (OpenFHE via heir_py) — hypothesis: serialization should be compatible. The toy validates this. |
| Key management | Client generates, sends public+eval keys once | Server caches keys per session; avoids sending keys on every request |
| Ciphertext transport | Base64-encoded in JSON | Simple, works with existing fetch-based API pattern |
| Server computation (toy) | Raw OpenFHE Python bindings (`openfhe` package) | heir_py's API likely does not support injecting external keys (see Risks). The toy endpoint uses `openfhe.EvalAdd` directly. Phase B will determine whether heir_py's `ctx.eval()` can work with client keys for the dot product. |
| CKKS parameters | Hardcoded on both sides for Phase A | Both client and server use identical parameters (ring dimension, mult depth, scaling mod size). Avoids parameter negotiation complexity in the toy. |
| Toy operation | Homomorphic addition (1 + 2 = 3) | Simplest possible operation to validate the full pipeline |

## Architecture

```
Browser (openfhe-wasm)                    Server (FastAPI + openfhe)
┌─────────────────────────┐               ┌──────────────────────────┐
│ 1. Generate CKKS keys   │               │                          │
│    (pk, sk, eval keys)  │               │                          │
│                         │   POST /fhe/  │                          │
│ 2. Send pk + eval keys  │───keys───────→│ 3. Cache client keys     │
│    + crypto context      │              │    + reconstruct context │
│                         │               │                          │
│ 4. Encrypt(1), Encrypt(2)│              │                          │
│    Serialize to base64   │  POST /fhe/  │                          │
│ 5. Send ciphertexts     │──toy-add─────→│ 6. Deserialize CTs       │
│                         │               │    EvalAdd (homomorphic) │
│                         │    base64     │    Serialize result      │
│ 8. Decrypt result ≈ 3   │←─────────────│ 7. Return encrypted sum  │
└─────────────────────────┘               └──────────────────────────┘
```

## Components

### Browser: `client/fhe-toy.html`

Standalone page that loads openfhe-wasm and provides:
- "Generate Keys" button — creates CKKS context with hardcoded parameters, generates keypair and eval keys
- "Run Test" button — encrypts 1 and 2, sends to server, decrypts result
- Status display showing each step and the final decrypted value

### Browser: openfhe-wasm build artifacts

- Built from [openfheorg/openfhe-wasm](https://github.com/openfheorg/openfhe-wasm) using Emscripten
- Must target an OpenFHE version compatible with the server's heir_py (see Risks)
- Produces `.wasm` binary + JS glue file
- Served as static assets from `client/wasm/`
- Expect a large WASM bundle (5-20MB) — acceptable for a proof-of-concept

### Server: `POST /fhe/keys` endpoint

- Accepts base64-encoded public key, evaluation keys, and serialized crypto context
- Stores in memory (single global slot for the toy)
- Returns confirmation

### Server: `POST /fhe/toy-add` endpoint

- Accepts two base64-encoded ciphertexts
- Deserializes ciphertexts using raw OpenFHE Python bindings with the cached client context/keys
- Performs `EvalAdd` (homomorphic addition)
- Serializes the encrypted result to base64
- Returns encrypted sum (client decrypts)
- Independent of existing `/fhe/*` routes — does not modify them

### Server: CKKS parameter alignment

- Both client (openfhe-wasm) and server (openfhe Python) use identical hardcoded CKKS parameters
- Parameters to align: ring dimension, multiplicative depth, scaling mod size, first mod size, security level
- The client sends the serialized crypto context alongside keys so the server can reconstruct a compatible context for deserialization

## Risks

1. **heir_py key injection (high likelihood):** The heir_py compiled context generates its own keys via `ctx.setup()` and does not expose methods to inject external keys. The toy endpoint will use raw `openfhe` Python bindings (`EvalAdd`) instead. Phase B will investigate whether heir_py's `ctx.eval()` can accept client-encrypted ciphertexts for the dot product, or whether we need to implement the dot product directly in OpenFHE.

2. **openfhe-wasm serialization API:** The WASM port exposes "a subset" of the C++ API. If `Serialize`/`Deserialize` methods for ciphertexts and keys are not in that subset, the entire approach is blocked. Verify this early.

3. **OpenFHE version mismatch:** If heir_py bundles a different OpenFHE version than what openfhe-wasm builds against, serialization formats may be incompatible. Check heir_py's bundled OpenFHE version and align openfhe-wasm build accordingly.

## What This Validates

- openfhe-wasm builds and loads in the browser
- CKKS parameter compatibility between browser OpenFHE and server OpenFHE
- Ciphertext serialization/deserialization works across the wire (base64 JSON transport)
- Client-generated keys work with server-side homomorphic evaluation
- End-to-end round-trip: client encrypt → server compute → client decrypt

## Out of Scope (Phase B)

- Integration with the facial recognition UI
- Dot product (128-element vectors)
- FHE toggle / key generation button in the main app
- Enroll/search with client-encrypted embeddings
- SSE streaming with client-side decryption
