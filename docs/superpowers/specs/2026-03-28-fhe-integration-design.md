# FHE Integration Design — Server-Side CKKS Demo

**Date:** 2026-03-28
**Status:** Approved
**Scope:** Phase 1 — Server-side FHE with openfhe-python (HEIR stretch goal)

## Goal

Add Fully Homomorphic Encryption (CKKS scheme) to the facial recognition system so that face embeddings can be matched via cosine similarity computed entirely on ciphertext. In Phase 1, the server performs encryption/decryption (demo mode). Phase 2 (future) moves encryption to the browser.

Long-term target is Desilo pi-heaan for production FHE. This phase uses OpenFHE + Google HEIR as a stepping stone.

## Architecture

```
Client (unchanged)                Server (FastAPI)
─────────────────                ──────────────────────────────────────
face-api.js                      POST /fhe/enroll
  → 128 floats (plaintext) ────→   openfhe-python encrypts embedding
                                    serializes ciphertext → SQLite

face-api.js                      POST /fhe/search
  → 128 floats (plaintext) ────→   openfhe-python encrypts query
                                    homomorphic dot-product on ciphertext pairs
                                    decrypts similarity scores
                                    returns ranked matches

                                 GET /fhe/identities
                                    returns metadata (no ciphertext)
```

### Key Insight

Cosine similarity of two unit vectors = their dot product. The client already L2-normalizes before sending, and the server normalizes again. So the homomorphic operation we need is just a dot product of two 128-element encrypted vectors.

## Components

### 1. FHE Crypto Module — `server/fhe/crypto.py`

Wraps openfhe-python for CKKS operations:

- `generate_keys()` → (public_key, secret_key, eval_keys, crypto_context)
- `encrypt(embedding: list[float], public_key, crypto_context)` → serialized ciphertext bytes
- `decrypt(ciphertext_bytes, secret_key, crypto_context)` → list[float]
- `serialize_key(key)` → bytes, `deserialize_key(bytes)` → key
- `load_ciphertext(bytes, crypto_context)` → ciphertext object

**Key persistence:** Keys are generated on first startup, serialized to `server/fhe/keys/` (secret_key.bin, public_key.bin, eval_keys.bin, crypto_context.bin). On subsequent startups, keys are loaded from disk. If key files are missing or corrupted, new keys are generated and all existing `fhe_identities` rows are purged (ciphertext encrypted under old keys is unrecoverable).

**State storage:** Crypto context and keys are stored on `app.state` during startup, accessed by route handlers via FastAPI dependency injection.

**CKKS Parameters (starting point):**
- Multiplicative depth: 2 (depth 1 for element-wise multiply, depth 2 as headroom for rescaling overhead)
- Ring dimension: 8192 (may increase to 16384 if accuracy requires it)
- Scaling mod size: 40 bits
- First mod size: 50 bits
- Batch size: 128 (matches embedding dimension)

### 2. Homomorphic Dot Product — `server/fhe/dot_product.py`

**Default implementation (Phase 1):** Pure openfhe-python using the standard rotation-and-sum pattern:
- Element-wise homomorphic multiplication of two ciphertexts
- Rotation-and-sum accumulation (log2(128) = 7 rotations + 7 additions)

This is implemented as a Python function calling openfhe-python's `EvalMult`, `EvalRotate`, and `EvalAdd` operations directly.

**HEIR stretch goal:** Compile the same dot product logic via Google HEIR targeting the OpenFHE backend. This would produce a C++ shared library (`.dylib` on macOS) that operates on serialized ciphertext byte buffers. Integration requires a C-wrapper layer to marshal between Python and the HEIR-generated code (serialize ciphertext → pass byte buffer → deserialize result). This is deferred unless the openfhe-python path proves too slow.

The `dot_product()` function interface is the same regardless of backend:
```python
def dot_product(ct_a, ct_b, crypto_context, eval_keys) -> ciphertext:
    """Compute dot product of two encrypted 128-element vectors."""
```

### 3. FHE API Routes — `server/fhe_routes.py`

New FastAPI router mounted at `/fhe/`.

#### `POST /fhe/enroll`

**Request** (same format as `/enroll`):
```json
{
  "label": "Agent Nova",
  "embedding": [0.11, -0.08, ...],  // 128 floats
  "metadata": {"role": "Field Operative"}  // optional
}
```

**Server logic:**
1. Validate label and embedding (reuse existing Pydantic models)
2. L2-normalize the embedding
3. Encrypt with CKKS → serialized ciphertext
4. Store in `fhe_identities` table (label, ciphertext blob, metadata)

**Response:**
```json
{
  "status": "saved",
  "label": "Agent Nova"
}
```

#### `POST /fhe/search`

**Request** (same format as `/search`):
```json
{
  "embedding": [0.11, -0.08, ...],  // 128 floats
  "top_k": 5,
  "threshold": 0.35
}
```

**Server logic:**
1. Validate and L2-normalize the query embedding
2. Encrypt query with CKKS
3. Load all stored ciphertexts from `fhe_identities`
4. For each stored ciphertext: compute homomorphic dot product
5. Decrypt each result to get similarity score (single float)
6. Filter by threshold, sort descending, take top_k

**Response** (same format as `/search`):
```json
{
  "count": 2,
  "best_match": {"label": "Agent Nova", "similarity": 0.8721, "metadata": {}},
  "matches": [...]
}
```

#### `POST /fhe/enroll/bulk`

Same pattern as `/enroll/bulk` — iterate and encrypt each entry. Max 50 entries.

#### `GET /fhe/identities`

Returns metadata only (label, created_at, metadata). No ciphertext in response.

### 4. Database Schema

New table alongside existing `identities`:

```sql
CREATE TABLE fhe_identities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL UNIQUE,
    ciphertext BLOB NOT NULL,
    metadata TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

Ciphertext is stored as a binary blob (OpenFHE serialization).

### 5. Client Changes

**None in Phase 1.** The client sends the same plaintext JSON it sends today. The FHE endpoints accept the same request format. The only difference is which URL the client hits.

A simple UI toggle could switch between `/enroll` and `/fhe/enroll` (optional, low priority).

## Dependencies

### Python (server)
- `openfhe` (openfhe-python) — CKKS encryption, decryption, key generation
- Existing: FastAPI, uvicorn, numpy, pydantic

### Build tools (HEIR stretch goal only)
- Google HEIR — compile dot product to OpenFHE-backed shared library
- Bazel (HEIR build system)
- OpenFHE C++ library (HEIR compilation target)

## Performance Considerations

- **Key generation:** ~1-5 seconds at startup. One-time cost.
- **Encryption:** ~10-50ms per embedding (128 slots in one ciphertext).
- **Dot product:** One homomorphic multiply + 7 rotations + 7 additions per comparison. ~50-200ms per pair.
- **Search across N identities:** O(N) comparisons. For demo scale (<100 identities), this is acceptable.
- **Ciphertext size:** ~32KB-128KB per embedding (much larger than 128 floats). SQLite handles this fine at demo scale.

## Error Handling

- FHE operations that fail (e.g., parameter mismatch, corrupted ciphertext) return 500 with a generic error message.
- Validation errors (bad embedding length, empty label) return 422, same as existing endpoints.
- If FHE initialization fails at startup (e.g., openfhe not installed), log a warning and disable `/fhe/` endpoints. Server still serves plaintext endpoints.
- `/health` endpoint extended to report FHE status: `{"status": "ok", "fhe": "ok"}` or `{"status": "ok", "fhe": "disabled"}`.

## Testing Strategy

- **Unit tests:** Encrypt → dot product → decrypt roundtrip with known vectors. Verify similarity scores match plaintext computation within CKKS approximation tolerance (~1e-4).
- **Integration tests:** Full enroll + search flow through FastAPI endpoints. Compare FHE results against plaintext endpoint results for same inputs.
- **Accuracy validation:** Run existing seed embeddings through both pipelines, verify match rankings are identical.
- **Threshold tolerance:** CKKS introduces approximation noise (~1e-4). For scores near the threshold boundary, this could flip match/no-match decisions. Tests should verify that FHE and plaintext pipelines produce the same ranking order, and the FHE threshold should be set slightly below the plaintext threshold (e.g., `threshold - 0.001`) to account for noise.

## Future Work (Phase 2)

- Move encryption to browser via openfhe-wasm
- Client generates keys, sends public + eval keys to server
- Server never sees plaintext embeddings
- Migrate HEIR backend to Desilo pi-heaan (inner product + argmax patterns from their docs)

## File Structure

```
server/
├── main.py                          # Existing — adds FHE router import + startup hook
├── fhe_routes.py                    # New FHE API router
├── fhe/
│   ├── __init__.py
│   ├── crypto.py                    # OpenFHE CKKS wrapper
│   ├── dot_product.py               # Homomorphic dot product (openfhe-python)
│   ├── keys/                        # Serialized keys (gitignored)
│   └── db.py                        # FHE identities DB operations
└── requirements.txt                 # Updated with openfhe
```
