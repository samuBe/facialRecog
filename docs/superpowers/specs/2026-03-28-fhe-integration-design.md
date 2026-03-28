# FHE Integration Design — Server-Side CKKS Demo

**Date:** 2026-03-28
**Status:** Approved
**Scope:** Phase 1 — Server-side FHE via `heir_py` Python API, full encrypted pipeline

## Goal

Add Fully Homomorphic Encryption (CKKS scheme) to the facial recognition system. The entire matching pipeline — dot products and argmax — runs on ciphertext. The server only decrypts the final one-hot result indicating the best match. In Phase 1, the server performs encryption/decryption (demo mode). Phase 2 (future) moves encryption to the browser.

Long-term target is Desilo pi-heaan for production FHE. This phase uses Google HEIR's Python API (`heir_py`) as a stepping stone.

## Architecture

```
Client (unchanged)                Server
─────────────────                ──────────────────────────────────────
face-api.js                      POST /fhe/enroll
  → 128 floats (plaintext) ────→   heir_py encrypts embedding
                                    serializes ciphertext → SQLite

face-api.js                      POST /fhe/search
  → 128 floats (plaintext) ────→   heir_py encrypts query
                                    dot_product() on each stored ciphertext
                                    → encrypted similarity scores vector
                                    argmax() on scores vector (all on ciphertext)
                                    → encrypted one-hot result
                                    decrypt one-hot → identify best match
                                    return match

                                 GET /fhe/identities
                                    returns metadata (no ciphertext)
```

### Key Insight

Cosine similarity of two unit vectors = their dot product. The client already L2-normalizes before sending, and the server normalizes again. So the homomorphic operation we need is:
1. **Dot product** of two encrypted 128-element vectors (for each enrolled identity)
2. **Argmax** over the resulting similarity scores (to find best match without decrypting scores)

## HEIR Python API (`heir_py`)

Instead of writing MLIR and compiling manually, we use HEIR's `@compile` decorator to define FHE functions in pure Python. HEIR handles CKKS parameter selection, key generation, encryption, and code generation internally.

### Compiled FHE Functions

#### Dot Product

```python
from heir import compile
from heir.mlir import F32, Secret, Tensor

@compile(scheme="ckks", backend="openfhe")
def dot_product(a: Secret[Tensor[128, F32]], b: Secret[Tensor[128, F32]]) -> Secret[F32]:
    result = 0.0
    for i in range(128):
        result += a[i] * b[i]
    return result
```

Usage:
```python
dot_product.setup()
enc_a = dot_product.encrypt_a(embedding_a)
enc_b = dot_product.encrypt_b(embedding_b)
result_enc = dot_product.eval(enc_a, enc_b)
score = dot_product.decrypt_result(result_enc)  # only for testing
```

#### Sign Function (Polynomial Approximation)

Adapted from Desilo's example. Approximates sign(x) in [-1, 1] using composed degree-7 polynomials:

```python
@compile(scheme="ckks", backend="openfhe")
def sign(x: Secret[F32]) -> Secret[F32]:
    # Polynomial p_{7,1}(x)
    p71 = [3.60471572e-36, 7.30445165, -5.05471704e-35, -3.46825871e1,
           1.16564665e-34, 5.98596518e1, -6.54298493e-35, -3.18755226e1]
    y = p71[0]
    for i in range(1, 8):
        y = y * x + p71[i]  # Horner's method — but reversed, may need adjustment

    # Polynomial p_{7,2}(y)
    p72 = [-9.46491402e-49, 2.40085652, 6.41744633e-48, -2.63125454,
           -7.25338565e-48, 1.54912675, 2.06916466e-48, -3.31172957e-1]
    result = p72[0]
    for i in range(1, 8):
        result = result * y + p72[i]
    return result
```

Note: The exact `heir_py` API for polynomial evaluation and CKKS slot operations needs validation during implementation. The polynomial coefficients are taken directly from Desilo's documentation.

#### Argmax

Adapted from Desilo's algorithm. Finds the index of the maximum value in an encrypted vector:

```python
def fhe_max(a, b):
    """max(a,b) = 0.5 * (a + b + |a - b|), where |x| = x * sign(x)"""
    diff = a - b
    abs_diff = diff * sign(diff)
    return 0.5 * (a + b + abs_diff)

def quick_max(scores, n):
    """Find max value via log2(n) rotation-and-max iterations.
    Requires bootstrapping between iterations to restore depth."""
    for i in range(log2(n)):
        rotated = rotate(scores, 2**i)
        scores = fhe_max(scores, rotated)
        scores = bootstrap(scores)  # restore multiplicative depth
    return scores

def argmax(scores, n):
    """Returns one-hot vector: 1 at position of max, 0 elsewhere."""
    max_val = quick_max(scores, n)
    diff = scores - max_val
    signs = sign(diff)        # -1 where less than max, ~0 at max
    return signs + 1          # 0 where less than max, ~1 at max
```

The sign function consumes significant multiplicative depth. Each `fhe_max` call uses ~10 levels. `quick_max` with N identities requires log2(N) iterations, each needing bootstrapping to restore depth.

### HEIR Bootstrapping

HEIR has CKKS bootstrapping support (`tests/Examples/openfhe/ckks/simple_ckks_bootstrapping/` in the repo). This is critical for argmax since the sign polynomial composition is deep. The `@compile` decorator with appropriate parameters should handle bootstrapping insertion, but this needs validation.

## Components

### 1. FHE Operations Module — `server/fhe/ops.py`

Contains the `@compile`-decorated functions:
- `dot_product(a, b)` — 128-element encrypted dot product
- `sign(x)` — polynomial approximation of sign function
- `fhe_search(query, stored_embeddings)` — orchestrates dot products + argmax

### 2. Key Management — `server/fhe/keys.py`

Uses HEIR-generated setup functions.

**Key persistence:** Keys are generated on first startup, serialized to `server/fhe/keys/`. On subsequent startups, loaded from disk. If missing/corrupted, regenerate and purge `fhe_identities` (old ciphertext is unrecoverable).

**State storage:** Crypto context and keys on `app.state`, accessed via FastAPI dependency injection.

### 3. FHE API Routes — `server/fhe_routes.py`

New FastAPI router mounted at `/fhe/`.

#### `POST /fhe/enroll`

**Request** (same format as `/enroll`):
```json
{
  "label": "Agent Nova",
  "embedding": [0.11, -0.08, ...],
  "metadata": {"role": "Field Operative"}
}
```

**Server logic:**
1. Validate label and embedding (reuse existing Pydantic models)
2. L2-normalize the embedding
3. Encrypt via `dot_product.encrypt_a()`
4. Serialize ciphertext → store in `fhe_identities` table

**Response:**
```json
{"status": "saved", "label": "Agent Nova"}
```

#### `POST /fhe/search`

**Request** (same format as `/search`):
```json
{
  "embedding": [0.11, -0.08, ...],
  "top_k": 5,
  "threshold": 0.35
}
```

**Server logic:**
1. Validate and L2-normalize the query embedding
2. Encrypt query
3. Load all stored ciphertexts from `fhe_identities`
4. For each stored ciphertext: compute encrypted dot product → collect into encrypted scores vector
5. Run `argmax()` on encrypted scores vector (includes bootstrapping)
6. Decrypt the one-hot result → identify which position(s) had highest similarity
7. Map positions back to identity labels

**Response** (same format as `/search`):
```json
{
  "count": 1,
  "best_match": {"label": "Agent Nova", "similarity": null, "metadata": {}},
  "matches": [...]
}
```

Note: With full encrypted argmax, individual similarity scores are not available (we only decrypt the one-hot result). The `similarity` field will be `null` unless we also decrypt scores separately for display purposes.

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

Ciphertext stored as binary blob (OpenFHE serialization format).

### 5. Client Changes

**None in Phase 1.** The client sends the same plaintext JSON. A UI toggle to switch between `/enroll` and `/fhe/enroll` is optional/low priority.

## Dependencies

### Python (server)
- `heir_py` — HEIR's Python package (includes heir-opt, heir-translate, OpenFHE integration)
- Existing: FastAPI, uvicorn, numpy, pydantic

### System
- OpenFHE C++ library (required by heir_py at runtime)
- C++ compiler (for HEIR's JIT compilation of decorated functions)

## Performance Considerations

- **Key generation:** ~1-5 seconds at startup (one-time).
- **`@compile` JIT:** First call to each decorated function triggers HEIR compilation. Cache the compiled result.
- **Encryption:** ~10-50ms per embedding.
- **Dot product:** ~50-200ms per pair.
- **Sign function:** Two degree-7 polynomial evaluations. ~100-500ms (significant multiplicative depth).
- **Bootstrapping:** ~500ms-2s per bootstrap operation. Required log2(N) times in quick_max.
- **Full search (N identities):** N dot products + log2(N) bootstrapped max iterations. For N=10: ~2-5s for dot products + ~5-20s for argmax. Total ~7-25s.
- **Ciphertext size:** ~32KB-128KB per embedding. SQLite handles this at demo scale.

The argmax is the expensive part. For a demo with <20 identities this is acceptable. For production, Desilo's optimized bootstrapping would be significantly faster.

## Error Handling

- FHE operations that fail return 500 with generic error.
- Validation errors return 422, same as existing.
- If `heir_py` import fails at startup, log warning and disable `/fhe/` endpoints. Plaintext endpoints unaffected.
- `/health` extended: `{"status": "ok", "fhe": "ok"}` or `{"status": "ok", "fhe": "disabled"}`.

## Testing Strategy

- **heir_py smoke test:** Verify `@compile` decorated dot product works with 128-element vectors.
- **Sign function accuracy:** Test against numpy sign for values in [-1, 1]. Verify approximation error < 0.01.
- **Argmax accuracy:** Encrypt known score vectors, run argmax, verify correct index identified.
- **Roundtrip test:** Encrypt → dot product → decrypt. Verify scores match plaintext within CKKS tolerance (~1e-3).
- **Integration tests:** Full enroll + search flow through FastAPI. Compare FHE best_match against plaintext endpoint for same inputs.
- **Bootstrapping test:** Verify argmax works correctly after bootstrapping restores depth.

## Implementation Risks

1. **`heir_py` CKKS support maturity:** The `@compile` examples in docs show `Secret[I64]` (integers). CKKS with `F32` tensors may need different API patterns. Validate early.
2. **Bootstrapping in heir_py:** HEIR has CKKS bootstrapping tests in C++, but exposing this through `@compile` decorator may require manual configuration. Fallback: use heir_py for dot product, implement argmax with openfhe-python directly.
3. **Polynomial evaluation:** The sign function's composed polynomials need enough multiplicative depth. HEIR's automatic depth management should handle this, but validate.
4. **Scores vector packing:** Collecting N dot product results into a single ciphertext for argmax requires packing/rotation operations. This may need manual ciphertext manipulation beyond what `@compile` supports.

## Future Work (Phase 2)

- Move encryption to browser via openfhe-wasm
- Client generates keys, sends public + eval keys to server
- Server never sees plaintext embeddings
- Migrate to Desilo pi-heaan (inner product + argmax patterns from their docs)

## File Structure

```
server/
├── main.py                          # Existing — adds FHE router import + startup hook
├── fhe_routes.py                    # New FHE API router
├── fhe/
│   ├── __init__.py
│   ├── ops.py                       # @compile decorated FHE functions
│   ├── keys.py                      # Key generation, persistence, loading
│   ├── db.py                        # FHE identities DB operations
│   └── keys/                        # Serialized keys (gitignored)
└── requirements.txt                 # Add heir_py
```
