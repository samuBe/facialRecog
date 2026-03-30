# Tactical Face Scanner

A facial-recognition demo with optional **Fully Homomorphic Encryption (FHE)** — the server can match faces on encrypted embeddings without ever seeing the raw biometric data.

- **Client** (`client/`): browser app for image upload/camera capture, local face detection + descriptor extraction (`face-api.js`), interactive face selection, and lookup/enrollment.
- **Server** (`server/`): FastAPI service with both plaintext and FHE-encrypted cosine-similarity search against a SQLite embedding store.

## Quick Start

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000.

> FHE compilation takes ~7 seconds on first startup. Watch the logs for "FHE subsystem ready".
> On Python 3.13+, the native `heir_py`/OpenFHE path is currently disabled by default in this app because local setup is segfaulting. Plaintext routes still work; use Python 3.12 for native `/fhe/*` routes, or override with `FACIALRECOG_ENABLE_UNSAFE_FHE=1` if you explicitly want to test the crashing path.

## Usage

1. Upload an image or start camera and capture a frame.
2. Click **Extract Faces** to detect faces locally in the browser.
3. Select a detected face from the thumbnail strip.
4. Set a label and click **Enroll Selected Face** to store the embedding.
5. Click **Lookup Selected Face** to search against enrolled identities.

### FHE Mode

Click the **FHE Mode** button in the header to switch to encrypted search:

- Embeddings are encrypted with CKKS homomorphic encryption before storage
- Search computes dot-product similarity entirely on ciphertext
- A real-time progress bar shows each encrypted comparison via SSE
- The server never sees raw embedding values in FHE mode

## API Endpoints

### Plaintext

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (includes FHE status) |
| `GET` | `/identities` | List enrolled identities |
| `POST` | `/enroll` | Enroll a face embedding |
| `POST` | `/enroll/bulk` | Bulk enroll (up to 50) |
| `POST` | `/search` | Cosine similarity search |

### FHE (encrypted)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/fhe/enroll` | Encrypt and enroll embedding |
| `POST` | `/fhe/enroll/bulk` | Bulk encrypted enroll |
| `POST` | `/fhe/search` | Homomorphic similarity search |
| `POST` | `/fhe/search/stream` | SSE streaming search with progress |
| `GET` | `/fhe/identities` | List FHE-enrolled identities |

## Architecture

```
Browser                              Server (FastAPI)
┌──────────────────────┐   JSON     ┌─────────────────────────────┐
│ face-api.js          │──────────→ │ Plaintext: numpy dot product│
│ → 128-float embedding│            │                             │
│                      │   JSON     │ FHE mode:                   │
│                      │──────────→ │ heir_py encrypts (CKKS)     │
│ FHE toggle           │            │ → homomorphic dot product   │
│ SSE progress bar     │ ←── SSE ──│ → decrypt result             │
└──────────────────────┘            └─────────────────────────────┘
```

### FHE Stack

- **[Google HEIR](https://github.com/google/heir)** (`heir_py`) — compiles MLIR dot-product to CKKS/OpenFHE
- **CKKS scheme** — approximate arithmetic on encrypted floating-point vectors
- **128-element encrypted dot product** — ~0.25s per comparison, ~1e-9 error vs plaintext

### Key Design Decisions

- **Server-side encryption (Phase 1):** The server encrypts on arrival as a demo. Phase 2 will move encryption to the browser via openfhe-wasm.
- **MLIR string mode:** `heir_py`'s `@compile` decorator doesn't support tensor indexing yet, so we pass MLIR directly.
- **In-memory ciphertext store:** heir_py's pybind11 ciphertext objects can't be serialized with pickle. UUID tokens in SQLite reference in-memory objects. Ciphertexts are purged on restart.
- **Plaintext argmax:** Encrypted argmax (via bootstrapping) is deferred to Phase 2.

## Project Structure

```
server/
├── main.py              # FastAPI app, routes, startup
├── fhe_routes.py        # /fhe/* encrypted endpoints + SSE
├── fhe/
│   ├── ops.py           # HEIR-compiled CKKS dot product
│   ├── crypto.py        # Encrypt/decrypt/serialize wrappers
│   └── db.py            # fhe_identities table operations
├── tests/               # pytest suite (13 tests)
└── requirements.txt     # FastAPI, numpy, heir_py

client/
├── index.html           # Single-page app
├── app.js               # Face detection, FHE toggle, SSE progress
└── styles.css           # Cyberpunk-themed UI
```

## Testing

```bash
cd server
source venv/bin/activate
python -m pytest tests/ -v
```

## Future Work

- **Phase 2:** Client-side encryption via openfhe-wasm (server never sees plaintext)
- **Encrypted argmax:** Find best match entirely on ciphertext using CKKS bootstrapping
- **Desilo pi-heaan:** Migration to production FHE library

## Notes

- Face detection runs locally in the browser with `face-api.js` — no images are sent to the server.
- First model load downloads weights from CDN (~5MB).
- The starter DB is seeded from `server/data/seed_embeddings.json` on first launch.
- FHE keys are regenerated on each server restart; encrypted identities are purged automatically.
- This is a research demo, not production-grade biometric security.
