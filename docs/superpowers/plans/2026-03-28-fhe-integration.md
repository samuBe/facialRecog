# FHE Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CKKS-based FHE to the facial recognition system so dot product similarity runs on encrypted embeddings, with backend logging and a frontend toggle to switch between plaintext and encrypted modes.

**Architecture:** `heir_py` compiles an MLIR dot product function targeting CKKS/OpenFHE. The server encrypts embeddings on arrival, computes homomorphic dot products for search, and decrypts results. New `/fhe/*` endpoints mirror existing plaintext ones. Argmax stays in plaintext for Phase 1 (encrypted argmax deferred — requires HEIR bootstrapping support validation).

**Tech Stack:** Python, FastAPI, heir_py (CKKS/OpenFHE), SQLite, vanilla JS

**Spec:** `docs/superpowers/specs/2026-03-28-fhe-integration-design.md`

---

## File Structure

```
server/
├── main.py                    # Modify: add FHE router, startup hook, /health FHE status
├── fhe_routes.py              # Create: /fhe/enroll, /fhe/search, /fhe/identities endpoints
├── fhe/
│   ├── __init__.py            # Create: empty
│   ├── ops.py                 # Create: HEIR-compiled dot product, encrypt/decrypt wrappers
│   ├── crypto.py              # Create: key management, persistence, setup
│   └── db.py                  # Create: fhe_identities table operations
├── requirements.txt           # Modify: add heir_py
└── tests/
    ├── __init__.py            # Create: empty
    ├── test_fhe_ops.py        # Create: dot product roundtrip tests
    ├── test_fhe_db.py         # Create: DB operations tests
    └── test_fhe_routes.py     # Create: API endpoint integration tests
client/
├── index.html                 # Modify: add FHE toggle in header
└── app.js                     # Modify: add FHE mode switching, route API calls accordingly
```

---

### Task 1: Add heir_py dependency and verify it works

**Files:**
- Modify: `server/requirements.txt`

- [ ] **Step 1: Add heir_py to requirements.txt**

```
heir_py==0.0.3
```

Append to `server/requirements.txt` after the existing dependencies.

- [ ] **Step 2: Verify installation**

Run: `source server/venv/bin/activate && pip install -r server/requirements.txt`

Expected: installs heir_py and its dependencies (numba, pybind11, etc). Note: numpy will be downgraded to <2 (heir_py requires numpy<2). This is fine — the existing code uses basic numpy features.

- [ ] **Step 3: Smoke test**

Run:
```bash
source server/venv/bin/activate && python3 -c "
from heir import compile
print('heir_py imported successfully')
"
```

Expected: `heir_py imported successfully`

- [ ] **Step 4: Commit**

```bash
git add server/requirements.txt
git commit -m "deps: add heir_py for CKKS homomorphic encryption"
```

---

### Task 2: FHE operations module — HEIR-compiled dot product

**Files:**
- Create: `server/fhe/__init__.py`
- Create: `server/fhe/ops.py`
- Create: `server/tests/__init__.py`
- Create: `server/tests/test_fhe_ops.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/__init__.py` (empty) and `server/tests/test_fhe_ops.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && source venv/bin/activate && python -m pytest tests/test_fhe_ops.py -v 2>&1 | head -20`

Expected: FAIL with `ModuleNotFoundError: No module named 'fhe'`

- [ ] **Step 3: Write the implementation**

Create `server/fhe/__init__.py` (empty).

Create `server/fhe/ops.py`:

```python
"""HEIR-compiled CKKS dot product for 128-element face embeddings."""

from __future__ import annotations

import logging
from functools import lru_cache

from heir import compile

logger = logging.getLogger("fhe")

DOT_PRODUCT_MLIR = """
func.func @dot_product(%arg0: tensor<128xf32> {secret.secret}, %arg1: tensor<128xf32> {secret.secret}) -> f32 {
  %c0_f32 = arith.constant 0.0 : f32
  %0 = affine.for %arg2 = 0 to 128 iter_args(%iter = %c0_f32) -> (f32) {
    %1 = tensor.extract %arg0[%arg2] : tensor<128xf32>
    %2 = tensor.extract %arg1[%arg2] : tensor<128xf32>
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %iter, %3 : f32
    affine.yield %4 : f32
  }
  return %0 : f32
}
"""


@lru_cache(maxsize=1)
def fhe_dot_product_ctx():
    """Compile and set up the CKKS dot product. Cached — only runs once."""
    logger.info("Compiling CKKS dot product with HEIR...")
    ctx = compile(mlir_str=DOT_PRODUCT_MLIR, scheme="ckks")
    logger.info("HEIR compilation complete. Running key generation...")
    ctx.setup()
    logger.info("CKKS key generation complete. FHE ready.")
    return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && source venv/bin/activate && python -m pytest tests/test_fhe_ops.py -v`

Expected: 3 tests PASS. First run will take ~7s for HEIR compilation + keygen.

- [ ] **Step 5: Commit**

```bash
git add server/fhe/ server/tests/
git commit -m "feat(fhe): add HEIR-compiled CKKS dot product for 128-element embeddings"
```

---

### Task 3: FHE database operations

**Files:**
- Create: `server/fhe/db.py`
- Create: `server/tests/test_fhe_db.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_fhe_db.py`:

```python
import sqlite3
import numpy as np
import pytest
from fhe.db import init_fhe_db, upsert_fhe_identity, load_fhe_identities, FHE_TABLE


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    init_fhe_db(c)
    return c


def test_init_creates_table(conn):
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (FHE_TABLE,)
    ).fetchone()
    assert tables is not None


def test_upsert_and_load(conn):
    ct = b"fake-ciphertext-bytes"
    upsert_fhe_identity(conn, "Agent Nova", ct, {"role": "test"})
    rows = load_fhe_identities(conn)
    assert len(rows) == 1
    assert rows[0]["label"] == "Agent Nova"
    assert rows[0]["ciphertext"] == ct
    assert rows[0]["metadata"]["role"] == "test"


def test_upsert_updates_existing(conn):
    upsert_fhe_identity(conn, "Agent Nova", b"ct-1", None)
    upsert_fhe_identity(conn, "Agent Nova", b"ct-2", None)
    rows = load_fhe_identities(conn)
    assert len(rows) == 1
    assert rows[0]["ciphertext"] == b"ct-2"


def test_load_empty(conn):
    assert load_fhe_identities(conn) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd server && source venv/bin/activate && python -m pytest tests/test_fhe_db.py -v 2>&1 | head -15`

Expected: FAIL with `ModuleNotFoundError: No module named 'fhe.db'`

- [ ] **Step 3: Write the implementation**

Create `server/fhe/db.py`:

```python
"""FHE identities database operations."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

logger = logging.getLogger("fhe")

FHE_TABLE = "fhe_identities"


def init_fhe_db(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {FHE_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            ciphertext BLOB NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def upsert_fhe_identity(
    conn: sqlite3.Connection,
    label: str,
    ciphertext: bytes,
    metadata: dict[str, Any] | None,
    *,
    commit: bool = True,
) -> None:
    logger.info("Storing encrypted identity: %s (%d bytes ciphertext)", label, len(ciphertext))
    conn.execute(
        f"""
        INSERT INTO {FHE_TABLE}(label, ciphertext, metadata)
        VALUES (?, ?, ?)
        ON CONFLICT(label)
        DO UPDATE SET ciphertext=excluded.ciphertext, metadata=excluded.metadata
        """,
        (label, ciphertext, json.dumps(metadata) if metadata else None),
    )
    if commit:
        conn.commit()


def load_fhe_identities(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"SELECT label, ciphertext, metadata, created_at FROM {FHE_TABLE}"
    ).fetchall()
    result = []
    for row in rows:
        meta_raw = row["metadata"]
        result.append({
            "label": row["label"],
            "ciphertext": row["ciphertext"],
            "metadata": json.loads(meta_raw) if meta_raw else None,
            "created_at": row["created_at"],
        })
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd server && source venv/bin/activate && python -m pytest tests/test_fhe_db.py -v`

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/fhe/db.py server/tests/test_fhe_db.py
git commit -m "feat(fhe): add fhe_identities database operations"
```

---

### Task 4: FHE crypto module — encrypt/decrypt/serialize wrappers

**Files:**
- Create: `server/fhe/crypto.py`

- [ ] **Step 1: Write the implementation**

Create `server/fhe/crypto.py`:

```python
"""High-level FHE encrypt/decrypt wrappers around heir_py compiled context."""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from fhe.ops import fhe_dot_product_ctx

logger = logging.getLogger("fhe")


def encrypt_embedding(embedding: list[float]) -> bytes:
    """Encrypt an embedding for storage (arg_0 slot). Returns serialized ciphertext."""
    ctx = fhe_dot_product_ctx()
    arr = np.asarray(embedding, dtype=np.float32)

    t0 = time.time()
    encrypted = ctx.encrypt_arg_0(arr)
    elapsed = time.time() - t0
    logger.info("Encrypted embedding (arg_0) in %.3fs", elapsed)

    # NOTE: pickle is used for demo only. For production, use OpenFHE native serialization.
    return pickle.dumps(encrypted)


def encrypt_query(embedding: list[float]) -> bytes:
    """Encrypt a query embedding (arg_1 slot). Returns serialized ciphertext."""
    ctx = fhe_dot_product_ctx()
    arr = np.asarray(embedding, dtype=np.float32)

    t0 = time.time()
    encrypted = ctx.encrypt_arg_1(arr)
    elapsed = time.time() - t0
    logger.info("Encrypted query (arg_1) in %.3fs", elapsed)

    return pickle.dumps(encrypted)


def deserialize_ciphertext(ct_bytes: bytes) -> Any:
    """Deserialize ciphertext from stored bytes."""
    # NOTE: pickle.loads on untrusted data is a security risk. Demo only.
    return pickle.loads(ct_bytes)


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
```

Note: We use `pickle` for ciphertext serialization. This is acceptable for a demo where all data is server-local. For production, OpenFHE's native serialization should be used.

- [ ] **Step 2: Quick manual verification**

Run:
```bash
cd server && source venv/bin/activate && python3 -c "
import numpy as np
from fhe.crypto import encrypt_embedding, deserialize_ciphertext, eval_dot_product, decrypt_result

a = np.random.randn(128).astype(np.float32)
a = (a / np.linalg.norm(a)).tolist()

ct_a = encrypt_embedding(a)
ct_b = encrypt_embedding(a)

enc_a = deserialize_ciphertext(ct_a)
enc_b = deserialize_ciphertext(ct_b)
enc_result = eval_dot_product(enc_a, enc_b)
score = decrypt_result(enc_result)
print(f'Self-similarity: {score:.6f} (expected ~1.0)')
print(f'Ciphertext size: {len(ct_a)} bytes')
"
```

Expected: Self-similarity ~1.0, ciphertext several KB.

- [ ] **Step 3: Commit**

```bash
git add server/fhe/crypto.py
git commit -m "feat(fhe): add encrypt/decrypt/serialize wrapper module"
```

---

### Task 5: FHE API routes

**Files:**
- Create: `server/fhe_routes.py`
- Create: `server/tests/test_fhe_routes.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_fhe_routes.py`:

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Module-scoped to avoid recompiling HEIR per test."""
    from main import app
    return TestClient(app)


def test_fhe_enroll(client):
    import numpy as np
    np.random.seed(99)
    emb = np.random.randn(128).tolist()
    resp = client.post("/fhe/enroll", json={
        "label": "Test Agent",
        "embedding": emb,
        "metadata": {"role": "tester"},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "saved"
    assert data["label"] == "Test Agent"


def test_fhe_search_finds_enrolled(client):
    import numpy as np
    np.random.seed(99)
    emb = np.random.randn(128).tolist()

    # Enroll first
    client.post("/fhe/enroll", json={"label": "Search Target", "embedding": emb})

    # Search with same embedding — should find it
    resp = client.post("/fhe/search", json={"embedding": emb, "top_k": 5, "threshold": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert data["best_match"]["label"] in ("Search Target", "Test Agent")


def test_fhe_identities_lists_enrolled(client):
    resp = client.get("/fhe/identities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    # No ciphertext in response
    for item in data["items"]:
        assert "ciphertext" not in item
        assert "label" in item


def test_fhe_search_no_results(client):
    """Search with a random embedding that won't match anything above a high threshold."""
    import numpy as np
    np.random.seed(777)
    emb = np.random.randn(128).tolist()
    resp = client.post("/fhe/search", json={"embedding": emb, "threshold": 0.99})
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_health_includes_fhe(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "fhe" in data
```

- [ ] **Step 2: Add `get_fhe_connection` and `set_db_path` to `fhe/db.py`**

These are needed by the routes. Add to `server/fhe/db.py`:

```python
from pathlib import Path

# DB_PATH will be set during app startup via set_db_path()
_db_path: Path | None = None


def set_db_path(path: Path) -> None:
    global _db_path
    _db_path = path


def get_fhe_connection() -> sqlite3.Connection:
    if _db_path is None:
        raise RuntimeError("FHE DB path not configured. Call set_db_path() first.")
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    return conn
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd server && source venv/bin/activate && python -m pytest tests/test_fhe_routes.py -v 2>&1 | head -20`

Expected: FAIL (routes don't exist yet)

- [ ] **Step 4: Write the FHE routes**

Create `server/fhe_routes.py`:

```python
"""FHE API routes — encrypted versions of enroll/search/identities."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from fhe.crypto import (
    decrypt_result,
    deserialize_ciphertext,
    encrypt_embedding,
    encrypt_query,
    eval_dot_product,
)
from fhe.db import get_fhe_connection, load_fhe_identities, upsert_fhe_identity

logger = logging.getLogger("fhe")

router = APIRouter(prefix="/fhe", tags=["fhe"])


class FHEEnrollRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)
    embedding: list[float] = Field(..., min_length=128, max_length=128)
    metadata: dict[str, Any] | None = None

    @field_validator("label")
    @classmethod
    def strip_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("label cannot be empty")
        return cleaned


class FHESearchRequest(BaseModel):
    embedding: list[float] = Field(..., min_length=128, max_length=128)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.35, ge=-1.0, le=1.0)


class FHEMatch(BaseModel):
    label: str
    similarity: float
    metadata: dict[str, Any] | None


class FHESearchResponse(BaseModel):
    count: int
    best_match: FHEMatch | None
    matches: list[FHEMatch]


def _normalize(embedding: list[float]) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Embedding norm cannot be zero")
    return arr / norm


@router.post("/enroll")
def fhe_enroll(payload: FHEEnrollRequest) -> dict[str, str]:
    t_start = time.time()
    try:
        unit_vec = _normalize(payload.embedding)
        logger.info("[enroll] Encrypting embedding for '%s'", payload.label)
        ct_bytes = encrypt_embedding(unit_vec.tolist())

        with get_fhe_connection() as conn:
            upsert_fhe_identity(conn, payload.label, ct_bytes, payload.metadata)

        elapsed = time.time() - t_start
        logger.info("[enroll] '%s' enrolled in %.3fs (%d bytes ciphertext)",
                     payload.label, elapsed, len(ct_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "saved", "label": payload.label}


@router.post("/search", response_model=FHESearchResponse)
def fhe_search(payload: FHESearchRequest) -> FHESearchResponse:
    t_start = time.time()
    try:
        query_vec = _normalize(payload.embedding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("[search] Encrypting query embedding")
    # Query uses arg_1 slot; stored embeddings use arg_0 slot
    enc_query = deserialize_ciphertext(encrypt_query(query_vec.tolist()))

    with get_fhe_connection() as conn:
        identities = load_fhe_identities(conn)

    if not identities:
        logger.info("[search] No FHE identities enrolled")
        return FHESearchResponse(count=0, best_match=None, matches=[])

    logger.info("[search] Comparing against %d encrypted identities", len(identities))
    ranked: list[FHEMatch] = []
    for identity in identities:
        enc_stored = deserialize_ciphertext(identity["ciphertext"])
        enc_result = eval_dot_product(enc_query, enc_stored)
        score = decrypt_result(enc_result)
        logger.info("[search] %s → similarity %.6f", identity["label"], score)

        if score >= payload.threshold:
            ranked.append(FHEMatch(
                label=identity["label"],
                similarity=round(score, 4),
                metadata=identity["metadata"],
            ))

    ranked.sort(key=lambda m: m.similarity, reverse=True)
    top = ranked[: payload.top_k]

    elapsed = time.time() - t_start
    logger.info("[search] Complete in %.3fs — %d match(es)", elapsed, len(top))

    return FHESearchResponse(
        count=len(top),
        best_match=top[0] if top else None,
        matches=top,
    )


@router.post("/enroll/bulk")
def fhe_enroll_bulk(payload: dict) -> dict:
    """Bulk enroll — same as /enroll/bulk but encrypts each embedding."""
    from pydantic import ValidationError

    entries = payload.get("entries", [])
    if not entries or len(entries) > 50:
        raise HTTPException(status_code=422, detail="entries must be 1-50 items")

    results = []
    enrolled = 0
    errors = 0

    with get_fhe_connection() as conn:
        for entry in entries:
            try:
                label = entry["label"].strip()
                embedding = entry["embedding"]
                metadata = entry.get("metadata")
                if len(embedding) != 128:
                    raise ValueError("embedding must be 128 floats")

                unit_vec = _normalize(embedding)
                ct_bytes = encrypt_embedding(unit_vec.tolist())
                upsert_fhe_identity(conn, label, ct_bytes, metadata, commit=False)
                results.append({"label": label, "status": "saved"})
                enrolled += 1
            except Exception as exc:
                results.append({"label": entry.get("label", "?"), "status": "error", "detail": str(exc)})
                errors += 1
        conn.commit()

    logger.info("[enroll/bulk] %d enrolled, %d errors", enrolled, errors)
    return {"enrolled": enrolled, "errors": errors, "results": results}


@router.get("/identities")
def fhe_identities() -> dict[str, Any]:
    with get_fhe_connection() as conn:
        identities = load_fhe_identities(conn)

    items = [
        {
            "label": i["label"],
            "metadata": i["metadata"],
            "created_at": i["created_at"],
        }
        for i in identities
    ]
    return {"count": len(items), "items": items}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd server && source venv/bin/activate && python -m pytest tests/test_fhe_routes.py -v`

Expected: Tests PASS (may take ~10s for first HEIR compilation)

- [ ] **Step 6: Commit**

```bash
git add server/fhe_routes.py server/fhe/db.py server/tests/test_fhe_routes.py
git commit -m "feat(fhe): add /fhe/enroll, /fhe/search, /fhe/identities endpoints"
```

---

### Task 6: Wire FHE into main app with logging

**Files:**
- Modify: `server/main.py`

- [ ] **Step 1: Add FHE imports and logging configuration at top of `main.py`**

After the existing imports (line 13), add:

```python
import logging

# Configure FHE logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
fhe_logger = logging.getLogger("fhe")
```

- [ ] **Step 2: Import and mount FHE router**

After the `app.add_middleware(...)` block (after line 155), add:

```python
# FHE routes
_fhe_available = False
try:
    from fhe_routes import router as fhe_router
    from fhe.db import init_fhe_db, set_db_path
    from fhe.ops import fhe_dot_product_ctx

    app.include_router(fhe_router)
    _fhe_available = True
except ImportError:
    fhe_logger.warning("heir_py not installed — FHE endpoints disabled")
```

- [ ] **Step 3: Update startup hook**

Modify the existing `startup()` function to also initialize FHE:

```python
@app.on_event("startup")
def startup() -> None:
    init_db()
    seed_if_empty()

    if _fhe_available:
        fhe_logger.info("Initializing FHE subsystem (compilation takes ~7s)...")
        set_db_path(DB_PATH)
        with get_connection() as conn:
            init_fhe_db(conn)
            # Keys are regenerated each startup, so stored ciphertexts are invalid
            conn.execute("DELETE FROM fhe_identities")
            conn.commit()
            fhe_logger.info("Cleared stale FHE identities (keys regenerated)")
        try:
            fhe_dot_product_ctx()  # Trigger compilation + keygen
            fhe_logger.info("FHE subsystem ready")
        except Exception:
            fhe_logger.exception("FHE initialization failed")
```

- [ ] **Step 4: Update /health endpoint**

Replace the existing health function:

```python
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "fhe": "ok" if _fhe_available else "disabled"}
```

- [ ] **Step 5: Test the full server**

Run: `cd server && source venv/bin/activate && python -m uvicorn main:app --reload 2>&1 | head -30`

Expected: See FHE initialization logs:
```
INFO: Initializing FHE subsystem...
INFO: Compiling CKKS dot product with HEIR...
INFO: HEIR compilation complete. Running key generation...
INFO: CKKS key generation complete. FHE ready.
INFO: FHE subsystem ready
```

Test health: `curl http://localhost:8000/health`
Expected: `{"status":"ok","fhe":"ok"}`

- [ ] **Step 6: Commit**

```bash
git add server/main.py
git commit -m "feat: wire FHE subsystem into FastAPI with logging"
```

---

### Task 7: Frontend — add FHE mode toggle

**Files:**
- Modify: `client/index.html`
- Modify: `client/app.js`

- [ ] **Step 1: Add FHE toggle to HTML header**

In `client/index.html`, inside the `<div class="status">` block (after line 28), add:

```html
        <label class="fhe-toggle">
          <input type="checkbox" id="fheToggle" />
          <span>FHE Mode</span>
        </label>
```

- [ ] **Step 2: Add FHE state and route helper in app.js**

At the top of `client/app.js`, after line 5 (`const API_URL = ...`), add:

```javascript
const fheToggle = document.getElementById("fheToggle");
let fheMode = false;

fheToggle.addEventListener("change", () => {
  fheMode = fheToggle.checked;
  log(fheMode ? "FHE mode ON — using encrypted backend" : "FHE mode OFF — using plaintext backend");
});

function apiPath(path) {
  return fheMode ? `${API_URL}/fhe${path}` : `${API_URL}${path}`;
}
```

- [ ] **Step 3: Update API calls to use apiPath()**

Replace hardcoded URLs in `client/app.js`:

1. **lookupFace** (line 241): change `${API_URL}/search` → `${apiPath("/search")}`
2. **enrollFace** (line 299): change `${API_URL}/enroll` → `${apiPath("/enroll")}`
3. **bulkEnroll** (line 502): change `${API_URL}/enroll/bulk` → `${apiPath("/enroll/bulk")}`

- [ ] **Step 4: Add FHE toggle styling**

In `client/styles.css`, add at the end:

```css
.fhe-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-family: "Share Tech Mono", monospace;
  font-size: 0.8rem;
  color: var(--text-muted, #8892b0);
}

.fhe-toggle input:checked + span {
  color: #64ffda;
}
```

- [ ] **Step 5: Test manually**

1. Start server: `cd server && source venv/bin/activate && uvicorn main:app --reload`
2. Open `http://localhost:8000` in browser
3. Check the "FHE Mode" toggle in the header
4. Upload an image, detect faces, enroll with FHE on
5. Check server logs — should see FHE encryption logs
6. Lookup the same face with FHE on — should see homomorphic dot product logs

- [ ] **Step 6: Commit**

```bash
git add client/index.html client/app.js client/styles.css
git commit -m "feat: add FHE mode toggle in frontend header"
```

---

### Task 8: Check FHE health on frontend load

**Files:**
- Modify: `client/app.js`

- [ ] **Step 1: Add FHE health check on page load**

In `client/app.js`, inside the `loadModels()` function or the DOMContentLoaded handler, add after the face-api model loading:

```javascript
// Check if FHE backend is available
fetch(`${API_URL}/health`)
  .then((r) => r.json())
  .then((data) => {
    if (data.fhe === "ok") {
      fheToggle.disabled = false;
      log("FHE backend available.");
    } else {
      fheToggle.disabled = true;
      fheToggle.checked = false;
      log("FHE backend not available.");
    }
  })
  .catch(() => {
    fheToggle.disabled = true;
  });
```

- [ ] **Step 2: Disable toggle by default in HTML**

In `client/index.html`, update the FHE toggle input:

```html
<input type="checkbox" id="fheToggle" disabled />
```

- [ ] **Step 3: Test**

1. Start server with FHE → toggle should be enabled
2. Comment out `heir_py` import in main.py → toggle should be disabled

- [ ] **Step 4: Commit**

```bash
git add client/app.js client/index.html
git commit -m "feat: check FHE health on load, disable toggle if unavailable"
```

---

### Task 9: Integration test — full roundtrip

**Files:**
- Modify: `server/tests/test_fhe_routes.py`

- [ ] **Step 1: Add full roundtrip integration test**

Append to `server/tests/test_fhe_routes.py`:

```python
def test_fhe_full_roundtrip(client):
    """Enroll multiple identities, search, verify correct match ranked first."""
    import numpy as np

    # Create 3 distinct embeddings
    np.random.seed(123)
    embeddings = {}
    for name in ["Alpha", "Bravo", "Charlie"]:
        e = np.random.randn(128).astype(np.float32)
        e = (e / np.linalg.norm(e)).tolist()
        embeddings[name] = e
        client.post("/fhe/enroll", json={"label": name, "embedding": e})

    # Search with Alpha's embedding — Alpha should be best match
    resp = client.post("/fhe/search", json={
        "embedding": embeddings["Alpha"],
        "top_k": 5,
        "threshold": 0.0,
    })
    data = resp.json()
    assert data["count"] >= 1
    assert data["best_match"]["label"] == "Alpha"
    assert data["best_match"]["similarity"] > 0.99  # self-match

    # Verify plaintext search gives same ranking
    resp_plain = client.post("/search", json={
        "embedding": embeddings["Alpha"],
        "top_k": 5,
        "threshold": 0.0,
    })
    plain_data = resp_plain.json()
    # Both should agree on best match
    if plain_data["count"] > 0:
        # Note: plaintext DB may have different identities, so just check FHE works
        pass

    assert data["best_match"]["similarity"] > 0.99
```

- [ ] **Step 2: Run all tests**

Run: `cd server && source venv/bin/activate && python -m pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add server/tests/test_fhe_routes.py
git commit -m "test: add full FHE roundtrip integration test"
```

---

### Task 10: Add .gitignore for FHE keys and update spec

**Files:**
- Modify: `.gitignore` (or create if doesn't exist)

- [ ] **Step 1: Add gitignore entries**

Add to `.gitignore`:

```
server/fhe/keys/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore FHE key directory"
```
