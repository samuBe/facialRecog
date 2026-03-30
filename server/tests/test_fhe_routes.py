import pytest
import pickle
from fastapi.testclient import TestClient

from fhe.runtime import native_fhe_enabled, native_fhe_unavailable_reason


pytestmark = pytest.mark.skipif(
    not native_fhe_enabled(),
    reason=native_fhe_unavailable_reason() or "native FHE unavailable",
)


@pytest.fixture(scope="module")
def client():
    """Module-scoped to avoid recompiling HEIR per test."""
    from main import app
    with TestClient(app) as c:
        yield c


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

    client.post("/fhe/enroll", json={"label": "Search Target", "embedding": emb})

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
    for item in data["items"]:
        assert "ciphertext" not in item
        assert "label" in item


def test_fhe_search_no_results(client):
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


def test_fhe_full_roundtrip(client):
    """Enroll multiple identities, search, verify correct match ranked first."""
    import numpy as np

    np.random.seed(123)
    embeddings = {}
    for name in ["Alpha", "Bravo", "Charlie"]:
        e = np.random.randn(128).astype(np.float32)
        e = (e / np.linalg.norm(e)).tolist()
        embeddings[name] = e
        client.post("/fhe/enroll", json={"label": name, "embedding": e})

    resp = client.post("/fhe/search", json={
        "embedding": embeddings["Alpha"],
        "top_k": 5,
        "threshold": 0.0,
    })
    data = resp.json()
    assert data["count"] >= 1
    assert data["best_match"]["label"] == "Alpha"
    assert data["best_match"]["similarity"] > 0.99


def test_fhe_search_stream_prunes_stale_rows_without_breaking_stream(client):
    from fhe.db import get_fhe_connection, load_fhe_identities, upsert_fhe_identity

    stale_token = pickle.dumps(b"missing-token")
    with get_fhe_connection() as conn:
        upsert_fhe_identity(conn, "Stale Agent", stale_token, {"role": "stale"})

    import numpy as np
    np.random.seed(11)
    emb = np.random.randn(128).tolist()

    with client.stream("POST", "/fhe/search/stream", json={
        "embedding": emb,
        "top_k": 5,
        "threshold": 0.35,
    }) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert '"type": "result"' in body
    assert "Error in input stream" not in body

    with get_fhe_connection() as conn:
        labels = [row["label"] for row in load_fhe_identities(conn)]
    assert "Stale Agent" not in labels
