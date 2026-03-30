from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from main import app


def test_openfhe_health_route_reports_disabled_or_ok() -> None:
    with TestClient(app) as client:
        resp = client.get("/fhe-openfhe/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in {"ok", "disabled"}
    assert "reason" in data


def test_openfhe_session_routes_return_503_when_backend_missing(monkeypatch) -> None:
    import fhe_openfhe_routes

    monkeypatch.setattr(
        fhe_openfhe_routes,
        "openfhe_available_reason",
        lambda: "openfhe not installed",
    )

    with TestClient(app) as client:
        resp = client.post("/fhe-openfhe/session")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "openfhe not installed"


def test_openfhe_session_creation_and_key_upload(monkeypatch) -> None:
    import fhe_openfhe_routes

    class FakeSession:
        session_id = "session-123"

        def load_client_keys(self, **kwargs) -> None:
            self.loaded = kwargs

    fake_session = FakeSession()

    monkeypatch.setattr(fhe_openfhe_routes, "openfhe_available_reason", lambda: None)
    monkeypatch.setattr(fhe_openfhe_routes, "create_client_session", lambda: fake_session)
    monkeypatch.setattr(fhe_openfhe_routes, "get_client_session", lambda session_id: fake_session if session_id == "session-123" else None)

    with TestClient(app) as client:
        create_resp = client.post("/fhe-openfhe/session")
        assert create_resp.status_code == 200
        assert create_resp.json()["session_id"] == "session-123"

        upload_resp = client.post(
            "/fhe-openfhe/session/session-123/keys",
            json={
                "crypto_context": base64.b64encode(b"cc").decode(),
                "public_key": base64.b64encode(b"pk").decode(),
                "eval_automorphism_key": base64.b64encode(b"auto").decode(),
            },
        )

    assert upload_resp.status_code == 200
    assert fake_session.loaded["crypto_context_bytes"] == b"cc"
    assert fake_session.loaded["public_key_bytes"] == b"pk"
    assert fake_session.loaded["eval_automorphism_key_bytes"] == b"auto"


def test_openfhe_search_returns_encrypted_candidates(monkeypatch) -> None:
    import fhe_openfhe_routes

    class FakeSession:
        def deserialize_ciphertext(self, ciphertext_bytes: bytes) -> bytes:
            return ciphertext_bytes

        def eval_dot_product_with_plaintext(self, encrypted_query: bytes, embedding: list[float]) -> bytes:
            return encrypted_query + b":" + str(embedding[0]).encode()

        def serialize_ciphertext(self, ciphertext: bytes) -> bytes:
            return ciphertext

    monkeypatch.setattr(fhe_openfhe_routes, "openfhe_available_reason", lambda: None)
    monkeypatch.setattr(fhe_openfhe_routes, "get_client_session", lambda session_id: FakeSession() if session_id == "session-123" else None)
    monkeypatch.setattr(
        fhe_openfhe_routes,
        "_load_plaintext_identities",
        lambda: [
            {"label": "Alpha", "embedding": [1.0] * 128, "metadata": {"role": "captain"}},
            {"label": "Bravo", "embedding": [0.5] * 128, "metadata": {"role": "wing"}},
        ],
    )

    with TestClient(app) as client:
        resp = client.post(
            "/fhe-openfhe/search",
            json={
                "session_id": "session-123",
                "encrypted_query": base64.b64encode(b"query").decode(),
                "top_k": 5,
                "threshold": 0.4,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["backend"] == "openfhe-client"
    assert data["candidate_count"] == 2
    assert data["count"] == 2
    assert data["candidates"][0]["label"] == "Alpha"
    assert base64.b64decode(data["candidates"][0]["encrypted_score"]) == b"query:1.0"
    assert base64.b64decode(data["candidates"][1]["encrypted_score"]) == b"query:0.5"
