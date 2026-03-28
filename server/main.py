from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
CLIENT_DIR = BASE_DIR.parent / "client"
DB_PATH = BASE_DIR / "faces.db"
SEED_PATH = BASE_DIR / "data" / "seed_embeddings.json"


class SearchRequest(BaseModel):
    embedding: list[float] = Field(..., min_length=128, max_length=128)
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.35, ge=-1.0, le=1.0)


class EnrollRequest(BaseModel):
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


class BulkEnrollRequest(BaseModel):
    entries: list[EnrollRequest] = Field(..., min_length=1, max_length=50)


class EntryResult(BaseModel):
    label: str
    status: str  # "saved" or "error"
    detail: str | None = None


class BulkEnrollResponse(BaseModel):
    enrolled: int
    errors: int
    results: list[EntryResult]


class Match(BaseModel):
    label: str
    similarity: float
    metadata: dict[str, Any] | None


class SearchResponse(BaseModel):
    count: int
    best_match: Match | None
    matches: list[Match]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL UNIQUE,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def embedding_to_unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Embedding norm cannot be zero")
    return arr / norm


def load_identities(conn: sqlite3.Connection) -> list[tuple[str, np.ndarray, dict[str, Any] | None]]:
    rows = conn.execute("SELECT label, embedding, metadata FROM identities").fetchall()
    parsed: list[tuple[str, np.ndarray, dict[str, Any] | None]] = []
    for row in rows:
        embedding = embedding_to_unit(json.loads(row["embedding"]))
        metadata_raw = row["metadata"]
        metadata = json.loads(metadata_raw) if metadata_raw else None
        parsed.append((row["label"], embedding, metadata))
    return parsed


def upsert_identity(conn: sqlite3.Connection, label: str, embedding: list[float], metadata: dict[str, Any] | None = None, *, commit: bool = True) -> None:
    unit_vec = embedding_to_unit(embedding)
    conn.execute(
        """
        INSERT INTO identities(label, embedding, metadata)
        VALUES (?, ?, ?)
        ON CONFLICT(label)
        DO UPDATE SET embedding=excluded.embedding, metadata=excluded.metadata
        """,
        (
            label,
            json.dumps(unit_vec.tolist()),
            json.dumps(metadata) if metadata else None,
        ),
    )
    if commit:
        conn.commit()


def seed_if_empty() -> None:
    if not SEED_PATH.exists():
        return

    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM identities").fetchone()["c"]
        if count > 0:
            return

        payload = json.loads(SEED_PATH.read_text())
        for entry in payload:
            upsert_identity(
                conn,
                label=entry["label"],
                embedding=entry["embedding"],
                metadata=entry.get("metadata"),
            )


app = FastAPI(title="Facial Embedding Lookup API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    seed_if_empty()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/identities")
def list_identities() -> dict[str, Any]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT label, metadata, created_at FROM identities ORDER BY created_at DESC"
        ).fetchall()

    identities = [
        {
            "label": row["label"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return {"count": len(identities), "items": identities}


@app.post("/enroll")
def enroll(payload: EnrollRequest) -> dict[str, str]:
    try:
        with get_connection() as conn:
            upsert_identity(conn, payload.label, payload.embedding, payload.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "saved", "label": payload.label}


@app.post("/enroll/bulk", response_model=BulkEnrollResponse)
def enroll_bulk(payload: BulkEnrollRequest) -> BulkEnrollResponse:
    results: list[EntryResult] = []
    enrolled = 0
    errors = 0

    with get_connection() as conn:
        for entry in payload.entries:
            try:
                upsert_identity(conn, entry.label, entry.embedding, entry.metadata, commit=False)
                results.append(EntryResult(label=entry.label, status="saved"))
                enrolled += 1
            except Exception as exc:
                results.append(EntryResult(label=entry.label, status="error", detail=str(exc)))
                errors += 1
        conn.commit()

    return BulkEnrollResponse(enrolled=enrolled, errors=errors, results=results)


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    try:
        query = embedding_to_unit(payload.embedding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with get_connection() as conn:
        identities = load_identities(conn)

    if not identities:
        return SearchResponse(count=0, best_match=None, matches=[])

    ranked: list[Match] = []
    for label, known_vec, metadata in identities:
        similarity = float(np.dot(query, known_vec))
        if similarity >= payload.threshold:
            ranked.append(Match(label=label, similarity=round(similarity, 4), metadata=metadata))

    ranked.sort(key=lambda match: match.similarity, reverse=True)
    top_matches = ranked[: payload.top_k]

    return SearchResponse(
        count=len(top_matches),
        best_match=top_matches[0] if top_matches else None,
        matches=top_matches,
    )


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(CLIENT_DIR / "index.html")


app.mount("/static", StaticFiles(directory=CLIENT_DIR), name="static")
