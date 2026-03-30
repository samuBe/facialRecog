"""FHE identities database operations."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("fhe")

FHE_TABLE = "fhe_identities"

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


def delete_fhe_identities(conn: sqlite3.Connection, labels: list[str], *, commit: bool = True) -> int:
    if not labels:
        return 0

    placeholders = ",".join("?" for _ in labels)
    cursor = conn.execute(
        f"DELETE FROM {FHE_TABLE} WHERE label IN ({placeholders})",
        labels,
    )
    if commit:
        conn.commit()
    return cursor.rowcount
