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
