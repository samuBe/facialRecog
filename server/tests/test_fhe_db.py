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
