"""
SQLite-backed ligand descriptor lookup (one SMILES per query, low RAM).

Used on Streamlit Cloud instead of joblib.load() on the full ~346 MB dict.
"""
from __future__ import annotations

import json
import os
import sqlite3
import zlib
from pathlib import Path
from typing import Dict, Optional

_TABLE = "ligand_lookup"
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    smiles TEXT PRIMARY KEY NOT NULL,
    payload BLOB NOT NULL
);
"""


def sqlite_lookup_path(manuscript_root: Path) -> Path:
    return manuscript_root / "ligand_feature_lookup.sqlite"


def sqlite_lookup_available(manuscript_root: Path) -> bool:
    p = sqlite_lookup_path(manuscript_root)
    return p.is_file() and p.stat().st_size > 100_000


def should_use_sqlite_lookup(manuscript_root: Path) -> bool:
    if not sqlite_lookup_available(manuscript_root):
        return False
    env = os.environ.get("GPCR_LIGAND_LOOKUP_SQLITE", "").strip().lower()
    if env in ("0", "false", "no"):
        return False
    if env in ("1", "true", "yes"):
        return True
    if os.environ.get("GPCR_FORCE_CLOUD_MODE", "").strip().lower() in ("1", "true", "yes"):
        return True
    if Path("/mount/src").is_dir():
        return True
    if str(os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "")).strip().lower() == "cloud":
        return True
    return True


def encode_feature_dict(features: Dict[str, float]) -> bytes:
    sparse = {k: float(v) for k, v in features.items() if v == v and abs(float(v)) > 1e-12}
    raw = json.dumps(sparse, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return zlib.compress(raw, level=6)


def decode_feature_blob(blob: bytes) -> Dict[str, float]:
    raw = zlib.decompress(blob)
    data = json.loads(raw.decode("utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def open_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def fetch_ligand_features(manuscript_root: Path, canonical_smiles: str) -> Dict[str, float]:
    db_path = sqlite_lookup_path(manuscript_root)
    if not db_path.is_file():
        return {}
    con = open_readonly(db_path)
    try:
        cur = con.execute(
            f"SELECT payload FROM {_TABLE} WHERE smiles = ?",
            (canonical_smiles,),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return {}
        return decode_feature_blob(row[0])
    finally:
        con.close()


def count_entries(manuscript_root: Path) -> Optional[int]:
    db_path = sqlite_lookup_path(manuscript_root)
    if not db_path.is_file():
        return None
    con = open_readonly(db_path)
    try:
        cur = con.execute(f"SELECT COUNT(*) FROM {_TABLE}")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def create_writable(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.executescript(_SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def bulk_insert(
    con: sqlite3.Connection,
    rows: list[tuple[str, bytes]],
    *,
    replace: bool = True,
) -> None:
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    con.executemany(
        f"{verb} INTO {_TABLE} (smiles, payload) VALUES (?, ?)",
        rows,
    )
