"""
Load ligand descriptor rows from *_NEW.xlsx only (manuscript training workbooks).

Uses per-SMILES workbook scans by default to avoid loading full receptor tables
into memory (important on Streamlit Cloud).
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .predict import _canonicalize_smiles

_EXCLUDE = {
    "Receptor", "Label", "ChEMBL_ID", "AID", "CID", "Activity", "SMILES",
    "Label_Type", "Label_Type_clean",
}

_WORKBOOK_SUFFIXES = (
    "agonist_enriched_NEW.xlsx",
    "antagonist_enriched_NEW.xlsx",
    "non_active_compounds_NEW.xlsx",
)

_RECEPTOR_TABLE_CACHE: "OrderedDict[str, Dict[str, Dict[str, float]]]" = OrderedDict()
_MAX_RECEPTOR_CACHE = int(os.environ.get("GPCR_RECEPTOR_CACHE_MAX", "2"))

_SMILES_ROW_CACHE: "OrderedDict[tuple[str, str], Dict[str, float]]" = OrderedDict()
_MAX_SMILES_ROW_CACHE = int(os.environ.get("GPCR_SMILES_ROW_CACHE_MAX", "128"))


def _data_root() -> Path:
    raw = os.environ.get("GPCR_DATA_ROOT", "").strip() or os.environ.get(
        "MANUSCRIPT_DATA_ROOT", ""
    ).strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2]


def _numeric_row_dict(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in row.index:
        if col in _EXCLUDE:
            continue
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(val):
            continue
        out[str(col)] = float(val)
    return out


def _numeric_row_dict_from_values(headers: list, values: tuple) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col, val in zip(headers, values):
        if not col or col in _EXCLUDE:
            continue
        try:
            fv = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if fv != fv:
            continue
        out[str(col)] = fv
    return out


def _find_smiles_row_in_xlsx(path: Path, canonical_smiles: str) -> Dict[str, float]:
    """Scan one workbook for one SMILES without loading the full sheet into RAM."""
    if not path.is_file():
        return {}
    try:
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.active
            rows = ws.iter_rows(values_only=True)
            header_row = next(rows, None)
            if not header_row:
                return {}
            headers = [str(h) if h is not None else "" for h in header_row]
            try:
                smiles_ix = next(i for i, h in enumerate(headers) if h == "SMILES")
            except StopIteration:
                return {}
            for values in rows:
                if values is None or smiles_ix >= len(values):
                    continue
                smi = values[smiles_ix]
                if smi is None or not str(smi).strip():
                    continue
                canon = _canonicalize_smiles(str(smi))
                if canon == canonical_smiles:
                    return _numeric_row_dict_from_values(headers, values)
        finally:
            wb.close()
    except Exception:
        pass

    try:
        df = pd.read_excel(path)
    except Exception:
        return {}
    if "SMILES" not in df.columns:
        return {}
    for _, row in df.iterrows():
        smi = row.get("SMILES")
        if pd.isna(smi) or not str(smi).strip():
            continue
        canon = _canonicalize_smiles(str(smi))
        if canon == canonical_smiles:
            return _numeric_row_dict(row)
    return {}


def _cache_smiles_row(cache_key: tuple[str, str], row: Dict[str, float]) -> Dict[str, float]:
    while len(_SMILES_ROW_CACHE) >= _MAX_SMILES_ROW_CACHE:
        _SMILES_ROW_CACHE.popitem(last=False)
    _SMILES_ROW_CACHE[cache_key] = row
    return dict(row)


def _lookup_smiles_across_workbooks(folder: str, canonical_smiles: str) -> Dict[str, float]:
    cache_key = (folder, canonical_smiles)
    if cache_key in _SMILES_ROW_CACHE:
        _SMILES_ROW_CACHE.move_to_end(cache_key)
        return dict(_SMILES_ROW_CACHE[cache_key])

    root = _data_root() / folder
    merged: Dict[str, float] = {}
    for suffix in _WORKBOOK_SUFFIXES:
        path = root / f"{folder}_{suffix}"
        if not path.exists():
            continue
        row = _find_smiles_row_in_xlsx(path, canonical_smiles)
        if row:
            merged.update(row)

    return _cache_smiles_row(cache_key, merged)


def _load_receptor_smiles_index(receptor: str) -> Dict[str, Dict[str, float]]:
    """Load all SMILES for one receptor (batch); LRU-cached and bounded."""
    if receptor in _RECEPTOR_TABLE_CACHE:
        _RECEPTOR_TABLE_CACHE.move_to_end(receptor)
        return _RECEPTOR_TABLE_CACHE[receptor]

    folder = _data_root() / receptor
    index: Dict[str, Dict[str, float]] = {}
    if not folder.is_dir():
        _RECEPTOR_TABLE_CACHE[receptor] = index
        return index

    for suffix in _WORKBOOK_SUFFIXES:
        path = folder / f"{receptor}_{suffix}"
        if not path.exists():
            continue
        df = pd.read_excel(path)
        if "SMILES" not in df.columns:
            continue
        for _, row in df.iterrows():
            smi = row.get("SMILES")
            if pd.isna(smi) or not str(smi).strip():
                continue
            canon = _canonicalize_smiles(str(smi))
            if not canon:
                continue
            prev = index.get(canon, {})
            prev.update(_numeric_row_dict(row))
            index[canon] = prev

    while len(_RECEPTOR_TABLE_CACHE) >= _MAX_RECEPTOR_CACHE:
        _RECEPTOR_TABLE_CACHE.popitem(last=False)
    _RECEPTOR_TABLE_CACHE[receptor] = index
    return index


def ligand_dict_from_new_workbooks(receptor: str, canonical_smiles: str) -> Dict[str, float]:
    """Return ligand columns from _NEW files for this receptor + SMILES, or {}."""
    folder = resolve_receptor_folder_name(receptor)
    if folder is None:
        return {}

    if os.environ.get("GPCR_LOAD_FULL_RECEPTOR_INDEX", "").strip().lower() in ("1", "true", "yes"):
        index = _load_receptor_smiles_index(folder)
        return dict(index.get(canonical_smiles, {}))

    return _lookup_smiles_across_workbooks(folder, canonical_smiles)


def resolve_receptor_folder_name(receptor_input: str) -> Optional[str]:
    from .receptor_names import resolve_receptor_folder

    return resolve_receptor_folder(receptor_input, _data_root())
