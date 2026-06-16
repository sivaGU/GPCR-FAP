"""
Build manuscript feature rows (full ligand + receptor + interaction columns).

Requires artifacts/manuscript/manifest.json with feature_columns produced by export script.
Ligand columns come from *_NEW.xlsx (training); use ligand_feature_lookup.joblib when built,
otherwise fall back to on-the-fly RDKit + Mordred (fewer columns — biased toward Inactive).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from rdkit import Chem

from .ligand_enrichment import build_ligand_descriptor_dict
from .ligand_lookup_store import (
    count_entries as sqlite_lookup_count,
    fetch_ligand_features,
    should_use_sqlite_lookup,
    sqlite_lookup_available,
    sqlite_lookup_path,
)
from .predict import _canonicalize_smiles
from .receptor_names import resolve_receptor_folder

_LIGAND_LOOKUP_CACHE: Optional[Tuple[Dict[str, Dict[str, float]], str]] = None

# Streamlit Cloud (~1 GB RAM): do not joblib.load the full lookup except at predict time.
_LIGAND_LOOKUP_META_CACHE: Optional[Dict[str, object]] = None


def _manuscript_root(project_root: Path) -> Path:
    return project_root / "artifacts" / "manuscript"


def _cloud_lite_mode() -> bool:
    """Streamlit Cloud: skip Mordred, workbooks, and other RAM-heavy inference paths."""
    if os.environ.get("GPCR_CLOUD_LITE", "").strip().lower() in ("0", "false", "no"):
        return False
    if os.environ.get("GPCR_CLOUD_LITE", "").strip().lower() in ("1", "true", "yes"):
        return True
    if Path("/mount/src").is_dir():
        return True
    return str(os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "")).strip().lower() == "cloud"


def load_feature_columns(project_root: Path) -> List[str]:
    manifest_path = _manuscript_root(project_root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Deploy manuscript exports under artifacts/manuscript/."
        )
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    cols = data.get("feature_columns")
    if not cols:
        raise ValueError("manifest.json must contain feature_columns list.")
    return list(cols)


def _try_import_shared_utilities():
    ml_root = os.environ.get("MANUSCRIPT_ML_ROOT", "").strip()
    if not ml_root:
        return None
    import sys

    p = Path(ml_root)
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
    try:
        import shared_utilities as su  # type: ignore

        return su
    except ImportError:
        return None


def ligand_lookup_meta(project_root: Path) -> Dict[str, object]:
    """Lightweight lookup stats (no joblib load). Used for Streamlit diagnostics on Cloud."""
    global _LIGAND_LOOKUP_META_CACHE
    if _LIGAND_LOOKUP_META_CACHE is not None:
        return _LIGAND_LOOKUP_META_CACHE

    root = _manuscript_root(project_root)
    meta_path = root / "ligand_feature_lookup_meta.json"
    joblib_path = root / "ligand_feature_lookup.joblib"
    sqlite_path = sqlite_lookup_path(root)
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                _LIGAND_LOOKUP_META_CACHE = dict(json.load(f))
                if sqlite_path.is_file():
                    _LIGAND_LOOKUP_META_CACHE.setdefault(
                        "sqlite_bytes", sqlite_path.stat().st_size
                    )
                return _LIGAND_LOOKUP_META_CACHE
        except Exception:
            pass

    if sqlite_lookup_available(root):
        n = sqlite_lookup_count(root)
        _LIGAND_LOOKUP_META_CACHE = {
            "n_smiles": n,
            "source": "sqlite",
            "storage": "sqlite",
            "sqlite_bytes": sqlite_path.stat().st_size,
        }
    elif joblib_path.is_file() and joblib_path.stat().st_size > 1_000_000:
        _LIGAND_LOOKUP_META_CACHE = {
            "n_smiles": None,
            "source": "_NEW.xlsx (meta missing — run build_manuscript_ligand_lookup.py)",
            "joblib_bytes": joblib_path.stat().st_size,
        }
    else:
        _LIGAND_LOOKUP_META_CACHE = {"n_smiles": 0, "source": "missing"}
    return _LIGAND_LOOKUP_META_CACHE


def ligand_lookup_entry_count(project_root: Path) -> int:
    root = _manuscript_root(project_root)
    if sqlite_lookup_available(root):
        n = sqlite_lookup_count(root)
        return int(n) if n is not None else 0
    if _skip_ligand_lookup_for_root(project_root):
        return 0
    meta = ligand_lookup_meta(project_root)
    n = meta.get("n_smiles")
    if isinstance(n, int) and n > 0:
        return n
    joblib_path = root / "ligand_feature_lookup.joblib"
    if joblib_path.is_file() and joblib_path.stat().st_size > 1_000_000:
        return -1  # present but count unknown without loading
    return 0


def _skip_ligand_lookup_for_root(project_root: Path) -> bool:
    if should_use_sqlite_lookup(_manuscript_root(project_root)):
        return False
    return os.environ.get("GPCR_SKIP_LIGAND_LOOKUP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _lookup_row_for_smiles(
    project_root: Path, canon: str
) -> Tuple[Dict[str, float], str]:
    root = _manuscript_root(project_root)
    if should_use_sqlite_lookup(root):
        row = fetch_ligand_features(root, canon)
        if row:
            return row, "sqlite"
        if _skip_ligand_lookup_for_root(project_root):
            return {}, "sqlite_miss"
    if _skip_ligand_lookup_for_root(project_root):
        return {}, "skipped_for_ram"
    lookup, src = _load_ligand_lookup(project_root)
    row = dict(lookup.get(canon, {}))
    if row:
        return row, src if src not in ("missing", "corrupt_or_incomplete") else "lookup"
    return {}, src


def _load_ligand_lookup(project_root: Path) -> Tuple[Dict[str, Dict[str, float]], str]:
    """Load SMILES → ligand descriptor dict (heavy — only for inference, not page load)."""
    global _LIGAND_LOOKUP_CACHE
    if _LIGAND_LOOKUP_CACHE is not None:
        return _LIGAND_LOOKUP_CACHE

    if _skip_ligand_lookup_for_root(project_root):
        _LIGAND_LOOKUP_CACHE = ({}, "skipped_for_ram")
        return _LIGAND_LOOKUP_CACHE

    path = _manuscript_root(project_root) / "ligand_feature_lookup.joblib"
    if not path.exists():
        _LIGAND_LOOKUP_CACHE = ({}, "missing")
        return _LIGAND_LOOKUP_CACHE

    try:
        payload = joblib.load(path)
        lookup = payload.get("lookup", {})
        _LIGAND_LOOKUP_CACHE = (lookup, str(payload.get("source", "_NEW.xlsx")))
    except Exception:
        _LIGAND_LOOKUP_CACHE = ({}, "corrupt_or_incomplete")
    return _LIGAND_LOOKUP_CACHE


def _ligand_row_for_smiles(
    project_root: Path,
    receptor_input: str,
    canon: str,
    mol: Chem.Mol,
) -> Dict[str, float]:
    from .ligand_enrichment import compute_rdkit_2d_descriptors

    lookup_row, lookup_src = _lookup_row_for_smiles(project_root, canon)
    row: Dict[str, float] = dict(lookup_row)
    if not _cloud_lite_mode():
        from .new_workbook_ligand import ligand_dict_from_new_workbooks

        wb = ligand_dict_from_new_workbooks(receptor_input, canon)
        if wb:
            row.update(wb)
    # Training lookup already has ~3k+ ligand columns — skip Mordred (large RAM spike on Cloud).
    rich_lookup = lookup_src in ("sqlite", "lookup") and len(row) >= 400
    if rich_lookup:
        return row
    use_mordred = not _cloud_lite_mode() and len(row) < 400
    if use_mordred:
        computed = build_ligand_descriptor_dict(canon, mol=mol, include_mordred=True) or {}
    else:
        computed = compute_rdkit_2d_descriptors(mol)
    for col, val in computed.items():
        if col not in row:
            row[col] = float(val)
    return row

def _receptor_features(receptor_input: str, data_root: Path) -> Optional[Dict[str, float]]:
    folder = resolve_receptor_folder(receptor_input, data_root)
    if folder is None:
        return None
    if os.environ.get("GPCR_POCKET_FEATURES_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        from .predict import _aggregate_receptor_feature_dict

        return _aggregate_receptor_feature_dict(receptor_input)
    su = _try_import_shared_utilities()
    if su is not None:
        rec = su.aggregate_receptor_features(folder)
        return rec if rec else None
    from .predict import _aggregate_receptor_feature_dict

    return _aggregate_receptor_feature_dict(receptor_input)


def _fill_row_from_parts(
    feature_columns: List[str],
    lig_row: Dict[str, float],
    rec_feats: Dict[str, float],
) -> np.ndarray:
    row_dict: Dict[str, float] = {}
    for col in feature_columns:
        if col in lig_row:
            row_dict[col] = lig_row[col]
        elif col in rec_feats:
            row_dict[col] = float(rec_feats[col])
        elif col.startswith("INT_") and "_X_" in col:
            lig_k, rec_k = col[4:].split("_X_", 1)
            row_dict[col] = float(lig_row.get(lig_k, 0.0)) * float(rec_feats.get(rec_k, 0.0))
        else:
            row_dict[col] = 0.0
    return np.array([float(row_dict.get(c, 0.0)) for c in feature_columns], dtype=np.float32)


def build_manuscript_feature_row(
    project_root: Path,
    receptor_input: str,
    ligand_smiles: str,
    feature_columns: Optional[List[str]] = None,
) -> Optional[np.ndarray]:
    """
    Build one row aligned to manuscript training columns (from manifest).
    """
    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return None

    if feature_columns is None:
        feature_columns = load_feature_columns(project_root)

    data_root = Path(os.environ.get("GPCR_DATA_ROOT", "").strip() or project_root)
    rec_feats = _receptor_features(receptor_input, data_root)
    if not rec_feats:
        return None

    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return None
    lig_row = _ligand_row_for_smiles(project_root, receptor_input, canon, mol)
    if not lig_row:
        return None

    return _fill_row_from_parts(feature_columns, lig_row, rec_feats)


def build_demo_2103_features(receptor_input: str, ligand_smiles: str) -> Optional[np.ndarray]:
    """Legacy 2103-dim demo bundle (10 RDKit + Morgan + 31 receptor + 14 interaction)."""
    from .predict import _compute_full_features

    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return None
    return _compute_full_features(receptor_input, canon)


_LIGAND_WORKBOOK_SUFFIXES = (
    "agonist_enriched_NEW.xlsx",
    "antagonist_enriched_NEW.xlsx",
    "non_active_compounds_NEW.xlsx",
)


def _gpcr_data_root_path() -> Path:
    raw = os.environ.get("GPCR_DATA_ROOT", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2]


def manuscript_data_health(project_root: Path) -> Dict[str, object]:
    """
    Check whether Cloud/local GPCR_DATA_ROOT can support manuscript predictions.

    Workbooks (*_NEW.xlsx per receptor) or ligand_feature_lookup.joblib are required
    for training-aligned ligand columns; pocket CSVs + Mordred alone are weaker.
    """
    gpcr_root = _gpcr_data_root_path()
    ml_root = Path(os.environ.get("MANUSCRIPT_ML_ROOT", "").strip() or (gpcr_root / "ML_code"))
    lookup_entries = ligand_lookup_entry_count(project_root)

    receptors_with_workbooks: List[str] = []
    if gpcr_root.is_dir():
        for child in sorted(gpcr_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name in ("Josh_Receptor_Features", "ML_code", "runtime_data"):
                continue
            for suffix in _LIGAND_WORKBOOK_SUFFIXES:
                if (child / f"{child.name}_{suffix}").is_file():
                    receptors_with_workbooks.append(child.name)
                    break

    josh = (gpcr_root / "Josh_Receptor_Features").is_dir()
    ml_ok = ml_root.is_dir()
    beta2_wb = (gpcr_root / "beta2" / "beta2_agonist_enriched_NEW.xlsx").is_file()
    manuscript_ready = josh and ml_ok
    has_lookup = lookup_entries != 0
    prediction_ready = manuscript_ready and (bool(receptors_with_workbooks) or has_lookup)

    issues: List[str] = []
    if not josh:
        issues.append("Missing Josh_Receptor_Features under GPCR_DATA_ROOT")
    if not ml_ok:
        issues.append("Missing ML_code (set MANUSCRIPT_ML_ROOT or include ML_code in zip)")
    if manuscript_ready and not receptors_with_workbooks and not has_lookup:
        issues.append(
            "No per-receptor *_NEW.xlsx workbooks and no ligand_feature_lookup.joblib — "
            "ligands use Mordred-only fallback (~half the descriptor signal vs local full data)"
        )

    return {
        "gpcr_data_root": str(gpcr_root.resolve()) if gpcr_root.exists() else str(gpcr_root),
        "manuscript_ready": manuscript_ready,
        "prediction_ready": prediction_ready,
        "receptors_with_workbooks": receptors_with_workbooks,
        "receptor_workbook_count": len(receptors_with_workbooks),
        "beta2_agonist_workbook": beta2_wb,
        "ligand_lookup_entries": lookup_entries if lookup_entries >= 0 else None,
        "issues": issues,
    }


def inference_feature_summary(
    project_root: Path,
    receptor_input: str,
    ligand_smiles: str,
) -> Optional[Dict[str, object]]:
    """Per-prediction ligand path: workbook hit vs Mordred fallback + nonzero count."""
    from .new_workbook_ligand import ligand_dict_from_new_workbooks

    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return None
    cols = load_feature_columns(project_root)
    vec = build_manuscript_feature_row(project_root, receptor_input, canon, feature_columns=cols)
    if vec is None:
        return None
    wb = ligand_dict_from_new_workbooks(receptor_input, canon)
    nz = int(np.count_nonzero(np.abs(vec) > 1e-12))
    n_cols = len(cols)
    lookup_row, lookup_src = _lookup_row_for_smiles(project_root, canon)
    if lookup_src == "skipped_for_ram":
        source = "mordred_cloud"
    elif len(wb) >= 500:
        source = "workbook"
    elif len(wb) > 0:
        source = "workbook_partial"
    elif lookup_src == "sqlite" and len(lookup_row) >= 500:
        source = "lookup_sqlite"
    elif lookup_src == "sqlite" and len(lookup_row) > 0:
        source = "lookup_sqlite_partial"
    elif len(lookup_row) >= 500:
        source = "lookup"
    elif len(lookup_row) > 0:
        source = "lookup_partial"
    else:
        source = "mordred_fallback"
    return {
        "canonical_smiles": canon,
        "workbook_ligand_keys": len(wb),
        "nonzero_features": nz,
        "manifest_features": n_cols,
        "nonzero_pct": round(100.0 * nz / n_cols, 1) if n_cols else 0.0,
        "ligand_source": source,
    }


def manuscript_debug_status(project_root: Path) -> Dict[str, object]:
    """
    Runtime diagnostics for manuscript feature-path resolution.

    Used by Streamlit to surface whether inference is using training-aligned assets
    or falling back to sparse/derived features.
    """
    root = _manuscript_root(project_root)
    manifest_path = root / "manifest.json"
    lookup_path = root / "ligand_feature_lookup.joblib"
    sqlite_path = sqlite_lookup_path(root)
    gpcr_root = Path(os.environ.get("GPCR_DATA_ROOT", "").strip() or project_root)
    ml_root = os.environ.get("MANUSCRIPT_ML_ROOT", "").strip()
    su = _try_import_shared_utilities()

    meta = ligand_lookup_meta(project_root)
    lookup_source = str(meta.get("source", "missing"))
    lookup_entries = ligand_lookup_entry_count(project_root)
    feature_columns_count = 0
    if manifest_path.exists():
        try:
            feature_columns_count = len(load_feature_columns(project_root))
        except Exception:
            feature_columns_count = 0

    health = manuscript_data_health(project_root)
    return {
        "gpcr_data_root": str(gpcr_root.resolve()),
        "ml_root": ml_root,
        "ml_root_exists": bool(ml_root and Path(ml_root).is_dir()),
        "shared_utilities_imported": su is not None,
        "manifest_exists": manifest_path.exists(),
        "manifest_feature_count": feature_columns_count,
        "ligand_lookup_exists": lookup_path.exists() or sqlite_path.exists(),
        "ligand_lookup_sqlite": sqlite_path.exists(),
        "ligand_lookup_entries": lookup_entries if lookup_entries >= 0 else meta.get("n_smiles"),
        "ligand_lookup_source": lookup_source,
        "ligand_lookup_joblib_bytes": meta.get("joblib_bytes"),
        "ligand_lookup_sqlite_bytes": meta.get("sqlite_bytes"),
        "prediction_ready": health["prediction_ready"],
        "receptor_workbook_count": health["receptor_workbook_count"],
        "beta2_agonist_workbook": health["beta2_agonist_workbook"],
        "data_issues": health["issues"],
    }
