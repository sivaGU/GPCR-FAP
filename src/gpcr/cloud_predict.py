"""
Minimal manuscript inference for Streamlit Cloud (~1 GB RAM).

Loads one model at a time (RF cloud / XGB / LGB), predicts, then frees memory.
Does not use GPCRPredictor. Ensemble is not supported on Cloud.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .manuscript_bundle import _align_proba, _joblib_load_model, _valid_model_path
from .manuscript_features import build_manuscript_feature_row, load_feature_columns
from .predict import PredictResult, _canonicalize_smiles

CloudModelType = Literal["rf", "lightgbm", "xgboost"]
CLOUD_MODEL_TYPES = ("rf", "lightgbm", "xgboost")
_MODEL_FOLDER = {"rf": "rf", "lightgbm": "lightgbm", "xgboost": "xgboost"}


def _model_dir(project_root: Path, evaluation_regime: str, model_type: str) -> Path:
    mt = model_type.strip().lower()
    folder = _MODEL_FOLDER.get(mt, mt)
    return Path(project_root) / "artifacts" / "manuscript" / evaluation_regime / folder


def _pick_model_path(
    project_root: Path,
    evaluation_regime: str,
    model_type: str,
    seed: int = 42,
) -> Optional[Path]:
    """Prefer cloud RF slim pickle on Streamlit Cloud; full RF locally when present."""
    mt = model_type.strip().lower()
    model_dir = _model_dir(project_root, evaluation_regime, mt)
    if not model_dir.is_dir():
        return None
    cloud_rf = model_dir / f"model_seed{seed}_cloud.pkl"
    full = model_dir / f"model_seed{seed}.pkl"
    on_cloud = os.environ.get("GPCR_CLOUD_LITE", "").strip().lower() in ("1", "true", "yes") or (
        str(os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "")).strip().lower() == "cloud"
    )
    if mt == "rf":
        if on_cloud and _valid_model_path(cloud_rf):
            return cloud_rf
        if _valid_model_path(full):
            return full
        if _valid_model_path(cloud_rf):
            return cloud_rf
        return None
    if _valid_model_path(full):
        return full
    for p in sorted(model_dir.glob("model_seed*.pkl")):
        if _valid_model_path(p):
            return p
    return None


def cloud_model_path(
    project_root: Path,
    evaluation_regime: str,
    model_type: str,
    seed: int = 42,
) -> Path:
    mt = model_type.strip().lower()
    pick = _pick_model_path(project_root, evaluation_regime, mt, seed)
    if pick is not None:
        return pick
    folder = _MODEL_FOLDER.get(mt)
    if folder is None:
        raise ValueError(f"Unsupported cloud model_type: {model_type}")
    name = f"model_seed{seed}_cloud.pkl" if mt == "rf" else f"model_seed{seed}.pkl"
    return _model_dir(project_root, evaluation_regime, mt) / name


def cloud_model_ready(
    project_root: Path,
    evaluation_regime: str,
    model_type: str,
    seed: int = 42,
    *,
    min_bytes: int = 50_000,
) -> bool:
    if model_type.strip().lower() not in CLOUD_MODEL_TYPES:
        return False
    path = _pick_model_path(project_root, evaluation_regime, model_type, seed)
    if path is None:
        path = cloud_model_path(project_root, evaluation_regime, model_type, seed)
    return path.is_file() and path.stat().st_size >= min_bytes


def _invalid(
    *,
    receptor: str,
    ligand_smiles: str,
    canon: str,
    error: str,
) -> PredictResult:
    return PredictResult(
        is_valid=False,
        receptor=receptor,
        ligand_smiles=ligand_smiles,
        canonical_smiles=canon,
        predicted_class="Unknown",
        class_id=-1,
        prob_agonist=0.0,
        prob_antagonist=0.0,
        prob_inactive=0.0,
        error=error,
    )


def _release_model(model: object) -> None:
    if hasattr(model, "estimators_"):
        model.estimators_ = []  # type: ignore[attr-defined]
    del model


def predict_cloud_manuscript(
    project_root: Path,
    receptor: str,
    ligand_smiles: str,
    evaluation_regime: str = "independent_ligand",
    seed: int = 42,
    model_type: str = "rf",
) -> PredictResult:
    root = Path(project_root)
    mt = model_type.strip().lower()
    if mt not in CLOUD_MODEL_TYPES:
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon="",
            error=f"Cloud supports rf, lightgbm, xgboost only (not {model_type}).",
        )

    canon = _canonicalize_smiles(ligand_smiles)
    if canon is None:
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon="",
            error="Invalid SMILES",
        )

    model_path = _pick_model_path(root, evaluation_regime, mt, seed)
    if model_path is None or not model_path.is_file():
        fallback = cloud_model_path(root, evaluation_regime, mt, seed)
        if mt == "rf":
            hint = "Deploy model_seed*_cloud.pkl via Git LFS."
        else:
            hint = f"Deploy {fallback.name} under artifacts/manuscript/{evaluation_regime}/{mt}/."
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon=canon,
            error=f"Missing model file. {hint}",
        )

    cols = load_feature_columns(root)
    vec = build_manuscript_feature_row(root, receptor, canon, feature_columns=cols)
    if vec is None:
        return _invalid(
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canon=canon,
            error="Could not build feature row (receptor pockets or ligand lookup).",
        )

    model: Optional[object] = None
    try:
        model = _joblib_load_model(model_path)
        X = np.ascontiguousarray(vec.reshape(1, -1), dtype=np.float64)
        raw = model.predict_proba(X)
        probs = _align_proba(raw, model.classes_, n_classes=3)[0]
        class_id = int(np.argmax(probs))
        names = ["Agonist", "Antagonist", "Inactive"]
        return PredictResult(
            is_valid=True,
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canonical_smiles=canon,
            predicted_class=names[class_id],
            class_id=class_id,
            prob_agonist=float(probs[0]),
            prob_antagonist=float(probs[1]),
            prob_inactive=float(probs[2]),
            prob_std_error=None,
        )
    finally:
        if model is not None:
            _release_model(model)
        del vec
        gc.collect()


def predict_cloud_rf(
    project_root: Path,
    receptor: str,
    ligand_smiles: str,
    evaluation_regime: str = "independent_ligand",
    seed: int = 42,
) -> PredictResult:
    """Backward-compatible wrapper."""
    return predict_cloud_manuscript(
        project_root,
        receptor,
        ligand_smiles,
        evaluation_regime=evaluation_regime,
        seed=seed,
        model_type="rf",
    )
