"""
Load manuscript-trained models (independent ligand, scaffold, LORO, stacking ensemble).

Artifacts are manuscript model exports under artifacts/manuscript/.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

EVALUATION_REGIMES = (
    "independent_ligand",
    "scaffold",
    "loro",
)

BASE_MODEL_TYPES = ("rf", "lightgbm", "xgboost", "ensemble")


def manuscript_artifacts_root(project_root: Path) -> Path:
    return project_root / "artifacts" / "manuscript"


def _align_proba(probs: np.ndarray, model_classes: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Map predict_proba columns to Agonist=0, Antagonist=1, Inactive=2."""
    out = np.zeros((probs.shape[0], n_classes), dtype=np.float64)
    for i in range(n_classes):
        if i in model_classes:
            col = int(np.where(model_classes == i)[0][0])
            out[:, i] = probs[:, col]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


class StackingEnsemblePredictor:
    """
    Manuscript §2.9 stacking: RF + XGB + LGB base learners → logistic regression meta (9 inputs).

    When ``tree_scaler`` is set (Code S21), XGB/LGB see StandardScaler-transformed features;
    RF always uses the raw feature vector.
    """

    def __init__(
        self,
        rf_model: Any,
        xgb_model: Any,
        lgb_model: Any,
        meta_model: Any,
        n_classes: int = 3,
        tree_scaler: Any = None,
    ):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.meta_model = meta_model
        self.tree_scaler = tree_scaler
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)
        nfi = getattr(rf_model, "n_features_in_", None)
        self.n_features_in_ = int(nfi) if nfi is not None else None

    def _tree_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.tree_scaler is not None:
            return self.tree_scaler.transform(X)
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_tree = self._tree_matrix(X)
        p_rf = _align_proba(self.rf_model.predict_proba(X), self.rf_model.classes_, self.n_classes)
        p_xgb = _align_proba(
            self.xgb_model.predict_proba(X_tree), self.xgb_model.classes_, self.n_classes
        )
        p_lgb = _align_proba(
            self.lgb_model.predict_proba(X_tree), self.lgb_model.classes_, self.n_classes
        )
        meta_x = np.hstack([p_rf, p_xgb, p_lgb])
        meta_probs = self.meta_model.predict_proba(meta_x)
        meta_classes = getattr(self.meta_model, "classes_", self.classes_)
        return _align_proba(meta_probs, meta_classes, self.n_classes)


def _joblib_load_model(path: Path) -> Any:
    """Load estimator; mmap on Cloud (GPCR_JOBLIB_MMAP=1) to reduce RAM spikes."""
    use_mmap = os.environ.get("GPCR_JOBLIB_MMAP", "").strip().lower() in ("1", "true", "yes")
    if use_mmap:
        try:
            return joblib.load(path, mmap_mode="r")
        except (TypeError, ValueError, OSError):
            pass
    return joblib.load(path)


def _valid_model_path(path: Path, min_bytes: int = 50_000) -> bool:
    """Skip placeholder seed files (tiny RandomForest stubs)."""
    if not path.is_file():
        return False
    if path.stat().st_size < min_bytes:
        return False
    return True


def _load_manifest(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _parse_seed_from_name(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def scan_manuscript_artifacts(project_root: Path) -> Dict[str, Any]:
    """
    Report which regime × model × seeds exist on disk (for Streamlit UI).

    Returns:
        {
          "has_manifest": bool,
          "n_features": int | None,
          "regimes": {
            "independent_ligand": {"rf": [42], "ensemble": [42], ...},
            ...
          },
          "seeds": [42],
        }
    """
    root = manuscript_artifacts_root(project_root)
    manifest = _load_manifest(root) if root.is_dir() else {}
    regimes: Dict[str, Dict[str, List[int]]] = {}
    all_seeds: Set[int] = set()

    if not root.is_dir():
        return {"has_manifest": False, "n_features": None, "regimes": {}, "seeds": []}

    for regime_dir in root.iterdir():
        if not regime_dir.is_dir() or regime_dir.name == "manifest.json":
            continue
        regime = regime_dir.name
        regimes[regime] = {}
        for model_dir in regime_dir.iterdir():
            if not model_dir.is_dir():
                continue
            mt = model_dir.name
            seeds: List[int] = []
            for p in model_dir.glob("*.pkl"):
                if not _valid_model_path(p, min_bytes=100_000 if mt == "ensemble" else 50_000):
                    continue
                sd = _parse_seed_from_name(p.name)
                if sd is not None:
                    seeds.append(sd)
            if seeds:
                regimes[regime][mt] = sorted(set(seeds))
                all_seeds.update(seeds)

    return {
        "has_manifest": (root / "manifest.json").is_file(),
        "n_features": manifest.get("n_features"),
        "regimes": regimes,
        "seeds": sorted(all_seeds),
    }


def resolve_manuscript_model_dir(
    project_root: Path,
    evaluation_regime: str,
    model_type: str,
    seed: int = 42,
) -> Optional[Path]:
    """
    Return folder containing model file(s) for regime × model_type × seed.

    Layout:
      artifacts/manuscript/{regime}/{model_type}/model_seed{seed}.pkl
      artifacts/manuscript/{regime}/ensemble/stacking_seed{seed}.pkl
    """
    root = manuscript_artifacts_root(project_root)
    regime = evaluation_regime.strip().lower()
    mt = model_type.strip().lower()
    if regime not in EVALUATION_REGIMES:
        return None

    if mt == "ensemble":
        folder = root / regime / "ensemble"
        if not folder.is_dir():
            return None
        for name in (f"stacking_seed{seed}.pkl", f"model_seed{seed}.pkl"):
            if _valid_model_path(folder / name, min_bytes=100_000):
                return folder
        return None

    folder = root / regime / mt
    if not folder.is_dir():
        return None
    if _valid_model_path(folder / f"model_seed{seed}.pkl"):
        return folder
    for p in sorted(folder.glob("model_seed*.pkl")):
        if _valid_model_path(p):
            return folder
    return None


def resolve_loro_model_path(
    project_root: Path,
    model_type: str,
    receptor_folder: str,
    seed: int = 42,
) -> Optional[Path]:
    """LORO: one model per held-out receptor folder."""
    root = manuscript_artifacts_root(project_root)
    mt = model_type.strip().lower()
    if mt == "ensemble":
        return None
    folder = root / "loro" / mt / receptor_folder
    if not folder.is_dir():
        return None
    p = folder / f"model_seed{seed}.pkl"
    if _valid_model_path(p):
        return p
    for cand in sorted(folder.glob("model_seed*.pkl")):
        if _valid_model_path(cand):
            return cand
    return None


def load_manuscript_models(
    project_root: Path,
    evaluation_regime: str,
    model_type: str,
    seed: int = 42,
    receptor_folder: Optional[str] = None,
) -> Tuple[List[Any], dict]:
    """
    Load estimator(s) for prediction.

    Returns (models_list, manifest_dict). For ensemble, list has one StackingEnsemblePredictor.
    For LORO, pass receptor_folder (Josh folder name, e.g. beta2).
    """
    root = manuscript_artifacts_root(project_root)
    regime = evaluation_regime.strip().lower()
    mt = model_type.strip().lower()

    manifest = _load_manifest(root)

    if regime == "loro":
        if not receptor_folder:
            raise ValueError("LORO models require receptor_folder (pocket folder name, e.g. beta2).")
        path = resolve_loro_model_path(project_root, mt, receptor_folder, seed=seed)
        if path is None:
            raise FileNotFoundError(
                f"No LORO model for receptor '{receptor_folder}' ({mt}, seed={seed}). "
                f"Export LORO models into artifacts/manuscript/loro/."
            )
        model_path = path if path.suffix == ".pkl" else path / f"model_seed{seed}.pkl"
        return [_joblib_load_model(model_path)], manifest

    model_dir = resolve_manuscript_model_dir(project_root, regime, mt, seed=seed)
    if model_dir is None:
        raise FileNotFoundError(
            f"Manuscript artifacts not found for regime={regime}, model={mt}, seed={seed}. "
            f"Expected under {root / regime}. Deploy exported models first."
        )

    if mt == "ensemble":
        for name in (f"stacking_seed{seed}.pkl", f"model_seed{seed}.pkl"):
            p = model_dir / name
            if _valid_model_path(p, min_bytes=100_000):
                return [_joblib_load_model(p)], manifest
        raise FileNotFoundError(f"No stacking bundle in {model_dir}")

    models = []
    use_cloud = os.environ.get("GPCR_CLOUD_LITE", "").strip().lower() in ("1", "true", "yes") or (
        str(os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "")).strip().lower() == "cloud"
    )
    cloud_path = model_dir / f"model_seed{seed}_cloud.pkl"
    seed_path = model_dir / f"model_seed{seed}.pkl"
    pick: Optional[Path] = None
    if use_cloud:
        if not _valid_model_path(cloud_path):
            raise FileNotFoundError(
                f"Streamlit Cloud requires {cloud_path.name} (~25 MB, 150 trees). "
                "Deploy model_seed*_cloud.pkl on Streamlit Cloud — do not use the full "
                f"model_seed{seed}.pkl (1000 trees) on Cloud; it exceeds ~1 GB RAM."
            )
        pick = cloud_path
    elif _valid_model_path(seed_path):
        pick = seed_path
    if pick is not None:
        models.append(_joblib_load_model(pick))
    else:
        for p in sorted(model_dir.glob("model_seed*.pkl")):
            if p.name.endswith("_cloud.pkl"):
                continue
            if _valid_model_path(p):
                models.append(_joblib_load_model(p))
    if not models:
        raise FileNotFoundError(f"No valid model_seed*.pkl in {model_dir}")
    return models, manifest


def manuscript_bundle_available(project_root: Path) -> bool:
    root = manuscript_artifacts_root(project_root)
    if not root.is_dir() or not (root / "manifest.json").exists():
        return False
    scan = scan_manuscript_artifacts(project_root)
    return bool(scan.get("regimes"))
