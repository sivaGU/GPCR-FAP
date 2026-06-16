"""
GPCR Class A Functional Activity prediction module.

Predicts Agonist/Antagonist/Inactive for GPCR Class A receptor-ligand pairs.
Supports multi-class classification with uncertainty quantification.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

from .receptor_names import resolve_receptor_folder


@dataclass
class PredictResult:
    """Result of a single prediction."""
    is_valid: bool
    receptor: str
    ligand_smiles: str
    canonical_smiles: str
    predicted_class: str  # "Agonist", "Antagonist", or "Inactive"
    class_id: int  # 0=Agonist, 1=Antagonist, 2=Inactive
    prob_agonist: float
    prob_antagonist: float
    prob_inactive: float
    prob_std_error: Optional[float] = None  # Standard error of the mean probability
    prob_std_dev: Optional[float] = None  # Standard deviation of ensemble predictions
    threshold: Optional[float] = None
    error: str = ""


# Based on manuscript-aligned pipeline: 31 receptor features + 14 interaction terms
RECEPTOR_FEATURES_DIM = 31
INTERACTION_TERMS_DIM = 14

RECEPTOR_FEATURE_ORDER = [
    "num_residues",
    "num_aromatic",
    "num_acidic",
    "num_basic",
    "num_charge_positive",
    "num_charge_negative",
    "num_charge_neutral",
    "num_polar",
    "num_nonpolar",
    "num_size_small",
    "num_size_medium",
    "num_size_large",
    "num_sulfur",
    "num_hydroxyl",
    "num_amide",
    "aromatic_ratio",
    "basic_ratio",
    "acidic_ratio",
    "charge_positive_ratio",
    "charge_negative_ratio",
    "charge_neutral_ratio",
    "polar_ratio",
    "nonpolar_ratio",
    "size_small_ratio",
    "size_medium_ratio",
    "size_large_ratio",
    "sulfur_ratio",
    "hydroxyl_ratio",
    "amide_ratio",
    "avg_distance",
    "avg_conservation",
]

INTERACTION_PAIRS: List[Tuple[str, str]] = [
    ("LogP", "aromatic_ratio"),
    ("LogP", "num_basic"),
    ("TPSA", "acidic_ratio"),
    ("HBD", "num_hydroxyl"),
    ("HBA", "num_basic"),
    ("HBA", "num_amide"),
    ("AromaticRings", "aromatic_ratio"),
    ("AromaticRings", "num_aromatic"),
    ("FormalCharge", "num_charge_positive"),
    ("FormalCharge", "num_charge_negative"),
    ("MolWt", "num_residues"),
    ("RotatableBonds", "num_residues"),
    ("LogP", "avg_conservation"),
    ("TPSA", "avg_conservation"),
]


def get_gpcr_data_root() -> Path:
    """Directory containing Josh_Receptor_Features/ (for UI and path display)."""
    return _resolve_gpcr_data_root()


def _resolve_gpcr_data_root() -> Path:
    """
    Resolve GPCR data root (directory that contains Josh_Receptor_Features/).

    Priority (aligned with GPCR-FAP training handoff):
    GPCR_DATA_ROOT env → sibling **GUI_Folder** (canonical pocket CSVs) →
    ./Josh_Receptor_Features next to this repo → legacy GPCRtryagain path.
    """
    env_root = os.environ.get("GPCR_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    project_root = Path(__file__).resolve().parents[2]
    sibling_gui = project_root.parent / "GUI_Folder"
    if (sibling_gui / "Josh_Receptor_Features").is_dir():
        return sibling_gui
    if (project_root / "Josh_Receptor_Features").is_dir():
        return project_root
    return project_root.parent / "GPCRtryagain - Delete - Copy"


def _coerce_numeric_sum(series: pd.Series) -> int:
    if series.dtype == bool:
        return int(series.sum())
    return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def _aggregate_receptor_feature_dict(receptor_name: str) -> Optional[Dict[str, float]]:
    """
    Build 31 receptor pocket features from *_pocket_residues_with_conservation.csv
    files in Josh_Receptor_Features/<folder>.
    """
    root = _resolve_gpcr_data_root()
    folder = resolve_receptor_folder(receptor_name, root)
    if folder is None:
        return None
    pocket_dir = root / "Josh_Receptor_Features" / folder
    if not pocket_dir.exists():
        return None

    files = list(pocket_dir.glob("*_pocket_residues_with_conservation.csv"))
    if not files:
        return None
    df = pd.read_csv(files[0])
    if df.empty:
        return None

    def count_category(column: str, value: str) -> int:
        if column not in df.columns:
            return 0
        return int((df[column].astype(str).str.lower() == value).sum())

    total = len(df)
    feats: Dict[str, float] = {
        "num_residues": float(total),
        "num_aromatic": float(_coerce_numeric_sum(df["aromatic"]) if "aromatic" in df.columns else 0),
        "num_acidic": float(_coerce_numeric_sum(df["acidic"]) if "acidic" in df.columns else 0),
        "num_basic": float(_coerce_numeric_sum(df["basic"]) if "basic" in df.columns else 0),
        "num_charge_positive": float(count_category("charge", "positive")),
        "num_charge_negative": float(count_category("charge", "negative")),
        "num_charge_neutral": float(count_category("charge", "neutral")),
        "num_polar": float(count_category("polarity", "polar")),
        "num_nonpolar": float(count_category("polarity", "nonpolar")),
        "num_size_small": float(count_category("size", "small")),
        "num_size_medium": float(count_category("size", "medium")),
        "num_size_large": float(count_category("size", "large")),
        "num_sulfur": float(_coerce_numeric_sum(df["sulfur"]) if "sulfur" in df.columns else 0),
        "num_hydroxyl": float(_coerce_numeric_sum(df["hydroxyl"]) if "hydroxyl" in df.columns else 0),
        "num_amide": float(_coerce_numeric_sum(df["amide"]) if "amide" in df.columns else 0),
    }

    nr = max(feats["num_residues"], 1.0)
    feats["aromatic_ratio"] = feats["num_aromatic"] / nr
    feats["basic_ratio"] = feats["num_basic"] / nr
    feats["acidic_ratio"] = feats["num_acidic"] / nr
    feats["charge_positive_ratio"] = feats["num_charge_positive"] / nr
    feats["charge_negative_ratio"] = feats["num_charge_negative"] / nr
    feats["charge_neutral_ratio"] = feats["num_charge_neutral"] / nr
    feats["polar_ratio"] = feats["num_polar"] / nr
    feats["nonpolar_ratio"] = feats["num_nonpolar"] / nr
    feats["size_small_ratio"] = feats["num_size_small"] / nr
    feats["size_medium_ratio"] = feats["num_size_medium"] / nr
    feats["size_large_ratio"] = feats["num_size_large"] / nr
    feats["sulfur_ratio"] = feats["num_sulfur"] / nr
    feats["hydroxyl_ratio"] = feats["num_hydroxyl"] / nr
    feats["amide_ratio"] = feats["num_amide"] / nr
    feats["avg_distance"] = float(df["distance_to_ligand"].mean()) if "distance_to_ligand" in df.columns else 0.0
    feats["avg_conservation"] = float(df["conservation_score"].mean()) if "conservation_score" in df.columns else 0.0
    return feats


def _zero_receptor_feature_dict() -> Dict[str, float]:
    """Fallback receptor feature dict when pocket files are unavailable."""
    return {k: 0.0 for k in RECEPTOR_FEATURE_ORDER}


def get_available_receptors() -> List[str]:
    """Return receptor folder names that have pocket feature CSVs."""
    from .receptor_names import list_receptor_folders

    return list_receptor_folders(_resolve_gpcr_data_root())


def _get_receptor_features(receptor_name: str) -> Optional[np.ndarray]:
    """Extract manuscript-style 31 receptor pocket features for a receptor."""
    feats = _aggregate_receptor_feature_dict(receptor_name)
    if feats is None:
        return None
    return np.array([float(feats.get(key, 0.0)) for key in RECEPTOR_FEATURE_ORDER], dtype=np.float32)


def _compute_ligand_features(smiles: str) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    """
    Compute ligand features: PhysChem (10) + Morgan ECFP4 (2048) = 2058 features.

    Morgan: radius 2, 2048 bits, bit vector (not counts) — matches feature_config.json / training export.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    ligand_core = {
        "MolWt": float(Descriptors.MolWt(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "HBD": float(Descriptors.NumHDonors(mol)),
        "HBA": float(Descriptors.NumHAcceptors(mol)),
        "RotatableBonds": float(Descriptors.NumRotatableBonds(mol)),
        "Rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "FormalCharge": float(Chem.GetFormalCharge(mol)),
    }

    # Preserve historical 10-dim ligand vector expected by existing models.
    phys = np.array(
        [
            ligand_core["MolWt"],
            ligand_core["TPSA"],
            ligand_core["LogP"],
            ligand_core["HBD"],
            ligand_core["HBA"],
            ligand_core["RotatableBonds"],
            ligand_core["Rings"],
            ligand_core["HeavyAtomCount"],
            ligand_core["FractionCSP3"],
            ligand_core["AromaticRings"],
        ],
        dtype=np.float32,
    )
    
    # ECFP4 fingerprint (2048 bits)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    
    return np.hstack([phys, arr]), ligand_core


def _compute_interaction_features(ligand_core: Dict[str, float], receptor_dict: Dict[str, float]) -> np.ndarray:
    """
    Compute interaction terms between ligand and receptor features.
    
    Compute manuscript-aligned 14 interaction terms from shared_utilities.
    """
    vals: List[float] = []
    for lig, rec in INTERACTION_PAIRS:
        vals.append(float(ligand_core.get(lig, 0.0)) * float(receptor_dict.get(rec, 0.0)))
    return np.array(vals, dtype=np.float32)


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _compute_full_features(receptor_name: str, ligand_smiles: str) -> Optional[np.ndarray]:
    """
    Compute full feature vector: ligand features + receptor features + interaction terms.

    Expected dimensions:
    - Ligand: 2058 (10 PhysChem + 2048 ECFP4)
    - Receptor: 31
    - Interaction: 14
    - Total: 2103 features

    Returns None if the receptor is unknown or pocket CSVs are missing (no zero-fill).
    """
    root = _resolve_gpcr_data_root()
    if resolve_receptor_folder(receptor_name, root) is None:
        return None

    ligand_out = _compute_ligand_features(ligand_smiles)
    if ligand_out is None:
        return None

    ligand_feats, ligand_core = ligand_out

    receptor_dict = _aggregate_receptor_feature_dict(receptor_name)
    receptor_feats = _get_receptor_features(receptor_name)
    if receptor_feats is None or receptor_dict is None:
        return None

    interaction_feats = _compute_interaction_features(ligand_core, receptor_dict)
    return np.hstack([ligand_feats, receptor_feats, interaction_feats])


class GPCRPredictor:
    """Loaded predictor state (models, class names, threshold)."""

    def __init__(
        self,
        models: List,  # List of trained models (ensemble)
        class_names: List[str] = None,
        threshold: Optional[float] = None,
        expected_feature_dim: Optional[int] = None,
        feature_mode: str = "manuscript",
        project_root: Optional[Path] = None,
        evaluation_regime: Optional[str] = None,
        model_type: Optional[str] = None,
        seed: int = 42,
    ):
        self.models = models
        self.class_names = class_names or ["Agonist", "Antagonist", "Inactive"]
        self.threshold = threshold
        self.expected_feature_dim = expected_feature_dim
        self.feature_mode = feature_mode
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.evaluation_regime = evaluation_regime
        self.model_type = model_type
        self.seed = int(seed)

    def _models_for_prediction(self, receptor: str) -> List:
        """LORO uses a separate model per receptor (held-out during training)."""
        if self.evaluation_regime != "loro" or self.model_type == "ensemble":
            return self.models
        from .manuscript_bundle import load_manuscript_models
        from .receptor_names import resolve_receptor_folder

        root = _resolve_gpcr_data_root()
        folder = resolve_receptor_folder(receptor, root)
        if folder is None:
            return self.models
        models, _ = load_manuscript_models(
            self.project_root,
            "loro",
            self.model_type or "rf",
            seed=self.seed,
            receptor_folder=folder,
        )
        return models

    def _build_feature_vector(self, receptor: str, canon: str) -> Tuple[Optional[np.ndarray], str]:
        from .manuscript_features import build_manuscript_feature_row

        vec = build_manuscript_feature_row(self.project_root, receptor, canon)
        if vec is None:
            return None, "Could not build manuscript feature row (check manifest, Mordred, receptor pocket)."
        return vec, ""

    def predict(self, receptor: str, ligand_smiles: str) -> PredictResult:
        """Run full pipeline for one receptor-ligand pair."""
        canon = _canonicalize_smiles(ligand_smiles)
        if canon is None:
            return PredictResult(
                is_valid=False,
                receptor=receptor,
                ligand_smiles=ligand_smiles,
                canonical_smiles="",
                predicted_class="Unknown",
                class_id=-1,
                prob_agonist=0.0,
                prob_antagonist=0.0,
                prob_inactive=0.0,
                error="Invalid SMILES",
            )
        
        features, feat_err = self._build_feature_vector(receptor, canon)
        if features is None:
            if feat_err and feat_err != "receptor_or_features":
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
                    error=feat_err,
                )
            root = _resolve_gpcr_data_root()
            folder = resolve_receptor_folder(receptor, root)
            if folder is None:
                avail = get_available_receptors()
                hint = (
                    f" Available pocket folders (sample): {', '.join(avail[:8])}..."
                    if avail
                    else " Set GPCR_DATA_ROOT to the directory that contains Josh_Receptor_Features."
                )
                err = (
                    f"Receptor '{receptor}' was not found under Josh_Receptor_Features "
                    f"(no matching folder or gene alias).{hint}"
                )
            else:
                err = (
                    f"Receptor '{receptor}' resolved to '{folder}' but pocket feature CSVs are missing."
                )
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
                error=err,
            )

        if (
            self.expected_feature_dim is not None
            and int(features.shape[0]) != int(self.expected_feature_dim)
        ):
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
                error=(
                    f"Descriptor vector length {int(features.shape[0])} does not match "
                    f"trained model expectation ({int(self.expected_feature_dim)}). "
                    "Check manifest.json feature_columns align with training (6633 features for manuscript models)."
                ),
            )

        X = features.reshape(1, -1)
        models = self._models_for_prediction(receptor)

        # Ensemble prediction
        all_probs = []
        for model in models:
            try:
                if hasattr(model, "base_model_probabilities"):
                    stacked = model.base_model_probabilities(X)
                    for i in range(stacked.shape[1]):
                        all_probs.append(stacked[0, i])
                    continue
                # Try predict_proba first (for sklearn models)
                if hasattr(model, 'predict_proba'):
                    raw = model.predict_proba(X)
                    classes = getattr(model, "classes_", np.array([0, 1, 2]))
                    if self.feature_mode == "manuscript":
                        from .manuscript_bundle import _align_proba

                        probs = _align_proba(raw, classes, n_classes=3)[0]
                    else:
                        probs = raw[0]
                # Try predict with probability output (for LightGBM/XGBoost)
                elif hasattr(model, 'predict'):
                    # Some models return probabilities directly
                    probs = model.predict(X, raw_score=False)[0]
                    # Ensure 3 classes
                    if len(probs) != 3:
                        # If binary, convert to 3-class
                        probs = np.array([probs[0], probs[1], 0.0])
                else:
                    continue
                
                # Ensure 3 probabilities
                if len(probs) == 3:
                    all_probs.append(probs)
            except Exception as e:
                continue
        
        if not all_probs:
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
                error="Model prediction failed",
            )
        
        # Average probabilities across ensemble
        mean_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        
        # Standard error of the mean
        std_error = std_probs / np.sqrt(len(all_probs))
        
        prob_agonist = float(mean_probs[0])
        prob_antagonist = float(mean_probs[1])
        prob_inactive = float(mean_probs[2])
        
        # Predicted class (highest probability)
        predicted_class_id = int(np.argmax(mean_probs))
        predicted_class = self.class_names[predicted_class_id]

        result = PredictResult(
            is_valid=True,
            receptor=receptor,
            ligand_smiles=ligand_smiles,
            canonical_smiles=canon,
            predicted_class=predicted_class,
            class_id=predicted_class_id,
            prob_agonist=prob_agonist,
            prob_antagonist=prob_antagonist,
            prob_inactive=prob_inactive,
            prob_std_error=float(std_error[predicted_class_id]),
            prob_std_dev=float(std_probs[predicted_class_id]),
            threshold=self.threshold,
            error="",
        )
        if os.environ.get("GPCR_CLOUD_LITE", "").strip().lower() in ("1", "true", "yes") or (
            str(os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "")).strip().lower() == "cloud"
        ):
            del X, features, all_probs, mean_probs, std_probs
            import gc

            gc.collect()
        return result


def load_predictor(
    artifact_dir: Union[str, Path],
    model_type: Optional[str] = None,
    evaluation_regime: Optional[str] = None,
    seed: int = 42,
) -> GPCRPredictor:
    """
    Load GPCR predictor from artifact directory.

    evaluation_regime (manuscript models):
      - "independent_ligand" — Table 1 / dev80-trained models (Code S9–S11)
      - "scaffold" — scaffold-split trained models (Code S15–S17)
      - "loro" — per-receptor leave-one-out models (Code S18–S20); model picked by receptor at predict time

    Requires artifacts/manuscript/ (exported manuscript models).
    Legacy demo_* bundles (2103 features) are not supported.
    """
    base = Path(artifact_dir)
    project_root = base if (base / "streamlit_app.py").exists() else base

    regime = (evaluation_regime or "").strip().lower() or "independent_ligand"
    if regime in ("independent", "indep"):
        regime = "independent_ligand"

    if regime in ("independent_ligand", "scaffold", "loro"):
        from .manuscript_bundle import load_manuscript_models, manuscript_bundle_available

        if not manuscript_bundle_available(project_root):
            raise FileNotFoundError(
                "Manuscript models not found. Export models into "
                "training repo, then copy artifacts/manuscript/ into this project."
            )
        mt = (model_type or "rf").lower()
        if regime == "loro" and mt == "ensemble":
            raise ValueError("Manuscript stacking ensemble was trained for independent-ligand evaluation only.")
        models, manifest = load_manuscript_models(
            project_root, regime, mt, seed=seed, receptor_folder=None
        )
        nfi = getattr(models[0], "n_features_in_", None)
        return GPCRPredictor(
            models=models,
            class_names=["Agonist", "Antagonist", "Inactive"],
            expected_feature_dim=int(nfi) if nfi is not None else None,
            feature_mode="manuscript",
            project_root=project_root,
            evaluation_regime=regime,
            model_type=mt,
            seed=seed,
        )

    raise FileNotFoundError(
        f"Unknown or missing manuscript evaluation regime: {evaluation_regime!r}. "
        "Use independent_ligand, scaffold, or loro. Export models with "
        "artifacts/manuscript/ from your training workspace."
    )


def predict_single(
    receptor: str,
    ligand_smiles: str,
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[GPCRPredictor] = None,
) -> PredictResult:
    """Predict for a single receptor-ligand pair."""
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    return predictor.predict(receptor, ligand_smiles)


def predict_batch(
    receptor_ligand_pairs: List[tuple],  # List of (receptor, ligand_smiles) tuples
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[GPCRPredictor] = None,
) -> List[PredictResult]:
    """Predict for a list of receptor-ligand pairs."""
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    return [predictor.predict(receptor, ligand) for receptor, ligand in receptor_ligand_pairs]
