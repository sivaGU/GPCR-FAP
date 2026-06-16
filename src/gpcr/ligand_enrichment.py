"""
Training-aligned ligand descriptors for manuscript inference.

Matches Code S3 (RDKit 2D) and Code S4 (Mordred, ignore_3D=True) used in enriched CSVs.
"""
from __future__ import annotations

import importlib
from typing import Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

_MORDRED_CALC = None


def _apply_numpy_mordred_patch() -> None:
    """Restore numpy symbols removed in NumPy 2.x before importing mordred."""
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]
    try:
        import numpy.core.records as _rec

        if not hasattr(np, "rec"):
            np.rec = _rec  # type: ignore[attr-defined]
    except ImportError:
        pass
    try:
        _ng = importlib.import_module("numpy._globals")
        if not hasattr(_ng, "VisibleDeprecationWarning"):
            setattr(_ng, "VisibleDeprecationWarning", Warning)
    except ModuleNotFoundError:
        pass


def compute_rdkit_2d_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Same descriptor set as Code S3 - add_rdkit_2d_descriptors.py."""
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "ExactMolWt": float(Descriptors.ExactMolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "HBD": float(Descriptors.NumHDonors(mol)),
        "HBA": float(Descriptors.NumHAcceptors(mol)),
        "RotatableBonds": float(Descriptors.NumRotatableBonds(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "FormalCharge": float(Chem.GetFormalCharge(mol)),
        "HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        "AromaticRings": float(Descriptors.NumAromaticRings(mol)),
        "AliphaticRings": float(Descriptors.NumAliphaticRings(mol)),
        "TotalRings": float(Descriptors.RingCount(mol)),
        "LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
        "Chi0v": float(Descriptors.Chi0v(mol)),
        "Chi1v": float(Descriptors.Chi1v(mol)),
        "Chi2v": float(Descriptors.Chi2v(mol)),
        "Chi0n": float(Descriptors.Chi0n(mol)),
        "Chi1n": float(Descriptors.Chi1n(mol)),
        "Chi2n": float(Descriptors.Chi2n(mol)),
        "Kappa1": float(Descriptors.Kappa1(mol)),
        "Kappa2": float(Descriptors.Kappa2(mol)),
        "Kappa3": float(Descriptors.Kappa3(mol)),
        "HallKierAlpha": float(Descriptors.HallKierAlpha(mol)),
        "Ipc": float(Descriptors.Ipc(mol)),
        "BertzCT": float(Descriptors.BertzCT(mol)),
        "BalabanJ": float(Descriptors.BalabanJ(mol)),
        "NumAtoms": float(mol.GetNumAtoms()),
        "NumBonds": float(mol.GetNumBonds()),
        "NumHeteroatoms": float(Descriptors.NumHeteroatoms(mol)),
        "NumSaturatedRings": float(Descriptors.NumSaturatedRings(mol)),
        "NumAromaticHeterocycles": float(Descriptors.NumAromaticHeterocycles(mol)),
        "NumSaturatedHeterocycles": float(Descriptors.NumSaturatedHeterocycles(mol)),
        "NumAliphaticHeterocycles": float(Descriptors.NumAliphaticHeterocycles(mol)),
        "NumAliphaticCarbocycles": float(Descriptors.NumAliphaticCarbocycles(mol)),
        "NumAromaticCarbocycles": float(Descriptors.NumAromaticCarbocycles(mol)),
        "EState_VSA1": float(Descriptors.EState_VSA1(mol)),
        "EState_VSA2": float(Descriptors.EState_VSA2(mol)),
        "EState_VSA3": float(Descriptors.EState_VSA3(mol)),
        "EState_VSA4": float(Descriptors.EState_VSA4(mol)),
        "EState_VSA5": float(Descriptors.EState_VSA5(mol)),
        "EState_VSA6": float(Descriptors.EState_VSA6(mol)),
        "EState_VSA7": float(Descriptors.EState_VSA7(mol)),
        "EState_VSA8": float(Descriptors.EState_VSA8(mol)),
        "EState_VSA9": float(Descriptors.EState_VSA9(mol)),
        "EState_VSA10": float(Descriptors.EState_VSA10(mol)),
        "EState_VSA11": float(Descriptors.EState_VSA11(mol)),
        "VSA_EState1": float(Descriptors.VSA_EState1(mol)),
        "VSA_EState2": float(Descriptors.VSA_EState2(mol)),
        "VSA_EState3": float(Descriptors.VSA_EState3(mol)),
        "VSA_EState4": float(Descriptors.VSA_EState4(mol)),
        "VSA_EState5": float(Descriptors.VSA_EState5(mol)),
        "VSA_EState6": float(Descriptors.VSA_EState6(mol)),
        "VSA_EState7": float(Descriptors.VSA_EState7(mol)),
        "VSA_EState8": float(Descriptors.VSA_EState8(mol)),
        "VSA_EState9": float(Descriptors.VSA_EState9(mol)),
        "VSA_EState10": float(Descriptors.VSA_EState10(mol)),
    }


def _mordred_value_to_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return 0.0
        return float(value)
    try:
        f = float(value)
        return 0.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _get_mordred_calculator():
    global _MORDRED_CALC
    if _MORDRED_CALC is not None:
        return _MORDRED_CALC
    _apply_numpy_mordred_patch()
    from mordred import Calculator, descriptors as md_descriptors

    _MORDRED_CALC = Calculator(md_descriptors, ignore_3D=True)
    return _MORDRED_CALC


def compute_mordred_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Full Mordred 2D panel (column names match enriched CSVs / manifest)."""
    try:
        calc = _get_mordred_calculator()
    except ImportError:
        return {}
    raw = calc(mol)
    return {str(k): _mordred_value_to_float(v) for k, v in raw.items()}


def build_ligand_descriptor_dict(
    smiles: str,
    *,
    mol: Optional[Chem.Mol] = None,
    include_mordred: bool = True,
) -> Optional[Dict[str, float]]:
    """RDKit + optional Mordred dict for one canonical SMILES."""
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    out = compute_rdkit_2d_descriptors(mol)
    if include_mordred:
        out.update(compute_mordred_descriptors(mol))
    return out
