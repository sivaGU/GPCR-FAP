"""
Resolve user-facing receptor names (gene symbols, aliases) to Josh_Receptor_Features folder names.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ALIASES_PATH = Path(__file__).resolve().parents[2] / "data" / "receptor_aliases.json"


@lru_cache(maxsize=1)
def _load_gene_to_folder() -> Dict[str, str]:
    """gene symbol (upper) -> pocket folder name."""
    mapping: Dict[str, str] = {}
    if _ALIASES_PATH.exists():
        with open(_ALIASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for gene, folder in (data.get("gene_to_folder") or {}).items():
            mapping[str(gene).strip().upper()] = str(folder).strip()
    return mapping


def _receptor_features_root(data_root: Path) -> Path:
    return data_root / "Josh_Receptor_Features"


def list_receptor_folders(data_root: Path) -> List[str]:
    """Folder names under Josh_Receptor_Features that contain pocket CSVs."""
    root = _receptor_features_root(data_root)
    if not root.is_dir():
        return []
    folders: List[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if list(p.glob("*_pocket_residues_with_conservation.csv")):
            folders.append(p.name)
    return sorted(folders)


def resolve_receptor_folder(receptor_input: str, data_root: Path) -> Optional[str]:
    """
    Map a receptor label to the Josh_Receptor_Features subdirectory name.

    Accepts folder names (beta2), gene symbols (ADRB2), or PDB-style entry names (ADA2A).
    """
    if not receptor_input or not str(receptor_input).strip():
        return None

    raw = str(receptor_input).strip()
    root = _receptor_features_root(data_root)
    if not root.is_dir():
        return None

    # Exact folder match (case-sensitive first, then case-insensitive)
    direct = root / raw
    if direct.is_dir() and list(direct.glob("*_pocket_residues_with_conservation.csv")):
        return raw

    lower_map = {p.name.lower(): p.name for p in root.iterdir() if p.is_dir()}
    hit = lower_map.get(raw.lower())
    if hit and list((root / hit).glob("*_pocket_residues_with_conservation.csv")):
        return hit

    # Gene / alias table
    alias = _load_gene_to_folder().get(raw.upper())
    if alias:
        folder = root / alias
        if folder.is_dir() and list(folder.glob("*_pocket_residues_with_conservation.csv")):
            return alias

    return None


def receptor_display_options(data_root: Path) -> List[Tuple[str, str]]:
    """
    (folder_name, display_label) for UI selectboxes.
    Labels include a gene alias when known.
    """
    gene_to_folder = _load_gene_to_folder()
    folder_to_genes: Dict[str, List[str]] = {}
    for gene, folder in gene_to_folder.items():
        folder_to_genes.setdefault(folder, []).append(gene)

    options: List[Tuple[str, str]] = []
    for folder in list_receptor_folders(data_root):
        genes = sorted(folder_to_genes.get(folder, []))
        # Prefer standard HGNC-style symbols in the label
        preferred = next(
            (g for g in genes if g.startswith(("ADR", "DRD", "HTR", "CHR", "CNR", "HRH", "S1PR", "FFAR", "P2RY"))),
            genes[0] if genes else None,
        )
        if preferred:
            label = f"{folder} ({preferred})"
        else:
            label = folder
        options.append((folder, label))
    return options
