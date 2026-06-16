"""
GPCR receptor + query-ligand 3D viewer (py3Dmol), styled after MBind's complex viewer.

Not AutoDock/Vina docking: the query ligand is embedded in 3D with RDKit and translated
so its heavy-atom centroid matches the co-crystal binding-site center (from *_ligand_only.pdb
when present). The native ligand is not drawn—only receptor cartoon + user ligand sticks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import py3Dmol
except ImportError:
    py3Dmol = None

# MBind-aligned palette: tan cartoon receptor, greenCarbon query sticks
RECEPTOR_CARTOON_HEX = "d2b48c"

# Keep only protein ATOMs in the receptor view (drops co-crystal ligand if mis-tagged as ATOM).
_STANDARD_PROTEIN_RESN = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "CYX", "GLN", "GLU", "GLY", "HIS", "HIE", "HID", "HIP",
        "ILE", "LEU", "LYS", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
)

# Default clip: keep receptor atoms within this distance (Å) of the binding-site center.
DEFAULT_BINDING_SITE_VIEW_RADIUS_A = 75.0
MIN_ATOM_LINES_AFTER_CLIP = 80  # fall back to full receptor if clip removes too much


def _resolve_data_root() -> Path:
    env_root = os.environ.get("GPCR_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    gpcr_main = Path(__file__).resolve().parents[2]
    if (gpcr_main / "Josh_Receptor_Features").is_dir():
        return gpcr_main
    sibling = gpcr_main.parent / "GUI_Folder"
    if (sibling / "Josh_Receptor_Features").is_dir():
        return sibling
    return gpcr_main.parent / "GPCRtryagain - Delete - Copy"


def resolve_receptor_structure_paths(receptor_folder: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (receptor_only_pdb, ligand_only_pdb) under Josh_Receptor_Features/<folder>/.

    Pairs by **matching PDB ID stem** so `<id>_ligand_only.pdb` always pairs with
    `<id>_receptor_only.pdb` (never the full complex *.pdb, never a mismatched ID).
    """
    from .receptor_names import resolve_receptor_folder

    root = _resolve_data_root()
    folder = resolve_receptor_folder(receptor_folder, root) or str(receptor_folder).strip()
    pocket = root / "Josh_Receptor_Features" / folder
    if not pocket.is_dir():
        return None, None

    lig_suffix, rec_suffix = "_ligand_only.pdb", "_receptor_only.pdb"
    ligand_files = sorted(pocket.glob(f"*{lig_suffix}"))
    receptor_files = sorted(pocket.glob(f"*{rec_suffix}"))
    if not ligand_files or not receptor_files:
        return None, None

    lig_stems = {p.name[: -len(lig_suffix)] for p in ligand_files}
    rec_stems = {p.name[: -len(rec_suffix)] for p in receptor_files}
    common = sorted(lig_stems & rec_stems)
    if not common:
        return None, None
    stem = common[0]
    rec = pocket / f"{stem}{rec_suffix}"
    ref_lig = pocket / f"{stem}{lig_suffix}"
    if not rec.is_file() or not ref_lig.is_file():
        return None, None
    return rec, ref_lig


def _sanitize_receptor_pdb_for_view(pdb_text: str) -> str:
    """
    Drop HETATM and any ATOM record that is not a standard protein residue (removes waters,
    co-crystal ligands, and ions that would otherwise show as extra sticks with the cartoon).
    """
    lines_out: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM"):
            continue
        if line.startswith("ATOM") and len(line) > 20:
            resn = line[17:20].strip().upper()
            if resn not in _STANDARD_PROTEIN_RESN:
                continue
        lines_out.append(line)
    return "\n".join(lines_out)


def _clip_receptor_pdb_near_site(
    sanitized_pdb_text: str,
    site_center: np.ndarray,
    radius_angstrom: float,
) -> str:
    """
    Keep only ATOM records within radius_angstrom (Å) of site_center (binding site / query ligand centroid).
    Preserves CRYST1 if present; appends END.
    """
    r = float(radius_angstrom)
    if r <= 0:
        return sanitized_pdb_text
    center = np.asarray(site_center, dtype=np.float64).reshape(3)
    header_lines: list[str] = []
    atom_lines: list[str] = []
    for line in sanitized_pdb_text.splitlines():
        if line.startswith("CRYST1"):
            header_lines.append(line)
            continue
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        pos = np.array([x, y, z], dtype=np.float64)
        if float(np.linalg.norm(pos - center)) <= r:
            atom_lines.append(line)
    if len(atom_lines) < MIN_ATOM_LINES_AFTER_CLIP:
        return sanitized_pdb_text
    parts = header_lines + atom_lines + ["END"]
    return "\n".join(parts)


def _pdb_line_element_symbol(line: str) -> str:
    """Element symbol from PDB v3 cols 77–78, with fallbacks from atom name (cols 13–16)."""
    elem = ""
    if len(line) >= 78:
        elem = line[76:78].strip().upper()
    if elem:
        return elem
    if len(line) < 16:
        return ""
    name = line[12:16].strip().upper()
    if not name:
        return ""
    two = name[:2]
    if two in ("BR", "CL", "FE", "ZN", "MG", "CA", "NA", "MN", "CU", "CO", "NI", "SE"):
        return two
    if name[0].isdigit() and len(name) > 1 and name[1].isalpha():
        return name[1:2]
    return name[0]


def _ligand_only_pdb_heavy_atom_centroid(pdb_text: str) -> Optional[np.ndarray]:
    """
    Centroid (mean x,y,z) of heavy atoms in a ligand-only PDB (ATOM/HETATM records).
    Coordinates use standard PDB columns 31–54 (0-based slices 30:54).
    """
    coords: list[list[float]] = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        elem = _pdb_line_element_symbol(line)
        if elem in ("H", "D"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        coords.append([x, y, z])
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64).mean(axis=0)


def _pdb_heavy_atom_com(pdb_text: str) -> Optional[np.ndarray]:
    """Heavy-atom centroid for mixed PDB text (e.g. legacy callers)."""
    return _ligand_only_pdb_heavy_atom_centroid(pdb_text)


def smiles_to_pdb_block_aligned(
    smiles: str,
    target_com: np.ndarray,
    random_seed: int = 42,
) -> Optional[str]:
    """ETKDG 3D structure for SMILES, translated so ligand COM matches target_com."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    embed_code = -1
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        embed_code = AllChem.EmbedMolecule(mol, params)
    except AttributeError:
        embed_code = AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    if embed_code != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=random_seed) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(i)
        positions.append([p.x, p.y, p.z])
    if not positions:
        return None
    lig_com = np.mean(positions, axis=0)
    delta = target_com - lig_com
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(
            i,
            (p.x + float(delta[0]), p.y + float(delta[1]), p.z + float(delta[2])),
        )
    mol_h = Chem.RemoveHs(mol)
    return Chem.MolToPDBBlock(mol_h, confId=0)


def build_gpcr_complex_view_html(
    receptor_pdb_text: str,
    query_ligand_pdb_text: str,
    width: int = 720,
    height: int = 520,
) -> Optional[str]:
    """MBind-style py3Dmol HTML: receptor cartoon + query ligand sticks only (no co-crystal ligand)."""
    if py3Dmol is None:
        return None
    if not receptor_pdb_text.strip() or not query_ligand_pdb_text.strip():
        return None
    try:
        view = py3Dmol.view(width=width, height=height)
        view.addModel(receptor_pdb_text, "pdb")
        view.setStyle({"model": 0}, {"cartoon": {"color": f"0x{RECEPTOR_CARTOON_HEX}"}})
        view.addModel(query_ligand_pdb_text, "pdb")
        view.setStyle({"model": 1}, {"stick": {"radius": 0.13, "colorscheme": "greenCarbon"}})
        view.zoomTo()
        return view._make_html()
    except Exception:
        return None


def build_aligned_complex_html_for_receptor(
    receptor_folder: str,
    canonical_smiles: str,
    width: int = 720,
    height: int = 520,
    binding_site_radius_angstrom: float = DEFAULT_BINDING_SITE_VIEW_RADIUS_A,
) -> Tuple[Optional[str], str]:
    """
    Load PDB assets for receptor_folder, align query ligand to the ligand-only PDB centroid, return (html, status_message).

    binding_site_radius_angstrom: receptor cartoon is clipped to protein atoms within this distance (Å)
    of the orthosteric site center so the view stays localized (typical UI range 50–100 Å).
    """
    rec_path, ref_lig_path = resolve_receptor_structure_paths(receptor_folder)
    if rec_path is None or not rec_path.is_file():
        return None, "No receptor PDB found for this target (expected <id>_receptor_only.pdb paired with ligand data)."
    if ref_lig_path is None or not ref_lig_path.is_file():
        return None, "No ligand-only PDB found for this target (expected <id>_ligand_only.pdb matching the receptor PDB id)."
    ref_text = ref_lig_path.read_text(encoding="utf-8", errors="ignore")
    if not ref_text.strip():
        return None, "Ligand-only PDB is empty; cannot place the query ligand."
    target_com = _ligand_only_pdb_heavy_atom_centroid(ref_text)
    if target_com is None:
        return None, "Could not compute a centroid from the ligand-only PDB (no parseable heavy-atom coordinates)."
    rec_text = _sanitize_receptor_pdb_for_view(
        rec_path.read_text(encoding="utf-8", errors="ignore")
    )
    query_pdb = smiles_to_pdb_block_aligned(canonical_smiles, target_com)
    if not query_pdb:
        return None, "RDKit could not build a 3D conformer for this SMILES."
    radius = float(binding_site_radius_angstrom)
    radius = max(50.0, min(100.0, radius))
    rec_for_view = _clip_receptor_pdb_near_site(rec_text, target_com, radius)
    # Reference ligand PDB is used only for binding-site centroid; not rendered in the viewer.
    html = build_gpcr_complex_view_html(rec_for_view, query_pdb, width=width, height=height)
    if html is None:
        if py3Dmol is None:
            return None, "Install py3Dmol for the 3D viewer: pip install py3Dmol"
        return None, "3D viewer failed to render (check PDB/SMILES)."
    return html, "ok"


def py3dmol_available() -> bool:
    return py3Dmol is not None
