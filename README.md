# GPCR-FAP GUI

Streamlit app for **Class A GPCR** receptor–ligand **multiclass** functional activity prediction (Agonist / Antagonist / Inactive), with optional **SMINA** docking and **py3Dmol** visualization tool.

- **Live app:** [https://gpcr-fap-final.streamlit.app](https://gpcrfap.streamlit.app/)
- **Repository:** https://github.com/sivaGU/GPCR-FAP

## Features

- Predict from SMILES or structure files (SDF, MOL, PDB, PDBQT, MOL2, CSV) for 68 bundled receptors.
- Manuscript models: RF, LightGBM, XGBoost (`artifacts/manuscript/`, **6,633** features).
- Optional SMINA top-pose docking and 3D complex view after prediction.

## Quick start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501 → choose **evaluation regime**, **model**, and **seed** → select receptor → enter SMILES → **Predict**.

## Manuscript models

Deploy under `artifacts/manuscript/`:

- `manifest.json`
- `ligand_feature_lookup.sqlite` (Git LFS)
- `independent_ligand/<model>/model_seed42.pkl` (Cloud RF: `model_seed42_cloud.pkl`)

Optional: set `GPCR_DATA_ROOT` if `Josh_Receptor_Features/` is outside the repo.

## Requirements

- Python 3.10+ (3.10–3.12 tested)
- `requirements.txt` (used by Streamlit Cloud and local installs)
- Receptor data: `Josh_Receptor_Features/` in repo root, or `GPCR_DATA_ROOT`
- Docking on Cloud: `docking_assets/smina_linux/` and `docking_assets/receptor_pdbs/`

## Project layout

```
├── streamlit_app.py
├── requirements.txt
├── artifacts/manuscript/
├── Josh_Receptor_Features/
├── docking_assets/
└── src/gpcr/
```

## Citation & contact

Cite the GPCR Class A functional activity prediction manuscript / Zenodo DOI when publishing.

**Dr. Sivanesan Dakshanamurthy** — sd233@georgetown.edu
