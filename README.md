# Glioma-Classification

<p align="left">
  <a href="https://github.com/PhDSandeepSingh/Glioma-Classification/releases/tag/v3.1.0"><img src="https://img.shields.io/badge/Release-v3.1.0-blue.svg?style=flat-square" alt="Release v3.1.0" /></a>
  <a href="https://github.com/PhDSandeepSingh/Glioma-Classification/tags"><img src="https://img.shields.io/badge/Tag-v3.1.0-0366d6.svg?style=flat-square" alt="Tag v3.1.0" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.13-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.13" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-green.svg?style=flat-square" alt="Code License: MIT" /></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg?style=flat-square" alt="Data License: CC BY 4.0" /></a>
  <a href="Data/External/README.md"><img src="https://img.shields.io/badge/External%20Data-TCGA%20%2F%20NCI%20GDC-orange.svg?style=flat-square" alt="External Data: TCGA / NCI GDC" /></a>
  <a href="environment/requirements-lock.txt"><img src="https://img.shields.io/badge/Pipeline-Deterministic%20%26%20Pinned-brightgreen.svg?style=flat-square" alt="Pipeline: Deterministic & Pinned" /></a>
  <a href="#license--terms-of-use"><img src="https://img.shields.io/badge/Scope-Research%20Use%20Only-red.svg?style=flat-square" alt="Scope: Research Use Only" /></a>
</p>

**Tags:** `machine-learning` · `glioma` · `tcga` · `somatic-mutations` · `nested-cross-validation` · `benchmarking` · `reproducible-research` · `xgboost` · `scikit-learn`

A reproducible, deterministic machine-learning pipeline and data-provenance framework for computational benchmarking of somatic-mutation-based glioma classification on TCGA/UCI cohorts.

### License & Terms of Use

| Component | License / Governance | Reference |
|---|---|---|
| **Pipeline, Notebook & Utility Code** | **MIT License** | [`LICENSE`](LICENSE) |
| **Base Cohort & Derived Label Release** | **CC BY 4.0** (Attribution 4.0 International) | [`Data/DATASHEET.md`](Data/DATASHEET.md) |
| **External Molecular Annotations** | **TCGA Open-Access / NCI GDC Data-Use Policies** | [`Data/External/README.md`](Data/External/README.md) |

> **Research Use Only:** All code, models, and derived datasets in this repository are provided strictly for computational research and methodological benchmarking. They are not intended for clinical diagnosis, prognosis, or patient management.

---

## Overview

This repository packages a self-contained, end-to-end computational workflow designed for strict reproducibility and leakage-free validation. Rather than relying on ad-hoc scripts, the entire pipeline is generated from structured Python source, executed headlessly under pinned dependencies, and verified against cryptographic hashes and pre-recorded model locks.

### Key Capabilities
- **Programmatic Notebook Generation:** `build_notebook.py` compiles the full analysis pipeline into `Glioma_Grading_R3.ipynb` from clean, version-controlled Python source.
- **Deterministic Headless Execution:** `run_nb.py` executes the full workflow end-to-end without interactive state, saving all outputs, figures, and summary tables.
- **Leakage-Free Nested Validation:** Implements nested and repeated nested cross-validation alongside inner-loop joint selection across multiple model families and imbalance-handling strategies.
- **Automated Provenance & Verification:** Tracks input checksums (SHA-256), pinned external commits, pre-test evaluation locks, and machine-readable reproducibility manifests.
- **Publication-Grade Visuals:** Includes programmatic scripts for high-resolution (600 DPI) figure rendering.

---

## Repository Structure

```text
Glioma-Classification/
├── Glioma_Grading_R3.ipynb        # Executed canonical notebook with saved outputs
├── build_notebook.py              # Source script that generates the Jupyter notebook
├── run_nb.py                      # Headless notebook runner
├── requirements.txt               # Direct Python dependencies
├── environment/
│   ├── requirements-lock.txt      # Full pinned pip dependency closure
│   └── environment.yml            # Conda environment specification
├── Data/
│   ├── DATASHEET.md               # Dataset documentation (Datasheets for Datasets)
│   ├── External/                  # Pinned upstream reference annotations & README
│   └── Release/                   # Versioned patient-level label release & schema
├── Results/
│   ├── Figures/                   # Generated analytical figures
│   ├── Models/                    # Serialized artifacts, locks & reproducibility report
│   └── Tables/                    # Structured CSV/JSON evaluation tables
├── tools/
│   ├── extract_external.py        # Deterministic subsetting for external reference files
│   ├── run_fully_nested_joint.py  # Nested inner-loop joint selection runner
│   └── generate_figure1_600dpi.py # Programmatic 600-DPI figure generator
├── CITATION.cff                   # Citation File Format metadata
├── .zenodo.json                   # Zenodo archive metadata
├── LICENSE                        # MIT License
└── VERSION                        # Semantic version tag (3.1.0)
```

---

## Getting Started

### Prerequisites
- **Python 3.13.x** (tested on CPython 3.13.16, Linux x86\_64 / macOS)
- `pip` or `conda` / `mamba`

> **Note:** `pip` cannot enforce the Python interpreter version automatically. Ensure your virtual environment is initialized with Python 3.13.

### Installation

#### Option A: Virtual Environment (`venv` + `pip`)
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r environment/requirements-lock.txt
```
*(For a looser, top-level install, use `pip install -r requirements.txt`.)*

#### Option B: Conda
```bash
conda env create -f environment/environment.yml
conda activate glioma-r3
```

---

## Reproducing the Pipeline

### 1. Standard Execution
To regenerate the notebook from source (optional) and execute the entire pipeline headlessly:
```bash
python build_notebook.py
python run_nb.py
```

### 2. Clean From-Scratch Run
Cached intermediate stages are keyed on input data hashes, run tags, and code hashes. To force a complete rebuild of all models, caches, tables, and figures:
```bash
rm -rf Results/
python build_notebook.py
python run_nb.py
```
*Typical runtime on a 2-core x86\_64 environment is ~18–21 minutes from scratch, or ~5 minutes when reusing valid caches.*

### 3. Standalone Utilities
- **Nested Joint Selection:**
  ```bash
  python tools/run_fully_nested_joint.py
  ```
- **External Reference Subsetting:**
  ```bash
  python3 -I tools/extract_external.py <download_dir> Data/External
  ```
- **High-Resolution Figure Rendering:**
  ```bash
  python tools/generate_figure1_600dpi.py
  ```

### 4. Verification Checks
After execution, you can verify deterministic reproduction by checking:
- `Results/Models/reproducibility_report.json` — environment, package versions, and input hashes.
- `Results/Models/locks_before_test.json` — pre-evaluation model and panel locks.
- `Results/ManuscriptNumbers.csv` — consolidated summary metrics table.

*(Serialized model pickles are tied to the pinned Python, `scikit-learn`, and `xgboost` versions in `environment/requirements-lock.txt` and should not be loaded under mismatched environments.)*

---

## Data Provenance

| Resource | Source / Identifier | License / Terms |
|---|---|---|
| Base Cohort Features | UCI Machine Learning Repository (Dataset 759, `doi:10.24432/C5R62J`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| External Molecular Annotations | cBioPortal Datahub (`lgggbm_tcga_pub`, pinned commit) | TCGA Open-Access / NCI GDC Policies |
| Derived Label Release | `Data/Release/` (documented in `Data/DATASHEET.md`) | CC BY 4.0 |
| Pipeline & Utility Code | This repository | [MIT License](LICENSE) |

SHA-256 checksums and pinned upstream commit hashes for external reference files are recorded in `Data/External/README.md`.

---

## Citation

If you use this codebase or data-processing pipeline in your work, please cite the software metadata in [`CITATION.cff`](CITATION.cff) and acknowledge the underlying UCI and TCGA/cBioPortal data sources.

---

## Contact

**Sandeep Singh**  
Department of Mathematics, Chandigarh University, Mohali, Punjab, India  
GitHub: [@PhDSandeepSingh](https://github.com/PhDSandeepSingh)
