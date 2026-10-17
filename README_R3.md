# Glioma-Classification (v3.0.0)

[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SandeepSinghNk/Glioma-Classification)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB.svg)](environment/requirements-lock.txt)

Official analysis code, reproducible notebook, locked evaluation artifacts, and versioned WHO 2021 patient-level label release (`v1.0`) for:

> **Diagnostic label–IDH discordance and machine-learning classifier performance on a legacy TCGA glioma grading benchmark**  
> Sandeep Singh, Ashok Pal, Nitin Mittal, Jatinder Kaur, and Fikreselam Gared  
> *Scientific Reports* (2026)

**Research use only.** Neither the trained classifiers nor the reconstructed WHO 2021 annotations are intended for clinical diagnosis or patient management.

---

## Key Findings

1. **Label–IDH Discordance (13.1%):** In the complete-case UCI/TCGA Glioma Grading benchmark ($n = 839$: 487 legacy LGG, 352 legacy GBM), 110 legacy WHO 2007 histological labels disagree with *IDH1*/*IDH2* mutation status (85 IDH-wildtype LGG and 25 IDH-mutant GBM). Consequently, the zero-parameter rule *"IDH1/2-wildtype $\Rightarrow$ GBM"* achieves **86.9% accuracy**.
2. **Model-Free WHO 2021 Reclassification:** Linking all 839 patients by TCGA case ID to published molecular annotations ([Ceccarelli et al., *Cell* 2016](https://doi.org/10.1016/j.cell.2015.12.028); [de Mendonça et al., *Nat. Commun.* 2025](https://doi.org/10.1038/s41467-025-59188-3)) shows that **68 of 85 (80.0%; range 51.8–84.0% across sensitivity variants)** IDH-wildtype legacy LGG meet WHO 2021 molecular criteria for *glioblastoma, IDH-wildtype*, whereas **23 of 23** assessable non-codeleted IDH-mutant legacy GBM are *astrocytoma, IDH-mutant, CNS WHO grade 4* (21 of 25 graded on histology alone without *CDKN2A/B* deletion).
3. **Classifier Performance vs. Simple Rules:** Under leakage-free $5\times 5$ nested cross-validation ($n = 671$) and locked single-run evaluation on the held-out test set ($n = 168$), the locked 23-feature XGBoost model reaches a test ROC-AUC of **0.932** (95% CI 0.893–0.964) but makes **25 errors** (accuracy 0.851) at its locked Platt-calibrated Youden threshold ($t = 0.351$), compared with **22 errors** (accuracy 0.869) for the zero-parameter *IDH1/2* rule (exact McNemar $p = 0.25$; Newcombe paired score 95% CI $-0.047$ to $+0.009$).
4. **Error Concentration:** Nearly all model errors (**84 of 86** out-of-fold; **22 of 25** test) occur in label–IDH discordant patients. Adding WHO 2021-defining copy-number (*EGFR* amplification, *CDKN2A/B* deletion, $+7/{-}10$) and 1p/19q co-deletion features does not materially raise legacy-label accuracy.

---

## Repository Structure

```text
Glioma-Classification/
├── README.md                                  # Project overview and reproduction guide
├── LICENSE                                    # MIT License (code) & CC BY 4.0 / TCGA terms (labels)
├── VERSION                                    # Release version (3.0.0)
├── CITATION.cff                               # Citation metadata (v3.0.0)
├── .zenodo.json                               # Zenodo archive metadata (v3.0.0)
├── Glioma_Grading_R3.ipynb                    # Executed canonical Jupyter notebook (all outputs saved)
├── build_notebook.py                          # Deterministic Python source that builds Glioma_Grading_R3.ipynb
├── run_nb.py                                  # Headless notebook execution runner
├── requirements.txt                           # Direct pinned Python dependencies
├── environment/
│   ├── requirements-lock.txt                  # Full pinned dependency lockfile (Python 3.13)
│   └── environment.yml                        # Conda environment specification
├── Data/
│   ├── TCGA_InfoWithGrade.csv                 # UCI analysis dataset (839 x 24, DOI: 10.24432/C5R62J)
│   ├── TCGA_GBM_LGG_Mutations_all.csv         # Raw TCGA export (862 records with TCGA Case_IDs)
│   ├── DATASHEET.md                           # Datasheet for the WHO 2021 label release (Gebru et al.)
│   ├── Discordant_Cases_Audit_WHO2021.csv     # Patient-level audit of the 110 label-IDH discordant cases
│   ├── WHO2021_Reconstruction_PatientLevel.csv# Reconstructed categories across sensitivity variants
│   ├── Release/
│   │   ├── TCGA_GliomaBenchmark_WHO2021_Labels_v1.0.csv        # Versioned label release (839 x 53)
│   │   └── TCGA_GliomaBenchmark_WHO2021_Labels_v1.0_Columns.csv# Column dictionary
│   └── External/                              # Pinned cBioPortal lgggbm_tcga_pub & MONET annotations + SHA-256
├── Results/
│   ├── ManuscriptNumbers.csv                  # Single source of truth for all 703 numbers in the manuscript
│   ├── AssetIndex.csv                         # Index of all 81 generated tables and figures
│   ├── Figures/                               # Publication-ready figures (300 DPI PNG)
│   ├── Tables/                                # All canonical result tables (CSV)
│   └── Models/                                # Locked pre-test decisions, predictions, and caches
└── tools/
    ├── extract_external.py                    # Reproducible extractor for upstream cBioPortal files
    └── fetch_demendonca.py                    # Verification script for de Mendonça et al. (2025) MONET labels
```

---

## Reproducing the Canonical Run

### 1. Environment Setup
The canonical pipeline was executed under **Python 3.13** with deterministic seeds and pinned dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements-lock.txt
# Or install direct dependencies:
# pip install -r requirements.txt
```

Alternatively, with Conda:
```bash
conda env create -f environment/environment.yml
conda activate glioma-r3
```

### 2. Regenerate and Execute the Notebook
```bash
# Optional: regenerate Glioma_Grading_R3.ipynb from build_notebook.py
python build_notebook.py

# Execute the full notebook headlessly and write all outputs to Results/
python run_nb.py
```

### 3. Verify Checksums
```bash
shasum -a 256 Data/Release/TCGA_GliomaBenchmark_WHO2021_Labels_v1.0.csv
# Expected: d7c9e905dbe11f94ca396394766d46088854f223e99f244af508f7ff35c9afbd
```

---

## License & Citation

- **Code:** Released under the [MIT License](LICENSE).
- **WHO 2021 Label Release (`Data/Release/`):** Released under CC BY 4.0, subject to the upstream terms of the [UCI Machine Learning Repository dataset (CC BY 4.0)](https://doi.org/10.24432/C5R62J), [NCI GDC / TCGA](https://gdc.cancer.gov/access-data/data-access-policies), and Broad GDAC Firehose.
- **Citation:** Please see [`CITATION.cff`](CITATION.cff) for software and article citation metadata.
