# Glioma-Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Journal: Scientific Reports](https://img.shields.io/badge/Journal-Scientific%20Reports-green.svg)](https://www.nature.com/srep/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SandeepSinghNk/Glioma-Classification/blob/main/Glioma_Classification_Pipeline.ipynb)
[![DOI](https://img.shields.io/badge/DOI-10.24432%2FC5R62J-lightgrey.svg)](https://doi.org/10.24432/C5R62J)

**Official repository for:**  
> **"Machine learning triage for glioma histological grading using a minimal mutation and age panel"**  
> *Sandeep Singh, Ashok Pal, Nitin Mittal, Jatinder Kaur, and Fikreselam Gared.*  
> *Under Review, Scientific Reports (2026).*

---

## 📌 Overview

This repository provides an open-source, fully reproducible machine-learning triage framework to discriminate histological Lower-Grade Glioma (LGG) from Glioblastoma (GBM) using clinical variables and somatic mutation markers from the UCI Glioma Grading dataset (derived from TCGA).

While numerous published studies claim classification accuracies of 96% to 98.6% on this benchmark, our critical audit reveals that these inflated figures were driven by pervasive data leakage (e.g., pre-split SMOTE/scaling) and arbitrary 0.5 deployment thresholds. This repository establishes a **leakage-free, calibrated, and reproducible benchmark** evaluated under nested cross-validation and a sealed test set touched exactly once.

---

## 🔬 Key Scientific Findings

1. **The "Parsimony Discovery" — Minimal Panel Sufficiency:**
   * Patient **Age** alone reaches AUC = **0.808**.
   * **IDH1** mutation status alone reaches AUC = **0.856**.
   * Combining just **Age + IDH1** achieves AUC = **0.899**.
   * Sequencing the full 20-gene mutation panel yields only a modest incremental benefit ($\Delta\text{AUC} = +0.020, p = 0.049$).
   * A minimal 4-feature panel (**IDH1, Age, TP53, PTEN**) retains **99.2%** of full-model discriminative capacity (test AUC = **0.908**).

2. **Rigorous Sealed Test-Set Performance ($N = 168$, touched once):**
   * **AUC:** **0.931** (95% CI: 0.893–0.963)
   * **Sensitivity:** **0.871** (61 / 70)
   * **Specificity:** **0.837** (82 / 98)
   * **Accuracy:** **0.851** (143 / 168)
   * **Brier Score:** **0.101** (vs. 0.243 no-skill baseline, 58% error reduction)
   * **Operating Threshold:** $\tau = 0.487$ (locked via Youden's $J$ on out-of-fold training data)

3. **WHO 2021 Neuropathological Discordance Audit:**
   * Analysis of the raw TCGA cohort identifies **129 discordant cases** (106 IDH1-wt LGGs and 23 IDH1-mut GBMs) where historical histological labels conflict with modern molecular taxonomy.
   * Full patient-level mapping is provided in [`Data/Discordant_Cases_Audit_WHO2021.csv`](Data/Discordant_Cases_Audit_WHO2021.csv).

---

## 📁 Repository Structure

```
Glioma-Classification/
├── Data/
│   ├── TCGA_InfoWithGrade.csv              # Primary analysis dataset (N=839, UCI ID 759)
│   ├── TCGA_GBM_LGG_Mutations_all.csv      # Raw TCGA provenance export (N=862, with Case IDs)
│   ├── Discordant_Cases_Audit_WHO2021.csv  # Empirical audit of 129 WHO 2021 discordant cases
│   └── README.md                           # Data dictionary & provenance documentation
├── Glioma_Classification_Pipeline.ipynb     # Canonical Jupyter notebook (Colab & local ready)
├── Glioma_Grading_R2.ipynb                 # R2 execution notebook mirror
├── Results/
│   ├── Figures/                            # High-resolution (600 DPI) publication figures
│   │   ├── Figure01_ClassDistribution.png
│   │   ├── Figure02_MutationHeatmap.png
│   │   ├── Figure03_AgeDistribution.png
│   │   ├── Figure04_CorrelationMatrix.png
│   │   ├── Figure05_BaselineComparison.png
│   │   ├── Figure06_ModelFamilyComparison.png
│   │   ├── Figure07_CalibrationCurve.png
│   │   ├── Figure08_ThresholdSensitivity.png
│   │   ├── Figure09_DecisionCurveAnalysis.png
│   │   ├── Figure10_SHAPImportance.png
│   │   ├── Figure11_SHAPBeeswarm.png
│   │   ├── Figure12_PermutationImportance.png
│   │   ├── Figure13_AblationAnalysis.png
│   │   ├── Figure14_FinalTestSetEvaluation.png
│   │   ├── Figure15_ThreeThresholdConfusionMatrices.png
│   │   ├── Figure16_LR_vs_XGB_DCA.png
│   │   └── Figure17_PerModelConfusionMatrices.png
│   ├── Tables/                             # All numerical results in CSV format
│   │   ├── Table00_DecisionsLog.csv
│   │   ├── Table01_DataDictionary.csv
│   │   ├── Table02_ExclusionAudit.csv
│   │   ├── Table03_MutationRatesByGrade.csv
│   │   ├── Table04_ImbalanceStrategyComparison.csv
│   │   ├── Table05_BaselineIncrementalValue.csv
│   │   ├── Table06_ModelFamilyComparison.csv
│   │   ├── Table07_StatisticalComparison.csv
│   │   ├── Table08_PrimaryModelBootstrapCI.csv
│   │   ├── Table09_CalibrationComparison.csv
│   │   ├── Table10_ThresholdCostSensitivity.csv
│   │   ├── Table11_DecisionCurveAnalysis.csv
│   │   ├── Table12_SHAPImportance.csv
│   │   ├── Table13_PermutationImportance.csv
│   │   ├── Table14_LeaveOneFeatureOut.csv
│   │   ├── Table15_ReducedPanelAblation.csv
│   │   ├── Table16_FinalTestSetResults.csv
│   │   ├── Table17_ConfusionMatrices_ThreeThresholds.csv
│   │   └── Table26_HeadToHead_XGB_vs_LR.csv
│   ├── Models/                             # Serialized calibrators, out-of-fold predictions, models
│   └── AssetIndex.csv                      # Index of all generated figures and tables
├── environment/
│   ├── reproducibility_report.json         # Execution environment & hardware cross-check audit
│   ├── requirements_primary.txt            # Pinned primary dependencies
│   └── requirements_secondary.txt          # Secondary validation environment
├── requirements.txt                        # Standard pip requirements
├── environment.yml                         # Conda environment specification
├── LICENSE                                 # MIT License
└── README.md                               # Project documentation
```

---

## ⚡ Quick Start

### Option 1: Run in Google Colab (Recommended)
You can run the entire pipeline in Google Colab with free GPU/CPU runtimes:
1. Click the **Open in Colab** badge above or open [`Glioma_Classification_Pipeline.ipynb`](Glioma_Classification_Pipeline.ipynb) directly.
2. Run the first setup cell to install dependencies:
   ```bash
   !pip install -q xgboost==3.0.1 shap==0.51.0 imbalanced-learn==0.14.0 statsmodels scikit-learn
   ```
3. Run all cells (`Runtime -> Run all`). All outputs will be automatically generated and saved to `Results/`.

### Option 2: Local Setup (pip)
```bash
# 1. Clone the repository
git clone https://github.com/SandeepSinghNk/Glioma-Classification.git
cd Glioma-Classification

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook Glioma_Classification_Pipeline.ipynb
```

### Option 3: Conda Setup
```bash
conda env create -f environment.yml
conda activate glioma-ml
jupyter notebook Glioma_Classification_Pipeline.ipynb
```

---

## 📊 Dataset & Provenance

* **Source:** UCI Machine Learning Repository (Dataset ID: [759](https://doi.org/10.24432/C5R62J)), derived from TCGA-LGG and TCGA-GBM cohorts.
* **Cohort Size:** $N = 839$ adult patients (487 LGG, 352 GBM) after excluding 23 patients with missing clinical values from the raw TCGA export ($N = 862$).
* **Duplicate Record Note:** Two rows sharing identical categorical values were verified against raw TCGA Case IDs as representing distinct biological patients (`TCGA-P5-A72W`, 35y 65d; and `TCGA-E1-A7Z4`, 35y 66d) whose ages binned identically to 35.18 years. Both are retained in the authentic analysis cohort.
* **Features:** 23 total predictors:
  * **Clinical (3):** Age at diagnosis (continuous, years), Gender (binary), Race (4-level nominal).
  * **Molecular (20):** Somatic mutation status of *IDH1, TP53, ATRX, PTEN, EGFR, CIC, MUC16, PIK3CA, NF1, PIK3R1, FUBP1, RB1, NOTCH1, BCOR, CSMD3, SMARCA4, GRIN2A, IDH2, FAT4, PDGFRA*.

---

## 🛠️ Methodological Highlights

* **Leakage-Free Cross-Validation:** All imputation, standard scaling, one-hot encoding, and class-weighting steps are executed strictly inside training folds using `scikit-learn` and `imbalanced-learn` pipelines.
* **Primary Model Selection Rule:** Model selection among Logistic Regression, Random Forest, XGBoost, and SVM is locked prior to test evaluation based on nested-CV mean outer-fold AUC. Logistic Regression was locked as primary (mean-of-fold AUC = 0.913).
* **Calibration & Threshold Optimization:** Deployment calibration is assessed via Cox calibration slopes and ECE; decision cutoffs are selected via Youden's $J$ on pooled out-of-fold probabilities.
* **Decision Curve Analysis (DCA):** Net clinical benefit is quantified against treat-all and treat-none strategies across the full threshold probability spectrum.

---

## 📝 Citation

If you use this codebase, models, or audit data in your research, please cite:

```bibtex
@article{singh2026glioma,
  title={Machine learning triage for glioma histological grading using a minimal mutation and age panel},
  author={Singh, Sandeep and Pal, Ashok and Mittal, Nitin and Kaur, Jatinder and Gared, Fikreselam},
  journal={Scientific Reports},
  year={2026},
  note={Under review}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
