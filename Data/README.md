# Dataset Documentation & Provenance

This directory contains the primary and provenance datasets used in:
> **"Machine learning triage for glioma histological grading using a minimal mutation and age panel"**  
> *Sandeep Singh, Ashok Pal, Nitin Mittal, Jatinder Kaur, and Fikreselam Gared (Scientific Reports, 2026).*

---

## 📂 File Descriptions

1. **`TCGA_InfoWithGrade.csv` ($N = 839 \times 24$)**
   * Primary analysis dataset from the UCI Machine Learning Repository (Dataset ID: [759](https://doi.org/10.24432/C5R62J)).
   * Comprises 839 adult patients with histologically confirmed diffuse glioma (487 LGG, 352 GBM).
   * Fully preprocessed and encoded without missing values.

2. **`TCGA_GBM_LGG_Mutations_all.csv` ($N = 862 \times 27$)**
   * Raw, unfiltered TCGA export preserving original clinical identifiers (`Case_ID`, `Project`, `Primary_Diagnosis`).
   * Used for exclusion provenance auditing and patient identity verification.

3. **`Discordant_Cases_Audit_WHO2021.csv` ($N = 129 \times 15$)**
   * Curated audit of the 129 patients in the analysis cohort whose legacy histological grade diverges from their *IDH1* mutation status under the WHO 2021 CNS classification.
   * Includes patient TCGA Case IDs, original primary histological diagnoses, and status of key co-occurring markers (*TP53*, *ATRX*, *EGFR*, *PTEN*, *CIC*, *FUBP1*).

---

## 📋 Data Dictionary

| Variable | Role | Type | Encoding / Unit |
| :--- | :--- | :--- | :--- |
| **`Grade`** | Target | Binary | `0` = Lower-Grade Glioma (LGG); `1` = Glioblastoma (GBM) |
| **`Gender`** | Clinical | Binary | `0` = Male (n=488, 58.2%); `1` = Female (n=351, 41.8%) |
| **`Age_at_diagnosis`** | Clinical | Continuous | Age in years at diagnosis (range: 14.37 – 89.29 years) |
| **`Race`** | Clinical | Nominal | `0` = White (n=765); `1` = Black or African American (n=59); `2` = Asian (n=14); `3` = American Indian or Alaska Native (n=1) |
| **`IDH1`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`TP53`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`ATRX`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`PTEN`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`EGFR`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`CIC`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`MUC16`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`PIK3CA`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`NF1`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`PIK3R1`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`FUBP1`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`RB1`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`NOTCH1`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`BCOR`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`CSMD3`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`SMARCA4`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`GRIN2A`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`IDH2`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`FAT4`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |
| **`PDGFRA`** | Molecular | Binary | `0` = Not Mutated; `1` = Mutated |

---

## 🔍 Exclusion & Clinical Provenance Audit

### 1. Exclusion Audit (862 Raw -> 839 Analysis Patients)
The 23 patients omitted during UCI's preprocessing from the raw 862-patient TCGA export are accounted for by missing clinical fields:
* Missing Gender: 4 cases
* Missing Age: 5 cases
* Missing / Not Reported Race: 22 cases
* **Union of Exclusions:** exactly 23 instances.

### 2. Patient Identity Audit for Identical Categorical Rows
Two rows in the preprocessed file match across all 24 fields when age is rounded to two decimal places (35.18 years). Cross-referencing raw TCGA Case IDs confirms:
* **Row 418:** `TCGA-P5-A72W` (Male, White, Age: 35 years 65 days, Grade: LGG)
* **Row 433:** `TCGA-E1-A7Z4` (Male, White, Age: 35 years 66 days, Grade: LGG)

Both patients carried matching mutation calls across the 20 genes. Because they are clinically and biologically distinct individuals, both records are retained in the canonical $N = 839$ analysis dataset.

---

## 🧬 WHO 2021 Neuropathological Discordance Summary

Under the WHO 2021 classification of CNS tumors, historical histological grading diverges from molecular taxonomy in 129 cases:
1. **106 IDH1-wildtype histological LGGs:** Under WHO 2021, IDH-wildtype diffuse astrocytomas harboring *EGFR* mutations ($n=30$), *PTEN* mutations ($n=20$), or *TERT* / $+7/-10$ alterations are reclassified as **Glioblastoma, IDH-wildtype, CNS WHO grade 4**, explaining their clinical aggressiveness despite initial histological diagnosis.
2. **23 IDH1-mutant histological GBMs:** All 23 cases exhibit secondary high-grade histological necrosis or microvascular proliferation, and 17 carry co-occurring *TP53* and *ATRX* mutations. Under WHO 2021, these tumors are classified as **Astrocytoma, IDH-mutant, CNS WHO grade 4**, because glioblastoma is strictly defined as IDH-wildtype.
