# Canonical Analysis Run — Results Summary (17 Nov 2026)

**Manuscript:** *Impact of diagnostic label–IDH discordance and WHO 2021 criteria on machine-learning glioma grading from somatic mutations*

All numbers below come from `Results/ManuscriptNumbers.csv` and were verified against the saved cross-validation and held-out test-set predictions.

## 1. Locked Development Decisions (decided on the 671-patient training partition before test evaluation)
| Decision | Value | Rule |
|---|---|---|
| Cohort | N = 839 (487 LGG / 352 GBM); no rows removed | 2 identical feature rows map to distinct TCGA patients (TCGA-P5-A72W, TCGA-E1-A7Z4); both retained |
| Train/test split | 671/168, stratified, seed 42 | Fixed a priori (80/20 stratified development/test split, seed 42) |
| Imbalance handling | Class-weight (balanced) | Highest mean CV AUC (`0.9020 ± 0.0184`); all four strategies within `0.0007` AUC (`Imbalance_AUC_spread = 0.0007`) |
| Primary model | **XGBoost** | Highest mean-of-fold nested-CV AUC (`0.915` [`0.9149 ± 0.0186`]); also wins pooled OOF AUC (`0.912` [`0.9116`]) |
| Calibration | Platt/sigmoid (cross-fitted) | Lowest ECE among non-degenerate methods (pile-up <= 10%) |
| Threshold | **0.351** (bootstrap 95% CI 0.257–0.567) | Youden's J on calibrated OOF probabilities (training only) |
| 4-feature panel | **IDH1, Age_at_diagnosis, IDH2, TP53** | Majority vote of nested per-fold top-4 selections (training only) |

## 2. Model Family & Statistical Comparisons (Development Partition, n = 671)
| Model / Comparison | Mean-of-Fold AUC (± SD) | Pooled OOF AUC | Paired DeLong p (vs. XGBoost) |
|---|---|---|---|
| **XGBoost (Primary)** | **0.915** (`0.9149 ± 0.0186`) | **0.912** (`0.9116`) | — |
| **LogisticRegression** | **0.912** (`0.9124 ± 0.0228`) | **0.911** (`0.9113`) | `p = 0.95` (`0.9508`) |
| **RandomForest** | **0.910** (`0.9105 ± 0.0142`) | **0.905** (`0.9050`) | `p = 0.20` (`0.2000`) |
| **SVM_RBF** | **0.909** (`0.9090 ± 0.0191`) | **0.894** (`0.8939`) | `p = 0.019` (`0.0188`) |
| **Soft-vote ensemble (4 families, nested)** | — | **0.912** (`0.9123`) | `p = 0.823` (`0.8227`, ΔAUC `+0.0007`) |
| **Age + IDH1 (LR, nested)** | — | **0.896** (`0.8963`) | `p = 0.0368` (ΔAUC `+0.0153`, 95% CI `+0.0009` to `+0.0297`) |
| **Fully nested inner-loop joint selector** | **0.908** (`0.9080 ± 0.0227`) | **0.905** (`0.9053`) | 84 errors at 0.5 (81/88 discordant; McNemar `p = 0.34` vs. `IDH1/2` rule) |

## 3. Held-Out Test Set (n = 168; 70 GBM, 98 LGG)
| Metric | Value (95% CI) |
|---|---|
| AUC | **0.932** (0.893–0.964) |
| Sensitivity | **0.929** (0.864–0.985) |
| Specificity | **0.796** (0.710–0.875) |
| Accuracy | 0.851 (0.792–0.905) |
| PPV / NPV | 0.765 / 0.940 |
| Brier | 0.101 (58% below the no-skill 0.243) |
| Confusion matrix | TP 65, TN 78, FP 20, FN 5 |
| Age + IDH1 (LR) | AUC 0.908 = 97.5% of primary; ΔAUC +0.023 (−0.010 to +0.056), DeLong `p = 0.17` |
| 4-feature panel | AUC 0.919 = 98.6% of primary; DeLong `p = 0.34` |

## 4. Key Empirical Findings
1. **Model family selection does not drive accuracy.** All four families lie within `0.006` mean-of-fold AUC (`0.909` to `0.915`); XGBoost vs. logistic regression `p = 0.95`, XGBoost vs. Random Forest `p = 0.20`, and a leakage-free nested 4-family soft-vote ensemble yields pooled OOF AUC `0.912` (`p = 0.823`). Across 10×5-fold repeated nested cross-validation, XGBoost achieves a mean-of-fold AUC of `0.916 ± 0.004` and a pooled OOF AUC of `0.913 ± 0.005` (range `0.903–0.921`).
2. **Incremental value over Age + IDH1 is small.** Development: `+0.015` AUC (95% CI `+0.001` to `+0.030`, `p = 0.0368`). Held-out test: `+0.023`, not statistically significant (`p = 0.17`). Decision curves: net-benefit gain `≈ 0.013` at the locked threshold (`0.336` vs. `0.323` for Age + IDH1, and `0.336` vs. `0.335` for the zero-parameter `IDH1/2` rule).
3. **Classification errors track diagnostic label–IDH discordance.**
   - Development (OOF, n = 671): 84 of 86 errors occur in the 105 IDH1-discordant patients (and 84 of 88 in the 88 IDH1/2-discordant patients; 2 errors in 583 concordant patients).
   - Held-out test (n = 168): 22 of 25 errors occur in the 24 IDH1-discordant patients (22 of 22 in IDH1/2-discordant patients; 3 errors in 146 concordant patients).
   - Across the cohort, 110 of 839 legacy labels (13.1%) disagree with *IDH1*/*IDH2* status (85 IDH-wildtype legacy LGG and 25 IDH-mutant legacy GBM), and 129 of 839 (15.4%) disagree with *IDH1* status alone (21 *IDH1*-wildtype LGG carry an *IDH2* mutation). Consequently, the zero-parameter rule `"IDH1/2-wildtype -> GBM"` achieves 86.9% accuracy (88 development errors vs. 86 model errors, exact McNemar `p = 0.69`; 22 test errors vs. 25 model errors, exact McNemar `p = 0.25`).
4. **WHO 2021 molecular audit and reconstruction.** Linking all 839 patients by TCGA case identifier to published molecular annotations (Ceccarelli et al., *Cell* 2016) shows that the legacy label agrees with glioblastoma, IDH-wildtype in 88.8% of assignable patients (n = 830) and with CNS WHO grade 4 in 90.0% (n = 827):
   - Of the 85 IDH-wildtype legacy LGG: 68 (80.0%; range 51.8–84.0% across sensitivity variants) meet molecular criteria for glioblastoma, IDH-wildtype (*TERT* promoter mutation, *EGFR* amplification, or `+7/-10`); 8 meet none of the tested criteria; 9 are not assessable due to missing *TERT* promoter status.
   - All 23 assessable non-codeleted IDH-mutant legacy GBM and 15 IDH-mutant legacy LGG with *CDKN2A/B* deep deletion map to astrocytoma, IDH-mutant, CNS WHO grade 4.
   - The locked model agrees with the reconstructed glioblastoma, IDH-wildtype vs. other distinction with accuracy 0.986 out-of-fold and 0.970 on the test set (72 of 79 development errors and 19 of 24 test errors agree with reconstructed tumour type, compared with 52 of 78 and 16 of 23 for CNS WHO grade 4).
   - Compared against the independent WHO 2021 TCGA reclassification of de Mendonça et al. (2025), the base reconstruction agrees in 811 of 816 mutually assigned patients (99.4%; 4-category Cohen's `κ = 0.99`, binary GBM-IDHwt `κ = 0.988`).
5. **Feature importance and ablation.** *IDH1* dominates (mean `|SHAP| = 1.656`; permutation importance `0.266`), followed by age (`0.553`) and *TP53* (`0.434`). In fully nested LOFO, no single feature's removal remains statistically significant after Holm correction (*IDH1* `p_Holm = 0.0515`).
6. **Threshold stability and hyperparameter tuning.** The calibrated Youden threshold is `0.351` (bootstrap 95% CI `0.257–0.567`; raw-scale Youden thresholds across 10 repeated nested-CV runs span `0.405–0.497`, mean `0.459 ± 0.039`) and is treated as a default operating point. Hyperparameter tuning adds `+0.014` AUC over library defaults (`p = 0.005`).

## 5. Software Environment & Reproducibility Release
- **Pinned Colab canonical environment (`Results/Models/reproducibility_report.json`):** Python `3.13.16`, `scikit-learn==1.6.1`, `xgboost==3.4.1`, `shap==0.54.1`, `imbalanced-learn==0.14.2`, `numpy==2.4.6`, `pandas==2.2.3`, `scipy==1.16.3`, `statsmodels==0.15.0`, `matplotlib==3.10.0`, `seaborn==0.13.2`.
- **Self-contained 64-column patient-level label release (`Data/Release/TCGA_GliomaBenchmark_WHO2021_Labels_v1.0.csv`):** Covers all 839 patients across 64 columns including `Age_at_diagnosis`, `GRADE_Ceccarelli`, `SNP6_array`, all 10 reconstruction variants (`a`–`h1`), `MolGBM_qualifying_criteria_count`, `WHO2021_cIMPACT11_cautious`, and `MONET_Method2021_category`.
