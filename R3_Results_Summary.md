# Revision 3 canonical run — results summary (12 Oct 2026)

**Manuscript:** *Impact of diagnostic label–IDH discordance and WHO 2021 criteria on machine-learning glioma grading from somatic mutations*

All numbers below come from `Results/ManuscriptNumbers.csv` and were re-checked independently from the saved test-set predictions.

## 1. Locks (all decided on the 671-patient training partition before the test set was opened)
| Decision | Value | Rule |
|---|---|---|
| Imbalance handling | Class weighting (now actually applied) | Highest mean CV AUC; all four strategies within 0.0001 AUC |
| Primary model | **XGBoost** | Highest mean-of-fold nested-CV AUC (0.915); also wins pooled-OOF (0.912) |
| Calibration | Platt (cross-fitted) | Lowest ECE among non-degenerate methods |
| Threshold | **0.351** (bootstrap 95 % CI 0.257–0.567) | Youden's J on calibrated OOF probabilities |
| 4-feature panel | **IDH1, Age, IDH2, TP53** | Majority vote of nested per-fold top-4 selections |

## 2. Sealed test set (n = 168; 70 GBM, 98 LGG)
| Metric | Value (95 % CI) |
|---|---|
| AUC | **0.932** (0.893–0.964) |
| Sensitivity | **0.929** (0.864–0.985) |
| Specificity | **0.796** (0.710–0.875) |
| Accuracy | 0.851 (0.792–0.905) |
| PPV / NPV | 0.765 / 0.940 |
| Brier | 0.101 (58 % below the no-skill 0.243) |
| Confusion matrix | TP 65, TN 78, FP 20, FN 5 |
| Age + IDH1 (LR) | AUC 0.908 = 97.5 % of primary; ΔAUC +0.023 (−0.010 to +0.056), DeLong p = 0.17 |
| 4-feature panel | AUC 0.919 = 98.6 % of primary; p = 0.34 |

## 3. What the run shows
1. **Model family does not matter.** All four families lie within 0.006 AUC; XGBoost vs logistic regression p = 0.95; a nested 4-family ensemble adds +0.001 (p = 0.73). The pre-specified rule picked XGBoost in R1, LR in R2 and XGBoost again here because the families are tied — this should be stated openly.
2. **Incremental value over Age + IDH1 is small.** Development: +0.015 AUC (95 % CI +0.001 to +0.030, p = 0.037). Sealed test: +0.023, not significant (p = 0.17). Decision curves: net-benefit gain ≈ 0.013 at the locked threshold (about 1 additional correctly triaged GBM per 100 patients).
3. **Almost every error is a label–IDH discordant case — this is the paper's strongest new finding.**
   - Development (OOF, n = 671): 84 of 86 errors are in the 105 discordant patients; accuracy 99.6 % in concordant vs 20 % in discordant patients.
   - Sealed test: 22 of 25 errors are in the 24 discordant patients; concordant-case AUC 0.999.
   - 129 of 839 legacy labels (15.4 %) disagree with IDH1 status, so an IDH1 rule is capped at 84.6 % accuracy — close to the 85.7–87.6 % reported on this benchmark (Mohamed; Tasci). Accuracy above ~95 % would require correctly classifying most discordant cases, which these features do not support (20 % accuracy in that group), so such reports warrant scrutiny for leakage.
   This directly answers Reviewer 4 (label validity) and gives the Editor a substantive empirical contribution beyond "good practice".
4. **WHO 2021 audit corrected.** The 106 IDH1-wild-type legacy LGG split into 21 IDH2-mutant (IDH-mutant tumours) and 85 IDH1/2-wild-type; plus 23 IDH1-mutant and 2 IDH2-mutant legacy GBM. Raw file: 108 IDH1-wt LGG, 2 excluded for missing demographics. IDH-status discordance overall: 110 patients (13.1 %).
5. **Feature importance.** IDH1 dominates (mean |SHAP| 1.66; permutation 0.27), then age and TP53. In nested LOFO no single feature's removal is significant after Holm correction (IDH1 p_Holm = 0.052) — the panel is redundant, not driven by one marker alone.
6. **Threshold is imprecise.** Bootstrap CI 0.26–0.57; raw-scale Youden thresholds across 10 repeated nested-CV runs 0.41–0.50. Report the threshold as a default, not a validated operating point (Reviewer 4, comment 5).
7. **Tuning helps a little:** +0.014 AUC over library defaults (p = 0.005).

## 4. Changes the manuscript needs (high level)
- Primary model: XGBoost (not LR). Every headline number changes: AUC 0.932, sensitivity 0.929, specificity 0.796, τ = 0.351, CM 65/78/20/5.
- Panel: IDH1 + Age + IDH2 + TP53 (98.6 % of test AUC), not IDH1 + Age + TP53 + PTEN.
- Incremental value: dev ΔAUC +0.015 (p = 0.037), test +0.023 (p = 0.17) — replace +0.020/+0.022 and p = 0.049/0.0010.
- Add the label–IDH discordance error analysis (Tables 21, 30) as a Results subsection and make it central to the Discussion and the reply to the Editor.
- Replace the WHO 2021 subsection with the corrected IDH2-aware audit (Tables 28–29).
- Disclose that the same test split was scored in earlier rounds.
- Software versions: Python 3.13.16, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.53.0, imbalanced-learn 0.14.2.
- Figure 16 is now the test-set decision curve (LR-vs-XGB DCA retired).

## 5. Added after independent review (12 Oct 2026, same canonical run)
- **IDH rules.** At its operating point the model is not more accurate than the no-fit rule "IDH1/2-wildtype implies GBM" (test accuracy 0.851 vs 0.869; 25 vs 22 errors). Its AUC is higher (0.932 vs 0.880, p < 0.001) because its output ranks patients within IDH groups.
- **Concordance by IDH1/2.** Concordant patients: 2 of 583 out-of-fold errors and 3 of 146 test errors. Discordant patients: 84 of 88 out-of-fold errors and 22 of 22 test errors. Every one of these errors follows IDH status.
- **WHO 2021 reconstruction** (Ceccarelli 2016 annotations linked by case ID). The legacy label agrees with glioblastoma, IDH-wildtype in 88.8% of patients (n = 830) and with CNS WHO grade 4 in 90.0% (n = 827).
  - Of the 85 IDH-wildtype legacy LGG: 68 are molecular glioblastomas; 8 meet none of the tested criteria; 9 are not assessable because TERT status is unknown.
  - 23 legacy GBM and 15 legacy LGG are IDH-mutant astrocytomas, grade 4.
  - The model agrees with the glioblastoma, IDH-wildtype vs other distinction with accuracy 0.986 out-of-fold and 0.970 on test.
  - 72 of 79 and 19 of 24 assessable legacy-label errors agree with this reconstructed distinction.

## Round 3 Updates (Fully Nested Joint Selection & 64-Column Reproducibility Release)
- **Fully nested inner-loop joint selection (canonical 5-fold CV, `6_Code/tools/run_fully_nested_joint.py`):** Evaluates all 12 combinations of 3 valid imbalance strategies (`No resampling`, fold-specific `Class-weight (balanced)`, `SMOTENC`) $\times$ 4 model families (`LogisticRegression`, `SVM_RBF`, `RandomForest`, `XGBoost`) strictly inside each outer fold inner 5-fold CV. Result: mean-of-fold AUC **$0.908 \pm 0.023$**, pooled out-of-fold AUC **$0.905$**, Brier **$0.108$**, **$84$ errors** out of $671$ at threshold $0.5$ (**$81/88$ discordant**, **$3/583$ concordant**), exact McNemar $p=0.34$ vs. the zero-parameter `IDH1/2` rule ($88$ errors).
- **Self-contained 64-column label release (`Supplementary_Data_1.xlsx` / `TCGA_GliomaBenchmark_WHO2021_Labels_v1.0.csv`):** Expanded from 53 to 64 columns across all 839 patients by adding `Age_at_diagnosis`, `GRADE_Ceccarelli`, `SNP6_array`, `WHO_grade4_d2`, `WHO_grade4_h1`, `WHO_GBM_IDHwt_h1`, `WHO_grade4_h0`, `WHO_GBM_IDHwt_h0`, `MolGBM_qualifying_criteria_count`, `WHO2021_cIMPACT11_cautious`, and `MONET_Method2021_category`.
