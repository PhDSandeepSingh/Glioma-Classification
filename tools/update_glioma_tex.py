from pathlib import Path

BASE = Path("/Users/sandy/Downloads/Research Work/Glioma/Revision 2/Submission Ready/Final")
tex_path = BASE / "1_Manuscript/Glioma.tex"
tex = tex_path.read_text(encoding="utf-8")

def replace_exact(src, old, new, label):
    if old not in src:
        raise AssertionError(f"Could not find target block: {label}")
    return src.replace(old, new, 1)

# 1. Abstract (L41)
old_abs = (
    r"\abstract{The widely used UCI/TCGA Glioma Grading benchmark labels patients as lower-grade glioma (LGG) or glioblastoma (GBM) "
    r"according to WHO 2007 histology, yet an analysis of this target revealed that it disagreed with IDH mutation status in 110 of 839 patients (13.1\%). "
    r"In published TCGA annotations, 68 of 85 IDH-wildtype legacy LGG met molecular criteria for glioblastoma (51.8--84.0\% across variants), "
    r"whereas all 23 non-codeleted IDH-mutant legacy GBM were CNS WHO grade~4 astrocytomas (21 of 25 on histology alone, lacking \gene{CDKN2A/B} deletion). "
    r"Under nested cross-validation, XGBoost trained on 20 mutations, age, sex and race reached a test AUC of 0.932 with 25 errors in 168 patients, "
    r"versus 22 errors for the rule ``IDH-wildtype implies GBM'' (McNemar $p=0.25$). Most errors (84 of 86 development; 22 of 25 test) occurred in discordant patients. "
    r"Of assessable development errors, 72 of 79 agreed with reconstructed tumour type but only 52 of 78 agreed with grade~4, and the rule matched or exceeded the model on both targets. "
    r"Adding copy-number and 1p/19q features did not materially increase legacy-label accuracy. "
    r"Within these features, classifier choice does not limit benchmark accuracy; tumour-type errors largely reflect outdated labels, whereas grade errors reflect missing histological information.}"
)
new_abs = (
    r"\abstract{The widely used UCI/TCGA Glioma Grading benchmark labels patients as lower-grade glioma (LGG) or glioblastoma (GBM) "
    r"according to WHO 2007 histology, yet an analysis of this target revealed that it disagreed with IDH mutation status in 110 of 839 patients (13.1\%). "
    r"In a post hoc reconstruction from published TCGA molecular annotations, 68 of 85 IDH-wildtype legacy LGG met approximate molecular criteria for glioblastoma "
    r"(51.8--84.0\% across sensitivity variants), whereas all 23 assessable non-codeleted IDH-mutant legacy GBM mapped to CNS WHO grade~4 astrocytoma "
    r"(21 of 25 via the legacy histological label alone, lacking \gene{CDKN2A/B} deep deletion). "
    r"In primary internal validation using 10-times-repeated 5-fold nested cross-validation on the development partition ($n=671$), XGBoost trained on 20 somatic mutations, "
    r"age, sex and race achieved a pooled out-of-fold AUC of $0.913\pm 0.005$ (canonical nested-CV AUC 0.912; 86 errors versus 88 for the zero-parameter rule "
    r"``IDH-wildtype implies GBM'', McNemar $p=0.69$), consistent with fully nested inner-loop joint selection of imbalance strategy and model family "
    r"(mean-of-fold AUC $0.908\pm 0.023$; pooled AUC 0.905). On the exploratory held-out test set ($n=168$; previously examined in earlier revisions), "
    r"the locked pipeline reached an AUC of 0.932 with 25 errors versus 22 rule errors ($p=0.25$). Most errors (84 of 86 development; 22 of 25 exploratory test) occurred in discordant patients. "
    r"Of assessable development errors, 72 of 79 agreed with reconstructed tumour type but only 52 of 78 agreed with grade~4, and the rule matched or exceeded the model on both targets. "
    r"Adding copy-number and 1p/19q features did not materially increase legacy-label accuracy. On this single-cohort benchmark, performance among the four tested tabular classifier families "
    r"was bounded primarily by label--IDH discordance and unmeasured histological criteria rather than classifier choice, and the multivariable model offered no predictive advantage over IDH mutation status alone.}"
)
tex = replace_exact(tex, old_abs, new_abs, "Abstract")

# 2. Introduction (L57)
old_intro = (
    r"Using the TCGA case identifiers in the raw export, the study linked every patient to published copy-number, 1p/19q and \gene{TERT} promoter data~\cite{ceccarelli2016molecular} "
    r"and reconstructed approximate WHO 2021 categories, which gives a model-free account of how the legacy label relates to current tumour type and grade. "
    r"The study evaluated four model families under a protocol in which every modelling decision was locked by nested cross-validation before a held-out test set was scored, "
    r"compared the model with no-fit rules based on IDH status, located its errors relative to label--IDH agreement and the reconstructed categories, and tested whether richer molecular inputs change the picture."
)
new_intro = (
    r"Using the TCGA case identifiers in the raw export, this study linked every patient to published copy-number, 1p/19q and \gene{TERT} promoter data~\cite{ceccarelli2016molecular} "
    r"and conducted a post hoc reconstruction of approximate WHO 2021 categories~\cite{louis2021who}, which gives a model-free account of how the legacy label relates to current tumour type and grade. "
    r"The study evaluated four tabular classifier families using 10-times-repeated 5-fold nested cross-validation (alongside fully nested inner-loop joint selection of imbalance strategy and model family) "
    r"as the primary internal performance assessment on the development partition ($n=671$), treated evaluation on the held-out test set ($n=168$) as exploratory because test-set results from earlier revision rounds had been examined, "
    r"compared the model with zero-parameter rules based on IDH status, located its errors relative to label--IDH agreement and the reconstructed categories, and tested whether richer molecular inputs change the picture."
)
tex = replace_exact(tex, old_intro, new_intro, "Introduction")

# 3. Fig 1 caption (L72)
old_fig1 = (
    r"\caption{\textbf{Study design.} Cohort derivation from the raw TCGA export, the label--IDH analysis, the training and test partitions, and the sequence of decisions locked on the training partition before the test set was scored. "
    r"The WHO 2021 reconstruction and its sensitivity analyses were added after the test set had been scored and do not affect any locked decision. Test-set results from earlier versions of this analysis had been seen (see Methods).}"
)
new_fig1 = (
    r"\caption{\textbf{Study design.} Cohort derivation from the raw TCGA export ($n=862$ to $n=839$), the label--IDH audit, the development ($n=671$) and exploratory held-out test ($n=168$; 98 legacy LGG, 70 legacy GBM) partitions, "
    r"and the development-partition pipeline incorporating fully nested joint inner-loop selection and 10-times-repeated 5-fold nested cross-validation as the primary internal validation. "
    r"Because test-set results from earlier revision rounds had been seen, test-set evaluation is interpreted as exploratory; the WHO 2021 reconstruction and its sensitivity analyses were conducted post hoc and do not affect any locked decision (Methods).}"
)
tex = replace_exact(tex, old_fig1, new_fig1, "Fig 1 caption")

# 4. Results: WHO 2021 sensitivity bounds & de Mendonca (L112-L114)
old_who112 = (
    r"Compared with the independent TCGA WHO 2021 reclassification of de Mendon\c{c}a et al.~\cite{mendonca2025updating} "
    r"(which uses curated IDH status without \gene{EGFR} amplification, \gene{CDKN2A/B} or grading), 811 of 816 mutually assigned patients agreed "
    r"(99.4\%; four-category $\kappa=0.990$; IDH-wildtype glioblastoma $\kappa=0.988$), with all five disagreements reflecting the IDH source "
    r"(among the 85 IDH-wildtype legacy LGG: 63 IDH-wildtype glioblastoma, 17 unclassified and 5 IDH-mutant astrocytoma; Supplementary Table~S37)."
)
new_who112 = (
    r"Each sensitivity variant bounds a distinct source of uncertainty in the retrospective annotations: variant~b (68 of 81; 84.0\%) bounds discrepancies between dataset mutation flags and curated IDH calls; "
    r"variant~c (44 of 85; 51.8\%) bounds morphological uncertainty by withholding molecular glioblastoma criteria from 32 IDH-wildtype legacy LGG with oligodendroglial or mixed diagnoses lacking central slide review or DNA-methylation classifier scores; "
    r"and the cIMPACT-NOW update~11 cautious variant (64 of 85; 75.3\%) and $\geq$2-criteria variant (45 of 85; 52.9\%; Supplementary Tables~S28 and S38) bound the risk of over-calling glioblastoma from an isolated molecular alteration "
    r"(noting that 7 of the 68 molecular glioblastomas lack a Ceccarelli histological grade and 26 lack \gene{TERT} promoter status). "
    r"Compared with the TCGA WHO 2021 reclassification of de Mendon\c{c}a et al.~\cite{mendonca2025updating} (which uses curated IDH status without \gene{EGFR} amplification, \gene{CDKN2A/B} or grading), "
    r"811 of 816 mutually assigned patients agreed (99.4\%; four-category $\kappa=0.990$; IDH-wildtype glioblastoma $\kappa=0.988$), with all five disagreements reflecting the IDH source "
    r"(among the 85 IDH-wildtype legacy LGG: 63 IDH-wildtype glioblastoma, 17 unclassified and 5 IDH-mutant astrocytoma; Supplementary Table~S37); "
    r"because both reclassifications draw from the same upstream Ceccarelli et al.~\cite{ceccarelli2016molecular} TCGA annotations, this high concordance confirms algorithmic rule consistency rather than independent histopathological validation."
)
tex = replace_exact(tex, old_who112, new_who112, "WHO L112")

old_who114 = (
    r"With respect to tumour grade, however, the two discordant groups differ: the 23 non-codeleted IDH-mutant legacy GBM with 1p/19q data are astrocytomas, IDH-mutant, CNS WHO grade~4~\cite{brat2020cimpact5}, "
    r"so their high-grade label remains accurate, whereas 15 IDH-mutant legacy LGG with \gene{CDKN2A/B} deletion are CNS WHO grade~4 despite carrying an LGG label (Table~\ref{tab:who}a)."
)
new_who114 = (
    r"With respect to tumour grade, however, the two discordant groups differ: treating the legacy TCGA-GBM project label as a proxy for histological microvascular proliferation or necrosis, "
    r"the 23 non-codeleted IDH-mutant legacy GBM with 1p/19q data map to astrocytoma, IDH-mutant, CNS WHO grade~4~\cite{brat2020cimpact5}, whereas 15 IDH-mutant legacy LGG with \gene{CDKN2A/B} deep deletion (GISTIC $-2$) "
    r"are upgraded to CNS WHO grade~4 despite carrying an LGG label (Table~\ref{tab:who}a). Because central neuropathology slide review was unavailable and 71 additional IDH-mutant non-codeleted legacy LGG carried single-copy (GISTIC $-1$) "
    r"\gene{CDKN2A/B} loss that could mask focal homozygous deletion in low-purity samples, reconstructed grade~2--3 versus grade~4 assignments remain approximate."
)
tex = replace_exact(tex, old_who114, new_who114, "WHO L114")

# 5. Results: Model families, fully nested joint selection & repeated nested CV (L118)
old_l118 = (
    r"The training (671 patients; 389 LGG, 282 GBM) and test (168; 98 LGG, 70 GBM) partitions were stratified by label. "
    r"Class weighting was locked among four imbalance strategies whose AUCs differed by at most 0.0001 (Supplementary Table~S6). "
    r"Under nested cross-validation the mean-of-fold AUC was 0.915 for XGBoost, 0.912 for logistic regression, 0.911 for random forest and 0.909 for a support vector machine (SVM) (Table~\ref{tab:dev}; Supplementary Table~S8; Supplementary Fig.~S5). "
    r"XGBoost was locked as the primary model (pooled out-of-fold AUC 0.912). Logistic regression and random forest performed similarly ($p=0.95$ and $p=0.31$), the SVM had a lower pooled AUC (0.894; $p=0.019$), "
    r"and a soft-voting ensemble of the four families did not improve on XGBoost (0.913; $p=0.73$; Supplementary Table~S9). "
    r"The same pre-specified rule selected XGBoost and logistic regression in earlier versions of this analysis (Methods), so the identity of the ``best'' family is unstable."
)
new_l118 = (
    r"The development (671 patients; 389 LGG, 282 GBM) and exploratory test (168; 98 LGG, 70 GBM) partitions were stratified by label. "
    r"In the pre-specified development pipeline, class weighting was selected among four imbalance strategies whose 5$\times$5 repeated cross-validation AUCs differed by at most 0.0001 (Supplementary Table~S6). "
    r"Under canonical 5-fold nested cross-validation, the mean-of-fold AUC was 0.915 for XGBoost, 0.912 for logistic regression, 0.911 for random forest and 0.909 for a support vector machine (SVM) (Table~\ref{tab:dev}; Supplementary Table~S8; Supplementary Fig.~S5). "
    r"XGBoost was designated as the primary model (pooled out-of-fold AUC 0.912). Because selecting the imbalance strategy on the full development partition and choosing the primary family across outer folds can introduce selection optimism, "
    r"this study also performed a fully nested inner-loop joint selection across all three valid imbalance strategies (no resampling, fold-specific class weighting, SMOTENC) and all four classifier families inside each outer training fold; "
    r"this completely unbiased estimator yielded a mean-of-fold AUC of $0.908\pm 0.023$ (pooled out-of-fold AUC 0.905, accuracy 87.5\%, 84 errors at 0.5; selecting SMOTENC~+~random forest, class-weighted logistic regression, SMOTENC~+~XGBoost, class-weighted XGBoost, and SMOTENC~+~logistic regression across the five folds). "
    r"In primary internal validation via 10-times-repeated 5-fold nested cross-validation across different outer-fold splits, XGBoost achieved a mean-of-fold AUC of $0.916\pm 0.004$ and a pooled out-of-fold AUC of $0.913\pm 0.005$ (range 0.903--0.921; Supplementary Table~S19). "
    r"Logistic regression and random forest performed similarly ($p=0.95$ and $p=0.31$), the SVM had a lower pooled AUC (0.894; $p=0.019$), and a soft-voting ensemble of the four families did not improve on XGBoost (0.913; $p=0.73$; Supplementary Table~S9). "
    r"The same pre-specified rule selected XGBoost and logistic regression in earlier versions of this analysis (Methods), so the identity of the ``best'' family is unstable."
)
tex = replace_exact(tex, old_l118, new_l118, "Results L118")

# 6. Table 2 (tab:dev)
old_tab2_row = r"XGBoost (primary) & $0.915\pm0.019$ & 0.912 & -- \\"
new_tab2_row = (
    r"XGBoost (primary, canonical 5-fold) & $0.915\pm0.019$ & 0.912 & -- \\" + "\n" +
    r"XGBoost (10$\times$5-fold repeated nested CV) & $0.916\pm0.004$ & $0.913\pm0.005$ & -- \\" + "\n" +
    r"Fully nested joint selector (imbalance $\times$ family) & $0.908\pm0.023$ & 0.905 & -- \\"
)
tex = replace_exact(tex, old_tab2_row, new_tab2_row, "Tab2 row")

# 7. Calibration / DCA (L160)
old_l160 = r"Out-of-fold, the net benefit of the primary model exceeded that of age plus \gene{IDH1} by 0.013 at a threshold probability of 0.35 (95\% CI 0.005--0.021; Supplementary Table~S13; Supplementary Fig.~S8)."
new_l160 = (
    r"Out-of-fold, the net benefit of the primary model at a threshold probability of 0.35 exceeded that of age plus \gene{IDH1} by 0.013 (0.336 versus 0.323; 95\% CI 0.005--0.021) "
    r"but did not exceed that of the zero-parameter \gene{IDH1/2} rule (0.336 versus 0.335; $\Delta\text{NB}=0.001$, 95\% CI $-0.004$ to 0.006; Supplementary Table~S13a; Supplementary Fig.~S8)."
)
tex = replace_exact(tex, old_l160, new_l160, "L160 DCA")

# 8. Exploratory held-out test set (L184-L193, L255)
old_l184 = (
    "\\subsection*{Held-out test set}\n\n"
    "The locked pipeline (XGBoost on 23 features, Platt calibration, threshold 0.351) reached a test AUC of 0.932 (95\\% CI 0.893--0.964), sensitivity 0.929, specificity 0.796 and accuracy 0.851 (65 true positives, 78 true negatives, 20 false positives, 5 false negatives; Table~\\ref{tab:test}; Fig.~\\ref{fig:test}a)."
)
new_l184 = (
    "\\subsection*{Exploratory evaluation on the held-out test set}\n\n"
    "Because the held-out test partition ($n=168$; 98 legacy LGG, 70 legacy GBM) had been examined in earlier versions of this analysis (Methods), the following test-set metrics are interpreted as exploratory alongside the primary 10-times-repeated nested cross-validation estimates above. "
    "On this partition, the locked pipeline (XGBoost on 23 features, Platt calibration, threshold 0.351) reached an exploratory test AUC of 0.932 (95\\% CI 0.893--0.964), sensitivity 0.929, specificity 0.796 and accuracy 0.851 (65 true positives, 78 true negatives, 20 false positives, 5 false negatives; Table~\\ref{tab:test}; Fig.~\\ref{fig:test}a)."
)
tex = replace_exact(tex, old_l184, new_l184, "L184 Held-out test set")

tex = replace_exact(
    tex,
    r"The test AUC lay above the nested cross-validation estimates (0.912--0.915), and ten repetitions of nested cross-validation gave pooled AUCs of 0.903--0.921 (Supplementary Table~S19; Supplementary Fig.~S14).",
    r"The exploratory test AUC lay above the primary internal nested cross-validation estimates (0.912--0.915 canonical; 0.903--0.921 across ten repetitions of nested cross-validation; Supplementary Table~S19; Supplementary Fig.~S14).",
    "L186 end"
)

tex = replace_exact(
    tex,
    r"\caption{\textbf{Held-out test-set performance ($n=168$; 70 GBM, 98 LGG).}",
    r"\caption{\textbf{Exploratory held-out test-set performance ($n=168$; 70 GBM, 98 LGG).}",
    "Table 3 caption"
)
tex = replace_exact(
    tex,
    r"\caption{\textbf{Held-out test performance and the location of errors.}",
    r"\caption{\textbf{Exploratory held-out test performance and the location of errors.}",
    "Fig 4 caption"
)

# 9. Discussion (L307, L309, L313, L315, L317)
old_d307 = (
    r"On the UCI/TCGA glioma grading benchmark, the accuracy of a locked model tracked the agreement between the legacy label and IDH status. Model families performed similarly, the full model was not significantly better than age and \gene{IDH1} on the test set, "
    r"it made no fewer errors than a rule that calls every IDH-wildtype tumour GBM, and almost every error occurred in a patient whose label disagreed with IDH status. Within the information these 23 features carry, the choice of algorithm does not limit accuracy."
)
new_d307 = (
    r"On the UCI/TCGA glioma grading benchmark, the accuracy of the multivariable model closely tracked the agreement between the legacy histological label and IDH mutation status. "
    r"Across primary 10-times-repeated nested cross-validation on the development partition and exploratory evaluation on the held-out test set, the four evaluated tabular classifier families performed similarly, "
    r"the 23-feature model did not significantly improve discrimination over age and \gene{IDH1} on the exploratory test set, it made no fewer classification errors than the zero-parameter rule calling every IDH-wildtype tumour GBM, "
    r"and almost every error occurred in a label--IDH discordant patient. Within this 23-feature representation on this single-cohort benchmark, observed performance among the four tested classifier families was constrained primarily by label--IDH discordance rather than choice of learning algorithm."
)
tex = replace_exact(tex, old_d307, new_d307, "Discussion D307")

old_d309 = (
    r"The reconstructed tumour-type assignments also agreed in 99.4\% of mutually assigned patients ($\kappa=0.990$) with the independent TCGA reclassification of de Mendon\c{c}a et al.~\cite{mendonca2025updating}, with all five disagreements reflecting differences in the IDH source. "
    r"When the target is tumour grade, by contrast, the limit is mainly missing histological information in the predictor set. The 23 non-codeleted IDH-mutant legacy GBM with 1p/19q data are astrocytomas of CNS WHO grade~4, so their high-grade label is correct, and 21 of the 25 IDH-mutant legacy GBM lack \gene{CDKN2A/B} deletion and are grade~4 on histology alone (necrosis or microvascular proliferation)."
)
new_d309 = (
    r"The reconstructed tumour-type assignments also agreed in 99.4\% of mutually assigned patients ($\kappa=0.990$) with the TCGA reclassification of de Mendon\c{c}a et al.~\cite{mendonca2025updating} "
    r"(which draws on the same underlying Ceccarelli et al.~\cite{ceccarelli2016molecular} annotations, thereby confirming algorithmic rule consistency rather than independent histopathological validation), with all five disagreements reflecting differences in the IDH source. "
    r"When the target is tumour grade, by contrast, the limit is mainly missing histological information in the predictor set. Treating the legacy GBM label as a proxy for histological microvascular proliferation or necrosis, "
    r"the 23 assessable non-codeleted IDH-mutant legacy GBM map to astrocytoma, CNS WHO grade~4, and 21 of the 25 IDH-mutant legacy GBM lack \gene{CDKN2A/B} deep deletion and are assigned grade~4 on legacy histology alone."
)
tex = replace_exact(tex, old_d309, new_d309, "Discussion D309")

old_d313_tail = (
    r"Reported accuracies of 85.7--87.6\%~\cite{tasci2022hierarchical,mohamed2025combinations}, like the 85.1\% observed here at the locked threshold (85.7\% at 0.5; 87.2\% in cross-validation), are close to the 86.9\% label--IDH agreement."
)
new_d313_tail = (
    r"Reported accuracies of 85.7--87.6\%~\cite{tasci2022hierarchical,mohamed2025combinations}, matching the 87.2\% out-of-fold cross-validation accuracy (87.5\% under fully nested joint selection; 85.1\% at the locked threshold on the exploratory test set, 85.7\% at 0.5), are close to the 86.9\% label--IDH agreement."
)
tex = replace_exact(tex, old_d313_tail, new_d313_tail, "Discussion D313a")

old_d313_end = (
    r"Because between-family differences (0.006 AUC) are smaller than split-to-split variation (0.903--0.921), this benchmark should not be used to rank classifiers; studies using it should stratify accuracy by label--IDH concordance, include the \gene{IDH1/2} rule baseline, and evaluate against current categories (Supplementary Data~1)."
)
new_d313_end = (
    r"Because differences among the four tested tabular classifier families on these 23 features (0.006 mean-of-fold AUC) were smaller than split-to-split variation across repeated nested cross-validation (0.903--0.921 pooled AUC), "
    r"small apparent gains on this single-cohort benchmark cannot reliably distinguish algorithm performance from sampling noise; future computational studies using this dataset should stratify accuracy by label--IDH concordance, benchmark against the \gene{IDH1/2} rule baseline, and evaluate against updated WHO 2021 categories (Supplementary Data~1)."
)
tex = replace_exact(tex, old_d313_end, new_d313_end, "Discussion D313b")

old_d315 = (
    r"Clinically, the findings argue against a role for a 20-gene mutation model as a triage tool where histology and IDH1 R132H immunohistochemistry are available. "
    r"The gain over age and \gene{IDH1} was 0.015 AUC in development and a non-significant 0.023 on the test set, with no test-set net-benefit advantage over age and \gene{IDH1} or the IDH rule. "
    r"Both the model and this comparator used sequenced \gene{IDH1} status. Immunohistochemistry detects only R132H, about 93\% of \gene{IDH1} mutations~\cite{hartmann2009type}, and sequencing of \gene{IDH1} codon 132 and \gene{IDH2} codon 172 is recommended when it is negative in grade 2--3 diffuse gliomas and in glioblastomas of patients younger than 55~years~\cite{weller2021glioblastoma}, consistent with the selection of \gene{IDH2} in the four-feature panel. "
    r"A clinically useful model would need to be trained against WHO 2021 integrated diagnoses and to include the markers that define them, such as \gene{TERT} promoter status, \gene{EGFR} amplification, +7/$-$10, 1p/19q and \gene{CDKN2A/B}~\cite{louis2021who,brat2018cimpact,brat2020cimpact5}."
)
new_d315 = (
    r"From a translational perspective, these findings indicate that the 23-feature multivariable classifier offers no predictive advantage over IDH status alone and is not suitable for clinical implementation or triage. "
    r"Across both development cross-validation and the exploratory test set, the model did not significantly outperform the zero-parameter \gene{IDH1/2} rule in classification accuracy or net benefit, "
    r"and its modest discrimination gain over age plus \gene{IDH1} in development ($\Delta\text{AUC}=0.015$, $p=0.037$) was not statistically significant on the exploratory test set ($\Delta\text{AUC}=0.023$, $p=0.17$). "
    r"Moreover, all models were trained on a single retrospective cohort using sequencing-derived \gene{IDH1}/\gene{IDH2} calls without independent external validation. "
    r"In this computational benchmark, adding 18 somatic mutation calls and demographic variables provides no meaningful predictive increment over IDH status alone. "
    r"This study is strictly a methodological audit of a public machine-learning benchmark and must not be interpreted as supporting any clinical application, diagnostic workflow or patient-management decision; "
    r"any future model intended for neuro-oncology research would need to be trained and externally validated directly against integrated WHO 2021 diagnoses using complete molecular and histopathological features~\cite{louis2021who,brat2018cimpact,brat2020cimpact5}."
)
tex = replace_exact(tex, old_d315, new_d315, "Discussion D315")

old_d317_mid = (
    r"Furthermore, the WHO 2021 reconstruction is approximate and was conducted post hoc: histological grade~4 features were inferred from the legacy GBM diagnosis, GISTIC calls approximate amplification and homozygous deletion, \gene{TERT} promoter status was missing for a subset of patients, methylation profiling was not used for classification, and the rules were fixed after the test set had been scored; accordingly, this study reports a range of rule variants. "
    r"A recent cIMPACT-NOW proposal also advises caution in calling a histologically low-grade tumour glioblastoma on a \gene{TERT} promoter mutation alone, and against using \gene{EGFR} amplification or +7/$-$10 as the sole defining feature in patients younger than 40~years~\cite{wesseling2026cimpact11}. "
    r"Test-set results from earlier versions of this analysis had been seen and the present partition had been scored once before; although all decisions in this version were locked on the training partition first, the test estimate is not fully independent, and the 70 test events limit its precision. Repeated nested cross-validation, which never uses the test set, gave similar results."
)
new_d317_mid = (
    r"Furthermore, the WHO 2021 reconstruction is an approximate, post hoc research reclassification rather than an integrated clinical pathology diagnosis: central slide review for microvascular proliferation, necrosis and mitotic index was unavailable "
    r"(so histological grade~4 features were inferred solely from the legacy TCGA-GBM project label, and grade~2 could not be separated from grade~3); "
    r"SNP~6.0 GISTIC~2.0 discrete calls ($+2$/$-2$) and arm-level summaries approximate formal WHO 2021 definitions for whole-arm 1p/19q co-deletion, \gene{EGFR} amplification, $+7/{-}10$ and homozygous \gene{CDKN2A/B} deletion without tumour-purity deconvolution "
    r"(71 IDH-mutant non-codeleted legacy LGG carried GISTIC $-1$ hemizygous \gene{CDKN2A/B} loss); \gene{TERT} promoter status was missing in 529 of 839 patients (including 26 of 68 molecular glioblastomas and 9 unassessable IDH-wildtype legacy LGG); "
    r"anatomical tumour location and genome-wide DNA-methylation classifier scores were unavailable (3 of 68 molecular glioblastomas and 11 of 17 unresolved IDH-wildtype legacy LGG carried a pilocytic-astrocytoma-like methylation cluster; Supplementary Table~S34d); and the reconstruction rules were formulated post hoc. "
    r"These uncertainties are bounded by reporting ten rule variants alongside the cIMPACT-NOW update~11 cautious~\cite{wesseling2026cimpact11} and $\geq$2-criteria sensitivity analyses "
    r"(51.8--84.0\% molecular glioblastomas among IDH-wildtype legacy LGG; 48.0--79.2\% of test errors and 64.0--94.8\% of development errors agreeing with reconstructed tumour type; Supplementary Tables~S26--S28 and S38). "
    r"Test-set results from earlier versions of this analysis had been seen and the present partition had been scored once before; although all pipeline decisions in this canonical run were locked on the development partition first, the held-out test set ($n=168$; 70 events) is not strictly independent, "
    r"so 10-times-repeated 5-fold nested cross-validation on the development partition ($n=671$) serves as the primary internal performance assessment and all test-set results are interpreted as exploratory."
)
tex = replace_exact(tex, old_d317_mid, new_d317_mid, "Discussion D317")

# 10. Conclusion (L322)
old_conc = (
    r"This study demonstrates that apparent classifier performance on the widely used UCI/TCGA glioma grading benchmark is governed by diagnostic label--IDH discordance rather than learning-algorithm capacity. "
    r"Across 839 complete-case patients, 13.1\% of legacy WHO 2007 histological labels disagree with \gene{IDH1}/\gene{IDH2} mutation status: 68 of 85 IDH-wildtype legacy LGG (80.0\%; 51.8--84.0\% across sensitivity variants) meet WHO 2021 molecular criteria for glioblastoma, IDH-wildtype, "
    r"whereas all 23 assessable non-codeleted IDH-mutant legacy GBM are CNS WHO grade~4 astrocytomas (21 of 25 graded on histology alone without \gene{CDKN2A/B} deletion). "
    r"Under leakage-free nested cross-validation, four classifier families performed similarly, and the locked 23-feature XGBoost model (test AUC 0.932; 25 errors in 168 patients) did not outperform the zero-parameter rule ``IDH-wildtype implies GBM'' (22 errors; McNemar $p=0.25$), "
    r"with 84 of 86 development errors and 22 of 25 test errors concentrated in discordant cases. Adding WHO 2021 copy-number and 1p/19q features did not materially increase legacy-label accuracy. "
    r"Future studies should retire this legacy target as an algorithm-ranking benchmark, evaluate predictions against integrated WHO 2021 categories using the released patient-level annotations (Supplementary Data~1), and train clinical decision-support models directly on current molecular and histological reference standards."
)
new_conc = (
    r"This computational audit demonstrates that on the widely used UCI/TCGA glioma grading benchmark, apparent classifier performance across standard tabular algorithms is primarily constrained by diagnostic label--IDH discordance and unmeasured histological grading criteria rather than differences among the tested model families. "
    r"Across 839 complete-case patients, 13.1\% of legacy WHO 2007 histological labels disagree with \gene{IDH1}/\gene{IDH2} mutation status: in a post hoc reconstruction from published TCGA annotations, "
    r"68 of 85 IDH-wildtype legacy LGG (80.0\%; 51.8--84.0\% across sensitivity variants) met approximate WHO 2021 molecular criteria for glioblastoma, IDH-wildtype, "
    r"whereas all 23 assessable non-codeleted IDH-mutant legacy GBM mapped to CNS WHO grade~4 astrocytoma (21 of 25 via the legacy histological label alone without \gene{CDKN2A/B} deep deletion). "
    r"In primary internal validation using 10-times-repeated 5-fold nested cross-validation on the development partition ($n=671$), the four tested classifier families performed similarly "
    r"(XGBoost pooled AUC $0.913\pm 0.005$; canonical nested-CV AUC 0.912, 86 errors versus 88 for the zero-parameter rule ``IDH-wildtype implies GBM'', McNemar $p=0.69$; fully nested joint inner-loop selection mean-of-fold AUC $0.908\pm 0.023$, pooled AUC 0.905), "
    r"and exploratory evaluation on the held-out test set ($n=168$; AUC 0.932, 25 errors versus 22 rule errors, $p=0.25$) likewise showed no advantage over the simple \gene{IDH1/2} rule, "
    r"with 84 of 86 development errors and 22 of 25 exploratory test errors concentrated in discordant cases. Adding copy-number and 1p/19q features did not materially increase legacy-label accuracy. "
    r"Because the multivariable model offers no predictive or net-benefit improvement over IDH status alone and lacks independent external validation, it carries no clinical utility; "
    r"future machine-learning studies on this benchmark should stratify accuracy by label--IDH concordance, compare against the \gene{IDH1/2} rule, and evaluate predictions against updated WHO 2021 annotations (Supplementary Data~1)."
)
tex = replace_exact(tex, old_conc, new_conc, "Conclusion")

# 11. Methods expansions
old_m337 = (
    r"For the reconstruction, each patient was linked by case identifier to the per-patient annotations of Ceccarelli et al.~\cite{ceccarelli2016molecular} "
    r"(cBioPortal study \texttt{lgggbm\_tcga\_pub}~\cite{cerami2012cbio}, datahub commit 18763f8): 1p/19q co-deletion, +7/$-$10, \gene{TERT} promoter status, "
    r"GISTIC copy-number calls for \gene{EGFR} (high-level amplification, +2) and \gene{CDKN2A}/\gene{CDKN2B} (deep deletion, $-2$), curated IDH status, supervised DNA-methylation cluster, and exome mutation calls for IDH and H3 K27/G34 hotspots (764 of the 839 patients were exome-sequenced)."
)
new_m337 = (
    r"For the post hoc reconstruction, each patient was linked by case identifier to the per-patient annotations of Ceccarelli et al.~\cite{ceccarelli2016molecular} "
    r"(cBioPortal study \texttt{lgggbm\_tcga\_pub}~\cite{cerami2012cbio}, datahub commit 18763f8): 1p/19q co-deletion (\texttt{IDH\_1P19Q\_SUBTYPE}), combined chromosome~7 gain and chromosome~10 loss (\texttt{CHR\_7\_GAIN\_CHR\_10\_LOSS}), "
    r"\gene{TERT} promoter status, GISTIC~2.0 discrete calls for \gene{EGFR} (high-level amplification, $+2$, excluding low-level gain $+1$) and \gene{CDKN2A}/\gene{CDKN2B} (deep deletion, $-2$, excluding single-copy loss $-1$), "
    r"curated IDH status, supervised DNA-methylation cluster, and exome mutation calls for IDH and H3 K27/G34 hotspots (764 of the 839 patients were exome-sequenced). "
    r"These research-grade annotations approximate, rather than replicate, the clinical assay definitions required by WHO 2021~\cite{louis2021who,brat2018cimpact,brat2020cimpact5}. "
    r"Specifically, whereas WHO 2021 defines oligodendroglioma by complete whole-arm 1p/19q co-deletion mediated by an unbalanced centromeric translocation $\mathrm{t}(1;19)(\mathrm{q10;p10})$, "
    r"the Ceccarelli et al.\ \texttt{Codel} call was derived from genome-wide Affymetrix SNP~6.0 array segmentation and GISTIC~2.0 arm-level analysis across all 162 codeleted cases (\texttt{SNP6 = Yes}); "
    r"this genome-wide array calling separates whole-arm 1p/19q loss from focal deletions that can cause false-positive fluorescence in situ hybridisation results "
    r"(96.3\% and 98.1\% of \texttt{Codel} tumours were \gene{TP53}- and \gene{ATRX}-wildtype, respectively; Supplementary Table~S34c), although raw segment coordinates are not exposed in the collapsed cBioPortal variable. "
    r"Likewise, GISTIC~2.0 $+2$ and $-2$ calls rely on sample-adaptive bulk-DNA $\log_2$ copy-number ratio thresholds without tumour-purity deconvolution, "
    r"and \texttt{CHR\_7\_GAIN\_CHR\_10\_LOSS} summarises arm-level SNP~6.0 calls without distinguishing whole-chromosome $+7/{-}10$ from partial-arm combinations."
)
tex = replace_exact(tex, old_m337, new_m337, "Methods M337")

old_m342 = (
    r"All model development, selection, calibration, threshold choice and panel selection used the training partition only. Before the test set was scored, the five locked decisions (imbalance strategy, primary model, calibration method, threshold and four-feature panel) were written to a file (Supplementary Table~S22), and the final model's family was checked against the lock programmatically."
)
new_m342 = (
    r"All model development, selection, calibration, threshold choice and panel selection used the development partition ($n=671$) only. Before the exploratory test evaluation was executed, the five locked decisions (imbalance strategy, primary model, calibration method, threshold and four-feature panel) were written to a file (Supplementary Table~S22), and the final model's family was checked against the lock programmatically. "
    r"Because test-set results from earlier versions of this analysis had been examined during revision (including an initial 838-patient split and an intermediate run on the present 671/168 split), the held-out test set ($n=168$) cannot be treated as strictly independent validation; "
    r"10-times-repeated 5-fold nested cross-validation on the development partition is therefore designated as the primary internal performance assessment, and all test-set results are interpreted as exploratory. "
    r"Similarly, all WHO 2021 reconstruction rules and sensitivity variants were formulated post hoc after initial error patterns had been inspected."
)
tex = replace_exact(tex, old_m342, new_m342, "Methods M342")

old_m348 = (
    r"The pre-specified rule locked the family with the highest mean-of-fold AUC, with ties broken by mean-of-fold Brier score; pooled out-of-fold AUC was reported alongside."
)
new_m348 = (
    r"In the pre-specified two-stage protocol, the imbalance strategy was selected on the development partition first and the primary family (XGBoost) was designated by highest mean-of-fold outer AUC (with ties broken by mean-of-fold Brier score); pooled out-of-fold AUC was reported alongside. "
    r"To eliminate any potential selection optimism from screening imbalance strategies on the full development partition or selecting the model family across outer folds, this study additionally implemented a fully nested inner-loop joint selection procedure: "
    r"within each outer training fold, all 12 combinations of the three valid imbalance strategies (no resampling, fold-specific class weighting with $\text{scale\_pos\_weight}=n_{\text{LGG,fold}}/n_{\text{GBM,fold}}$, and SMOTENC) and four model families were tuned via inner 5-fold cross-validation, and the single winning pipeline was scored once on that outer validation fold."
)
tex = replace_exact(tex, old_m348, new_m348, "Methods M348")

tex = replace_exact(
    tex,
    "For the test set, the locked recalibrator was refitted once on all training out-of-fold probabilities",
    "For the exploratory test set, the locked recalibrator (Platt scaling) was refitted once on all training out-of-fold probabilities",
    "Methods M354"
)
tex = replace_exact(
    tex,
    r"The final primary model, the age plus \gene{IDH1} logistic regression and the four-feature panel model were each tuned by 5-fold randomised search on the full training partition, refitted, and scored once on the test set, together with the no-fit rules.",
    r"For the exploratory test-set evaluation, the final primary model, the age plus \gene{IDH1} logistic regression and the four-feature panel model were each tuned by 5-fold randomised search on the full development partition, refitted, and scored once on the held-out test set, together with the zero-parameter rules.",
    "Methods M360"
)
tex = replace_exact(
    tex,
    "oracle accuracy at the best threshold selected on the out-of-fold predictions (an upper bound)",
    "oracle accuracy at the best threshold selected on the out-of-fold predictions (an upper bound that uses the evaluated predictions to choose the threshold)",
    "Methods M363"
)
old_m366 = r"The patient-level analysis and reconstruction are released as a versioned label file (version 1.0; 839 rows, 53 columns) with a column dictionary and a datasheet:"
new_m366 = (
    r"The patient-level analysis and post hoc WHO 2021 reconstruction are released as a self-contained, versioned label file (\texttt{Supplementary Data~1}; version 1.0; 839 rows, 64 columns) "
    r"accompanied by a column dictionary and datasheet~\cite{gebru2021datasheets}, including explicit patient-level flags for all reconstruction variants, qualifying molecular glioblastoma criteria counts, "
    r"the \texttt{cIMPACT-NOW} update~11 cautious rule, two-criterion flags, and the de Mendon\c{c}a et al.\ comparison:"
)
tex = replace_exact(tex, old_m366, new_m366, "Methods M366")

tex_path.write_text(tex, encoding="utf-8")
print("Updated 1_Manuscript/Glioma.tex successfully.")
