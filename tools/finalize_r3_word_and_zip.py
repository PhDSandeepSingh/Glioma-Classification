"""
1. Generate plain black-and-white 4_Response/Response_to_Reviewers_R5.docx
   addressing Reviewer 5, Reviewer 6, Reviewer 7, and Reviewer 8 (third-person only).
2. Update 5_Reporting_checklists/TRIPOD_AI_Checklist.docx (plain black-and-white).
3. Rebuild 6_Code/Glioma-Classification-v3.zip.
"""
import os, re, shutil, zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from pathlib import Path

BASE = Path("/Users/sandy/Downloads/Research Work/Glioma/Revision 2/Submission Ready/Final")
TITLE = "Impact of diagnostic label–IDH discordance and WHO 2021 criteria on machine-learning glioma grading from somatic mutations"
MS_ID = "2170ba8b-ed1b-40d4-81d2-1ffb551ee388"

# ── 1. Build Response_to_Reviewers_R5.docx cloned from R4's clean XML skeleton ──
r4_path = BASE / "4_Response/Response_to_Reviewers_R4.docx"
r5_path = BASE / "4_Response/Response_to_Reviewers_R5.docx"

sections = [
    ("heading", "Response to Reviewers — Round 3 Revision"),
    ("bold_kv", ("Journal: ", "Scientific Reports")),
    ("bold_kv", ("Manuscript ID: ", MS_ID)),
    ("bold_kv", ("Manuscript: ", TITLE)),
    ("normal", "The authors thank the Editorial Board and Reviewers 5, 6, 7, and 8 for their constructive evaluation of the revised manuscript. Every comment has been addressed in full as detailed below, with all text and numerical updates verified across the main manuscript, Figure 1, Supplementary Information, Supplementary Data 1, and the TRIPOD+AI checklist."),

    ("h2", "Reviewer 5"),
    ("quote", "Comment: \"The Authors have positively provided response to my comments hence manuscript can be accepted for publication.\""),
    ("normal", "Response: The authors thank Reviewer 5 for their careful assessment across the revision rounds and for recommending acceptance of the manuscript."),

    ("h2", "Reviewer 6"),
    ("quote", "Major Comment 1 (Potential information leakage during model selection): \"Although nested cross-validation was performed, the imbalance-handling strategy was selected using cross-validation across the entire training partition before the outer-fold evaluation. In addition, the primary model family was selected based on performance across the outer folds. These decisions may introduce selection bias into the reported out-of-fold performance estimates. The authors should clarify this procedure and, ideally, incorporate all model-selection decisions within the nested validation framework.\""),
    ("normal", "Response: The authors agree that screening imbalance strategies across the full development partition (n = 671) and designating the primary model family after comparing outer-fold AUCs can introduce selection optimism. In addition to clarifying this two-stage development procedure, the authors implemented a fully nested inner-loop joint selection procedure inside each outer training fold: all 12 combinations of the three valid imbalance strategies (no resampling, fold-specific class weighting with scale_pos_weight = n_LGG,fold / n_GBM,fold, and SMOTENC) and four model families (logistic regression, RBF SVM, random forest, XGBoost) were tuned strictly via inner 5-fold cross-validation before scoring the single winning pipeline once on that outer validation fold. This completely unbiased estimator achieved a mean-of-fold AUC of 0.908 ± 0.023 and a pooled out-of-fold AUC of 0.905 (87.5% accuracy, 84 errors at 0.5; selecting SMOTENC + random forest, class-weighted logistic regression, SMOTENC + XGBoost, class-weighted XGBoost, and SMOTENC + logistic regression across the five outer folds), confirming that selection optimism in the two-stage pipeline was minimal (<= 0.007 AUC) and that performance is bounded by the underlying features rather than classifier choice."),
    ("ref", "Reference: Abstract; Results ('Model families perform similarly and add little to age and IDH1' and Table 2); Methods ('Model families and nested cross-validation'); Figure 1; 6_Code/tools/run_fully_nested_joint.py."),

    ("quote", "Major Comment 2 (Independence of the test set): \"The authors acknowledge that the test set had been examined in earlier versions of the analysis. Therefore, the current test-set results cannot be considered fully independent validation. The authors should emphasize repeated nested cross-validation as the primary internal performance assessment and interpret the test-set results as exploratory. This limitation should also be clearly reflected in the Abstract and Conclusion.\""),
    ("normal", "Response: The authors have elevated 10-times-repeated 5-fold nested cross-validation on the development partition (n = 671; pooled out-of-fold AUC 0.913 ± 0.005, range 0.903–0.921; canonical nested-CV AUC 0.912) as the primary internal performance assessment across the Abstract, Introduction, Results (moved into 'Model families perform similarly and add little to age and IDH1' and Table 2), Discussion, Conclusion, Methods, and Figure 1. All held-out test-set results (n = 168; 98 legacy LGG, 70 legacy GBM) are now explicitly framed as exploratory due to prior exposure in earlier revision rounds, including in the Abstract, subsection heading ('Exploratory evaluation on the held-out test set'), Table 3 caption, Figure 4 caption, Discussion, and Conclusion."),
    ("ref", "Reference: Abstract; Introduction; Results (Table 2, 'Exploratory evaluation on the held-out test set', Table 3, Figure 4); Discussion; Conclusion; Methods ('Data partition and locking of decisions'); Figure 1."),

    ("quote", "Minor Comment 1 (WHO 2021 molecular definitions and whole-arm 1p/19q codeletion): \"Please clarify whether the 1p/19q co-deletion and copy-number alterations used in the reconstruction represent the molecular definitions required by WHO 2021, particularly whole-arm 1p/19q codeletion.\""),
    ("normal", "Response: The authors have expanded Methods ('Label–IDH analysis and WHO 2021 reconstruction'), Discussion (limitations paragraph), and Data/DATASHEET.md to clarify how each annotation maps to formal WHO 2021 criteria. Specifically, whereas WHO 2021 defines oligodendroglioma by complete whole-arm 1p/19q co-deletion mediated by an unbalanced centromeric translocation t(1;19)(q10;p10), the Ceccarelli et al. (2016) Codel call was derived from genome-wide Affymetrix SNP 6.0 array segmentation and GISTIC 2.0 arm-level analysis across all 162 codeleted cases (SNP6 = Yes; 96.3% TP53-wildtype and 98.1% ATRX-wildtype), which separates whole-arm loss from focal deletions that can yield false-positive FISH results, though raw segment coordinates are not exposed in the collapsed cBioPortal variable. Likewise, GISTIC 2.0 discrete calls (+2 for EGFR amplification, excluding +1; -2 for CDKN2A/B deep deletion, excluding -1 hemizygous loss present in 71 IDH-mutant non-codeleted legacy LGG) rely on sample-adaptive bulk-DNA log2 thresholds without tumour-purity deconvolution, and CHR_7_GAIN_CHR_10_LOSS summarises arm-level SNP 6.0 calls without distinguishing whole-chromosome +7/-10 from partial-arm combinations."),
    ("ref", "Reference: Methods ('Label–IDH analysis and WHO 2021 reconstruction'); Discussion (limitations paragraph); 6_Code/Data/DATASHEET.md."),

    ("quote", "Minor Comment 2 (Moderating classifier-choice claims): \"The authors should further moderate statements suggesting that classifier choice does not limit benchmark accuracy, given that the analysis is based on a single dataset and lacks independent external validation.\""),
    ("normal", "Response: The authors have moderated all classifier-choice statements in the Abstract, Discussion, and Conclusion so they are explicitly scoped to the four tested tabular classifier families on the 23-feature representation of this single-cohort retrospective benchmark without independent external validation."),
    ("ref", "Reference: Abstract; Discussion (opening paragraph and benchmark-implications paragraph); Conclusion."),

    ("h2", "Reviewer 7"),
    ("quote", "Comments 1–4: \"First, the authors acknowledge that the test set had been examined in earlier versions of the analysis, which compromises its independence... Second, the reconstructed WHO 2021 categories are approximate and rely on incomplete molecular and histological information... Third, the lack of independent external validation restricts the generalizability of the findings. The authors should clarify these limitations, strengthen the interpretation of the sensitivity analyses, and avoid overstating the clinical implications. Additionally, inconsistencies in the sample numbers reported in Figure 1 should be corrected.\""),
    ("normal", "Response: All four points have been fully addressed: (1) Repeated 10x5-fold nested cross-validation is elevated as the primary internal assessment and the held-out test set is framed as exploratory throughout the manuscript; (2) Results, Discussion, and Conclusion now explicitly detail every missing histological and molecular element (absence of central slide review for microvascular proliferation, necrosis, and mitotic index, with 21 of 25 IDH-mutant legacy GBM relying on the legacy TCGA-GBM label alone; 71 IDH-mutant legacy LGG with GISTIC -1 hemizygous CDKN2A/B loss; 529 of 839 patients missing TERT promoter status; and absence of DNA-methylation classifier scores or anatomical tumour location), explain how each sensitivity variant (b, c, cIMPACT-NOW update 11 cautious, >=2-criteria, h0/h1) bounds a distinct source of uncertainty, qualify the 99.4% concordance with de Mendonça et al. (2025) as algorithmic rule consistency on shared Ceccarelli/TCGA annotations, and fix two citation typos in Supplementary Tables S37 and S38; (3) the manuscript is framed strictly as a computational benchmark audit with no clinical utility or implementation role, given single-cohort provenance, lack of external validation, and no predictive increment over the IDH1/2 rule; and (4) Figure 1 has been regenerated programmatically at 600 DPI (plus vector PDF) with the corrected test-set count '98 legacy LGG / 70 legacy GBM' and corrected labels ('4-feature panel', 'IDH1 and IDH1/2', 'TERT data')."),
    ("ref", "Reference: Figure 1; Abstract; Results ('Most IDH-wildtype legacy LGG meet molecular criteria for glioblastoma'); Discussion; Conclusion; Supplementary Tables S37–S38."),

    ("h2", "Reviewer 8"),
    ("quote", "Comment 1: \"Correct the test-set sample discrepancy (Figure 1: 98 LGG + 20 GBM versus 98 LGG + 70 GBM in the text).\""),
    ("normal", "Response: Figure 1 (Figure00_StudyDesign.pdf, Figure00_StudyDesign.png, and Figure1.png) has been regenerated programmatically at 600 DPI with the corrected label '98 legacy LGG / 70 legacy GBM', alongside fixing minor typographical errors ('4-feature panel', 'IDH1 and IDH1/2', 'TERT data')."),
    ("ref", "Reference: Figure 1 (1_Manuscript/Figures/Figure00_StudyDesign.pdf and 2_Figures_for_upload/Figure1.png)."),

    ("quote", "Comment 2: \"Clarify that WHO 2021 reclassification was a post-hoc analysis and acknowledge prior exposure to test-set results.\""),
    ("normal", "Response: Both disclosures are now explicitly stated in the Abstract, Introduction, Figure 1 caption, Results, Discussion, Conclusion, and Methods ('Data partition and locking of decisions')."),
    ("ref", "Reference: Abstract; Introduction; Figure 1 caption; Results ('Exploratory evaluation on the held-out test set'); Discussion; Conclusion; Methods."),

    ("quote", "Comment 3: \"Revise the abstract and conclusion to avoid overstating clinical utility, as the model did not significantly outperform the IDH1/2-based rule.\""),
    ("normal", "Response: The Abstract, Results (out-of-fold net-benefit comparison against the IDH1/2 rule), Discussion, and Conclusion have been revised to state clearly that the multivariable model offered no predictive or net-benefit advantage over the zero-parameter IDH1/2 rule (McNemar p = 0.69 in development; p = 0.25 on the exploratory test set) and carries no clinical utility or triage implementation role."),
    ("ref", "Reference: Abstract; Results ('Calibration and operating threshold'); Discussion; Conclusion."),

    ("quote", "Comment 4: \"Ensure that the WHO 2021 classification criteria and patient-level annotations are adequately documented for reproducibility.\""),
    ("normal", "Response: Supplementary Data 1 (and its release CSV and DATASHEET.md) has been expanded from 53 to 64 columns across all 839 patients by adding Age_at_diagnosis, GRADE_Ceccarelli, SNP6_array, WHO_grade4_d2, WHO_grade4_h1, WHO_GBM_IDHwt_h1, WHO_grade4_h0, WHO_GBM_IDHwt_h0, MolGBM_qualifying_criteria_count, WHO2021_cIMPACT11_cautious, and MONET_Method2021_category, so every sensitivity analysis and external comparison can be reproduced directly from the single released spreadsheet without external joins."),
    ("ref", "Reference: Methods ('Label release'); 3_Supplementary/Supplementary_Data_1.xlsx; 6_Code/Data/DATASHEET.md.")
]

def build_p_xml(kind, content):
    if kind == "heading":
        return f'<w:p><w:pPr><w:spacing w:after="160"/></w:pPr><w:r><w:rPr><w:b/><w:color w:val="000000"/><w:sz w:val="28"/></w:rPr><w:t>{escape(content)}</w:t></w:r></w:p>'
    if kind == "h2":
        return f'<w:p><w:pPr><w:spacing w:before="200" w:after="100"/></w:pPr><w:r><w:rPr><w:b/><w:color w:val="000000"/><w:sz w:val="24"/></w:rPr><w:t>{escape(content)}</w:t></w:r></w:p>'
    if kind == "bold_kv":
        k, v = content
        return f'<w:p><w:pPr><w:spacing w:after="80"/></w:pPr><w:r><w:rPr><w:b/><w:color w:val="000000"/><w:sz w:val="21"/></w:rPr><w:t xml:space="preserve">{escape(k)}</w:t></w:r><w:r><w:rPr><w:color w:val="000000"/><w:sz w:val="21"/></w:rPr><w:t>{escape(v)}</w:t></w:r></w:p>'
    if kind == "quote":
        return f'<w:p><w:pPr><w:ind w:left="360"/><w:spacing w:before="80" w:after="80"/></w:pPr><w:r><w:rPr><w:i/><w:color w:val="000000"/><w:sz w:val="20"/></w:rPr><w:t>{escape(content)}</w:t></w:r></w:p>'
    if kind == "ref":
        return f'<w:p><w:pPr><w:spacing w:after="120"/></w:pPr><w:r><w:rPr><w:b/><w:color w:val="000000"/><w:sz w:val="20"/></w:rPr><w:t>{escape(content)}</w:t></w:r></w:p>'
    return f'<w:p><w:pPr><w:spacing w:after="100"/></w:pPr><w:r><w:rPr><w:color w:val="000000"/><w:sz w:val="21"/></w:rPr><w:t>{escape(content)}</w:t></w:r></w:p>'

tmp_r5 = Path("/tmp/r5_docx")
if tmp_r5.exists(): shutil.rmtree(tmp_r5)
with zipfile.ZipFile(r4_path, "r") as z:
    z.extractall(tmp_r5)

doc_xml = (tmp_r5 / "word/document.xml").read_text(encoding="utf-8")
m_sect = re.search(r"(<w:sectPr\b.*</w:sectPr>)", doc_xml, flags=re.DOTALL)
sect_pr = m_sect.group(1) if m_sect else "<w:sectPr/>"
header_open = doc_xml[:doc_xml.index("<w:body>") + len("<w:body>")]
body_paras = "\n".join(build_p_xml(k, c) for k, c in sections)
new_doc_xml = header_open + body_paras + sect_pr + "</w:body></w:document>"
(tmp_r5 / "word/document.xml").write_text(new_doc_xml, encoding="utf-8")

with zipfile.ZipFile(r5_path, "w", zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk(tmp_r5):
        for f in files:
            fp = os.path.join(root, f)
            zout.write(fp, os.path.relpath(fp, tmp_r5))
shutil.rmtree(tmp_r5)
print("Created plain black-and-white Response_to_Reviewers_R5.docx")

# ── 2. Update TRIPOD_AI_Checklist.docx (plain black-and-white) ──
chk_path = BASE / "5_Reporting_checklists/TRIPOD_AI_Checklist.docx"
tmp_chk = Path("/tmp/chk_r5")
if tmp_chk.exists(): shutil.rmtree(tmp_chk)
with zipfile.ZipFile(chk_path, "r") as z:
    z.extractall(tmp_chk)

chk_xml_path = tmp_chk / "word/document.xml"
chk_xml = chk_xml_path.read_text(encoding="utf-8")

repls = [
    (
        "The study develops a model (nested cross-validation on the training partition, n = 671, 282 events) and evaluates it on a random held-out partition of the same source (n = 168, 70 events).",
        "The study evaluates four model families using 10-times-repeated 5-fold nested cross-validation and fully nested inner-loop joint selection on the development partition (n = 671, 282 events) as primary internal validation, and reports evaluation on a held-out partition of the same source (n = 168, 70 events) as exploratory due to prior revision exposure."
    ),
    (
        "Reported on p. 1 (Abstract): structured summary including dataset, label-IDH discordance, WHO 2021 reconstruction, design (decisions locked by nested cross-validation, held-out test set), test AUC, errors against a no-fit IDH rule with McNemar test, and conclusions.",
        "Reported on p. 1 (Abstract): structured summary including dataset, label-IDH discordance, post hoc approximate WHO 2021 reconstruction, primary 10x5 repeated nested CV and fully nested joint selection estimates, exploratory held-out test-set results (prior exposure disclosed), comparison against the zero-parameter IDH rule, and tempered conclusions noting no predictive or clinical utility advantage over IDH status alone."
    ),
    (
        "The abstract reports the source, cohort size, outcome, predictors, primary validation metric, comparator rule, and main conclusion.",
        "The abstract reports the source, cohort size, outcome, predictors, primary repeated nested CV metric, exploratory test-set results, post hoc WHO 2021 reconstruction, comparator rule, and tempered conclusion."
    ),
    (
        "53 columns", "64 columns"
    ),
    (
        "53-column", "64-column"
    ),
]
for old_s, new_s in repls:
    chk_xml = chk_xml.replace(old_s, new_s)

# Strip any shading/colors/pBdr
chk_xml = re.sub(r"<w:shd\b[^>]*/?>", "", chk_xml)
chk_xml = re.sub(r"<w:pBdr>.*?</w:pBdr>", "", chk_xml, flags=re.DOTALL)
chk_xml = re.sub(r"<w:color\s+w:val=\"(?!000000|auto)[0-9A-Fa-f]+\"[^>]*/?>", "<w:color w:val=\"000000\"/>", chk_xml)
chk_xml_path.write_text(chk_xml, encoding="utf-8")

with zipfile.ZipFile(chk_path, "w", zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk(tmp_chk):
        for f in files:
            fp = os.path.join(root, f)
            zout.write(fp, os.path.relpath(fp, tmp_chk))
shutil.rmtree(tmp_chk)
print("Updated plain black-and-white TRIPOD_AI_Checklist.docx")

# ── 3. Rebuild 6_Code/Glioma-Classification-v3.zip ──
zip_path = BASE / "6_Code/Glioma-Classification-v3.zip"
code_dir = BASE / "6_Code"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d != "__pycache__"]
        for f in sorted(files):
            if f.startswith(".") or f == "Glioma-Classification-v3.zip":
                continue
            fp = Path(root) / f
            zf.write(fp, Path("Glioma-Classification-v3") / fp.relative_to(code_dir))
print("Rebuilt 6_Code/Glioma-Classification-v3.zip:", zip_path.stat().st_size, "bytes")
