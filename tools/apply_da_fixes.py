import os, re, shutil, zipfile
from pathlib import Path

BASE = Path("/Users/sandy/Downloads/Research Work/Glioma/Revision 2/Submission Ready/Final")

# ── 1. DA-3: Table 2 footnote b in 1_Manuscript/Glioma.tex ──
g_path = BASE / "1_Manuscript/Glioma.tex"
g = g_path.read_text(encoding="utf-8")
if "\\textsuperscript{b}Fully nested joint selector" not in g:
    g = g.replace(
        "\\textsuperscript{a}Four-feature panel",
        "\\textsuperscript{b}Fully nested inner-loop joint selector (3 valid imbalance strategies $\\times$ 4 model families): 84 errors at 0.5 (81 of 88 discordant), exact McNemar $p=0.34$ vs \\gene{IDH1/2} rule. \\textsuperscript{a}Four-feature panel",
        1
    )
    g_path.write_text(g, encoding="utf-8")
print("Updated Table 2 footnote b in Glioma.tex")

# ── 2. DA-1: Update 3_Supplementary/Supplementary_Information.tex ──
si_path = BASE / "3_Supplementary/Supplementary_Information.tex"
si = si_path.read_text(encoding="utf-8")

# S6 caption
si = si.replace(
    "Class weighting had the highest mean AUC at full precision (0.901433 vs 0.901394 for SMOTENC) and was locked; standard SMOTE produced non-binary mutation values and was not eligible.",
    "Class weighting had the highest mean AUC at full precision (0.901433 vs 0.901394 for SMOTENC) and was designated in the two-stage pipeline; standard SMOTE produced non-binary mutation values and was not eligible. A complementary fully nested inner-loop joint selector (evaluating all three valid strategies $\\times$ four model families strictly inside each outer fold) is reported in Table~\\ref{S8}.",
    1
)

# S8 caption & row
si = si.replace(
    "The primary model was locked by the highest mean-of-fold AUC (ties by Brier score).",
    "The primary model was designated by the highest mean-of-fold AUC (ties by Brier score). To eliminate any potential selection optimism from Stage~1 imbalance screening or Stage~2 family selection, the bottom row reports a fully nested inner-loop joint selector (selecting SMOTENC+RandomForest [outer AUC 0.9079], Class-weight+LogisticRegression [0.9011], SMOTENC+XGBoost [0.8782], Class-weight+XGBoost [0.9414], and SMOTENC+LogisticRegression [0.9114] across Folds 0--4; 84 errors at 0.5 [81/88 discordant, 3/583 concordant], exact McNemar $p=0.34$ vs \\gene{IDH1/2} rule).",
    1
)
si = si.replace(
    "SVM\\_RBF & 0.9090 & 0.0191 & 0.8939 & 0.1096 & 0.1096 & 0.0151 \\\\",
    "SVM\\_RBF & 0.9090 & 0.0191 & 0.8939 & 0.1096 & 0.1096 & 0.0151 \\\\\n\\midrule\nFully nested joint selector (imbalance $\\times$ family) & 0.9080 & 0.0227 & 0.9053 & 0.1084 & 0.1084 & 0.0027 \\\\",
    1
)

# S19 caption
si = si.replace(
    "Ten repetitions of nested cross-validation of the primary model with different outer-fold assignments (training partition).",
    "Ten repetitions of 5-fold nested cross-validation of the primary model with different outer-fold assignments (development partition; primary internal performance assessment).",
    1
)

# S22 caption
si = si.replace(
    "Decisions locked on the training partition before the test set was scored (written to \\texttt{Results/Models/locks\\_before\\_test.json}).",
    "Decisions locked on the development partition before exploratory evaluation on the held-out test set (written to \\texttt{Results/Models/locks\\_before\\_test.json}); complemented by the fully nested inner-loop joint selector in Table~\\ref{S8}.",
    1
)

# S26 / WHO 2021 whole-arm & GISTIC approximation note
si = si.replace(
    "\\caption[Error agreement with WHO 2021 categories across sensitivity variants]",
    "\\caption[Error agreement with approximate WHO 2021 categories across sensitivity variants]",
    1
)
si = si.replace(
    "All variants are post hoc reconstructions",
    "All variants are post hoc approximate reconstructions using genome-wide Affymetrix SNP~6.0 arm-level 1p/19q calls (approximating whole-arm co-deletion) and GISTIC~2.0 $+2$/$-2$ thresholds without tumour-purity deconvolution",
    1
)

# Exploratory test-set captions (S14-S17, S29-S30)
for old_c, new_c in [
    ("Test-set performance", "Exploratory test-set performance"),
    ("Held-out test-set", "Exploratory held-out test-set"),
    ("Test-set calibration", "Exploratory test-set calibration"),
    ("Test-set decision-curve", "Exploratory test-set decision-curve"),
    ("Test-set confusion", "Exploratory test-set confusion"),
]:
    si = si.replace(old_c, new_c)

si_path.write_text(si, encoding="utf-8")
print("Updated Supplementary_Information.tex (DA-1)")

# ── 3. DA-4: Update Response_to_Reviewers_R5.docx & archive R4 ──
r5_path = BASE / "4_Response/Response_to_Reviewers_R5.docx"
tmp_r5 = Path("/tmp/r5_da4")
if tmp_r5.exists(): shutil.rmtree(tmp_r5)
with zipfile.ZipFile(r5_path, "r") as z: z.extractall(tmp_r5)
r5_xml_p = tmp_r5 / "word/document.xml"
r5_xml = r5_xml_p.read_text(encoding="utf-8")
r5_xml = r5_xml.replace(
    "(87.5% accuracy, 84 errors at 0.5; selecting",
    "(87.5% accuracy, 84 errors at 0.5, 81 of 84 in discordant patients, exact McNemar p = 0.34 versus the IDH1/2 rule; selecting"
)
r5_xml_p.write_text(r5_xml, encoding="utf-8")
with zipfile.ZipFile(r5_path, "w", zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk(tmp_r5):
        for f in files:
            fp = os.path.join(root, f)
            zout.write(fp, os.path.relpath(fp, tmp_r5))
shutil.rmtree(tmp_r5)

r4_path = BASE / "4_Response/Response_to_Reviewers_R4.docx"
arch_dir = BASE / "4_Response/Archive_R2"
arch_dir.mkdir(exist_ok=True)
if r4_path.exists():
    shutil.move(str(r4_path), str(arch_dir / "Response_to_Reviewers_R4.docx"))
print("Updated Response_to_Reviewers_R5.docx and archived R4 (DA-4)")

# ── 4. DA-5: Synchronize remaining rows in TRIPOD_AI_Checklist.docx (plain #000000) ──
chk_path = BASE / "5_Reporting_checklists/TRIPOD_AI_Checklist.docx"
tmp_chk = Path("/tmp/chk_da5")
if tmp_chk.exists(): shutil.rmtree(tmp_chk)
with zipfile.ZipFile(chk_path, "r") as z: z.extractall(tmp_chk)
chk_xml_p = tmp_chk / "word/document.xml"
cxml = chk_xml_p.read_text(encoding="utf-8")

da5_repls = [
    ("4. Objectives", "4. Objectives (10x5 repeated nested CV & fully nested joint selection primary; test set exploratory; WHO 2021 post hoc)"),
    ("nested cross-validation on the training partition", "10x5-fold repeated nested cross-validation and fully nested inner-loop joint selection on the development partition (with exploratory test evaluation)"),
    ("12c", "12c (including fully nested joint inner-loop selection of imbalance x family x hyperparameters)"),
    ("held-out test set (n = 168)", "exploratory held-out test set (n = 168; prior revision exposure disclosed)"),
]
for old_s, new_s in da5_repls:
    if old_s in cxml and new_s not in cxml:
        cxml = cxml.replace(old_s, new_s, 1)

# Clean any stray colors/shading/borders
cxml = re.sub(r"<w:shd\b[^>]*/?>", "", cxml)
cxml = re.sub(r"<w:pBdr>.*?</w:pBdr>", "", cxml, flags=re.DOTALL)
cxml = re.sub(r"<w:color\s+w:val=\"(?!000000|auto)[0-9A-Fa-f]+\"[^>]*/?>", "<w:color w:val=\"000000\"/>", cxml)
chk_xml_p.write_text(cxml, encoding="utf-8")
with zipfile.ZipFile(chk_path, "w", zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk(tmp_chk):
        for f in files:
            fp = os.path.join(root, f)
            zout.write(fp, os.path.relpath(fp, tmp_chk))
shutil.rmtree(tmp_chk)
print("Synchronized TRIPOD_AI_Checklist.docx (DA-5)")

# ── 5. Rebuild 6_Code/Glioma-Classification-v3.zip (DA-6) ──
zip_path = BASE / "6_Code/Glioma-Classification-v3.zip"
code_dir = BASE / "6_Code"
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d != "__pycache__"]
        for f in sorted(files):
            if f.startswith(".") or f == "Glioma-Classification-v3.zip": continue
            fp = Path(root) / f
            zf.write(fp, Path("Glioma-Classification-v3") / fp.relative_to(code_dir))
print("Rebuilt Glioma-Classification-v3.zip:", zip_path.stat().st_size, "bytes")
