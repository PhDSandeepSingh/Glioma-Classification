import os, re, zipfile
from pathlib import Path

BASE = Path("/Users/sandy/Downloads/Research Work/Glioma/Revision 2/Submission Ready/Final")
tex_path = BASE / "1_Manuscript/Glioma.tex"
tex = tex_path.read_text(encoding="utf-8")

new_abs = (
    r"\abstract{The UCI/TCGA Glioma Grading benchmark labels gliomas as lower-grade glioma (LGG) or glioblastoma (GBM) using legacy WHO 2007 histology, "
    r"yet this target disagreed with \gene{IDH1}/\gene{IDH2} mutation status in 110 of 839 patients (13.1\%). "
    r"In a post hoc reconstruction from published TCGA annotations, 68 of 85 IDH-wildtype legacy LGG (80.0\%; 51.8--84.0\% across variants) met approximate WHO 2021 molecular criteria for glioblastoma, "
    r"whereas all 23 assessable non-codeleted IDH-mutant legacy GBM mapped to CNS WHO grade~4 astrocytoma (21 of 25 via histology alone, lacking \gene{CDKN2A/B} deletion). "
    r"In primary internal validation via 10-times-repeated 5-fold nested cross-validation ($n=671$), XGBoost trained on 20 mutations, age, sex and race achieved a pooled out-of-fold AUC of $0.913\pm0.005$ "
    r"(canonical AUC 0.912; 86 errors versus 88 for the rule ``IDH-wildtype implies GBM'', McNemar $p=0.69$; fully nested joint-selection AUC 0.905) "
    r"and an exploratory test AUC of 0.932 ($n=168$, examined in earlier revisions; 25 versus 22 rule errors, $p=0.25$). "
    r"Most errors (84/86 development; 22/25 test) occurred in discordant patients: 72/79 assessable development errors agreed with tumour type versus 52/78 with grade~4, and copy-number/1p/19q features did not raise accuracy. "
    r"On this single-cohort benchmark, label--IDH discordance and unmeasured histology---not classifier choice---bound accuracy, offering no clinical utility over IDH status alone.}"
)

plain = new_abs.replace(r"\abstract{", "")[:-1]
plain = re.sub(r"\\gene\{([^}]+)\}", r"\1", plain)
plain = plain.replace(r"\%", "%").replace("~", " ").replace("---", "—").replace("--", "–").replace("``", "\"").replace("''", "\"").replace("$", "").replace(r"\pm", "±")
words = plain.split()
print(f"Verified Scientific Reports Abstract word count: {len(words)} words (limit <= 200)")
assert len(words) <= 200, f"Word count {len(words)} exceeds 200!"
print("Plain text:\n" + plain)

m = re.search(r"\\abstract\{.*?\}\n", tex, flags=re.DOTALL)
assert m, "Could not locate \\abstract{...} in Glioma.tex"
tex = tex[:m.start()] + new_abs + "\n" + tex[m.end():]
tex_path.write_text(tex, encoding="utf-8")
