"""
Programmatic 600-DPI PNG + Vector PDF generator for Figure 1 (Study Design).
Fixes all R3 reviewer items (98 LGG / 70 GBM, 4-feature panel, IDH1, TERT data)
and updates labels for fully nested selection, exploratory test set, and post-hoc WHO 2021.
"""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path("/Users/sandy/Downloads/Research Work/Glioma/Revision 2/Submission Ready/Final")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(11.5, 8.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def draw_box(x, y, w, h, lines, fc="#F7F8FA", ec="#7A869A", lw=1.5, align="center",
             title_bold=True, fontsize=9.2, title_size=9.8, line_spacing=1.32):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.4",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
    )
    ax.add_patch(patch)
    n = len(lines)
    if align == "center":
        cx = x + w / 2.0
        # vertically center lines
        step = min(3.1, (h - 1.4) / max(n, 1))
        top_y = y + h / 2.0 + (n - 1) * step / 2.0
        for idx, line in enumerate(lines):
            is_bold = (idx == 0 and title_bold)
            ax.text(
                cx, top_y - idx * step, line,
                ha="center", va="center",
                fontsize=title_size if is_bold else fontsize,
                fontweight="bold" if is_bold else "normal",
                color="#1A202C", zorder=3
            )
    else:
        lx = x + 1.8
        top_y = y + h - 2.3
        step = (h - 3.6) / max(n, 1)
        for idx, line in enumerate(lines):
            is_bold = (idx == 0 and title_bold)
            ax.text(
                lx, top_y - idx * step, line,
                ha="left", va="top",
                fontsize=title_size if is_bold else fontsize,
                fontweight="bold" if is_bold else "normal",
                color="#1A202C", zorder=3
            )

def draw_arrow(x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5.5,head_width=3.5",
        color="#2D3748", linewidth=1.6, zorder=4
    )
    ax.add_patch(arr)

# Left column geometry: x=2.5, w=45.0 (center=25.0)
# Right column geometry: x=53.5, w=44.0 (center=75.5)

# Row 1 (y=86..96.5)
draw_box(2.5, 86.5, 45.0, 9.8, [
    "Raw TCGA-LGG/GBM export (UCI ID 759)",
    "n = 862"
], fc="#F4F5F7", ec="#718096")

draw_box(53.5, 86.5, 44.0, 9.8, [
    "Excluded: missing sex, age or race",
    "n = 23 (sex 4, age 5, race 22)"
], fc="#FFF5F5", ec="#C53030")

# Row 2 (y=71.5..82.5)
draw_box(2.5, 71.5, 45.0, 10.5, [
    "Analysis cohort (n = 839)",
    "487 legacy LGG / 352 legacy GBM; 23 features"
], fc="#F4F5F7", ec="#718096")

draw_box(53.5, 70.5, 44.0, 12.0, [
    "Label–IDH audit (TCGA Case IDs)",
    "110 labels disagree with IDH1/2 status (13.1%)",
    "(129 with IDH1 status alone; 15.4%)"
], fc="#FEFCBF", ec="#D69E2E")

# Row 3 (y=56..66.5)
draw_box(2.5, 56.0, 45.0, 10.5, [
    "Development partition (n = 671; 80%)",
    "389 legacy LGG / 282 legacy GBM"
], fc="#F4F5F7", ec="#718096")

draw_box(53.5, 56.0, 44.0, 10.5, [
    "Exploratory held-out test set (n = 168; 20%)",
    "98 legacy LGG / 70 legacy GBM"
], fc="#EBF8FF", ec="#2B6CB0")

# Row 4 Left: Primary internal validation & locked pipeline (y=8.5..51.5)
draw_box(2.5, 8.5, 45.0, 43.0, [
    "Development-partition pipeline (no test data)",
    "1  Imbalance (4 arms, 5×5 CV) & fully nested joint inner-loop selection",
    "    (imbalance × family × hyperparameters, 5×5 nested CV)",
    "2  Four model families (LR, SVM-RBF, RF, XGBoost); primary = XGBoost",
    "3  Primary internal validation: 10×5-fold repeated nested CV",
    "4  Baselines (age, IDH1, age + IDH1, clinical); DeLong & McNemar",
    "5  Calibration: out-of-fold Platt scaling; Youden threshold = 0.351",
    "6  Interpretability: fold-aggregated SHAP, permutation, nested LOFO",
    "7  Nested reduced panels -> 4-feature panel locked (IDH1, age, IDH2, TP53)",
    "8  Decision-curve analysis (DCA) & tuning sensitivity check"
], fc="#F0FFF4", ec="#2F855A", align="left", fontsize=8.0, title_size=9.5)

# Row 4 Right top: Exploratory evaluation (y=33..51.5)
draw_box(53.5, 33.5, 44.0, 18.0, [
    "Exploratory evaluation of locked models",
    "Primary 23-feature XGBoost (Platt, threshold 0.351)",
    "Age + IDH1 baseline; 4-feature panel",
    "IDH1 and IDH1/2 no-fit rules (zero-parameter)"
], fc="#EBF8FF", ec="#2B6CB0", fontsize=8.8)

# Row 4 Right bottom: Post-hoc WHO 2021 (y=8.5..28.5)
draw_box(53.5, 8.5, 44.0, 20.0, [
    "Post-hoc WHO 2021 reconstruction & error audit",
    "Errors stratified by label–IDH concordance",
    "Approximate categories from TCGA SNP 6.0 1p/19q,",
    "+7/−10, GISTIC 2.0 EGFR/CDKN2A/B, and TERT data",
    "Sensitivity variants (a–g, cIMPACT-NOW 11, ≥2 criteria) & MONET"
], fc="#FEFCBF", ec="#D69E2E", fontsize=8.3)

# Arrows
draw_arrow(25.0, 86.2, 25.0, 82.3)   # Raw -> Cohort
draw_arrow(47.8, 91.4, 53.2, 91.4)   # Raw -> Excluded
draw_arrow(25.0, 71.2, 25.0, 66.8)   # Cohort -> Dev
draw_arrow(47.8, 76.5, 53.2, 76.5)   # Cohort -> Audit
draw_arrow(47.8, 72.5, 53.2, 62.5)   # Cohort -> Test
draw_arrow(25.0, 55.7, 25.0, 51.8)   # Dev -> Dev pipeline
draw_arrow(75.5, 55.7, 75.5, 51.8)   # Test -> Exploratory eval
draw_arrow(75.5, 33.2, 75.5, 28.8)   # Exploratory eval -> Post-hoc WHO 2021

# Footer note
ax.text(
    2.5, 4.2,
    "No test-set information enters any development-partition step on the left. Because test-set results had been seen in earlier revision rounds,\n"
    "10×5-fold repeated nested CV serves as the primary internal performance assessment and test-set evaluation is exploratory; WHO 2021 reconstruction is post hoc.",
    ha="left", va="center", fontsize=8.1, style="italic", color="#2D3748"
)

plt.tight_layout(pad=0.3)
fig_pdf = BASE / "1_Manuscript/Figures/Figure00_StudyDesign.pdf"
fig_png = BASE / "1_Manuscript/Figures/Figure00_StudyDesign.png"
fig_up  = BASE / "2_Figures_for_upload/Figure1.png"

fig.savefig(fig_pdf, format="pdf", bbox_inches="tight")
fig.savefig(fig_png, format="png", dpi=600, bbox_inches="tight")
shutil.copy2(fig_png, fig_up)
plt.close(fig)
print("Generated 600-DPI Figure 1 PNG and vector PDF:", fig_png.stat().st_size, "bytes")

jf = BASE / "6_Code/Results/Tables/FullyNested_JointSelection_Summary.json"
if jf.exists():
    print("\n=== JOINT SELECTION SUMMARY ===")
    print(jf.read_text())
else:
    print("\nJoint selection script still running...")
