# Glioma-Classification — Revision 3/4 analysis

**Manuscript:** *Impact of diagnostic label–IDH discordance and WHO 2021 criteria on machine-learning glioma grading from somatic mutations* (*Scientific Reports*)

This is a single canonical run of the full pipeline, prepared for the second-round revision for *Scientific Reports*.

**Research use only.** Neither the model nor the reconstructed WHO 2021 labels are for clinical diagnosis.

## Contents
| Path | What it is |
|---|---|
| `Glioma_Grading_R3.ipynb` | Executed notebook with all outputs saved. This is the record of the canonical run. |
| `build_notebook.py` | Source of the notebook as plain Python. Running it regenerates the `.ipynb`. |
| `run_nb.py` | Executes the notebook headlessly and saves the outputs. |
| `requirements_r3.txt` | Direct dependencies, pinned to the versions used in the canonical run. |
| `environment/` | `requirements-lock.txt`: the full pinned dependency closure. `environment.yml`: the conda equivalent. |
| `Data/` | The UCI analysis file (839 × 24), the raw TCGA export (862 patients), the label–IDH audit (`Discordant_Cases_Audit_WHO2021.csv`) and the patient-level WHO 2021 reconstruction (`WHO2021_Reconstruction_PatientLevel.csv`). |
| `Data/Release/` | The versioned patient-level label release (`TCGA_GliomaBenchmark_WHO2021_Labels_v1.0.csv`) and its column dictionary. |
| `Data/DATASHEET.md` | Datasheet for the label release, following Gebru et al. |
| `Data/External/` | Published TCGA molecular annotations (Ceccarelli et al., Cell 2016; cBioPortal `lgggbm_tcga_pub`), used only for the post hoc WHO 2021 reconstruction. The README there records pinned commits and SHA-256 checksums. |
| `Results/Tables`, `Results/Figures` | Every table and figure cited in the manuscript. |
| `Results/ManuscriptNumbers.csv` | Every number the manuscript cites, with its source table. Update the manuscript from this file only. |
| `Results/Models/` | Locks recorded before the test set was opened, test-set predictions, cached intermediate results and the reproducibility report. |
| `tools/extract_external.py` | Script that cut the four R4 files in `Data/External/` from the full cBioPortal files (rows and columns only). Run as `python3 -I tools/extract_external.py <download_dir> Data/External`. The original is in the revision package at `Revision_R4/engineering/extract_external.py`. |
| `CITATION.cff`, `.zenodo.json` | Citation and archive metadata (see "Release and citation"). |

## Reproduce

**Requirements**
- **Python 3.13.x.** The canonical run used CPython 3.13.16 on Linux x86_64. pip cannot enforce the interpreter version, so create the environment with Python 3.13.
- **Package versions** match `Results/Models/reproducibility_report.json`.

**Install**
```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-lock.txt   # exact environment, or: pip install -r requirements_r3.txt
# conda alternative:  conda env create -f environment/environment.yml && conda activate glioma-r3
```

**Run**
```bash
python build_notebook.py      # optional: regenerate the notebook from source
python run_nb.py              # executes the notebook headlessly and saves all outputs
```

**Runtime** (Linux x86_64, 2 CPU cores, CPython 3.13.16):
- From scratch (empty `Results/Models/`, all 12 caches rebuilt, `build_notebook.py` then `run_nb.py`): **1,258 s (21.0 min)**. This is the final clean R4 run of 12 Oct 2026.
- Re-execution with valid caches: about 5 min (286 s, `run_stage2.log`).

**From-scratch run.** Delete `Results/` first. Cached steps are keyed on the data hash, the run tag and a hash of the code that produced them.

**Results check**
- Compare the regenerated `Results/ManuscriptNumbers.csv` with the released copy.
- The test-set predictions (`Results/Models/test_set_predictions.csv`) and the locks (`Results/Models/locks_before_test.json`) should be identical.
- The fitted model pickles are tied to the pinned Python, scikit-learn and xgboost versions. Do not load them under other versions.

## Data provenance

| Input | Identifier / pin | Licence / terms | SHA-256 |
|---|---|---|---|
| `Data/TCGA_InfoWithGrade.csv` | UCI ML Repository dataset 759, doi:10.24432/C5R62J | CC BY 4.0 | `1b25614f…195144a1` |
| `Data/TCGA_GBM_LGG_Mutations_all.csv` | same dataset (raw TCGA export with case IDs) | CC BY 4.0 | `6c617f37…44e6b312` |
| `Data/External/*.csv` (6 files) | cBioPortal datahub `public/lgggbm_tcga_pub`, commit `18763f8eee86d4886d7210068f745f1e509498ee`, downloaded 12 Oct 2026 | TCGA open-access data (NCI GDC policies; study LICENSE) | listed in `Data/External/README.md` |

**UCI files**
- Both files are byte-identical to the copies used in the earlier submissions.
- The analysis-file hash is also recorded in `reproducibility_report.json`.

**External files**
- Each external subset was re-derived from the full upstream file at its pinned commit and found identical.
- The upstream full-file checksums (Git LFS object IDs) and the commit that last modified each upstream file are listed in `Data/External/README.md`.

**Derived data**
- `WHO2021_Reconstruction_PatientLevel.csv`, `Discordant_Cases_Audit_WHO2021.csv` and `Data/Release/` are generated by the notebook.
- They are not inputs.

## Release and citation
- **Archived release (DOI): TODO(authors).** Tag a GitHub release (for example `v3.0.0`) and archive it with the Zenodo–GitHub integration. Zenodo reads `.zenodo.json`, which takes precedence over `CITATION.cff`, and mints a version DOI. Paste that DOI into:
  - `CITATION.cff` (the commented `doi:` line);
  - this README;
  - the manuscript's Code availability statement;
  - the reference list.
- **Licences**
  - Code: MIT. The `LICENSE` file from the earlier repository release must be copied into this folder before tagging.
  - Label release (`Data/Release/`): proposed CC BY 4.0, to match the UCI source. This is an author decision; see `Data/DATASHEET.md`.
  - TCGA-derived inputs remain subject to the NCI GDC/TCGA data-use policies.
- **How to cite.** Cite the archived release (see `CITATION.cff`), the article, and the data sources:
  - the UCI dataset;
  - Ceccarelli et al., Cell 2016;
  - cBioPortal;
  - the TCGA Research Network.

## Reporting checklists
The revision package (`Revision_R4/compliance/`) contains:
- **TRIPOD+AI** (Collins et al., BMJ 2024): a completed checklist (`TRIPOD_AI_Checklist.docx` / `.pdf` / `.md`). Locations are given as manuscript section names, tables, figures and Supplementary items. It is summarised by topic in Supplementary Table S23.
- **PROBAST+AI** (Moons et al., BMJ 2025): a self-assessment with signalling-question answers (`PROBAST_AI_Assessment.md`). Its summary is proposed as a Supplementary table.
- **Datasheet** for the label release: `Data/DATASHEET.md`.
- **Data provenance and verification:** `data_provenance.md` and `verify_external_sources.py`.

## Fixes relative to the Revision-2 notebook
1. All model-dependent steps (final refit, SHAP, LOFO, reduced panel, ensemble, tuning check) follow the locked primary model; nothing is hard-coded to XGBoost. The final model's family is asserted before the test set is scored.
2. The locked imbalance strategy (class weighting) is now actually applied to every model family.
3. The LOFO and reduced-panel ablations are fully nested: hyper-parameters are re-tuned inside each outer fold. ΔAUC uses pooled out-of-fold AUC on both sides, with paired DeLong tests and Holm correction.
4. The 4-feature panel carried to the test set is the nested majority-vote panel, locked before the test set is opened.
5. The ensemble is a leakage-free nested soft vote, compared with the primary model.
6. The duplicate-row audit and the WHO 2021 discordance audit are built from the raw TCGA file inside the notebook. The row-level mapping was verified on all 24 fields, and IDH2 is included.
7. New: errors stratified by agreement between the legacy label and IDH status (Tables 21, 30, 30b, 31; Figure 18); no-fit IDH1 and IDH1/2 rules as comparators (Table 32); test-set decision curves (Figure 16).
8. New: a post hoc reconstruction of WHO 2021 categories from published TCGA 1p/19q, chromosome 7/10, TERT and EGFR/CDKN2A/B copy-number data, and the agreement of the locked model with the reconstructed targets (Section 20f; Tables 33, 33b, 34). It is descriptive only and not used for any locked decision.
9. R4: post hoc sensitivity analyses of the reconstruction (Sections 20g–20o), a versioned label release (`Data/Release/`) with a datasheet, pinned external sources with checksums, and pinned Python and package versions.

## Round 3 Reproducibility Additions
- `tools/run_fully_nested_joint.py`: Fully nested inner-loop joint selection across 3 valid imbalance strategies × 4 model families (`mean-of-fold AUC 0.908 ± 0.023`, `pooled OOF AUC 0.905`, `84` errors at `0.5` [`81/88` discordant], McNemar `p = 0.34` vs `IDH1/2` rule).
- `Data/Release/TCGA_GliomaBenchmark_WHO2021_Labels_v1.0.csv`: Expanded 64-column self-contained patient-level WHO 2021 label release.
