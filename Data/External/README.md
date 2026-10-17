# External molecular annotations (TCGA LGG/GBM; Ceccarelli et al., Cell 2016)

These files are used **only** for the post hoc WHO 2021 reconstruction and its sensitivity analyses (notebook Sections 20f and 20j–20o). They are not used for model fitting, model selection, or any locked decision.

## Source and pin

| | |
|---|---|
| Study | cBioPortal `lgggbm_tcga_pub`, "Merged Cohort of LGG and GBM (TCGA, Cell 2016)" |
| Publication | Ceccarelli M. et al. *Cell* 164, 550–563 (2016), doi:10.1016/j.cell.2015.12.028 |
| Repository | https://github.com/cBioPortal/datahub, `public/lgggbm_tcga_pub/` |
| Pinned snapshot | `master` = **`18763f8eee86d4886d7210068f745f1e509498ee`** (state on 12 Oct 2026) |
| Downloaded | 12 October 2026, from the Git LFS media endpoint |
| Upstream licence | `public/lgggbm_tcga_pub/LICENSE`: TCGA data under the Broad Institute GDAC TCGA Analysis Pipeline License. Use is also subject to the NCI GDC/TCGA data-use and publication policies. |

The upstream data files are stored in Git LFS. The SHA-256 of each full file equals its LFS object ID, and each file is identical at the snapshot and at the last commit that modified it:

| Upstream file | Last-modifying commit | Full-file SHA-256 | Bytes |
|---|---|---|---|
| `data_clinical_sample.txt` | `04f170590dbd1ac6e49c2d334decb4dbdb4c14b9` (2021-12-01) | `cf622c83218f6a1a6f02a5e2cc8cc5c365ec90c71193faa693e113c5326e45ac` | 374,033 |
| `data_cna.txt` | `86690e1ed9752b1dcd50b5657f5f05eafa4b6b78` (2026-01-30) | `f37d9f804124bfaa5e6a704aa3dabc37b7b9876db081f0c83ad8d12783e854af` | 57,746,107 |
| `data_mutations.txt` | `86690e1ed9752b1dcd50b5657f5f05eafa4b6b78` (2026-01-30) | `052db47ed90e4a4e1f58ec977b8bd92628ae6bc58e3375e1b6541cbbec635039` | 126,144,206 |

The last-modifying commits were read with `git log -1` on a full-history clone, which is equivalent to `https://api.github.com/repos/cBioPortal/datahub/commits?path=public/lgggbm_tcga_pub/<file>&per_page=1`. They were re-checked on 12 Oct 2026 after all six files were in place.

To re-download a file at its pinned commit:

```
https://media.githubusercontent.com/media/cBioPortal/datahub/<commit>/public/lgggbm_tcga_pub/<file>
```

## Files in this folder

The files are row and column subsets of the upstream files. No values were changed, and upstream `NA` is written as an empty cell. `PATIENT_ID` is the first 12 characters of the TCGA sample barcode.

| File | Upstream | Subset rule | Rows | SHA-256 |
|---|---|---|---|---|
| `Ceccarelli2016_molecular_annotations.csv` | `data_clinical_sample.txt` | Columns PATIENT_ID, GRADE, IDH_STATUS, IDH_CODEL_SUBTYPE, IDH_1P19Q_SUBTYPE, CHR_7_GAIN_CHR_10_LOSS, TERT_PROMOTER_STATUS, ORIGINAL_SUBTYPE, SNP6 | 1,122 | `6296dc3ae825e7247f29a10c23351490a3cb1c626847e13e40dd7adb4f8f5445` |
| `Ceccarelli2016_GISTIC_EGFR_CDKN2AB.csv` | `data_cna.txt` (GISTIC 2.0 discrete) | Genes EGFR, CDKN2A, CDKN2B, one row per sample. Codes: 2 = high-level amplification; −2 = deep (homozygous) deletion | 1,084 | `900203e6d02ee34183685f041efd94f296f4c0b9999ff59d2709a7166c2193ca` |
| `Ceccarelli2016_methylation_BRAF.csv` | `data_clinical_sample.txt` | Columns PATIENT_ID, WHOLE_EXOME_SEQUENCING, SUPERVISED_DNA_METHYLATION_CLUSTER, BRAF_V600E_STATUS, BRAF_KIAA1549_FUSION | 1,122 | `189bfe88909819935d5ad7501937e327de350e739326a02088094f709ba32fe0` |
| `Ceccarelli2016_IDH_mutations.csv` | `data_mutations.txt` | Hugo_Symbol in {IDH1, IDH2}; all variant classes | 430 | `48064b04b87ddca85be3e55c4f36d4826299fdb5a542fbec52f2e0432c3cc08e` |
| `Ceccarelli2016_H3_mutations.csv` | `data_mutations.txt` | Hugo_Symbol in {H3F3A, H3-3A, HIST1H3B, H3C2, HIST1H3C, H3C3}; all variant classes | 6 | `1932f2c5ccabc74ca9654f2304835e4e0800dd708989cda2364711168f6cc680` |
| `Ceccarelli2016_exome_sequenced_samples.csv` | `data_mutations.txt` | Sample list in the `#sequenced_samples:` header line. This is the denominator: a patient not listed was not sequenced, which is different from wild-type. | 812 | `9545e52392b56fd4c7b20e3b7ed6e74c2c75ec11513d93000442d17299932539` |

On 12 Oct 2026 every subset was re-derived from the pinned upstream files and found identical to the file in this folder.

## Notes

- **Columns of the mutation subsets** (`*_IDH_mutations.csv`, `*_H3_mutations.csv`): PATIENT_ID, Hugo_Symbol, Chromosome, Start_Position, Variant_Classification, Reference_Allele, Tumor_Seq_Allele2, Tumor_Sample_Barcode, t_ref_count, t_alt_count, HGVSp_Short (in file order). `*_exome_sequenced_samples.csv`: Tumor_Sample_Barcode, PATIENT_ID.
- **Mutation numbering.** The MAF numbers proteins from the initiator methionine. H3F3A `p.K28M` is H3 **K27M**, and `p.G35R` is **G34R**.
- **H3 genes with records.** Only H3F3A and HIST1H3C have records.
- **Gene symbols.** The 2026-01-30 upstream commit migrated outdated gene symbols. Check the H3 symbol list if the files are re-downloaded from a later commit.
- **Copy-number sample count.** The GISTIC profile description says "563 cases", but the file has 1,084 sample columns. All columns are kept.
- **Extraction script.** The four files added in R4 (methylation/BRAF, IDH, H3, sequenced samples) were extracted on 12 Oct 2026 with `extract_external.py` (`python3 -I extract_external.py <download_dir> Data/External`; it only subsets rows and columns). The two R3 files, also downloaded on 12 Oct 2026, are plain column and row subsets as described above. The script is in the release as `tools/extract_external.py`, identical to the original in the revision package, `Revision_R4/engineering/extract_external.py` (SHA-256 `3d50d4b7233fcbbf05e70c19a31e4ce349900038f565c60bdce8e8b02e47085b`). The H3 calls used in the WHO 2021 sensitivity analyses come from `Ceccarelli2016_H3_mutations.csv`, interpreted against `Ceccarelli2016_exome_sequenced_samples.csv`. No other external files were added in R4.
- **Verification script.** `verify_external_sources.py` (R4 compliance package) checks every hash above and rebuilds the subsets from the upstream files.
