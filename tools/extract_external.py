import sys, hashlib, pandas as pd
D, OUT = sys.argv[1], sys.argv[2]
mf = f"{D}/data_mutations.txt"
cf = f"{D}/data_clinical_sample.txt"
gf = f"{D}/data_cna.txt"

# 1. Ceccarelli2016_exome_sequenced_samples.csv
seq = open(mf).readline().split()[1:]
pd.DataFrame({"Tumor_Sample_Barcode": seq, "PATIENT_ID": [s[:12] for s in seq]}).to_csv(
    f"{OUT}/Ceccarelli2016_exome_sequenced_samples.csv", index=False)

# 2 & 3. Ceccarelli2016_H3_mutations.csv and Ceccarelli2016_IDH_mutations.csv
cols = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "HGVSp_Short",
        "Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "t_ref_count", "t_alt_count"]
keep = []
for ch in pd.read_csv(mf, sep="\t", comment="#", usecols=cols, chunksize=5000, dtype=str, low_memory=False):
    keep.append(ch[ch.Hugo_Symbol.isin(["H3F3A", "H3-3A", "HIST1H3B", "H3C2", "HIST1H3C", "H3C3", "IDH1", "IDH2"])])
m = pd.concat(keep)
m.insert(0, "PATIENT_ID", m.Tumor_Sample_Barcode.str[:12])
m[m.Hugo_Symbol.str.startswith(("H3", "HIST"))].to_csv(f"{OUT}/Ceccarelli2016_H3_mutations.csv", index=False)
m[m.Hugo_Symbol.isin(["IDH1", "IDH2"])].to_csv(f"{OUT}/Ceccarelli2016_IDH_mutations.csv", index=False)

# 4 & 5. Ceccarelli2016_methylation_BRAF.csv and Ceccarelli2016_molecular_annotations.csv
s = pd.read_csv(cf, sep="\t", comment="#")
s[["PATIENT_ID", "WHOLE_EXOME_SEQUENCING", "SUPERVISED_DNA_METHYLATION_CLUSTER",
   "BRAF_V600E_STATUS", "BRAF_KIAA1549_FUSION"]].to_csv(
    f"{OUT}/Ceccarelli2016_methylation_BRAF.csv", index=False)
s[["PATIENT_ID", "GRADE", "IDH_STATUS", "IDH_CODEL_SUBTYPE", "IDH_1P19Q_SUBTYPE",
   "CHR_7_GAIN_CHR_10_LOSS", "TERT_PROMOTER_STATUS", "ORIGINAL_SUBTYPE", "SNP6"]].to_csv(
    f"{OUT}/Ceccarelli2016_molecular_annotations.csv", index=False)

# 6. Ceccarelli2016_GISTIC_EGFR_CDKN2AB.csv
g = pd.read_csv(gf, sep="\t", comment="#")
g = g[g["Hugo_Symbol"].isin(["EGFR", "CDKN2A", "CDKN2B"])].set_index("Hugo_Symbol")
sample_cols = [c for c in g.columns if c.startswith("TCGA-")]
gt = g[sample_cols].T
gt.index.name = "SAMPLE_ID"
gt = gt.reset_index()
gt.insert(0, "PATIENT_ID", gt["SAMPLE_ID"].str[:12])
gt[["PATIENT_ID", "SAMPLE_ID", "EGFR", "CDKN2A", "CDKN2B"]].to_csv(
    f"{OUT}/Ceccarelli2016_GISTIC_EGFR_CDKN2AB.csv", index=False)

for f in [mf, cf, gf]:
    print(f.split("/")[-1], hashlib.sha256(open(f, "rb").read()).hexdigest())
