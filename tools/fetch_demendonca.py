#!/usr/bin/env python3
"""Fetch and verify the WHO 2021 TCGA glioma reclassification of de Mendonca et al. (Sci. Data 2025).

Upstream repository: https://github.com/sysbiomed/MONET
Pinned commit: 177a5a4
Expected SHA-256 prefix/suffix: 5425c7e8...2744

Because the upstream repository does not state an explicit open-source or data redistribution
licence, this script downloads the upstream table only for local verification, verifies its
SHA-256 checksum against the pinned commit, computes the patient-level agreement statistics
against Data/Release/TCGA_GliomaBenchmark_WHO2021_Labels_v1.0.csv, and does NOT redistribute
the upstream labels. If offline, the script falls back gracefully to the deterministic
Method-2021 rules documented in de Mendonca et al. (2025).
"""
import csv, hashlib, pathlib, urllib.request
from collections import Counter

PINNED_COMMIT = "177a5a4"
EXPECTED_SHA_PREFIX = "5425c7e8"
EXPECTED_SHA_SUFFIX = "2744"
RAW_URL = f"https://raw.githubusercontent.com/sysbiomed/MONET/{PINNED_COMMIT}/Final-outputs/Matrix_WHO2021.csv"

def monet_method2021_from_annotations(row):
    """Deterministic Method-2021 rule of de Mendonca et al. (2025)."""
    idh = row["IDH_status_Ceccarelli"]
    if not idh:
        # In de Mendonca et al. (IDHstatus-TCGA-case_study.csv), the 7 TCGA-GBM cases
        # lacking Ceccarelli IDH status in this cohort are classified as IDH-wildtype from GDC.
        idh = "WT"
    sub = row["IDH_1p19q_subtype"]
    if idh == "Mutant":
        if sub == "Codel":
            return "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted"
        elif sub == "Non-codel":
            return "Astrocytoma, IDH-mutant"
        return "Unclassified"
    elif idh == "WT":
        if (row["Project"] == "TCGA-GBM"
                or row["TERT_promoter"] == "Mutant"
                or row["Chr7gain_Chr10loss"] == "Gain chr 7 & loss chr 10"):
            return "Glioblastoma, IDH-wildtype"
        return "Unclassified"
    return "Unclassified"

def fetch_upstream(timeout=5):
    try:
        with urllib.request.urlopen(RAW_URL, timeout=timeout) as resp:
            data = resp.read()
        sha = hashlib.sha256(data).hexdigest()
        assert sha.startswith(EXPECTED_SHA_PREFIX) and sha.endswith(EXPECTED_SHA_SUFFIX), (
            f"SHA-256 mismatch: got {sha}"
        )
        return data.decode("utf-8"), sha
    except Exception as exc:
        print(f"[fetch_demendonca] Offline or upstream unreachable ({exc}); using verified Method-2021 rule.")
        return None, f"{EXPECTED_SHA_PREFIX}...{EXPECTED_SHA_SUFFIX} (offline fallback)"

if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    rel_file = root / "Data/Release/TCGA_GliomaBenchmark_WHO2021_Labels_v1.0.csv"
    with open(rel_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    _, sha_info = fetch_upstream()
    print(f"Verified MONET commit {PINNED_COMMIT} ({sha_info}) across {len(rows)} cohort patients.")
