"""
Fully nested inner-loop joint selection across:
  3 valid imbalance strategies (No resampling, Class-weight (balanced) with fold-specific SPW, SMOTENC)
  x 4 model families (LogisticRegression, SVM_RBF, RandomForest, XGBoost)
  x hyperparameters (same RandomizedSearchCV budgets as build_notebook.py)
"""
import json, os, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

BASE = Path("/Users/sandy/Downloads/Research Work/Glioma/Revision 2/Submission Ready/Final/6_Code")
RANDOM_STATE = 42
N_JOBS = 1

df = pd.read_csv(BASE / "Data/TCGA_InfoWithGrade.csv")
df = df.dropna(subset=["Age_at_diagnosis", "Gender", "Race", "Grade"]).reset_index(drop=True)
ALL_FEATURES = [c for c in df.columns if c != "Grade"]
X = df[ALL_FEATURES]
y = df["Grade"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
Xtr = X_train.reset_index(drop=True)
ytr = y_train.reset_index(drop=True)

def build_preprocessor(features):
    t = []
    if "Age_at_diagnosis" in features:
        t.append(("age", StandardScaler(), ["Age_at_diagnosis"]))
    if "Gender" in features:
        t.append(("gender", "passthrough", ["Gender"]))
    if "Race" in features:
        t.append(("race", OneHotEncoder(categories=[[0, 1, 2, 3]], drop="first", handle_unknown="ignore"), ["Race"]))
    rest = [f for f in features if f not in ("Age_at_diagnosis", "Gender", "Race")]
    if rest:
        t.append(("genes", "passthrough", rest))
    return ColumnTransformer(t, remainder="drop", verbose_feature_names_out=False)

FAMILIES = ["LogisticRegression", "SVM_RBF", "RandomForest", "XGBoost"]
STRATEGIES = ["No resampling", "Class-weight (balanced)", "SMOTENC"]

SEARCH_SPACES = {
    "LogisticRegression": ({"clf__C": loguniform(1e-3, 1e2)}, 25),
    "SVM_RBF": ({"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e-1)}, 12),
    "RandomForest": ({"clf__n_estimators": randint(100, 500), "clf__max_depth": [None, 3, 5, 8, 12],
                      "clf__min_samples_leaf": randint(1, 8), "clf__max_features": ["sqrt", "log2", None]}, 25),
    "XGBoost": ({"clf__n_estimators": randint(100, 400), "clf__max_depth": randint(2, 6),
                 "clf__learning_rate": loguniform(0.01, 0.3), "clf__subsample": [0.6, 0.8, 1.0],
                 "clf__colsample_bytree": [0.6, 0.8, 1.0]}, 25),
}

def make_estimator(family, strategy, spw_fold):
    cw = "balanced" if strategy == "Class-weight (balanced)" else None
    if family == "LogisticRegression":
        return LogisticRegression(max_iter=5000, class_weight=cw, random_state=RANDOM_STATE)
    if family == "SVM_RBF":
        return SVC(kernel="rbf", probability=True, class_weight=cw, random_state=RANDOM_STATE)
    if family == "RandomForest":
        return RandomForestClassifier(class_weight=cw, random_state=RANDOM_STATE, n_jobs=1)
    if family == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1,
                             scale_pos_weight=spw_fold if cw else 1.0)
    raise ValueError(family)

def make_pipeline(features, family, strategy, spw_fold):
    steps = [("preprocess", build_preprocessor(features)), ("clf", make_estimator(family, strategy, spw_fold))]
    if strategy == "SMOTENC":
        cat_idx = [i for i, f in enumerate(features) if f != "Age_at_diagnosis"]
        sampler = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
        return ImbPipeline([("resample", sampler)] + steps)
    return Pipeline(steps)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
logf = open(BASE / "Results/Tables/joint_progress.log", "w")

def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()

def run_joint_nested_cv(outer_splitter, label):
    t0 = time.time()
    n_repeats = getattr(outer_splitter, "n_repeats", 1)
    n_splits = 5
    repeat_oofs = [np.zeros(len(ytr)) for _ in range(n_repeats)]
    fold_records = []
    for idx, (tr, te) in enumerate(outer_splitter.split(Xtr, ytr)):
        rep_i = idx // n_splits
        fold_i = idx % n_splits
        X_f_tr, y_f_tr = Xtr.iloc[tr][ALL_FEATURES], ytr.iloc[tr]
        X_f_te, y_f_te = Xtr.iloc[te][ALL_FEATURES], ytr.iloc[te]
        spw_fold = float((y_f_tr == 0).sum() / (y_f_tr == 1).sum())

        best_inner_auc = -1.0
        best_choice = None
        for strat in STRATEGIES:
            for fam in FAMILIES:
                params, n_iter = SEARCH_SPACES[fam]
                search = RandomizedSearchCV(
                    make_pipeline(ALL_FEATURES, fam, strat, spw_fold),
                    params, n_iter=n_iter, scoring="roc_auc",
                    cv=inner_cv, random_state=RANDOM_STATE, n_jobs=N_JOBS, refit=True
                )
                search.fit(X_f_tr, y_f_tr)
                if search.best_score_ > best_inner_auc:
                    best_inner_auc = float(search.best_score_)
                    best_choice = (strat, fam, search)

        sel_strat, sel_fam, sel_search = best_choice
        p = sel_search.predict_proba(X_f_te)[:, 1]
        repeat_oofs[rep_i][te] = p
        auc_f = float(roc_auc_score(y_f_te, p))
        brier_f = float(brier_score_loss(y_f_te, p))
        fold_records.append({
            "repeat": rep_i, "fold": fold_i,
            "selected_strategy": sel_strat, "selected_family": sel_fam,
            "inner_cv_auc": best_inner_auc, "outer_auc": auc_f, "outer_brier": brier_f,
            "spw_fold": spw_fold,
        })
        log(f"[{label}] rep={rep_i} fold={fold_i} -> {sel_strat} + {sel_fam} (inner={best_inner_auc:.4f}, outer={auc_f:.4f}) [{time.time()-t0:.1f}s]")

    rep_df = pd.DataFrame(fold_records)
    rep_summaries = []
    for r_i in range(n_repeats):
        sub = rep_df[rep_df["repeat"] == r_i]
        oof = repeat_oofs[r_i]
        preds_05 = (oof >= 0.5).astype(int)
        rep_summaries.append({
            "repeat": r_i,
            "mean_fold_auc": float(sub["outer_auc"].mean()),
            "pooled_oof_auc": float(roc_auc_score(ytr, oof)),
            "pooled_oof_brier": float(brier_score_loss(ytr, oof)),
            "oof_acc_05": float(accuracy_score(ytr, preds_05)),
            "oof_errors_05": int((preds_05 != ytr.values).sum()),
            "oof_mcc_05": float(matthews_corrcoef(ytr, preds_05)),
        })
    sum_df = pd.DataFrame(rep_summaries)
    log(f"[{label}] DONE in {time.time()-t0:.1f}s")
    return rep_df, sum_df, repeat_oofs

outer_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
f5_folds, f5_sum, f5_oofs = run_joint_nested_cv(outer_5, "Canonical_5x5")
f5_folds.to_csv(BASE / "Results/Tables/Table_FullyNested_Joint_Canonical5Fold.csv", index=False)
np.save(BASE / "Results/Models/oof_joint_nested_canonical.npy", f5_oofs[0])

outer_10x5 = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
r10_folds, r10_sum, r10_oofs = run_joint_nested_cv(outer_10x5, "Repeated_10x5")
r10_sum.to_csv(BASE / "Results/Tables/Table_FullyNested_Joint_Repeated10x5.csv", index=False)
r10_folds.to_csv(BASE / "Results/Tables/Table_FullyNested_Joint_Repeated10x5_Folds.csv", index=False)

out = {
    "canonical_5x5": {
        "mean_fold_auc": float(f5_sum["mean_fold_auc"].iloc[0]),
        "sd_fold_auc": float(f5_folds["outer_auc"].std()),
        "pooled_oof_auc": float(f5_sum["pooled_oof_auc"].iloc[0]),
        "pooled_oof_brier": float(f5_sum["pooled_oof_brier"].iloc[0]),
        "oof_acc_05": float(f5_sum["oof_acc_05"].iloc[0]),
        "oof_errors_05": int(f5_sum["oof_errors_05"].iloc[0]),
        "oof_mcc_05": float(f5_sum["oof_mcc_05"].iloc[0]),
        "folds": f5_folds.to_dict(orient="records"),
    },
    "repeated_10x5": {
        "mean_of_fold_auc_mean": float(r10_sum["mean_fold_auc"].mean()),
        "mean_of_fold_auc_sd": float(r10_sum["mean_fold_auc"].std()),
        "pooled_oof_auc_mean": float(r10_sum["pooled_oof_auc"].mean()),
        "pooled_oof_auc_sd": float(r10_sum["pooled_oof_auc"].std()),
        "pooled_oof_auc_min": float(r10_sum["pooled_oof_auc"].min()),
        "pooled_oof_auc_max": float(r10_sum["pooled_oof_auc"].max()),
        "pooled_oof_brier_mean": float(r10_sum["pooled_oof_brier"].mean()),
        "pooled_oof_brier_sd": float(r10_sum["pooled_oof_brier"].std()),
        "oof_acc_05_mean": float(r10_sum["oof_acc_05"].mean()),
        "oof_errors_05_mean": float(r10_sum["oof_errors_05"].mean()),
        "selection_counts": {f"{a} | {b}": int(c) for (a, b), c in r10_folds.groupby(["selected_strategy", "selected_family"]).size().items()},
    }
}
(BASE / "Results/Tables/FullyNested_JointSelection_Summary.json").write_text(json.dumps(out, indent=2))
log("ALL SAVED!")
logf.close()
