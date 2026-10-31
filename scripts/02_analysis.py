# -*- coding: utf-8 -*-
"""02_analysis.py — Fall detection generalization on SisFall (window level).

Models: rule-based impact+tilt baseline, logistic regression, random forest,
histogram gradient boosting (all CPU, sklearn).
Protocols:
  P1 LOSO (leave-one-subject-out), pooled across 38 folds.
  P2 Leave-one-fall-type-out (LOFO): the detector has never seen that fall type.
  P3 Age-group transfer: train young (SA01-23) -> test elderly (SE01-15) and
     the reverse.
  P4 Device-position transfer: waist features only vs chest features only.
Metrics: ROC-AUC, PR-AUC (average precision), recall at fixed FPR=0.05,
window F1 at the pooled threshold. Subject-level bootstrap CIs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
RES = ROOT / "results"
RNG = np.random.default_rng(CFG["seed"])

w = pd.read_parquet(ROOT / "data" / "harmonised" / "windows.parquet")
feat = json.loads((ROOT / "results" / "feature_names.json").read_text())
w["age_group"] = np.where(w["subject"].str.startswith("SA"), "young", "elderly")
meta = {"windows": len(w), "subjects": int(w["subject"].nunique()),
        "fall_windows": int(w["label"].sum()),
        "young_windows": int((w["age_group"] == "young").sum()),
        "elderly_windows": int((w["age_group"] == "elderly").sum()),
        "fall_types": sorted(w.loc[w["is_fall_trial"] == 1, "code"].unique())}

X = w[feat].to_numpy()
y = w["label"].to_numpy()
subjects = w["subject"].to_numpy()


def rule_predict(w_):
    """classic two-stage rule: impact above 2.5 g AND large posture change."""
    return ((w_["mag_max"] >= CFG["impact_g"]) &
            ((w_["tilt_max"] >= 60) | (w_["tilt_p95"] >= 60))).astype(int).to_numpy()


def metrics(ytrue, yscore):
    out = {"roc_auc": float(roc_auc_score(ytrue, yscore)),
           "pr_auc": float(average_precision_score(ytrue, yscore))}
    fpr, tpr, thr = roc_curve(ytrue, yscore)
    # recall at FPR <= 0.05
    ok = fpr <= 0.05
    out["recall_at_fpr05"] = float(tpr[ok].max()) if ok.any() else 0.0
    prec, rec, th = precision_recall_curve(ytrue, yscore)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    out["f1_max"] = float(np.nanmax(f1))
    i = int(np.nanargmax(f1))
    out["threshold"] = float(th[min(i, len(th) - 1)]) if len(th) else 0.5
    out["precision_at_f1"] = float(prec[i]); out["recall_at_f1"] = float(rec[i])
    return out


def subject_boot_ci(ytrue, yscore, subjects_, stat="roc_auc", n=500):
    vals = []
    subs = np.unique(subjects_)
    for _ in range(n):
        pick = RNG.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([np.where(subjects_ == s)[0] for s in pick])
        yy, ss = ytrue[idx], yscore[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, ss) if stat == "roc_auc"
                    else average_precision_score(yy, ss))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def make_model(name):
    if name == "logit":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                      n_jobs=-1, random_state=CFG["seed"],
                                      class_weight="balanced_subsample")
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=CFG["seed"],
                                              max_iter=300,
                                              class_weight="balanced")
    raise ValueError(name)


# ---------------- P1 LOSO ----------------
folds = []
models = ["logit", "rf", "hgb"]
oof = {m: np.full(len(w), np.nan) for m in models}
subs = sorted(w["subject"].unique())
for k, s in enumerate(subs):
    te = subjects == s
    tr = ~te
    Xtr, ytr = X[tr], y[tr]
    for m in models:
        mod = make_model(m)
        if m == "logit":
            sc = StandardScaler().fit(Xtr)
            mod.fit(sc.transform(Xtr), ytr)
            oof[m][te] = mod.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            mod.fit(Xtr, ytr)
            oof[m][te] = mod.predict_proba(X[te])[:, 1]
    if k % 10 == 0:
        print(f"LOSO fold {k + 1}/{len(subs)}")
# rule baseline needs the frame too
oof["rule"] = rule_predict(w)

rows = []
for m in models + ["rule"]:
    met = metrics(y, oof[m])
    ci = subject_boot_ci(y, oof[m], subjects)
    rows.append({"protocol": "LOSO", "model": m, **met,
                 "auc_ci_lo": ci[0], "auc_ci_hi": ci[1]})
# per-device-position variants: waist-only / chest-only features
waist_feat = [f for f in feat if f not in ("rms2", "range2")]
chest_feat = ["rms2", "range2"] + [f for f in feat if f not in ("rms2", "range2")]
# simple device-restricted models (HGB only)
for dev, cols in [("waist_only", waist_feat), ("chest_only", ["rms2"])]:
    Xd = w[cols].to_numpy() if dev == "waist_only" else \
        w[["rms2", "range2"]].to_numpy()
    oofd = np.full(len(w), np.nan)
    for s in subs:
        te = subjects == s
        mod = make_model("hgb")
        mod.fit(Xd[~te], y[~te])
        oofd[te] = mod.predict_proba(Xd[te])[:, 1]
    met = metrics(y, oofd)
    rows.append({"protocol": f"device_{dev}", "model": "hgb", **met,
                 "auc_ci_lo": None, "auc_ci_hi": None})
pd.DataFrame(rows).to_csv(RES / "main_results.csv", index=False)

# ROC curve points for the figure (HGB, pooled OOF)
fpr, tpr, _ = roc_curve(y, oof["hgb"])
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(RES / "roc_hgb.csv", index=False)
np.save(RES.parent / "scratch" / "oof_hgb.npy", oof["hgb"])

# ---------------- P2 LOFO ----------------
rows = []
fall_types = meta["fall_types"]
for ft in fall_types:
    te_mask = (w["code"] == ft) & (w["is_fall_trial"] == 1)
    # negatives: all ADL windows + pre/post-impact windows of held-out type
    tr_mask = ~(w["is_fall_trial"] == 1) | ~((w["code"] == ft))
    tr = tr_mask & ~te_mask
    mod = make_model("hgb")
    mod.fit(X[tr], y[tr])
    te_idx = np.where(te_mask | ((~tr) & (~te_mask)))[0]
    te_idx = np.where((w["is_fall_trial"] == 1) & (w["code"] == ft))[0]
    ys = mod.predict_proba(X[te_idx])[:, 1]
    # recall at the global LOSO threshold and at FPR-matched operating point:
    # threshold from ADL windows scored by the same model
    adl_idx = np.where(w["is_fall_trial"] == 0)[0]
    adl_scores = mod.predict_proba(X[adl_idx])[:, 1]
    thr = np.quantile(adl_scores, 0.95)
    rows.append({"fall_type": ft, "n_windows": int(len(te_idx)),
                 "recall_at_thr95": float((ys >= thr).mean()),
                 "mean_score": float(ys.mean()),
                 "median_score": float(np.median(ys))})
pd.DataFrame(rows).to_csv(RES / "lofo_types.csv", index=False)

# ---------------- P3 age transfer -> moved to 02b_age_transfer.py
# (SisFall design: falls only from young subjects; see 02b)

# ---------------- feature importance (HGB, full-data fit) ----------------
mod = make_model("hgb")
mod.fit(X, y)
from sklearn.inspection import permutation_importance
pi = permutation_importance(mod, X[:20000], y[:20000], n_repeats=5,
                            random_state=CFG["seed"], n_jobs=-1)
imp = pd.DataFrame({"feature": feat, "importance": pi.importances_mean,
                    "std": pi.importances_std}).sort_values("importance", ascending=False)
imp.to_csv(RES / "feature_importance.csv", index=False)

(RES / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print(json.dumps(meta, indent=1))
print("done")
