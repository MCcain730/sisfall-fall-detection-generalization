# -*- coding: utf-8 -*-
"""02b_age_transfer.py — Cross-age deployment test.

SisFall design fact: falls were simulated ONLY by young subjects (SA01-23);
elderly subjects (SE01-15) performed ADLs only (SE06 contributes 15 fall
trials, kept as a single-faller case study). The deployment question is
therefore: does a fall detector trained on young subjects raise false alarms
on elderly daily activity, and does it still catch an elderly faller?

Outputs results/age_transfer.csv with:
  young_adl_fpr, elderly_adl_fpr at the young-ADL 95th-percentile threshold,
  se06_fall_recall (single elderly faller), se06_adl_fpr.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

w = pd.read_parquet(ROOT / "data" / "harmonised" / "windows.parquet")
oof = np.load(ROOT / "scratch" / "oof_hgb.npy")  # pooled LOSO scores, row-aligned
w["score"] = oof
w["age_group"] = np.where(w["subject"].str.startswith("SA"), "young", "elderly")
w["is_young_faller"] = w["age_group"].eq("young") & w["is_fall_trial"].eq(1)

young_adl = w[(w["age_group"] == "young") & (w["is_fall_trial"] == 0)]
thr = float(np.quantile(young_adl["score"], 0.95))

elderly_adl = w[(w["age_group"] == "elderly") & (w["is_fall_trial"] == 0)]
se06_falls = w[(w["subject"] == "SE06") & (w["is_fall_trial"] == 1)]
se06_all = w[w["subject"] == "SE06"]
young_fall_windows = w[(w["age_group"] == "young") & (w["label"] == 1)]

rows = [
    {"metric": "threshold (young ADL 95th pct)", "value": thr},
    {"metric": "young ADL FPR", "value": float((young_adl["score"] >= thr).mean())},
    {"metric": "elderly ADL FPR", "value": float((elderly_adl["score"] >= thr).mean())},
    {"metric": "elderly ADL FPR ratio (elderly/young)",
     "value": float((elderly_adl["score"] >= thr).mean() /
                    max((young_adl["score"] >= thr).mean(), 1e-9))},
    {"metric": "young fall-window recall", "value": float((young_fall_windows["score"] >= thr).mean())},
    {"metric": "SE06 (single elderly faller) fall-window recall",
     "value": float((se06_falls["score"] >= thr).mean())},
    {"metric": "SE06 n fall windows", "value": int(len(se06_falls))},
    {"metric": "SE06 ADL FPR", "value": float((se06_all[se06_all["is_fall_trial"] == 0]["score"] >= thr).mean())},
]
pd.DataFrame(rows).to_csv(RES / "age_transfer.csv", index=False)
# score distributions for the figure
pd.DataFrame({
    "group": (["young ADL"] * len(young_adl) + ["elderly ADL"] * len(elderly_adl) +
              ["SE06 fall windows"] * len(se06_falls)),
    "score": np.concatenate([young_adl["score"].to_numpy(),
                             elderly_adl["score"].to_numpy(),
                             se06_falls["score"].to_numpy()]),
}).to_csv(RES / "age_scores.csv", index=False)
print(json.dumps(rows, indent=1))
