"""Build tables.json + narrative_numbers.json for the fall-detection paper."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / ".." / "results"

flow = json.loads((RES / "flow_counts.json").read_text())
meta = json.loads((RES / "meta.json").read_text())
main = pd.read_csv(RES / "main_results.csv")
age = pd.read_csv(RES / "age_transfer.csv")
lofo = pd.read_csv(RES / "lofo_types.csv")
imp = pd.read_csv(RES / "feature_importance.csv")


def r3(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def r1(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def r2(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def fp(p):
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


out = {}

out["table1"] = {
    "header": ["", "Subjects", "Trials", "Windows"],
    "widths": [2400, 1400, 1400, 1400],
    "rows": [
        ["Young (SA01-SA23)", "23", "", f"{meta['young_windows']:,}"],
        ["Elderly (SE01-SE15)", "15", "", f"{meta['elderly_windows']:,}"],
        ["All", f"{meta['subjects']}", f"{flow['fall_trials'] + flow['adl_trials']:,}",
         f"{meta['windows']:,}"],
    ],
    "note": ("SisFall trials annotated as 19 activities of daily living (2,702 trials) "
             "and 15 fall types (1,798 trials), two IMUs at 200 Hz. Windows: 1 s, 50% "
             "overlap; a fall-trial window is positive only if it overlaps the impact "
             "(peak resultant acceleration +/- 0.25 s), other fall-trial windows are "
             "kept as hard negatives."),
}

MLAB = {"logit": "Logistic regression", "rf": "Random forest",
        "hgb": "Gradient boosting", "rule": "Rule baseline (2.5 g + tilt)"}
rows = []
for _, s_ in main[main["protocol"] == "LOSO"].iterrows():
    rows.append([MLAB[s_["model"]], r3(s_["roc_auc"]),
                 f"{r3(s_['auc_ci_lo'])} to {r3(s_['auc_ci_hi'])}",
                 r3(s_["pr_auc"]), r1(100 * s_["recall_at_fpr05"]),
                 r1(100 * s_["f1_max"])])
out["table2"] = {
    "header": ["Model", "ROC-AUC", "95% CI", "PR-AUC", "Recall@5%FPR (%)", "F1 (%)"],
    "widths": [2200, 1100, 1700, 1000, 1500, 900],
    "rows": rows,
    "note": ("Leave-one-subject-out protocol pooled over 38 folds (window level). "
             "Recall@5%FPR: fall recall when the false-positive rate is fixed at 5%. "
             "CIs are subject-level bootstrap (500 resamples)."),
}

rows = []
for _, s_ in lofo.iterrows():
    rows.append([s_["fall_type"], f"{int(s_['n_windows']):,}",
                 r1(100 * s_["recall_at_thr95"]), r3(s_["mean_score"])])
med = lofo["recall_at_thr95"].median()
rows.append(["Median over types", "", r1(100 * med), ""])
out["table3"] = {
    "header": ["Held-out fall type", "Windows", "Recall@5%FPR (%)", "Mean score"],
    "widths": [2200, 1400, 2000, 1400],
    "rows": rows,
    "note": ("Leave-one-fall-type-out: the detector is trained with one of the 15 fall "
             "types entirely absent and tested on it at the same 5% false-positive "
             "operating point (threshold from ADL windows scored by the same model). "
             "Fall types: F01-F03 slip while walking; F04-F07 fainting while walking; "
             "F08-F10 falls when trying to sit; F11-F13 falls when trying to stand; "
             "F14-F15 vertical/forward fainting falls."),
}

rows = []
at = pd.read_csv(RES / "age_transfer.csv")
AV = {r["metric"]: r["value"] for _, r in at.iterrows()}
rows = [
    ["Young ADL false-alarm rate", r2(100 * AV["young ADL FPR"]) + "%"],
    ["Elderly ADL false-alarm rate", r2(100 * AV["elderly ADL FPR"]) + "%"],
    ["Inflation ratio (elderly / young)", r2(AV["elderly ADL FPR ratio (elderly/young)"]) + "x"],
    ["SE06 (single elderly faller) ADL false-alarm rate",
     r2(100 * AV["SE06 ADL FPR"]) + "%"],
    ["Young fall-window recall at the same threshold",
     r1(100 * AV["young fall-window recall"]) + "%"],
    ["SE06 fall-window recall at the same threshold",
     r1(100 * AV["SE06 (single elderly faller) fall-window recall"]) + "%"],
]
out["table4"] = {
    "header": ["Cross-age deployment metric", "Value"],
    "widths": [4200, 1600],
    "rows": rows,
    "note": ("Cross-age deployment test. SisFall's falls were simulated by the 23 "
             "young subjects; the 15 elderly subjects performed daily activities only "
             "(SE06 is the single exception, contributing 15 fall trials). Threshold: "
             "95th percentile of the LOSO score on young ADL windows. All values from "
             "results/age_transfer.csv."),
}

DLAB = {"device_waist_only": "Waist-unit features only",
        "device_chest_only": "Chest-unit features only"}
rows = []
for v, lab in DLAB.items():
    s_ = main[main["protocol"] == v]
    if len(s_) == 0:
        continue
    s_ = s_.iloc[0]
    rows.append([lab, r3(s_["roc_auc"]), r3(s_["pr_auc"]),
                 r1(100 * s_["recall_at_fpr05"])])
out["table5"] = {
    "header": ["Sensor-position variant", "ROC-AUC", "PR-AUC", "Recall@5%FPR (%)"],
    "widths": [3000, 1300, 1300, 1700],
    "rows": rows,
    "note": ("Device-position transfer: the same gradient-boosting protocol restricted "
             "to the features of one body-worn unit (waist ADXL345 vs chest MMA8451Q), "
             "leave-one-subject-out."),
}

(ROOT / "tables.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
age = pd.read_csv(RES / "age_transfer.csv")
narr = {"meta": meta, "flow": flow, "main": main.to_dict("records"),
        "age": age.to_dict("records"), "lofo": lofo.to_dict("records"),
        "imp": imp.to_dict("records")}
(RES / "narrative_numbers.json").write_text(json.dumps(narr, indent=1), encoding="utf-8")
print("tables.json + narrative_numbers.json written")
