# -*- coding: utf-8 -*-
"""01_build.py — SisFall raw txt -> windowed feature parquet.

File naming: <CODE>_<SUBJ>_R<rep>.txt where CODE starts with D (ADL) or F (fall).
Sensors (200 Hz): ADXL345 xyz (+/-16 g, LSB 32/8192 g), ITG3200 gyro xyz,
MMA8451Q xyz (+/-8 g, LSB 16/8192 g). Windows 1 s, 50% overlap.
A fall-trial window is labelled fall only if it overlaps the impact region
(peak resultant acceleration +/- 0.25 s); other fall-trial windows are kept
as hard negatives (pre/post impact). Audit trail in results/flow_counts.json.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
FS = CFG["fs"]
W = int(CFG["window_s"] * FS)
STEP = int(W * (1 - CFG["overlap"]))

flow = {}
rows = []
feat_names = None

z0 = zipfile.ZipFile(ROOT / CFG["raw_zip"])
if "SisFall_dataset.zip" in z0.namelist():
    if not (ROOT / "data" / "raw" / "SisFall_dataset.zip").exists():
        z0.extract("SisFall_dataset.zip", ROOT / "data" / "raw")
    z = zipfile.ZipFile(ROOT / "data" / "raw" / "SisFall_dataset.zip")
else:
    z = z0
names = [n for n in z.namelist() if n.lower().endswith(".txt")]
flow["files_raw"] = len(names)

for name in sorted(names):
    base = Path(name).name
    if "_" not in base:
        continue
    parts = base.replace(".txt", "").split("_")
    code, subj = parts[0], parts[1]
    rep = parts[2] if len(parts) > 2 else "R01"
    is_fall = code.upper().startswith("F")
    kind = code.split("_")[0][:3]
    try:
        df = pd.read_csv(io.BytesIO(z.read(name)), header=None, sep=r"[,;\s]+",
                         engine="python", comment=";")
    except Exception:
        flow.setdefault("files_unreadable", []).append(base)
        continue
    if df.shape[1] < 9:
        flow.setdefault("files_unreadable", []).append(base)
        continue
    vals = df.iloc[:, :9].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
    if len(vals) < W:
        flow.setdefault("files_too_short", []).append(base)
        continue
    a1 = vals[:, :3] * CFG["adxl_lsb_g"]     # ADXL345, g
    g1 = vals[:, 3:6] * CFG["gyro_lsb_dps"]  # ITG3200, deg/s
    a2 = vals[:, 6:9] * CFG["mma_lsb_g"]     # MMA8451Q, g
    sm = np.sqrt((a1 ** 2).sum(axis=1))      # resultant accel (ADXL)
    # impact centre: peak resultant within the trial
    impact = int(np.argmax(sm))
    n = len(sm)
    flow["rows_signals"] = flow.get("rows_signals", 0) + n
    starts = list(range(0, n - W + 1, STEP))
    if starts[-1] != n - W:
        starts.append(n - W)
    for s0 in starts:
        s1 = s0 + W
        seg = a1[s0:s1]
        sg = g1[s0:s1]
        seg2 = a2[s0:s1]
        x, y, zz = seg[:, 0], seg[:, 1], seg[:, 2]
        mag = np.sqrt((seg ** 2).sum(axis=1))
        mag2 = np.sqrt((seg2 ** 2).sum(axis=1))
        gz = sg[:, 2]
        # gravity direction ~ mean accel of the WHOLE trial (standing posture proxy)
        grav = a1.mean(axis=0)
        gn = grav / (np.linalg.norm(grav) + 1e-9)
        tilt = np.degrees(np.arccos(np.clip(mag * 0 + (seg @ gn) /
                                            (np.linalg.norm(seg, axis=1) + 1e-9), -1, 1)))
        # features
        feat = [
            mag.mean(), mag.std(), mag.min(), mag.max(), mag.max() - mag.min(),
            np.sqrt((mag ** 2).mean()),                        # RMS
            np.abs(x).sum() + np.abs(y).sum() + np.abs(zz).sum(),  # SMA
            mag.max() - mag.min(),                              # placeholder dup guard
            np.std(np.diff(mag)),                              # jerk proxy
            (np.abs(zz) > 1).mean(),                           # vertical-g fraction
            tilt.mean(), tilt.max(), np.percentile(tilt, 95),
            np.sqrt((sg ** 2).sum(axis=1)).mean(),             # gyro magnitude mean
            np.sqrt((sg ** 2).sum(axis=1)).max(),
            np.std(gz),
            np.sqrt((mag2 ** 2).mean()),                       # second-device RMS
            mag2.max() - mag2.min(),
        ]
        # spectral features on resultant (rFFT)
        f = np.abs(np.fft.rfft(mag * np.hanning(len(mag))))
        freqs = np.fft.rfftfreq(len(mag), 1 / FS)
        tot = f.sum() + 1e-9
        feat += [float(freqs[1:][np.argmax(f[1:])]), float(tot),
                 float(f[(freqs > 0.5) & (freqs <= 3)].sum() / tot),
                 float(f[(freqs > 3) & (freqs <= 8)].sum() / tot)]
        feat = feat[:18] + feat[18:22]
        if feat_names is None:
            feat_names = (["mag_mean", "mag_std", "mag_min", "mag_max", "mag_range",
                           "rms", "sma", "peak_to_peak", "jerk_std", "vert_frac",
                           "tilt_mean", "tilt_max", "tilt_p95", "gyro_mean",
                           "gyro_max", "gyro_std", "rms2", "range2",
                           "dom_freq", "spec_tot", "spec_low", "spec_mid"])
        impact_overlap = (s0 <= impact <= s1) or (abs(impact - (s0 + s1) / 2) <= W / 4)
        label = int(is_fall and impact_overlap)
        rows.append({
            "subject": subj, "code": kind, "is_fall_trial": int(is_fall),
            "rep": rep, "w_start": s0, "label": label,
            "impact_rel": int(impact - s0),
            **{f: float(v) for f, v in zip(feat_names, feat)}})

feat_names = feat_names[:18] + ["dom_freq", "spec_tot", "spec_low", "spec_mid"]
w = pd.DataFrame(rows)
flow["rows_raw"] = len(w)
flow["subjects"] = int(w["subject"].nunique())
flow["fall_windows"] = int(w["label"].sum())
flow["fall_trials"] = int(w.loc[w["is_fall_trial"] == 1].groupby(["subject", "code", "rep"]).ngroups)
flow["adl_trials"] = int(w.loc[w["is_fall_trial"] == 0].groupby(["subject", "code", "rep"]).ngroups)
flow["codes_fall"] = sorted(w.loc[w["is_fall_trial"] == 1, "code"].unique().tolist())
flow["codes_adl"] = sorted(w.loc[w["is_fall_trial"] == 0, "code"].unique().tolist())
flow["files_unreadable_n"] = len(flow.get("files_unreadable", []))
flow["files_too_short_n"] = len(flow.get("files_too_short", []))

w.to_parquet(ROOT / "data" / "harmonised" / "windows.parquet")
(RES / "feature_names.json").write_text(json.dumps(feat_names), encoding="utf-8")
(RES / "flow_counts.json").write_text(json.dumps(flow, indent=1), encoding="utf-8")
meta = {k: flow[k] for k in ["rows_raw", "subjects", "fall_windows", "fall_trials",
                             "adl_trials", "files_raw"]}
print(json.dumps(meta, indent=1))
