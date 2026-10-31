"""EJSS figures for the fall-detection generalization paper. PNG 600dpi + PDF."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
FIG = BASE / "figures" / "ejss"
FIG.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
COL1, COL2 = 85 * MM, 174 * MM
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 600, "figure.dpi": 200,
})
C = {"blue": "#0072B2", "sky": "#56B4E9", "green": "#009E73", "orange": "#E69F00",
     "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
flow = json.loads((RES / "flow_counts.json").read_text())
meta = json.loads((RES / "meta.json").read_text())


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / (name.rsplit(".", 1)[0] + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


# ---------------- Figure 1: flow ----------------
fig, ax = plt.subplots(figsize=(COL2, 2.0))
ax.axis("off")
ax.set_xlim(0, 720)
ax.set_ylim(0, 180)


def fbox(x, y, w, h, lines, fc="#F2F2F2"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="0.1", lw=0.7))
    n = len(lines)
    for i, (txt, weight, fs) in enumerate(lines):
        ax.text(x + w / 2, y + h * (n - i - 0.5) / n, txt, ha="center", va="center",
                fontsize=fs, fontweight=weight, color="#111111" if weight == 500 else "#333333")


def farrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.1"))


fbox(15, 110, 180, 55, [("SisFall (200 Hz, two IMUs)", 500, 7.0),
                        (f"{flow['subjects']} subjects (23 young, 15 elderly)", 400, 6.2),
                        (f"4,500 trials analysed", 400, 6.2)])
fbox(245, 110, 200, 55, [("1-s windows, 50% overlap", 500, 7.0),
                         (f"{meta['windows']:,} windows", 400, 6.4),
                         (f"{meta['fall_windows']:,} fall windows", 400, 6.4),
                         ("pre/post-impact kept as hard negatives", 400, 6.2)])
fbox(495, 110, 210, 55, [("Protocols", 500, 7.0),
                         ("P1 leave-one-subject-out", 400, 6.2),
                         ("P2 leave-one-FALL-TYPE-out", 400, 6.2),
                         ("P3 age transfer  P4 device", 400, 6.2)])
farrow(195, 137, 245, 137)
farrow(445, 137, 495, 137)
fbox(15, 25, 180, 55, [("Labels", 500, 7.0),
                       (f"ADL trials: {flow['adl_trials']:,}", 400, 6.2),
                       (f"fall trials: {flow['fall_trials']:,} (15 types)", 400, 6.2),
                       ("impact window +/- 0.25 s = positive", 400, 6.2)])
fbox(245, 25, 200, 55, [("Models", 500, 7.0),
                        ("rule: impact>2.5g + tilt>60 deg", 400, 6.2),
                        ("logit / random forest / HGB", 400, 6.2),
                        ("class-balanced training", 400, 6.2)])
farrow(340, 80, 340, 110)
save(fig, "fig1_flow.png")

# ---------------- Figure 2: ROC ----------------
roc = pd.read_csv(RES / "roc_hgb.csv")
main = pd.read_csv(RES / "main_results.csv")
fig, ax = plt.subplots(figsize=(COL1 + 10 * MM, 62 * MM))
ax.plot(roc["fpr"], roc["tpr"], color=C["blue"], lw=1.1,
        label="HGB (LOSO, AUC "
              f"{main[(main.protocol == 'LOSO') & (main.model == 'hgb')]['roc_auc'].iloc[0]:.3f})")
ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls=":")
ax.set_xlabel("False-positive rate (window level)")
ax.set_ylabel("True-positive rate")
ax.legend(loc="lower right")
save(fig, "fig2_roc.png")

# ---------------- Figure 3: LOFO per fall type ----------------
lofo = pd.read_csv(RES / "lofo_types.csv")
fig, ax = plt.subplots(figsize=(COL2, 2.4))
x = np.arange(len(lofo))
ax.bar(x, lofo["recall_at_thr95"] * 100, width=0.6, color=C["verm"], edgecolor="white",
       lw=0.3)
ax.axhline(np.median(lofo["recall_at_thr95"]) * 100, color="0.2", lw=0.8, ls="--")
ax.text(len(lofo) - 0.5, np.median(lofo["recall_at_thr95"]) * 100 + 1.2,
        f"median {np.median(lofo['recall_at_thr95']) * 100:.1f}%", fontsize=6.0, ha="right")
ax.set_xticks(x, lofo["fall_type"])
ax.set_xlabel("Held-out fall type (detector never trained on it)")
ax.set_ylabel("Fall recall at 5% FPR (%)")
save(fig, "fig3_lofo.png")

# ---------------- Figure 4: cross-age deployment ----------------
age = pd.read_csv(RES / "age_transfer.csv")
AV = {r["metric"]: r["value"] for _, r in age.iterrows()}
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
ax = axes[0]
labels = ["Young ADL", "Elderly ADL", "SE06 ADL"]
vals = [100 * AV["young ADL FPR"], 100 * AV["elderly ADL FPR"],
        100 * AV["SE06 ADL FPR"]]
ax.bar(range(3), vals, width=0.55, color=[C["blue"], C["verm"], C["sky"]],
       edgecolor="white")
for i, v in enumerate(vals):
    ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=6.2)
ax.set_xticks(range(3), labels)
ax.set_ylabel("False-alarm rate at fixed threshold (%)")
ax = axes[1]
labels = ["Young fall windows", "SE06 fall windows"]
vals = [100 * AV["young fall-window recall"],
        100 * AV["SE06 (single elderly faller) fall-window recall"]]
ax.bar(range(2), vals, width=0.45, color=[C["green"], C["verm"]], edgecolor="white")
for i, v in enumerate(vals):
    ax.text(i, v + 2.5, f"{v:.1f}", ha="center", fontsize=6.2)
ax.set_xticks(range(2), labels)
ax.set_ylabel("Fall-window recall (%)")
ax.set_ylim(0, 110)
fig.tight_layout(w_pad=1.5)
save(fig, "fig4_age.png")

# ---------------- Figure 5: feature importance ----------------
imp = pd.read_csv(RES / "feature_importance.csv").head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(COL1 + 12 * MM, 52 * MM))
ax.barh(imp["feature"], imp["importance"], color=C["purple"], edgecolor="white", lw=0.3)
ax.set_xlabel("Permutation importance (AUC drop)")
save(fig, "fig5_importance.png")

print("all figures done")
