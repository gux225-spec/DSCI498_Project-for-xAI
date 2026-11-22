"""Figures for the Nov 10 progress talk, from existing results only (no model runs, no ADS)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"E:\PycharmProject\DSCI498_Project for xAI")
OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
from filterxai import data, targets, xai  # noqa: E402

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, CRIT = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

CASE = 25  # row in the 40 explained validation designs: v2_validation_0063
va = data.load("validation")
f = va["frequency_GHz"]
setup = np.load(ROOT / "results/step2/setup.npz")
d = setup["explain_idx"][CASE]
TAG = "PRNet_lam0.3_s2026_b16_lossless10.0"
vp = np.load(ROOT / f"results/step1/runs/{TAG}/val_pred.npz")
y, yp = va["Y_dB"][d], vp["pred_dB"][d]
R, M = data.REGIONS_GHZ, targets.MASK
info = {"design_id": str(va["design_id"][d])}

# ---------- Figure 1: response vs specification (spec-first view) ----------
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=220)
ax.plot(f, y[:, 0], color=BLUE, lw=2, label="S21 (ADS)")
ax.plot(f, yp[:, 0], color=BLUE, lw=1.6, ls=(0, (4, 3)), label="S21 (PR-Net)")
ax.plot(f, y[:, 1], color=AQUA, lw=2, label="S11 (ADS)")
lim = [(R["low_stopband"], M["low_stopband_S21_max"], "S21 ≤ −20"), (R["passband"], M["passband_S21_min"], "S21 ≥ −0.5"),
       (R["passband"], M["passband_S11_max"], "S11 ≤ −10"), (R["high_stopband"], M["high_stopband_S21_max"], "S21 ≤ −15")]
for (a, b), v, _ in lim:
    ax.plot([a, b], [v, v], color=INK, lw=1.6)
hs = (f >= R["high_stopband"][0]) & (f <= R["high_stopband"][1])
ax.fill_between(f[hs], M["high_stopband_S21_max"], y[hs, 0], where=y[hs, 0] > M["high_stopband_S21_max"],
                color=CRIT, alpha=0.25, lw=0)
k = np.flatnonzero(hs)[np.argmax(y[hs, 0])]
viol = float(y[k, 0] - M["high_stopband_S21_max"])
info.update(high_stop_peak_GHz=float(f[k]), high_stop_peak_dB=float(y[k, 0]), high_stop_violation_dB=viol)
exceed = f[hs][y[hs, 0] > M["high_stopband_S21_max"]]
info.update(exceed_from_GHz=float(exceed.min()), exceed_to_GHz=float(exceed.max()))
ax.annotate(f"{viol:.1f} dB over limit\nat {f[k]:.2f} GHz", xy=(f[k] + 0.12, y[k, 0] - 4), xytext=(13.45, -46),
            color=CRIT, fontsize=13, fontweight="bold", arrowprops=dict(arrowstyle="-", color=CRIT, lw=1.2))
ax.text(9.05, M["low_stopband_S21_max"] + 1.2, "limit −20 dB", color=INK2, fontsize=12)
ax.text(14.0, M["high_stopband_S21_max"] + 1.2, "limit −15 dB", color=INK2, fontsize=12)
ax.set_xlim(9, 15); ax.set_ylim(-60, 3)
ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
ax.legend(loc="lower left", frameon=False, fontsize=12, ncol=1)
fig.tight_layout(); fig.savefig(OUT / "fig_spec_first.png"); plt.close(fig)

# ---------- Figure 2: three explainers on the same case and target ----------
A = np.load(ROOT / f"results/step2/attr/{TAG}.npz")
names = list(A["names"])
t = names.index("T2_mask_violation_dB")
x_ref = setup["x_ref"]
shap = A["shap_ref"][CASE, t]
ig = A["ig"][CASE, t]
lime_d = A["lime_coef"][0][CASE, t] * (A["x"][CASE] - x_ref)
feat = ["L1", "L2", "S1", "S2", "Wc1", "Wc2", "ctrl U", "ctrl N", "ctrl M"]
order = np.argsort(-np.abs(shap))
info.update(F_x=float(A["F_x"][CASE, t]), F_ref=float(A["F_ref"][t]), shap=dict(zip(feat, shap.round(3).tolist())),
            ig=dict(zip(feat, ig.round(3).tolist())), lime_delta=dict(zip(feat, lime_d.round(3).tolist())),
            lime_r2=float(A["lime_r2"][0][CASE, t]), ig_resid=float(A["ig_resid"][CASE, t]))
fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=220)
yy = np.arange(len(feat))
h = 0.26
for i, (v, c, lab) in enumerate([(shap, BLUE, "SHAP"), (ig, ORANGE, "IG"), (lime_d, AQUA, "LIME-delta")]):
    ax.barh(yy + (i - 1) * h, v[order], height=h * 0.9, color=c, label=lab)
ax.set_yticks(yy, [feat[j] for j in order]); ax.invert_yaxis()
ax.axvline(0, color=BASE, lw=1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Contribution to predicted mask violation (dB, vs reference design)")
ax.legend(frameon=False, fontsize=12, loc="lower right")
ax.tick_params(axis="y", colors=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig_three_methods.png"); plt.close(fig)

# ---------- Figure 3: deletion curve, SHAP order vs 100 random orders ----------
v = A["v_ref"][CASE, :, t]
full = (1 << 9) - 1
gap = v[full] - v[0]


def deletion(order):
    bits = np.concatenate([[0], np.cumsum(1 << np.asarray(order))])
    return (v[full - bits] - v[0]) / gap


rng = np.random.default_rng(0)
rand = np.stack([deletion(rng.permutation(9)) for _ in range(100)])
cur = deletion(np.argsort(-shap * np.sign(gap)))
kx = np.arange(10)
fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=220)
ax.fill_between(kx, np.percentile(rand, 5, 0), np.percentile(rand, 95, 0), color=MUTED, alpha=0.22, lw=0,
                label="100 random orders (5–95%)")
ax.plot(kx, np.median(rand, 0), color=MUTED, lw=1.6, ls=(0, (4, 3)), label="random median")
ax.plot(kx, cur, color=BLUE, lw=2.2, marker="o", ms=6, label="SHAP order")
ax.set_xlabel("Inputs reset to reference (most important first)")
ax.set_ylabel("Remaining share of output gap")
ax.set_xticks(kx); ax.set_ylim(-0.1, max(1.1, float(np.percentile(rand, 95, 0).max()) + 0.05))
ax.legend(frameon=False, fontsize=11.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.6)
fig.tight_layout(); fig.savefig(OUT / "fig_deletion.png"); plt.close(fig)
info["deletion_auc_shap"] = float(np.trapezoid(cur, dx=1 / 9))
info["deletion_auc_random_median"] = float(np.median(np.trapezoid(rand, dx=1 / 9, axis=1)))
(OUT / "case_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
print(json.dumps(info, indent=1))
