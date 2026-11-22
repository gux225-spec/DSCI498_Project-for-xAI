"""Distribution / sensitivity preview on the TRAIN split only (no model fitting beyond OLS).

Usage: python audit_distribution.py <release_dir>
"""
import sys, csv
import numpy as np

rel = sys.argv[1]
D = rel + r"\data"
names = ["L1", "L2", "S1", "S2", "Wc1", "Wc2"]

z = np.load(f"{D}\\core_train.npz", allow_pickle=False)
g = z["geometry_mil"]
src_legacy = np.array([not i.startswith("v2_") for i in z["design_id"]])

print("legacy Wc1 unique:", np.unique(np.round(g[src_legacy, 4], 6)), "Wc2 unique:", np.unique(np.round(g[src_legacy, 5], 6)))
print("v2 geometry corr (train):")
print(np.round(np.corrcoef(g[~src_legacy].T), 3))

with open(f"{D}\\design_metrics.csv", newline="", encoding="utf-8-sig") as fh:
    rows = list(csv.DictReader(fh))
for src in ("legacy_4d_actual", "new_6d_actual"):
    sub = [r for r in rows if r["source"] == src]
    nb, valid = {}, 0
    for r in sub:
        nb[r["n_connected_3dB_bands"]] = nb.get(r["n_connected_3dB_bands"], 0) + 1
        valid += r["single_band_center_valid"] == "True"
    print(src, "n_bands:", nb, "single_band_center_valid:", valid, "/", len(sub))
for sp in ("train", "validation", "test"):
    sub = [r for r in rows if r["split"] == sp]
    print(sp, "mask pass:", sum(r["mask_pass_sampled_grid"] == "True" for r in sub), "/", len(sub))

tr = [r for r in rows if r["split"] == "train" and r["single_band_center_valid"] == "True"]
G = np.array([[float(r[n]) for n in names] for r in tr])
Gs = (G - G.mean(0)) / G.std(0)
A = np.c_[np.ones(len(G)), Gs]
for target in ("f0_arithmetic_GHz", "BW_GHz", "ripple_fixed_pass_dB", "return_loss_fixed_pass_worst_dB"):
    y = np.array([float(r[target]) for r in tr])
    ok = np.isfinite(y)
    coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    pred = A[ok] @ coef
    r2 = 1 - np.sum((y[ok] - pred) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
    print(f"OLS {target:35s} R2={r2:.3f}  std-coef per 1 sd: " + " ".join(f"{n}={c:+.4f}" for n, c in zip(names, coef[1:])))

Y = z["Y_dB"]
f = z["frequency_Hz"] / 1e9
for lo, hi, tag in ((9, 10.8, "low stop"), (11.6, 12.4, "pass"), (13.2, 15, "high stop")):
    m = (f >= lo) & (f <= hi)
    print(f"{tag:9s} S21 mean {Y[:, m, 0].mean():7.2f} sd {Y[:, m, 0].std():5.2f} | S11 mean {Y[:, m, 1].mean():7.2f} sd {Y[:, m, 1].std():5.2f} | S11<-40dB frac {np.mean(Y[:, m, 1] < -40):.3f}")
