"""Pre-planning numbers from TRAIN only (plus diagnostic scenario names). No model selection on val/test.

Usage: python preplan_checks.py <release_dir>
"""
import sys
import numpy as np

rel = sys.argv[1]
D = rel + r"\data"
z = np.load(f"{D}\\core_train.npz", allow_pickle=False)
f = z["frequency_Hz"] / 1e9
Y = z["Y_dB"]
S = z["S_complex"]
g = z["geometry_mil"]
names = ["L1", "L2", "S1", "S2", "Wc1", "Wc2"]

# 1) differentiable passband centroid proxy: linear in geometry?
m = (f >= 10.5) & (f <= 13.5)
w = np.abs(S[:, m, 1, 0]) ** 2
cent = (w * f[m]).sum(1) / w.sum(1)
A = np.c_[np.ones(len(g)), (g - g.mean(0)) / g.std(0)]
coef, *_ = np.linalg.lstsq(A, cent, rcond=None)
r2 = 1 - np.sum((cent - A @ coef) ** 2) / np.sum((cent - cent.mean()) ** 2)
print(f"|S21|^2 centroid 10.5-13.5 GHz: R2={r2:.4f} " + " ".join(f"{n}={c:+.4f}" for n, c in zip(names, coef[1:])))

# 2) PCA reconstruction error on train curves (in-sample; indicates compressibility only)
for ch, tag in ((0, "S21_dB"), (1, "S11_dB")):
    X = Y[:, :, ch]
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = np.cumsum(s ** 2) / np.sum(s ** 2)
    out = []
    for k in (8, 16, 32, 64):
        R = (X - mu) @ Vt[:k].T @ Vt[:k] + mu
        e = np.abs(R - X)
        pb = (f >= 11.6) & (f <= 12.4)
        out.append(f"k={k}: var={ev[k-1]:.5f} MAE={e.mean():.3f} max={e.max():.1f} passMAE={e[:, pb].mean():.3f}")
    print(tag, " | ".join(out))

# 3) complex (real/imag) compressibility for comparison
Xc = np.concatenate([S[:, :, 1, 0].real, S[:, :, 1, 0].imag, S[:, :, 0, 0].real, S[:, :, 0, 0].imag], 1)
mu = Xc.mean(0)
_, s, _ = np.linalg.svd(Xc - mu, full_matrices=False)
ev = np.cumsum(s ** 2) / np.sum(s ** 2)
print("complex RI concat var explained:", {k: round(float(ev[k-1]), 5) for k in (8, 16, 32, 64)})

# 4) diagnostic scenario inventory
d = np.load(f"{D}\\diagnostic_scenarios.npz", allow_pickle=False)
print("diagnostic keys:", {k: d[k].shape for k in d.files})
for k in d.files:
    if d[k].dtype.kind == "U":
        print(k, d[k].tolist())
