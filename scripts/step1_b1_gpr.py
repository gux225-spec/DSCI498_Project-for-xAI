"""Step 1, B1: PCA of complex curves (train only) + one ARD Matern-2.5 GPR per PCA score.

Inputs are the same 9 normalized inputs as the neural models. PCA components are nested,
so ranks 16/32 reuse the first GPRs fitted for rank 64. Validation only.
Writes results/step1/runs/B1_r{rank}/metrics.json and val_pred.npz.
Usage: python scripts/step1_b1_gpr.py
"""
from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, metrics  # noqa: E402

RANKS = (16, 32, 64)
tr, va = data.load("train"), data.load("validation")
f = tr["frequency_GHz"]


def flat(S):  # (N, F, 2) complex -> (N, 4F) = [Re S21, Im S21, Re S11, Im S11]
    return np.concatenate([S[..., 0].real, S[..., 0].imag, S[..., 1].real, S[..., 1].imag], axis=1)


def unflat(Y):
    F_ = len(f)
    return np.stack([Y[:, :F_] + 1j * Y[:, F_:2 * F_], Y[:, 2 * F_:3 * F_] + 1j * Y[:, 3 * F_:]], axis=-1)


pca = PCA(n_components=max(RANKS), random_state=2026).fit(flat(tr["S"]))
Z = pca.transform(flat(tr["S"]))
n_in = tr["X"].shape[1]
preds, kernels = [], []
t0 = time.time()
for c in range(max(RANKS)):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(n_in), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-4, (1e-10, 1e-1))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1, random_state=2026)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(tr["X"], Z[:, c])
    preds.append(gp.predict(va["X"]))
    kernels.append(str(gp.kernel_))
    print(f"component {c + 1:2d} var {pca.explained_variance_ratio_[c]:.2e} ({time.time() - t0:.0f}s) {gp.kernel_}", flush=True)
P = np.stack(preds, 1)

for r in RANKS:
    S_hat = unflat(pca.mean_ + P[:, :r] @ pca.components_[:r])
    res = metrics.evaluate(f, va["Y_dB"], metrics.to_db(S_hat), va["single_band"], metrics.standard_groups(va), va["S"], S_hat)
    out = data.ROOT / "results" / "step1" / "runs" / f"B1_r{r}"
    out.mkdir(parents=True, exist_ok=True)
    recon = unflat(pca.mean_ + pca.transform(flat(va["S"]))[:, :r] @ pca.components_[:r])
    summary = {"tag": f"B1_r{r}", "model": "B1", "rank": r, "seed": None,
               "pca_explained_variance": float(pca.explained_variance_ratio_[:r].sum()),
               "pca_reconstruction_only_S21_MAE_dB": float(np.abs(metrics.to_db(recon) - va["Y_dB"])[..., 0].mean()),
               "training": {"seconds": round(time.time() - t0, 1), "kernels": kernels[:r]}, "validation": res}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(out / "val_pred.npz", design_id=va["design_id"], pred_dB=metrics.to_db(S_hat), pred_S=S_hat)
    a = res["all"]
    print("VAL", f"B1_r{r}", {k: round(a[k], 4) for k in metrics.GOALS}, "composite", round(a["composite_score"], 4), flush=True)
