"""Step 3, B1 on test: refit the frozen B1 (PCA 64 components as in step 1, GPRs for the first 32 scores,
identical kernels / seeds) because step 1 saved no fitted model, check that the validation predictions
reproduce results/step1/runs/B1_r32/val_pred.npz, then predict test.
Writes results/step3/test/B1_r32/{metrics.json, test_pred.npz}. Usage: python scripts/step3_b1_test.py
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

RANK = 32
tr, va, te = data.load("train"), data.load("validation"), data.load("test", allow_test_labels=True)
f = tr["frequency_GHz"]
flat = lambda S: np.concatenate([S[..., 0].real, S[..., 0].imag, S[..., 1].real, S[..., 1].imag], axis=1)  # noqa: E731


def unflat(Y):
    F_ = len(f)
    return np.stack([Y[:, :F_] + 1j * Y[:, F_:2 * F_], Y[:, 2 * F_:3 * F_] + 1j * Y[:, 3 * F_:]], axis=-1)


pca = PCA(n_components=64, random_state=2026).fit(flat(tr["S"]))
Z = pca.transform(flat(tr["S"]))
pv, pt = [], []
t0 = time.time()
for c in range(RANK):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(9), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-4, (1e-10, 1e-1))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1, random_state=2026)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(tr["X"], Z[:, c])
    pv.append(gp.predict(va["X"]))
    pt.append(gp.predict(te["X"]))
    print(f"component {c + 1:2d} ({time.time() - t0:.0f}s)", flush=True)
S_va = unflat(pca.mean_ + np.stack(pv, 1) @ pca.components_[:RANK])
S_te = unflat(pca.mean_ + np.stack(pt, 1) @ pca.components_[:RANK])
repro = float(np.abs(S_va - np.load(data.ROOT / "results" / "step1" / "runs" / "B1_r32" / "val_pred.npz")["pred_S"]).max())
res = metrics.evaluate(f, te["Y_dB"], metrics.to_db(S_te), te["single_band"], metrics.standard_groups(te), te["S"], S_te)
out = data.ROOT / "results" / "step3" / "test" / "B1_r32"
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps({"tag": "B1_r32", "model": "B1", "rank": RANK, "seed": None,
                                              "validation_reproduction_max_abs_dS": repro, "test": res}, indent=2), encoding="utf-8")
np.savez_compressed(out / "test_pred.npz", design_id=te["design_id"], pred_dB=metrics.to_db(S_te), pred_S=S_te)
print("validation reproduction max |dS|", repro)
print("TEST B1_r32", {k: round(res["all"][k], 4) for k in metrics.GOALS}, "composite", round(res["all"]["composite_score"], 4))
