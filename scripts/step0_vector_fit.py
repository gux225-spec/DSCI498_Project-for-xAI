"""Step 0: vector-fitting concept labels for train and validation (N=10 shared poles).

For each design, S21 and S11 are fitted with 5 shared conjugate pole pairs. Outputs:
  results/step0/vf_concepts_{split}.npz   poles, residues, constants, natural frequency, Q, fit errors
  results/step0/vf_summary.json           fit quality, pole-structure counts, PR-Net init (train only)
  results/step0/vf_poles_train.png        pole map
Test is not touched. Usage: python scripts/step0_vector_fit.py
"""
from pathlib import Path
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, vectfit  # noqa: E402

N_POLES = 10
IN_BAND = (9.0, 15.0)
Q_RESONATOR = 5.0
out_dir = data.ROOT / "results" / "step0"
out_dir.mkdir(parents=True, exist_ok=True)


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-15))


def fit_split(split):
    d = data.load(split)
    f, S = d["frequency_GHz"], d["S"]
    pb = data.region_masks(f)["passband"]
    n = len(S)
    poles = np.zeros((n, N_POLES), complex)
    res = np.zeros((n, 2, N_POLES), complex)
    const = np.zeros((n, 2))
    err = {k: np.zeros(n) for k in ("max_abs_dS", "S21_dB_MAE", "S11_dB_MAE", "S11_passband_dB_MAE", "S21_passband_dB_MAE")}
    n_cplx = np.zeros(n, int)
    t0 = time.time()
    for i in range(n):
        p, r, c, Hf = vectfit.vector_fit(f, S[i], N_POLES)
        poles[i], res[i], const[i] = p, r, c
        n_cplx[i] = int((p.imag > 0).sum())
        err["max_abs_dS"][i] = np.abs(Hf - S[i]).max()
        err["S21_dB_MAE"][i] = np.abs(db(Hf[:, 0]) - d["Y_dB"][i, :, 0]).mean()
        err["S11_dB_MAE"][i] = np.abs(db(Hf[:, 1]) - d["Y_dB"][i, :, 1]).mean()
        err["S11_passband_dB_MAE"][i] = np.abs(db(Hf[pb, 1]) - d["Y_dB"][i, pb, 1]).mean()
        err["S21_passband_dB_MAE"][i] = np.abs(db(Hf[pb, 0]) - d["Y_dB"][i, pb, 0]).mean()
    elapsed = time.time() - t0
    # upper-half-plane poles sorted by natural frequency -> one row of 5 "concepts" per design
    fn, q = vectfit.natural_frequency_q(poles)
    upper = np.where(poles.imag > 0, 1, 0).astype(bool)
    fn_u = np.full((n, N_POLES // 2), np.nan)
    q_u = np.full((n, N_POLES // 2), np.nan)
    for i in range(n):
        k = np.flatnonzero(upper[i])
        k = k[np.argsort(fn[i, k])]
        fn_u[i, :len(k)], q_u[i, :len(k)] = fn[i, k], q[i, k]
    resonators = np.sum((fn_u >= IN_BAND[0]) & (fn_u <= IN_BAND[1]) & (q_u >= Q_RESONATOR), axis=1)
    np.savez_compressed(out_dir / f"vf_concepts_{split}.npz", design_id=d["design_id"], poles=poles,
                        residues=res, constants=const, natural_frequency_GHz=fn_u, Q=q_u,
                        n_complex_pairs=n_cplx, n_in_band_resonators=resonators, source=d["source"],
                        single_band=d["single_band"], mask_pass=d["mask_pass"], **err)
    return d, err, fn_u, q_u, n_cplx, resonators, elapsed


def q(x, ps=(0.5, 0.95, 1.0)):
    x = x[np.isfinite(x)]
    return {f"p{int(p * 100)}": float(np.quantile(x, p)) for p in ps}


summary = {"N_poles": N_POLES, "in_band_resonator_rule": f"natural frequency in {IN_BAND} GHz and Q >= {Q_RESONATOR}"}
store = {}
for split in ("train", "validation"):
    d, err, fn_u, q_u, n_cplx, reso, elapsed = fit_split(split)
    store[split] = (d, fn_u, q_u, reso)
    groups = {"all": np.ones(len(reso), bool), "single_band": d["single_band"], "multiband": ~d["single_band"],
              "legacy_4d": d["source"] == "legacy_4d_actual", "new_6d": d["source"] == "new_6d_actual"}
    summary[split] = {
        "designs": int(len(reso)), "seconds": round(elapsed, 1),
        "all_5_pairs_complex": int((n_cplx == 5).sum()),
        "in_band_resonator_count": {str(k): int((reso == k).sum()) for k in np.unique(reso)},
        "fit_error": {g: {k: q(v[m]) for k, v in err.items()} for g, m in groups.items()},
        "worst_designs_S11_passband": [
            {"design_id": str(d["design_id"][i]), "S11_passband_dB_MAE": float(err["S11_passband_dB_MAE"][i]),
             "single_band": bool(d["single_band"][i]), "resonators": int(reso[i])}
            for i in np.argsort(err["S11_passband_dB_MAE"])[::-1][:5]],
    }
    print(split, json.dumps({k: summary[split][k] for k in ("designs", "seconds", "all_5_pairs_complex", "in_band_resonator_count")}))
    print("  fit (all):", json.dumps({k: round(v["p50"], 4) for k, v in summary[split]["fit_error"]["all"].items()}),
          "worst:", json.dumps({k: round(v["p100"], 4) for k, v in summary[split]["fit_error"]["all"].items()}))

# Concept labels must not depend on the VF starting point: refit validation from a different init.
d, fn_ref, q_ref, _ = store["validation"]
beta = 2 * np.pi * np.linspace(7.0, 17.0, N_POLES // 2)
p0 = np.ravel(np.column_stack([-0.05 * beta + 1j * beta, -0.05 * beta - 1j * beta]))
dfn, dq = [], []
for i in range(len(d["S"])):
    p, *_ = vectfit.vector_fit(d["frequency_GHz"], d["S"][i], N_POLES, iters=30, poles=p0)
    fn, qq = vectfit.natural_frequency_q(p[p.imag > 0])
    o = np.argsort(fn)
    dfn.append(np.abs(fn[o] - fn_ref[i]))
    dq.append(np.abs(qq[o] / q_ref[i] - 1))
summary["init_robustness_validation"] = {
    "alt_init": "5 pairs over 7-17 GHz, damping 0.05, 30 iterations",
    "max_abs_dfn_MHz": (1e3 * np.max(dfn, 0)).tolist(), "max_rel_dQ": np.max(dq, 0).tolist()}
print("init robustness:", json.dumps(summary["init_robustness_validation"]))

# Sorting by frequency gives stable identities only if the in-band poles never cross.
_, fn_u, q_u, _ = store["train"]
summary["pole_identity_train"] = {
    "min_gap_MHz_pole3_minus_pole2": float(1e3 * np.min(fn_u[:, 2] - fn_u[:, 1])),
    "min_gap_MHz_pole4_minus_pole3": float(1e3 * np.min(fn_u[:, 3] - fn_u[:, 2])),
    "fraction_middle_resonator_lowest_Q": float(np.mean(q_u[:, 2] < np.minimum(q_u[:, 1], q_u[:, 3])))}
print("pole identity:", json.dumps(summary["pole_identity_train"]))

# PR-Net initialization from train only: per sorted pole index
summary["prnet_init_train"] = {
    f"pole_{k + 1}": {"natural_frequency_GHz": q(fn_u[:, k], (0.01, 0.5, 0.99)), "Q": q(q_u[:, k], (0.01, 0.5, 0.99))}
    for k in range(N_POLES // 2)}
(out_dir / "vf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary["prnet_init_train"], indent=1))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(7, 4.5))
d, fn_u, q_u, _ = store["train"]
for k in range(N_POLES // 2):
    for sb, mk in ((True, "o"), (False, "x")):
        m = d["single_band"] == sb
        ax.scatter(fn_u[m, k], q_u[m, k], s=8, marker=mk, alpha=0.5,
                   label=f"pole {k + 1}" + ("" if sb else " (multiband)"), color=f"C{k}")
ax.set_yscale("log")
ax.axvspan(*IN_BAND, color="0.92", zorder=-1)
ax.set_xlabel("natural frequency |p|/2π (GHz)")
ax.set_ylabel("Q")
ax.set_title("Vector-fitting poles, train designs (N=10, sorted by frequency)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(out_dir / "vf_poles_train.png", dpi=150)
