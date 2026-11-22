"""Step 2 setup: target checks (train only), fixed explained designs, backgrounds, diagnostic scenarios.

1. Targets on TRAIN true curves: smooth vs hard T2 (log-sum-exp tau fixed in targets.py), agreement with
   the dataset's own metrics, linear R2 on the 6 geometry inputs, T0 standardized OLS ranking, target std.
2. Explained validation designs (plan §4): 40 single-band designs stratified by source x mask pass, seed 2026.
   Validation has very few mask passes, so every single-band pass is taken and the remaining slots go to
   failing designs, split across sources by largest remainder, sampled with seed 2026.
3. Backgrounds from train: lhs32 = 32 new_6d designs (seed 2026), rand32 = 32 any-source designs (seed 2027),
   paper100 = 100 any-source designs (seed 2026, paper protocol).
4. Reference point: ADS baseline geometry, controls at train mean (= 0 after train-only normalization).
5. The 12 one-factor baseline diagnostics (+ baseline) from diagnostic_scenarios.npz: normalized inputs,
   ADS targets on the 601-point core grid, and vector-fitting concepts (N=10, as in step 0).
Writes results/step2/setup.{json,npz}. Usage: python scripts/step2_setup.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, targets, vectfit  # noqa: E402

OUT = data.ROOT / "results" / "step2"
OUT.mkdir(parents=True, exist_ok=True)
N_EXPLAIN, SEED = 40, 2026
tr, va = data.load("train"), data.load("validation")
f = tr["frequency_GHz"]
names = tr["feature_names"]
T = targets.CurveTargets(f)


def curve_targets(db):
    with torch.no_grad():
        return T(torch.tensor(db, dtype=torch.float32)).numpy()


def r2_ols(X, y):
    A = np.c_[np.ones(len(X)), X]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return 1 - np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2), coef[1:]


# 1. target checks on train
Ytr = curve_targets(tr["Y_dB"])
hard = targets.hard_targets(f, tr["Y_dB"])
with open(data.RELEASE / "data" / "design_metrics.csv", encoding="utf-8-sig") as fh:
    import csv
    rows = {r["design_id"]: r for r in csv.DictReader(fh) if r["split"] == "train"}
ds_ripple = np.array([float(rows[i]["ripple_fixed_pass_dB"]) for i in tr["design_id"]])
ds_rl = np.array([float(rows[i]["return_loss_fixed_pass_worst_dB"]) for i in tr["design_id"]])
check = {"tau_dB": targets.TAU_DB, "smooth_vs_hard": {}, "linear_R2_geometry": {}, "train_std": {}}
for k, h in hard.items():
    s = Ytr[:, targets.CURVE_TARGETS.index(k)]
    check["smooth_vs_hard"][k] = {"spearman": float(spearmanr(s, h)[0]), "pearson": float(np.corrcoef(s, h)[0, 1]),
                                  "mean_offset": float(np.mean(s - h)), "max_abs_offset": float(np.max(np.abs(s - h)))}
check["dataset_agreement"] = {
    "hard_ripple_vs_dataset_max_abs": float(np.max(np.abs(hard["T2_ripple_dB"] - ds_ripple))),
    "hard_worst_S11_vs_dataset_return_loss_max_abs": float(np.max(np.abs(np.abs(hard["T2_worst_S11_dB"]) - np.abs(ds_rl)))),
    "hard_mask_pass_matches_dataset": bool(np.array_equal(hard["T2_mask_violation_dB"] <= 1e-9, tr["mask_pass"]))}
G = tr["X"][:, :6]
for j, k in enumerate(targets.CURVE_TARGETS):
    check["linear_R2_geometry"][k] = float(r2_ols(G, Ytr[:, j])[0])
    check["train_std"][k] = float(Ytr[:, j].std())
r2, coef = r2_ols(tr["X"], Ytr[:, 0])  # 9 normalized inputs -> standardized coefficients
check["T0_OLS_9_inputs"] = {"R2": float(r2), "coef_per_train_std": dict(zip(names, map(float, coef))),
                            "rank_by_abs": [names[i] for i in np.argsort(-np.abs(coef))]}

# 2. explained designs
rng = np.random.default_rng(SEED)
sb = va["single_band"]
pick = list(np.flatnonzero(sb & va["mask_pass"]))
fails = {s: np.flatnonzero(sb & ~va["mask_pass"] & (va["source"] == s)) for s in sorted(set(va["source"]))}
slots = N_EXPLAIN - len(pick)
tot = sum(len(v) for v in fails.values())
quota = {s: slots * len(v) / tot for s, v in fails.items()}
alloc = {s: int(np.floor(q)) for s, q in quota.items()}
for s in sorted(quota, key=lambda s: quota[s] - alloc[s], reverse=True)[:slots - sum(alloc.values())]:
    alloc[s] += 1
for s, idx in fails.items():
    pick += list(rng.choice(idx, alloc[s], replace=False))
pick = np.array(sorted(pick))
strata = {f"{s}|{'pass' if p else 'fail'}": int(np.sum((va["source"][pick] == s) & (va["mask_pass"][pick] == p)))
          for s in fails for p in (True, False)}

# 3. backgrounds
new6 = np.flatnonzero(tr["source"] == "new_6d_actual")
bg = {"lhs32": np.sort(np.random.default_rng(2026).choice(new6, 32, replace=False)),
      "rand32": np.sort(np.random.default_rng(2027).choice(len(tr["X"]), 32, replace=False)),
      "paper100": np.sort(np.random.default_rng(2026).choice(len(tr["X"]), 100, replace=False))}

# 4. reference point (normalized exactly as the release loader does)
sc = data.scalers()["candidates"]
order = data.scalers()["candidate_order"]
idx = [order.index(n) for n in names[:6]]
mean, sd = np.array(sc["mean"])[idx], np.array(sc["effective_std"])[idx]
norm_geom = lambda g: (np.asarray(g) - mean) / sd  # noqa: E731
x_ref = np.r_[norm_geom(data.REFERENCE_GEOMETRY_MIL), np.zeros(3)].astype(np.float32)
assert np.allclose(norm_geom(tr["X_raw"][:, :6]), tr["X"][:, :6], atol=1e-4), "normalization mismatch"
lo = np.r_[norm_geom([v[0] for v in json.loads((data.RELEASE / "plan.json").read_text(encoding="utf-8"))["bounds_mil"].values()]),
           tr["X"][:, 6:].min(0)].astype(np.float32)
hi = np.r_[norm_geom([v[1] for v in json.loads((data.RELEASE / "plan.json").read_text(encoding="utf-8"))["bounds_mil"].values()]),
           tr["X"][:, 6:].max(0)].astype(np.float32)

# 5. diagnostics: baseline + 12 one-factor scenarios
dz = np.load(data.RELEASE / "data" / "diagnostic_scenarios.npz")
dg = dz["physical_parameters"][:13, :6]
assert np.allclose(dz["frequency_Hz"][:601] / 1e9, f) and np.allclose(dg[0], data.REFERENCE_GEOMETRY_MIL)
x_diag = np.c_[norm_geom(dg), np.zeros((13, 3))].astype(np.float32)
y_diag = curve_targets(dz["Y_dB"][:13, :601])
S_diag = np.stack([dz["S_complex"][:13, :601, 1, 0], dz["S_complex"][:13, :601, 0, 0]], -1)
conc = []
for i in range(13):
    p, *_ = vectfit.vector_fit(f, S_diag[i], 10)
    fn, q = vectfit.natural_frequency_q(p[p.imag > 0])
    o = np.argsort(fn)
    conc.append(np.r_[fn[o][1:4], q[o][1:4]])
conc = np.array(conc)

np.savez_compressed(OUT / "setup.npz", explain_idx=pick, explain_id=va["design_id"][pick], x_explain=va["X"][pick],
                    x_ref=x_ref, lo=lo, hi=hi, bg_lhs32=tr["X"][bg["lhs32"]], bg_rand32=tr["X"][bg["rand32"]],
                    bg_paper100=tr["X"][bg["paper100"]], x_train=tr["X"], y_train=Ytr,
                    y_explain_true=curve_targets(va["Y_dB"][pick]), x_diag=x_diag, y_diag=y_diag, concepts_diag=conc,
                    diag_id=dz["design_id"][:13])
setup = {"target_check_train": check, "explained": {"n": int(len(pick)), "strata": strata,
         "design_id": va["design_id"][pick].tolist(), "rule": "all single-band mask passes + failing designs by source (largest remainder), seed 2026"},
         "backgrounds": {k: tr["design_id"][v].tolist() for k, v in bg.items()},
         "reference_normalized": dict(zip(names, map(float, x_ref))),
         "lime_bounds_normalized": {"lo": lo.tolist(), "hi": hi.tolist()},
         "diagnostics": {"ids": dz["design_id"][:13].tolist(), "targets": targets.CURVE_TARGETS,
                         "ADS_targets": y_diag.tolist(), "vf_concepts": {"names": targets.CONCEPT_TARGETS, "values": conc.tolist()}}}
(OUT / "setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
print(json.dumps({k: setup[k] for k in ("target_check_train", "explained")}, indent=1)[:6000])
