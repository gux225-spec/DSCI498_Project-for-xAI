"""Step 2: explanations on the fixed validation designs (plan §4). Test is never loaded.

For each frozen surrogate of one kind (3 seeds, results/step1/final_models.json):
  exact SHAP (512 coalitions) with background ref / lhs32 / rand32; IG (64 steps, raised to 128/256 per
  design and target while the completeness residual exceeds 1%); own LIME (scale 0.25 x 5 seeds, scales
  0.1 and 0.5 x 1 seed); local gradients; per-feature intervention sensitivity (rand32); predictions on the
  13 baseline diagnostic scenarios.
Reference seed only: exact SHAP with the paper100 background, shap.GradientExplainer (paper100,
  nsamples 200), lime package defaults x 20 runs (paper protocol), and for PR-Net the concept-group Shapley
  of the curve targets (8 groups: f2, f3, f4, Q2, Q3, Q4, residues, background poles + constants; the
  reference design's concepts fill the absent groups).
Designs: the 40 explained validation designs, then the baseline (= reference point) as row 40.
Writes results/step2/attr/<tag>.npz (+ <tag>_paper.npz for the reference seed).
Usage: python scripts/step2_xai.py {PRNet,M1,M0} [test]
  With "test" (step 3, after the freeze) the same procedure reads results/step3/setup_test.{npz,json} and
  writes results/step3/attr/; nothing else changes.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, xai, zoo  # noqa: E402
from filterxai.train import DEV  # noqa: E402

kind = sys.argv[1]
TEST = len(sys.argv) > 2 and sys.argv[2] == "test"
SDIR, SNAME = (data.ROOT / "results" / "step3", "setup_test") if TEST else (data.ROOT / "results" / "step2", "setup")
OUT = SDIR / "attr"
OUT.mkdir(parents=True, exist_ok=True)
z = np.load(SDIR / f"{SNAME}.npz")
setup = json.loads((SDIR / f"{SNAME}.json").read_text(encoding="utf-8"))
REF_SEED = json.loads(zoo.REGISTRY.read_text(encoding="utf-8"))["reference_seed"]
x_ref, lo, hi = z["x_ref"], z["lo"], z["hi"]
X = np.vstack([z["x_explain"], x_ref[None]]).astype(np.float32)
STD = np.array(list(setup["target_check_train"]["train_std"].values()))
LIME_RUNS = [(0.25, s) for s in range(5)] + [(0.1, 0), (0.5, 0)]


def tic(msg, t0=[time.time()]):  # noqa: B006
    print(f"  [{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


def adaptive_ig(m):
    """IG at 64 steps; entries with relative completeness residual > 1% recomputed at 128 then 256."""
    phi, delta = xai.integrated_gradients(m, X, x_ref, 64, DEV)
    steps = np.full(phi.shape[:2], 64)
    std = np.r_[STD, np.ones(phi.shape[1] - len(STD))]  # T3 scale: 1 (GHz or Q units)
    scale = np.maximum(np.abs(delta), 0.01 * std)
    for s in (128, 256):
        bad = np.abs(phi.sum(-1) - delta) / scale > 0.01
        rows = np.flatnonzero(bad.any(1))
        if not len(rows):
            break
        p2, _ = xai.integrated_gradients(m, X[rows], x_ref, s, DEV)
        sub = bad[rows]
        phi[rows] = np.where(sub[..., None], p2, phi[rows])
        steps[rows] = np.where(sub, s, steps[rows])
    resid = np.abs(phi.sum(-1) - delta) / scale
    return phi, delta, steps, resid


def concept_group_shapley(m):
    """PR-Net only: Shapley over 8 concept groups for the curve targets, reference-design concepts as baseline."""
    net = m.net
    with torch.no_grad():
        c_x = net.concepts(torch.tensor(X, device=DEV))
        c_r = net.concepts(torch.tensor(x_ref[None], device=DEV))
    groups = [("fn", 1), ("fn", 2), ("fn", 3), ("q", 1), ("q", 2), ("q", 3), ("rn", None), ("bg", None)]
    n = len(groups)
    masks = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(bool)
    vals = []
    for d in range(len(X)):
        fn = c_r[0].repeat(2 ** n, 1).clone()
        q = c_r[1].repeat(2 ** n, 1).clone()
        rn = c_r[2].repeat(2 ** n, 1, 1).clone()
        dd = c_r[3].repeat(2 ** n, 1).clone()
        for gi, (name, k) in enumerate(groups):
            on = torch.tensor(masks[:, gi], device=DEV)
            if name in ("fn", "q"):
                tgt, src = (fn, c_x[0]) if name == "fn" else (q, c_x[1])
                tgt[on, k] = src[d, k]
            elif name == "rn":
                rn[on] = c_x[2][d]
            else:
                for j in (0, 4):
                    fn[on, j], q[on, j] = c_x[0][d, j], c_x[1][d, j]
                dd[on] = c_x[3][d]
        with torch.no_grad():
            S = net.response(fn, q, rn, dd)
            db = 10 * torch.log10(torch.clamp(torch.abs(S) ** 2, min=1e-30))
            vals.append(m.T(db).cpu().numpy())
    v = np.stack(vals)  # (D, 256, T_curve)
    W = xai.shapley_matrix(n)
    return np.einsum("ic,dct->dti", W, v), v[:, 0], v[:, -1], [f"{a}{'' if k is None else k + 1}" for a, k in groups]


def paper_protocol(m):
    import shap
    from lime import lime_tabular
    e = shap.GradientExplainer(m, torch.tensor(z["bg_paper100"], device=DEV))
    gx = np.asarray(e.shap_values(torch.tensor(X, device=DEV), nsamples=200, rseed=REF_SEED)).transpose(0, 2, 1)
    tic("GradientExplainer")
    T = len(m.names)
    lw = np.zeros((len(X), 20, T, 9))
    for d in range(len(X)):
        cache = {}

        def pred(t):
            def f(Zs):
                k = Zs.tobytes()
                if k not in cache:
                    cache.clear()
                    cache[k] = m.targets(Zs.astype(np.float32))
                return cache[k][:, t]
            return f
        for r in range(20):
            for t in range(T):
                ex = lime_tabular.LimeTabularExplainer(z["x_train"], mode="regression", random_state=r)
                le = ex.explain_instance(X[d], pred(t), num_features=9)
                for j, w in le.local_exp[next(iter(le.local_exp))]:
                    lw[d, r, t, j] = w
    tic("LIME paper x20")
    return gx, lw


for m in zoo.load(kind):
    print(m.tag, flush=True)
    res = {"names": np.array(m.names), "x": X, "F_x": m.targets(X), "F_ref": m.targets(x_ref[None])[0]}
    v_ref = xai.coalition_values(m.targets, X, x_ref[None])
    res["v_ref"], res["shap_ref"] = v_ref, xai.shap_from_values(v_ref)
    for b in ("lhs32", "rand32"):
        v = xai.coalition_values(m.targets, X, z[f"bg_{b}"])
        res[f"shap_{b}"], res[f"base_{b}"] = xai.shap_from_values(v), v[:, 0]
    tic("exact SHAP x3 backgrounds")
    res["ig"], res["ig_delta"], res["ig_steps"], res["ig_resid"] = adaptive_ig(m)
    tic(f"IG (steps used: {dict(zip(*np.unique(res['ig_steps'], return_counts=True)))})")
    res["lime_runs"] = np.array(LIME_RUNS)
    lc, lr = zip(*[xai.lime(m.targets, X, lo, hi, scale=s, seed=sd) for s, sd in LIME_RUNS])
    res["lime_coef"], res["lime_r2"] = np.stack(lc), np.stack(lr)  # (R, D, T, 9), (R, D, T)
    tic("own LIME")
    res["grad"] = xai.local_gradient(m, X, DEV)
    res["interv_rand32"] = xai.intervention_sensitivity(m.targets, X, z["bg_rand32"])
    res["F_diag"] = m.targets(z["x_diag"])
    res["grad_diag0"] = xai.local_gradient(m, z["x_diag"][:1], DEV)[0]
    tic("gradients, interventions, diagnostics")
    np.savez_compressed(OUT / f"{m.tag}.npz", **res)
    if f"_s{REF_SEED}" in m.tag:
        extra = {}
        v100 = xai.coalition_values(m.targets, X, z["bg_paper100"])
        extra["shap_paper100"], extra["base_paper100"] = xai.shap_from_values(v100), v100[:, 0]
        tic("exact SHAP paper100")
        extra["gradexp"], extra["lime_paper"] = paper_protocol(m)
        if kind == "PRNet":
            extra["cg_shap"], extra["cg_v0"], extra["cg_vN"], names = concept_group_shapley(m)
            extra["cg_groups"] = np.array(names)
            tic("concept-group Shapley")
        np.savez_compressed(OUT / f"{m.tag}_paper.npz", **extra)
print("done", kind)
