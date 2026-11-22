"""Step 4b (exploratory, plan §13.1): all 15 two-parameter subspaces + all-6 for every test case, with
PR-Net or M0 as the search surrogate. Search and ADS confirmation are the pre-registered ones from
scripts/step4_tune.py (copied here unchanged so that the pre-registered script stays untouched).
For M0 the explanation-based selections are recomputed with M0 (same explainer settings).
The PR-Net run used the per-subspace search(); the rounds now call search_batch(), the same procedure
batched (checked against PR-Net round-0 proposals with --check-batch before the M0 run).
Writes results/step4b/<kind>/{pairs.json, selections.json}.
Usage: python scripts/step4b_explore.py {PRNet,M0} [--check-batch]
"""
from itertools import combinations
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, targets, xai, zoo  # noqa: E402
from filterxai.train import DEV  # noqa: E402

kind = sys.argv[1]
ROOT = data.ROOT
PRE = json.loads((ROOT / "results" / "step4" / "prereg.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "step4b" / kind
OUT.mkdir(parents=True, exist_ok=True)
ADS_DIR = ROOT / "results" / "step4" / "ads"
CACHES = [ADS_DIR / "cases", ROOT / "results" / "step3" / "ads" / "cases"]
ADS_PY = r"E:\ADS2026\tools\python\python.exe"
G = data.GEOMETRY
T_IDX = targets.CURVE_TARGETS.index("T2_mask_violation_dB")
F = np.linspace(9.0, 15.0, 601)
TC = targets.CurveTargets(F, DEV)

d = data.load("test", allow_test_labels=True)
idx = np.array(PRE["cases"]["test"]["index"])
X0, G0 = d["X"][idx].astype(np.float32), d["X_raw"][idx, :6]
z2 = np.load(ROOT / "results" / "step2" / "setup.npz")
x_ref, lo, hi = z2["x_ref"], z2["lo"], z2["hi"]
sc = data.scalers()
ci = [sc["candidate_order"].index(n) for n in G]
MEAN, SD = np.array(sc["candidates"]["mean"])[ci], np.array(sc["candidates"]["effective_std"])[ci]
bounds = json.loads((data.RELEASE / "plan.json").read_text(encoding="utf-8"))["bounds_mil"]
LO_MIL, HI_MIL = np.array([bounds[k][0] for k in G]), np.array([bounds[k][1] for k in G])
SPAN_MIL = HI_MIL - LO_MIL
SPAN_N = (hi - lo)[:6]
to_mil = lambda x: x[:6] * SD + MEAN  # noqa: E731
model = zoo.load(kind, 2026)[0]
for p in model.parameters():
    p.requires_grad_(False)
STD = json.loads((ROOT / "results" / "step2" / "setup.json").read_text(encoding="utf-8"))["target_check_train"]["train_std"]


def jid(g):
    return hashlib.sha1(" ".join(f"{v:.12g}" for v in g).encode()).hexdigest()[:16]


def cached(j):
    for c in CACHES:
        p = c / f"{j}.npz"
        if p.exists():
            return p
    return None


def ads(geoms, tag):
    geoms = [np.round(np.clip(g, LO_MIL, HI_MIL), 9) for g in geoms]
    ids = [jid(g) for g in geoms]
    todo = {j: g for j, g in zip(ids, geoms) if cached(j) is None}
    if todo:
        jf = ADS_DIR / f"jobs_4b_{kind}_{tag}.json"
        jf.write_text(json.dumps([{"id": j, "geometry": dict(zip(G, map(float, g)))} for j, g in todo.items()]))
        for workers in (4, 1, 1):
            subprocess.run([ADS_PY, str(ROOT / "scripts" / "ads_sim.py"), str(jf), str(ADS_DIR), "--workers", str(workers)],
                           check=True, capture_output=True)
            if all(cached(j) for j in todo):
                break
        if any(cached(j) is None for j in todo):
            raise RuntimeError("ADS failed")
    out = []
    for j in ids:
        s = np.load(cached(j))["S_complex"]
        db = 10 * np.log10(np.maximum(np.abs(np.stack([s[:, 1, 0], s[:, 0, 0]], -1)) ** 2, 1e-30))[None]
        out.append({"hard": float(targets.hard_targets(F, db)["T2_mask_violation_dB"][0])})
    return out


def viol_margin(db, m):
    s21, s11 = db[..., 0], db[..., 1]
    p21, p11 = s21[:, TC.pb], s11[:, TC.pb]
    M = targets.MASK
    return (targets.srelu(M["passband_S21_min"] + m + targets.smax(-p21)) + targets.srelu(targets.smax(p11) - (M["passband_S11_max"] - m))
            + targets.srelu(targets.smax(s21[:, TC.low]) - (M["low_stopband_S21_max"] - m))
            + targets.srelu(targets.smax(s21[:, TC.high]) - (M["high_stopband_S21_max"] - m)))


def J(x, x0, m):
    dn = (x[:, :6] - x0[:6]) / torch.tensor(SPAN_N, device=DEV)
    return viol_margin(model.db(x), m) + 0.01 * (dn ** 2).sum(-1)


def search(x0, S, m):
    x0t = torch.tensor(x0, device=DEV)
    lo_t, hi_t = torch.tensor(lo, device=DEV), torch.tensor(hi, device=DEV)
    S = list(S)
    if len(S) == 2:
        a = torch.linspace(0, 1, 41, device=DEV)
        g1, g2 = torch.meshgrid(lo_t[S[0]] + a * (hi_t[S[0]] - lo_t[S[0]]), lo_t[S[1]] + a * (hi_t[S[1]] - lo_t[S[1]]), indexing="ij")
        X = x0t.repeat(41 * 41, 1)
        X[:, S[0]], X[:, S[1]] = g1.ravel(), g2.ravel()
        with torch.no_grad():
            starts = X[torch.argsort(J(X, x0t, m))[:5]]
        steps = 200
    else:
        r = np.random.default_rng(2026)
        U = lo[None, :] + r.random((510, 9)) * (hi - lo)[None, :]
        starts = torch.tensor(np.vstack([x0[None], np.r_[x_ref[:6], x0[6:]][None], np.c_[U[:, :6], np.repeat(x0[None, 6:], 510, 0)]]),
                              dtype=torch.float32, device=DEV)
        steps = 300
    z = starts[:, S].clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.01)
    base = starts.detach().clone()
    for _ in range(steps):
        X = base.clone()
        X[:, S] = z
        loss = J(X, x0t, m).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.data = torch.max(torch.min(z.data, hi_t[S]), lo_t[S])
    with torch.no_grad():
        X = base.clone()
        X[:, S] = z
        best = X[torch.argmin(J(X, x0t, m))]
    return best.cpu().numpy()


def search_batch(jobs):
    """search() for many (x0, S, m) jobs at once. Adam is elementwise and rows are independent, so this is
    the same procedure as calling search() per job (up to float rounding), just one optimizer loop per k."""
    lo_t, hi_t = torch.tensor(lo, device=DEV), torch.tensor(hi, device=DEV)
    span = torch.tensor(SPAN_N, device=DEV)

    def Jrows(X, X0, Mg):
        return viol_margin(model.db(X), Mg) + 0.01 * (((X[:, :6] - X0[:, :6]) / span) ** 2).sum(-1)

    out = [None] * len(jobs)
    a = torch.linspace(0, 1, 41, device=DEV)
    r = np.random.default_rng(2026)
    U = lo[None, :] + r.random((510, 9)) * (hi - lo)[None, :]
    for size, steps in ((2, 200), (6, 300)):
        sel_jobs = [n for n, (_, S, _) in enumerate(jobs) if len(S) == size]
        if not sel_jobs:
            continue
        starts, x0s, ms, masks, owner = [], [], [], [], []
        for n in sel_jobs:
            x0, S, m = jobs[n]
            x0t = torch.tensor(x0, device=DEV)
            S = list(S)
            if size == 2:
                g1, g2 = torch.meshgrid(lo_t[S[0]] + a * (hi_t[S[0]] - lo_t[S[0]]), lo_t[S[1]] + a * (hi_t[S[1]] - lo_t[S[1]]), indexing="ij")
                X = x0t.repeat(41 * 41, 1)
                X[:, S[0]], X[:, S[1]] = g1.ravel(), g2.ravel()
                with torch.no_grad():
                    st = X[torch.argsort(J(X, x0t, m))[:5]]
            else:
                st = torch.tensor(np.vstack([x0[None], np.r_[x_ref[:6], x0[6:]][None], np.c_[U[:, :6], np.repeat(x0[None, 6:], 510, 0)]]),
                                  dtype=torch.float32, device=DEV)
            mk = torch.zeros(9, device=DEV)
            mk[S] = 1.0
            starts.append(st)
            x0s.append(x0t.repeat(len(st), 1))
            ms.append(torch.full((len(st),), m, device=DEV))
            masks.append(mk.repeat(len(st), 1))
            owner += [n] * len(st)
        base, X0b, Mg, Mk = torch.cat(starts), torch.cat(x0s), torch.cat(ms), torch.cat(masks)
        owner = np.array(owner)
        Z = base.clone().requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=0.01)
        for _ in range(steps):
            X = base + Mk * (Z - base)
            loss = Jrows(X, X0b, Mg).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                Z.data = torch.max(torch.min(Z.data, hi_t), lo_t)
        with torch.no_grad():
            X = base + Mk * (Z - base)
            Jv = Jrows(X, X0b, Mg).cpu().numpy()
        Xn = X.cpu().numpy()
        for n in sel_jobs:
            rows = np.flatnonzero(owner == n)
            out[n] = Xn[rows[np.argmin(Jv[rows])]]
    return out


def top2(phi):
    return sorted(np.argsort(-np.abs(phi[:6]))[:2].tolist())


def adaptive_ig(x):
    phi, delta = xai.integrated_gradients(model, x, x_ref, 64, DEV)
    scale = max(abs(delta[0, T_IDX]), 0.01 * STD["T2_mask_violation_dB"])
    for s in (128, 256):
        if abs(phi[0, T_IDX].sum() - delta[0, T_IDX]) / scale <= 0.01:
            break
        phi, delta = xai.integrated_gradients(model, x, x_ref, s, DEV)
    return phi[0, T_IDX]


if "--check-batch" in sys.argv:  # equivalence of search_batch() and search() on 3 cases x 16 subspaces, margin 0
    subs = [tuple(p) for p in combinations(range(6), 2)] + [tuple(range(6))]
    jobs = [(X0[i], S, 0.0) for i in range(3) for S in subs]
    xb = search_batch(jobs)
    diff = [float(np.max(np.abs(to_mil(b) - to_mil(search(x0, list(S), m))) / SPAN_MIL)) for b, (x0, S, m) in zip(xb, jobs)]
    print(f"max |batched - per-subspace| over {len(jobs)} searches: {max(diff):.2e} span units (median {np.median(diff):.1e})")
    sys.exit(0)

# selections with this model (model-independent ones copied from step 4)
s4 = json.loads((ROOT / "results" / "step4" / "test" / "selections.json").read_text(encoding="utf-8"))["selections"]
sel = {}
for i in range(len(idx)):
    x = X0[i:i + 1]
    shap = xai.shap_from_values(xai.coalition_values(model.targets, x, x_ref[None]))[0, T_IDX]
    lc, _ = xai.lime(model.targets, x, lo, hi, scale=0.25, seed=0)
    lc = lc[0, T_IDX]
    grad = xai.local_gradient(model, x, DEV)[0, T_IDX]
    sel[i] = {"shap_ref": top2(shap), "ig": top2(adaptive_ig(x)), "lime_delta": top2(lc * (x[0] - x_ref)),
              "grad_x_span": top2(grad[:6] * SPAN_N), "lime_slope_x_span": top2(lc[:6] * SPAN_N),
              "random": s4[str(i)]["random"], "ads_true_shap": s4[str(i)]["ads_true_shap"]}
(OUT / "selections.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
print("selections done", flush=True)

SUBS = [tuple(p) for p in combinations(range(6), 2)] + [tuple(range(6))]
x0_ads = ads([G0[i] for i in range(len(idx))], "x0")
traj = {(i, S): {"calls": [], "prev": None, "done": False} for i in range(len(idx)) for S in SUBS}
for k in range(10):
    m = 0.25 * k
    props = {}
    active = [key for key, t in traj.items() if not t["done"]]
    found = search_batch([(X0[i], S, m) for i, S in active])
    for (i, S), x in zip(active, found):
        t = traj[(i, S)]
        if t["prev"] is not None and np.max(np.abs(x - t["prev"])) < 1e-6:
            continue
        t["prev"] = x
        props[(i, S)] = x
    res = ads([to_mil(x) for x in props.values()], f"r{k}")
    for (key, x), r in zip(props.items(), res):
        g = to_mil(x)
        t = traj[key]
        t["calls"].append({"margin": m, "hard": r["hard"], "change": float(np.linalg.norm((g - G0[key[0]]) / SPAN_MIL))})
        if r["hard"] <= 1e-9:
            t["done"] = True
    print(f"round {k}: {len(props)} proposals, solved {sum(t['done'] for t in traj.values())}/{len(traj)}", flush=True)

rev_g = [np.where(np.isin(np.arange(6), sel[i]["shap_ref"]), data.REFERENCE_GEOMETRY_MIL, G0[i]) for i in range(len(idx))]
rev = ads(rev_g, "revert")
out = []
for i in range(len(idx)):
    row = {"design_id": PRE["cases"]["test"]["design_id"][i], "x0_hard": x0_ads[i]["hard"],
           "x0_surrogate": float(model.targets(X0[i:i + 1])[0, T_IDX]), "subspaces": {},
           "revert": {"hard": rev[i]["hard"], "change": float(np.linalg.norm((rev_g[i] - G0[i]) / SPAN_MIL))}}
    for S in SUBS:
        calls = traj[(i, S)]["calls"]
        ok = [n for n, c in enumerate(calls) if c["hard"] <= 1e-9]
        best = min(calls, key=lambda c: c["hard"])
        row["subspaces"][",".join(G[j] for j in S)] = {
            "success": bool(ok), "calls_to_success": ok[0] + 1 if ok else None, "ads_calls": len(calls),
            "best_hard": best["hard"], "final_change": calls[ok[0]]["change"] if ok else best["change"]}
    out.append(row)
(OUT / "pairs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("done", kind, flush=True)
