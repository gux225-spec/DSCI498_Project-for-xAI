"""Step 4 (RQ3): explanation-guided tuning with ADS confirmation, exactly as pre-registered in
results/step4/prereg.json (plan §5 C).

Per case: choose k = 2 geometry parameters by each selector, search the PR-Net surrogate in that subspace
with margin escalation (0.25 dB per ADS call, at most 10 calls), confirm every proposal in ADS; plus the
all-6 surrogate search, ADS coordinate search (60 calls), ADS true-Shapley top-2 (64 extra calls) and the
SHAP revert-to-baseline (1 call). ADS results are cached by geometry (shared with step 3).
Writes results/step4/<split>/{results.json, selections.json}.
Usage: python scripts/step4_tune.py {validation,test}
"""
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

split = sys.argv[1]
ROOT = data.ROOT
PRE = json.loads((ROOT / "results" / "step4" / "prereg.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "step4" / split
OUT.mkdir(parents=True, exist_ok=True)
ADS_DIR = ROOT / "results" / "step4" / "ads"
CACHES = [ADS_DIR / "cases", ROOT / "results" / "step3" / "ads" / "cases"]
ADS_PY = r"E:\ADS2026\tools\python\python.exe"
G = data.GEOMETRY
T_IDX = targets.CURVE_TARGETS.index("T2_mask_violation_dB")
F = np.linspace(9.0, 15.0, 601)
TC = targets.CurveTargets(F, DEV)

# cases and coordinates
d = data.load(split, allow_test_labels=split == "test")
idx = np.array(PRE["cases"][split]["index"])
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
model = zoo.load("PRNet", 2026)[0]
assert model.tag == PRE["model"]
for p in model.parameters():  # only input gradients are needed
    p.requires_grad_(False)
STD = json.loads((ROOT / "results" / "step2" / "setup.json").read_text(encoding="utf-8"))["target_check_train"]["train_std"]


# ---------- ADS ----------
def jid(g):
    return hashlib.sha1(" ".join(f"{v:.12g}" for v in g).encode()).hexdigest()[:16]


def cached(j):
    for c in CACHES:
        p = c / f"{j}.npz"
        if p.exists():
            return p
    return None


def ads(geoms, tag):
    """geoms: list of (6,) mil arrays -> list of dicts {hard, smooth}; simulates missing ones in a batch."""
    geoms = [np.round(np.clip(g, LO_MIL, HI_MIL), 9) for g in geoms]
    ids = [jid(g) for g in geoms]
    todo = {j: g for j, g in zip(ids, geoms) if cached(j) is None}
    if todo:
        ADS_DIR.mkdir(parents=True, exist_ok=True)
        jf = ADS_DIR / f"jobs_{split}_{tag}.json"
        jf.write_text(json.dumps([{"id": j, "geometry": dict(zip(G, map(float, g)))} for j, g in todo.items()]))
        for workers in (4, 1, 1):
            subprocess.run([ADS_PY, str(ROOT / "scripts" / "ads_sim.py"), str(jf), str(ADS_DIR), "--workers", str(workers)],
                           check=True, capture_output=True)
            if all(cached(j) for j in todo):
                break
        missing = [j for j in todo if cached(j) is None]
        if missing:
            raise RuntimeError(f"ADS failed for {len(missing)} jobs")
    out = []
    for j in ids:
        s = np.load(cached(j))["S_complex"]
        db = 10 * np.log10(np.maximum(np.abs(np.stack([s[:, 1, 0], s[:, 0, 0]], -1)) ** 2, 1e-30))[None]
        hard = float(targets.hard_targets(F, db)["T2_mask_violation_dB"][0])
        with torch.no_grad():
            smooth = float(TC(torch.tensor(db, dtype=torch.float32, device=DEV))[0, T_IDX])
        out.append({"hard": hard, "smooth": smooth})
    return out


# ---------- surrogate ----------
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
    """Pre-registered surrogate search in subspace S (list of geometry indices); returns normalized x (9,)."""
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


# ---------- selectors ----------
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


sel, ads_true = {}, {}
MASK6 = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(bool)
coal_geoms = [np.where(m, G0[i], data.REFERENCE_GEOMETRY_MIL) for i in range(len(idx)) for m in MASK6]
coal = ads(coal_geoms, "coalitions")
W6 = xai.shapley_matrix(6)
for i in range(len(idx)):
    x = X0[i:i + 1]
    v = xai.coalition_values(model.targets, x, x_ref[None])
    shap = xai.shap_from_values(v)[0, T_IDX]
    lc, _ = xai.lime(model.targets, x, lo, hi, scale=0.25, seed=0)
    lc = lc[0, T_IDX]
    grad = xai.local_gradient(model, x, DEV)[0, T_IDX]
    phi_ads = W6 @ np.array([c["smooth"] for c in coal[64 * i:64 * (i + 1)]])
    sel[i] = {"shap_ref": top2(shap), "ig": top2(adaptive_ig(x)), "lime_delta": top2(lc * (x[0] - x_ref)),
              "random": sorted(np.random.default_rng(2026 + i).choice(6, 2, replace=False).tolist()),
              "grad_x_span": top2(grad[:6] * SPAN_N), "lime_slope_x_span": top2(lc[:6] * SPAN_N),
              "ads_true_shap": top2(phi_ads), "all6_surrogate": list(range(6))}
    ads_true[i] = phi_ads.tolist()
(OUT / "selections.json").write_text(json.dumps({"design_id": PRE["cases"][split]["design_id"],
                                                 "selections": sel, "ads_true_shap_phi": ads_true}, indent=1), encoding="utf-8")
print("selections done", flush=True)

# ---------- surrogate-guided trajectories (shared per unique subspace) ----------
x0_ads = ads([G0[i] for i in range(len(idx))], "x0")
traj = {(i, tuple(S)): {"calls": [], "prev": None, "done": False} for i in sel for S in map(tuple, sel[i].values())}
for k in range(10):
    m = 0.25 * k
    props = {}
    for (i, S), t in traj.items():
        if t["done"]:
            continue
        x = search(X0[i], list(S), m)
        if t["prev"] is not None and np.max(np.abs(x - t["prev"])) < 1e-6:
            t["calls"].append({"margin": m, "repeat": True})
            continue
        t["prev"] = x
        props[(i, S)] = x
    res = ads([to_mil(x) for x in props.values()], f"round{k}")
    for (key, x), r in zip(props.items(), res):
        g = to_mil(x)
        t = traj[key]
        t["calls"].append({"margin": m, "repeat": False, "geometry": g.tolist(), "hard": r["hard"], "smooth": r["smooth"],
                           "change": float(np.linalg.norm((g - G0[key[0]]) / SPAN_MIL))})
        if r["hard"] <= 1e-9:
            t["done"] = True
    print(f"round {k}: {len(props)} proposals, solved {sum(t['done'] for t in traj.values())}/{len(traj)} subspaces", flush=True)

# ---------- revert top-2 SHAP features to baseline ----------
rev_g = [np.where(np.isin(np.arange(6), sel[i]["shap_ref"]), data.REFERENCE_GEOMETRY_MIL, G0[i]) for i in range(len(idx))]
rev = ads(rev_g, "revert")

# ---------- ADS coordinate search (one evaluation per case per round) ----------
cs = [{"x": G0[i].copy(), "v": x0_ads[i]["hard"], "step": 0.25 * SPAN_MIL.copy(), "j": 0, "sign": 1, "improved": False,
       "evals": 0, "done": x0_ads[i]["hard"] <= 1e-9, "trace": []} for i in range(len(idx))]


def next_candidate(c):
    """Advance (j, sign) until a candidate inside the bounds that differs from x; handles step halving."""
    for _ in range(64):
        cand = c["x"].copy()
        cand[c["j"]] = np.clip(cand[c["j"]] + c["sign"] * c["step"][c["j"]], LO_MIL[c["j"]], HI_MIL[c["j"]])
        if abs(cand[c["j"]] - c["x"][c["j"]]) > 1e-9:
            return cand
        advance(c, accepted=False)
    return None


def advance(c, accepted):
    if accepted or c["sign"] == -1:
        c["sign"] = 1
        c["j"] += 1
        if c["j"] == 6:
            c["j"] = 0
            if not c["improved"]:
                c["step"] /= 2
            c["improved"] = False
    else:
        c["sign"] = -1


while any(not c["done"] for c in cs):
    act = [(i, next_candidate(c)) for i, c in enumerate(cs) if not c["done"]]
    for i, g in act:
        if g is None:  # no move possible inside the bounds
            cs[i]["done"] = True
    act = [(i, g) for i, g in act if g is not None]
    if not act:
        break
    res = ads([g for _, g in act], f"coord{cs[act[0][0]]['evals']}")
    for (i, g), r in zip(act, res):
        c = cs[i]
        c["evals"] += 1
        acc = r["hard"] < c["v"] - 1e-12
        if acc:
            c["x"], c["v"], c["improved"] = g, r["hard"], True
        c["trace"].append({"geometry": g.tolist(), "hard": r["hard"], "accepted": bool(acc)})
        advance(c, acc)
        if c["v"] <= 1e-9 or c["evals"] >= 60:
            c["done"] = True
    print(f"coordinate: active {sum(not c['done'] for c in cs)}", flush=True)

# ---------- assemble per case and method ----------
results = []
for i in range(len(idx)):
    row = {"design_id": PRE["cases"][split]["design_id"][i], "x0_hard": x0_ads[i]["hard"],
           "x0_surrogate": float(model.targets(X0[i:i + 1])[0, T_IDX]), "methods": {}}
    for name, S in sel[i].items():
        calls = [c for c in traj[(i, tuple(S))]["calls"] if not c["repeat"]]
        ok = [n for n, c in enumerate(calls) if c["hard"] <= 1e-9]
        best = min(calls, key=lambda c: c["hard"]) if calls else None
        final = calls[ok[0]] if ok else best
        row["methods"][name] = {"subspace": [G[j] for j in S], "success": bool(ok), "calls_to_success": ok[0] + 1 if ok else None,
                                "ads_calls": len(calls), "best_hard": best["hard"] if best else x0_ads[i]["hard"],
                                "final_change": final["change"] if final else 0.0,
                                "extra_calls": 64 if name == "ads_true_shap" else 0}
    rv = rev[i]
    row["methods"]["shap_revert_no_search"] = {"subspace": [G[j] for j in sel[i]["shap_ref"]], "success": rv["hard"] <= 1e-9,
                                               "calls_to_success": 1 if rv["hard"] <= 1e-9 else None, "ads_calls": 1, "best_hard": min(rv["hard"], x0_ads[i]["hard"]),
                                               "final_change": float(np.linalg.norm((rev_g[i] - G0[i]) / SPAN_MIL)), "extra_calls": 0}
    c = cs[i]
    okc = [n for n, t in enumerate(c["trace"]) if t["hard"] <= 1e-9]
    row["methods"]["ads_coordinate"] = {"subspace": G, "success": c["v"] <= 1e-9, "calls_to_success": okc[0] + 1 if okc else None,
                                        "ads_calls": c["evals"], "best_hard": c["v"],
                                        "final_change": float(np.linalg.norm((c["x"] - G0[i]) / SPAN_MIL)), "extra_calls": 0}
    row["trajectories"] = {",".join(G[j] for j in S): traj[(i, tuple(S))]["calls"] for S in {tuple(s) for s in sel[i].values()}}
    row["coordinate_trace"] = c["trace"]
    results.append(row)
(OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
summary = {name: {"success": int(sum(r["methods"][name]["success"] for r in results)), "n": len(results)} for name in results[0]["methods"]}
print(json.dumps(summary, indent=1))
