"""Step 3: ADS job list for the true-Shapley game and local finite differences (freeze protocol).

Designs: the 40 validation + 40 test explained designs. Per design:
  coalitions  all 64 subsets of the 6 geometry features; present features from the design, absent ones
              from the ADS baseline geometry (empty = baseline, full = the design itself, both re-simulated
              as a reproduction check against the dataset)
  finite diff h = 2% of the declared span per parameter; x + h and x - h when both are in bounds,
              otherwise only the in-bound side (one-sided with the design itself)
Jobs are keyed by geometry (12 significant digits), so duplicates are simulated once.
Writes results/step3/ads/{jobs.json, map.json}. Usage: python scripts/step3_ads_jobs.py
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data  # noqa: E402

OUT = data.ROOT / "results" / "step3" / "ads"
OUT.mkdir(parents=True, exist_ok=True)
G = data.GEOMETRY
REF = data.REFERENCE_GEOMETRY_MIL
bounds = json.loads((data.RELEASE / "plan.json").read_text(encoding="utf-8"))["bounds_mil"]
LO, HI = np.array([bounds[k][0] for k in G]), np.array([bounds[k][1] for k in G])
H = 0.02 * (HI - LO)
MASK6 = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(bool)

va = data.load("validation")
z2 = np.load(data.ROOT / "results" / "step2" / "setup.npz")
z3 = np.load(data.ROOT / "results" / "step3" / "setup_test.npz")
designs = [("validation", str(i), va["X_raw"][j, :6]) for i, j in zip(z2["explain_id"], z2["explain_idx"])]
designs += [("test", str(i), g[:6]) for i, g in zip(z3["explain_id"], z3["x_raw_explain"])]

jobs, mp = {}, []


def job(g):
    key = " ".join(f"{v:.12g}" for v in g)
    jid = hashlib.sha1(key.encode()).hexdigest()[:16]
    jobs.setdefault(jid, {"id": jid, "geometry": dict(zip(G, map(float, g)))})
    return jid


for split, did, x in designs:
    coal = [job(np.where(m, x, REF)) for m in MASK6]
    fd = []
    for k in range(6):
        up, dn = x.copy(), x.copy()
        up[k] += H[k]
        dn[k] -= H[k]
        fd.append({"plus": job(up) if up[k] <= HI[k] + 1e-9 else None, "minus": job(dn) if dn[k] >= LO[k] - 1e-9 else None,
                   "h": float(H[k])})
    mp.append({"split": split, "design_id": did, "geometry": dict(zip(G, map(float, x))), "coalitions": coal, "fd": fd})
(OUT / "jobs.json").write_text(json.dumps(list(jobs.values())), encoding="utf-8")
(OUT / "map.json").write_text(json.dumps({"mask_bits": "coalition c contains geometry feature i iff bit i of c",
                                          "features": G, "designs": mp}, indent=1), encoding="utf-8")
n_req = sum(64 + sum((d["plus"] is not None) + (d["minus"] is not None) for d in m["fd"]) for m in mp)
print(f"{len(mp)} designs, {n_req} requested simulations, {len(jobs)} unique jobs")
