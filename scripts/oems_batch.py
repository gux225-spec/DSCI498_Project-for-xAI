"""Batch openEMS full-wave checks of step-4 tuning decisions (stand-in for the planned HFSS runs).

Question: when SHAP-guided tuning fixed a design in ADS (circuit), does the fix also hold in a
full-wave model? For each selected case we simulate the original test design (x0) and the
SHAP-tuned design (first ADS-passing geometry on the shap_ref trajectory).

Case rule (fixed before any openEMS case result was seen): walk the 30 step-4 test cases in
prereg order; keep a case if shap_ref succeeded in ADS and its SHAP parameter pair has not been
used yet; stop at N_CASES. Baseline is run separately first as calibration vs Momentum.

Usage: python scripts/oems_batch.py [select|run|report]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "fullwave_openems"
N_CASES = 3
NAMES = ["L1", "L2", "S1", "S2", "Wc1", "Wc2"]


def select():
    prereg = json.loads((ROOT / "results/step4/prereg.json").read_text(encoding="utf-8"))
    res = {r["design_id"]: r for r in json.loads((ROOT / "results/step4/test/results.json").read_text(encoding="utf-8"))}
    test = np.load(ROOT / "data/filter_supplement_v2/data/core_test.npz")
    ids = list(test["design_id"])
    cases, used = [], set()
    for did in prereg["cases"]["test"]["design_id"]:
        m = res[did]["methods"]["shap_ref"]
        pair = ",".join(m["subspace"])
        if not m["success"] or pair in used:
            continue
        traj = res[did]["trajectories"][pair]
        tuned = next(t for t in traj if t["hard"] <= 1e-9)
        g = tuned["geometry"]
        tuned["geometry"] = g if isinstance(g, str) else " ".join(map(str, g))  # stored as str or list
        x0 = test["geometry_mil"][ids.index(did)]
        cases.append({"design_id": did, "shap_pair": pair,
                      "x0": dict(zip(NAMES, map(float, x0))),
                      "tuned": dict(zip(NAMES, map(float, tuned["geometry"].split()))),
                      "ads_violation_x0_dB": res[did]["x0_hard"], "ads_violation_tuned_dB": tuned["hard"]})
        used.add(pair)
        if len(cases) == N_CASES:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps({"rule": __doc__.split("Case rule")[1].split("Usage")[0].strip(),
                                                "cases": cases}, indent=2))
    for c in cases:
        print(c["design_id"], c["shap_pair"], round(c["ads_violation_x0_dB"], 2), "->", c["ads_violation_tuned_dB"])
    return cases


def run():
    """python scripts/oems_batch.py run <mesh level from oems_mesh_study.LEVELS>"""
    import oems_filter as o
    from oems_mesh_study import LEVELS
    level = sys.argv[2]
    cases = json.loads((OUT / "cases.json").read_text())["cases"]
    for c in cases:
        for tag in ["x0", "tuned"]:
            d = OUT / "cases" / level / f"{c['design_id']}_{tag}"
            if (d / "summary.json").exists():
                print("skip", d.name, flush=True)
                continue
            d.mkdir(parents=True, exist_ok=True)
            print("run", d.name, c[tag], flush=True)
            r = o.simulate(c[tag], d / "sim", verbose=0, mesh_overrides=LEVELS[level])
            np.savez(d / "response.npz", frequency_Hz=r["frequency_Hz"], S_complex=r["S_complex"])
            y = o.to_db(r["S_complex"])
            v, comps = o.mask_violation(r["frequency_Hz"], y)
            s = {"geometry_mil": c[tag], "elapsed_s": r["elapsed_s"], "mesh_lines_xyz": r["mesh_lines_xyz"],
                 "mesh_level": level, "mesh": r["mesh"], **o.band_metrics(r["frequency_Hz"], y[:, 0]),
                 "mask_violation_dB": v, "mask_components_dB": comps}
            (d / "summary.json").write_text(json.dumps(s, indent=2))
            print("done", d.name, f"{r['elapsed_s']:.0f}s", "violation", round(v, 3), flush=True)


def ads_jobs():
    """Job list for scripts/ads_sim.py (ADS Python) so every openEMS geometry has a fresh ADS pair."""
    cases = json.loads((OUT / "cases.json").read_text())["cases"]
    jobs = [{"id": f"{c['design_id']}_{t}", "geometry": c[t]} for c in cases for t in ("x0", "tuned")]
    jobs.append({"id": "baseline", "geometry": {"L1": 138.5, "L2": 136.0, "S1": 3.3, "S2": 18.0, "Wc1": 25.5, "Wc2": 31.5}})
    (OUT / "ads_jobs.json").write_text(json.dumps(jobs, indent=1))
    print(len(jobs), "jobs")


if __name__ == "__main__":
    {"select": select, "run": run, "ads_jobs": ads_jobs}[sys.argv[1] if len(sys.argv) > 1 else "select"]()
