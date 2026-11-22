"""Step 1 sanity check: band-edge extraction reproduces design_metrics.csv on train/validation,
and the vector-fitting curves (a near-perfect 'model') score near zero on every metric.
Usage: python scripts/step1_check_metrics.py
"""
from pathlib import Path
import csv
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, metrics, vectfit  # noqa: E402

with open(data.RELEASE / "data" / "design_metrics.csv", encoding="utf-8-sig", newline="") as fh:
    ref = {r["design_id"]: r for r in csv.DictReader(fh) if r["split"] in ("train", "validation")}

for split in ("train", "validation"):
    d = data.load(split)
    f = d["frequency_GHz"]
    e = metrics.band_edges(f, d["Y_dB"][..., 0])
    sb = d["single_band"]
    lo = np.array([float(ref[i]["f_lower_GHz"] or "nan") for i in d["design_id"]])
    hi = np.array([float(ref[i]["f_upper_GHz"] or "nan") for i in d["design_id"]])
    dl, dh = np.abs(e[sb, 0] - lo[sb]) * 1e3, np.abs(e[sb, 1] - hi[sb]) * 1e3
    print(f"{split}: single-band {sb.sum()}, clipped edges {np.isnan(e[sb]).sum()}, "
          f"max |edge - csv| lower {np.nanmax(dl):.2e} MHz upper {np.nanmax(dh):.2e} MHz")

    z = np.load(data.ROOT / "results" / "step0" / f"vf_concepts_{split}.npz")
    S_vf = np.stack([vectfit.evaluate(f, z["poles"][i], z["residues"][i], z["constants"][i]) for i in range(len(d["S"]))])
    r = metrics.evaluate(f, d["Y_dB"], metrics.to_db(S_vf), sb, metrics.standard_groups(d), d["S"], S_vf)["all"]
    print("  VF as model:", {k: round(r[k], 4) for k in metrics.GOALS}, "composite", round(r["composite_score"], 4))
