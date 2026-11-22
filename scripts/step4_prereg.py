"""Step 4 pre-registration: fixes the cases and every tuning setting before any tuning code runs.

Writes results/step4/prereg.json (refuses to overwrite). The settings mirror docs/algorithm_plan_v2.md §5 C.
Usage: python scripts/step4_prereg.py
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data  # noqa: E402

OUT = data.ROOT / "results" / "step4" / "prereg.json"
if OUT.exists():
    raise SystemExit("pre-registration already exists: " + str(OUT))
cases = {}
for split, n in (("test", 30), ("validation", 5)):
    d = data.load(split, allow_test_labels=split == "test")
    pool = np.flatnonzero(d["single_band"] & ~d["mask_pass"])
    pick = np.sort(np.random.default_rng(2026).choice(pool, n, replace=False))
    cases[split] = {"pool_size": int(len(pool)), "index": pick.tolist(), "design_id": d["design_id"][pick].tolist()}
plan = data.ROOT / "docs" / "algorithm_plan_v2.md"
rec = {
    "written_utc": datetime.now(timezone.utc).isoformat(),
    "statement": "Written before any tuning code was run; no setting below may change after test-case results are seen.",
    "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    "cases": cases, "case_rule": "single-band designs failing the mask, seed 2026; validation cases only for debugging code",
    "model": "PRNet_lam0.3_s2026_b16_lossless10.0", "target": "T2_mask_violation_dB (smooth, tau 0.05 dB)",
    "fixed_inputs": "negative controls stay at the design's own values",
    "k": 2,
    "selectors": {
        "primary": ["shap_ref", "ig", "lime_delta", "random"],
        "secondary": ["grad_x_span", "lime_slope_x_span", "ads_true_shap", "shap_revert_no_search"],
        "reference": ["all6_surrogate", "ads_coordinate"],
        "attribution": "|phi| over the 6 geometry features; SHAP/IG/LIME settings exactly as in step 2/3",
        "random_seed": "2026 + case index (0-based within the 30 test cases)"},
    "surrogate_search": {
        "objective": "V_m(x) + 0.01 * ||(x - x0)/span||^2, V_m = smooth mask violation with all four limits tightened by m dB",
        "k2": "41x41 grid over the box, best 5 grid points refined by 200 projected Adam steps (lr 0.01, normalized units)",
        "k6": "512 starts (x0, baseline, 510 uniform random, seed 2026) x 300 projected Adam steps"},
    "ads_confirmation": {"budget_per_method_case": 10, "margin_dB": "0.25 * (k - 1) for call k = 1..10",
                         "repeat_proposal": "no simulation, margin escalates", "stop": "first ADS pass (hard violation <= 1e-9)",
                         "shared_trajectories": "identical subspaces share simulations; counts are per method as if independent"},
    "ads_coordinate": {"budget": 60, "initial_step": "25% of declared span", "order": "L1, L2, S1, S2, Wc1, Wc2; +step then -step",
                       "accept": "hard ADS violation decreases", "shrink": "halve all steps after a full cycle without improvement"},
    "ads_true_shap": "64 coalition simulations per case (single reference, same game as step 3), counted separately",
    "metrics": ["ADS-confirmed success", "ADS calls to success", "best ADS hard violation on the trajectory",
                "normalized geometry change ||delta/span||_2 of the final design", "overlap of selected parameter pairs"],
    "tests": {"primary": "shap_ref / ig / lime_delta vs random: exact McNemar on success, Wilcoxon signed-rank on best violation, Holm over the three",
              "secondary": "descriptive only"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(json.dumps(cases, indent=1))
