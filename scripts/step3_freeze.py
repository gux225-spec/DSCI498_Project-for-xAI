"""Step 3: configuration freeze. Records SHA256 of every artifact that determines the test results and the
step-3 protocol, BEFORE any test label is read (data.load refuses test labels until this file exists).

Writes results/step3/freeze.json. Refuses to overwrite an existing freeze.
Usage: python scripts/step3_freeze.py
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step3" / "freeze.json"
if OUT.exists():
    raise SystemExit("freeze already exists: " + str(OUT))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
reg = json.loads((ROOT / "results" / "step1" / "final_models.json").read_text(encoding="utf-8"))
files = [ROOT / "results" / "step1" / "final_models.json", ROOT / "results" / "step2" / "setup.json",
         ROOT / "results" / "step2" / "setup.npz", ROOT / "ads" / "baseline.ckt",
         ROOT / "docs" / "algorithm_plan_v2.md", ROOT / "docs" / "training_xai_protocol_v2.json"]
files += sorted((ROOT / "src" / "filterxai").glob("*.py"))
files += [ROOT / "scripts" / n for n in ("step1_train.py", "step1_b1_gpr.py", "step2_setup.py", "step2_xai.py", "step2_eval.py", "ads_sim.py")]
for kind in ("PRNet", "M1", "M0"):
    files += [ROOT / "results" / "step1" / "runs" / t / "model.pt" for t in reg[kind]["runs"]]

protocol = {
    "test_accuracy": "PRNet, M1, M0 (3 seeds each) and B1 (rank 32, refit with the step-1 settings, validation predictions must reproduce) on all 216 test designs; same gate metrics and groups as step 1; compared with the validation results and Suslova et al. 2026",
    "test_explained_designs": "40 single-band test designs by the step-2 rule: all single-band mask passes + failing designs by source (largest remainder), seed 2026",
    "test_XAI": "identical to step 2 (scripts/step2_xai.py, same reference, backgrounds, LIME settings, seeds); no setting may change after reading test labels",
    "ADS_true_shapley": "6 geometry features, single-reference interventional game (absent features at the ADS baseline geometry), 64 coalitions per design for the 40 validation + 40 test explained designs; empty and full coalitions taken from existing simulations; curve targets via filterxai.targets; T3 truth by vector fitting (N=10) of every coalition response",
    "ADS_local_finite_difference": "per geometry parameter h = 2% of the declared span; central difference when x +- h are both in bounds, otherwise one-sided using the design itself; compared with model gradients",
    "comparisons": {
      "explanation_vs_network": "explainer (SHAP 9-feature geometry part, IG, LIME-delta) vs the surrogate's exact 6-feature game (controls at reference, read from the 512-coalition table)",
      "network_vs_physics": "surrogate 6-feature game vs ADS 6-feature game",
      "end_to_end": "explainer vs ADS game",
      "metrics": "Spearman of |phi| over 6 geometry features, sign agreement on features with |phi_ADS| >= 1% of target train std, MAE of phi / target train std"},
    "paper_criterion_test": "per design and target: (a) paper protocol rho(GradientExplainer, median lime-package LIME) vs rho(GradientExplainer, ADS Shapley); (b) matched game rho(SHAP_ref, LIME-delta) vs rho(SHAP_ref, ADS Shapley); report Spearman across design-target pairs and mean truth agreement for designs above vs below the median SHAP-LIME rho",
    "not_in_step3": "C counterfactual tuning (step 4)"}
rec = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
       "statement": "No test label was read before this record. Every file below is frozen for step 3.",
       "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}, "step3_protocol": protocol}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(f"frozen {len(files)} files -> {OUT}")
