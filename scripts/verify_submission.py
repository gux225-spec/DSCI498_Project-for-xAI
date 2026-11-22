"""Read-only verification of archived experiments; writes a new audit directory only.

Usage: python scripts/verify_submission.py [--root PROJECT] [--output DIRECTORY] [--live-models]
Requires NumPy and SciPy; --live-models additionally requires PyTorch.
No training, ADS, openEMS installation, or simulation is invoked.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from math import comb
from pathlib import Path
import platform
import re
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest, rankdata, wilcoxon


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def masks(f):
    return [(f >= a - 1e-9) & (f <= b + 1e-9)
            for a, b in [(11.6, 12.4), (9, 10.8), (13.2, 15)]]


def hard_violation(f, db):
    pb, low, high = masks(f)
    return (np.maximum(0, -.5 - db[..., pb, 0].min(-1))
            + np.maximum(0, db[..., pb, 1].max(-1) + 10)
            + np.maximum(0, db[..., low, 0].max(-1) + 20)
            + np.maximum(0, db[..., high, 0].max(-1) + 15))


def curve_targets(f, db):
    """Independent NumPy formulation of the eleven curve targets, in stored order."""
    pb, low, high = masks(f)
    cen = (f >= 10.5 - 1e-9) & (f <= 13.5 + 1e-9)
    s21, s11 = db[..., 0], db[..., 1]
    weight = 10 ** (s21[:, cen] / 10)
    centroid = (weight * f[cen]).sum(-1) / weight.sum(-1)
    sm = lambda x: .05 * logsumexp(x / .05, axis=-1)
    sr = lambda x: .05 * np.logaddexp(0, x / .05)
    p21, p11 = s21[:, pb], s11[:, pb]
    worst = sm(p11)
    viol = (sr(-.5 + sm(-p21)) + sr(worst + 10)
            + sr(sm(s21[:, low]) + 20) + sr(sm(s21[:, high]) + 15))
    return np.column_stack([centroid] + [s21[:, np.argmin(abs(f - v))] for v in [10.8, 11.6, 12, 12.4, 13.2]]
                           + [s11[:, np.argmin(abs(f - 12))], p21.mean(-1),
                              sm(p21) + sm(-p21), worst, viol])


def band_edges(f, db, relative=True):
    """Peak-connected interval; relative peak-3 dB or absolute -3 dB, with interpolation."""
    out = np.full((len(db), 2), np.nan)
    for row, y in enumerate(db):
        peak = int(np.argmax(y))
        level = y[peak] - 3 if relative else -3
        if y[peak] < level:
            continue
        lo = hi = peak
        while lo > 0 and y[lo - 1] >= level:
            lo -= 1
        while hi < len(y) - 1 and y[hi + 1] >= level:
            hi += 1
        if lo > 0:
            out[row, 0] = f[lo - 1] + (level-y[lo - 1]) * (f[lo]-f[lo - 1]) / (y[lo]-y[lo - 1])
        if hi < len(y) - 1:
            out[row, 1] = f[hi] + (level-y[hi]) * (f[hi + 1]-f[hi]) / (y[hi + 1]-y[hi])
    return out


def shapley(values, n):
    """Enumerate marginal differences without importing the project's Shapley matrix."""
    phi = np.zeros(values.shape[:-2] + (values.shape[-1], n))
    for j in range(n):
        for coalition in range(1 << n):
            if coalition & (1 << j):
                continue
            phi[..., j] += (values[..., coalition | (1 << j), :] - values[..., coalition, :]) / (n * comb(n-1, coalition.bit_count()))
    return phi


def rho_abs(a, b):
    ar, br = rankdata(np.abs(a), axis=-1), rankdata(np.abs(b), axis=-1)
    ar, br = ar-ar.mean(-1, keepdims=True), br-br.mean(-1, keepdims=True)
    den = np.sqrt((ar*ar).sum(-1) * (br*br).sum(-1))
    return np.divide((ar*br).sum(-1), den, out=np.full(den.shape, np.nan), where=den > 0)


def holm(values):
    order = sorted(values, key=values.get)
    result, running = {}, 0.
    for i, name in enumerate(order):
        running = max(running, min(1., (len(order)-i)*values[name]))
        result[name] = running
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live-models", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output or root / "results/submission_audit").resolve()
    # An accidental output path must never overwrite a historical experiment file.
    protected = [root / "data", root / "src", root / "scripts", root / "report"]
    if any(out == p or p in out.parents for p in protected):
        raise SystemExit("Use a new audit output directory, outside data/src/scripts/report")
    report = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Archived-data recomputation and optional sample inference; no retraining or new simulation",
              "checks": [], "summary": {}, "environment": {"python": platform.python_version()}}
    for name in ["numpy", "scipy", "torch", "matplotlib", "scikit-learn", "shap", "lime"]:
        try:
            report["environment"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["environment"][name] = None

    def check(name, ok, details=None, warning=False):
        report["checks"].append({"name": name, "status": "pass" if ok else ("warning" if warning else "fail"), "details": details})

    def section(name, fn):
        print(name, flush=True)
        try:
            fn()
        except Exception as exc:
            check(name, False, f"{type(exc).__name__}: {exc}")

    release = root / "data/filter_supplement_v2"
    registry = read_json(root / "results/step1/final_models.json")
    sys.path.insert(0, str(root / "src"))
    from filterxai import data
    datasets = {}

    def integrity():
        files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
        for path in files:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        check("Python source parses", True, {"files": len(files)})
        hashes = read_json(release / "artifact_hashes.json")
        bad = [name for name, expected in hashes.items() if not (release/name).is_file() or digest(release/name) != expected]
        check("Dataset release SHA256 manifest", not bad, {"files": len(hashes), "mismatches": bad})
        freeze = read_json(root / "results/step3/freeze.json")
        changed = [name for name, expected in freeze["sha256"].items()
                   if not (root/name).is_file() or digest(root/name) != expected]
        documented = {"docs/algorithm_plan_v2.md", "docs/training_xai_protocol_v2.json",
                      "scripts/step2_xai.py", "scripts/step2_eval.py"}
        check("Frozen model weights and other unchanged artifacts", set(changed) <= documented,
              {"total": len(freeze["sha256"]), "matched": len(freeze["sha256"])-len(changed), "changed": changed})
        check("Historical freeze is not an immutable current snapshot", not changed,
              "Four historical hash differences are documented in plan section 12; original prior versions are not reconstructed.", warning=True)
        report["summary"]["freeze"] = {"recorded_utc": freeze["frozen_utc"], "changed": changed}
        prereg = read_json(root / "results/step4/prereg.json")
        report["summary"]["preregistration"] = {"recorded_utc": prereg["written_utc"],
            "plan_currently_matches_recorded_hash": digest(root/"docs/algorithm_plan_v2.md") == prereg["plan_sha256"]}

    def inputs():
        for split, count in [("train", 1008), ("validation", 216), ("test", 216)]:
            d = data.load(split, allow_test_labels=split == "test")
            datasets[split] = d
            check(f"{split} shapes and finite values", d["X"].shape == (count, 9) and d["S"].shape == (count, 601, 2)
                  and np.isfinite(d["X"]).all() and np.isfinite(d["S"]).all() and np.isfinite(d["Y_dB"]).all())
            check(f"{split} unique design IDs", len(set(d["design_id"])) == count)
            check(f"{split} frequency grid", np.allclose(d["frequency_GHz"], np.linspace(9, 15, 601), atol=1e-12, rtol=0))
            max_db = float(abs(20*np.log10(np.maximum(abs(d["S"]), 1e-150))-d["Y_dB"]).max())
            check(f"{split} dB derived from complex response", max_db < 1e-9, {"max_error_dB": max_db})
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            ids = set(datasets[a]["design_id"]) & set(datasets[b]["design_id"])
            ga = {tuple(row) for row in datasets[a]["X_raw"][:, :6]}
            gb = {tuple(row) for row in datasets[b]["X_raw"][:, :6]}
            check(f"{a}/{b} disjoint designs and geometry", not ids and not ga & gb)
        check("Train normalization", abs(datasets["train"]["X"].mean(0)).max() < 1e-6
              and abs(datasets["train"]["X"].std(0)-1).max() < 1e-6)
        default_test = data.load("test")
        check("Default test access does not return labels", default_test["labels_locked"] and "S" not in default_test and "Y_dB" not in default_test)
        report["summary"]["splits"] = {k: len(v["X"]) for k, v in datasets.items()}

    def accuracy():
        te = datasets["test"]
        f, true = te["frequency_GHz"], te["Y_dB"]
        true_edges = band_edges(f, true[..., 0])
        result = {}
        for kind in ["PRNet", "M1", "M0", "B1"]:
            rows = []
            for tag in registry[kind]["runs"]:
                base = root / "results/step3/test" / tag
                with np.load(base/"test_pred.npz", allow_pickle=False) as z:
                    pred = z["pred_dB"]
                    check(f"{tag} prediction IDs", np.array_equal(z["design_id"], te["design_id"]))
                stored = read_json(base/"metrics.json")
                err = abs(pred-true).mean((0, 1))
                check(f"{tag} saved S-parameter MAE", abs(err[0]-stored["test"]["all"]["S21_full_MAE_dB"]) < 1e-6
                      and abs(err[1]-stored["test"]["all"]["S11_full_MAE_dB"]) < 1e-6)
                ep = band_edges(f, pred[..., 0])
                valid = te["single_band"] & np.isfinite(true_edges).all(1) & np.isfinite(ep).all(1)
                et, ep = true_edges[valid], ep[valid]
                fc = float(abs(et.mean(1)-ep.mean(1)).mean()*1000)
                bt, bp = np.diff(et, axis=1).ravel(), np.diff(ep, axis=1).ravel()
                r2 = float(1-((bp-bt)**2).sum()/((bt-bt.mean())**2).sum())
                if "paper_measures_test" in stored:
                    s = stored["paper_measures_test"]
                    check(f"{tag} frequency and bandwidth metrics", abs(fc-s["center_frequency_MAE_MHz"]) < 1e-6
                          and abs(r2-s["bandwidth_R2"]) < 1e-9 and int(valid.sum()) == s["n"])
                rows.append([float(err[0]), float(err[1]), fc, r2, int(valid.sum())])
            mean = np.mean(rows, axis=0)
            result[kind] = dict(zip(["S21_MAE_dB", "S11_MAE_dB", "f0_MAE_MHz", "BW_R2", "n_valid_single_band"], mean.tolist()))
        report["summary"]["test_accuracy"] = result
        check("Frequency metric denominator disclosed", True, "Curve MAE uses 216 designs; f0/BW uses 202 valid single-band designs.")

    def attributions():
        with np.load(root/"results/step3/ads/truth.npz", allow_pickle=False) as z:
            truth = {k: z[k] for k in z.files}
        phi = shapley(truth["v_curve"], 6)
        check("ADS curve Shapley rebuilt from coalition values", np.max(abs(phi-truth["phi_curve"])) < 2e-5,
              {"designs": len(phi), "max_error": float(np.max(abs(phi-truth["phi_curve"])))})
        check("ADS Shapley efficiency", np.max(abs(phi.sum(-1)-(truth["v_curve"][:, -1]-truth["v_curve"][:, 0]))) < 2e-5)
        mapping = read_json(root/"results/step3/ads/map.json")["designs"]
        check("ADS mapping agrees with truth array IDs", [m["design_id"] for m in mapping] == truth["design_id"].tolist())
        f = datasets["test"]["frequency_GHz"]
        ds = {str(did): response for d in datasets.values() for did, response in zip(d["design_id"], d["S"])}
        full_error, target_error, missing = 0., 0., []
        for i, entry in enumerate(mapping):
            response = []
            for key in entry["coalitions"]:
                p = root / "results/step3/ads/cases" / f"{key}.npz"
                if not p.exists():
                    missing.append(key)
                    continue
                with np.load(p, allow_pickle=False) as z:
                    s = z["S_complex"]
                    response.append(np.stack([s[:, 1, 0], s[:, 0, 0]], axis=-1))
            if len(response) != 64:
                continue
            s = np.stack(response)
            full_error = max(full_error, float(abs(s[-1]-ds[entry["design_id"]]).max()))
            target = curve_targets(f, 20*np.log10(np.maximum(abs(s), 1e-15)))
            target_error = max(target_error, float(abs(target-truth["v_curve"][i]).max()))
        check("All 80 ADS coalition tables available", not missing, {"missing_unique": sorted(set(missing))})
        check("Full ADS coalitions reproduce original responses", full_error == 0, {"max_abs_dS": full_error})
        check("ADS target tables recomputed from cached complex responses", target_error < 1e-4, {"max_float64_vs_saved_float32_error": target_error})
        stored = read_json(root/"results/step3/ads_eval.json")
        groups = {"T0": [0], "T1": list(range(1, 7)), "T1b": [7], "T2": [8, 9, 10]}
        summary = {}
        for kind in ["PRNet", "M1", "M0"]:
            correlations = []
            efficiencies = []
            for tag in registry[kind]["runs"]:
                with np.load(root/"results/step3/attr"/f"{tag}.npz", allow_pickle=False) as a:
                    rebuilt = shapley(a["v_ref"], 9)
                    check(f"{tag} exact SHAP rebuilt", np.max(abs(rebuilt-a["shap_ref"])) < 2e-5)
                    efficiencies.append(float(abs(a["shap_ref"].sum(-1)-(a["v_ref"][:, -1]-a["v_ref"][:, 0])).max()))
                    correlations.append(rho_abs(a["shap_ref"][:40, :11, :6], truth["phi_curve"][40:]))
            values = {g: float(np.mean([np.nanmean(r[:, ix]) for r in correlations])) for g, ix in groups.items()}
            check(f"{kind} test SHAP vs ADS curve rank correlations", all(abs(values[g]-stored["decomposition"][kind]["test"]["shap_ref_vs_ADS"]["spearman_abs"][g]) < 1e-10 for g in groups))
            summary[kind] = {"spearman_abs": values, "max_efficiency_error": max(efficiencies)}
        report["summary"]["attribution"] = summary
        check("Shapley reference scope disclosed", True, "Agreement is with one simulator-defined Shapley game. IG/LIME definition differences are not automatically failures of faithfulness.")

    def tuning():
        rows = read_json(root/"results/step4/test/results.json")
        saved = read_json(root/"results/step4/step4_report.json")
        prereg = read_json(root/"results/step4/prereg.json")
        check("Tuning cases agree with preregistration", [r["design_id"] for r in rows] == prereg["cases"]["test"]["design_id"])
        summary = {}
        for name in rows[0]["methods"]:
            success = sum(r["methods"][name]["success"] for r in rows)
            calls = sum(r["methods"][name]["ads_calls"]+r["methods"][name]["extra_calls"] for r in rows)
            summary[name] = {"success": int(success), "n": len(rows), "ADS_calls": int(calls)}
            check(f"{name} success and call totals", success == saved["summary"][name]["success"] and calls == saved["summary"][name]["total_ads_calls"])
        raw_mc, raw_w, tests = {}, {}, {}
        rnd = np.array([r["methods"]["random"]["best_hard"] for r in rows])
        for name in ["shap_ref", "ig", "lime_delta"]:
            b = sum(r["methods"][name]["success"] and not r["methods"]["random"]["success"] for r in rows)
            c = sum(not r["methods"][name]["success"] and r["methods"]["random"]["success"] for r in rows)
            vals = np.array([r["methods"][name]["best_hard"] for r in rows])
            raw_mc[name] = float(binomtest(b, b+c, .5).pvalue) if b+c else 1.
            raw_w[name] = float(wilcoxon(vals, rnd, zero_method="wilcox").pvalue)
            tests[name] = {"only_method": int(b), "only_random": int(c), "mcnemar_p": raw_mc[name], "wilcoxon_p": raw_w[name]}
        adj_mc, adj_w = holm(raw_mc), holm(raw_w)
        for name in tests:
            tests[name].update(mcnemar_p_holm=adj_mc[name], wilcoxon_p_holm=adj_w[name])
            check(f"{name} primary statistics", all(abs(tests[name][key]-saved["primary_tests"][name][key]) < 1e-12
                  for key in ["mcnemar_p", "wilcoxon_p", "mcnemar_p_holm", "wilcoxon_p_holm"]))
        report["summary"]["tuning"] = summary
        report["summary"]["primary_tests"] = tests
        exploratory = {}
        sr = read_json(root/"results/step4b/step4b_report.json")
        for kind in ["PRNet", "M0"]:
            pairs = read_json(root/"results/step4b"/kind/"pairs.json")
            keys = [k for k in pairs[0]["subspaces"] if len(k.split(",")) == 2]
            check(f"{kind} exhaustive parameter pair count", len(keys) == 15)
            feasible = np.array([[r["subspaces"][k]["success"] for k in keys] for r in pairs])
            all6 = ",".join(data.GEOMETRY)
            vals = {"fixed_L1L2": sum(r["subspaces"]["L1,L2"]["success"] for r in pairs),
                    "random_expected": float(feasible.mean(1).sum()), "oracle_any_pair": int(feasible.any(1).sum()),
                    "all6": sum(r["subspaces"][all6]["success"] for r in pairs)}
            check(f"{kind} exploratory baselines", all(abs(v-sr["models"][kind]["success"][k]) < 1e-10 for k, v in vals.items()))
            exploratory[kind] = vals
        report["summary"]["exploratory"] = exploratory

    def fullwave():
        fw = root/"results/fullwave_openems"
        saved = read_json(fw/"report.json")
        result = []
        for row in saved["cases"]:
            did = row["design_id"]
            entry = {"design_id": did}
            for tag in ["x0", "tuned"]:
                for solver, path in [("ads", fw/"ads/cases"/f"{did}_{tag}.npz"),
                                     ("oems", fw/"cases/m4"/f"{did}_{tag}"/"response.npz")]:
                    with np.load(path, allow_pickle=False) as z:
                        f, s = z["frequency_Hz"]/1e9, z["S_complex"]
                    db = np.stack([20*np.log10(abs(s[:, 1, 0])), 20*np.log10(abs(s[:, 0, 0]))], -1)
                    hard = float(hard_violation(f, db))
                    er, ea = band_edges(f, db[None, :, 0]), band_edges(f, db[None, :, 0], relative=False)
                    entry[f"{solver}_{tag}"] = {"violation_dB": hard, "relative_BW_GHz": float(np.diff(er)[0, 0]),
                        "absolute_BW_GHz": float(np.diff(ea)[0, 0]), "relative_f0_GHz": float(er.mean())}
                    original = row[f"{solver}_{tag}"]
                    check(f"{did} {tag} {solver} mask and archived absolute bandwidth", abs(hard-original["mask_violation_dB"]) < 1e-8
                          and abs(float(np.diff(ea)[0, 0])-original["BW_GHz"]) < 1e-8)
            entry["relative_BW_reduction_MHz"] = 1000*(entry["ads_tuned"]["relative_BW_GHz"]-entry["oems_tuned"]["relative_BW_GHz"])
            result.append(entry)
        report["summary"]["fullwave"] = result
        check("Three ADS-passing tuned cases fail archived full-wave mask", len(result) == 3 and all(r["ads_tuned"]["violation_dB"] <= 1e-9 and r["oems_tuned"]["violation_dB"] > 0 for r in result))
        check("Bandwidth conventions require separate labels", False,
              "Historical model metrics use peak-relative -3 dB; full-wave summaries use absolute -3 dB. Both are retained and recomputed here; the hard-mask verdict is unaffected.", warning=True)

    def documents_and_case():
        tex = (root/"report/main.tex").read_text(encoding="utf-8")
        bib = (root/"report/references.bib").read_text(encoding="utf-8")
        keys = set(re.findall(r"@\w+\s*\{\s*([^,]+)", bib))
        cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex) for k in group.split(",")}
        figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
        check("LaTeX citation keys resolve", not cited-keys, {"missing": sorted(cited-keys)})
        check("LaTeX figure files exist", all((root/"report/figures"/name).exists() for name in figs))
        todos = [i for i, line in enumerate(tex.splitlines(), 1) if "\\todo{" in line and not line.startswith("\\newcommand")]
        final_source = root/"report/final/main.tex"
        if final_source.exists():
            final_tex = final_source.read_text(encoding="utf-8")
            check("Final source has no TODO placeholders", "\\todo" not in final_tex)
            final_bib = (root/"report/final/references.bib").read_text(encoding="utf-8")
            final_keys = set(re.findall(r"@\w+\s*\{\s*([^,]+)", final_bib))
            final_cites = {k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", final_tex) for k in g.split(",")}
            check("Final citation keys resolve", not final_cites-final_keys)
            artifact_checks = read_json(root/"results/submission_audit/final_artifacts.json")
            for doc in artifact_checks["documents"]:
                lo, hi = doc["page_requirement"]
                check("Compiled artifact integrity: "+doc["path"], (root/doc["path"]).exists()
                      and digest(root/doc["path"]) == doc["sha256"] and lo <= doc["pages"] <= hi,
                      {"pages": doc["pages"], "page_requirement": [lo, hi]})
            report["summary"]["final_artifacts"] = artifact_checks
            check("Original draft placeholders retained as history", True, {"draft_TODO_lines": todos, "final_source": "report/final/main.tex"})
        else:
            check("Draft still contains TODOs and is not the final report", not todos, {"lines": todos}, warning=True)
        with np.load(root/"results/step3/setup_test.npz", allow_pickle=False) as z:
            i = z["explain_id"].tolist().index("test_0005")
            actual = z["y_explain_true"]
        with np.load(root/"results/step3/attr"/f"{registry['PRNet']['runs'][0]}.npz", allow_pickle=False) as a:
            t = a["names"].tolist().index("T2_mask_violation_dB")
            report["summary"]["test_0005_mask_quality"] = {"LIME_R2_scale025_seed0": float(a["lime_r2"][0, i, t]),
                "IG_relative_residual": float(a["ig_resid"][i, t]), "IG_steps": int(a["ig_steps"][i, t]),
                "smooth_model_vs_ADS_error_dB": float(abs(a["F_x"][i, t]-actual[i, t]))}
        check("Historical violation-error statistic mixes smooth and hard targets", False,
              "step4b_report.py compares surrogate smooth violation with ADS hard violation. Its 0.062 dB median is not a same-target prediction error; the case card also uses a population median for one design.", warning=True)
        check("Conclusion and generalization limits", True,
              "No measured practitioner time/decision gain; three selected full-wave cases are not a failure-rate estimate; baseline mesh sensitivity does not prove case convergence; missing two-port power alone does not isolate radiation.")

    def inference():
        import torch
        from filterxai import zoo
        from filterxai.train import DEV
        live = {}
        for kind in ["PRNet", "M1", "M0"]:
            model = zoo.load(kind, 2026)[0]
            with torch.no_grad():
                pred = model.db(torch.tensor(datasets["test"]["X"][:3], device=DEV)).cpu().numpy()
            with np.load(root/"results/step3/test"/model.tag/"test_pred.npz", allow_pickle=False) as z:
                error = float(abs(pred-z["pred_dB"][:3]).max())
            live[kind] = {"designs": 3, "max_abs_difference_dB": error, "device": DEV}
            check(f"{kind} current-weight sample inference", error < .02, live[kind])
        report["summary"]["live_inference"] = live

    for title, fn in [("Integrity and historical hashes", integrity), ("Data and split checks", inputs),
                      ("Saved prediction accuracy", accuracy), ("Attribution and ADS coalition checks", attributions),
                      ("Tuning counts and primary statistics", tuning), ("Full-wave response checks", fullwave),
                      ("Draft and interpretation checks", documents_and_case)]:
        section(title, fn)
    if args.live_models:
        section("Current-weight sample inference", inference)
    report["counts"] = {s: sum(c["status"] == s for c in report["checks"]) for s in ["pass", "warning", "fail"]}
    report["status"] = "needs_revision" if report["counts"]["fail"] else "verified_with_documented_caveats"
    report["not_run"] = ["Model retraining", "New ADS/openEMS/Momentum simulations", "Vector-fitting refits", "LaTeX compilation", "Practitioner performance study"]
    if not args.live_models:
        report["not_run"].append("Current-weight inference (enable with --live-models)")
    out.mkdir(parents=True, exist_ok=True)
    (out/"verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Archived result verification", "", f"Recorded UTC: {report['recorded_utc']}", "", f"Status: **{report['status']}**", "",
             report["scope"], "", "## Recomputed test accuracy", "", "| Model | S21 MAE dB | S11 MAE dB | f0 MAE MHz | BW R2 | f0 and BW designs |", "|---|---:|---:|---:|---:|---:|"]
    for kind, v in report["summary"].get("test_accuracy", {}).items():
        lines.append(f"| {kind} | {v['S21_MAE_dB']:.6f} | {v['S11_MAE_dB']:.6f} | {v['f0_MAE_MHz']:.6f} | {v['BW_R2']:.6f} | {int(v['n_valid_single_band'])} |")
    lines += ["", "Curve errors use all 216 test designs; frequency and bandwidth errors use the valid single-band subset.", "", "## Recomputed tuning results", "", "| Method | Success | ADS calls including reference attribution |", "|---|---:|---:|"]
    for name, v in report["summary"].get("tuning", {}).items():
        lines.append(f"| {name} | {v['success']}/{v['n']} | {v['ADS_calls']} |")
    lines += ["", "Calls follow the original per-method accounting, including shared cached trajectories as independent method calls; these are not unique new solver launches.", "", "## Findings and retained caveats", ""]
    for c in report["checks"]:
        if c["status"] != "pass":
            lines.append(f"- **{c['status'].upper()} — {c['name']}**: {c['details']}")
    lines += ["", "## Full-wave bandwidth definitions", "", "| Tuned design | ADS hard V dB | openEMS hard V dB | Relative bandwidth reduction MHz |", "|---|---:|---:|---:|"]
    for row in report["summary"].get("fullwave", []):
        lines.append(f"| {row['design_id']} | {row['ads_tuned']['violation_dB']:.6f} | {row['oems_tuned']['violation_dB']:.6f} | {row['relative_BW_reduction_MHz']:.3f} |")
    lines += ["", "## Scope limits", "", "The following were outside this verification run:", ""] + [f"- {item}" for item in report["not_run"]]
    lines += ["", "## Detailed checks", "", "| Check | Status |", "|---|---|"] + [f"| {c['name']} | {c['status']} |" for c in report["checks"]]
    lines += ["", "Full numeric values, settings, environment and check details are in `verification.json` alongside this file.", ""]
    (out/"verification.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(out)}, indent=2))
    return 1 if report["counts"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
