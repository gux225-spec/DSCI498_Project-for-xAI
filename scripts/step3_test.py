"""Step 3 (after freeze): test accuracy of the frozen neural models and the fixed test explained designs.

1. PRNet, M1, M0 (3 seeds each, results/step1/final_models.json) on all 216 test designs: step-1 gate metrics
   and groups, plus the paper's accuracy measures on single-band designs (center frequency = mean of the
   -3 dB edges, MAE in MHz; bandwidth R^2). The same paper measures are computed on validation for reference.
2. 40 test explained designs by the step-2 rule (all single-band mask passes + failing designs by source,
   largest remainder, seed 2026) -> results/step3/setup_test.{npz,json} with the step-2 reference,
   bounds, backgrounds and diagnostics unchanged.
Writes results/step3/test/<tag>/{metrics.json, test_pred.npz}. Usage: python scripts/step3_test.py
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, metrics, targets, zoo  # noqa: E402
from filterxai.train import DEV  # noqa: E402

S3 = data.ROOT / "results" / "step3"
reg = json.loads(zoo.REGISTRY.read_text(encoding="utf-8"))
te, va = data.load("test", allow_test_labels=True), data.load("validation")
f = te["frequency_GHz"]


def paper_measures(true_db, pred_db, single_band):
    et, ep = metrics.band_edges(f, true_db[..., 0]), metrics.band_edges(f, pred_db[..., 0])
    ok = single_band & np.isfinite(et).all(1) & np.isfinite(ep).all(1)
    f0t, f0p = et[ok].mean(1), ep[ok].mean(1)
    bwt, bwp = np.diff(et[ok], axis=1)[:, 0], np.diff(ep[ok], axis=1)[:, 0]
    return {"n": int(ok.sum()), "center_frequency_MAE_MHz": float(1e3 * np.abs(f0p - f0t).mean()),
            "center_frequency_R2": float(1 - np.sum((f0p - f0t) ** 2) / np.sum((f0t - f0t.mean()) ** 2)),
            "bandwidth_MAE_MHz": float(1e3 * np.abs(bwp - bwt).mean()),
            "bandwidth_R2": float(1 - np.sum((bwp - bwt) ** 2) / np.sum((bwt - bwt.mean()) ** 2))}


@torch.no_grad()
def predict(m, X):
    x = torch.tensor(X, device=DEV)
    db = m.db(x).cpu().numpy()
    S = None if m.kind == "M0" else (m.net(x) if m.kind == "PRNet" else m.net.curves(x, m.fenc)).cpu().numpy()
    return db, S


# 1. test accuracy
for kind in ("PRNet", "M1", "M0"):
    for m in zoo.load(kind):
        db, S = predict(m, te["X"])
        res = metrics.evaluate(f, te["Y_dB"], db, te["single_band"], metrics.standard_groups(te), te["S"] if S is not None else None, S)
        db_va, _ = predict(m, va["X"])
        out = S3 / "test" / m.tag
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({"tag": m.tag, "model": kind, "test": res,
                                                      "paper_measures_test": paper_measures(te["Y_dB"], db, te["single_band"]),
                                                      "paper_measures_validation": paper_measures(va["Y_dB"], db_va, va["single_band"])}, indent=2), encoding="utf-8")
        np.savez_compressed(out / "test_pred.npz", design_id=te["design_id"], pred_dB=db, **({"pred_S": S} if S is not None else {}))
        print("TEST", m.tag, {k: round(res["all"][k], 4) for k in metrics.GOALS}, "composite", round(res["all"]["composite_score"], 4), flush=True)

# 2. test explained designs (same rule and seed as step 2)
N_EXPLAIN, SEED = 40, 2026
rng = np.random.default_rng(SEED)
sb = te["single_band"]
pick = list(np.flatnonzero(sb & te["mask_pass"]))
fails = {s: np.flatnonzero(sb & ~te["mask_pass"] & (te["source"] == s)) for s in sorted(set(te["source"]))}
slots = N_EXPLAIN - len(pick)
tot = sum(len(v) for v in fails.values())
quota = {s: slots * len(v) / tot for s, v in fails.items()}
alloc = {s: int(np.floor(q)) for s, q in quota.items()}
for s in sorted(quota, key=lambda s: quota[s] - alloc[s], reverse=True)[:slots - sum(alloc.values())]:
    alloc[s] += 1
for s, idx in fails.items():
    pick += list(rng.choice(idx, alloc[s], replace=False))
pick = np.array(sorted(pick))
strata = {f"{s}|{'pass' if p else 'fail'}": int(np.sum((te["source"][pick] == s) & (te["mask_pass"][pick] == p)))
          for s in fails for p in (True, False)}
z2 = dict(np.load(data.ROOT / "results" / "step2" / "setup.npz"))
T = targets.CurveTargets(f)
with torch.no_grad():
    y_true = T(torch.tensor(te["Y_dB"][pick], dtype=torch.float32)).numpy()
z2.update(explain_idx=pick, explain_id=te["design_id"][pick], x_explain=te["X"][pick], y_explain_true=y_true,
          x_raw_explain=te["X_raw"][pick])
np.savez_compressed(S3 / "setup_test.npz", **z2)
s2 = json.loads((data.ROOT / "results" / "step2" / "setup.json").read_text(encoding="utf-8"))
s2["explained"] = {"split": "test", "n": int(len(pick)), "strata": strata, "design_id": te["design_id"][pick].tolist(),
                   "rule": "all single-band mask passes + failing designs by source (largest remainder), seed 2026"}
(S3 / "setup_test.json").write_text(json.dumps(s2, indent=2), encoding="utf-8")
print("test explained:", strata)
