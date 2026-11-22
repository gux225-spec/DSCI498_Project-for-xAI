"""Step 1: train one surrogate configuration and score it on validation (test never loaded).

Usage:
  python scripts/step1_train.py PRNet --lam 0.3 --seed 2026
  python scripts/step1_train.py M1 --k 0 --lam 0.3 --seed 2026
  python scripts/step1_train.py M0 --seed 2026
Writes results/step1/runs/<tag>/{metrics.json, model.pt, val_pred.npz}.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, metrics, models, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["PRNet", "M0", "M1"])
ap.add_argument("--lam", type=float, default=0.3)
ap.add_argument("--k", type=int, default=0)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--max-epochs", type=int, default=2000)
ap.add_argument("--patience", type=int, default=50)
ap.add_argument("--batch", type=int, default=None, help="designs (PRNet/M0) or rows (M1); default per protocol")
ap.add_argument("--mu", type=float, default=0.0, help="PRNet power-balance penalty weight (0 = off)")
ap.add_argument("--pen", choices=["passive", "lossless"], default="lossless")
ap.add_argument("--tag", default=None)
a = ap.parse_args()

tag = a.tag or {"PRNet": f"PRNet_lam{a.lam}_s{a.seed}", "M0": f"M0_s{a.seed}",
                "M1": f"M1_k{a.k}_lam{a.lam}_s{a.seed}"}[a.model] + (f"_b{a.batch}" if a.batch else "")
tag += f"_{a.pen}{a.mu}" if a.tag is None and a.model == "PRNet" and a.mu else ""
bkw = {"batch": a.batch} if a.batch else {}
out = data.ROOT / "results" / "step1" / "runs" / tag
out.mkdir(parents=True, exist_ok=True)
print(tag, "device", train.DEV, flush=True)

ctx = train.Context()
if a.model == "PRNet":
    init = models.prnet_init_from_vf(data.ROOT / "results" / "step0" / "vf_concepts_train.npz")
    model, info = train.train_prnet(ctx, init, a.lam, a.seed, a.max_epochs, a.patience, mu=a.mu, pen=a.pen, **bkw)
elif a.model == "M0":
    model, info = train.train_m0(ctx, a.seed, a.max_epochs, a.patience, **bkw)
else:
    model, info = train.train_m1(ctx, a.k, a.lam, a.seed, a.max_epochs, a.patience, **bkw)

pred_db, pred_S = train.predict(a.model, model, ctx.Xva)
va = ctx.va
res = metrics.evaluate(ctx.f, va["Y_dB"], pred_db, va["single_band"], metrics.standard_groups(va),
                       va["S"] if pred_S is not None else None, pred_S)
summary = {"tag": tag, "model": a.model, "lam": a.lam if a.model != "M0" else None,
           "k": a.k if a.model == "M1" else None, "seed": a.seed,
           **({"mu": a.mu, "pen": a.pen} if a.model == "PRNet" else {}),
           "training": {k: v for k, v in info.items() if k != "history"}, "validation": res}
if a.model == "PRNet":  # concept check vs validation vector fitting
    with torch.no_grad():
        fn, q, _, _ = model.concepts(ctx.Xva)
    z = np.load(data.ROOT / "results" / "step0" / "vf_concepts_validation.npz")
    fn, q = fn.cpu().numpy(), q.cpu().numpy()
    summary["concepts_vs_vf"] = {
        "natural_frequency_MAE_MHz": (1e3 * np.abs(fn - z["natural_frequency_GHz"]).mean(0)).tolist(),
        "Q_median_rel_error": np.median(np.abs(q / z["Q"] - 1), 0).tolist(),
    }
(out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(out / "history.json").write_text(json.dumps(info["history"]), encoding="utf-8")
torch.save(model.state_dict(), out / "model.pt")
np.savez_compressed(out / "val_pred.npz", design_id=va["design_id"], pred_dB=pred_db,
                    **({"pred_S": pred_S} if pred_S is not None else {}))
r = res["all"]
print("VAL", tag, {k: round(r[k], 4) for k in metrics.GOALS}, "composite", round(r["composite_score"], 4),
      "epochs", info["epochs_run"], "s", info["seconds"])
if "concepts_vs_vf" in summary:
    print("CONCEPTS", json.dumps(summary["concepts_vs_vf"]))
