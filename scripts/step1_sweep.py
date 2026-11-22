"""Step 1 driver: sequential hyperparameter selection on validation, then extra seeds.

Selection criterion: validation composite score (mean of metric/goal over the six gate metrics).
  PRNet  batch {16, 64} at lam 0.3, then lam {0.1, 0.3, 1} at the chosen batch
  M0     batch {16, 64} (paper layout otherwise fixed)
  M1     k {0, 1, 2, 4, 8} at lam 0.3, then lam {0.1, 0.3, 1} at the chosen k
  then seeds 2027, 2028 for each chosen configuration.
Runs whose metrics.json already exists are reused. Usage: python scripts/step1_sweep.py
"""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "step1" / "runs"
TRAIN = ROOT / "scripts" / "step1_train.py"
log = {"criterion": "validation composite score", "stages": []}


def run(model, seed=2026, lam=None, k=None, batch=None):
    tag = {"PRNet": f"PRNet_lam{lam}_s{seed}", "M0": f"M0_s{seed}", "M1": f"M1_k{k}_lam{lam}_s{seed}"}[model]
    tag += f"_b{batch}" if batch else ""
    mfile = RUNS / tag / "metrics.json"
    if not mfile.exists():
        cmd = [sys.executable, str(TRAIN), model, "--seed", str(seed)]
        cmd += ["--lam", str(lam)] if lam is not None else []
        cmd += ["--k", str(k)] if k is not None else []
        cmd += ["--batch", str(batch)] if batch else []
        print(">>", " ".join(cmd[2:]), flush=True)
        subprocess.run(cmd, check=True)
    m = json.loads(mfile.read_text(encoding="utf-8"))
    score = m["validation"]["all"]["composite_score"]
    print(f"   {tag}: composite {score:.4f}", flush=True)
    return tag, score


def choose(stage, options):
    results = {opt: run(**kw) for opt, kw in options.items()}
    best = min(results, key=lambda o: results[o][1])
    log["stages"].append({"stage": stage, "scores": {str(o): s for o, (_, s) in results.items()}, "chosen": str(best)})
    (ROOT / "results" / "step1" / "sweep_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return best


# batch 64 is the protocol default and is passed as None so tags match earlier runs
pb = choose("PRNet batch", {16: dict(model="PRNet", lam=0.3, batch=16), 64: dict(model="PRNet", lam=0.3)})
pb = pb if pb == 16 else None
pl = choose("PRNet lambda", {l: dict(model="PRNet", lam=l, batch=pb) for l in (0.1, 0.3, 1.0)})
mb = choose("M0 batch", {16: dict(model="M0", batch=16), 64: dict(model="M0")})
mb = mb if mb == 16 else None
mk = choose("M1 k", {k: dict(model="M1", k=k, lam=0.3) for k in (0, 1, 2, 4, 8)})
ml = choose("M1 lambda", {l: dict(model="M1", k=mk, lam=l) for l in (0.1, 0.3, 1.0)})

final = {"PRNet": dict(model="PRNet", lam=pl, batch=pb), "M0": dict(model="M0", batch=mb),
         "M1": dict(model="M1", k=mk, lam=ml)}
log["chosen"] = {name: {k: v for k, v in kw.items() if k != "model"} for name, kw in final.items()}
log["seeds"] = {name: [run(seed=s, **kw)[0] for s in (2026, 2027, 2028)] for name, kw in final.items()}
(ROOT / "results" / "step1" / "sweep_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
print(json.dumps(log["chosen"]), flush=True)
