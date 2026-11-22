"""Step 1b: does a power-balance soft penalty help PR-Net? Validation only (test never loaded).

Loss = rf_loss + mu * penalty, penalty on P = |S21|^2 + |S11|^2 (train.power_penalty):
  passive   relu(P - 1)^2   lossless   (P - 1)^2
Base configuration fixed from step 1: lam 0.3, batch 16. Early stopping stays on rf_loss.

Decision rule (fixed before running):
  A  seed 2026, pen {passive, lossless} x mu {0.1, 1, 10}. Eligible: passes the gate and
     composite <= 1.10 x baseline seed-2026 composite. Choose the eligible setting with the smallest
     validation max |P - 1|. None eligible -> keep the baseline.
  B  seeds 2027, 2028 for the chosen setting. Recommend adopting it if every seed passes the gate,
     3-seed mean composite <= 1.10 x baseline mean, and 3-seed mean max |P - 1| <= 0.5 x baseline mean.
Power balance is also checked on a 10x denser grid (6001 points, 9-15 GHz) using the saved weights,
so a model that is lossless only at the 601 training frequencies is visible.
Runs whose metrics.json exists are reused. Usage: python scripts/step1b_passivity.py
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filterxai import data, metrics, models  # noqa: E402

S1 = ROOT / "results" / "step1"
RUNS = S1 / "runs"
TRAIN = ROOT / "scripts" / "step1_train.py"
BASE = "PRNet_lam0.3_s{seed}_b16"
PENS, MUS, SEEDS = ("passive", "lossless"), (0.1, 1.0, 10.0), (2026, 2027, 2028)
PARALLEL = 3
TOL_COMPOSITE, TOL_POWER = 1.10, 0.5


def tag(seed, pen=None, mu=None):
    return BASE.format(seed=seed) + (f"_{pen}{mu}" if pen else "")


def train_all(jobs):
    """jobs: list of (seed, pen, mu); trains missing ones, PARALLEL at a time."""
    todo = [j for j in jobs if not (RUNS / tag(*j) / "metrics.json").exists()]
    for i in range(0, len(todo), PARALLEL):
        procs = []
        for seed, pen, mu in todo[i:i + PARALLEL]:
            t = tag(seed, pen, mu)
            (RUNS / t).mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(TRAIN), "PRNet", "--lam", "0.3", "--batch", "16", "--seed", str(seed),
                   "--mu", str(mu), "--pen", pen]
            print(">>", t, flush=True)
            procs.append((t, subprocess.Popen(cmd, stdout=open(RUNS / t / "stdout.txt", "w"), stderr=subprocess.STDOUT)))
        for t, p in procs:
            if p.wait():
                raise RuntimeError(f"{t} failed, see {RUNS / t / 'stdout.txt'}")
            print("   done", t, flush=True)


ctx_va = data.load("validation")
X_va = torch.tensor(ctx_va["X"], dtype=torch.float32)
F_DENSE = np.linspace(9.0, 15.0, 6001)
INIT = models.prnet_init_from_vf(ROOT / "results" / "step0" / "vf_concepts_train.npz")


@torch.no_grad()
def dense_power(t):
    state = torch.load(RUNS / t / "model.pt", map_location="cpu")
    state.pop("s")
    m = models.PRNet(F_DENSE, INIT)
    m.load_state_dict(state, strict=False)
    m.eval()
    d = (torch.abs(m(X_va)) ** 2).sum(-1).numpy() - 1
    return {"max_abs": float(np.abs(d).max()), "max_excess": float(d.max()), "max_deficit": float(-d.min()),
            "mean_abs": float(np.abs(d).mean()), "p95_abs": float(np.quantile(np.abs(d), 0.95)),
            "frac_abs_gt_0.02": float((np.abs(d) > 0.02).mean())}


def summary(t):
    m = json.loads((RUNS / t / "metrics.json").read_text(encoding="utf-8"))
    v = m["validation"]["all"]
    return {"tag": t, "composite": v["composite_score"], "passes_gate": all(v["passes_gate"].values()),
            **{k: v[k] for k in metrics.GOALS}, "complex_RMSE": v["complex_RMSE"],
            "power_max_abs": v["power_balance_max_abs"], "power_mean_abs": v["power_balance_mean_abs"],
            "dense": dense_power(t), "epochs": m["training"]["epochs_run"],
            "resonator_f_MAE_MHz": m["concepts_vs_vf"]["natural_frequency_MAE_MHz"][1:4],
            "resonator_Q_rel_err": m["concepts_vs_vf"]["Q_median_rel_error"][1:4]}


# stage A
grid = [(2026, p, mu) for p in PENS for mu in MUS]
train_all(grid)
base = {s: summary(tag(s)) for s in SEEDS}
stage_a = {f"{p}{mu}": summary(tag(2026, p, mu)) for _, p, mu in grid}
limit_a = TOL_COMPOSITE * base[2026]["composite"]
eligible = {k: r for k, r in stage_a.items() if r["passes_gate"] and r["composite"] <= limit_a}
chosen = min(eligible, key=lambda k: eligible[k]["power_max_abs"]) if eligible else None
print("stage A eligible:", list(eligible), "chosen:", chosen, flush=True)

# stage B
report = {"rule": __doc__.split("Decision rule (fixed before running):")[1].split("Power balance")[0].strip(),
          "baseline": base, "stage_A": stage_a, "stage_A_composite_limit": limit_a, "chosen": chosen}
if chosen:
    pen, mu = next((p, m) for _, p, m in grid if f"{p}{m}" == chosen)
    train_all([(s, pen, mu) for s in SEEDS[1:]])
    pen_runs = {s: summary(tag(s, pen, mu)) for s in SEEDS}
    mean = lambda rs, k: float(np.mean([r[k] for r in rs.values()]))  # noqa: E731
    checks = {
        "every_seed_passes_gate": all(r["passes_gate"] for r in pen_runs.values()),
        "mean_composite_ok": mean(pen_runs, "composite") <= TOL_COMPOSITE * mean(base, "composite"),
        "mean_power_halved": mean(pen_runs, "power_max_abs") <= TOL_POWER * mean(base, "power_max_abs"),
    }
    report.update({"stage_B": pen_runs, "stage_B_checks": checks, "recommend_adopt": all(checks.values())})
else:
    report["recommend_adopt"] = False
(S1 / "passivity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# markdown
COLS = [("composite", "composite", 3), ("S21_full_MAE_dB", "S21 MAE", 3), ("edge_MAE_MHz", "带边沿 MAE", 2),
        ("edge_P95_MHz", "带边沿 P95", 1), ("power_max_abs", "max\\|P−1\\| (601)", 3),
        ("power_mean_abs", "mean\\|P−1\\|", 4)]
head = "| 配置 | " + " | ".join(c[1] for c in COLS) + " | max\\|P−1\\| (6001) | 最大超出 / 最大亏损 | 过门槛 | 谐振 f MAE (MHz) |"
row = lambda name, r: (f"| {name} | " + " | ".join(f"{r[k]:.{n}f}" for k, _, n in COLS) +  # noqa: E731
                       f" | {r['dense']['max_abs']:.3f} | {r['dense']['max_excess']:.3f} / {r['dense']['max_deficit']:.3f}"
                       f" | {'是' if r['passes_gate'] else '否'} | " + " / ".join(f"{x:.1f}" for x in r["resonator_f_MAE_MHz"]) + " |")
lines = ["# PR-Net 功率平衡软惩罚（验证集）", "", "决策规则（运行前固定）：", "", report["rule"], "",
         "## A：seed 2026", "", head, "|---" * (len(COLS) + 5) + "|", row("基线 μ=0", base[2026])]
lines += [row(k, r) for k, r in stage_a.items()]
lines += ["", f"composite 上限 {limit_a:.4f}；合格：{', '.join(eligible) or '无'}；选定：{chosen or '无（保留基线）'}"]
if chosen:
    lines += ["", f"## B：{chosen} 的 3 个种子 vs 基线", "", head, "|---" * (len(COLS) + 5) + "|"]
    lines += [row(f"基线 s{s}", base[s]) for s in SEEDS] + [row(f"{chosen} s{s}", report["stage_B"][s]) for s in SEEDS]
    lines += ["", "检查：" + "；".join(f"{k} = {v}" for k, v in report["stage_B_checks"].items())]
lines += ["", f"**建议{'采用' if report['recommend_adopt'] else '不采用'}**（按上面的规则）。"]
(S1 / "passivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
