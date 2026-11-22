"""Step 1 report: pool seeds of the chosen configurations, apply the §2.5 primary-model rule.

Reads results/step1/sweep_log.json and runs/*/metrics.json; writes
results/step1/validation_report.{json,md}. Validation only.
Usage: python scripts/step1_report.py
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "results" / "step1"
RUNS = S1 / "runs"
log = json.loads((S1 / "sweep_log.json").read_text(encoding="utf-8"))
load = lambda tag: json.loads((RUNS / tag / "metrics.json").read_text(encoding="utf-8"))  # noqa: E731

GROUPS = ("all", "legacy_4d", "new_6d", "single_band", "multiband")
KEYS = list(metrics.GOALS) + ["composite_score"]
EXTRA = ("complex_RMSE", "power_balance_max_abs")


def pool(tags):
    ms = [load(t) for t in tags]
    for m in ms:  # per-group composite over the finite gate metrics (multiband has no single-band edges)
        for g in GROUPS:
            v = m["validation"][g]
            v["composite_score"] = float(np.nanmean([v[k] / goal for k, goal in metrics.GOALS.items()]))
    out = {}
    for g in GROUPS:
        out[g] = {}
        for k in KEYS + list(EXTRA):
            v = [m["validation"][g].get(k) for m in ms]
            if all(x is not None for x in v):
                out[g][k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                             "worst": float(np.max(v))}
    out["all"]["passes_gate_every_seed"] = {k: all(m["validation"]["all"]["passes_gate"][k] for m in ms) for k in metrics.GOALS}
    out["epochs"] = [m["training"].get("epochs_run") for m in ms]
    if "concepts_vs_vf" in ms[0]:
        out["concepts_vs_vf"] = {k: np.mean([m["concepts_vs_vf"][k] for m in ms], 0).tolist() for k in ms[0]["concepts_vs_vf"]}
    return out


models = {name: pool(tags) for name, tags in log["seeds"].items()}
b1 = {t.name: load(t.name) for t in sorted(RUNS.glob("B1_r*"))}
if b1:
    best_b1 = min(b1, key=lambda t: b1[t]["validation"]["all"]["composite_score"])
    models["B1"] = pool([best_b1])
    models["B1"]["rank"] = b1[best_b1]["rank"]

passes = {n: all(m["all"]["passes_gate_every_seed"].values()) for n, m in models.items()}
if passes.get("PRNet"):
    primary = "PRNet"
    rule = "PR-Net passes the gate on every seed -> primary by design (§2.5)"
elif passes.get("M1"):
    primary = "M1"
    rule = "PR-Net fails, M1 passes -> M1 takes over RQ3; PR-Net keeps concept analysis (§2.5)"
else:
    primary = None
    rule = "Neither PR-Net nor M1 passes on every seed -> revisit before XAI"
report = {"chosen_configs": log["chosen"], "sweep": log["stages"], "models": models,
          "passes_gate_every_seed": passes, "primary_model": primary, "rule": rule,
          "B1_ranks": {t: b1[t]["validation"]["all"]["composite_score"] for t in b1}}
(S1 / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# markdown
fmt = lambda d: f"{d['mean']:.3f} ± {d['std']:.3f}" if d["std"] else f"{d['mean']:.3f}"  # noqa: E731
names = [n for n in ("PRNet", "M1", "M0", "B1") if n in models]
lines = ["# 第 1 步验证集结果", "", f"主模型判定：**{primary}**（{rule}）", "",
         "选定配置：" + "；".join(f"{n} {json.dumps(c)}" for n, c in log["chosen"].items()) +
         (f"；B1 rank {models['B1']['rank']}" if "B1" in models else ""), "",
         "## 门槛指标（验证集，3 个种子均值 ± 标准差；B1 为确定性模型）", "",
         "| 指标 | 目标 | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
for k in KEYS:
    goal = metrics.GOALS.get(k, "")
    lines.append(f"| {k} | {goal} | " + " | ".join(fmt(models[n]["all"][k]) for n in names) + " |")
lines.append("| 每个种子都过门槛 | | " + " | ".join("是" if passes[n] else "否" for n in names) + " |")
lines += ["", "## 分组（composite score 均值）", "", "| 组 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for g in GROUPS[1:]:
    lines.append(f"| {g} | " + " | ".join(f"{models[n][g]['composite_score']['mean']:.3f}" for n in names) + " |")
if "concepts_vs_vf" in models.get("PRNet", {}):
    c = models["PRNet"]["concepts_vs_vf"]
    lines += ["", "## PR-Net 极点 vs validation vector fitting（3 个种子均值）", "",
              "| 极点 | 1 背景 | 2 谐振 | 3 谐振 | 4 谐振 | 5 背景 |", "|---|---|---|---|---|---|",
              "| 自然频率 MAE (MHz) | " + " | ".join(f"{x:.1f}" for x in c["natural_frequency_MAE_MHz"]) + " |",
              "| Q 相对误差中位数 | " + " | ".join(f"{x:.3f}" for x in c["Q_median_rel_error"]) + " |"]
lines += ["", "## 超参数选择过程（seed 2026）", ""]
for st in log["stages"]:
    lines.append(f"- {st['stage']}: " + ", ".join(f"{o} → {s:.4f}" for o, s in st["scores"].items()) + f"；选 {st['chosen']}")
if b1:
    lines.append("- B1 rank: " + ", ".join(f"{t} → {s:.4f}" for t, s in report["B1_ranks"].items()))
(S1 / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
