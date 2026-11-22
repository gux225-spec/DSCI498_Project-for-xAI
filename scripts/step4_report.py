"""Step 4 report (RQ3): per-method ADS-confirmed results and the pre-registered tests.

Reads results/step4/test/{results.json, selections.json}; writes results/step4/step4_report.{json,md}.
Primary tests (prereg): SHAP / IG / LIME-delta vs random; exact McNemar on success, Wilcoxon signed-rank
on the best ADS violation, Holm over the three comparisons. Everything else is descriptive.
Usage: python scripts/step4_report.py [split]
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
split = sys.argv[1] if len(sys.argv) > 1 else "test"
D = ROOT / "results" / "step4" / split
R = json.loads((D / "results.json").read_text(encoding="utf-8"))
S = json.loads((D / "selections.json").read_text(encoding="utf-8"))["selections"]
M = list(R[0]["methods"])
LAB = {"shap_ref": "SHAP（单一参考）", "ig": "IG", "lime_delta": "LIME-delta", "random": "随机 2 个",
       "grad_x_span": "梯度 × 跨度", "lime_slope_x_span": "LIME 斜率 × 跨度", "ads_true_shap": "ADS 真值 Shapley 前 2",
       "shap_revert_no_search": "SHAP 前 2 改回基线（不搜索）", "all6_surrogate": "6 个参数代理搜索", "ads_coordinate": "ADS 坐标搜索"}
n = len(R)
col = lambda m, k: np.array([r["methods"][m][k] if r["methods"][m][k] is not None else np.nan for r in R], float)  # noqa: E731

summary = {}
for m in M:
    ok = col(m, "success").astype(bool)
    calls = col(m, "ads_calls") + col(m, "extra_calls")
    summary[m] = {"success": int(ok.sum()), "n": n, "rate": float(ok.mean()),
                  "calls_to_success_median": float(np.nanmedian(col(m, "calls_to_success"))) if ok.any() else None,
                  "total_ads_calls": int(np.nansum(calls)),
                  "best_violation_median": float(np.median(col(m, "best_hard"))),
                  "best_violation_mean": float(np.mean(col(m, "best_hard"))),
                  "change_median_success": float(np.median(col(m, "final_change")[ok])) if ok.any() else None}

tests = {}
rnd_ok, rnd_v = col("random", "success").astype(bool), col("random", "best_hard")
for m in ("shap_ref", "ig", "lime_delta"):
    ok, v = col(m, "success").astype(bool), col(m, "best_hard")
    b, c = int(np.sum(ok & ~rnd_ok)), int(np.sum(~ok & rnd_ok))
    p_mc = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    diff = v - rnd_v
    p_w = wilcoxon(v, rnd_v, zero_method="wilcox").pvalue if np.any(diff != 0) else 1.0
    tests[m] = {"only_method_ok": b, "only_random_ok": c, "mcnemar_p": float(p_mc), "wilcoxon_p": float(p_w),
                "median_violation_diff": float(np.median(diff))}
for key in ("mcnemar_p", "wilcoxon_p"):  # Holm over the three primary comparisons
    order = sorted(tests, key=lambda m: tests[m][key])
    adj, run = {}, 0.0
    for k, m in enumerate(order):
        run = max(run, min(1.0, (3 - k) * tests[m][key]))
        adj[m] = run
    for m in tests:
        tests[m][key + "_holm"] = adj[m]

pairs = {m: [tuple(S[str(i)][m]) for i in range(n)] for m in S["0"] if m != "all6_surrogate"}
overlap = {m: {"same_as_shap": float(np.mean([a == b for a, b in zip(pairs[m], pairs["shap_ref"])])),
               "same_as_ads_truth": float(np.mean([a == b for a, b in zip(pairs[m], pairs["ads_true_shap"])]))} for m in pairs}
G = ["L1", "L2", "S1", "S2", "Wc1", "Wc2"]
pair_counts = {m: {",".join(G[j] for j in p): pairs[m].count(p) for p in sorted(set(pairs[m]))} for m in pairs}
rep = {"split": split, "n_cases": n, "summary": summary, "primary_tests": tests, "selection_overlap": overlap, "pair_counts": pair_counts}
(D.parent / f"step4_report{'' if split == 'test' else '_' + split}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

f2 = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
L = [f"# 第 4 步：解释引导的调谐（RQ3，{split}，{n} 个未通过 mask 的单通带设计）", "",
     "设置全部按 `results/step4/prereg.json` 预先登记；成功 = ADS 复核的原始 mask 违反量为 0。", "",
     "| 方法 | 成功 | 成功率 | 成功所需仿真（中位数） | 总仿真次数 | 最好违反量 中位数 / 均值 (dB) | 成功设计的几何改动（中位数，跨度单位） |",
     "|---|---|---|---|---|---|---|"]
for m in M:
    s = summary[m]
    L.append(f"| {LAB.get(m, m)} | {s['success']}/{n} | {s['rate']:.2f} | {f2(s['calls_to_success_median'])} | {s['total_ads_calls']} | "
             f"{s['best_violation_median']:.2f} / {s['best_violation_mean']:.2f} | {f2(s['change_median_success'])} |")
L += ["", "ADS 真值 Shapley 前 2 的总仿真次数含每例 64 次联盟仿真。", "",
      "## 预先登记的主要检验（vs 随机 2 个）", "",
      "| 方法 | 只有该方法成功 | 只有随机成功 | McNemar p（Holm） | 最好违反量差值中位数 (dB) | Wilcoxon p（Holm） |", "|---|---|---|---|---|---|"]
for m, t in tests.items():
    L.append(f"| {LAB[m]} | {t['only_method_ok']} | {t['only_random_ok']} | {t['mcnemar_p']:.3g}（{t['mcnemar_p_holm']:.3g}） | "
             f"{t['median_violation_diff']:.2f} | {t['wilcoxon_p']:.3g}（{t['wilcoxon_p_holm']:.3g}） |")
L += ["", "## 选中的参数组", "", "| 方法 | 与 SHAP 相同的比例 | 与 ADS 真值前 2 相同的比例 | 参数组（次数） |", "|---|---|---|---|"]
for m in pairs:
    L.append(f"| {LAB.get(m, m)} | {overlap[m]['same_as_shap']:.2f} | {overlap[m]['same_as_ads_truth']:.2f} | "
             + "；".join(f"{k} ×{v}" for k, v in sorted(pair_counts[m].items(), key=lambda kv: -kv[1])) + " |")
L += ["", "## 逐例结果（✓ = ADS 确认通过；数字 = 轨迹中最好的 ADS 违反量 dB）", "",
      "| 设计 | 初始违反量 | " + " | ".join(LAB.get(m, m) for m in M) + " |", "|---" * (len(M) + 2) + "|"]
for r in R:
    L.append(f"| {r['design_id']} | {r['x0_hard']:.2f} | " + " | ".join(
        "✓" if r["methods"][m]["success"] else f"{r['methods'][m]['best_hard']:.2f}" for m in M) + " |")
(D.parent / f"step4_report{'' if split == 'test' else '_' + split}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:20]))
