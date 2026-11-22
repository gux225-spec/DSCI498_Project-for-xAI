"""Step 3 report: test accuracy (vs validation), ADS true-Shapley decomposition, finite differences, paper criterion.

Reads results/step3/test/*/metrics.json, results/step1/runs/*/metrics.json, results/step3/ads_eval.json.
Writes results/step3/step3_report.md (test-set XAI internals are in results/step3/xai_report_test.md).
Usage: python scripts/step3_report.py
"""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S3 = ROOT / "results" / "step3"
reg = json.loads((ROOT / "results" / "step1" / "final_models.json").read_text(encoding="utf-8"))
ev = json.loads((S3 / "ads_eval.json").read_text(encoding="utf-8"))
GOALS = {"S21_full_MAE_dB": 0.5, "S21_passband_MAE_dB": 0.2, "S11_full_MAE_dB": 2.0, "S11_MAE_true_ge_m30dB": 1.0,
         "edge_MAE_MHz": 10.0, "edge_P95_MHz": 30.0}
rd = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
G4 = ("T0", "T1", "T1b", "T2")
n2 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"  # noqa: E731

acc = {}
for kind in ("PRNet", "M1", "M0"):
    te = [rd(S3 / "test" / t / "metrics.json") for t in reg[kind]["runs"]]
    va = [rd(ROOT / "results" / "step1" / "runs" / t / "metrics.json")["validation"]["all"] for t in reg[kind]["runs"]]
    acc[kind] = {"test": [m["test"]["all"] for m in te], "val": va, "paper_test": [m["paper_measures_test"] for m in te],
                 "paper_val": [m["paper_measures_validation"] for m in te]}
b1t = rd(S3 / "test" / "B1_r32" / "metrics.json")
acc["B1"] = {"test": [b1t["test"]["all"]], "val": [rd(ROOT / "results" / "step1" / "runs" / "B1_r32" / "metrics.json")["validation"]["all"]]}

L = ["# 第 3 步：冻结后的 test 结果", "",
     "冻结记录：`results/step3/freeze.json`（30 个文件的 SHA256 和第 3 步协议，写于读取任何 test 标签之前；"
     "`data.load` 在冻结记录存在前拒绝返回 test 标签）。", "",
     "## test 精度（216 个设计；3 个种子均值，括号内为验证集）", "", "| 指标 | 目标 | PR-Net | M1 | M0 | B1 |", "|---|---|---|---|---|---|"]
for k, g in list(GOALS.items()) + [("composite_score", None)]:
    L.append(f"| {k} | {g or ''} | " + " | ".join(
        f"{np.mean([m[k] for m in acc[n]['test']]):.3f}（{np.mean([m[k] for m in acc[n]['val']]):.3f}）" for n in ("PRNet", "M1", "M0", "B1")) + " |")
L.append("| 每个种子都过门槛（test） | | " + " | ".join(
    "是" if all(all(m[k] <= g for k, g in GOALS.items()) for m in acc[n]["test"]) else "否" for n in ("PRNet", "M1", "M0", "B1")) + " |")
L.append("| 最大 \\|P−1\\|（test） | 诊断 | " + " | ".join(
    f"{np.mean([m['power_balance_max_abs'] for m in acc[n]['test']]):.3f}" if "power_balance_max_abs" in acc[n]["test"][0] else "—"
    for n in ("PRNet", "M1", "M0", "B1")) + " |")
L += ["", f"B1 重拟合复现验证集预测：max |ΔS| = {b1t['validation_reproduction_max_abs_dS']:.1e}。", "",
      "PR-Net 的 test 带边沿 MAE（3.2 MHz）大于 P95（4.1 MHz），来自单个设计 v2_test_0034：其真实通带内凹陷只比 −3 dB 阈值高 0.18 dB，"
      "PR-Net 预测的凹陷略低于阈值，通带被判为断开，某一侧边沿跳变 0.5–1.1 GHz。这是带边沿定义在阈值处不连续造成的，不是曲线整体错误；"
      "去掉这个设计，PR-Net 的 test 带边沿 MAE 为 1.3–1.7 MHz。", "",
      "**与论文同口径的精度**（单通带设计；中心频率 = −3 dB 边沿中点；论文：S21 MAE 1.11 dB、中心频率误差 18 MHz、带宽 R² = −1.28）", "",
      "| 模型 | S21 全曲线 MAE (dB) | 中心频率 MAE (MHz) | 带宽 MAE (MHz) | 带宽 R² |", "|---|---|---|---|---|"]
for n in ("PRNet", "M1", "M0"):
    p = acc[n]["paper_test"]
    L.append(f"| {n} | {np.mean([m['S21_full_MAE_dB'] for m in acc[n]['test']]):.3f} | {np.mean([q['center_frequency_MAE_MHz'] for q in p]):.2f} | "
             f"{np.mean([q['bandwidth_MAE_MHz'] for q in p]):.2f} | {np.mean([q['bandwidth_R2'] for q in p]):.3f} |")

r = ev["reproduction"]
v = ev["vector_fitting"]
L += ["", "## ADS 真值 Shapley", "",
      f"80 个被解释设计（验证集 40 + test 40）× 64 个联盟 + 局部有限差分。完整联盟与数据集的最大 |ΔS| = {r['full_coalition_vs_dataset_max_abs_dS']:.1e}，"
      f"空联盟与基线 {r['empty_vs_baseline_max_abs_dS']:.1e}。联盟响应 vector fitting：{v['unique_responses']} 条，拟合误差中位数 {v['median_fit_error']:.1e}"
      f"（最大 {v['max_abs_fit_error']:.1e}），{v['frac_with_3_in_band_resonators']:.3f} 有 3 个带内谐振器，"
      f"{v['designs_with_complete_concept_game']}/80 个设计的概念博弈完整。", "",
      "**物理真值的全局排序（ADS，mean |φ| 前三）**", "", "| 目标 | 验证集 | test |", "|---|---|---|"]
for t in ev["ADS_global_importance"]["validation"]:
    L.append(f"| {t} | {'>'.join(ev['ADS_global_importance']['validation'][t])} | {'>'.join(ev['ADS_global_importance']['test'][t])} |")

L += ["", "## 分解：解释 → 网络 → 物理（6 个几何特征的同一单一参考博弈；3 个种子平均）", "",
      "|φ| 的 Spearman；四个数依次为 T0 / T1 / T1b / T2（PR-Net 另附 T3）。"]
for sp in ("validation", "test"):
    L += ["", f"### {sp}", "", "| 模型 | 比较 | Spearman | 符号一致率 | MAE / train std |", "|---|---|---|---|---|"]
    for kind in ("PRNet", "M1", "M0"):
        d = ev["decomposition"][kind][sp]
        gs = G4 + (("T3",) if kind == "PRNet" else ())
        for key, lab in [("shap_ref_vs_model6", "SHAP vs 网络"), ("ig_vs_model6", "IG vs 网络"), ("lime_delta_vs_model6", "LIME-delta vs 网络"),
                         ("model6_vs_ADS", "**网络 vs ADS**"), ("shap_ref_vs_ADS", "SHAP vs ADS"), ("ig_vs_ADS", "IG vs ADS"),
                         ("lime_delta_vs_ADS", "LIME-delta vs ADS")]:
            L.append(f"| {kind} | {lab} | " + " | ".join(" / ".join(n2(d[key][m].get(g)) for g in gs) for m in ("spearman_abs", "sign_agree", "mae_over_std")) + " |")

L += ["", "## 局部有限差分（h = 2% 声明跨度；11 个曲线目标；3 个种子平均）", "",
      "| 模型 | 集合 | 梯度 vs ADS 有限差分：Spearman / 符号 | LIME 斜率 vs ADS 有限差分：Spearman / 符号 |", "|---|---|---|---|"]
for kind in ("PRNet", "M1", "M0"):
    for sp in ("validation", "test"):
        d = ev["finite_difference"][kind][sp]
        cell = lambda key: " / ".join(f"{np.nanmean([d[key][m][g] for g in G4]):.2f}" for m in ("spearman_abs", "sign_agree"))  # noqa: E731
        L.append(f"| {kind} | {sp} | {cell('grad_vs_ADS_fd')} | {cell('lime_slope_vs_ADS_fd')} |")

L += ["", "## 检验论文的判据：SHAP–LIME 一致能否预测与物理真值一致", "",
      "每个（设计, 目标）一对：判据 = SHAP–LIME 排序 ρ；真值一致 = 该 SHAP 与 ADS 真值 Shapley 的排序 ρ。验证集 + test 共 80 个设计，参考种子。", "",
      "注意：论文协议的 GradientExplainer 以 100 个 train 设计为背景，而 ADS 真值是单一参考博弈，两者回答的问题不完全相同，"
      "这是论文协议下真值一致只有 0.66–0.74 的原因之一（与第 2 步中单一参考 vs 分布背景只有 0.70–0.86 一致）。"
      "\"同一博弈\" 一行用单一参考 SHAP 与 LIME-delta 作判据，与 ADS 真值是同一个博弈，不受这一影响。", "",
      "| 模型 | 协议 | 判据均值 | 真值一致均值 | 判据 vs 真值一致 Spearman | 判据高于中位数时的真值一致 | 低于中位数时 | 判据 ≥ 0.8 时的真值一致 |", "|---|---|---|---|---|---|---|---|"]
for kind in ("PRNet", "M1", "M0", "all"):
    for lab, nm in (("paper_protocol", "论文协议"), ("matched_game", "同一博弈")):
        c = ev["paper_criterion"][kind][lab]
        L.append(f"| {kind} | {nm} | {c['mean_criterion']:.2f} | {c['mean_truth_agreement']:.2f} | {c['spearman_criterion_vs_truth']:.2f} | "
                 f"{c['truth_agreement_above_median_criterion']:.2f} | {c['truth_agreement_below_median_criterion']:.2f} | {n2(c['truth_agreement_when_criterion_ge_0.8'])} |")
(S3 / "step3_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote", S3 / "step3_report.md")
