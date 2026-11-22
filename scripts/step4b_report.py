"""Step 4b report (exploratory, plan §13.1): feasible pairs, fixed-L1L2 and exact random baselines, pair ranks,
and PR-Net vs M0 as search surrogate. Descriptive only.

Reads results/step4b/{PRNet,M0}/{pairs.json, selections.json} and results/step4/test/results.json.
Writes results/step4b/step4b_report.{json,md}. Usage: python scripts/step4b_report.py
"""
from pathlib import Path
import json

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "step4b"
G = ["L1", "L2", "S1", "S2", "Wc1", "Wc2"]
s4 = json.loads((ROOT / "results" / "step4" / "test" / "results.json").read_text(encoding="utf-8"))
coord = [r["methods"]["ads_coordinate"]["success"] for r in s4]
SELS = ["shap_ref", "ig", "lime_delta", "grad_x_span", "lime_slope_x_span", "ads_true_shap", "random"]
LAB = {"shap_ref": "SHAP（单一参考）", "ig": "IG", "lime_delta": "LIME-delta", "grad_x_span": "梯度 × 跨度",
       "lime_slope_x_span": "LIME 斜率 × 跨度", "ads_true_shap": "ADS 真值 Shapley 前 2", "random": "随机 2 个（第 4 步的那一次抽样）",
       "fixed_L1L2": "固定 L1 + L2", "random_expected": "随机 2 个（精确期望 = 可行组比例）", "oracle_any_pair": "任一组可行（2 参数上限）",
       "all6": "6 个参数代理搜索", "revert": "SHAP 前 2 改回基线（不搜索）", "ads_coordinate": "ADS 坐标搜索（与模型无关）"}
key = lambda S: ",".join(G[j] for j in S)  # noqa: E731
rep = {"note": "exploratory, not pre-registered", "models": {}}
for kind in ("PRNet", "M0"):
    P = json.loads((D / kind / "pairs.json").read_text(encoding="utf-8"))
    S = json.loads((D / kind / "selections.json").read_text(encoding="utf-8"))
    n = len(P)
    pair_keys = [k for k in P[0]["subspaces"] if len(k.split(",")) == 2]
    feas = np.array([[P[i]["subspaces"][k]["success"] for k in pair_keys] for i in range(n)])  # (n, 15)
    best = np.array([[P[i]["subspaces"][k]["best_hard"] for k in pair_keys] for i in range(n)])
    rank = np.array([rankdata(b, method="average") for b in best])  # 1 = lowest best violation
    succ, ranks, ch = {}, {}, {}
    for m in SELS:
        j = [pair_keys.index(key(S[str(i)][m])) for i in range(n)]
        succ[m] = int(feas[np.arange(n), j].sum())
        ranks[m] = float(np.mean(rank[np.arange(n), j]))
        ch[m] = float(np.median([P[i]["subspaces"][pair_keys[j[i]]]["final_change"] for i in range(n) if feas[i, j[i]]] or [np.nan]))
    jl = pair_keys.index("L1,L2")
    succ["fixed_L1L2"], ranks["fixed_L1L2"] = int(feas[:, jl].sum()), float(rank[:, jl].mean())
    succ["random_expected"] = float(feas.mean(1).sum())
    succ["oracle_any_pair"] = int(feas.any(1).sum())
    all6 = [P[i]["subspaces"][",".join(G)] for i in range(n)]
    succ["all6"] = int(sum(a["success"] for a in all6))
    ch["all6"] = float(np.median([a["final_change"] for a in all6 if a["success"]] or [np.nan]))
    succ["revert"] = int(sum(P[i]["revert"]["hard"] <= 1e-9 for i in range(n)))
    succ["ads_coordinate"] = int(sum(coord))
    out = {"n": n, "success": succ, "mean_rank_of_chosen_pair_among_15": ranks, "change_median_success": ch,
           "feasible_pairs_per_case": feas.sum(1).tolist(),
           "most_often_feasible_pairs": dict(sorted(zip(pair_keys, map(int, feas.sum(0))), key=lambda kv: -kv[1])[:6]),
           "x0_surrogate_abs_error_median_dB": float(np.median([abs(P[i]["x0_surrogate"] - P[i]["x0_hard"]) for i in range(n)])),
           "all6_ads_calls_median": float(np.median([a["ads_calls"] for a in all6]))}
    if kind == "PRNet":  # reproduction of step 4 via lookup
        mism = 0
        for i, r in enumerate(s4):
            for m in ("shap_ref", "ig", "lime_delta", "random", "grad_x_span", "lime_slope_x_span", "ads_true_shap"):
                look = P[i]["subspaces"][",".join(r["methods"][m]["subspace"])]["success"]
                mism += look != r["methods"][m]["success"]
            mism += P[i]["subspaces"][",".join(G)]["success"] != r["methods"]["all6_surrogate"]["success"]
        out["step4_reproduction_mismatches"] = int(mism)
    out["selection_overlap_with_PRNet"] = None
    rep["models"][kind] = out
    rep["models"][kind]["_sel"] = S
sp, sm = rep["models"]["PRNet"].pop("_sel"), rep["models"]["M0"].pop("_sel")
rep["models"]["M0"]["selection_overlap_with_PRNet"] = {m: float(np.mean([sp[str(i)][m] == sm[str(i)][m] for i in range(len(sp))]))
                                                       for m in ("shap_ref", "ig", "lime_delta", "grad_x_span", "lime_slope_x_span")}
(D / "step4b_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

A, B = rep["models"]["PRNet"], rep["models"]["M0"]
f = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)  # noqa: E731
L = ["# 第 4 步探索性分析（事后添加，非预先登记）", "",
     "30 个 test 案例 × 15 组 2 参数 + 6 参数，每组都按预先登记的搜索和 ADS 确认运行。成功 = 10 次 ADS 调用内通过。", "",
     f"PR-Net 穷举对第 4 步结果的复现：不一致 {A.get('step4_reproduction_mismatches')} 处（共 30 × 8）。", "",
     "| 方法 | PR-Net 成功（/30） | M0 成功（/30） | 所选参数组在 15 组中的平均排名（PR-Net / M0，1 = 最好） | 成功设计的几何改动中位数（PR-Net / M0） |",
     "|---|---|---|---|---|"]
for m in SELS + ["fixed_L1L2", "random_expected", "oracle_any_pair", "all6", "revert", "ads_coordinate"]:
    rk = A["mean_rank_of_chosen_pair_among_15"]
    rkb = B["mean_rank_of_chosen_pair_among_15"]
    L.append(f"| {LAB[m]} | {f(A['success'][m])} | {f(B['success'][m])} | "
             + (f"{rk[m]:.1f} / {rkb[m]:.1f}" if m in rk else "—") + " | "
             + (f"{A['change_median_success'][m]:.2f} / {B['change_median_success'][m]:.2f}" if m in A["change_median_success"] else "—") + " |")
L += ["", f"每例可行参数组数（15 组中）：PR-Net 中位数 {np.median(A['feasible_pairs_per_case']):.0f}（范围 {min(A['feasible_pairs_per_case'])}–{max(A['feasible_pairs_per_case'])}），"
      f"M0 中位数 {np.median(B['feasible_pairs_per_case']):.0f}（范围 {min(B['feasible_pairs_per_case'])}–{max(B['feasible_pairs_per_case'])}）。", "",
      "最常可行的参数组（案例数）：PR-Net " + "；".join(f"{k} {v}" for k, v in A["most_often_feasible_pairs"].items())
      + "｜M0 " + "；".join(f"{k} {v}" for k, v in B["most_often_feasible_pairs"].items()), "",
      f"初始设计处代理的违反量误差（中位数）：PR-Net {A['x0_surrogate_abs_error_median_dB']:.2f} dB，M0 {B['x0_surrogate_abs_error_median_dB']:.2f} dB。"
      f"6 参数搜索的 ADS 调用次数中位数：PR-Net {A['all6_ads_calls_median']:.0f}，M0 {B['all6_ads_calls_median']:.0f}。", "",
      "M0 与 PR-Net 选中相同参数组的比例：" + "；".join(f"{LAB[m]} {v:.2f}" for m, v in B["selection_overlap_with_PRNet"].items())]
(D / "step4b_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
