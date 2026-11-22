"""Step 2 evaluation on validation: §5 A (faithfulness), paper protocol, and the free parts of B.

Reads results/step2/setup.{json,npz} and results/step2/attr/*.npz; writes results/step2/xai_report.{json,md}.
Explainers compared in target units: SHAP (single reference), IG, LIME-delta = LIME slope * (x - ref).
Rows 0-39 are the explained validation designs, row 40 the baseline (= reference).
Usage: python scripts/step2_eval.py [test]
  With "test" (step 3) the same evaluation reads results/step3/setup_test.* and results/step3/attr and
  writes results/step3/xai_report_test.{json,md}.
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, targets, xai  # noqa: E402

TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
S2 = data.ROOT / "results" / ("step3" if TEST else "step2")
SNAME, RNAME = ("setup_test", "xai_report_test") if TEST else ("setup", "xai_report")
reg = json.loads((data.ROOT / "results" / "step1" / "final_models.json").read_text(encoding="utf-8"))
setup = json.loads((S2 / f"{SNAME}.json").read_text(encoding="utf-8"))
z = np.load(S2 / f"{SNAME}.npz")
FEAT = list(setup["reference_normalized"])
GEO, CTRL = slice(0, 6), slice(6, 9)
KINDS = ("PRNet", "M1", "M0")
E = slice(0, 40)
vf = np.load(data.ROOT / "results" / "step0" / "vf_concepts_train.npz")
STD = dict(setup["target_check_train"]["train_std"])
STD.update(dict(zip(targets.CONCEPT_TARGETS, np.r_[np.nanstd(vf["natural_frequency_GHz"][:, 1:4], 0),
                                                   np.nanstd(vf["Q"][:, 1:4], 0)])))


def load(kind):
    runs = [dict(np.load(S2 / "attr" / f"{t}.npz")) for t in reg[kind]["runs"]]
    ref = reg[kind]["runs"][[f"_s{reg['reference_seed']}" in t for t in reg[kind]["runs"]].index(True)]
    paper = dict(np.load(S2 / "attr" / f"{ref}_paper.npz"))
    for r in runs:
        r["lime_delta"] = r["lime_coef"][0] * (r["x"] - z["x_ref"])[:, None, :]
    return runs, runs[[f"_s{reg['reference_seed']}" in t for t in reg[kind]["runs"]].index(True)], paper


def by_group(names, arr, fn=np.nanmedian):
    """arr (..., T) over designs -> {group: value} aggregated over designs and the group's targets."""
    out = {}
    for g, ts in targets.GROUPS.items():
        idx = [names.index(t) for t in ts if t in names]
        if idx:
            out[g] = float(fn(np.asarray(arr)[..., idx]))
    return out


def per_target(names, arr, fn=np.nanmedian):
    return {t: float(fn(np.asarray(arr)[..., i])) for i, t in enumerate(names)}


EXPL = ("shap_ref", "ig", "lime_delta")
rep = {"explainers": {"shap_ref": "exact SHAP, single ADS-baseline reference (512 coalitions)",
                      "ig": "integrated gradients from the same reference", "lime_delta": "own LIME slope x (x - ref)"},
       "designs": setup["explained"], "models": {}}
cache = {}
for kind in KINDS:
    runs, R, P = load(kind)
    cache[kind] = (runs, R, P)
    names = list(R["names"])
    gap_scale = np.array([STD[t] for t in names])
    out = {"tags": reg[kind]["runs"]}

    # A1 additivity / completeness / LIME fit
    out["A1"] = {
        "shap_ref_additivity_max_abs": float(np.max(np.abs(R["shap_ref"].sum(-1) - (R["F_x"] - R["F_ref"])))),
        "shap_lhs32_additivity_max_abs": float(np.max(np.abs(R["shap_lhs32"].sum(-1) - (R["F_x"] - R["base_lhs32"])))),
        "ig_resid_rel": {"median": float(np.median(R["ig_resid"][E])), "p95": float(np.quantile(R["ig_resid"][E], 0.95)),
                         "max": float(R["ig_resid"][E].max()), "frac_gt_1pct": float(np.mean(R["ig_resid"][E] > 0.01))},
        "ig_steps_counts": {str(k): int(v) for k, v in zip(*np.unique(R["ig_steps"][E], return_counts=True))},
        "lime_heldout_R2_median": by_group(names, R["lime_r2"][0][E]),
        "lime_heldout_R2_frac_below_0.5": by_group(names, R["lime_r2"][0][E] < 0.5, np.mean),
        "ig_frac_gt_1pct_per_target": per_target(names, R["ig_resid"][E] > 0.01, np.mean)}

    # A2 deletion / insertion on the single-reference game
    gap = R["F_x"] - R["F_ref"]
    ok = np.abs(gap[E]) >= 1e-3 * gap_scale
    out["A2"] = {"skipped_small_gap_frac": float(1 - ok.mean())}
    for e in EXPL:
        di = xai.deletion_insertion(R["v_ref"][E], R[e][E])
        for k in di:
            di[k][~ok] = np.nan
        out["A2"][e] = {"del_pct_median": by_group(names, di["del_pct"]), "ins_pct_median": by_group(names, di["ins_pct"]),
                        "del_top5pct_frac": by_group(names, np.where(ok, di["del_pct"] <= 0.05, np.nan), np.nanmean),
                        "ins_top5pct_frac": by_group(names, np.where(ok, di["ins_pct"] <= 0.05, np.nan), np.nanmean),
                        "del_auc_median": by_group(names, di["del_auc"]), "ins_auc_median": by_group(names, di["ins_auc"])}

    # A3 negative controls: attribution share vs measured intervention share
    share = lambda a: np.abs(a[..., CTRL]).sum(-1) / np.maximum(np.abs(a).sum(-1), 1e-30)  # noqa: E731
    out["A3"] = {e: by_group(names, share(R[e][E]), np.mean) for e in EXPL + ("shap_rand32",)}
    out["A3"]["intervention_rand32"] = by_group(names, share(R["interv_rand32"][E]), np.mean)

    # A4 stability (Spearman of |phi| over 9 features, mean over designs and targets of a group)
    st = {}
    for e in EXPL:
        st[f"seeds_{e}"] = by_group(names, np.nanmean([xai.rank_corr(a[e][E], b[e][E]) for a, b in itertools.combinations(runs, 2)], 0), np.nanmean)
    bgs = {"ref": R["shap_ref"], "lhs32": R["shap_lhs32"], "rand32": R["shap_rand32"], "paper100": P["shap_paper100"]}
    for a, b in itertools.combinations(bgs, 2):
        st[f"background_{a}_vs_{b}"] = by_group(names, xai.rank_corr(bgs[a][E], bgs[b][E]), np.nanmean)
    lc = R["lime_coef"]
    st["lime_seeds_scale0.25"] = by_group(names, np.nanmean([xai.rank_corr(lc[i][E], lc[j][E]) for i, j in itertools.combinations(range(5), 2)], 0), np.nanmean)
    st["lime_scale_0.1_vs_0.25"] = by_group(names, xai.rank_corr(lc[5][E], lc[0][E]), np.nanmean)
    st["lime_scale_0.5_vs_0.25"] = by_group(names, xai.rank_corr(lc[6][E], lc[0][E]), np.nanmean)
    st["shap_ref_vs_ig"] = by_group(names, xai.rank_corr(R["shap_ref"][E], R["ig"][E]), np.nanmean)
    st["shap_ref_vs_lime_delta"] = by_group(names, xai.rank_corr(R["shap_ref"][E], R["lime_delta"][E]), np.nanmean)
    out["A4"] = st

    # A5 GradientExplainer (100 backgrounds) vs exact SHAP on the same background
    gx, ex = P["gradexp"][E], P["shap_paper100"][E]
    out["A5"] = {"spearman_abs": by_group(names, xai.rank_corr(gx, ex), np.nanmean),
                 "rel_L1_error_median": by_group(names, np.abs(gx - ex).sum(-1) / np.maximum(np.abs(ex).sum(-1), 1e-30)),
                 "gradexp_additivity_rel_median": by_group(names, np.abs(gx.sum(-1) - (R["F_x"][E] - P["base_paper100"][E]))
                                                          / np.maximum(np.abs(R["F_x"][E] - P["base_paper100"][E]), 1e-30))}

    # paper protocol: SHAP (GradientExplainer) vs median LIME (lime package, 20 runs)
    lp = P["lime_paper"][E]  # (D, 20, T, 9)
    med = np.median(lp, 1)
    rho = xai.rank_corr(gx, med)
    top3 = np.argsort(-np.abs(med), -1)[..., :3]
    sel = np.take_along_axis(lp, top3[:, None], -1)  # (D, 20, T, 3)
    out["paper_protocol"] = {
        "rho_SHAP_LIME_median_over_designs": per_target(names, rho), "rho_by_group_mean": by_group(names, rho, np.nanmean),
        "rho_T1b_quantiles": [float(q) for q in np.nanquantile(rho[:, names.index("T1b_passband_mean_S21_dB")], [0.1, 0.5, 0.9])],
        "lime_top3_sign_constant_frac": by_group(names, np.all(np.sign(sel) == np.sign(sel[:, :1]), 1).mean(-1), np.mean),
        "lime_top3_rel_spread_median": by_group(names, (sel.std(1) / np.maximum(np.abs(sel.mean(1)), 1e-30)).mean(-1))}
    rep["models"][kind] = out

# explanations themselves: global mean |phi| (reference seed, single-reference SHAP) and top-3 per target
rep["global_importance"] = {}
for kind in KINDS:
    R = cache[kind][1]
    names = list(R["names"])
    imp = np.abs(R["shap_ref"][E]).mean(0)  # (T, 9)
    rep["global_importance"][kind] = {t: {"mean_abs_shap": dict(zip(FEAT, map(float, imp[i]))),
                                          "top3": [FEAT[j] for j in np.argsort(-imp[i])[:3]]} for i, t in enumerate(names)}
# cross-model agreement of single-reference SHAP (reference seeds, curve targets)
cur = targets.CURVE_TARGETS
rep["cross_model_rank_corr"] = {f"{a}_vs_{b}": by_group(cur, xai.rank_corr(cache[a][1]["shap_ref"][E][:, :11], cache[b][1]["shap_ref"][E][:, :11]), np.nanmean)
                                for a, b in itertools.combinations(KINDS, 2)}

# B free 1: baseline one-factor diagnostics (ADS) vs model finite differences and baseline attributions
yd, cd = z["y_diag"], z["concepts_diag"]
span = z["x_diag"][2::2] - z["x_diag"][1::2]  # (6, 9) high - low, normalized units
ads_fd = (yd[2::2] - yd[1::2]).T  # (T_curve, 6)
ads_fd_c = (cd[2::2] - cd[1::2]).T  # (6 concepts, 6)
rep["B_diagnostics"] = {"ADS_fd": {t: dict(zip(FEAT[:6], map(float, ads_fd[i]))) for i, t in enumerate(cur)},
                        "ADS_fd_concepts": {t: dict(zip(FEAT[:6], map(float, ads_fd_c[i]))) for i, t in enumerate(targets.CONCEPT_TARGETS)}}


def compare(ref_fd, est, names_t):
    """sign agreement (excluding |ADS| < 1% train std) and Spearman of |.| over the 6 parameters, per target."""
    sgn, rho = [], []
    for i, t in enumerate(names_t):
        big = np.abs(ref_fd[i]) >= 0.01 * STD[t]
        sgn.append(np.mean(np.sign(ref_fd[i][big]) == np.sign(est[i][big])) if big.any() else np.nan)
        rho.append(spearmanr(np.abs(ref_fd[i]), np.abs(est[i]))[0])
    return {"sign_agree": dict(zip(names_t, map(float, sgn))), "spearman_abs": dict(zip(names_t, map(float, rho))),
            "sign_agree_mean": float(np.nanmean(sgn)), "spearman_abs_mean": float(np.nanmean(rho))}


for kind in KINDS:
    runs, R, _ = cache[kind]
    names = list(R["names"])
    blk = {}
    for tag, r in zip(reg[kind]["runs"], runs):
        fd = (r["F_diag"][2::2] - r["F_diag"][1::2]).T  # (T, 6)
        ests = {"model_fd": fd, "grad_x_span": np.einsum("tj,kj->tk", r["grad_diag0"], span),
                "lime_slope_x_span": np.einsum("tj,kj->tk", r["lime_coef"][0][40], span),
                "shap_lhs32_at_baseline": r["shap_lhs32"][40][:, :6], "shap_rand32_at_baseline": r["shap_rand32"][40][:, :6]}
        blk[tag] = {k: compare(ads_fd, v[:11], cur) for k, v in ests.items()}
        if kind == "PRNet":
            blk[tag]["concepts"] = {k: compare(ads_fd_c, v[11:], targets.CONCEPT_TARGETS)
                                    for k, v in ests.items() if k in ("model_fd", "grad_x_span", "lime_slope_x_span")}
            blk[tag]["concepts"]["model_fd_values"] = {t: dict(zip(FEAT[:6], map(float, fd[11 + i]))) for i, t in enumerate(targets.CONCEPT_TARGETS)}
    rep["B_diagnostics"][kind] = blk

# B free 2: T0 ranking vs train OLS (standardized coefficients; LIME slopes and gradients share the unit)
ols = setup["target_check_train"]["T0_OLS_9_inputs"]["coef_per_train_std"]
ols_v = np.array([ols[f] for f in FEAT])
rep["B_T0_vs_OLS"] = {"OLS_rank": setup["target_check_train"]["T0_OLS_9_inputs"]["rank_by_abs"][:6]}
for kind in KINDS:
    R = cache[kind][1]
    ests = {"lime_slope_mean": R["lime_coef"][0][E][:, 0].mean(0), "grad_mean": R["grad"][E][:, 0].mean(0),
            "shap_ref_mean_abs": np.abs(R["shap_ref"][E][:, 0]).mean(0), "ig_mean_abs": np.abs(R["ig"][E][:, 0]).mean(0)}
    rep["B_T0_vs_OLS"][kind] = {k: {"top3": [FEAT[j] for j in np.argsort(-np.abs(v[GEO]))[:3]],
                                    "spearman_abs_6geom": float(spearmanr(np.abs(v[GEO]), np.abs(ols_v[GEO]))[0]),
                                    **({"rel_err_vs_OLS": dict(zip(FEAT[:6], map(float, v[GEO] / ols_v[GEO] - 1)))} if "slope" in k or "grad" in k else {})}
                                for k, v in ests.items()}

# concept-group Shapley (PR-Net reference seed)
P = cache["PRNet"][2]
cg, groups = P["cg_shap"][E], list(P["cg_groups"])
R = cache["PRNet"][1]
sh = np.abs(cg) / np.maximum(np.abs(cg).sum(-1, keepdims=True), 1e-30)
rep["concept_group_shapley"] = {
    "groups": groups, "additivity_max_abs": float(np.max(np.abs(P["cg_shap"].sum(-1) - (P["cg_vN"] - P["cg_v0"])))),
    "vN_matches_model_max_abs": float(np.max(np.abs(P["cg_vN"] - R["F_x"][:, :11]))),
    "mean_abs_share": {t: dict(zip(groups, map(float, sh[:, i].mean(0)))) for i, t in enumerate(cur)}}

(S2 / f"{RNAME}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

# markdown
M = rep["models"]
G4 = ("T0", "T1", "T1b", "T2")
n2 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"  # noqa: E731
n3 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"  # noqa: E731
grp = lambda d, gs=G4: " / ".join(n2(d.get(g)) for g in gs)  # noqa: E731
L = [f"# {'第 3 步：test 上的 XAI（冻结后，与第 2 步相同的流程）' if TEST else '第 2 步：验证集上的 XAI'}", "",
     f"被解释设计：{'test' if TEST else '验证集'} 40 个单通带设计（{json.dumps(setup['explained']['strata'], ensure_ascii=False)}）+ 基线；"
     "参考点 = ADS 基线几何，负对照取 train 均值。模型：各 3 个种子，参考种子 2026。表中四个数依次为目标组 T0 / T1 / T1b / T2（PR-Net 另列 T3）。", "",
     "## A1 内部检查", "", "| 模型 | SHAP 加和误差（最大） | IG 残差 > 1% 的比例 | LIME 留出 R² 中位数 | LIME R² < 0.5 的比例 |", "|---|---|---|---|---|"]
for k in KINDS:
    a = M[k]["A1"]
    L.append(f"| {k} | {a['shap_ref_additivity_max_abs']:.1e} | {a['ig_resid_rel']['frac_gt_1pct']:.3f}（"
             + "、".join(f"{t} {v:.2f}" for t, v in a["ig_frac_gt_1pct_per_target"].items() if v > 0.1) + f"） | {grp(a['lime_heldout_R2_median'])} | {grp(a['lime_heldout_R2_frac_below_0.5'])} |")
L += ["", "## A2 删除 / 插入（单一参考博弈，相对 100 个随机顺序；落在随机顺序前 5% 的比例）", "",
      "| 模型 | 解释器 | 删除 | 插入 |", "|---|---|---|---|"]
for k in KINDS:
    for e in EXPL:
        L.append(f"| {k} | {e} | {grp(M[k]['A2'][e]['del_top5pct_frac'])} | {grp(M[k]['A2'][e]['ins_top5pct_frac'])} |")
L += ["", "## A3 负对照：归因占比 vs 模型对负对照的实际干预敏感度", "",
      "| 模型 | SHAP | IG | LIME-delta | SHAP（rand32 背景） | 实际干预占比 |", "|---|---|---|---|---|---|"]
for k in KINDS:
    a = M[k]["A3"]
    L.append(f"| {k} | " + " | ".join(grp(a[e]) for e in EXPL + ("shap_rand32", "intervention_rand32")) + " |")
L += ["", "## A4 稳定性（9 个特征 |归因| 的 Spearman，设计和目标上平均）", "", "| 比较 | PR-Net | M1 | M0 |", "|---|---|---|---|"]
for key, lab in [("seeds_shap_ref", "种子间：SHAP"), ("seeds_ig", "种子间：IG"), ("seeds_lime_delta", "种子间：LIME"),
                 ("background_ref_vs_lhs32", "背景：单一参考 vs lhs32"), ("background_ref_vs_paper100", "背景：单一参考 vs paper100"),
                 ("background_lhs32_vs_rand32", "背景：lhs32 vs rand32"), ("lime_seeds_scale0.25", "LIME 采样种子"),
                 ("lime_scale_0.5_vs_0.25", "LIME 邻域 0.5 vs 0.25"), ("shap_ref_vs_ig", "SHAP vs IG"), ("shap_ref_vs_lime_delta", "SHAP vs LIME-delta")]:
    L.append(f"| {lab} | " + " | ".join(grp(M[k]["A4"][key]) for k in KINDS) + " |")
L += ["", "## A5 GradientExplainer（100 背景，nsamples 200）vs 同一背景的精确 SHAP", "",
      "| 模型 | Spearman | 相对 L1 误差中位数 | 加和相对误差中位数 |", "|---|---|---|---|"]
for k in KINDS:
    a = M[k]["A5"]
    L.append(f"| {k} | {grp(a['spearman_abs'])} | {grp(a['rel_L1_error_median'])} | {grp(a['gradexp_additivity_rel_median'])} |")
L += ["", "## 论文协议（GradientExplainer vs lime 包 20 次的中位数；论文正向任务 ρ = 0.86）", "",
      "| 模型 | ρ 均值 | T1b ρ 的 10% / 50% / 90% 分位 | LIME top-3 符号恒定比例 | LIME top-3 相对离散度 |", "|---|---|---|---|---|"]
for k in KINDS:
    a = M[k]["paper_protocol"]
    L.append(f"| {k} | {grp(a['rho_by_group_mean'])} | {' / '.join(n2(q) for q in a['rho_T1b_quantiles'])} | "
             f"{grp(a['lime_top3_sign_constant_frac'])} | {grp(a['lime_top3_rel_spread_median'])} |")
L += ["", "## B 物理一致性（无需新仿真）", "", "### 基线处 12 个单参数 ADS 场景（±5% 声明边界）", "",
      "符号一致率 / |·| 的 Spearman（6 个参数，11 个曲线目标平均，3 个种子平均）。", "",
      "| 模型 | 模型有限差分 | 梯度 × 跨度 | LIME 斜率 × 跨度 | SHAP lhs32（基线处） |", "|---|---|---|---|---|"]
for k in KINDS:
    b = rep["B_diagnostics"][k]
    cell = lambda e: f"{np.mean([v[e]['sign_agree_mean'] for v in b.values()]):.2f} / {np.mean([v[e]['spearman_abs_mean'] for v in b.values()]):.2f}"  # noqa: E731
    L.append(f"| {k} | {cell('model_fd')} | {cell('grad_x_span')} | {cell('lime_slope_x_span')} | {cell('shap_lhs32_at_baseline')} |")
bc = [v["concepts"] for v in rep["B_diagnostics"]["PRNet"].values()]
L += ["", "PR-Net 概念（T3，ADS 场景曲线逐条 vector fitting 作真值）：模型有限差分 "
      f"{np.mean([c['model_fd']['sign_agree_mean'] for c in bc]):.2f} / {np.mean([c['model_fd']['spearman_abs_mean'] for c in bc]):.2f}；"
      f"梯度 × 跨度 {np.mean([c['grad_x_span']['sign_agree_mean'] for c in bc]):.2f} / {np.mean([c['grad_x_span']['spearman_abs_mean'] for c in bc]):.2f}。", "",
      "### T0 已知答案 vs train OLS（排序 " + " > ".join(rep["B_T0_vs_OLS"]["OLS_rank"][:3]) + "）", "",
      "| 模型 | LIME 斜率 | 梯度 | SHAP | IG |", "|---|---|---|---|---|"]
for k in KINDS:
    b = rep["B_T0_vs_OLS"][k]
    L.append(f"| {k} | " + " | ".join(f"{'>'.join(b[e]['top3'])}（ρ {b[e]['spearman_abs_6geom']:.2f}）"
                                       for e in ("lime_slope_mean", "grad_mean", "shap_ref_mean_abs", "ig_mean_abs")) + " |")
L += ["", "## 解释结果", "", "### 全局重要性（参考种子，单一参考 SHAP 的 mean |φ| 前三）", "", "| 目标 | PR-Net | M1 | M0 |", "|---|---|---|---|"]
for t in rep["global_importance"]["PRNet"]:
    L.append(f"| {t} | " + " | ".join(">".join(rep["global_importance"][k][t]["top3"]) if t in rep["global_importance"][k] else "—" for k in KINDS) + " |")
L += ["", "跨模型一致性（SHAP 排序 Spearman）：" + "；".join(f"{a} {grp(v)}" for a, v in rep["cross_model_rank_corr"].items()), "",
      "### PR-Net 概念分组 Shapley（|φ| 占比，40 个设计平均）", "", "| 目标 | " + " | ".join(rep["concept_group_shapley"]["groups"]) + " |",
      "|---" * (len(rep["concept_group_shapley"]["groups"]) + 1) + "|"]
for t, v in rep["concept_group_shapley"]["mean_abs_share"].items():
    L.append(f"| {t} | " + " | ".join(n2(x) for x in v.values()) + " |")
(S2 / f"{RNAME}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote", S2 / f"{RNAME}.json", "and .md")
