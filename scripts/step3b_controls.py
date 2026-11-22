"""Step 3b: post-hoc robustness checks of the step-3 ADS-truth comparisons (no new simulation, no frozen
setting changed; added after the step-3 results and labelled as such).

a  naive baseline: train OLS slope (9 normalized inputs) x (x - ref) as an "explanation", vs ADS truth
b  minor features: agreement on the 4 geometry features other than L1, L2 (Spearman of |phi|, sign),
   plus top-1 hit and top-2 set match over all 6
c  95% bootstrap intervals over designs (2000 resamples) for the mean per-design agreement
Explainers use seed-averaged per-design agreement (3 seeds). Writes results/step3/controls.{json,md}.
Usage: python scripts/step3b_controls.py
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, targets, xai  # noqa: E402

S3 = data.ROOT / "results" / "step3"
W6 = xai.shapley_matrix(6)
CUR, CON = targets.CURVE_TARGETS, targets.CONCEPT_TARGETS
reg = json.loads((data.ROOT / "results" / "step1" / "final_models.json").read_text(encoding="utf-8"))
tz = np.load(S3 / "ads" / "truth.npz")
truth_c, truth_k = tz["phi_curve"], tz["phi_concept"]  # (80, 11, 6), (80, 6, 6)
z2 = np.load(data.ROOT / "results" / "step2" / "setup.npz")
x_ref = z2["x_ref"]
setup = json.loads((data.ROOT / "results" / "step2" / "setup.json").read_text(encoding="utf-8"))
vfz = np.load(data.ROOT / "results" / "step0" / "vf_concepts_train.npz")
STD = np.r_[[setup["target_check_train"]["train_std"][t] for t in CUR],
            np.nanstd(vfz["natural_frequency_GHz"][:, 1:4], 0), np.nanstd(vfz["Q"][:, 1:4], 0)]
MINOR = [2, 3, 4, 5]  # S1, S2, Wc1, Wc2
rng = np.random.default_rng(2026)
SPL = {"validation": slice(0, 40), "test": slice(40, 80), "all": slice(0, 80)}


def load_attr(tag):
    a = [dict(np.load(data.ROOT / "results" / d / "attr" / f"{tag}.npz")) for d in ("step2", "step3")]
    out = {k: np.concatenate([a[0][k][:40], a[1][k][:40]]) for k in ("shap_ref", "ig", "v_ref", "x")}
    lc = np.concatenate([a[0]["lime_coef"][0][:40], a[1]["lime_coef"][0][:40]])
    out["lime_delta"] = lc * (out["x"] - x_ref)[:, None, :]
    out["model6"] = np.einsum("ic,dct->dti", W6, out["v_ref"][:, :64, :])
    return out


# a. naive OLS baseline (train only)
tr = np.load(data.ROOT / "results" / "step2" / "setup.npz")
Xtr, Ytr = tr["x_train"], tr["y_train"]
fn_tr, q_tr = vfz["natural_frequency_GHz"][:, 1:4], vfz["Q"][:, 1:4]
Ytr_all = np.c_[Ytr, fn_tr, q_tr]
A = np.c_[np.ones(len(Xtr)), Xtr]
ok = np.isfinite(Ytr_all).all(1)
beta = np.linalg.lstsq(A[ok], Ytr_all[ok], rcond=None)[0][1:]  # (9, 17)
X80 = load_attr(reg["PRNet"]["runs"][0])["x"]
naive = (beta.T[None, :, :6] * (X80 - x_ref)[:, None, :6])  # (80, 17, 6)
truth = np.concatenate([truth_c, truth_k], 1)  # (80, 17, 6)


def agree(est, tru):
    """per (design, target): spearman |.| over 6, minor-4 spearman, sign on big ref features (all / minor), top1, top2."""
    big = np.abs(tru) >= 0.01 * STD[None, :tru.shape[1], None]
    with np.errstate(invalid="ignore"):
        sign_all = np.nanmean(np.where(big, np.sign(est) == np.sign(tru), np.nan), -1)
        sign_min = np.nanmean(np.where(big[..., MINOR], np.sign(est[..., MINOR]) == np.sign(tru[..., MINOR]), np.nan), -1)
    top = lambda a, k: np.sort(np.argsort(-np.abs(a), -1)[..., :k], -1)  # noqa: E731
    return {"rho6": xai.rank_corr(est, tru), "rho_minor4": xai.rank_corr(est[..., MINOR], tru[..., MINOR]),
            "sign_all": sign_all, "sign_minor": sign_min,
            "top1_hit": (top(est, 1) == top(tru, 1))[..., 0].astype(float),
            "top2_set": np.all(top(est, 2) == top(tru, 2), -1).astype(float)}


def boot(per_design):
    """per_design: (80,) values -> mean and 95% bootstrap CI for each split."""
    out = {}
    for sp, sl in SPL.items():
        v = per_design[sl]
        v = v[np.isfinite(v)]
        b = np.array([rng.choice(v, len(v)).mean() for _ in range(2000)])
        out[sp] = [float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
    return out


def group_mean(arr, names, g):
    idx = [names.index(t) for t in targets.GROUPS[g] if t in names]
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr[:, idx], 1)  # (80,)


rep = {"note": "post-hoc checks added after step 3; frozen settings unchanged", "models": {}}
for kind in ("PRNet", "M1", "M0"):
    names = CUR + (CON if kind == "PRNet" else [])
    T = len(names)
    runs = [load_attr(t) for t in reg[kind]["runs"]]
    ests = {"model6": [r["model6"] for r in runs], "shap_ref": [r["shap_ref"][..., :6] for r in runs],
            "ig": [r["ig"][..., :6] for r in runs], "lime_delta": [r["lime_delta"][..., :6] for r in runs]}
    res = {}
    for e, arrs in ests.items():
        per = [agree(a[:, :T], truth[:, :T]) for a in arrs]
        res[e] = {m: np.nanmean([p[m] for p in per], 0) for m in per[0]}  # seed-averaged (80, T)
    res["naive_OLS"] = agree(naive[:, :T], truth[:, :T])
    out = {}
    for e, mets in res.items():
        out[e] = {}
        for g in targets.GROUPS:
            if not any(t in names for t in targets.GROUPS[g]):
                continue
            out[e][g] = {m: boot(group_mean(v, names, g)) for m, v in mets.items()}
    rep["models"][kind] = out
(S3 / "controls.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

# markdown (all 80 designs; mean [95% CI])
c = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
L = ["# 第 3 步补充检查（事后添加，冻结设置未改）", "",
     "a 朴素基线 = train 线性回归斜率 × (x − 参考)；b 去掉 L1、L2 后其余 4 个参数上的一致度，及前 1 / 前 2 是否命中；"
     "c 对设计 bootstrap 2000 次的 95% 区间。80 个设计（验证集 + test），解释器为 3 个种子平均。", ""]
for m, lab in (("rho6", "6 个参数 |φ| Spearman"), ("rho_minor4", "次要 4 个参数 Spearman"), ("sign_minor", "次要参数符号一致率"),
               ("top1_hit", "前 1 命中率"), ("top2_set", "前 2 集合命中率")):
    L += [f"## {lab}", "", "| 模型 | 解释 | T0 | T1 | T1b | T2 | T3 |", "|---|---|---|---|---|---|---|"]
    for kind in ("PRNet", "M1", "M0"):
        for e in ("naive_OLS", "model6", "shap_ref", "ig", "lime_delta"):
            o = rep["models"][kind][e]
            L.append(f"| {kind} | {e} | " + " | ".join(c(o[g][m]["all"]) if g in o else "—" for g in ("T0", "T1", "T1b", "T2", "T3")) + " |")
    L.append("")
(S3 / "controls.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote", S3 / "controls.md")
