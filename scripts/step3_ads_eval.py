"""Step 3: ADS true Shapley and the comparisons fixed in results/step3/freeze.json.

Inputs: results/step3/ads/{map.json, cases/*.npz}, explanations in results/step2/attr (validation) and
results/step3/attr (test). Per design (40 validation + 40 test):
  ADS game     v_ADS(c) = target of the ADS response of coalition c (6 geometry features, absent = baseline);
               curve targets via filterxai.targets; T3 via vector fitting (N=10) of each coalition response
  model game   the surrogate's 6-feature game = coalitions 0..63 of its 512-coalition table (controls absent)
  explainers   SHAP (single reference, geometry part), IG, LIME-delta; GradientExplainer for the paper test
Comparisons (per design and target): Spearman of |phi| over the 6 features, sign agreement where
|phi_ADS| >= 1% of the target's train std, MAE of phi / train std. Finite differences: ADS gradient
(h = 2% span) vs model gradient and LIME slope. Paper criterion: does SHAP-LIME agreement predict
agreement with ADS truth (paper protocol and matched game)?
Writes results/step3/ads_eval.{json,md} and results/step3/ads/truth.npz.
Usage: python scripts/step3_ads_eval.py
"""
from pathlib import Path
import itertools
import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from filterxai import data, targets, vectfit, xai  # noqa: E402

S3 = data.ROOT / "results" / "step3"
CASES = S3 / "ads" / "cases"
F = np.linspace(9.0, 15.0, 601)
W6 = xai.shapley_matrix(6)
CUR, CON = targets.CURVE_TARGETS, targets.CONCEPT_TARGETS
FEAT = data.GEOMETRY


def load_S(jid):
    s = np.load(CASES / f"{jid}.npz")["S_complex"]
    return np.stack([s[:, 1, 0], s[:, 0, 0]], -1)  # (601, 2) = [S21, S11]


def vf_concepts(jid):
    S = load_S(jid)
    p, r, c, Hf = vectfit.vector_fit(F, S, 10)
    up = p[p.imag > 0]
    fn, q = vectfit.natural_frequency_q(up)
    o = np.argsort(fn)
    fn, q = fn[o], q[o]
    ok = len(fn) == 5
    err = float(np.abs(Hf - S).max())
    if not ok:
        return np.full(6, np.nan), err, int(np.sum((fn >= 9) & (fn <= 15) & (q >= 5)))
    return np.r_[fn[1:4], q[1:4]], err, int(np.sum((fn >= 9) & (fn <= 15) & (q >= 5)))


def main():
    import torch
    mp_ = json.loads((S3 / "ads" / "map.json").read_text(encoding="utf-8"))["designs"]
    T = targets.CurveTargets(F)
    setup = json.loads((data.ROOT / "results" / "step2" / "setup.json").read_text(encoding="utf-8"))
    vfz = np.load(data.ROOT / "results" / "step0" / "vf_concepts_train.npz")
    STD = dict(setup["target_check_train"]["train_std"])
    STD.update(dict(zip(CON, np.r_[np.nanstd(vfz["natural_frequency_GHz"][:, 1:4], 0), np.nanstd(vfz["Q"][:, 1:4], 0)])))
    std_c, std_k = np.array([STD[t] for t in CUR]), np.array([STD[t] for t in CON])

    # reproduction: full coalition vs dataset, empty vs baseline
    ds = {}
    for split in ("validation", "test"):
        d = data.load(split, allow_test_labels=split == "test")
        ds.update({str(i): s for i, s in zip(d["design_id"], d["S"])})
    base = np.load(data.RELEASE / "data" / "diagnostic_scenarios.npz")["S_complex"][0, :601]
    repro_full = max(float(np.abs(load_S(m["coalitions"][63]) - ds[m["design_id"]]).max()) for m in mp_)
    repro_empty = float(np.abs(load_S(mp_[0]["coalitions"][0]) - np.stack([base[:, 1, 0], base[:, 0, 0]], -1)).max())

    # ADS games
    def curve_t(jids):
        S = np.stack([load_S(j) for j in jids])
        db = 10 * np.log10(np.maximum(np.abs(S) ** 2, 1e-30))
        with torch.no_grad():
            return T(torch.tensor(db, dtype=torch.float32)).numpy()
    v_c = np.stack([curve_t(m["coalitions"]) for m in mp_])  # (80, 64, 11)
    uniq = sorted({j for m in mp_ for j in m["coalitions"]})
    res = {j: vf_concepts(j) for j in uniq}  # ~0.03 s per fit; a process pool exhausted Windows commit memory
    v_k = np.stack([[res[j][0] for j in m["coalitions"]] for m in mp_])  # (80, 64, 6)
    vf_err = np.array([res[j][1] for j in uniq])
    n_res = np.array([res[j][2] for j in uniq])
    phi_c = np.einsum("ic,dct->dti", W6, v_c)  # (80, 11, 6)
    phi_k = np.einsum("ic,dct->dti", W6, v_k)  # nan where any coalition fit lacked 5 pairs
    # ADS finite differences (per normalized unit, to compare with model gradients)
    sc = data.scalers()
    idx = [sc["candidate_order"].index(n) for n in FEAT]
    sd = np.array(sc["candidates"]["effective_std"])[idx]
    fd = np.zeros((len(mp_), len(CUR), 6))
    fd_step = np.zeros((len(mp_), 6))
    for di, m in enumerate(mp_):
        f0 = v_c[di, 63]
        for k, e in enumerate(m["fd"]):
            p = curve_t([e["plus"]])[0] if e["plus"] else None
            q = curve_t([e["minus"]])[0] if e["minus"] else None
            g = (p - q) / (2 * e["h"]) if p is not None and q is not None else ((p - f0) / e["h"] if p is not None else (f0 - q) / e["h"])
            fd[di, :, k] = g * sd[k]
            fd_step[di, k] = e["h"] / sd[k]
    np.savez_compressed(S3 / "ads" / "truth.npz", design_id=[m["design_id"] for m in mp_], split=[m["split"] for m in mp_],
                        v_curve=v_c, v_concept=v_k, phi_curve=phi_c, phi_concept=phi_k, fd_curve=fd)

    reg = json.loads((data.ROOT / "results" / "step1" / "final_models.json").read_text(encoding="utf-8"))
    ref_seed = reg["reference_seed"]
    SPL = {"validation": slice(0, 40), "test": slice(40, 80)}

    def attr(kind, tag):
        a = [dict(np.load(data.ROOT / "results" / d / "attr" / f"{tag}.npz")) for d in ("step2", "step3")]
        out = {k: np.concatenate([a[0][k][:40], a[1][k][:40]]) for k in ("shap_ref", "ig", "grad", "v_ref", "x")}
        out["lime_coef"] = np.concatenate([a[0]["lime_coef"][0][:40], a[1]["lime_coef"][0][:40]])
        out["lime_delta"] = out["lime_coef"] * (out["x"] - np.load(data.ROOT / "results" / "step2" / "setup.npz")["x_ref"])[:, None, :]
        if f"_s{ref_seed}" in tag:
            p = [dict(np.load(data.ROOT / "results" / d / "attr" / f"{tag}_paper.npz")) for d in ("step2", "step3")]
            out["gradexp"] = np.concatenate([p[0]["gradexp"][:40], p[1]["gradexp"][:40]])
            out["lime_paper_med"] = np.concatenate([np.median(p[0]["lime_paper"][:40], 1), np.median(p[1]["lime_paper"][:40], 1)])
        out["names"] = list(a[0]["names"])
        return out

    def cmp(a, b, ref_for_sign, std):
        """a, b: (D, T, 6). Returns per (D, T): spearman |.|, sign agreement on big ref features, MAE/std."""
        rho = xai.rank_corr(a, b)
        big = np.abs(ref_for_sign) >= 0.01 * std[None, :, None]
        agree = np.where(big, np.sign(a) == np.sign(b), np.nan)
        with np.errstate(invalid="ignore"):
            sign = np.nanmean(agree, -1)  # nan where no feature is large in the reference
        mae = np.abs(a - b).mean(-1) / std[None, :]
        return rho, sign, mae

    def grp(names, arr, sl):
        names = names[:arr.shape[1]]  # finite-difference comparisons cover the curve targets only
        return {g: float(np.nanmean(arr[sl][:, [names.index(t) for t in ts if t in names]]))
                for g, ts in targets.GROUPS.items() if any(t in names for t in ts)}

    rep = {"reproduction": {"full_coalition_vs_dataset_max_abs_dS": repro_full, "empty_vs_baseline_max_abs_dS": repro_empty},
           "vector_fitting": {"unique_responses": len(uniq), "max_abs_fit_error": float(np.nanmax(vf_err)),
                              "median_fit_error": float(np.median(vf_err)),
                              "frac_with_3_in_band_resonators": float(np.mean(n_res >= 3)),
                              "designs_with_complete_concept_game": int(np.isfinite(phi_k).all((1, 2)).sum())},
           "ADS_global_importance": {}, "decomposition": {}, "finite_difference": {}, "paper_criterion": {}}
    for sp, sl in SPL.items():
        imp = np.abs(phi_c[sl]).mean(0)
        rep["ADS_global_importance"][sp] = {t: [FEAT[j] for j in np.argsort(-imp[i])[:3]] for i, t in enumerate(CUR)}
        impk = np.nanmean(np.abs(phi_k[sl]), 0)
        rep["ADS_global_importance"][sp].update({t: [FEAT[j] for j in np.argsort(-impk[i])[:3]] for i, t in enumerate(CON)})

    crit_rows = []
    for kind in ("PRNet", "M1", "M0"):
        dec, fdr = {}, {}
        per_seed = []
        for tag in reg[kind]["runs"]:
            A = attr(kind, tag)
            names = A["names"]
            v6 = A["v_ref"][:, :64, :]  # controls absent
            m6 = np.einsum("ic,dct->dti", W6, v6)  # (80, T, 6)
            truth = np.concatenate([phi_c, phi_k], 1)[:, :len(names)] if kind == "PRNet" else phi_c
            std = np.r_[std_c, std_k][:len(names)]
            r = {}
            for e in ("shap_ref", "ig", "lime_delta"):
                ex = A[e][..., :6]
                r[f"{e}_vs_model6"] = cmp(ex, m6, m6, std)
                r[f"{e}_vs_ADS"] = cmp(ex, truth, truth, std)
            r["model6_vs_ADS"] = cmp(m6, truth, truth, std)
            g = A["grad"][:, :11, :6]
            r["grad_vs_ADS_fd"] = cmp(g, fd, fd * fd_step[:, None, :], std_c)
            r["lime_slope_vs_ADS_fd"] = cmp(A["lime_coef"][:, :11, :6], fd, fd * fd_step[:, None, :], std_c)
            per_seed.append((names, r))
            if f"_s{ref_seed}" in tag:
                # paper criterion: (a) paper protocol, (b) matched single-reference game
                rho_paper = xai.rank_corr(A["gradexp"], A["lime_paper_med"])  # 9 features, as in the paper
                rho_truth_a = xai.rank_corr(A["gradexp"][..., :6], truth)
                rho_match = xai.rank_corr(A["shap_ref"][..., :6], A["lime_delta"][..., :6])
                rho_truth_b = xai.rank_corr(A["shap_ref"][..., :6], truth)
                for di in range(80):
                    for ti, t in enumerate(names):
                        crit_rows.append((kind, di, t, rho_paper[di, ti], rho_truth_a[di, ti], rho_match[di, ti], rho_truth_b[di, ti]))
        for sp, sl in SPL.items():
            dec[sp] = {}
            for key in per_seed[0][1]:
                vals = [[grp(n, r[key][i], sl) for n, r in per_seed] for i in range(3)]
                dec[sp][key] = {lab: {g: float(np.mean([v[g] for v in vals[i]])) for g in vals[i][0]}
                                for i, lab in enumerate(("spearman_abs", "sign_agree", "mae_over_std"))}
        rep["decomposition"][kind] = {sp: {k: v for k, v in d.items() if "fd" not in k} for sp, d in dec.items()}
        rep["finite_difference"][kind] = {sp: {k: v for k, v in d.items() if "fd" in k} for sp, d in dec.items()}

    # paper criterion statistics
    rows = np.array([r[3:] for r in crit_rows], float)
    kinds = np.array([r[0] for r in crit_rows])
    tnames = np.array([r[2] for r in crit_rows])
    for kind in ("PRNet", "M1", "M0", "all"):
        m = (kinds == kind) if kind != "all" else np.ones(len(kinds), bool)
        out = {}
        for lab, (pi, ti) in {"paper_protocol": (0, 1), "matched_game": (2, 3)}.items():
            x, y = rows[m, pi], rows[m, ti]
            ok = np.isfinite(x) & np.isfinite(y)
            hi = x[ok] >= np.median(x[ok])
            out[lab] = {"n_pairs": int(ok.sum()), "spearman_criterion_vs_truth": float(spearmanr(x[ok], y[ok])[0]),
                        "mean_criterion": float(x[ok].mean()), "mean_truth_agreement": float(y[ok].mean()),
                        "truth_agreement_above_median_criterion": float(y[ok][hi].mean()),
                        "truth_agreement_below_median_criterion": float(y[ok][~hi].mean()),
                        "frac_criterion_ge_0.8": float(np.mean(x[ok] >= 0.8)),
                        "truth_agreement_when_criterion_ge_0.8": float(y[ok][x[ok] >= 0.8].mean()) if np.any(x[ok] >= 0.8) else None}
            if kind == "all":
                out[lab]["per_target_spearman"] = {t: float(spearmanr(x[ok & (tnames[m] == t)], y[ok & (tnames[m] == t)])[0])
                                                   for t in CUR if np.sum(ok & (tnames[m] == t)) > 5}
        rep["paper_criterion"][kind] = out
    (S3 / "ads_eval.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("reproduction", "vector_fitting")}, indent=1))


if __name__ == "__main__":
    main()
