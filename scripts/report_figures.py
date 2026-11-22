"""English, print-ready figures for the LaTeX draft report (report/figures/*.pdf).

Reads only existing result files; no simulation, no training.
Usage (openems env, CSXCAD_INSTALL_PATH set because oems_filter is imported):
    python scripts/report_figures.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import oems_filter as o  # noqa: E402

RES = ROOT / "results"
FW = RES / "fullwave_openems"
EM = ROOT / "data" / "filter_supplement_v2" / "em_diagnostics"
FIG = ROOT / "report" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.alpha": 0.35,
})
# Categorical slots 1-4 (fixed order) + ink/neutral.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, GRAY, RED = "#52514e", "#a3a29d", "#e34948"
# Physical widths (in) chosen so each figure prints at ~100% in the single-column draft
# (text width 6.5 in): COL_W at 0.55-0.6\linewidth, CASE_W at 0.65\linewidth, SMALL_W in a half-width minipage.
COL_W, CASE_W, SMALL_W = 3.45, 4.2, 3.05


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if os.environ.get("FIG_PREVIEW_DIR"):  # optional PNG previews outside the report folder
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / f"{name}.png", dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_decomposition():
    """Spearman of |phi| vs ADS ground-truth Shapley (test), PR-Net vs M0."""
    a = json.loads((RES / "step3" / "ads_eval.json").read_text(encoding="utf-8"))["decomposition"]
    targets = ["T0", "T1", "T1b", "T2"]
    series = [("model6_vs_ADS", "Network vs ADS", INK, "D"), ("shap_ref_vs_ADS", "SHAP vs ADS", BLUE, "o"),
              ("ig_vs_ADS", "IG vs ADS", ORANGE, "s"), ("lime_delta_vs_ADS", r"LIME-$\Delta$ vs ADS", AQUA, "^")]
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 2.3), sharey=True, gridspec_kw={"wspace": 0.16})
    for ax, model, title in zip(axs, ["PRNet", "M0"], ["PR-Net (main model)", "M0 (weak baseline)"]):
        d = a[model]["test"]
        for k, (key, lab, col, mk) in enumerate(series):
            y = np.arange(len(targets)) + (k - 1.5) * 0.17
            x = [d[key]["spearman_abs"][t] for t in targets]
            ax.plot(x, y, mk, color=col, ms=4, mec="white", mew=0.5, label=lab, ls="none")
        ax.set_title(title, pad=3)
        ax.set_xlim(0.68, 1.01)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0], ["0.7", "0.8", "0.9", "1"])
        ax.grid(axis="x")
        ax.set_yticks(range(len(targets)), targets)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("Explanation target")
    fig.supxlabel(r"Spearman of $|\phi|$ with ADS ground truth (test, 40 designs)", fontsize=8, y=-0.06)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 1.12), frameon=False, handletextpad=0.2, columnspacing=1.2)
    save(fig, "fig_decomposition")


def fig_tuning():
    """Pre-registered step-4 success (PR-Net surrogate, 30 test cases) + step-4b fixed-pair rule."""
    s = json.loads((RES / "step4" / "step4_report.json").read_text(encoding="utf-8"))["summary"]
    b = json.loads((RES / "step4b" / "step4b_report.json").read_text(encoding="utf-8"))["models"]["PRNet"]["success"]
    rows = [  # (label, successes, ADS calls or None, group)
        ("SHAP top-2", s["shap_ref"]["success"], s["shap_ref"]["total_ads_calls"], "xai"),
        ("IG top-2", s["ig"]["success"], s["ig"]["total_ads_calls"], "xai"),
        (r"LIME-$\Delta$ top-2", s["lime_delta"]["success"], s["lime_delta"]["total_ads_calls"], "xai"),
        (r"Gradient$\times$span top-2", s["grad_x_span"]["success"], s["grad_x_span"]["total_ads_calls"], "sens"),
        (r"LIME slope$\times$span top-2", s["lime_slope_x_span"]["success"], s["lime_slope_x_span"]["total_ads_calls"], "sens"),
        ("ADS true Shapley top-2", s["ads_true_shap"]["success"], s["ads_true_shap"]["total_ads_calls"], "ref"),
        ("Fixed L1+L2 (no explanation)*", b["fixed_L1L2"], None, "base"),
        ("Random 2 parameters", s["random"]["success"], s["random"]["total_ads_calls"], "base"),
        ("SHAP top-2, revert, no search", s["shap_revert_no_search"]["success"], s["shap_revert_no_search"]["total_ads_calls"], "base"),
        ("Surrogate search, all 6 params", s["all6_surrogate"]["success"], s["all6_surrogate"]["total_ads_calls"], "ref"),
        ("ADS coordinate search", s["ads_coordinate"]["success"], s["ads_coordinate"]["total_ads_calls"], "ref"),
    ]
    col = {"xai": BLUE, "sens": "#86b6ef", "ref": INK, "base": GRAY}
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    y = np.arange(len(rows))
    for i, (lab, n, calls, g) in enumerate(rows):
        ax.barh(i, n, color=col[g], height=0.66, edgecolor="white", linewidth=0.8)
        txt = f"{n}/30" + (f"  ({calls} ADS)" if calls is not None else "")
        ax.text(n + 0.4, i, txt, va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 39)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("Designs fixed (ADS-confirmed mask violation = 0), of 30")
    ax.axvline(30, color=INK, lw=0.5, ls=":")
    ax.grid(axis="x")
    save(fig, "fig_tuning")


def _load(p):
    d = np.load(p)
    return d["frequency_Hz"], d["S_complex"]


def _mask(ax, which):
    segs = [(9, 10.8, -20), (13.2, 15, -15), (11.6, 12.4, -0.5)] if which == "S21" else [(11.6, 12.4, -10)]
    for a, b, lim in segs:
        ax.plot([a, b], [lim, lim], color=RED, lw=1.0, solid_capstyle="butt", zorder=1)
        if not (which == "S21" and lim == -0.5):
            ax.fill_between([a, b], lim, 5, color=RED, alpha=0.07, lw=0, zorder=0)


def fig_fullwave_summary():
    rep = json.loads((FW / "report.json").read_text(encoding="utf-8"))
    rows, base = rep["cases"], rep["baseline_oems_case_level"]["mask_violation_dB"]
    fig, ax = plt.subplots(figsize=(SMALL_W, 1.95))
    yt, yl = [], []
    for i, r in enumerate(rows):
        for k, (src, col, nm) in enumerate((("ads", ORANGE, "ADS"), ("oems", BLUE, "openEMS"))):
            y = 2.6 * i + k
            a, b = r[f"{src}_x0"]["mask_violation_dB"], r[f"{src}_tuned"]["mask_violation_dB"]
            ax.annotate("", xy=(b, y), xytext=(a, y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, shrinkA=3.5, shrinkB=3.5, mutation_scale=7))
            ax.plot(a, y, "o", ms=4.5, mfc="white", mec=col, mew=1.2)
            ax.plot(b, y, "o", ms=4.5, color=col, mec="white", mew=0.5)
            ax.text(b + (0.5 if b >= a else -0.5), y, f"{b:.1f}", va="center", ha="left" if b >= a else "right", fontsize=6.5, color=INK)
            yt.append(y)
            yl.append(f"{r['design_id'].replace('_', ' ')} {nm}")
    ax.axvline(0, color=INK, lw=0.6)
    ax.axvline(base, color=BLUE, lw=0.6, ls=":")
    ax.text(base + 0.3, -0.95, f"openEMS baseline {base:.2f} dB", fontsize=6.5, color=INK, va="bottom")
    ax.set_yticks(yt, yl)
    ax.set_ylim(2.6 * len(rows) - 1.2, -1.1)
    ax.set_xlim(-2.4, 22.5)
    ax.set_xlabel("Mask violation (dB), 0 = pass")
    ax.grid(axis="x")
    save(fig, "fig_fullwave_summary")


def fig_fullwave_case(design_id="test_0005", level="m4"):
    fig, axs = plt.subplots(1, 2, figsize=(CASE_W, 2.0), gridspec_kw={"wspace": 0.38})
    for t, col, lab in (("x0", GRAY, "original $x_0$"), ("tuned", BLUE, "SHAP-tuned")):
        fa, Sa = _load(FW / "ads" / "cases" / f"{design_id}_{t}.npz")
        fe, Se = _load(FW / "cases" / level / f"{design_id}_{t}" / "response.npz")
        ya, ye = o.to_db(Sa), o.to_db(Se)
        for j, ax in enumerate(axs):
            ax.plot(fa / 1e9, ya[:, j], "--", color=col, lw=0.9, label=f"{lab}, ADS circuit")
            ax.plot(fe / 1e9, ye[:, j], "-", color=col, lw=1.3, label=f"{lab}, openEMS full-wave")
    for ax, nm in zip(axs, ["S21", "S11"]):
        _mask(ax, nm)
        ax.set(xlabel="Frequency (GHz)", ylabel=f"$|{nm[0]}_{{{nm[1:]}}}|$ (dB)", xlim=(9, 15),
               ylim=(-50, 2) if nm == "S21" else (-40, 2))
        ax.grid()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=6.5, frameon=False, ncol=2, columnspacing=1.2)
    save(fig, "fig_fullwave_case")


def fig_baseline_solvers(level="m4"):
    fig, axs = plt.subplots(1, 2, figsize=(CASE_W, 2.0), gridspec_kw={"wspace": 0.38})
    curves =[("ADS circuit", FW / "ads" / "cases" / "baseline.npz", "--", "#0b0b0b"),
              ("Momentum (30 cells/$\\lambda$)", EM / "native_baseline_601.npz", "-", ORANGE),
              (f"openEMS ({level})", FW / "mesh_study" / level / "response.npz", "-", BLUE)]
    for lab, p, ls, col in curves:
        f, S = _load(p)
        y = o.to_db(S)
        for j, ax in enumerate(axs):
            ax.plot(f / 1e9, y[:, j], ls, color=col, lw=1.1, label=lab)
    for ax, nm in zip(axs, ["S21", "S11"]):
        _mask(ax, nm)
        ax.set(xlabel="Frequency (GHz)", ylabel=f"$|{nm[0]}_{{{nm[1:]}}}|$ (dB)", xlim=(9, 15),
               ylim=(-50, 2) if nm == "S21" else (-40, 2))
        ax.grid()
    axs[0].legend(loc="lower center", frameon=False, fontsize=6.5)
    save(fig, "fig_baseline_solvers")


def fig_mesh():
    """openEMS f0 rises, Momentum f0 falls with refinement; ADS between."""
    rep = json.loads((FW / "report.json").read_text(encoding="utf-8"))["baseline"]
    oe = rep["openems_mesh_study"]
    mo = rep["momentum_mesh"]
    fig, ax = plt.subplots(figsize=(SMALL_W, 1.95))
    ax.plot(range(1, len(oe) + 1), [m["f0_GHz"] for m in oe], "o-", color=BLUE, ms=4, lw=1.1, label="openEMS (levels m1-m4)")
    ax.plot(range(1, len(mo) + 1), [m["f0_GHz"] for m in mo], "s-", color=ORANGE, ms=4, lw=1.1,
            label="Momentum (30/45/60/90 cells/$\\lambda$)")
    ax.axhline(rep["ads"]["f0_GHz"], color="#0b0b0b", ls="--", lw=0.8)
    ax.text(0.85, rep["ads"]["f0_GHz"] + 0.008, "ADS circuit", va="bottom", fontsize=6.5)
    ax.set(xticks=[1, 2, 3, 4], xticklabels=["coarsest", "", "", "finest"], ylabel="Baseline $f_0$ (GHz)", xlim=(0.8, 4.4))
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3)
    save(fig, "fig_mesh")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_decomposition, fig_tuning, fig_fullwave_summary, fig_fullwave_case, fig_baseline_solvers, fig_mesh):
        f()
        print("ok", f.__name__)
