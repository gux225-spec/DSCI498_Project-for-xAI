"""Summarise the openEMS full-wave checks: mesh study (baseline) and step-4 tuning cases.

Writes results/fullwave_openems/report.md, report.json and figures/.
Usage: python scripts/oems_report.py [mesh_level=m4]      |  python scripts/oems_report.py ads_check
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
C_ADS, C_OEMS, C_MOM, C_X0, C_TUNED, C_INK = "#eb6834", "#2a78d6", "#eb6834", "#8a8984", "#2a78d6", "#52514e"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import oems_filter as o  # noqa: E402

OUT = ROOT / "results" / "fullwave_openems"
EM = ROOT / "data" / "filter_supplement_v2" / "em_diagnostics"
FIG = OUT / "figures"
COMP_ZH = {"passband_S21": "通带 S21", "passband_S11": "通带 S11", "low_stopband_S21": "低阻带", "high_stopband_S21": "高阻带"}


def load(path):
    d = np.load(path)
    return d["frequency_Hz"], d["S_complex"]


def metrics(f, S):
    y = o.to_db(S)
    v, comps = o.mask_violation(f, y) if len(f) == 601 else (float("nan"), {})
    return {**o.band_metrics(f, y[:, 0]), "mask_violation_dB": v, "mask_components_dB": comps}


def cases():
    return json.loads((OUT / "cases.json").read_text())["cases"]


def ads_check():
    for c in cases():
        for t in ("x0", "tuned"):
            m = metrics(*load(OUT / "ads" / "cases" / f"{c['design_id']}_{t}.npz"))
            print(c["design_id"], t, "ADS now", round(m["mask_violation_dB"], 4),
                  "step4 record", round(c[f"ads_violation_{t}_dB"], 4))


def shade_mask(ax, which):
    """Mask limits as red segments; forbidden side lightly shaded (stopbands / S11 only)."""
    kw = dict(color="#e34948", alpha=0.07, lw=0)
    segs = [(9, 10.8, -20), (13.2, 15, -15), (11.6, 12.4, -0.5)] if which == "S21" else [(11.6, 12.4, -10)]
    for a, b, lim in segs:
        ax.plot([a, b], [lim, lim], color="#e34948", lw=1.2, solid_capstyle="butt")
        if not (which == "S21" and lim == -0.5):
            ax.fill_between([a, b], lim, 5, **kw)
    if which == "S21":
        ax.axvspan(11.6, 12.4, color="#e34948", alpha=0.03, lw=0)


def fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main(level):
    FIG.mkdir(parents=True, exist_ok=True)
    rep = {"case_mesh_level": level}

    # ---- 1. baseline: mesh study vs Momentum vs ADS
    ads_base = metrics(*load(OUT / "ads" / "cases" / "baseline.npz"))
    mesh_rows = []
    for d in sorted((OUT / "mesh_study").glob("m*/summary.json")):
        s = json.loads(d.read_text())
        mesh_rows.append({k: s[k] for k in ["level", "mesh_lines_xyz", "elapsed_s", "f0_GHz", "BW_GHz", "mask_violation_dB"]})
    mom = [{"cells_per_wavelength": 30, "f0_GHz": metrics(*load(EM / "native_baseline_601.npz"))["f0_GHz"]}]
    fm = json.loads((EM / "focused_mesh_checks.json").read_text())["comparisons"]
    mom += [{"cells_per_wavelength": c["mesh_to"], "f0_GHz": c["center_to_GHz"]} for c in fm]
    rep["baseline"] = {"ads": ads_base, "openems_mesh_study": mesh_rows, "momentum_mesh": mom,
                       "momentum_601_30cells": metrics(*load(EM / "native_baseline_601.npz"))}

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), tight_layout=True)
    curves = [("ADS 电路", OUT / "ads" / "cases" / "baseline.npz", "--", "#0b0b0b"),
              ("Momentum 30 cells/λ", EM / "native_baseline_601.npz", "-", C_MOM),
              (f"openEMS {level}", OUT / "mesh_study" / level / "response.npz", "-", C_OEMS)]
    for lab, p, ls, col in curves:
        f, S = load(p)
        y = o.to_db(S)
        for j, ax in enumerate(axs):
            ax.plot(f / 1e9, y[:, j], ls, color=col, lw=1.5, label=lab)
    for ax, nm in zip(axs, ["S21", "S11"]):
        shade_mask(ax, nm)
        ax.set(xlabel="Frequency (GHz)", ylabel=f"{nm} (dB)", xlim=(9, 15), ylim=(-50 if nm == "S21" else -40, 2), title=f"Baseline {nm}")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.savefig(FIG / "baseline_three_solvers.png", dpi=130)
    plt.close(fig)

    # ---- 2. step-4 cases
    rows = []
    for c in cases():
        r = {"design_id": c["design_id"], "shap_pair": c["shap_pair"]}
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), tight_layout=True)
        for t, col in (("x0", C_X0), ("tuned", C_TUNED)):
            fa, Sa = load(OUT / "ads" / "cases" / f"{c['design_id']}_{t}.npz")
            fe, Se = load(OUT / "cases" / level / f"{c['design_id']}_{t}" / "response.npz")
            r[f"ads_{t}"], r[f"oems_{t}"] = metrics(fa, Sa), metrics(fe, Se)
            ya, ye = o.to_db(Sa), o.to_db(Se)
            lab = "原始" if t == "x0" else "SHAP 调谐后"
            for j, ax in enumerate(axs):
                ax.plot(fa / 1e9, ya[:, j], "--", color=col, lw=1.2, label=f"{lab} ADS")
                ax.plot(fe / 1e9, ye[:, j], "-", color=col, lw=1.8, label=f"{lab} openEMS")
        for ax, nm in zip(axs, ["S21", "S11"]):
            shade_mask(ax, nm)
            ax.set(xlabel="Frequency (GHz)", ylabel=f"{nm} (dB)", xlim=(9, 15), ylim=(-50 if nm == "S21" else -40, 2),
                   title=f"{c['design_id']}  (SHAP: {c['shap_pair']})  {nm}")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.savefig(FIG / f"{c['design_id']}.png", dpi=130)
        plt.close(fig)
        r["oems_improvement_dB"] = r["oems_x0"]["mask_violation_dB"] - r["oems_tuned"]["mask_violation_dB"]
        r["oems_tuned_passes"] = r["oems_tuned"]["mask_violation_dB"] <= 1e-9
        r["f0_shift_oems_minus_ads_MHz"] = {t: 1e3 * (r[f"oems_{t}"]["f0_GHz"] - r[f"ads_{t}"]["f0_GHz"]) for t in ("x0", "tuned")}
        rows.append(r)
    rep["cases"] = rows
    base_oems = json.loads((OUT / "mesh_study" / level / "summary.json").read_text())
    summary_figure(rows, base_oems["mask_violation_dB"])
    rep["baseline_oems_case_level"] = base_oems
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    # ---- 3. markdown
    L = [f"# 全波复核（openEMS FDTD，代替原计划的 HFSS）", "",
         "原计划在 HFSS 里跑少量案例，检验解释引导的调谐决策是否在全波模型里成立。HFSS 学生版未获批，改用开源 openEMS（v0.37.0）。"
         "版图与 Momentum 诊断模板完全相同（5 个金属轮廓，`scripts/oems_filter.py`），基板 H=15 mil、Er=3.78、无损、理想导体（零厚度 PEC）。",
         "频率网格与 ADS 主数据相同：9–15 GHz，601 点；掩模与 `src/filterxai/targets.py` 相同。", "",
         "## 1. 基线校准：网格研究", "",
         "| openEMS 网格 | 网格线 (x,y,z) | 用时 (s) | f0 (GHz) | 带宽 (GHz) | 掩模违反 (dB) |", "|---|---|---|---|---|---|"]
    for m in mesh_rows:
        L.append(f"| {m['level']} | {m['mesh_lines_xyz']} | {m['elapsed_s']:.0f} | {m['f0_GHz']:.4f} | {m['BW_GHz']:.4f} | {m['mask_violation_dB']:.2f} |")
    L += ["", "| 参照 | f0 (GHz) |", "|---|---|", f"| ADS 电路 | {ads_base['f0_GHz']:.4f} |"]
    L += [f"| Momentum {m['cells_per_wavelength']} cells/λ | {m['f0_GHz']:.4f} |" for m in mom]
    L += ["", f"ADS 带宽 {ads_base['BW_GHz']:.4f} GHz；Momentum（30 cells/λ, 601 点）带宽 {rep['baseline']['momentum_601_30cells']['BW_GHz']:.4f} GHz，"
          f"掩模违反 {rep['baseline']['momentum_601_30cells']['mask_violation_dB']:.2f} dB。", "",
          "openEMS 的 f0 随加密单调上升，Momentum 的 f0 随加密单调下降，两者都未完全收敛，但从两侧向 ADS 附近收拢。"
          "因此数据集 README 中记录的 Momentum–电路 约 122 MHz 偏差，至少一部分来自 Momentum 的网格；全波基线中心频率目前只能界定在两条收敛序列之间。",
          "", "![基线](figures/baseline_three_solvers.png)", "",
          f"## 2. 第 4 步调谐案例（openEMS 网格 {level}）", "",
          "选例规则（在看到任何 openEMS 案例结果前写入 `cases.json`）：按 prereg 顺序取 SHAP 调谐在 ADS 中成功、且参数对互不重复的前 3 个 test 案例。",
          "每例仿真原始设计 x0 与 SHAP 调谐后的设计（ADS 轨迹上第一个通过的几何）。", "",
          "| 设计 | SHAP 参数对 | ADS 违反 x0 → 调谐 (dB) | openEMS 违反 x0 → 调谐 (dB) | openEMS 改善 (dB) | openEMS 调谐后通过 | f0 偏移 openEMS−ADS (MHz) x0 / 调谐 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['design_id']} | {r['shap_pair']} | {fmt(r['ads_x0']['mask_violation_dB'])} → {fmt(r['ads_tuned']['mask_violation_dB'])} | "
                 f"{fmt(r['oems_x0']['mask_violation_dB'])} → {fmt(r['oems_tuned']['mask_violation_dB'])} | {fmt(r['oems_improvement_dB'])} | "
                 f"{'是' if r['oems_tuned_passes'] else '否'} | {fmt(r['f0_shift_oems_minus_ads_MHz']['x0'], 0)} / {fmt(r['f0_shift_oems_minus_ads_MHz']['tuned'], 0)} |")
    L += ["", f"同一网格下 openEMS 基线本身的掩模违反为 {base_oems['mask_violation_dB']:.2f} dB（ADS 为 0）。", "",
          "openEMS 调谐后设计的违反分项 (dB)：", "", "| 设计 | " + " | ".join(COMP_ZH.values()) + " |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['design_id']} | " + " | ".join(fmt(r["oems_tuned"]["mask_components_dB"][k]) for k in COMP_ZH) + " |")
    L += [""] + [f"![{r['design_id']}](figures/{r['design_id']}.png)" for r in rows]
    L += conclusions(rows, ads_base, base_oems, mesh_rows)
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def summary_figure(rows, base_viol):
    """Dumbbell: mask violation x0 (hollow) -> tuned (filled), per case, ADS vs openEMS."""
    fig, ax = plt.subplots(figsize=(8, 3.6), tight_layout=True)
    yt, yl = [], []
    for i, r in enumerate(rows):
        for k, (src, col, nm) in enumerate((("ads", C_ADS, "ADS"), ("oems", C_OEMS, "openEMS"))):
            y = 3 * i + k
            a, b = r[f"{src}_x0"]["mask_violation_dB"], r[f"{src}_tuned"]["mask_violation_dB"]
            ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=2, shrinkA=5, shrinkB=5))
            ax.plot(a, y, "o", ms=8, mfc="white", mec=col, mew=2)
            ax.plot(b, y, "o", ms=8, color=col, mec="white", mew=1)
            ax.text(b + (0.4 if b >= a else -0.4), y, f"{b:.1f}", va="center", ha="left" if b >= a else "right", fontsize=8, color=C_INK)
            yt.append(y)
            yl.append(f"{r['design_id']}  {nm}")
    ax.axvline(0, color=C_INK, lw=1)
    ax.axvline(base_viol, color=C_OEMS, lw=1, ls=":")
    ax.text(base_viol + 0.2, -0.7, f"openEMS 基线自身违反 {base_viol:.2f} dB", fontsize=8, color=C_INK, va="bottom")
    ax.set(yticks=yt, yticklabels=yl, xlabel="掩模违反 (dB)，0 = 通过；空心 = 原始 x0，实心 = SHAP 调谐后",
           title="SHAP 调谐决策：ADS 中全部修好，全波中 0/3 通过、2/3 改善")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=-2.2)
    ax.set_ylim(3 * len(rows) - 1.5, -1.3)
    fig.savefig(FIG / "summary_violation.png", dpi=130)
    plt.close(fig)


def conclusions(rows, ads_base, base_oems, mesh_rows):
    n = len(rows)
    n_pass = sum(r["oems_tuned_passes"] for r in rows)
    better = [r for r in rows if r["oems_improvement_dB"] > 0]
    worse = [r for r in rows if r["oems_improvement_dB"] <= 0]
    df = [r["f0_shift_oems_minus_ads_MHz"]["tuned"] for r in rows]
    bw = [1e3 * (r["oems_tuned"]["BW_GHz"] - r["ads_tuned"]["BW_GHz"]) for r in rows]
    f0_ads_t = [r["ads_tuned"]["f0_GHz"] for r in rows]
    last_step = 1e3 * (mesh_rows[-1]["f0_GHz"] - mesh_rows[-2]["f0_GHz"])
    rad = {"oems": power_loss(OUT / "mesh_study" / mesh_rows[-1]["level"] / "response.npz"),
           "mom": power_loss(EM / "native_baseline_601.npz")}
    L = ["", "![汇总](figures/summary_violation.png)", "", "## 3. 结论", "",
         f"**C1. 调谐决策“修好了这个设计”在全波下不成立：{n_pass}/{n} 通过。** "
         f"ADS 里 3 个调谐后设计的违反都是 0，openEMS 里分别为 "
         + "、".join(f"{r['oems_tuned']['mask_violation_dB']:.1f}" for r in rows) + " dB，"
         f"都远大于同网格下基线自身的 {base_oems['mask_violation_dB']:.2f} dB，也远大于网格最后一级加密带来的变化（基线违反 m3→m4 只变了 "
         f"{abs(mesh_rows[-2]['mask_violation_dB'] - mesh_rows[-1]['mask_violation_dB']):.2f} dB）。所以“不通过”不是网格没收敛造成的。", "",
         f"**C2. 调谐方向在 {len(better)}/{n} 个案例中成立，在 {len(worse)}/{n} 个案例中失败。** "
         + "；".join(f"{r['design_id']}（{r['shap_pair']}）改善 {r['oems_improvement_dB']:.1f} dB" for r in better) + "。"
         + "".join(f"{r['design_id']}（{r['shap_pair']}）在全波里反而变差 {-r['oems_improvement_dB']:.1f} dB。" for r in worse), "",
         f"**C3. 失败的机制是同一个：电路模型系统性地高估带宽、略高估中心频率，而调谐把设计推到了掩模边缘。** "
         f"调谐后设计 openEMS 相对 ADS 的 f0 偏移为 " + "、".join(f"{d:+.0f}" for d in df) + " MHz，"
         f"带宽偏差为 " + "、".join(f"{d:+.0f}" for d in bw) + " MHz。"
         f"基线上也有同样的带宽差（openEMS {base_oems['BW_GHz']:.3f} vs ADS {ads_base['BW_GHz']:.3f} GHz），"
         "而 Momentum 给出的带宽（约 1.13 GHz）与 openEMS 一致，所以“带宽变窄约 0.12–0.14 GHz”是两个独立全波求解器都看到的，不是 openEMS 的个别问题。"
         f"ADS 调谐后的 f0 为 " + "、".join(f"{x:.3f}" for x in f0_ads_t) + " GHz（掩模中心 12.0 GHz）。"
         "偏低的设计在全波中再往下移，碰到低阻带；即使基本居中的设计也失败了，说明仅带宽变窄就足以让 ADS 中“刚好通过”的设计掉出通带要求。"
         "全波违反都集中在通带 S21/S11 和低阻带，高阻带始终为 0（分项见上表）。", ""]
    for r in worse:
        a0, at = r["ads_x0"], r["ads_tuned"]
        L += [f"**C4. {r['design_id']} 为什么变差。** 它的原始问题是匹配（ADS 中通带 S11 违反 {a0['mask_components_dB']['passband_S11']:.1f} dB，"
              f"f0 = {a0['f0_GHz']:.3f} GHz 本来是居中的）。SHAP 选的 {r['shap_pair']} 在 ADS 里通过把 f0 下移 "
              f"{1e3 * (a0['f0_GHz'] - at['f0_GHz']):.0f} MHz 到 {at['f0_GHz']:.3f} GHz 来“修好”它——这是在电路模型里成立、但恰好朝着全波偏移方向的修法。"
              f"全波里 f0 落到 {r['oems_tuned']['f0_GHz']:.3f} GHz，新增了 {r['oems_tuned']['mask_components_dB']['low_stopband_S21']:.1f} dB 的低阻带违反，"
              "原有的 S11 问题也没有消失。也就是说，解释给出的“该调哪两个参数”在代理/电路模型内部是对的，但它修复的路径依赖于该模型的系统误差。", ""]
    L += ["**对项目三个问题的回答。**", "",
          "- *真不真（true）*：SHAP 归因在它所解释的 ADS 模型内部可用（3/3 在 ADS 里修好）；但它不携带关于“模型本身偏差”的任何信息，换到全波模型后决策结果不再成立。",
          f"- *有没有用（useful）*：作为方向提示有用（{len(better)}/{n} 个案例全波违反大幅下降，约 13 dB）；作为“改完即可交付”的判据没用（0/{n}）。",
          "- *站不站得住（hold up）*：在模型偏移下站不住。站得住的最低要求是：调谐停止条件留出余量（至少覆盖约 0.13 GHz 带宽差和约 50 MHz f0 偏移），"
          "或者调谐后必须经过一次全波复核。这一条应写进给接收端排障人员的使用说明。", "",
          "## 4. 这些结论不依赖什么、依赖什么", "",
          f"- 不依赖网格细节：最后一级加密 f0 只变了 {last_step:.0f} MHz、违反只变了约 0.1 dB，与 5–10 dB 的调谐后违反相比可以忽略。",
          "- 依赖相同的理想化物理（零厚度 PEC、无损介质），这与 ADS 主数据一致，所以差异来自建模层级（电路 vs 全波），不是来自损耗。加入损耗只会让通带更差。",
          f"- 只有 {n} 个案例，结论是“存在明确的反例和系统性机制”，不是统计意义上的失败率估计。",
          "- openEMS 不是 HFSS；但 Momentum 的独立结果在带宽上与它一致，可作为交叉印证。",
          f"- 通带 S21 的不足不是 openEMS 的伪影：基线通带内不守恒功率 1−|S11|²−|S21|²，openEMS 平均 {rad['oems'][1]:.0%}，"
          f"Momentum 平均 {rad['mom'][1]:.0%}（ADS 为 0）。两个全波求解器一致，这是开放微带的辐射，"
          f"约折合 {-10 * np.log10(1 - rad['oems'][1]):.1f} dB 的通带插损，电路模型完全没有。",
          f"- openEMS 的一个已知缺陷：带外（9 GHz）不守恒功率 {rad['oems'][0]:.0%}，Momentum 只有 {rad['mom'][0]:.0%}，"
          "说明 openEMS 端口/吸收边界有额外误差，表现为阻带 −30 dB 以下的波纹。它不影响掩模判定（阻带限值是 −20/−15 dB），但带外细节不要按 openEMS 解读。"]
    return L


def power_loss(path):
    """(fraction at 9 GHz, mean fraction 11.6-12.4 GHz) of 1-|S11|^2-|S21|^2."""
    f, S = load(path)
    f = f / 1e9
    lost = 1 - abs(S[:, 0, 0]) ** 2 - abs(S[:, 1, 0]) ** 2
    pb = (f >= 11.6) & (f <= 12.4)
    return float(lost[np.argmin(abs(f - 9))]), float(lost[pb].mean())


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "m4"
    ads_check() if arg == "ads_check" else main(arg)
