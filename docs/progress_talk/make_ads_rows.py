"""Hierarchy figure from the user's ADS screenshots: big -> small, with circles and arrows.

Row 'schematic': spar_sim test bench -> cmstpfil (two mirrored halves) -> halffil (TL1, CLin1, CLin2).
Row 'layout'   : cmstpfil layout -> halffil layout -> zoom on CLin1 (needs the layout screenshots).
Usage: python make_ads_rows.py <ads_dir> <out_dir> [check]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Ellipse, FancyBboxPatch
from PIL import Image

ADS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CHECK = len(sys.argv) > 3
RED, INK, HOT = "#D03B3B", "#1B1B1B", "#FF5A5A"
plt.rcParams.update({"font.family": "Arial"})


def crop(name, box):
    return Image.open(ADS / name).convert("RGB").crop(box)


def row(panels, links, fname, heights, gap_in=0.55, top=0.42, label_size=11):
    """panels: [(img, title, marks)], marks: list of ('ellipse'|'box', (x, y, w, h) in crop px, line tags or None).
    heights: displayed height (in) of each panel, centred vertically. links: [(panel_i, mark_j)] -> arrow to panel i+1."""
    widths = [p[0].width / p[0].height * h for p, h in zip(panels, heights)]
    W = sum(widths) + gap_in * (len(panels) - 1)
    Hf = max(heights) + top
    fig = plt.figure(figsize=(W, Hf), dpi=200)
    axes, x = [], 0.0
    for (im, title, marks, *extra), w, hh in zip(panels, widths, heights):
        ax = fig.add_axes([x / W, (max(heights) - hh) / 2 / Hf, w / W, hh / Hf])
        ax.imshow(im)
        for fn in extra:
            fn(ax)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#9A9890"); s.set_linewidth(1.2)
        ax.set_title(title, fontsize=13, color=INK, loc="left", pad=6, fontweight="bold")
        for kind, (mx, my, mw, mh), lab in marks:
            col = HOT if kind.startswith("dark") else RED  # brighter red on the black layout canvas
            if kind.endswith("ellipse"):
                ax.add_patch(Ellipse((mx + mw / 2, my + mh / 2), mw, mh, fill=False, ec=col, lw=2.6))
            else:
                ax.add_patch(FancyBboxPatch((mx, my), mw, mh, boxstyle="round,pad=2", fill=False, ec=col, lw=2.4))
            for ly, text in (lab or []):  # per-line tags right of the box, at image y = ly
                ax.text(mx + mw + 10, ly, "→ " + text, color=RED, fontsize=label_size, fontweight="bold",
                        va="center", ha="left")
        axes.append(ax)
        x += w + gap_in
    for link in links:
        i, j = link[:2]
        style = link[2] if len(link) > 2 else {}
        kind, (mx, my, mw, mh), _ = panels[i][2][j]
        nxt = axes[i + 1]
        h = panels[i + 1][0].height
        start = style.get("start", (mx + mw, my + mh / 2))
        end = style.get("end", (0, h * 0.5))
        con = ConnectionPatch(xyA=start, coordsA=axes[i].transData, xyB=end, coordsB=nxt.transData,
                              arrowstyle="-|>", color=RED, lw=2.2, mutation_scale=18,
                              connectionstyle=style.get("conn", "arc3,rad=0"))
        fig.add_artist(con)
    fig.savefig(OUT / fname, dpi=200)
    plt.close(fig)
    print("saved", OUT / fname, f"{W:.2f} x {Hf:.2f} in")


OUT.mkdir(parents=True, exist_ok=True)
if CHECK:  # dump raw crops to calibrate boxes
    for n, b in [("sch_testbench.png", (205, 395, 845, 655)), ("sch_full.png", (425, 375, 1000, 565)),
                 ("sch_half.png", (205, 170, 1215, 640))]:
        crop(n, b).save(OUT / ("raw_" + n))
    sys.exit()

tb = crop("sch_testbench.png", (205, 395, 845, 655))
full = crop("sch_full.png", (425, 375, 1000, 565))
half = crop("sch_half.png", (610, 170, 1258, 640))
tb = crop("sch_testbench.png", (205, 395, 575, 655))
row([(tb, "Test bench", [("ellipse", (238, -2, 128, 100), None)]),
     (full, "Filter = two mirrored halves", [("ellipse", (400, 2, 160, 138), None)]),
     (half, "One half: CLin1 + CLin2", [("box", (6, 222, 152, 95), [(237, "Wc1"), (268, "S1"), (300, "L1")]),
                                        ("box", (299, 300, 150, 95), [(315, "Wc2"), (347, "S2"), (379, "L2")])])],
    [(0, 0), (1, 0)], "fig_ads_schematic.png", heights=[1.55, 1.0, 2.35])

# ---------------- layout row ----------------
lfull = crop("lay_full.png", (180, 345, 1700, 795))
lhalf = crop("lay_half.png", (170, 320, 1700, 810))
zoom = crop("lay_half.png", (560, 330, 1180, 610))
WH = "white"


def dims(ax):
    """CLin1 close-up (crop px): strip a x 42-569 y 37-132; strip b y 147-242; gap y 132-147."""
    ax.annotate("", (42, 18), (569, 18), arrowprops=dict(arrowstyle="<->", color=WH, lw=1.8, shrinkA=0, shrinkB=0))
    ax.text(305, 18, " L1 = 138.5 mil ", color=INK, fontsize=12, fontweight="bold", ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.annotate("", (95, 147), (95, 242), arrowprops=dict(arrowstyle="<->", color=WH, lw=1.8, shrinkA=0, shrinkB=0))
    ax.text(108, 195, "Wc1 = 25.5 mil", color=WH, fontsize=12, fontweight="bold", va="center")
    ax.annotate("S1 = 3.3 mil", xy=(470, 139.5), xytext=(455, 196), color=WH, fontsize=12, fontweight="bold",
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color=WH, lw=1.4))


row([(lfull, "Full layout", [("dark_box", (742, 178, 752, 252), None)]),
     (lhalf, "One half (halffil)", [("dark_box", (428, 42, 535, 216), None)]),
     (zoom, "CLin1 up close", [], dims)],
    [(0, 0), (1, 0, {"start": (963, 42), "end": (0, 30), "conn": "arc3,rad=-0.25"})],
    "fig_ads_layout.png", heights=[1.0, 1.25, 1.9])
