"""To-scale top view of the edge-coupled filter in the ADS schematic (cmstpfil = halffil + mirrored halffil).

Values from the user's ADS screenshots (halffil): feed TL1 W=32 L=100 mil; CLin1 W=25.5 S=3.3 L=138.5;
CLin2 W=31.5 S=18 L=136; substrate H=15 mil, Er=3.78, lossless. Sections in order CLin1, CLin2, CLin2, CLin1.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).parent / "figs" / "fig_geometry.png"
CU, CU2, INK, INK2, SUB = "#B8612A", "#D9A27C", "#1B1B1B", "#4A4A48", "#F1F4F7"
plt.rcParams.update({"font.family": "Arial", "font.size": 13})

secs = [("1", 138.5, 25.5, 3.3), ("2", 136.0, 31.5, 18.0), ("2", 136.0, 31.5, 18.0), ("1", 138.5, 25.5, 3.3)]
FEED_L, FEED_W = 100.0, 32.0
fig, ax = plt.subplots(figsize=(7.0, 2.45), dpi=220)
ax.add_patch(Rectangle((0, -secs[0][2] / 2), FEED_L, FEED_W, color=CU))  # feed shares the coupled-side edge
geo = []
x = FEED_L
prev = None  # (top, bottom) of the previous section's strip b
for k, (n, L, W, S) in enumerate(secs):
    col = CU if k < 2 else CU2
    if prev is None:
        top_a = W / 2  # first strip centred on the feed line
    elif k < 2:
        top_a = prev[0]  # first half: joined strips share the top edge (as in the ADS layout)
    else:
        top_a = prev[1] + W  # mirrored half: joined strips share the bottom edge
    ya = top_a - W / 2
    yb = ya - (W + S)
    ax.add_patch(Rectangle((x, ya - W / 2), L, W, color=col))
    ax.add_patch(Rectangle((x, yb - W / 2), L, W, color=col))
    geo.append((n, x, L, W, S, ya, yb))
    prev = (yb + W / 2, yb - W / 2)
    x, yc = x + L, yb
y_out = yc + secs[-1][2] / 2 - FEED_W
ax.add_patch(Rectangle((x, y_out), FEED_L, FEED_W, color=CU2))
xmax, ymin = x + FEED_L, y_out


def dim_h(x0, x1, y, text):
    ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=INK, lw=1, shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y + 4, text, ha="center", va="bottom", color=INK, fontsize=13, fontweight="bold")


for n, x0, L, W, S, ya, yb in geo[:2]:
    dim_h(x0, x0 + L, ya + W / 2 + 14, f"L{n} = {L:g}")
    # gap S between strips
    top_b, bot_a = yb + W / 2, ya - W / 2
    if n == "1":
        ax.annotate(f"S1 = {S:g}", xy=(x0 + 4, (top_b + bot_a) / 2), xytext=(2, -52), color=INK, fontsize=12,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.9), ha="left", va="center")
    else:
        ax.text(x0 + 6, (top_b + bot_a) / 2, f"S2 = {S:g}", color=INK, fontsize=10.5, va="center", ha="left")
    # width W of strip b
    wx = x0 + 10
    ax.annotate("", (wx, yb - W / 2), (wx, yb + W / 2), arrowprops=dict(arrowstyle="<->", color="white", lw=1, shrinkA=0, shrinkB=0))
    ax.text(wx + 6, yb, f"Wc{n} = {W:g}", va="center", ha="left", color="white", fontsize=10.5, fontweight="bold")
ax.text(geo[3][1] + 70, -72, "mirror image of\nsections 1–2", color=INK2, fontsize=12, style="italic", ha="center",
        va="center")
ax.text(0, ymin - 4, "Top view to scale (mil), as in the ADS layout  ·  substrate H = 15 mil, εr = 3.78, lossless",
        color=INK2, fontsize=11, va="top")
ax.set_xlim(-8, xmax + 8)
ax.set_ylim(ymin - 38, 52)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout(pad=0.2)
fig.savefig(OUT)
print("saved", OUT, "extent", xmax, ymin)
