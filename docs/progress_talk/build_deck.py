"""Build the Nov 10 progress-talk deck (python-pptx) and a PIL preview of every slide from the same spec."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
FIG = HERE / "figs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "deck.pptx"
PREV = HERE / "preview"
PREV.mkdir(exist_ok=True)

NAVY, COPPER, TINT, INK, INK2, MUTED, WHITE = "13293D", "B8612A", "EAF0F5", "1B1B1B", "4A4A48", "77756F", "FFFFFF"
ICE = "C9D7E3"
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5

# ---------------- slide specs ----------------
# text element: dict(kind="text", x, y, w, h, paras=[[(text, opts), ...], ...], size, color, font, bold, align, anchor, space)
def T(x, y, w, h, paras, size=16, color=INK, font=BODY, bold=False, italic=False, align="l", anchor="t", space=6, name=None):
    if isinstance(paras, str):
        paras = [paras]
    paras = [[(p, {})] if isinstance(p, str) else p for p in paras]
    return dict(kind="text", x=x, y=y, w=w, h=h, paras=paras, size=size, color=color, font=font, bold=bold,
                italic=italic, align=align, anchor=anchor, space=space, name=name)


def R(x, y, w, h, fill, shape="rect", name=None):
    return dict(kind="shape", x=x, y=y, w=w, h=h, fill=fill, shape=shape, name=name)


def I(path, x, y, w, h, alt, name=None):
    return dict(kind="img", path=str(path), x=x, y=y, w=w, h=h, alt=alt, name=name)


def TITLE(text, color=NAVY):
    return T(0.6, 0.42, 12.1, 0.95, text, size=32, color=color, font=HEAD, bold=True, anchor="m", name="Title")


def NUM(n, x, y, d=0.46, fill=COPPER):
    return [R(x, y, d, d, fill, "oval"), T(x, y, d, d, str(n), size=16, color=WHITE, bold=True, align="c", anchor="m")]


B = lambda s: (s, {"bold": True})  # noqa: E731

slides = []

# 0 Title (dark)
slides.append(dict(bg=NAVY, els=[
    T(0.8, 2.0, 11.7, 1.7, "Can an RF engineer trust the explanation?", size=44, color=WHITE, font=HEAD, bold=True, anchor="b"),
    T(0.8, 3.85, 11.7, 1.0, "Testing SHAP, IG and LIME on a neural surrogate of a microstrip bandpass filter",
      size=20, color=ICE),
    T(0.8, 6.2, 11.7, 0.5, "DSCI 498 Explainable AI  ·  Progress talk  ·  November 10, 2026", size=14, color=ICE),
], notes=(
    "Hi, I'm [your name]. My project asks a simple question: when a neural network predicts how an RF filter "
    "behaves, can the engineer who reads its explanation actually trust it?")))

# 1 Framing
cards = [("DATA", "1,440 ADS circuit simulations; 6 geometry + 3 unused control inputs"),
         ("MODEL", "PR-Net pole–residue surrogate; predicts S21 / S11 from 9 to 15 GHz"),
         ("STAKEHOLDER", "Receive-side RF troubleshooter (interviewed Nov 5)"),
         ("DECISION", "Which failed spec and band to check next; what to hand to the designer")]
els = [TITLE("One model, one decision, one question")]
for i, (lab, txt) in enumerate(cards):
    cx, cy = 0.6 + (i % 2) * 3.6, 1.75 + (i // 2) * 2.45
    els += [R(cx, cy, 3.4, 2.15, TINT, "round"),
            T(cx + 0.25, cy + 0.22, 2.9, 0.4, lab, size=13, color=COPPER, bold=True),
            T(cx + 0.25, cy + 0.7, 2.9, 1.3, txt, size=17, color=INK)]
els += [R(7.95, 1.75, 4.8, 4.6, NAVY, "round"),
        T(8.3, 2.05, 4.15, 0.4, "PRINCIPLE: CONTRASTIVE", size=13, color="E3A77C", bold=True),
        T(8.3, 2.5, 4.15, 0.95, "Why does this design fail the spec, rather than pass like the reference?",
          size=17, color=WHITE, italic=True),
        T(8.3, 3.7, 4.15, 0.4, "QUESTION", size=13, color="E3A77C", bold=True),
        T(8.3, 4.15, 4.15, 1.95, "For a receive-side RF troubleshooter deciding which failed spec and band to "
          "investigate next, does a contrastive, spec-first explanation improve decision quality without losing "
          "faithfulness?", size=16, color=WHITE)]
slides.append(dict(bg=WHITE, els=els, notes=(
    "The data are 1,440 ADS circuit simulations of a microstrip bandpass filter, with six geometry inputs plus three "
    "control inputs that the simulator never uses. The model is PR-Net, a pole-residue network that predicts the "
    "full S-parameter response. The stakeholder is a receive-side RF troubleshooter, deciding which failed "
    "specification and band to check next, and what to hand to the component designer. I use the contrastive "
    "principle: why does this design fail the specification, rather than pass like the reference?")))

# 1b ADS hierarchy: schematic row and layout row, big -> small
els = [TITLE("From the ADS test bench to the six inputs"),
       T(0.6, 1.4, 4.0, 0.3, "SCHEMATIC", size=12, color=COPPER, bold=True),
       I(FIG / "fig_ads_schematic.png", 0.6, 1.72, 8.9, 2.576, "ADS schematic hierarchy: the test bench contains the "
         "filter; the filter is two mirrored halves; one half holds coupled sections CLin1 (W 25.5, S 3.3, L 138.5 mil) "
         "and CLin2 (W 31.5, S 18, L 136 mil), mapped to inputs Wc1, S1, L1 and Wc2, S2, L2"),
       R(9.85, 1.95, 2.9, 2.1, TINT, "round"),
       T(10.1, 2.15, 2.45, 1.75, [[B("Six inputs")], "Width, gap and length of the two coupled-line sections: "
         "Wc1, S1, L1 and Wc2, S2, L2"], size=14, color=INK),
       T(0.6, 4.45, 4.0, 0.3, "LAYOUT", size=12, color=COPPER, bold=True),
       I(FIG / "fig_ads_layout.png", 0.6, 4.77, 12.1, 2.23, "ADS layout hierarchy: full layout, one half, and a "
         "close-up of CLin1 with L1 = 138.5 mil, Wc1 = 25.5 mil and gap S1 = 3.3 mil marked")]
slides.append(dict(bg=WHITE, els=els, notes=(
    "Here is where the six inputs come from. In the ADS schematic, the test bench holds the filter, the filter is two "
    "mirrored halves, and each half has two coupled-line sections. Each section has a width, a gap and a length: "
    "Wc1, S1 and L1, and Wc2, S2 and L2. The layout below shows the same chain physically. In the close-up, the gap "
    "S1 between the coupled strips is only 3.3 mil.")))

# 2 Interview
reqs = [("Start with the failed spec", "Which band, which limit, how many dB over"),
        ("Flag case-level quality", "An average agreement score says nothing about this one case"),
        ("Separate diagnosis from action", "Attribution is not a validated tuning recommendation")]
els = [TITLE("The engineer needs the failed spec before the attribution"),
       T(0.6, 1.35, 12.1, 0.45, "One RF engineer  ·  internet voice call, Nov 5  ·  daily work: transceivers, antennas, "
         "mixing; component dimensions rarely come up", size=14, color=INK2)]
for i, (hd, ds) in enumerate(reqs):
    y = 2.15 + i * 1.3
    els += NUM(i + 1, 0.6, y + 0.05)
    els += [T(1.25, y, 4.6, 0.45, hd, size=18, color=NAVY, bold=True),
            T(1.25, y + 0.47, 4.6, 0.7, ds, size=15, color=INK2)]
els += [I(FIG / "fig_spec_first.png", 6.25, 1.95, 6.5, 3.61, "S21 and S11 of validation design v2_validation_0063 "
          "against the specification mask; high-stopband rejection exceeds the -15 dB limit by 8.6 dB at 13.20 GHz"),
        T(6.25, 5.62, 6.5, 0.7, "Spec-first view, validation design v2_validation_0063: high-stopband rejection fails "
          "13.20–13.56 GHz, 8.6 dB over the −15 dB limit (PR-Net dashed, ADS solid)", size=11, color=MUTED),
        T(0.6, 6.55, 12.1, 0.45, "One interview: these are design requirements, not evidence that the tool already "
          "helps troubleshooting.", size=12, color=MUTED, italic=True)]
slides.append(dict(bg=WHITE, els=els, notes=(
    "On November 5th I interviewed one RF engineer by internet voice call. Their daily work is mostly transceivers, "
    "antennas and mixing; component dimensions rarely come up, so a ranking like 'L2 is important' does not tell "
    "them what to do next. They gave three requirements. First, start with the failed specification: which band, "
    "which limit, and by how many dB. The figure shows that view for one validation design: high-stopband "
    "rejection fails near 13.2 gigahertz, 8.6 dB over the limit. Second, give case-level quality information, "
    "because an average score says nothing about this one explanation. Third, separate a diagnostic attribution "
    "from a validated tuning recommendation. This is one interview, so these are design requirements, not yet "
    "evidence that the tool helps.")))

# 3 Methods
meth = [("2A78D6", "SHAP", "exact split of the gap to the reference, over all 512 input coalitions"),
        ("EB6834", "IG", "model gradients integrated along the straight reference-to-design path"),
        ("1BAF7A", "LIME-delta", "local linear fit; slope × distance to the reference")]
els = [TITLE("All three methods name L1 first, then disagree"),
       I(FIG / "fig_three_methods.png", 0.5, 1.5, 6.4, 4.07, "Bar chart of SHAP, IG and LIME-delta contributions to "
         "the predicted mask violation for design v2_validation_0063; all rank L1 first, IG ranks S2 second, SHAP and "
         "LIME rank L2 second, LIME gives 0.9 dB to an unused control"),
       T(0.6, 5.65, 6.2, 0.7, "Same case; target = predicted mask violation (dB) relative to a passing reference design",
         size=11, color=MUTED)]
for i, (c, nm, ds) in enumerate(meth):
    y = 1.6 + i * 0.95
    els += [R(7.3, y + 0.09, 0.26, 0.26, c),
            T(7.75, y, 5.0, 0.4, nm, size=17, color=NAVY, bold=True),
            T(7.75, y + 0.38, 5.0, 0.5, ds, size=14, color=INK2)]
els += [R(7.3, 4.6, 5.45, 1.05, TINT, "round"),
        T(7.55, 4.75, 5.0, 0.8, [[B("Case-level warning: "), ("LIME's local fit R² is only 0.74, and it gives "
          "0.9 dB to a control input the model barely uses.", {})]], size=15, color=INK),
        T(7.3, 5.9, 5.45, 0.75, "Across 40 validation designs, rank agreement with SHAP: IG 0.93, LIME-delta 0.85",
          size=14, color=INK2)]
slides.append(dict(bg=WHITE, els=els, notes=(
    "I explain one number: the predicted mask violation in dB, relative to a reference design that passes. SHAP "
    "splits that difference exactly, using all 512 input coalitions. Integrated gradients integrates the model's "
    "gradient along a straight path. LIME fits a local linear model, and I turn its slopes into contributions. On "
    "this case all three put L1 first, but they disagree on second place: IG favors S2, while SHAP and LIME favor L2, "
    "because the response is nonlinear and the methods answer different questions. LIME also gives almost 1 dB to a "
    "control input the model barely uses, and its local fit R-squared is only 0.74: exactly the case-level warning "
    "the engineer asked for. Across 40 validation designs, rank agreement with SHAP is 0.93 for "
    "IG and 0.85 for LIME.")))

# 4 Faithfulness
els = [TITLE("Top-ranked inputs close the gap faster than chance"),
       I(FIG / "fig_deletion.png", 0.5, 1.5, 6.0, 4.0, "Deletion curve for design v2_validation_0063: following the "
         "SHAP order, three input resets remove the whole output gap; 100 random orders remove it much more slowly"),
       T(0.6, 5.6, 5.8, 0.7, "Same case: reset inputs to the reference one by one, in the explainer's order",
         size=11, color=MUTED),
       dict(kind="table", x=7.0, y=1.65, w=5.75, h=2.2, size=15,
            rows=[["Explainer", "Deletion", "Insertion"], ["SHAP", "80%", "89%"], ["IG", "69%", "86%"],
                  ["LIME-delta", "64%", "93%"]]),
       T(7.0, 3.95, 5.75, 0.75, "Share of design–target pairs that beat 95% of 100 random orders (PR-Net, 40 "
         "validation designs, 3 spec targets)", size=11, color=MUTED)]
for i, (hd, ds) in enumerate([("Supports", "the explanations track what the model does; no method wins everywhere"),
                              ("Limits", "tests the model, not physics; not verifiably written down before results, "
                               "so the next test is pre-registered")]):
    y = 4.85 + i * 0.95
    els += [T(7.0, y, 1.3, 0.45, hd, size=15, color=COPPER, bold=True),
            T(8.3, y, 4.45, 0.9, ds, size=15, color=INK)]
slides.append(dict(bg=WHITE, els=els, notes=(
    "For faithfulness, I ask: do the top-ranked inputs really account for the model's output change? I reset inputs "
    "to the reference one at a time, in the explainer's order. A faithful order closes the gap quickly. On this case, "
    "three SHAP-ranked inputs close the whole gap; random orders, in gray, take much longer. Across 40 validation "
    "designs and three spec targets, SHAP beats 95 percent of random orders in 80 percent of cases for deletion and "
    "89 for insertion; IG and LIME are lower for deletion. So the explanations track the model, but no method wins "
    "everywhere. Two cautions: this tests the model, not physics, and I cannot prove it was written down beforehand, "
    "so my next test is pre-registered.")))

# 5 Next steps (dark)
steps = [("Known-answer test", "Wrap the frozen model with planted equations; pre-register what each method should report"),
         ("Spec-first case card", "Failed spec, band and dB first, with quality flags; review it with the same engineer"),
         ("Unusual conditions", "Retrain with a planted shortcut; check 3 cases in HFSS full-wave simulation")]
els = [TITLE("Next: test the explanations where the answer is known", color=WHITE)]
for i, (hd, ds) in enumerate(steps):
    x = 0.6 + i * 4.1
    els += [R(x, 1.9, 3.85, 2.85, "1E3A52", "round")]
    els += NUM(i + 1, x + 0.3, 2.2)
    els += [T(x + 0.3, 2.9, 3.3, 0.5, hd, size=19, color=WHITE, bold=True),
            T(x + 0.3, 3.45, 3.3, 1.15, ds, size=15, color=ICE)]
els += [T(0.6, 5.35, 12.1, 0.5, "Draft report Dec 2  ·  Final report and code Dec 9", size=15, color=ICE),
        T(0.6, 6.05, 12.1, 0.6, "Thank you", size=24, color=WHITE, font=HEAD, bold=True)]
slides.append(dict(bg=NAVY, els=els, notes=(
    "Next, I will test the explanations where the right answer is known in advance. I will wrap the frozen model with "
    "planted equations, such as an additive term, an interaction and a saturating term, and pre-register what each "
    "method should report. Then I will build the spec-first case card with quality flags and review it with the same "
    "engineer. Finally, for unusual conditions, I will retrain with a planted shortcut and check three cases in HFSS "
    "full-wave simulation. Thank you.")))


# ---------------- pptx writer ----------------
def rgb(h):
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
blank = prs.slide_layouts[6]
SHAPES = {"rect": MSO_SHAPE.RECTANGLE, "round": MSO_SHAPE.ROUNDED_RECTANGLE, "oval": MSO_SHAPE.OVAL}
AL = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}
AN = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}
for s in slides:
    sl = prs.slides.add_slide(blank)
    sl.background.fill.solid(); sl.background.fill.fore_color.rgb = rgb(s["bg"])
    for e in s["els"]:
        if e["kind"] == "shape":
            sh = sl.shapes.add_shape(SHAPES[e["shape"]], Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            sh.fill.solid(); sh.fill.fore_color.rgb = rgb(e["fill"]); sh.line.fill.background(); sh.shadow.inherit = False
            if e["shape"] == "round":
                sh.adjustments[0] = 0.08
        elif e["kind"] == "img":
            pic = sl.shapes.add_picture(e["path"], Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            pic._element.nvPicPr.cNvPr.set("descr", e["alt"])
        elif e["kind"] == "table":
            rows = e["rows"]
            gt = sl.shapes.add_table(len(rows), len(rows[0]), Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            tb = gt.table
            tb.columns[0].width = Inches(e["w"] * 0.42)
            for j in (1, 2):
                tb.columns[j].width = Inches(e["w"] * 0.29)
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    c = tb.cell(i, j)
                    c.fill.solid(); c.fill.fore_color.rgb = rgb(NAVY if i == 0 else (TINT if i % 2 == 0 else WHITE))
                    c.vertical_anchor = MSO_ANCHOR.MIDDLE
                    p = c.text_frame.paragraphs[0]
                    p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                    r = p.add_run(); r.text = val
                    r.font.size, r.font.name = Pt(e["size"]), BODY
                    r.font.bold = i == 0 or (i == 1)
                    r.font.color.rgb = rgb(WHITE if i == 0 else INK)
        else:
            tx = sl.shapes.add_textbox(Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            if e["name"]:
                tx.name = e["name"]
            tf = tx.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            tf.vertical_anchor = AN[e["anchor"]]
            for k, para in enumerate(e["paras"]):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                p.alignment = AL[e["align"]]
                p.space_after = Pt(e["space"])
                for text, o in para:
                    r = p.add_run(); r.text = text
                    f = r.font
                    f.size, f.name = Pt(e["size"]), e["font"]
                    f.bold = o.get("bold", e["bold"]); f.italic = o.get("italic", e["italic"])
                    f.color.rgb = rgb(o.get("color", e["color"]))
    sl.notes_slide.notes_text_frame.text = s["notes"]
prs.save(OUT)

# ---------------- preview renderer + overflow check ----------------
DPI = 110
FONTS = {("Calibri", False, False): "calibri.ttf", ("Calibri", True, False): "calibrib.ttf",
         ("Calibri", False, True): "calibrii.ttf", ("Calibri", True, True): "calibriz.ttf",
         ("Cambria", False, False): "cambria.ttc", ("Cambria", True, False): "cambriab.ttf",
         ("Cambria", False, True): "cambriai.ttf", ("Cambria", True, True): "cambriaz.ttf"}


def font(name, bold, italic, pt):
    return ImageFont.truetype("C:/Windows/Fonts/" + FONTS[(name, bold, italic)], int(round(pt * DPI / 72)))


def wrap(words_runs, maxw, e):
    """words_runs: list of (word, opts). Returns lines of [(word, opts)] using PIL metrics."""
    lines, cur, curw = [], [], 0
    for wd, o in words_runs:
        fnt = font(e["font"], o.get("bold", e["bold"]), o.get("italic", e["italic"]), e["size"])
        ww = fnt.getlength(wd + " ")
        if cur and curw + fnt.getlength(wd) > maxw:
            lines.append(cur); cur, curw = [], 0
        cur.append((wd, o, fnt)); curw += ww
    if cur:
        lines.append(cur)
    return lines


problems = []
for n, s in enumerate(slides):
    img = Image.new("RGB", (int(W * DPI), int(H * DPI)), "#" + s["bg"])
    dr = ImageDraw.Draw(img)
    P = lambda v: int(round(v * DPI))  # noqa: E731
    for e in s["els"]:
        box = (P(e["x"]), P(e["y"]), P(e["x"] + e["w"]), P(e["y"] + e["h"]))
        if e["kind"] == "shape":
            if e["shape"] == "oval":
                dr.ellipse(box, fill="#" + e["fill"])
            elif e["shape"] == "round":
                dr.rounded_rectangle(box, radius=P(0.08 * min(e["w"], e["h"])), fill="#" + e["fill"])
            else:
                dr.rectangle(box, fill="#" + e["fill"])
        elif e["kind"] == "img":
            im = Image.open(e["path"]).convert("RGB").resize((box[2] - box[0], box[3] - box[1]))
            img.paste(im, box[:2])
        elif e["kind"] == "table":
            rows = e["rows"]; rh = e["h"] / len(rows)
            cw = [e["w"] * 0.42, e["w"] * 0.29, e["w"] * 0.29]
            for i, row in enumerate(rows):
                xx = e["x"]
                for j, val in enumerate(row):
                    fill = NAVY if i == 0 else (TINT if i % 2 == 0 else WHITE)
                    dr.rectangle((P(xx), P(e["y"] + i * rh), P(xx + cw[j]), P(e["y"] + (i + 1) * rh)), fill="#" + fill,
                                 outline="#DDDDDD")
                    fnt = font(BODY, i <= 1, False, e["size"])
                    tw = fnt.getlength(val)
                    tx = P(xx + 0.1) if j == 0 else P(xx + cw[j] / 2) - tw / 2
                    dr.text((tx, P(e["y"] + i * rh + rh / 2) - fnt.size * 0.55), val, font=fnt,
                            fill="#" + (WHITE if i == 0 else INK))
                    xx += cw[j]
        else:
            maxw = P(e["w"])
            allines = []
            for para in e["paras"]:
                wr = [(wd, o) for text, o in para for wd in text.split(" ") if wd]
                allines.append(wrap(wr, maxw, e))
            lh = e["size"] * DPI / 72 * 1.2
            total = sum(len(ls) for ls in allines) * lh + (len(allines) - 1) * e["space"] * DPI / 72
            if total > P(e["h"]) + 1:
                problems.append(f"slide {n}: text overflow ({total / DPI:.2f}in > {e['h']}in): {e['paras'][0][0][0][:50]}")
            y0 = box[1] + {"t": 0, "m": (P(e["h"]) - total) / 2, "b": P(e["h"]) - total}[e["anchor"]]
            for ls in allines:
                for line in ls:
                    lw = sum(f.getlength(wd + " ") for wd, _, f in line) - line[-1][2].getlength(" ")
                    x0 = box[0] + (0 if e["align"] == "l" else (maxw - lw) / 2)
                    for wd, o, f in line:
                        dr.text((x0, y0), wd, font=f, fill="#" + o.get("color", e["color"]))
                        x0 += f.getlength(wd + " ")
                    y0 += lh
                y0 += e["space"] * DPI / 72
        if e["x"] < 0.45 or e["y"] < 0.3 or e["x"] + e["w"] > W - 0.45 or e["y"] + e["h"] > H - 0.3:
            problems.append(f"slide {n}: element near edge: {e['kind']} at {e['x']},{e['y']} {e['w']}x{e['h']}")
    img.save(PREV / f"slide-{n}.png")
words = sum(len(s["notes"].split()) for s in slides)
print("saved", OUT, "| notes words:", words, "| ~min at 140 wpm:", round(words / 140, 2))
print("\n".join(problems) or "no overflow/edge problems")
