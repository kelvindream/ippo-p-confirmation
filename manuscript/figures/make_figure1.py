#!/usr/bin/env python3
"""Render Figure 1 (method schematic) for the IPPO-P manuscript.

Run: env -u LD_LIBRARY_PATH python3 make_figure1.py   (from paper/figures/)
Output: figure1_method_schematic.pdf / .svg / .png (300 dpi)

Design contract
- Core conclusion: the intervention is an additive, scaled pressure prior on
  the actor logits, applied identically during rollout sampling and the PPO
  update; actor/critic architecture, reward, and observations are unchanged.
- Archetype: schematic-led composite (single panel, three horizontal bands).
- Colour roles: blue = computation/flow, orange = the intervention (pressure
  prior and training-consistency links), grey = context and stored state.
- Spec source: draft.md Figure 1 caption + Sections 3.3-3.4 + the frozen
  method lock (pressure_zscore_hold: per-intersection centering, SD floor 1,
  clip to [-3, 3], scale 2.25, temperature 1.0).
- Glyph floor: every rendered glyph (including mathtext subscripts) stays
  >= 5 pt; all math-bearing text uses >= 7.2 pt parents.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",   # editable text in SVG
    "pdf.fonttype": 42,       # editable TrueType text in PDF
    "font.size": 7,
})

# Okabe-Ito derived roles
BLUE = "#0072B2"; ORANGE = "#D55E00"; GREY = "#5A5A5A"
BAND_ROLLOUT = "#F0F6FB"; BAND_SHARED = "#FFF4E8"; BAND_UPDATE = "#F5F5F5"

fig, ax = plt.subplots(figsize=(7.3, 4.0))
ax.set_xlim(0, 104); ax.set_ylim(0, 40.5); ax.axis("off")

# ---------------- background bands ----------------
ax.add_patch(Rectangle((0, 26.2), 104, 12.4, fc=BAND_ROLLOUT, ec="none", zorder=0))
ax.add_patch(Rectangle((0, 18.0), 104, 7.8, fc=BAND_SHARED, ec="none", zorder=0))
ax.add_patch(Rectangle((0, 2.0), 104, 15.8, fc=BAND_UPDATE, ec="none", zorder=0))

def box(x, y, w, h, text, fc="white", ec=BLUE, fs=7.2, dashed=False, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.55",
                                fc=fc, ec=ec, lw=lw,
                                linestyle="--" if dashed else "-", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="black", linespacing=1.35, zorder=4)

def arrow(x1, y1, x2, y2, color=BLUE, style="-", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=lw,
                                 linestyle=style,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))

# ---------------- band titles + headline ----------------
ax.text(52, 39.5, "Actor and critic networks unchanged; the prior is an "
        "additive logit shift, not a new network", ha="center",
        fontsize=7.2, style="italic", color=GREY, zorder=4)
ax.text(2, 36.7, "Rollout (sampling)", fontsize=8, color=BLUE,
        fontweight="bold", zorder=4)
ax.text(56.5, 21.9, "same $p_t$ feeds both lanes", fontsize=7.2, color=ORANGE,
        style="italic", ha="left", va="center", zorder=4)
ax.text(2, 15.5, "PPO update", fontsize=8, color=BLUE, fontweight="bold",
        zorder=4)

# ---------------- rollout band ----------------
box(3.2, 27.5, 15, 6, "Observation\n$o_t$ (unchanged)")
box(22.6, 27.5, 16.6, 6, "Actor logits $z$\n(architecture\nunchanged)", fs=7.0)

# hero junction: scaled prior added to logits
circ = Circle((45.25, 30.5), 2.1, fc="white", ec=ORANGE, lw=1.4, zorder=4)
ax.add_patch(circ)
ax.text(45.25, 30.42, "+", ha="center", va="center", fontsize=11,
        color=ORANGE, fontweight="bold", zorder=5)

box(51.1, 27.5, 27.5, 6, "Action distribution $\\pi(a\\,|\\,o_t)$\n"
    "$=\\mathrm{softmax}(z + 2.25\\,p_t)$, $T=1.0$")
box(82.8, 27.5, 16.4, 6, "sample $a_t$;\nstore $\\log\\pi_{old}$", fc="#FBFBFB",
    ec=GREY, dashed=True)

# flush connectors: each arrow spans exactly the gap between box edges
arrow(18.75, 30.5, 22.05, 30.5)          # obs -> logits
arrow(39.75, 30.5, 43.15, 30.5)          # logits -> (+)
arrow(47.35, 30.5, 50.55, 30.5)          # (+) -> distribution
arrow(79.15, 30.5, 82.25, 30.5)          # distribution -> sample

# ---------------- shared prior strip ----------------
box(2, 18.4, 16.5, 5.8, "Pressure state\n(incoming $-$ outgoing\nhalting queues)", fc="#FBFBFB",
    ec=GREY, fs=6.2)
box(22.6, 18.4, 30, 5.8, "Pressure prior module $p_t$\n"
    "per-intersection centering,\n"
    "population-SD scaling (floor 1),\n"
    "clip $[-3,3]$ (non-binding, $n{=}4$)",
    fc="white", ec=ORANGE, fs=6.6)

arrow(18.95, 21.3, 22.05, 21.3, GREY)    # pressure state -> module
arrow(45.25, 24.85, 45.25, 28.35, ORANGE)  # module -> (+); scale shown as 2.25 p_t in the distribution formula

# ---------------- update band ----------------
box(26.5, 6.0, 25.5, 6, "Stored rollout tuple\n$(o_t,\\,p_t,\\,a_t,\\,"
    "\\log\\pi_{old})$")
box(56.7, 6.0, 17, 6, "$\\log\\pi_{new}(a_t\\,|\\,o_t)$\nsame prior path")
box(77.9, 6.0, 22.5, 6, "Clipped-ratio objective\n$\\min(r_t A_t,\\,"
    "\\mathrm{clip}(r_t,1{\\pm}\\epsilon)A_t)$")

arrow(45.25, 17.8, 45.25, 12.65, ORANGE, style=(0, (3, 2)), lw=1.1)  # p_t reuse
ax.text(46.1, 15.2, "reuse $p_t$", fontsize=7.2, color=ORANGE, ha="left",
        va="center", zorder=4)
arrow(52.55, 9, 56.15, 9)                # tuple -> new logp
arrow(74.75, 9, 77.35, 9)                # new logp -> PPO loss

# training-consistency link: stored (a_t, old logp) + same prior -> ratio
arrow(91, 26.9, 69, 12.7, ORANGE, style=(0, (2, 2)), lw=1.0, rad=-0.25)
ax.text(92, 15.8, "identical prior,\ntemperature, action", fontsize=6.8,
        color=ORANGE, ha="center", va="center", linespacing=1.3, zorder=4)

for ext in ("pdf", "svg", "png"):
    fig.savefig(f"figure1_method_schematic.{ext}", dpi=300,
                bbox_inches="tight")
print("figure1_method_schematic.pdf/.svg/.png written")
