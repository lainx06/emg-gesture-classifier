"""Slide 4 / paper figure: degradation axis attribution.

Two panels sharing a condition axis. Left = Day 1 baseline accuracy relative
to clinical; right = 28-day accuracy loss relative to clinical. Signal-axis
conditions are muted, spatial conditions are highlighted, so the pattern
reads before anyone parses a number or an axis direction.
"""

import os

# Output goes to the figures/ directory at the project root, regardless of
# which directory this script is run from.
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK    = "#1a1a1a"
MUTED  = "#8a8a8a"
GREY   = "#6e6e6e"
RED    = "#b5482a"
GRID   = "#dcdcdc"

# label, (day1 mean, lo, hi), (drop mean, lo, hi), is_spatial
DATA = [
    ("Rate only",     (+0.16, -0.06, +0.58),   (-0.18, -1.78, +0.48),   False),
    ("Depth only",    (-0.09, -0.27, +0.00),   (-0.10, -0.29, +0.03),   False),
    ("Rate + depth",  (+0.18, -0.01, +0.72),   (-0.27, -1.88, +0.34),   False),
    ("Spatial only",  (-11.10, -14.67, -8.31), (+9.09, +3.32, +14.66),  True),
    ("Commodity",     (-10.23, -14.82, -7.25), (+8.57, +2.78, +14.38),  True),
]


def panel(ax, values, title, xlabel, xlim, s):
    n = len(DATA)
    for i, ((label, d1, dr, spatial), vals) in enumerate(zip(DATA, values)):
        y = n - 1 - i
        mean, lo, hi = vals
        color = RED if spatial else GREY
        lw = 3.2 if spatial else 2.4
        ms = 11 if spatial else 8

        ax.plot([lo, hi], [y, y], color=color, linewidth=lw,
                solid_capstyle="round", zorder=3)
        ax.plot([lo, lo], [y - 0.13, y + 0.13], color=color, linewidth=lw * 0.7, zorder=3)
        ax.plot([hi, hi], [y - 0.13, y + 0.13], color=color, linewidth=lw * 0.7, zorder=3)
        ax.plot([mean], [y], "o", color=color, markersize=ms,
                markeredgecolor="white", markeredgewidth=1.6, zorder=4)

    ax.axvline(0, color=INK, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_yticks(range(n))
    ax.set_yticklabels([d[0] for d in DATA][::-1], fontsize=s["tick"], color=INK)
    for lbl, (_, _, _, spatial) in zip(ax.get_yticklabels()[::-1], DATA):
        if spatial:
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.tick_params(axis="x", labelsize=s["tick"], colors=INK, length=4)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel, fontsize=s["axlabel"], color=INK, labelpad=8)
    ax.set_title(title, fontsize=s["panel"], fontweight="bold",
                 color=INK, pad=12)

    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def render(path, figsize, dpi, s, suptitle=None):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True,
                             gridspec_kw={"wspace": 0.10})

    panel(axes[0], [d[1] for d in DATA],
          "Day 1 accuracy", "points vs. clinical   (negative = worse)",
          (-17.5, 4.0), s)
    panel(axes[1], [d[2] for d in DATA],
          "28-day accuracy loss", "points vs. clinical   (positive = worse)",
          (-4.0, 17.5), s)

    if suptitle:
        fig.suptitle(suptitle, fontsize=s["title"], fontweight="bold",
                     color=INK, x=0.02, ha="left", y=0.99)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    render(os.path.join(OUT, "slide4_ablation.png"), (13.333, 7.0), 200,
           s={"title": 21, "panel": 17, "axlabel": 13, "tick": 14, "note": 12},
           suptitle="Only channel count matters")

    render(os.path.join(OUT, "fig_ablation.pdf"), (7.16, 3.2), 300,
           s={"title": 11, "panel": 9.5, "axlabel": 8, "tick": 8.5, "note": 7.5})