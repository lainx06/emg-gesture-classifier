"""Processing and evaluation pipeline diagram.

Box widths are computed from the longest detail line so text never spills.
Branch positions are derived from the box grid so nothing is clipped.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK   = "#1a1a1a"
MUTED = "#6f6f6f"
BOX   = "#f0f0f0"
RED   = "#b5482a"
BLUE  = "#2c4f6b"

# Detail lines kept short so they fit inside the boxes.
STAGES = [
    ("Load",      ["GRABMyo records", "43 subjects", "3 sessions"]),
    ("Derive",    ["Bipolar channels", "F1\u2212F9, F5\u2212F13", "trim 0.5 s"]),
    ("Degrade",   ["2048 \u2192 1000 Hz", "10-bit quantize", "six conditions"]),
    ("Filter",    ["20 Hz high-pass", "4th-order", "causal only"]),
    ("Featurize", ["200 / 100 ms", "RMS MAV WL ZC", "100,233 windows"]),
]

BRANCHES = [
    ("Evaluate", BLUE, ["Train d1, test d29", "5 classifiers", "Wilcoxon + CIs"]),
    ("Deploy",   RED,  ["C++ on ATmega328P", "16 MHz, 2 KB SRAM", "latency + flash"]),
]

# geometry
BW, BH   = 2.55, 1.60
GAP      = 0.30
X0       = 0.35
Y_ROW    = 3.30
BR_W, BR_H = 3.30, 1.45
Y_BR     = 0.95
Y_TITLE  = 5.55

N = len(STAGES)
CHAIN_W = N * BW + (N - 1) * GAP
CHAIN_MID = X0 + CHAIN_W / 2
FIG_W = X0 + CHAIN_W + 0.35


def box(ax, x, y, w, h, title, lines, s, edge=INK, tcol=INK, fill=BOX, lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center",
            fontsize=s["stage"], fontweight="bold", color=tcol, zorder=3)
    for k, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.66 - k * 0.31, ln, ha="center",
                va="center", fontsize=s["detail"], color=MUTED, zorder=3)


def build(ax, s, show_title):
    if show_title:
        ax.text(X0, Y_TITLE, "What the pipeline does", ha="left",
                va="center", fontsize=s["title"], fontweight="bold", color=INK)

    for i, (title, lines) in enumerate(STAGES):
        x = X0 + i * (BW + GAP)
        box(ax, x, Y_ROW, BW, BH, title, lines, s)
        if i < N - 1:
            ax.add_patch(FancyArrowPatch(
                (x + BW + 0.04, Y_ROW + BH / 2),
                (x + BW + GAP - 0.04, Y_ROW + BH / 2),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.6,
                color=INK, zorder=3))

    # split from the last stage down to the branch bus
    last_mid = X0 + (N - 1) * (BW + GAP) + BW / 2
    bus_y = Y_BR + BR_H + 0.50
    ax.add_patch(FancyArrowPatch((last_mid, Y_ROW - 0.04), (last_mid, bus_y),
                                 arrowstyle="-", linewidth=1.7, color=INK, zorder=3))

    total_br = 2 * BR_W + 0.90
    bx0 = min(last_mid - total_br / 2, FIG_W - total_br - 0.35)
    bx0 = max(bx0, X0)

    left_mid = bx0 + BR_W / 2
    right_mid = bx0 + BR_W + 0.90 + BR_W / 2
    ax.add_patch(FancyArrowPatch((left_mid, bus_y), (right_mid, bus_y),
                                 arrowstyle="-", linewidth=1.7, color=INK, zorder=3))

    for j, (title, col, lines) in enumerate(BRANCHES):
        bx = bx0 + j * (BR_W + 0.90)
        mid = bx + BR_W / 2
        ax.add_patch(FancyArrowPatch((mid, bus_y - 0.02), (mid, Y_BR + BR_H + 0.04),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.7, color=col, zorder=3))
        box(ax, bx, Y_BR, BR_W, BR_H, title, lines, s,
            edge=col, tcol=col, fill="white", lw=1.9)

    ax.text(X0, 0.35,
            "Every condition derived from the same recordings, so all "
            "comparisons are exactly paired per subject.",
            ha="left", va="center", fontsize=s["note"], color=INK)


def render(path, figsize, dpi, s, show_title=True):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, Y_TITLE + 0.45)
    ax.axis("off")
    build(ax, s, show_title)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    render("slide_methods.png", (13.333, 6.6), 200,
           s={"title": 21, "stage": 15, "detail": 10, "note": 11.5},
           show_title=True)
    render("fig_pipeline.pdf", (7.16, 3.6), 300,
           s={"title": 11, "stage": 8.5, "detail": 5.8, "note": 6.5},
           show_title=False)