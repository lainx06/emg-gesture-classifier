"""Six-condition ablation design matrix.

PNG for the presentation (16:9), PDF for the paper (single column).
Source box is sized to its content so the three lines never collide.
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK      = "#1a1a1a"
MUTED    = "#7a7a7a"
CLEAN_BG = "#f0f0f0"
DEG_BG   = "#b5482a"
DEG_FG   = "#ffffff"
HEAD_BG  = "#1a1a1a"
HEAD_FG  = "#ffffff"
ROW_HL   = "#f7e4dd"

ROWS = [
    ("Clinical",     ("8", "2048 Hz", "float"),  (0, 0, 0)),
    ("Rate only",    ("8", "1 kHz",   "float"),  (0, 1, 0)),
    ("Depth only",   ("8", "2048 Hz", "10-bit"), (0, 0, 1)),
    ("Rate + depth", ("8", "1 kHz",   "10-bit"), (0, 1, 1)),
    ("Spatial only", ("2", "2048 Hz", "float"),  (1, 0, 0)),
    ("Commodity",    ("2", "1 kHz",   "10-bit"), (1, 1, 1)),
]
COLS = ["Channels", "Sample rate", "Bit depth"]

X0      = 0.50
LABEL_W = 2.60
CELL_W  = 2.20
GAP     = 0.10
TABLE_W = LABEL_W + 3 * (CELL_W + GAP) - GAP

# vertical layout, absolute
Y_TITLE   = 9.15
SRC_Y     = 7.10          # bottom of source box
SRC_H     = 1.55          # tall enough for 3 comfortably spaced lines
ARROW_TOP = SRC_Y - 0.10
ARROW_BOT = 6.25
Y_HEAD    = 5.50
CELL_H    = 0.72
ROW_GAP   = 0.10
Y_LEGEND  = Y_HEAD - 6 * (CELL_H + ROW_GAP) - 0.42
Y_FOOT    = Y_LEGEND - 0.42
Y_BOTTOM  = Y_FOOT - 0.38


def cell(ax, x, y, w, h, fc, radius=0.06):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor="white", linewidth=1.6, zorder=2))


def build(ax, s_title, s_head, s_body, s_label, s_note, show_title):
    if show_title:
        ax.text(X0, Y_TITLE, "Six conditions, three axes",
                ha="left", va="center", fontsize=s_title,
                fontweight="bold", color=INK)

    # source box: three evenly spaced lines inside a taller frame
    ax.add_patch(FancyBboxPatch(
        (X0, SRC_Y), TABLE_W, SRC_H,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor="white", edgecolor=INK, linewidth=1.8, zorder=2))
    cx = X0 + TABLE_W / 2
    line_y = [SRC_Y + SRC_H * f for f in (0.76, 0.48, 0.20)]
    ax.text(cx, line_y[0], "GRABMyo", ha="center", va="center",
            fontsize=s_head, fontweight="bold", color=INK, zorder=3)
    ax.text(cx, line_y[1], "43 subjects   \u00b7   recorded on days 1, 8, and 29",
            ha="center", va="center", fontsize=s_body, color=INK, zorder=3)
    ax.text(cx, line_y[2], "8 channels   \u00b7   2048 Hz   \u00b7   floating point",
            ha="center", va="center", fontsize=s_body - 1, color=MUTED, zorder=3)

    ax.add_patch(FancyArrowPatch(
        (cx, ARROW_TOP), (cx, ARROW_BOT), arrowstyle="-|>",
        mutation_scale=16, linewidth=1.8, color=INK, zorder=3))
    ax.text(cx + 0.22, (ARROW_TOP + ARROW_BOT) / 2,
            "degrade one axis at a time", ha="left", va="center",
            fontsize=s_note, style="italic", color=MUTED, zorder=3)

    # header
    cell(ax, X0, Y_HEAD, LABEL_W, CELL_H, HEAD_BG)
    ax.text(X0 + 0.22, Y_HEAD + CELL_H / 2, "Condition", ha="left", va="center",
            fontsize=s_head - 1, fontweight="bold", color=HEAD_FG, zorder=3)
    for j, cname in enumerate(COLS):
        x = X0 + LABEL_W + GAP + j * (CELL_W + GAP)
        cell(ax, x, Y_HEAD, CELL_W, CELL_H, HEAD_BG)
        ax.text(x + CELL_W / 2, Y_HEAD + CELL_H / 2, cname, ha="center",
                va="center", fontsize=s_head - 1, fontweight="bold",
                color=HEAD_FG, zorder=3)

    # rows
    for i, (label, vals, degs) in enumerate(ROWS):
        y = Y_HEAD - (i + 1) * (CELL_H + ROW_GAP)
        two_ch = degs[0] == 1
        cell(ax, X0, y, LABEL_W, CELL_H, ROW_HL if two_ch else CLEAN_BG)
        ax.text(X0 + 0.22, y + CELL_H / 2, label, ha="left", va="center",
                fontsize=s_label, fontweight="bold" if two_ch else "normal",
                color=INK, zorder=3)
        for j, (v, d) in enumerate(zip(vals, degs)):
            x = X0 + LABEL_W + GAP + j * (CELL_W + GAP)
            cell(ax, x, y, CELL_W, CELL_H, DEG_BG if d else CLEAN_BG)
            ax.text(x + CELL_W / 2, y + CELL_H / 2, v, ha="center", va="center",
                    fontsize=s_body, fontweight="bold" if d else "normal",
                    color=DEG_FG if d else INK, zorder=3)

    # legend and footer on separate lines
    sw = 0.34
    ax.add_patch(FancyBboxPatch((X0, Y_LEGEND - 0.11), sw, 0.22,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=DEG_BG, edgecolor="none", zorder=3))
    ax.text(X0 + sw + 0.16, Y_LEGEND, "degraded from the clinical baseline",
            ha="left", va="center", fontsize=s_note, color=MUTED, zorder=3)
    ax.text(X0, Y_FOOT,
            "All six conditions derived from the same recordings   \u00b7   "
            "train day 1, test day 29",
            ha="left", va="center", fontsize=s_note, color=INK, zorder=3)


def render(path, figsize, dpi, sizes, show_title=True):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, X0 + TABLE_W + 0.5)
    ax.set_ylim(Y_BOTTOM, Y_TITLE + 0.5)
    ax.axis("off")
    build(ax, *sizes, show_title=show_title)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    render(os.path.join(OUT, "slide3_design.png"), (13.333, 7.5), 200,
           sizes=(21, 15, 14, 14, 12), show_title=True)
    render(os.path.join(OUT, "fig_design.pdf"), (7.16, 4.6), 300,
           sizes=(11, 9.5, 9, 9, 8), show_title=False)