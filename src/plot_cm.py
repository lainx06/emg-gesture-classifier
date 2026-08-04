"""Generate 2x2 confusion matrix figure for Clinical vs Commodity conditions."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from confusion_matrices import get_cm   # single source of truth

# Sized for full text width of a two-column IEEE page (7.16 in).
# Include with \includegraphics[width=\textwidth]{...} so it renders 1:1.
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,      # embed TrueType, not Type 3 (IEEE requires this)
})


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, "cache")
    fig_dir = os.path.join(base_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    configs = [
        ("clinical",  1, "Clinical, Day 1"),
        ("clinical",  3, "Clinical, Day 29"),
        ("commodity", 1, "Commodity, Day 1"),
        ("commodity", 3, "Commodity, Day 29"),
    ]
    labels = ["Extension", "Close", "Rest"]

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.6))
    axes = axes.flatten()

    for ax, (cond, sess, title) in zip(axes, configs):
        print(f"Processing {title}...")
        cm = get_cm(os.path.join(cache_dir, f"{cond}_cache.npz"), sess)
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar=False, vmin=0, vmax=100, square=True,
                    linewidths=0.5, linecolor="white",
                    annot_kws={"size": 8})

        acc = np.trace(cm) / cm.sum() * 100
        ax.set_title(f"{title}  ({acc:.1f}%)", fontweight="bold")
        ax.set_ylabel("True class")
        ax.set_xlabel("Predicted class")

    fig.tight_layout()
    out_path = os.path.join(fig_dir, "confusion_matrices.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")


if __name__ == "__main__":
    main()