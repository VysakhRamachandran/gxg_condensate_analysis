"""Plot position-resolved sticker engagement."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
})

SYS_ORDER = ["9E", "9C", "9A", "9P", "9N", "X2", "X2S", "X3", "X3S"]
SYS_CLASS_COLOR = {**{s: "#a1d99b" for s in ["9E", "9C", "9A", "9P", "9N"]},
                   **{s: "#fcae91" for s in ["X2", "X2S", "X3", "X3S"]}}
HIGHLIGHT = {"X3S": "#d62728", "X3": "#ff7f0e"}


def plot_per_position_bars(summary):
    """One panel per multi-sticker system showing per-position engagement."""
    pat_systems = ["X2", "X2S", "X3", "X3S"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, sys in zip(axes, pat_systems):
        sub = summary[summary["system"] == sys].copy()
        sub = sub.sort_values("position")
        x = np.arange(len(sub))
        ax.bar(x, sub["mean_partners"].values,
               yerr=sub["sem_partners"].values,
               color=HIGHLIGHT.get(sys, SYS_CLASS_COLOR[sys]),
               edgecolor="k", linewidth=0.5, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"pos {int(p)}" for p in sub["position"]])
        ax.set_title(f"{sys}", fontsize=11, fontweight="bold",
                     color=HIGHLIGHT.get(sys, "k"))
        for xi, v in zip(x, sub["mean_partners"].values):
            ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
        # also annotate engagement fraction below x label
        for xi, fr in zip(x, sub["engaged_frac"].values):
            ax.text(xi, -0.15, f"{fr*100:.0f}% engaged",
                    ha="center", fontsize=8, color="dimgray",
                    transform=ax.get_xaxis_transform())
    axes[0].set_ylabel("mean partner peptides per sticker")
    fig.suptitle("Per-position sticker engagement (patterned systems)\n"
                 "X3 stickers (4,5,6 adjacent) vs X3S stickers (2,5,8 spread)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "15_sticker_engagement_bars.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 15_sticker_engagement_bars.png")


def plot_position_in_chain(summary):
    """Aggregate by absolute position in the 9-residue chain."""
    df = summary[summary["system"].isin(SYS_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for sys in SYS_ORDER:
        sub = df[df["system"] == sys].sort_values("position")
        color = HIGHLIGHT.get(sys, SYS_CLASS_COLOR[sys])
        marker = "D" if sys in HIGHLIGHT or sys.startswith("X") else "s"
        ax.plot(sub["position"], sub["mean_partners"], marker=marker,
                color=color,
                ms=10 if sys in HIGHLIGHT else 8,
                lw=2.2 if sys in HIGHLIGHT else 1.4,
                markeredgecolor="k",
                markeredgewidth=0.8 if sys in HIGHLIGHT else 0.4,
                label=sys, zorder=4 if sys in HIGHLIGHT else 2)
    ax.set_xlabel("sticker position within 9-residue chain")
    ax.set_ylabel("mean partner peptides per sticker")
    ax.set_xticks(range(1, 10))
    ax.set_title("Position-resolved sticker engagement\n"
                 "(higher = sticker reaches more partner peptides)")
    ax.legend(ncol=3, fontsize=9, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "16_position_in_chain.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 16_position_in_chain.png")


def plot_central_vs_terminal(summary):
    """For X3/X3S, compare central sticker(s) vs terminal sticker(s)."""
    rows = []
    # X3: 4,5,6 — central is 5, terminals are 4 and 6
    # X3S: 2,5,8 — central is 5, terminals are 2 and 8
    for sys in ["X3", "X3S"]:
        sub = summary[summary["system"] == sys].set_index("position")
        center = sub.loc[5, "mean_partners"]
        terminals = sub.drop(5)["mean_partners"].mean()
        rows.append({"system": sys, "center_pos5": center,
                     "terminal_avg": terminals,
                     "ratio_center_vs_term": center / terminals})
    df = pd.DataFrame(rows)
    print("\nCentral vs terminal sticker engagement:")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.2, 4))
    width = 0.35
    x = np.arange(len(df))
    ax.bar(x - width/2, df["center_pos5"], width,
           color=["#ff7f0e", "#d62728"], edgecolor="k",
           label="central sticker (pos 5)")
    ax.bar(x + width/2, df["terminal_avg"], width,
           color="#bbbbbb", edgecolor="k",
           label="terminal stickers (avg)")
    for xi, c, t in zip(x, df["center_pos5"], df["terminal_avg"]):
        ax.text(xi - width/2, c + 0.05, f"{c:.2f}", ha="center", fontsize=9)
        ax.text(xi + width/2, t + 0.05, f"{t:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(df["system"], fontsize=11,
                                          fontweight="bold")
    ax.set_ylabel("mean partner peptides per sticker")
    ax.set_title("Central vs terminal sticker engagement\n"
                 "(does the middle sticker get buried?)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "17_central_vs_terminal.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 17_central_vs_terminal.png")
    return df


def plot_diversification(div_summary, summary):
    """Two-panel: per-peptide union vs sum, and diversification ratio."""
    df = div_summary.set_index("system")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # ---- left: union (peptide degree) vs sum (sum of per-sticker partners)
    pat_order = ["X2", "X2S", "X3", "X3S"]
    n_stickers = {"X2": 2, "X2S": 2, "X3": 3, "X3S": 3}
    x = np.arange(len(pat_order))
    width = 0.38
    union_vals = [df.loc[s, "union_partners"] for s in pat_order]
    sum_vals   = [df.loc[s, "sum_partners"] for s in pat_order]
    colors = [HIGHLIGHT.get(s, SYS_CLASS_COLOR[s]) for s in pat_order]
    ax1.bar(x - width/2, sum_vals, width, color="#cccccc", edgecolor="k",
            label="sum across stickers")
    ax1.bar(x + width/2, union_vals, width, color=colors, edgecolor="k",
            label="union (distinct partner peptides)")
    for xi, su, un, sys in zip(x, sum_vals, union_vals, pat_order):
        ax1.text(xi - width/2, su + 0.15, f"{su:.1f}", ha="center",
                 fontsize=9)
        ax1.text(xi + width/2, un + 0.15, f"{un:.1f}", ha="center",
                 fontsize=9, fontweight="bold",
                 color=HIGHLIGHT.get(sys, "k"))
    ax1.set_xticks(x); ax1.set_xticklabels(pat_order, fontweight="bold")
    ax1.set_ylabel("partner peptides")
    ax1.set_title("Sum across stickers vs union\n"
                  "(gap = redundancy; small gap = diversified contacts)")
    ax1.legend(frameon=False, fontsize=9)

    # ---- right: diversification ratio & jaccard
    ratio_vals = [df.loc[s, "diversification_ratio"] for s in pat_order]
    jac_vals   = [df.loc[s, "mean_jaccard"] for s in pat_order]
    ax2.bar(x - width/2, ratio_vals, width,
            color=colors, edgecolor="k",
            label="diversification ratio (union/sum)")
    ax2.bar(x + width/2, jac_vals, width,
            color="#aaaaaa", edgecolor="k",
            label="mean Jaccard overlap between sticker pairs")
    for xi, r, j in zip(x, ratio_vals, jac_vals):
        ax2.text(xi - width/2, r + 0.01, f"{r:.2f}", ha="center",
                 fontsize=9, fontweight="bold")
        ax2.text(xi + width/2, j + 0.01, f"{j:.2f}", ha="center",
                 fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(pat_order, fontweight="bold")
    ax2.set_ylabel("dimensionless")
    ax2.set_title("Partner-set diversification across stickers\n"
                  "(higher ratio = more distinct partners; lower Jaccard = less overlap)")
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.set_ylim(0, 1.05)

    fig.suptitle("X3S advantage comes from partner DIVERSIFICATION, "
                 "not higher per-sticker engagement",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "18_diversification.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 18_diversification.png")


if __name__ == "__main__":
    summary = pd.read_csv(os.path.join(DATA, "15_sticker_engagement_summary.csv"))
    div_summary = pd.read_csv(
        os.path.join(DATA, "15_sticker_diversification_summary.csv"))
    pd.set_option("display.float_format", "{:.2f}".format)
    print(summary.to_string(index=False))
    print("\n", div_summary.to_string(index=False))
    plot_per_position_bars(summary)
    plot_position_in_chain(summary)
    plot_central_vs_terminal(summary)
    plot_diversification(div_summary, summary)
    print("Done.")
