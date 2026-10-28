"""Money plot: sticker multivalency (left) and the network->dynamics link (right).

Combines the two halves of the network result into one figure.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

from .observables import all_systems_observables

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

ORDER = ["3E", "3C", "3A", "3P", "3N",
         "9E", "9C", "9A", "9P", "9N",
         "X2", "X2S", "X3", "X3S"]


def sys_class_color(sys):
    if sys.startswith("3"):  return "#9ecae1"
    if sys.startswith("9"):  return "#a1d99b"
    return "#fcae91"

HIGHLIGHT = "#d62728"  # X3S
HIGHLIGHT2 = "#ff7f0e"  # X3
HIGHLIGHTS = {"X3S": HIGHLIGHT, "X3": HIGHLIGHT2}


def sys_marker(sys):
    if sys.startswith("3"): return "o"
    if sys.startswith("9"): return "s"
    return "D"


def fig_money():
    metrics = pd.read_csv(os.path.join(DATA, "10_network_metrics.csv"),
                          index_col="system").reindex(ORDER)
    obs = all_systems_observables().drop(columns=["chain", "is_pattern"])
    df = metrics.join(obs)

    fig = plt.figure(figsize=(12.4, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.28)

    # --- LEFT: sticker-mean-degree bars ---------------------------------
    ax1 = fig.add_subplot(gs[0])
    vals = metrics["sticker_mean_degree"].values
    colors = [HIGHLIGHTS.get(s, sys_class_color(s)) for s in metrics.index]
    edgecolors = ["k" if s not in HIGHLIGHTS else "k"
                  for s in metrics.index]
    lw = [0.6 if s not in HIGHLIGHTS else 1.4 for s in metrics.index]
    bars = ax1.bar(range(len(metrics)), vals, color=colors,
                   edgecolor=edgecolors, linewidth=lw)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics.index, fontsize=10)
    ax1.set_ylabel("mean sticker partners per peptide",
                   fontsize=11)
    ax1.set_title("(a) Sticker-restricted multivalency", fontsize=11)
    # annotate X3S — placed inside the empty mid-range of the y-axis
    x3s_i = list(metrics.index).index("X3S")
    ax1.set_ylim(0, max(vals) * 1.22)
    ax1.annotate(f"X3S: {vals[x3s_i]:.1f} partners\n(~2× X2/X2S)",
                 xy=(x3s_i, vals[x3s_i]),
                 xytext=(6.5, vals[x3s_i] * 0.55),
                 fontsize=10, color=HIGHLIGHT, ha="left",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=HIGHLIGHT,
                                 lw=1.2, connectionstyle="arc3,rad=-0.15"))
    # group dividers
    for x in [4.5, 9.5]:
        ax1.axvline(x, color="grey", lw=0.4, ls=":")
    ymax = ax1.get_ylim()[1]
    ax1.text(2, ymax * 0.98, "3-residue", ha="center",
             color="grey", fontsize=9)
    ax1.text(7, ymax * 0.98, "9-residue", ha="center",
             color="grey", fontsize=9)
    ax1.text(11.5, ymax * 0.98, "patterned", ha="center",
             color="grey", fontsize=9)
    # value labels above each bar
    for x, v in enumerate(vals):
        ax1.text(x, v + 0.07, f"{v:.1f}", ha="center", fontsize=8,
                 color="dimgray")

    # --- RIGHT: percolation_pc vs tau_pw_eff scatter --------------------
    ax2 = fig.add_subplot(gs[1])
    x = df["percolation_pc"].values
    y = df["tau_pw_eff"].values
    mask = np.isfinite(x) & np.isfinite(y)
    for s, xx, yy in zip(df.index[mask], x[mask], y[mask]):
        is_hl = s in HIGHLIGHTS
        ax2.scatter(xx, yy,
                    color=HIGHLIGHTS.get(s, sys_class_color(s)),
                    marker=sys_marker(s),
                    s=180 if is_hl else 95,
                    edgecolors="k",
                    linewidths=1.2 if is_hl else 0.5,
                    zorder=4 if is_hl else 3)
        ax2.annotate(s, (xx, yy),
                     xytext=(7, 4), textcoords="offset points",
                     fontsize=10 if is_hl else 9,
                     fontweight="bold" if is_hl else "normal",
                     color=HIGHLIGHTS.get(s, "k"))
    # linear regression line
    coef = np.polyfit(x[mask], y[mask], 1)
    xline = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax2.plot(xline, np.polyval(coef, xline), "--", color="grey", lw=1,
             zorder=1)
    r, p = spearmanr(x[mask], y[mask])
    ax2.text(0.04, 0.94,
             f"Spearman ρ = {r:+.2f}\np = {p:.1e}",
             transform=ax2.transAxes, fontsize=11, va="top",
             bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85,
                       boxstyle="round,pad=0.3"))
    ax2.set_xlabel("network robustness  (percolation pₒ, "
                   "fraction of edges removed before LCC < N/2)",
                   fontsize=10)
    ax2.set_ylabel(r"peptide–water contact lifetime $\tau_{\rm p-w}$  (ns)",
                   fontsize=11)
    ax2.set_title("(b) Network robustness $\\to$ water dynamics",
                  fontsize=11)

    # legend for marker shape
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ecae1",
               markeredgecolor="k", markersize=10, label="3-residue"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#a1d99b",
               markeredgecolor="k", markersize=10, label="9-residue"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#fcae91",
               markeredgecolor="k", markersize=10, label="patterned"),
    ]
    ax2.legend(handles=leg, loc="lower right", fontsize=9, frameon=False)

    fig.suptitle("Sticker patterning drives peptide multivalency, which "
                 "sets network robustness and water dynamics",
                 fontsize=12, y=1.01)
    fig.savefig(os.path.join(OUT, "14_money_plot.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved 14_money_plot.png")


if __name__ == "__main__":
    fig_money()
