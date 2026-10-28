"""Composite Figure 5: multivalency-by-diversification explains the
patterning effect.

(a) Sticker-restricted multivalency bars (X3S highlights).
(b) Network percolation_pc → peptide–water lifetime scatter (ρ=+0.91).
(c) Partner-set diversification: union vs sum, ratio, Jaccard.
(d) Rg–Ree scatter for patterned chains (X3 extended, X3S branched).
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
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11, "axes.titlesize": 11,
})

ORDER = ["3E", "3C", "3A", "3P", "3N",
         "9E", "9C", "9A", "9P", "9N",
         "X2", "X2S", "X3", "X3S"]

def sys_color(s):
    if s.startswith("3"): return "#9ecae1"
    if s.startswith("9"): return "#a1d99b"
    return "#fcae91"

HL = {"X3S": "#d62728", "X3": "#ff7f0e"}

def col(s): return HL.get(s, sys_color(s))
def mk(s):
    if s.startswith("3"): return "o"
    if s.startswith("9"): return "s"
    return "D"


def panel_a(ax, metrics):
    """Sticker-restricted multivalency bars."""
    vals = metrics["sticker_mean_degree"].values
    colors = [col(s) for s in metrics.index]
    edgecols = ["k" for s in metrics.index]
    lw = [1.4 if s in HL else 0.6 for s in metrics.index]
    ax.bar(range(len(metrics)), vals, color=colors,
           edgecolor=edgecols, linewidth=lw)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics.index, fontsize=9)
    ax.set_ylabel("mean sticker partners / peptide")
    ax.set_title("(A)  Sticker-restricted multivalency", loc="left",
                 fontsize=11)
    # group dividers
    for x in [4.5, 9.5]:
        ax.axvline(x, color="grey", lw=0.4, ls=":")
    ymax = vals.max() * 1.22
    ax.set_ylim(0, ymax)
    ax.text(2,    ymax * 0.94, "3-residue", ha="center",
            color="grey", fontsize=8)
    ax.text(7,    ymax * 0.94, "9-residue", ha="center",
            color="grey", fontsize=8)
    ax.text(11.5, ymax * 0.94, "patterned", ha="center",
            color="grey", fontsize=8)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.06, f"{v:.1f}", ha="center", fontsize=7,
                color="dimgray")


def panel_b(ax, df):
    """Percolation pc vs tau_p-w scatter, regression, ρ annotation."""
    x = df["percolation_pc"].values
    y = df["tau_pw_eff"].values
    mask = np.isfinite(x) & np.isfinite(y)
    # manual label-side hints to reduce overlap
    label_side = {
        "3E": (8, -8), "3C": (8, 6), "3A": (8, -10), "3P": (-22, -2),
        "3N": (8, 4),
        "9E": (-18, 8), "9C": (8, 6), "9A": (10, -8),
        "9P": (8, -10), "9N": (8, -4),
        "X2": (8, -8), "X2S": (8, -2),
        "X3": (10, 6), "X3S": (-18, -14),
    }
    for s, xx, yy in zip(df.index[mask], x[mask], y[mask]):
        is_hl = s in HL
        ax.scatter(xx, yy, color=col(s), marker=mk(s),
                   s=170 if is_hl else 80,
                   edgecolors="k",
                   linewidths=1.1 if is_hl else 0.45, zorder=4 if is_hl else 3)
        off = label_side.get(s, (6, 4))
        ax.annotate(s, (xx, yy), xytext=off,
                    textcoords="offset points",
                    fontsize=9 if is_hl else 8,
                    fontweight="bold" if is_hl else "normal",
                    color=HL.get(s, "k"))
    coef = np.polyfit(x[mask], y[mask], 1)
    xl = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xl, np.polyval(coef, xl), "--", color="grey", lw=0.9, zorder=1)
    r, p = spearmanr(x[mask], y[mask])
    ax.text(0.04, 0.94, f"Spearman ρ = {r:+.2f}\np = {p:.1e}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85,
                      boxstyle="round,pad=0.3"))
    ax.set_xlabel(r"network robustness $p_c$  (bond percolation)")
    ax.set_ylabel(r"$\tau_{\rm p-w}$  (ns)")
    ax.set_title("(B)  Chain connectivity modulates water dynamics",
                 loc="left", fontsize=11)
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sys_color("3E"),
               markeredgecolor="k", markersize=9, label="3-residue"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=sys_color("9E"),
               markeredgecolor="k", markersize=9, label="9-residue"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=sys_color("X2"),
               markeredgecolor="k", markersize=9, label="patterned"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8, frameon=False)


def panel_c(ax, div):
    """Diversification ratio + Jaccard for patterned systems."""
    pat = ["X2", "X2S", "X3", "X3S"]
    x = np.arange(len(pat))
    width = 0.38
    rd = [div.loc[div["system"] == s, "diversification_ratio"].values[0]
          for s in pat]
    jc = [div.loc[div["system"] == s, "mean_jaccard"].values[0]
          for s in pat]
    cols = [col(s) for s in pat]
    bars1 = ax.bar(x - width/2, rd, width, color=cols, edgecolor="k",
                   label="diversification (union / sum)")
    bars2 = ax.bar(x + width/2, jc, width, color="#aaaaaa", edgecolor="k",
                   label="mean Jaccard between sticker pairs")
    for xi, r_, j_ in zip(x, rd, jc):
        ax.text(xi - width/2, r_ + 0.01, f"{r_:.2f}", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(xi + width/2, j_ + 0.01, f"{j_:.2f}", ha="center",
                fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(pat, fontweight="bold")
    ax.set_ylabel("dimensionless")
    ax.set_title("(C)  Partner-set diversification", loc="left",
                 fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def panel_d(ax, conf):
    """Rg vs Ree for patterned chains (in condensate)."""
    inside = conf[conf["in_condensate"]].copy()
    # filter PBC artifacts
    inside = inside[(inside["Rg"] < 12) & (inside["Ree"] < 32)]
    for sys in ["X2", "X2S", "X3", "X3S"]:
        sub = inside[inside["system"] == sys]
        ax.scatter(sub["Rg"], sub["Ree"], color=col(sys),
                   s=18, alpha=0.55, edgecolors="none",
                   label=f"{sys}  ⟨Rg⟩={sub['Rg'].mean():.2f}, "
                         f"⟨Ree⟩={sub['Ree'].mean():.2f}")
    ax.set_xlabel(r"$R_g$  (Å)")
    ax.set_ylabel(r"end-to-end distance (Å)")
    ax.set_title("(D)  Chain geometry: X3 stretches end-to-end, X3S more branched",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              handlelength=0.8)


def main():
    metrics = pd.read_csv(os.path.join(DATA, "10_network_metrics.csv"),
                          index_col="system").reindex(ORDER)
    obs = all_systems_observables().drop(columns=["chain", "is_pattern"])
    df_perc = metrics.join(obs)
    div = pd.read_csv(os.path.join(DATA, "15_sticker_diversification_summary.csv"))
    conf = pd.read_csv(os.path.join(DATA, "20_chain_conformation.csv"))

    fig = plt.figure(figsize=(13.0, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.27)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, metrics)
    panel_b(ax_b, df_perc)
    panel_c(ax_c, div)
    panel_d(ax_d, conf)

    out_path = os.path.join(OUT, "manuscript_figs", "Fig5.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
