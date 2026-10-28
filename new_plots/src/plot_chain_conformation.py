"""Plot Rg and end-to-end distributions per system, inside vs outside condensate."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ORDER = ["9E", "9C", "9A", "9P", "9N", "X2", "X2S", "X3", "X3S"]
HIGHLIGHT = {"X3S": "#d62728", "X3": "#ff7f0e"}
CLASS_COLOR = {**{s: "#a1d99b" for s in ["9E", "9C", "9A", "9P", "9N"]},
               **{s: "#fcae91" for s in ["X2", "X2S", "X3", "X3S"]}}


def color_for(sys):
    return HIGHLIGHT.get(sys, CLASS_COLOR[sys])


def fig_distributions(df):
    """Violins of Rg and Ree (inside-condensate chains only)."""
    inside = df[df["in_condensate"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for ax, col, ylabel in [(axes[0], "Rg",  r"$R_g$  (Å)"),
                            (axes[1], "Ree", r"end-to-end distance (Å)")]:
        data, colors = [], []
        for s in ORDER:
            vals = inside[inside["system"] == s][col].dropna().values
            data.append(vals)
            colors.append(color_for(s))
        parts = ax.violinplot(data, showmedians=True, widths=0.85)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_edgecolor("k"); pc.set_alpha(0.7)
        for key in ["cmins", "cmaxes", "cbars", "cmedians"]:
            parts[key].set_color("k"); parts[key].set_linewidth(0.7)
        ax.set_xticks(range(1, len(ORDER) + 1))
        ax.set_xticklabels(ORDER, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{col} of chains inside condensate")
    fig.suptitle("Chain conformation per system (in-condensate chains)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "21_chain_conformation_violins.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 21_chain_conformation_violins.png")


def fig_inside_vs_outside(summary):
    """Mean Rg inside vs outside condensate per system."""
    s = summary.copy()
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    width = 0.38
    x = np.arange(len(ORDER))
    inside  = [s.loc[(s["system"] == sys) & (s["in_condensate"] == True),  "Rg_mean"].values[0]
               for sys in ORDER]
    outside = [s.loc[(s["system"] == sys) & (s["in_condensate"] == False), "Rg_mean"].values[0]
               for sys in ORDER]
    in_std  = [s.loc[(s["system"] == sys) & (s["in_condensate"] == True),  "Rg_std"].values[0]
               for sys in ORDER]
    out_std = [s.loc[(s["system"] == sys) & (s["in_condensate"] == False), "Rg_std"].values[0]
               for sys in ORDER]
    ax.bar(x - width/2, outside, width, yerr=out_std,
           color="#cccccc", edgecolor="k", label="dilute (outside)",
           capsize=3)
    ax.bar(x + width/2, inside, width, yerr=in_std,
           color=[color_for(s) for s in ORDER], edgecolor="k",
           label="in condensate", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontweight="bold")
    ax.set_ylabel(r"$R_g$  (Å)")
    ax.set_title("Chain $R_g$ inside vs outside condensate (error bars = stdev across chains)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "22_Rg_inside_vs_outside.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 22_Rg_inside_vs_outside.png")


def fig_rg_ree_scatter(df):
    """Rg vs Ree scatter (per-chain) for patterned systems only — geometry."""
    inside = df[df["in_condensate"]].copy()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True)
    for ax, sys in zip(axes, ["X2", "X2S", "X3", "X3S"]):
        sub = inside[inside["system"] == sys]
        ax.scatter(sub["Rg"], sub["Ree"], color=color_for(sys),
                   s=18, alpha=0.45, edgecolors="none")
        ax.set_title(f"{sys}\n⟨Rg⟩={sub['Rg'].mean():.2f}  ⟨Ree⟩={sub['Ree'].mean():.2f}",
                     fontsize=10, fontweight="bold",
                     color=color_for(sys) if sys in HIGHLIGHT else "k")
        ax.set_xlabel(r"$R_g$  (Å)")
        ax.axhline(sub["Ree"].mean(), color="k", lw=0.6, ls=":")
        ax.axvline(sub["Rg"].mean(), color="k", lw=0.6, ls=":")
    axes[0].set_ylabel("end-to-end distance (Å)")
    fig.suptitle("Per-chain Rg–Ree distributions (in-condensate, patterned systems)\n"
                 "X3 is more end-to-end extended; X3S spans larger Rg (more branched)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "23_Rg_Ree_scatter.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 23_Rg_Ree_scatter.png")


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA, "20_chain_conformation.csv"))
    # filter PBC artifacts: max physical Rg for L=9 fully extended ≈ 10 Å,
    # max Ree ≈ 30 Å. Anything above is a wrapped-across-box artifact.
    n_before = len(df)
    df = df[(df["Rg"] < 12) & (df["Ree"] < 32)].copy()
    n_after = len(df)
    print(f"Filtered {n_before - n_after} PBC-artifact chains "
          f"({100*(n_before-n_after)/n_before:.1f}% of rows)")
    # recompute summary
    summary = (df.groupby(["system", "in_condensate"])
                 .agg(Rg_mean=("Rg", "mean"), Rg_std=("Rg", "std"),
                      Ree_mean=("Ree", "mean"), Ree_std=("Ree", "std"),
                      n=("Rg", "size"))).reset_index()
    summary.to_csv(os.path.join(DATA,
                                "20_chain_conformation_summary_filtered.csv"),
                   index=False)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(summary.to_string(index=False))
    fig_distributions(df)
    fig_inside_vs_outside(summary)
    fig_rg_ree_scatter(df)
    print("Done.")
