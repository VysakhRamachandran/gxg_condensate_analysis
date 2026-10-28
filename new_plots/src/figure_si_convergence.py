"""Supplementary figure: convergence checks for the production trajectories.

Three diagnostics that should all plateau by the second half of the run:
  (A) Largest-cluster fraction vs simulation time.
  (B) Water-to-peptide mass ratio vs time.
  (C) Block-averaged τ_eff (peptide–peptide contact correlation) computed
       over the 1st, 2nd, 3rd, 4th quartiles of the production trajectory
       — error bars give a direct estimate of statistical uncertainty.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT  = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")
GXG  = "/Volumes/potoyan-lab/will/gxg-project/gxg_comps"
VYS  = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/data"

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11, "axes.titlesize": 11,
})

ORDER = ["3E", "3C", "3A", "3P", "3N",
         "9E", "9C", "9A", "9P", "9N",
         "X2", "X2S", "X3", "X3S"]

# system -> (cluster-size CSV, cluster-water CSV)
SIZE_PATH = {
    "3E":  f"{GXG}/3_res/equal/size.csv",
    "3C":  f"{GXG}/3_res/charged/size.csv",
    "3A":  f"{GXG}/3_res/aromatic/size.csv",
    "3P":  f"{GXG}/3_res/polar/size.csv",
    "3N":  f"{GXG}/3_res/nonpolar/size.csv",
    "9E":  f"{GXG}/9_res/equal/size.csv",
    "9C":  f"{GXG}/9_res/charged/size.csv",
    "9A":  f"{GXG}/9_res/aromatic/size.csv",
    "9P":  f"{GXG}/9_res/polar/size.csv",
    "9N":  f"{GXG}/9_res/nonpolar/size.csv",
    "X2":  f"{GXG}/9_res/equal/2_close/size.csv",
    "X2S": f"{GXG}/9_res/equal/2_spread/size.csv",
    "X3":  f"{GXG}/9_res/equal/3_close/size.csv",
    "X3S": f"{GXG}/9_res/equal/3_spread/size.csv",
}

WATER_PATH = {  # cluster_water csvs from Vysakh's data dir
    "3E":  f"{VYS}/equal_cluster_water.csv",
    "3C":  f"{VYS}/charged_cluster_water.csv",
    "3A":  f"{VYS}/aromatic_cluster_water.csv",
    "3P":  f"{VYS}/polar_cluster_water.csv",
    "3N":  f"{VYS}/nonpolar_cluster_water.csv",
    "9E":  f"{VYS}/equal_cluster_water_9res.csv",
    "9C":  f"{VYS}/charged_cluster_water_9res.csv",
    "9A":  f"{VYS}/aromatic_cluster_water_9res.csv",
    "9P":  f"{VYS}/polar_cluster_water_9res.csv",
    "9N":  f"{VYS}/nonpolar_cluster_water_9res.csv",
    "X2":  f"{VYS}/2_close_cluster_water_9res.csv",
    "X2S": f"{VYS}/2_spread_cluster_water_9res.csv",
    "X3":  f"{VYS}/3_close_cluster_water_9res.csv",
    "X3S": f"{VYS}/3_spread_cluster_water_9res.csv",
}

# Approximate dt per frame in size.csv: trajectory was saved every 100 frames
# of an MD step. From the file the "frame" column goes 4000, 4100, 4200, …
# We will plot in units of the column directly (no unit conversion claim).

CLASS_COLOR = {
    "3E": "#444",  "3C": "#1f77b4", "3A": "#d62728",
    "3P": "#2ca02c", "3N": "#ff7f0e",
    "9E": "#444",  "9C": "#1f77b4", "9A": "#d62728",
    "9P": "#2ca02c", "9N": "#ff7f0e",
    "X2": "#9467bd", "X2S": "#c5b0d5",
    "X3": "#ff7f0e", "X3S": "#d62728",
}


def _moving_average(y, window=50):
    if len(y) < window: return y
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values


def panel_a_cluster_size(axes):
    """3 subpanels: 3-residue, 9-residue, patterned cluster fraction vs time.
    Raw trace shown faint; moving average overlaid."""
    groups = [("3-residue", ["3E","3C","3A","3P","3N"]),
              ("9-residue", ["9E","9C","9A","9P","9N"]),
              ("patterned", ["X2","X2S","X3","X3S"])]
    for ax, (title, sys_list) in zip(axes, groups):
        for s in sys_list:
            path = SIZE_PATH[s]
            if not os.path.exists(path): continue
            df = pd.read_csv(path)
            x = df["frame"].values
            y = df["prot-size"].values
            ax.plot(x, y, lw=0.35, color=CLASS_COLOR[s], alpha=0.25)
            ax.plot(x, _moving_average(y, 50), lw=1.6,
                    color=CLASS_COLOR[s], label=s)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("frame number")
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color="grey", ls=":", lw=0.4)
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower right")
    axes[0].set_ylabel("largest-cluster fraction")
    axes[0].text(-0.1, 1.15,
                 "(A)  Largest-cluster fraction vs simulation time",
                 transform=axes[0].transAxes, fontsize=11)


def panel_b_water_content(axes):
    """Same three sub-groups, but water/peptide mass ratio vs frame."""
    groups = [("3-residue", ["3E","3C","3A","3P","3N"]),
              ("9-residue", ["9E","9C","9A","9P","9N"]),
              ("patterned", ["X2","X2S","X3","X3S"])]
    for ax, (title, sys_list) in zip(axes, groups):
        for s in sys_list:
            path = WATER_PATH[s]
            if not os.path.exists(path): continue
            df = pd.read_csv(path)
            x = df["frame"].values
            y = (df["water_mass"] / df["protein_mass"]).values
            ax.plot(x, y, lw=0.35, color=CLASS_COLOR[s], alpha=0.25)
            ax.plot(x, _moving_average(y, 50), lw=1.6,
                    color=CLASS_COLOR[s], label=s)
        ax.set_xlabel("frame number")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    axes[0].set_ylabel("water / peptide  (mass ratio)")
    axes[0].text(-0.1, 1.15,
                 "(B)  Water-to-peptide ratio vs simulation time",
                 transform=axes[0].transAxes, fontsize=11)


def _block_tau(values, n_blocks=4):
    """Quartile-block means of an array.  Returns (means, sems_within_blocks)."""
    arr = np.asarray(values)
    arr = arr[np.isfinite(arr)]
    if arr.size < n_blocks * 2:
        return np.array([np.nan]*n_blocks), np.array([np.nan]*n_blocks)
    chunks = np.array_split(arr, n_blocks)
    means = np.array([c.mean() for c in chunks])
    sems  = np.array([c.std(ddof=1) / np.sqrt(len(c)) for c in chunks])
    return means, sems


def panel_c_block_water(ax):
    """Quartile block-means of water/peptide for each system — direct
    stability check: error bars come from intra-block variance."""
    x = np.arange(len(ORDER))
    width = 0.18
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]  # 4 quartiles
    for q in range(4):
        means = []
        sems  = []
        for s in ORDER:
            path = WATER_PATH[s]
            if not os.path.exists(path):
                means.append(np.nan); sems.append(np.nan); continue
            df = pd.read_csv(path)
            wpp = (df["water_mass"] / df["protein_mass"]).values
            m, e = _block_tau(wpp, n_blocks=4)
            means.append(m[q]); sems.append(e[q])
        ax.bar(x + (q - 1.5) * width, means, width,
               yerr=sems, color=colors[q],
               edgecolor="k", linewidth=0.4,
               label=f"Q{q+1}", capsize=1.5)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=9,
                                          rotation=45, ha="right")
    ax.set_ylabel("water / peptide  (mass ratio)")
    ax.set_title("(C)  Block-averaged water/peptide over trajectory quartiles",
                 loc="left", fontsize=11)
    ax.legend(fontsize=9, frameon=False, ncol=4, loc="upper right",
              title="quartile")
    for x_ in [4.5, 9.5]:
        ax.axvline(x_, color="grey", lw=0.4, ls=":")


def main():
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.30,
                          height_ratios=[1, 1, 1])
    ax_a = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ax_b = [fig.add_subplot(gs[1, j]) for j in range(3)]
    ax_c = fig.add_subplot(gs[2, :])
    panel_a_cluster_size(ax_a)
    panel_b_water_content(ax_b)
    panel_c_block_water(ax_c)
    out = os.path.join(OUT, "manuscript_figs", "FigS1_convergence.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
