"""Plot peptide-peptide contact network metrics computed by contact_network.py."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from .observables import all_systems_observables

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ORDER = ["3E", "3C", "3A", "3P", "3N",
         "9E", "9C", "9A", "9P", "9N",
         "X2", "X2S", "X3", "X3S"]


def sys_class_color(sys):
    if sys.startswith("3"): return "#1f77b4"
    if sys.startswith("9"): return "#2ca02c"
    return "#d62728"


def load_data():
    metrics = pd.read_csv(os.path.join(DATA, "10_network_metrics.csv"),
                          index_col="system").reindex(ORDER)
    dists = np.load(os.path.join(DATA, "10_degree_distributions.npz"))
    return metrics, dists


def plot_mean_degree(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, col, title, yl in [
        (axes[0], "mean_degree",
         "Mean # partner peptides per peptide",
         "mean degree (peptide–peptide)"),
        (axes[1], "sticker_mean_degree",
         "Sticker-restricted contact degree",
         "mean degree (sticker–sticker)"),
    ]:
        vals = metrics[col].values
        colors = [sys_class_color(s) for s in metrics.index]
        ax.bar(range(len(metrics)), vals, color=colors,
               edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics.index, rotation=0, fontsize=9)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10)
        for x, v in enumerate(vals):
            ax.text(x, v + 0.02 * max(vals), f"{v:.1f}", ha="center",
                    fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "10_mean_degree.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 10_mean_degree.png")


def plot_degree_distributions(dists):
    """Stacked violins of pooled per-peptide degrees by system."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, prefix, title in [(axes[0], "deg_", "peptide–peptide degree distribution"),
                              (axes[1], "stk_", "sticker–sticker degree distribution")]:
        data = []
        for s in ORDER:
            k = f"{prefix}{s}"
            if k in dists.files:
                data.append(dists[k])
            else:
                data.append(np.array([0]))
        parts = ax.violinplot(data, showmedians=True, widths=0.85)
        for pc, s in zip(parts["bodies"], ORDER):
            pc.set_facecolor(sys_class_color(s))
            pc.set_edgecolor("k")
            pc.set_alpha(0.65)
        for key in ["cmins", "cmaxes", "cbars", "cmedians"]:
            parts[key].set_color("k"); parts[key].set_linewidth(0.6)
        ax.set_xticks(range(1, len(ORDER) + 1))
        ax.set_xticklabels(ORDER, fontsize=9)
        ax.set_ylabel("degree (# distinct partners per peptide)")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "11_degree_distributions.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 11_degree_distributions.png")


def plot_lcc_percolation(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title, yl in [
        (axes[0], "lcc_fraction",
         "Largest connected component fraction",
         "LCC size / N_peptides"),
        (axes[1], "clustering",
         "Avg clustering coefficient",
         "clustering coefficient"),
        (axes[2], "percolation_pc",
         "Bond-percolation threshold",
         "fraction of edges removed before LCC < 0.5N"),
    ]:
        vals = metrics[col].values
        colors = [sys_class_color(s) for s in metrics.index]
        ax.bar(range(len(metrics)), vals, color=colors,
               edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics.index, rotation=0, fontsize=9)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "12_lcc_clustering_percolation.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 12_lcc_clustering_percolation.png")


def plot_degree_vs_observables(metrics):
    """Correlate network metrics with hydration / dynamic observables."""
    obs = all_systems_observables().drop(columns=["chain", "is_pattern"])
    df = metrics.join(obs)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    pairs = [
        ("sticker_mean_degree", "water_per_peptide"),
        ("sticker_mean_degree", "waterbridges_per_res"),
        ("sticker_mean_degree", "sasa_per_atom"),
        ("percolation_pc",      "tau_pw_eff"),
        ("clustering",          "tau_in"),
        ("mean_degree",         "tau_all"),
    ]
    from scipy.stats import spearmanr
    for ax, (xk, yk) in zip(axes.flat, pairs):
        x = df[xk].values; y = df[yk].values
        mask = np.isfinite(x) & np.isfinite(y)
        for s, xx, yy in zip(df.index[mask], x[mask], y[mask]):
            ax.scatter(xx, yy, color=sys_class_color(s),
                       marker=("o" if s.startswith("3") else
                               ("s" if s.startswith("9") else "D")),
                       s=80, edgecolors="k", linewidths=0.4, zorder=3)
            ax.annotate(s, (xx, yy), xytext=(5, 3),
                         textcoords="offset points", fontsize=8)
        if mask.sum() >= 3:
            r, p = spearmanr(x[mask], y[mask])
            ax.text(0.04, 0.96, f"ρ={r:+.2f}, p={p:.2f}",
                    transform=ax.transAxes, fontsize=9, va="top")
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
    fig.suptitle("Network metrics vs condensate observables", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "13_network_vs_observables.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 13_network_vs_observables.png")


if __name__ == "__main__":
    metrics, dists = load_data()
    pd.set_option("display.float_format", "{:.3f}".format)
    print(metrics)
    plot_mean_degree(metrics)
    plot_degree_distributions(dists)
    plot_lcc_percolation(metrics)
    plot_degree_vs_observables(metrics)
    print("Done.")
