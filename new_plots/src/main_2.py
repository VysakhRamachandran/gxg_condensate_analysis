"""Driver: build descriptor x observable matrix and produce
(a) a combined table CSV, (b) a scatter-matrix figure with Spearman ρ,
(c) a focused figure highlighting sticker_spread vs water dynamics for the
    patterned-system inversion.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

from .descriptors import all_systems_descriptor_table
from .observables import all_systems_observables

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")
os.makedirs(DATA, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# colors for the three system classes
def system_color(sys):
    if sys.startswith("3"):
        return "#1f77b4"
    if sys.startswith("9"):
        return "#2ca02c"
    return "#d62728"

def system_marker(sys):
    if sys.startswith("3"): return "o"
    if sys.startswith("9"): return "s"
    return "D"


def build_table():
    d = all_systems_descriptor_table()
    o = all_systems_observables().drop(columns=["chain", "is_pattern"])
    df = d.join(o)
    df.to_csv(os.path.join(DATA, "07_descriptor_observable_table.csv"))
    return df


DESCRIPTORS = ["FCR", "f_aromatic", "f_sticker", "hydropathy",
               "sc_volume", "kappa_xg", "sticker_spread"]
OBSERVABLES = ["water_per_peptide", "waterbridges_per_res",
               "sasa_per_atom", "tau_pw_eff",
               "tau_in", "tau_out", "tau_all"]

OBS_LABEL = {
#    "water_per_peptide": r"water/peptide",
#    "waterbridges_per_res": r"WB / res",
    "sasa_per_atom": "SASA/atom",
#    "tau_pw_eff": r"$\tau_{\rm p-w}$",
#    "tau_in": r"$\tau_{\rm in}$",
#    "tau_out": r"$\tau_{\rm out}$",
    "tau_all": r"$\tau_{\rm all}$",
}
DESC_LABEL = {
    "FCR": "FCR",
    "f_aromatic": r"$f_{\rm aro}$",
    "f_sticker": r"$f_{\rm stk}$",
    "hydropathy": "KD hydropathy",
    "sc_volume": r"$\langle V_{\rm sc}\rangle$",
    "kappa_xg": r"$\kappa_{X/G}$",
    "sticker_spread": r"$\sigma_{\rm pos}^{\rm stk}$",
}


def scatter_matrix(df):
    n_rows = len(OBSERVABLES)
    n_cols = len(DESCRIPTORS)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.0 * n_cols, 1.7 * n_rows),
                              sharex="col")
    for i, obs in enumerate(OBSERVABLES):
        for j, desc in enumerate(DESCRIPTORS):
            ax = axes[i, j]
            x = df[desc].values
            y = df[obs].values
            mask = np.isfinite(x) & np.isfinite(y)
            for sys, xx, yy in zip(df.index[mask], x[mask], y[mask]):
                ax.scatter(xx, yy,
                           color=system_color(sys), marker=system_marker(sys),
                           s=42, edgecolors="k", linewidths=0.4, zorder=3)
            if mask.sum() >= 3:
                rho, pval = spearmanr(x[mask], y[mask])
                txt = f"ρ={rho:+.2f}"
                color = "k"
                if pval < 0.01: color = "darkred"
                elif pval < 0.05: color = "darkorange"
                ax.text(0.05, 0.92, txt, transform=ax.transAxes,
                        fontsize=8, color=color, va="top")
            if i == n_rows - 1:
                ax.set_xlabel(DESC_LABEL[desc], fontsize=9)
            if j == 0:
                ax.set_ylabel(OBS_LABEL[obs], fontsize=9)
            ax.tick_params(labelsize=7)
    # legend on top
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=8, label="3-residue"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ca02c",
               markersize=8, label="9-residue"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728",
               markersize=8, label="patterned"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Sequence descriptors vs condensate observables (Spearman ρ in each panel)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "07_descriptor_observable_matrix.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved descriptor x observable scatter matrix")


def correlation_heatmap(df):
    """Spearman correlation between descriptors and observables, color-coded."""
    rho = np.zeros((len(OBSERVABLES), len(DESCRIPTORS)))
    pval = np.zeros_like(rho)
    for i, obs in enumerate(OBSERVABLES):
        for j, desc in enumerate(DESCRIPTORS):
            x = df[desc].values; y = df[obs].values
            mask = np.isfinite(x) & np.isfinite(y)
            r, p = spearmanr(x[mask], y[mask])
            rho[i, j] = r; pval[i, j] = p
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DESCRIPTORS)))
    ax.set_xticklabels([DESC_LABEL[d] for d in DESCRIPTORS], rotation=30,
                       ha="right")
    ax.set_yticks(range(len(OBSERVABLES)))
    ax.set_yticklabels([OBS_LABEL[o] for o in OBSERVABLES])
    for i in range(len(OBSERVABLES)):
        for j in range(len(DESCRIPTORS)):
            marker = "**" if pval[i, j] < 0.01 else ("*" if pval[i, j] < 0.05 else "")
            ax.text(j, i, f"{rho[i, j]:+.2f}{marker}", ha="center",
                    va="center",
                    color="white" if abs(rho[i, j]) > 0.6 else "k",
                    fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Descriptor–observable correlation matrix\n"
                 "(* p<0.05, ** p<0.01; N=14)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "08_correlation_heatmap.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved correlation heatmap")


def patterned_focus(df):
    """Sticker-spread axis vs three observables for the patterned subset."""
    sub = df.loc[["X2", "X3", "X2S", "X3S"]].copy()
    sub = sub.sort_values("sticker_spread")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, obs, ylabel in zip(
            axes,
            ["water_per_peptide", "tau_in", "tau_all"],
            [r"water/peptide", r"$\tau_{\rm in}$ (ns)",
             r"$\tau_{\rm all}$ (ns)"]):
        ax.plot(sub["sticker_spread"], sub[obs], "-o",
                color="#d62728", lw=1.6, ms=10,
                markeredgecolor="k", markeredgewidth=0.5)
        for s, x, y in zip(sub.index, sub["sticker_spread"], sub[obs]):
            ax.annotate(s, (x, y), xytext=(7, 3), textcoords="offset points",
                        fontsize=10)
        rho, p = spearmanr(sub["sticker_spread"], sub[obs])
        ax.set_title(f"{ylabel}  vs sticker spread\nρ={rho:+.2f}, p={p:.2f}",
                     fontsize=10)
        ax.set_xlabel(r"sticker spread $\sigma_{\rm pos}^{\rm stk}$")
        ax.set_ylabel(ylabel)
    fig.suptitle("Patterned systems: composition is fixed; only sticker position differs",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "09_patterned_focus.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("Saved patterned focus")


if __name__ == "__main__":
    df = build_table()
    pd.set_option("display.float_format", "{:.3f}".format)
    print(df)
    scatter_matrix(df)
    correlation_heatmap(df)
    patterned_focus(df)
    print("All done. Outputs in", OUT)
