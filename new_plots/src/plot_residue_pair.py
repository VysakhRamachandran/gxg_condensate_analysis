"""Plot residue-pair enrichment matrices and a top-pair summary."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")
AA20 = list("ARNDCEQGHILKMFPSTWYV")
# group AA into chemistry classes for a tidier axis order
CLASSES = {
    "charged+": "RK", "charged-": "DE",
    "polar": "NQSTH",
    "aromatic": "FYW",
    "non-polar": "AVLIMC",
    "special": "GP",
}
ORDERED_AA = list("RKDEHNQSTCFYWAVLIMGP")
ORDER_IDX = [AA20.index(a) for a in ORDERED_AA]

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "font.size": 10,
})


def load_enrichment(label):
    df = pd.read_csv(os.path.join(DATA, f"resmat_enr_{label}.csv"),
                     index_col=0)
    df = df.loc[ORDERED_AA, ORDERED_AA]
    return df


def heatmap_panel(ax, df, title, vmin=0.1, vmax=10, show_y=True):
    M = df.values.copy()
    # cap and clip
    M = np.where(np.isnan(M), 1.0, M)
    M = np.clip(M, vmin, vmax)
    im = ax.imshow(M, cmap="RdBu_r", norm=LogNorm(vmin=vmin, vmax=vmax),
                   aspect="auto")
    ax.set_xticks(range(len(ORDERED_AA)))
    ax.set_xticklabels(ORDERED_AA, fontsize=7)
    if show_y:
        ax.set_yticks(range(len(ORDERED_AA)))
        ax.set_yticklabels(ORDERED_AA, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="bold")
    # group separators
    boundaries = [2, 4, 5, 9, 12, 18]  # after RK, DE, H, NQSTC, FYW, AVLIM
    for b in boundaries:
        ax.axhline(b - 0.5, color="grey", lw=0.4)
        ax.axvline(b - 0.5, color="grey", lw=0.4)
    return im


def fig_heatmaps():
    # show the most interesting systems
    sel = ["3C", "3A", "3P", "9C", "9A", "X3S"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.0))
    for ax, lbl in zip(axes.flat, sel):
        df = load_enrichment(lbl)
        im = heatmap_panel(ax, df, lbl, show_y=True)
    # one shared colorbar
    cbar_ax = fig.add_axes([1.0, 0.15, 0.018, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="enrichment (obs / random-mix expected)")
    fig.suptitle("Residue-pair inter-chain contact enrichment\n"
                 "(red = pairs that contact each other more than expected from mole fractions)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "19_residue_pair_heatmaps.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 19_residue_pair_heatmaps.png")


def fig_top_pairs():
    df = pd.read_csv(os.path.join(DATA, "19_residue_pair_top.csv"))
    sys_order = ["3E", "3C", "3A", "3P", "3N",
                 "9E", "9C", "9A", "9P", "9N",
                 "X2", "X2S", "X3", "X3S"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for xi, s in enumerate(sys_order):
        sub = df[df["system"] == s].sort_values("rank")
        for _, r in sub.iterrows():
            ax.text(xi, -r["rank"] + 1,
                    f"{r['pair']}\n×{r['enrichment']:.1f}",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(facecolor=_color_for_pair(r["pair"]),
                              edgecolor="grey", boxstyle="round,pad=0.18",
                              alpha=0.85))
    ax.set_xticks(range(len(sys_order)))
    ax.set_xticklabels(sys_order, fontsize=10, fontweight="bold")
    ax.set_yticks(range(-4, 1))
    ax.set_yticklabels(["5th", "4th", "3rd", "2nd", "top"][::-1])
    ax.set_ylim(-4.7, 0.7)
    ax.set_xlim(-0.7, len(sys_order) - 0.3)
    ax.set_title("Top-5 most-enriched inter-chain residue pairs per system "
                 "(enrichment vs random mixing)")
    ax.set_xlabel("system")
    ax.set_ylabel("rank")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "20_residue_pair_top.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 20_residue_pair_top.png")


def _color_for_pair(pair):
    a, b = pair.split("-")
    aro = set("FYW"); pos = set("RK"); neg = set("DE")
    pol = set("NQSTH"); npl = set("AVLIMC")
    if {a, b} <= aro:                       return "#ffc4c4"  # aro-aro
    if (a in aro and b in pos) or (b in aro and a in pos): return "#ffd4a0"  # aro-cation
    if (a in pos and b in neg) or (b in pos and a in neg): return "#c4d4ff"  # salt
    if {a, b} <= pol:                       return "#d4f0c4"
    if {a, b} <= npl:                       return "#e4e4e4"
    return "#fff"


if __name__ == "__main__":
    fig_heatmaps()
    fig_top_pairs()
    print("Done.")
