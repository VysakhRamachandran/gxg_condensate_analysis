"""Composite Figure 2: composition, length, and patterning control cluster structure.

(A) SASA-per-atom vs largest-cluster fraction across systems.
(B) Per-residue mean inter-chain contact count (residue × system heatmap).
(C) Peptide–peptide contact lifetime τ_eff ranked per system.
(D) Network largest-connected-component fraction.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")
VYS_DATA = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/data"
CORR = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/corr_results"
GXG = "/Volumes/potoyan-lab/will/gxg-project/gxg_comps"

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11, "axes.titlesize": 11,
})

ORDER = ["3E", "3C", "3A", "3P", "3N",
         "9E", "9C", "9A", "9P", "9N",
         "X2", "X2S", "X3", "X3S"]

CLASS_COLOR = {**{s: "#1f77b4" for s in ["3E", "3C", "3A", "3P", "3N"]},
               **{s: "#2ca02c" for s in ["9E", "9C", "9A", "9P", "9N"]},
               **{s: "#d62728" for s in ["X2", "X2S", "X3", "X3S"]}}
HL = {"X3S": "#d62728", "X3": "#ff7f0e"}
def col(s): return HL.get(s, CLASS_COLOR[s])
def mk(s):
    if s.startswith("3"): return "o"
    if s.startswith("9"): return "s"
    return "D"


def _sasa_for_sys(sys):
    if sys in ("X2", "X2S", "X3", "X3S"):
        df = pd.read_csv(os.path.join(VYS_DATA, "sasa_peratom_9res_spacers.csv"))
        keymap = {"X2": "2_close", "X2S": "2_spread",
                  "X3": "3_close", "X3S": "3_spread"}
        return df[keymap[sys]].values
    if sys.startswith("3"):
        df = pd.read_csv(os.path.join(VYS_DATA, "sasa_peratom_3res.csv"))
        kmap = {"3E": "equal", "3C": "charged", "3A": "aromatic",
                "3P": "polar", "3N": "nonpolar"}
        return df[kmap[sys]].values
    df = pd.read_csv(os.path.join(VYS_DATA, "sasa_peratom_9res.csv"))
    kmap = {"9E": "equal", "9C": "charged", "9A": "aromatic",
            "9P": "polar", "9N": "nonpolar"}
    return df[kmap[sys]].values


def _clustersize_for_sys(sys):
    """Largest-cluster fraction (peptides in dense phase / total)."""
    kmap = {"3E": ("3_res", "equal"), "3C": ("3_res", "charged"),
            "3A": ("3_res", "aromatic"), "3P": ("3_res", "polar"),
            "3N": ("3_res", "nonpolar"),
            "9E": ("9_res", "equal"), "9C": ("9_res", "charged"),
            "9A": ("9_res", "aromatic"), "9P": ("9_res", "polar"),
            "9N": ("9_res", "nonpolar"),
            "X2": ("9_res/equal", "2_close"),
            "X2S": ("9_res/equal", "2_spread"),
            "X3": ("9_res/equal", "3_close"),
            "X3S": ("9_res/equal", "3_spread")}
    chain_dir, sub = kmap[sys]
    if sys in ("X2", "X2S", "X3", "X3S"):
        path = os.path.join(GXG, chain_dir, sub, "size.csv")
    else:
        path = os.path.join(GXG, chain_dir, sub, "size.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "prot-size" in df.columns:
        return df["prot-size"].values
    return None


def panel_a(ax):
    """SASA per atom vs largest-cluster fraction.
    All systems get a uniform square frame; the text label IS the marker
    and the text color tells the class. Highlights (X3, X3S) drawn bold
    with a heavier square outline."""
    pts = {}
    for sys in ORDER:
        sasa = _sasa_for_sys(sys)
        size = _clustersize_for_sys(sys)
        if sasa is None or size is None:
            continue
        pts[sys] = (float(np.nanmean(size)), float(np.nanmean(sasa)))
    # set explicit axis range so text labels don't blow up autoscale
    xs = np.array([p[0] for p in pts.values()])
    ys = np.array([p[1] for p in pts.values()])
    ax.set_xlim(xs.min() - 0.05, xs.max() + 0.04)
    ax.set_ylim(ys.min() - 0.004, ys.max() + 0.005)
    # manual nudges for systems whose data positions are essentially identical
    # (e.g. X2 and X2S sit on top of each other). nudge is in DATA coords.
    # nudges (DATA coords) to spread stacked labels into empty corners.
    # X2S → bottom-left;  9E → bottom-right; everything else nudged radially
    # to minimize overlap with the central peptide cluster.
    nudge = {
        "X2":   (-0.025,  0.0014),
        "X2S":  (-0.035, -0.0040),   # bottom-left
        "9E":   (+0.035, -0.0040),   # bottom-right
        "9C":   (+0.030,  0.0010),
        "9A":   (-0.030,  0.0014),
        "9P":   (+0.005,  0.0014),
        "9N":   (-0.025, -0.0010),
        "X3":   (-0.014, -0.0010),
        "X3S":  (+0.014,  0.0010),
        "3C":   (+0.010,  0.0006),
        "3A":   (-0.010, -0.0006),
    }
    for sys, (x, y) in pts.items():
        is_hl = sys in HL
        dx, dy = nudge.get(sys, (0.0, 0.0))
        xx, yy = x + dx, y + dy
        # small grey dot at the true data position; coloured text label
        # near it (nudged when necessary). No square — color carries class.
        ax.scatter(x, y, marker="o",
                   s=22 if is_hl else 12,
                   color=col(sys), zorder=2, alpha=0.85)
        if dx != 0 or dy != 0:
            ax.plot([x, xx], [y, yy], color="grey", lw=0.4, zorder=1)
        ax.text(xx, yy, sys,
                ha="center", va="center",
                fontsize=11 if is_hl else 9,
                fontweight="bold",
                color=col(sys),
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.65, pad=0.5),
                zorder=3)
    ax.set_xlabel("largest-cluster fraction (peptides in dense phase)")
    ax.set_ylabel(r"SASA per atom  (nm²)")
    ax.set_title("(A)  Cluster compactness across systems",
                 loc="left", fontsize=11)
    # uniform circle markers (squares are dropped); color = class
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="#1f77b4", markersize=9, label="3-residue"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="#2ca02c", markersize=9, label="9-residue"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markeredgecolor="#d62728", markersize=9, label="patterned"),
    ]
    ax.legend(handles=leg, loc="lower left", fontsize=9, frameon=False)


def panel_b(ax):
    """Per-residue mean inter-chain contact count, residue × system heatmap."""
    df3 = pd.read_csv(os.path.join(GXG, "3_res/total_conts_df.csv"),
                      index_col=0)
    df9 = pd.read_csv(os.path.join(GXG, "9_res/total_conts_df.csv"),
                      index_col=0)
    # row order: residues with chemistry grouping
    ordered_aa = list("RKDEHNQSTCFYWAVLIMGP")
    # combine
    # row index (system) -> need to relabel
    df3 = df3.rename(index={"equal": "3E", "charged": "3C",
                            "aromatic": "3A", "polar": "3P",
                            "nonpolar": "3N"})
    df9 = df9.rename(index={"equal": "9E", "charged": "9C",
                            "aromatic": "9A", "polar": "9P",
                            "nonpolar": "9N"})
    combined = pd.concat([df3, df9])
    # 3-letter to 1-letter
    AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
              "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
              "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
              "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    combined.columns = [AA3to1[c] for c in combined.columns]
    combined = combined[ordered_aa]
    combined = combined.reindex(["3E", "3C", "3A", "3P", "3N",
                                  "9E", "9C", "9A", "9P", "9N"])
    im = ax.imshow(combined.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(combined.columns)))
    ax.set_xticklabels(combined.columns, fontsize=9)
    ax.set_yticks(range(len(combined.index)))
    ax.set_yticklabels(combined.index, fontsize=10, fontweight="bold")
    # group separators on x-axis
    for b in [2, 4, 5, 9, 12, 18]:
        ax.axvline(b - 0.5, color="white", lw=0.8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85,
                        label="mean inter-chain\ncontacts per residue")
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("(B)  Residue-resolved inter-chain contact counts",
                 loc="left", fontsize=11)


def _corr_teff(sys):
    if sys.startswith("3"):
        with open(os.path.join(CORR, "3res_corr_results.pkl"), "rb") as fh:
            d = pickle.load(fh)
        key = {"3E": "equal", "3C": "charged", "3A": "aromatic",
               "3P": "polar", "3N": "nonpolar"}[sys]
    elif sys in ("X2", "X2S", "X3", "X3S"):
        with open(os.path.join(CORR, "9res_corr_results_spacers.pkl"), "rb") as fh:
            d = pickle.load(fh)
        key = {"X2": "2_close", "X2S": "2_spread",
               "X3": "3_close", "X3S": "3_spread"}[sys]
    else:
        with open(os.path.join(CORR, "9_res_corr_results.pkl"), "rb") as fh:
            d = pickle.load(fh)
        key = {"9E": "equal", "9C": "charged", "9A": "aromatic",
               "9P": "polar", "9N": "nonpolar"}[sys]
    if key not in d:
        return np.nan
    return float(d[key]["t_eff"])


def panel_c(ax):
    """Contact correlation lifetime τ_eff per system."""
    vals = [_corr_teff(s) for s in ORDER]
    colors = [col(s) for s in ORDER]
    ax.bar(range(len(ORDER)), vals, color=colors, edgecolor="k",
           linewidth=0.5)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=9)
    ax.set_ylabel(r"$\tau_{\rm eff}$  (ns)")
    ax.set_title("(C)  Peptide–peptide contact lifetime",
                 loc="left", fontsize=11)
    ax.set_yscale("log")
    for x in [4.5, 9.5]:
        ax.axvline(x, color="grey", lw=0.4, ls=":")
    # labels
    for x, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(x, v * 1.15, f"{v:.1f}", ha="center", fontsize=7,
                    color="dimgray")


def panel_d(ax):
    """Representative condensate snapshot (X3S) with chemistry-class legend.
    Loaded from manuscript_figs/snapshots/snap_X3S.png (PyMOL render)."""
    import matplotlib.image as mpimg
    png = os.path.join(OUT, "manuscript_figs", "snapshots", "snap_X3S.png")
    if os.path.exists(png):
        img = mpimg.imread(png)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "snapshot not rendered\n(run src/render_snapshots.py)",
                ha="center", va="center", transform=ax.transAxes,
                color="grey")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(D)  X3S condensate snapshot",
                 loc="left", fontsize=11)
    # small chemistry-class legend at the bottom-right of the snapshot
    from matplotlib.patches import Patch
    # colors match PyMOL render: marine→#1a4789, firebrick, salmon,
    # forest→#228B22, sand→#c2b280
    leg = [
        Patch(facecolor="#dadada",   edgecolor="grey", label="G (spacer)"),
        Patch(facecolor="firebrick", edgecolor="grey", label="cationic (R/K/H)"),
        Patch(facecolor="#1a4789",   edgecolor="grey", label="anionic (D/E)"),
        Patch(facecolor="salmon",    edgecolor="grey", label="aromatic (F/Y/W)"),
        Patch(facecolor="#228B22",   edgecolor="grey", label="polar (S/T/N/Q/C)"),
        Patch(facecolor="#c2b280",   edgecolor="grey", label="non-polar"),
    ]
    ax.legend(handles=leg, loc="lower left", fontsize=7,
              frameon=True, framealpha=0.85, ncol=2, handlelength=1.0,
              handleheight=0.9, columnspacing=0.6, labelspacing=0.25)


def main():
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    panel_a(ax_a); panel_b(ax_b); panel_c(ax_c); panel_d(ax_d)
    out = os.path.join(OUT, "manuscript_figs", "Fig2.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
