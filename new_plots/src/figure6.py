"""Composite Figure 6: sequence-grammar correlations and condensate water
is structurally distinct.

(a) Spearman ρ heatmap (descriptors × observables) — orthogonal sequence axes.
(b) Mean tetrahedral order q per system per region (inner/interface/bulk).
(c) Water g_OO(r) by region (selected systems).
(d) Top-5 enriched residue pairs per system, color-coded by chemistry.
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

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11, "axes.titlesize": 11,
})


# ---------- shared lookups -------------------------------------------------
DESCRIPTORS = ["FCR", "f_aromatic", "f_sticker", "hydropathy",
               "sc_volume", "kappa_xg", "sticker_spread"]
OBSERVABLES = ["water_per_peptide", "waterbridges_per_res",
               "sasa_per_atom", "tau_pw_eff",
               "tau_in", "tau_out", "tau_all"]
OBS_LABEL = {
    "water_per_peptide": r"water/pep",
    "waterbridges_per_res": r"WB / res",
    "sasa_per_atom": "SASA/atom",
    "tau_pw_eff": r"$\tau_{\rm p-w}$",
    "tau_in": r"$\tau_{\rm in}$",
    "tau_out": r"$\tau_{\rm out}$",
    "tau_all": r"$\tau_{\rm all}$",
}
DESC_LABEL = {
    "FCR": "FCR",
    "f_aromatic": r"$f_{\rm aro}$",
    "f_sticker": r"$f_{\rm stk}$",
    "hydropathy": "KD",
    "sc_volume": r"$\langle V_{\rm sc}\rangle$",
    "kappa_xg": r"$\kappa_{X/G}$",
    "sticker_spread": r"$\sigma_{\rm pos}^{\rm stk}$",
}
SYS_ORDER = ["3E", "3C", "3A", "3P", "3N",
             "9E", "9C", "9A", "9P", "9N",
             "X2", "X2S", "X3", "X3S"]
REGION_COLOR = {"inner": "#d62728", "interface": "#ff7f0e",
                "bulk": "#1f77b4"}


# ---------- panel (a): correlation heatmap ---------------------------------
def panel_a(ax):
    d = all_systems_descriptor_table()
    o = all_systems_observables().drop(columns=["chain", "is_pattern"])
    df = d.join(o)
    rho = np.zeros((len(OBSERVABLES), len(DESCRIPTORS)))
    for i, obs in enumerate(OBSERVABLES):
        for j, desc in enumerate(DESCRIPTORS):
            x = df[desc].values; y = df[obs].values
            mask = np.isfinite(x) & np.isfinite(y)
            r, _ = spearmanr(x[mask], y[mask])
            rho[i, j] = r
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DESCRIPTORS)))
    ax.set_xticklabels([DESC_LABEL[d] for d in DESCRIPTORS],
                       rotation=0, fontsize=10)
    ax.set_yticks(range(len(OBSERVABLES)))
    ax.set_yticklabels([OBS_LABEL[o] for o in OBSERVABLES], fontsize=10)
    # thin grid lines to separate cells
    for i in range(len(OBSERVABLES) + 1):
        ax.axhline(i - 0.5, color="white", lw=0.6)
    for j in range(len(DESCRIPTORS) + 1):
        ax.axvline(j - 0.5, color="white", lw=0.6)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, label="Spearman ρ")
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("(A)  Observable Correlations of Sequence Descriptors",
                 fontsize=11, loc="left")


# ---------- panel (b): mean q per region ----------------------------------
def panel_b(ax):
    summary = pd.read_csv(os.path.join(DATA, "25_water_q_summary.csv"))
    sys_list = ["9E", "9C", "9A", "9P", "9N", "X2", "X2S", "X3", "X3S"]
    width = 0.27
    x = np.arange(len(sys_list))
    for i, region in enumerate(["inner", "interface", "bulk"]):
        means, errs = [], []
        for sys in sys_list:
            sub = summary[(summary["system"] == sys) &
                          (summary["region"] == region)]
            if sub.empty:
                means.append(np.nan); errs.append(np.nan)
            else:
                means.append(float(sub["q_mean"].iloc[0]))
                errs.append(float(sub["q_std"].iloc[0])
                            / np.sqrt(float(sub["n"].iloc[0])))
        ax.bar(x + (i - 1) * width, means, width, yerr=errs,
               color=REGION_COLOR[region], edgecolor="k",
               label=region, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(sys_list, fontsize=9,
                                          fontweight="bold")
    ax.set_ylabel(r"tetrahedral order $\langle q\rangle$")
    ax.set_title("(B)  Inner-core water has reduced tetrahedral order",
                 fontsize=11, loc="left")
    ax.axhline(0.5, color="grey", lw=0.5, ls=":")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    ax.set_ylim(0.05, 0.6)


# ---------- panel (c): g_OO(r) selected ------------------------------------
def panel_c(axes):
    """3-system small-multiples of g_OO(r)."""
    g = pd.read_csv(os.path.join(DATA, "25_water_gOO.csv"))
    sel = [("9C", "9C"), ("X3", "X3"), ("X3S", "X3S")]
    for ax, (sys, title) in zip(axes, sel):
        for region in ["inner", "interface", "bulk"]:
            sub = g[(g["system"] == sys) & (g["region"] == region)]
            ax.plot(sub["r"], sub["g"], color=REGION_COLOR[region],
                    lw=1.8, label=region)
        ax.axhline(1.0, color="grey", lw=0.4, ls=":")
        ax.set_xlim(2.0, 7.5)
        ax.set_xlabel(r"$r$ (Å)")
        ax.set_title(title, fontsize=10, fontweight="bold")
    axes[0].set_ylabel(r"$g_{\rm OO}(r)$")
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    # super-title for the 3-panel block
    axes[0].text(0.0, 1.18,
                 "(C)  Water O–O g(r) by region (inner-core suppression universal)",
                 transform=axes[0].transAxes, fontsize=11,
                 fontweight="normal", ha="left")


# ---------- panel (d): top residue pairs ----------------------------------
def _color_for_pair(pair):
    a, b = pair.split("-")
    aro = set("FYW"); pos = set("RK"); neg = set("DE")
    pol = set("NQSTH"); npl = set("AVLIMC")
    if {a, b} <= aro:
        return "#ffc4c4"
    if (a in aro and b in pos) or (b in aro and a in pos):
        return "#ffd4a0"
    if (a in pos and b in neg) or (b in pos and a in neg):
        return "#c4d4ff"
    if {a, b} <= pol:
        return "#d4f0c4"
    if {a, b} <= npl:
        return "#e4e4e4"
    return "#f5e6d3"  # mixed / other — light cream so it's visible against bg


def panel_d(ax):
    """Top-5 inter-chain residue pairs as a color-only grid.

    rows = system, columns = rank (top → 5th). Cell color = chemistry
    class of the pair. Reading rule: predominantly pink/orange rows
    mean aromatic + cation–π dominate the system's contact chemistry.
    """
    df = pd.read_csv(os.path.join(DATA, "19_residue_pair_top.csv"))
    # build colored grid with per-cell color AND alpha-by-enrichment
    grid_color = np.empty((len(SYS_ORDER), 5), dtype=object)
    grid_enr   = np.full((len(SYS_ORDER), 5), np.nan)
    for yi, s in enumerate(SYS_ORDER):
        sub = df[df["system"] == s].sort_values("rank").reset_index(drop=True)
        for xi in range(5):
            if xi < len(sub):
                grid_color[yi, xi] = _color_for_pair(sub.iloc[xi]["pair"])
                grid_enr[yi, xi] = float(sub.iloc[xi]["enrichment"])
            else:
                grid_color[yi, xi] = "#ffffff"
    # log-enrichment range across all cells (excluding NaNs)
    log_enr = np.log10(grid_enr)
    log_min = np.nanmin(log_enr)   # ~ log10(5) ≈ 0.7
    log_max = np.nanmax(log_enr)   # ~ log10(200) ≈ 2.3

    def alpha_for(enr):
        if not np.isfinite(enr):
            return 0.0
        x = (np.log10(enr) - log_min) / (log_max - log_min)
        # map [0,1] to [0.30, 1.0] so the lowest cells are still visible
        return 0.30 + 0.70 * float(np.clip(x, 0, 1))

    from matplotlib.patches import Rectangle
    for yi in range(len(SYS_ORDER)):
        for xi in range(5):
            a = alpha_for(grid_enr[yi, xi])
            # draw a thin grey baseline (so empty cells aren't invisible)
            ax.add_patch(Rectangle((xi - 0.42, yi - 0.40), 0.84, 0.80,
                                    facecolor="white",
                                    edgecolor="lightgrey", linewidth=0.5))
            ax.add_patch(Rectangle((xi - 0.42, yi - 0.40), 0.84, 0.80,
                                    facecolor=grid_color[yi, xi],
                                    alpha=a,
                                    edgecolor="white", linewidth=1.0))
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.6, len(SYS_ORDER) - 0.4)
    ax.invert_yaxis()
    ax.set_xticks(range(5))
    ax.set_xticklabels(["top", "2nd", "3rd", "4th", "5th"], fontsize=10)
    ax.set_yticks(range(len(SYS_ORDER)))
    ax.set_yticklabels(SYS_ORDER, fontsize=10, fontweight="bold")
    ax.set_xlabel("rank within top-5")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(D)  Chemistry class of top-5 enriched inter-chain pairs",
                 fontsize=11, loc="left", pad=32)
    # gradient reference bar as an inset axes anchored ABOVE the cells,
    # between the title and the first system row.
    inset = ax.inset_axes([0.30, 1.08, 0.55, 0.035])
    n_grad = 200
    grad_img = np.linspace(0.30, 1.0, n_grad).reshape(1, -1)
    inset.imshow(grad_img, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                 extent=[log_min, log_max, 0, 1])
    inset.set_yticks([])
    tick_enrs = [5, 10, 30, 95]
    tick_pos = [np.log10(v) for v in tick_enrs if log_min <= np.log10(v) <= log_max]
    tick_lbl = [f"×{v}" for v in tick_enrs if log_min <= np.log10(v) <= log_max]
    inset.set_xticks(tick_pos)
    inset.set_xticklabels(tick_lbl, fontsize=8)
    inset.set_xlim(log_min, log_max)
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.tick_params(axis="x", length=2, pad=1)
    # chemistry-class legend with example pairs
    from matplotlib.patches import Patch
    leg = [
        Patch(facecolor="#ffc4c4", edgecolor="grey", label="aromatic–aromatic   (e.g. W–W)"),
        Patch(facecolor="#ffd4a0", edgecolor="grey", label="cation–π                    (e.g. R–W)"),
        Patch(facecolor="#c4d4ff", edgecolor="grey", label="salt bridge                  (e.g. R–D)"),
        Patch(facecolor="#d4f0c4", edgecolor="grey", label="polar–polar               (e.g. Q–Q)"),
        Patch(facecolor="#e4e4e4", edgecolor="grey", label="non-polar                 (e.g. L–I)"),
        Patch(facecolor="#f5e6d3", edgecolor="grey", label="other / mixed"),
    ]
    ax.legend(handles=leg, loc="lower center", ncol=2, fontsize=9,
              frameon=False, bbox_to_anchor=(0.5, -0.32),
              handlelength=1.4, handleheight=1.2, columnspacing=1.0)


# ---------- driver ---------------------------------------------------------
def main():
    fig = plt.figure(figsize=(15.5, 10.5))
    # 3-row layout
    #   row 0: heatmap (a) | tetrahedral q bars (b)
    #   row 1: g_OO 3-system panel (c)  | residue-pair grid (d)
    # gridspec with nested grids for the small-multiples in panel (c)
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                             width_ratios=[1.0, 1.4],
                             hspace=0.50, wspace=0.30)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    # nested gridspec for 3-subaxes inside outer[1, 0]
    gs_c = outer[1, 0].subgridspec(1, 3, wspace=0.18)
    ax_c1 = fig.add_subplot(gs_c[0])
    ax_c2 = fig.add_subplot(gs_c[1], sharey=ax_c1)
    ax_c3 = fig.add_subplot(gs_c[2], sharey=ax_c1)
    plt.setp(ax_c2.get_yticklabels(), visible=False)
    plt.setp(ax_c3.get_yticklabels(), visible=False)
    ax_d = fig.add_subplot(outer[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c([ax_c1, ax_c2, ax_c3])
    panel_d(ax_d)

    fig.suptitle("Figure 6  ·  Orthogonal sequence axes govern condensate water; "
                 "the condensate interior is a structurally distorted liquid",
                 fontsize=12, y=1.0)
    out_path = os.path.join(OUT, "manuscript_figs", "Fig6.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
