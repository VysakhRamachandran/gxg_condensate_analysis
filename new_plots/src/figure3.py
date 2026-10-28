"""Composite Figure 3: water content and spatial distribution.

(A) Composition × chain-length heatmap of water/peptide and waterbridges/res.
(B) Continuous radial density profiles (peptide vs water) for X3 vs X3S.
(C) Per-frame water/peptide distributions (violins, all systems).
(D) Inner vs outer water fraction across systems.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")
VYS_DATA = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/data"

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11, "axes.titlesize": 11,
})

SINGLE_COMPS = ["equal", "charged", "aromatic", "polar", "nonpolar"]
SINGLE_LBL = {"equal": "E", "charged": "C", "aromatic": "A",
              "polar": "P", "nonpolar": "N"}
PATTERN_KEYS = ["2_close", "2_spread", "3_close", "3_spread"]
PATTERN_LBL = {"2_close": "X2", "2_spread": "X2S",
               "3_close": "X3", "3_spread": "X3S"}

COMP_COLOR = {"E": "#444", "C": "#1f77b4", "A": "#d62728",
              "P": "#2ca02c", "N": "#ff7f0e"}
PAT_COLOR = {"X2": "#9467bd", "X2S": "#c5b0d5",
             "X3": "#ff7f0e", "X3S": "#d62728"}
HL = {"X3S": "#d62728", "X3": "#ff7f0e"}

ORDER = ["3E", "3C", "3A", "3P", "3N",
         "9E", "9C", "9A", "9P", "9N",
         "X2", "X2S", "X3", "X3S"]


def _wpp(csv):
    df = pd.read_csv(csv)
    return df["water_mass"] / df["protein_mass"]


def panel_a(ax):
    """Composition × chain-length heatmaps (water/peptide and WB/res).
    Uses a nested 1×2 gridspec inside `ax` so the two quantities each get
    their own colorbar (they have different units and ranges).
    """
    rows = []
    for comp in SINGLE_COMPS:
        for chain, suf in [(3, ""), (9, "_9res")]:
            cw = os.path.join(VYS_DATA, f"{comp}_cluster_water{suf}.csv")
            wb = os.path.join(VYS_DATA, f"{comp}_waterbridges_{chain}res.csv")
            if not (os.path.exists(cw) and os.path.exists(wb)):
                continue
            wppv = float(_wpp(cw).mean())
            df_cw = pd.read_csv(cw)
            n_res = df_cw["cluster_size"].mean()
            wb_df = pd.read_csv(wb)
            wb_col = "WaterBridgeCount" if "WaterBridgeCount" in wb_df else wb_df.columns[0]
            wbpr = float(wb_df[wb_col].mean() / n_res)
            rows.append({"comp": SINGLE_LBL[comp], "chain": chain,
                         "water_per_peptide": wppv,
                         "wb_per_res": wbpr})
    df = pd.DataFrame(rows)
    cols = ["E", "C", "A", "P", "N"]
    pivot_w = df.pivot(index="chain", columns="comp",
                       values="water_per_peptide").reindex(columns=cols)
    pivot_b = df.pivot(index="chain", columns="comp",
                       values="wb_per_res").reindex(columns=cols)
    # split parent ax into two side-by-side sub-axes; aim for square cells
    parent_bbox = ax.get_position()
    ax.set_visible(False)
    fig = ax.figure
    # each heatmap will be ~square: use most of the parent width but leave
    # room for colorbars on each subpanel
    avail = parent_bbox.width
    cbar_each = 0.045   # fraction of parent width reserved for each colorbar
    pad = 0.04          # gap between the two heatmaps
    w = (avail - pad - 2 * cbar_each) / 2  # each heatmap data area
    sub_left  = fig.add_axes([parent_bbox.x0,
                              parent_bbox.y0,
                              w, parent_bbox.height])
    sub_right = fig.add_axes([parent_bbox.x0 + w + cbar_each + pad,
                              parent_bbox.y0,
                              w, parent_bbox.height])

    def _draw(sub_ax, pivot, cmap, title, cbar_label):
        Mw = np.vstack([pivot.loc[3].values, pivot.loc[9].values])
        im = sub_ax.imshow(Mw, cmap=cmap, aspect="auto")
        sub_ax.set_xticks(range(5))
        sub_ax.set_xticklabels(cols, fontsize=11, fontweight="bold")
        sub_ax.set_yticks([0, 1])
        sub_ax.set_yticklabels(["3-res", "9-res"], fontsize=10)
        cbar = plt.colorbar(im, ax=sub_ax, shrink=0.9,
                            fraction=0.06, pad=0.06)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(cbar_label, fontsize=9)
        sub_ax.set_title(title, loc="center", fontsize=10, pad=4)
        for spine in sub_ax.spines.values():
            spine.set_visible(False)
        sub_ax.tick_params(axis="both", length=0)

    _draw(sub_left,  pivot_w, "viridis",
          "water / peptide",
          "mass ratio (dimensionless)")
    _draw(sub_right, pivot_b, "magma",
          "waterbridges / residue",
          "count per residue")
    # parent-style header on the LEFT sub_ax (since `ax` is hidden)
    sub_left.text(-0.18, 1.30,
                  "(A)  Hydration & water-bridged contacts",
                  transform=sub_left.transAxes,
                  fontsize=11, fontweight="normal", ha="left")


def panel_b(ax):
    """Continuous radial profiles for X3 vs X3S — overlay peptide & water."""
    df = pd.read_csv(os.path.join(DATA, "24_radial_profiles.csv"))
    for sys in ["X3", "X3S"]:
        sub = df[df["system"] == sys].sort_values("r")
        r = sub["r"].values
        rho_p = sub["rho_pep_heavy"].values
        ax.plot(r, rho_p, color=HL[sys], lw=2.4,
                label=f"{sys} peptide")
    ax.set_xlabel(r"distance from condensate COM  $r$  (Å)")
    ax.set_ylabel(r"$\rho_{\rm pep\,heavy}$  (Å$^{-3}$)", color="k",
                  labelpad=10)
    ax2 = ax.twinx()
    for sys in ["X3", "X3S"]:
        sub = df[df["system"] == sys].sort_values("r")
        r = sub["r"].values
        rho_w = sub["rho_water_O"].values
        ax2.plot(r, rho_w, "--", color=HL[sys], lw=1.6,
                 label=f"{sys} water")
    ax2.set_ylabel(r"$\rho_{\rm water\,O}$  (Å$^{-3}$)", color="dimgray",
                   labelpad=10)
    ax2.tick_params(axis="y", labelcolor="dimgray")
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False,
              loc="center right", bbox_to_anchor=(1.0, 0.35))
    ax.set_title("(B)  Radial density profiles, X3 vs X3S",
                 loc="left", fontsize=11)


def panel_c(ax):
    """Violin of per-frame water/peptide ratios across all systems."""
    systems = []
    # 3-residue
    for comp in SINGLE_COMPS:
        cw = os.path.join(VYS_DATA, f"{comp}_cluster_water.csv")
        if os.path.exists(cw):
            systems.append((f"3{SINGLE_LBL[comp]}", _wpp(cw).dropna().values,
                            COMP_COLOR[SINGLE_LBL[comp]]))
    n_three = len(systems)
    # 9-residue
    for comp in SINGLE_COMPS:
        cw = os.path.join(VYS_DATA, f"{comp}_cluster_water_9res.csv")
        if os.path.exists(cw):
            systems.append((f"9{SINGLE_LBL[comp]}", _wpp(cw).dropna().values,
                            COMP_COLOR[SINGLE_LBL[comp]]))
    n_nine_end = len(systems)
    for pk in PATTERN_KEYS:
        cw = os.path.join(VYS_DATA, f"{pk}_cluster_water_9res.csv")
        if os.path.exists(cw):
            systems.append((PATTERN_LBL[pk], _wpp(cw).dropna().values,
                            PAT_COLOR[PATTERN_LBL[pk]]))
    data = [s[1] for s in systems]; labels = [s[0] for s in systems]
    colors = [s[2] for s in systems]
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_edgecolor("k"); pc.set_alpha(0.7)
    for key in ["cmins", "cmaxes", "cbars", "cmedians"]:
        parts[key].set_color("k"); parts[key].set_linewidth(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("water / peptide  (mass ratio)")
    ax.set_title("(C)  Per-frame water-to-peptide distributions",
                 loc="left", fontsize=11)
    ax.axvline(n_three + 0.5, color="grey", lw=0.4, ls=":")
    ax.axvline(n_nine_end + 0.5, color="grey", lw=0.4, ls=":")


def panel_d(ax):
    """Inner vs outer water fraction across systems."""
    # use combined_inner_outer_ratios.csv if present, otherwise compute per system
    src = os.path.join(VYS_DATA, "combined_inner_outer_ratios.csv")
    if os.path.exists(src):
        df = pd.read_csv(src)
        # remap system label
        sysmap = {"equal": "3E", "charged": "3C", "aromatic": "3A",
                  "polar": "3P", "nonpolar": "3N",
                  "equal_9res": "9E", "charged_9res": "9C",
                  "aromatic_9res": "9A", "polar_9res": "9P",
                  "nonpolar_9res": "9N",
                  "2_close": "X2", "2_spread": "X2S",
                  "3_close": "X3", "3_spread": "X3S"}
        # this file only has some systems; compute means
        if "system" in df.columns and "inner_ratio" in df.columns:
            summary = (df.groupby("system")
                         .agg(inner=("inner_ratio", "mean"),
                              outer=("outer_ratio", "mean")))
            # only single-sticker 3-res systems in this file; let's reconstruct
            # from individual inner_outer_water files for completeness
            pass
    # Reconstruct from per-system files
    rows = []
    file_map = {}
    for comp in SINGLE_COMPS:
        file_map[f"3{SINGLE_LBL[comp]}"] = os.path.join(
            VYS_DATA, f"{comp}_inner_outer_water.csv")
        file_map[f"9{SINGLE_LBL[comp]}"] = os.path.join(
            VYS_DATA, f"{comp}_inner_outer_water_9res.csv")
    for pk in PATTERN_KEYS:
        file_map[PATTERN_LBL[pk]] = os.path.join(
            VYS_DATA, f"{pk}_inner_outer_water_9res.csv")
    for sys in ORDER:
        path = file_map.get(sys)
        if path and os.path.exists(path):
            df_ = pd.read_csv(path)
            inner = df_["inner_water_unique"]
            outer = df_["outer_water"]
            tot   = inner + outer
            rows.append({"sys": sys,
                         "inner": float((inner / tot).mean()),
                         "outer": float((outer / tot).mean())})
    df2 = pd.DataFrame(rows).set_index("sys").reindex(ORDER)
    x = np.arange(len(ORDER))
    inner = df2["inner"].values * 100  # percent
    # color bars by class (3-res blue, 9-res green, patterned red/orange)
    def _bar_color(s):
        if s.startswith("3"): return "#1f77b4"
        if s.startswith("9"): return "#2ca02c"
        return HL.get(s, "#d62728")
    bar_colors = [_bar_color(s) for s in ORDER]
    ax.bar(x, inner, color=bar_colors, edgecolor="k", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=9,
                                          fontweight="bold")
    ax.set_ylabel("inner-core water  (% of cluster water)")
    ax.set_title("(D)  Inner-core water fraction (zoomed; remainder is outer)",
                 loc="left", fontsize=11)
    for x_ in [4.5, 9.5]:
        ax.axvline(x_, color="grey", lw=0.4, ls=":")
    # group labels
    ymax = inner.max() * 1.15
    ax.set_ylim(0, ymax)
    ax.text(2,    ymax * 0.95, "3-residue", ha="center",
            color="grey", fontsize=9)
    ax.text(7,    ymax * 0.95, "9-residue", ha="center",
            color="grey", fontsize=9)
    ax.text(11.5, ymax * 0.95, "patterned", ha="center",
            color="grey", fontsize=9)
    # add percentage labels above each bar
    for xi, v in zip(x, inner):
        if np.isfinite(v):
            ax.text(xi, v + 0.15, f"{v:.1f}", ha="center",
                    fontsize=8, color="dimgray")


def main():
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.45,
                          width_ratios=[1.05, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    panel_a(ax_a); panel_b(ax_b); panel_c(ax_c); panel_d(ax_d)
    out = os.path.join(OUT, "manuscript_figs", "Fig3.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
