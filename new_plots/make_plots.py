"""
Alternative visualizations of the GXG condensate / water-dynamics data.

Replaces bar charts and single-exp decay panels with:
  1) heatmap of composition x chain length (water content, water-bridged contacts)
  2) violin plots of per-frame water content distributions
  3) slope/parallel plot of overall vs inner-core peptide-water contact lifetimes
     (the X3S/X3 inversion)
  4) 2D scatter of tau_inner vs tau_outer (water survival)
  5) survival-probability collapse: SP vs t/tau
  6) stretched-exponential (tau, beta) scatter from KWW refits

Outputs to /Volumes/potoyan-lab/will/gxg-project/new_plots/
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

DATA = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/data"
CORR = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/corr_results"
OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

# ---- system labels -----------------------------------------------------------
SINGLE_COMPS = ["equal", "charged", "aromatic", "polar", "nonpolar"]
SINGLE_LABELS = {"equal": "E", "charged": "C", "aromatic": "A",
                 "polar": "P", "nonpolar": "N"}
PATTERN_KEYS = ["2_close", "2_spread", "3_close", "3_spread"]
PATTERN_LABELS = {"2_close": "X2", "2_spread": "X2S",
                  "3_close": "X3", "3_spread": "X3S"}
COMP_COLOR = {"E": "#444", "C": "#1f77b4", "A": "#d62728",
              "P": "#2ca02c", "N": "#ff7f0e"}
PAT_COLOR = {"X2": "#9467bd", "X2S": "#c5b0d5",
             "X3": "#8c564b", "X3S": "#e377c2"}


# ---- helpers -----------------------------------------------------------------
def water_per_peptide_series(csv):
    df = pd.read_csv(csv)
    return df["water_mass"] / df["protein_mass"]


def waterbridges_series(csv):
    df = pd.read_csv(csv)
    return df["WaterBridgeCount"] if "WaterBridgeCount" in df else df.iloc[:, 0]


def biexp(t, A, t1, t2):
    return A * np.exp(-t / t1) + (1 - A) * np.exp(-t / t2)


def kww(t, beta, tau):
    return np.exp(-((t / tau) ** beta))


def fit_kww(t, y):
    t = np.asarray(t)
    y = np.asarray(y)
    mask = (t > 0) & np.isfinite(y) & (y > 1e-4)
    try:
        popt, _ = curve_fit(kww, t[mask], y[mask],
                            p0=(0.6, max(t[mask].max() / 5, 0.05)),
                            bounds=([0.05, 1e-4], [2.0, t[mask].max() * 5]),
                            maxfev=20000)
        return popt[0], popt[1]
    except Exception:
        return np.nan, np.nan


def tau_integral(t, y):
    """integral_0^inf of y(t) dt by trapezoid; y is normalized to 1 at t=0."""
    t = np.asarray(t)
    y = np.asarray(y)
    mask = (y > 1e-4) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    return float(np.trapezoid(y[mask], t[mask]))


# ---- 1) heatmap of single-sticker systems -----------------------------------
def plot_heatmap_singles():
    rows = []
    for comp in SINGLE_COMPS:
        for chain, suffix in [(3, ""), (9, "_9res")]:
            cw = os.path.join(DATA, f"{comp}_cluster_water{suffix}.csv")
            wb_3 = os.path.join(DATA, f"{comp}_waterbridges_3res.csv")
            wb_9 = os.path.join(DATA, f"{comp}_waterbridges_9res.csv")
            wb = wb_3 if chain == 3 else wb_9
            if not (os.path.exists(cw) and os.path.exists(wb)):
                continue
            wpp = water_per_peptide_series(cw).mean()
            df_cw = pd.read_csv(cw)
            # cluster_size is the number of residues in the largest cluster
            n_residues = df_cw["cluster_size"].mean()
            wb_series = waterbridges_series(wb)
            wb_per_res = wb_series.mean() / n_residues
            rows.append({"comp": SINGLE_LABELS[comp], "chain": chain,
                         "water_per_peptide": wpp,
                         "wb_per_res": wb_per_res})
    df = pd.DataFrame(rows)
    if df.empty:
        print("heatmap: no data"); return

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for ax, value, title, cmap in [
        (axes[0], "water_per_peptide", "water-to-peptide mass ratio", "viridis"),
        (axes[1], "wb_per_res", "water-bridged contacts / residue", "magma"),
    ]:
        pivot = df.pivot(index="chain", columns="comp", values=value)
        pivot = pivot.reindex(columns=["E", "C", "A", "P", "N"])
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{c}-res" for c in pivot.index])
        ax.set_title(title, fontsize=10)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if cmap == "magma" and v < pivot.values[np.isfinite(pivot.values)].mean()
                            else "k", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Composition × chain length (mean across trajectory)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_heatmap_singles.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("[1] heatmap saved")


# ---- 2) violin plots of water-per-peptide -----------------------------------
def plot_violin_water():
    systems = []
    # All 3-residue first
    for comp in SINGLE_COMPS:
        cw = os.path.join(DATA, f"{comp}_cluster_water.csv")
        if os.path.exists(cw):
            s = water_per_peptide_series(cw).dropna()
            systems.append((f"3{SINGLE_LABELS[comp]}", s.values,
                            COMP_COLOR[SINGLE_LABELS[comp]]))
    n_three = len(systems)
    # All 9-residue
    for comp in SINGLE_COMPS:
        cw = os.path.join(DATA, f"{comp}_cluster_water_9res.csv")
        if os.path.exists(cw):
            s = water_per_peptide_series(cw).dropna()
            systems.append((f"9{SINGLE_LABELS[comp]}", s.values,
                            COMP_COLOR[SINGLE_LABELS[comp]]))
    n_nine_end = len(systems)
    # Patterned (file naming: e.g. 2_close_cluster_water_9res.csv)
    for pk in PATTERN_KEYS:
        cw = os.path.join(DATA, f"{pk}_cluster_water_9res.csv")
        if os.path.exists(cw):
            s = water_per_peptide_series(cw).dropna()
            systems.append((PATTERN_LABELS[pk], s.values, PAT_COLOR[PATTERN_LABELS[pk]]))

    if not systems:
        print("violin: no data"); return

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    data = [s[1] for s in systems]
    labels = [s[0] for s in systems]
    colors = [s[2] for s in systems]
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_edgecolor("k"); pc.set_alpha(0.6)
    for key in ["cmins", "cmaxes", "cbars", "cmedians"]:
        parts[key].set_color("k"); parts[key].set_linewidth(0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("water / peptide  (mass ratio)")
    ax.set_title("Per-frame distributions reveal spread hidden by bar means")
    ax.axvline(n_three + 0.5, color="grey", lw=0.4, ls=":")
    ax.axvline(n_nine_end + 0.5, color="grey", lw=0.4, ls=":")
    ymax = ax.get_ylim()[1]
    ax.text((1 + n_three) / 2, ymax * 0.95, "3-residue",
            ha="center", color="grey", fontsize=9)
    ax.text((n_three + 1 + n_nine_end) / 2, ymax * 0.95, "9-residue",
            ha="center", color="grey", fontsize=9)
    ax.text((n_nine_end + 1 + len(systems)) / 2, ymax * 0.95, "patterned",
            ha="center", color="grey", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_violin_water_content.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("[2] violin saved")


# ---- helper: fit tau from SP CSV --------------------------------------------
def tau_from_sp_csv(csv, col):
    df = pd.read_csv(csv)
    if col not in df.columns:
        return np.nan
    y = df[col].values
    t = np.arange(len(y)) * 0.005  # ns (5 ps dt based on tM grids above)
    return tau_integral(t, y)


# ---- 3) slope plot: overall vs inner-core peptide-water tau -----------------
def plot_slope_inversion():
    # Use SP CSVs split into Inner/Outer for the four patterned systems
    csv_map = {
        "X2": "sp_9res_2_close_new.csv",
        "X2S": "sp_9res_2_spread_new.csv",
        "X3": "sp_9res_3_close_new.csv",
        "X3S": "sp_9res_3_spread_new.csv",
    }
    overall_map = {
        "X2": "sp_all_9res_2_close_new.csv",
        "X2S": "sp_all_9res_2_spread_new.csv",
        "X3": "sp_all_9res_3_close_new.csv",
        "X3S": "sp_all_9res_3_spread_new.csv",
    }
    rows = []
    for lbl in ["X2", "X2S", "X3", "X3S"]:
        inner_csv = os.path.join(DATA, csv_map[lbl])
        overall_csv = os.path.join(DATA, overall_map[lbl])
        tau_in = tau_from_sp_csv(inner_csv, "Inner")
        tau_out = tau_from_sp_csv(inner_csv, "Outer")
        tau_all = tau_from_sp_csv(overall_csv, "SP_all")
        rows.append({"sys": lbl, "tau_inner": tau_in,
                     "tau_outer": tau_out, "tau_all": tau_all})
    df = pd.DataFrame(rows)
    print(df)

    # Two-panel: (overall -> inner) and (overall -> outer)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, target_col, target_label, title in [
        (axL, "tau_inner", "inner core", "overall → inner core"),
        (axR, "tau_outer", "outer layer", "overall → outer interface"),
    ]:
        x = [0, 1]
        for _, r in df.iterrows():
            y2 = r[target_col]
            ax.plot(x, [r.tau_all, y2], "-o",
                    color=PAT_COLOR[r.sys], lw=2.2, ms=8, label=r.sys)
            ax.text(-0.04, r.tau_all, r.sys, ha="right", va="center",
                    color=PAT_COLOR[r.sys], fontsize=10, fontweight="bold")
            ax.text(1.04, y2, r.sys, ha="left", va="center",
                    color=PAT_COLOR[r.sys], fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["overall", target_label])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylabel(r"$\tau_{\rm water\ survival}$  (ns)")
        ax.set_title(title)
    fig.suptitle("Parallel-coordinate view: rankings preserved core-side, partly inverted outer-side",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_slope_overall_vs_regions.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("[3] slope plot saved")
    return df


# ---- 4) tau_inner vs tau_outer scatter --------------------------------------
def plot_tau_inner_outer():
    rows = []
    # 3-residue compositions
    for comp in SINGLE_COMPS:
        f = os.path.join(DATA, f"sp_3res_{comp}_new.csv")
        if os.path.exists(f):
            rows.append({"sys": f"3{SINGLE_LABELS[comp]}",
                         "tau_in": tau_from_sp_csv(f, "Inner"),
                         "tau_out": tau_from_sp_csv(f, "Outer"),
                         "color": COMP_COLOR[SINGLE_LABELS[comp]],
                         "marker": "o"})
    # 9-residue compositions
    for comp in SINGLE_COMPS:
        f = os.path.join(DATA, f"sp_9res_{comp}_new.csv")
        if os.path.exists(f):
            rows.append({"sys": f"9{SINGLE_LABELS[comp]}",
                         "tau_in": tau_from_sp_csv(f, "Inner"),
                         "tau_out": tau_from_sp_csv(f, "Outer"),
                         "color": COMP_COLOR[SINGLE_LABELS[comp]],
                         "marker": "s"})
    # patterned
    for pk in PATTERN_KEYS:
        f = os.path.join(DATA, f"sp_9res_{pk}_new.csv")
        if os.path.exists(f):
            rows.append({"sys": PATTERN_LABELS[pk],
                         "tau_in": tau_from_sp_csv(f, "Inner"),
                         "tau_out": tau_from_sp_csv(f, "Outer"),
                         "color": PAT_COLOR[PATTERN_LABELS[pk]],
                         "marker": "D"})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for _, r in df.iterrows():
        ax.scatter(r.tau_out, r.tau_in, color=r.color, marker=r.marker,
                   s=110, edgecolors="k", linewidths=0.6, zorder=3)
        ax.annotate(r.sys, (r.tau_out, r.tau_in),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    lo = min(df.tau_in.min(), df.tau_out.min()) * 0.7
    hi = max(df.tau_in.max(), df.tau_out.max()) * 1.15
    ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"outer-layer $\tau$  (ns)")
    ax.set_ylabel(r"inner-core $\tau$  (ns)")
    ax.set_title("Inner-core vs outer-layer water survival timescales")
    # legend for markers
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                  markersize=10, label="3-residue"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                  markersize=10, label="9-residue"),
           Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
                  markersize=10, label="patterned")]
    ax.legend(handles=leg, loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_tau_inner_vs_outer.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("[4] tau-tau scatter saved")
    return df


# ---- 5) survival probability collapse plot ----------------------------------
def plot_sp_collapse():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                    sharey=True)
    series = []
    # Use all_water (overall SP) CSVs for the patterned and compositional 9-mers
    for comp in SINGLE_COMPS:
        f = os.path.join(DATA, f"sp_all_9res_{comp}_new.csv")
        if os.path.exists(f):
            y = pd.read_csv(f)["SP_all"].values
            t = np.arange(len(y)) * 0.005
            tau = tau_integral(t, y)
            series.append((f"9{SINGLE_LABELS[comp]}", t, y, tau,
                           COMP_COLOR[SINGLE_LABELS[comp]], "-"))
    for pk in PATTERN_KEYS:
        f = os.path.join(DATA, f"sp_all_9res_{pk}_new.csv")
        if os.path.exists(f):
            y = pd.read_csv(f)["SP_all"].values
            t = np.arange(len(y)) * 0.005
            tau = tau_integral(t, y)
            series.append((PATTERN_LABELS[pk], t, y, tau,
                           PAT_COLOR[PATTERN_LABELS[pk]], "--"))
    for lbl, t, y, tau, c, ls in series:
        ax1.plot(t, y, ls, color=c, label=lbl, lw=1.4)
    ax1.set_xlabel("time (ns)")
    ax1.set_ylabel(r"SP(t)")
    ax1.set_title("raw survival probability")
    ax1.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper right")
    ax1.set_xscale("log"); ax1.set_xlim(0.005, 1.0)

    for lbl, t, y, tau, c, ls in series:
        if not np.isfinite(tau) or tau <= 0:
            continue
        ax2.plot(t / tau, y, ls, color=c, label=lbl, lw=1.4)
    ax2.set_xlabel(r"$t / \tau$")
    ax2.set_title(r"scaled by $\tau$ (integral)")
    ax2.set_xscale("log"); ax2.set_xlim(0.02, 50)
    fig.suptitle("Survival-probability collapse test: do curves overlay when rescaled?",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_sp_collapse.png"), bbox_inches="tight")
    plt.close(fig)
    print("[5] SP collapse saved")


# ---- 6) biexp fast/slow scatter from existing pickle fits -------------------
def plot_biexp_fast_slow():
    """Use the existing biexp fits stored in the pickles.
    para = [A, t1, t2] with corrFit = A*exp(-t/t1) + (1-A)*exp(-t/t2).
    Plot the slow timescale t2 vs the amplitude (1-A) of the slow component,
    revealing how much of the decay is in the long-lived component.
    """
    pkl3 = pickle.load(open(os.path.join(CORR, "3res_corr_results.pkl"), "rb"))
    pkl9 = pickle.load(open(os.path.join(CORR, "9_res_corr_results.pkl"), "rb"))
    pklP = pickle.load(open(os.path.join(CORR, "9res_corr_results_spacers.pkl"),
                            "rb"))
    rows = []
    for comp, d in pkl3.items():
        A, t1, t2 = d["para"]
        rows.append({"sys": f"3{SINGLE_LABELS[comp]}",
                     "A_slow": 1 - A, "t_slow": t2, "t_fast": t1,
                     "t_eff": d["t_eff"],
                     "color": COMP_COLOR[SINGLE_LABELS[comp]],
                     "marker": "o"})
    for comp, d in pkl9.items():
        A, t1, t2 = d["para"]
        rows.append({"sys": f"9{SINGLE_LABELS[comp]}",
                     "A_slow": 1 - A, "t_slow": t2, "t_fast": t1,
                     "t_eff": d["t_eff"],
                     "color": COMP_COLOR[SINGLE_LABELS[comp]],
                     "marker": "s"})
    for pk, d in pklP.items():
        A, t1, t2 = d["para"]
        rows.append({"sys": PATTERN_LABELS[pk],
                     "A_slow": 1 - A, "t_slow": t2, "t_fast": t1,
                     "t_eff": d["t_eff"],
                     "color": PAT_COLOR[PATTERN_LABELS[pk]],
                     "marker": "D"})
    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # Left: slow-component amplitude vs slow timescale
    for _, r in df.iterrows():
        ax1.scatter(r.t_slow, r.A_slow, color=r.color, marker=r.marker, s=130,
                    edgecolors="k", linewidths=0.6, zorder=3)
        ax1.annotate(r.sys, (r.t_slow, r.A_slow),
                     xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"slow timescale $\tau_{\rm slow}$  (ns)")
    ax1.set_ylabel(r"slow-component amplitude  $1-A$")
    ax1.set_title("Two-timescale decomposition of\npeptide--water contact correlation")

    # Right: t_eff bar across systems, ordered by t_eff
    df_sorted = df.sort_values("t_eff")
    ax2.barh(range(len(df_sorted)), df_sorted["t_eff"],
             color=df_sorted["color"], edgecolor="k", linewidth=0.6)
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted["sys"])
    ax2.set_xlabel(r"effective lifetime $\tau_{\rm eff}$  (ns)")
    ax2.set_title("Systems ranked by peptide--water $\\tau_{\\rm eff}$")
    ax2.set_xscale("log")

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                  markersize=10, label="3-residue"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                  markersize=10, label="9-residue"),
           Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
                  markersize=10, label="patterned")]
    ax1.legend(handles=leg, loc="best", fontsize=9, frameon=False)

    fig.suptitle("Slow component dominates: single τ hides this structure",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "06_biexp_decomposition.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("[6] biexp decomposition saved")
    return df


if __name__ == "__main__":
    plot_heatmap_singles()
    plot_violin_water()
    plot_slope_inversion()
    plot_tau_inner_outer()
    plot_sp_collapse()
    plot_biexp_fast_slow()
    print("Done. Outputs in", OUT)
