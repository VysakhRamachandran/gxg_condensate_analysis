"""Composite Figure 4: water dynamics are spatially heterogeneous and tied
to sticker chemistry.

(A) τ_inner vs τ_outer log-log scatter.
(B) Parallel-coordinate slope plot: overall → inner; overall → outer.
(C) Survival probability raw and rescaled by τ (collapse test).
(D) Effective peptide–water lifetime ranked across systems.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")
VYS_DATA = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/data"

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11, "axes.titlesize": 11,
})

ORDER = ["3E", "3C", "3A", "3P", "3N",
         "9E", "9C", "9A", "9P", "9N",
         "X2", "X2S", "X3", "X3S"]
SINGLE_LBL = {"E": "equal", "C": "charged", "A": "aromatic",
              "P": "polar", "N": "nonpolar"}
PAT_KEY = {"X2": "2_close", "X2S": "2_spread",
           "X3": "3_close", "X3S": "3_spread"}

COMP_COLOR = {"E": "#444", "C": "#1f77b4", "A": "#d62728",
              "P": "#2ca02c", "N": "#ff7f0e"}
PAT_COLOR = {"X2": "#9467bd", "X2S": "#c5b0d5",
             "X3": "#ff7f0e", "X3S": "#d62728"}
HL = {"X3S": "#d62728", "X3": "#ff7f0e"}


def col(s):
    if s in PAT_COLOR:
        return PAT_COLOR[s]
    return COMP_COLOR[s[1]]


def mk(s):
    if s.startswith("3"): return "o"
    if s.startswith("9"): return "s"
    return "D"


def biexp(t, A, t1, t2):
    return A * np.exp(-t / t1) + (1 - A) * np.exp(-t / t2)


def fit_teff(y, dt=0.005):
    t = np.arange(len(y)) * dt
    try:
        p, _ = curve_fit(biexp, t, y, p0=(0.5, 0.01, 0.5),
                         bounds=([0, 1e-5, 1e-3], [1, 5, 50]),
                         maxfev=20000)
        A, t1, t2 = p
        return A * t1 + (1 - A) * t2
    except Exception:
        return np.nan


def _sp_inner_outer(sys):
    if sys.startswith("3"):
        path = os.path.join(VYS_DATA, f"sp_3res_{SINGLE_LBL[sys[1]]}_new.csv")
    elif sys in PAT_KEY:
        path = os.path.join(VYS_DATA, f"sp_9res_{PAT_KEY[sys]}_new.csv")
    else:
        path = os.path.join(VYS_DATA, f"sp_9res_{SINGLE_LBL[sys[1]]}_new.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _sp_all(sys):
    if sys.startswith("3"):
        path = os.path.join(VYS_DATA, f"sp_all_3res_{SINGLE_LBL[sys[1]]}_new.csv")
    elif sys in PAT_KEY:
        path = os.path.join(VYS_DATA, f"sp_all_9res_{PAT_KEY[sys]}_new.csv")
    else:
        path = os.path.join(VYS_DATA, f"sp_all_9res_{SINGLE_LBL[sys[1]]}_new.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def collect_taus():
    rows = []
    for sys in ORDER:
        io = _sp_inner_outer(sys)
        all_ = _sp_all(sys)
        row = {"sys": sys}
        if io is not None and "Inner" in io.columns and "Outer" in io.columns:
            row["tau_in"] = fit_teff(io["Inner"].values)
            row["tau_out"] = fit_teff(io["Outer"].values)
        if all_ is not None:
            col0 = "SP_all" if "SP_all" in all_.columns else all_.columns[0]
            row["tau_all"] = fit_teff(all_[col0].values)
        rows.append(row)
    return pd.DataFrame(rows).set_index("sys").reindex(ORDER)


def panel_a(ax, taus):
    """tau_in vs tau_out scatter on linear axes (data spans <1 decade)."""
    # manual label offsets to declutter
    label_offset = {
        "3E": (10, 4), "3C": (10, 4), "3A": (10, -10),
        "3P": (-26, -2), "3N": (10, -8),
        "9E": (10, 6), "9C": (-22, 6), "9A": (10, -10),
        "9P": (10, 4), "9N": (-22, -2),
        "X2": (10, -10), "X2S": (-26, 4),
        "X3": (10, 6), "X3S": (10, 6),
    }
    for s, r in taus.iterrows():
        is_hl = s in HL
        ax.scatter(r["tau_out"], r["tau_in"], color=col(s),
                   marker=mk(s),
                   s=220 if is_hl else 110,
                   edgecolors="k", linewidths=1.2 if is_hl else 0.5,
                   zorder=4 if is_hl else 3)
        off = label_offset.get(s, (8, 4))
        ax.annotate(s, (r["tau_out"], r["tau_in"]),
                    xytext=off, textcoords="offset points",
                    fontsize=10 if is_hl else 9,
                    fontweight="bold" if is_hl else "normal",
                    color=HL.get(s, "k"))
    ax.set_xlabel(r"$\tau_{\rm outer}$  (ns)")
    ax.set_ylabel(r"$\tau_{\rm inner}$  (ns)")
    ax.set_title("(A)  Inner-core water lives ~25× longer than surface",
                 loc="left", fontsize=11)
    ax.set_xlim(0.0, 0.085)
    ax.set_ylim(0.0, 2.3)
    # reference line illustrating 25× ratio
    xs = np.linspace(0, 0.085, 50)
    ax.plot(xs, 25 * xs, "--", color="grey", lw=0.7, zorder=1)
    ax.text(0.072, 25 * 0.072 - 0.15, r"$\tau_{\rm in}=25\,\tau_{\rm out}$",
            color="grey", fontsize=9, ha="right")
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=9, label="3-residue"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
               markersize=9, label="9-residue"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
               markersize=9, label="patterned"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8, frameon=False)


def panel_b(ax, taus):
    """Slope plot of overall → inner and overall → outer for patterned."""
    pat = ["X2", "X2S", "X3", "X3S"]
    x = [0, 1, 2]
    labels = ["outer", "overall", "inner"]
    for s in pat:
        r = taus.loc[s]
        ys = [r["tau_out"], r["tau_all"], r["tau_in"]]
        ax.plot(x, ys, "-o", color=col(s), lw=2.4, ms=10,
                markeredgecolor="k", markeredgewidth=0.6,
                label=s)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10,
                                          fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\tau_{\rm water\,survival}$  (ns)")
    ax.set_title("(B)  Patterned systems across regions",
                 loc="left", fontsize=11)
    ax.set_xlim(-0.2, 2.2)
    ax.legend(loc="upper left", fontsize=9, frameon=False,
              ncol=4, columnspacing=0.7, handlelength=1.5)


def panel_c(ax):
    """Water survival probability decays — student-style plot.
    Shows 9-residue compositions + patterned systems together (decay
    visible on 1 ns window so curves separate clearly)."""
    # Show all 9-residue + patterned for completeness
    systems = ["9E", "9C", "9A", "9P", "9N", "X2", "X2S", "X3", "X3S"]
    for s in systems:
        all_ = _sp_all(s)
        if all_ is None:
            continue
        col0 = "SP_all" if "SP_all" in all_.columns else all_.columns[0]
        y = all_[col0].values
        t = np.arange(len(y)) * 0.005
        is_hl = s in HL
        ls = "--" if s in PAT_KEY else "-"
        ax.plot(t, y, ls, color=col(s),
                lw=2.4 if is_hl else 1.5,
                alpha=1.0 if is_hl else 0.85,
                label=s)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(r"water survival probability  SP(t)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_title("(C)  Water survival probability decays",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)


def panel_d(ax, taus):
    """Bar chart of tau_pw_eff per system, log y, no inline value labels."""
    obs = pd.read_csv(os.path.join(DATA, "07_descriptor_observable_table.csv"),
                      index_col=0).reindex(ORDER)
    vals = obs["tau_pw_eff"].values
    colors = [col(s) for s in ORDER]
    ax.bar(range(len(ORDER)), vals, color=colors, edgecolor="k",
           linewidth=0.5)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=9, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\tau_{\rm p-w,eff}$  (ns)")
    ax.set_title("(D)  Peptide–water contact lifetime per system",
                 loc="left", fontsize=11)
    for x_ in [4.5, 9.5]:
        ax.axvline(x_, color="grey", lw=0.4, ls=":")
    # group labels INSIDE the plot near the top of each region
    trans = ax.get_xaxis_transform()  # x in data, y in axes-fraction
    ax.text(2,    0.95, "3-residue", ha="center",
            transform=trans, color="grey", fontsize=9)
    ax.text(7,    0.95, "9-residue", ha="center",
            transform=trans, color="grey", fontsize=9)
    ax.text(11.5, 0.95, "patterned", ha="center",
            transform=trans, color="grey", fontsize=9)


def main():
    taus = collect_taus()
    fig = plt.figure(figsize=(13.0, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    panel_a(ax_a, taus)
    panel_b(ax_b, taus)
    panel_c(ax_c)
    panel_d(ax_d, taus)
    out = os.path.join(OUT, "manuscript_figs", "Fig4.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
