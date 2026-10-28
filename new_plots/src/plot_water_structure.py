"""Plot water structure (tetrahedral q, g_OO) inside vs interface vs bulk."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
DATA = os.path.join(OUT, "data")

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

ORDER = ["9E", "9C", "9A", "9P", "9N", "X2", "X2S", "X3", "X3S"]
REGION_COLOR = {"inner": "#d62728", "interface": "#ff7f0e",
                "bulk": "#1f77b4"}


def fig_q_violins():
    q = pd.read_csv(os.path.join(DATA, "25_water_q_raw.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, region in zip(axes, ["inner", "interface", "bulk"]):
        data, labels = [], []
        for sys in ORDER:
            vals = q[(q["system"] == sys) & (q["region"] == region)]["q"].values
            if len(vals) > 0:
                data.append(vals)
                labels.append(sys)
        if not data:
            continue
        parts = ax.violinplot(data, showmedians=True, widths=0.85)
        for pc in parts["bodies"]:
            pc.set_facecolor(REGION_COLOR[region]); pc.set_edgecolor("k")
            pc.set_alpha(0.7)
        for key in ["cmins", "cmaxes", "cbars", "cmedians"]:
            parts[key].set_color("k"); parts[key].set_linewidth(0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(r"tetrahedral order $q$"
                      if region == "inner" else "")
        ax.set_title(f"{region} water  ({REGION_COLOR[region]})",
                     color=REGION_COLOR[region], fontsize=11)
        ax.axhline(0.5, color="grey", lw=0.5, ls=":")
    fig.suptitle("Tetrahedral order parameter $q$ by region "
                 "(q=0.95 ice, ~0.55 bulk TIP3P)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "26_water_q_violins.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 26_water_q_violins.png")


def fig_q_means(summary):
    fig, ax = plt.subplots(figsize=(11, 4.0))
    width = 0.27
    x = np.arange(len(ORDER))
    for i, region in enumerate(["inner", "interface", "bulk"]):
        means = []; errs = []
        for sys in ORDER:
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
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontweight="bold")
    ax.set_ylabel(r"mean tetrahedral order $\langle q\rangle$")
    ax.set_title("Mean tetrahedral order by system and region")
    ax.axhline(0.5, color="grey", lw=0.5, ls=":")
    ax.legend(frameon=False, fontsize=10)
    # zoom y to make region differences visible
    valid = summary["q_mean"].dropna()
    if not valid.empty:
        ax.set_ylim(valid.min() - 0.02, valid.max() + 0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "27_water_q_means.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 27_water_q_means.png")


def fig_gOO():
    g = pd.read_csv(os.path.join(DATA, "25_water_gOO.csv"))
    sel = ["9E", "9C", "9N", "X3", "X3S"]
    fig, axes = plt.subplots(1, len(sel), figsize=(2.4 * len(sel), 3.0),
                              sharey=True, sharex=True)
    for ax, sys in zip(axes, sel):
        for region in ["inner", "interface", "bulk"]:
            sub = g[(g["system"] == sys) & (g["region"] == region)]
            ax.plot(sub["r"], sub["g"], color=REGION_COLOR[region],
                    lw=1.6, label=region)
        ax.set_title(sys, fontweight="bold")
        ax.set_xlabel(r"$r$ (Å)")
        ax.set_xlim(2.0, 7.5)
        ax.axhline(1.0, color="grey", lw=0.4, ls=":")
    axes[0].set_ylabel(r"$g_{\rm OO}(r)$")
    axes[0].legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle("Water O–O radial distribution by region", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "28_water_gOO.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 28_water_gOO.png")


if __name__ == "__main__":
    summary = pd.read_csv(os.path.join(DATA, "25_water_q_summary.csv"))
    pd.set_option("display.float_format", "{:.3f}".format)
    print(summary.to_string(index=False))
    fig_q_violins()
    fig_q_means(summary)
    fig_gOO()
    print("Done.")
