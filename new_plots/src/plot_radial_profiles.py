"""Plot continuous radial density profiles for peptide and water."""
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
COMP_COLOR = {"9E": "#444", "9C": "#1f77b4", "9A": "#d62728",
              "9P": "#2ca02c", "9N": "#ff7f0e"}
PAT_COLOR  = {"X2": "#9467bd", "X2S": "#c5b0d5",
              "X3":  "#ff7f0e", "X3S": "#d62728"}
COLOR = {**COMP_COLOR, **PAT_COLOR}


def fig_pep_vs_water():
    df = pd.read_csv(os.path.join(DATA, "24_radial_profiles.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    sys_groups = [
        ("9E 9C 9A 9P 9N".split(), "single-sticker 9-residue systems",
         COMP_COLOR),
        ("X2 X2S X3 X3S".split(), "patterned 9-residue systems",
         PAT_COLOR),
    ]
    for col, (sys_list, title, color_map) in enumerate(sys_groups):
        ax_p, ax_w = axes[0, col], axes[1, col]
        for sys in sys_list:
            sub = df[df["system"] == sys]
            c = color_map[sys]
            ax_p.plot(sub["r"], sub["rho_pep_heavy"], "-",
                      color=c, lw=2.0,
                      label=sys)
            ax_w.plot(sub["r"], sub["rho_water_O"], "-",
                      color=c, lw=2.0, label=sys)
        ax_p.set_title(f"peptide density — {title}", fontsize=10)
        ax_w.set_title(f"water-O density — {title}", fontsize=10)
        ax_w.set_xlabel(r"distance from condensate COM  $r$ (Å)")
        ax_p.set_ylabel(r"$\rho_{\rm pep\ heavy}$ (atoms/Å³)")
        ax_w.set_ylabel(r"$\rho_{\rm water}$ (waters/Å³)")
        ax_p.legend(fontsize=8, frameon=False, ncol=2)
        ax_w.legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Continuous radial density profiles from condensate COM",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "24_radial_profiles.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 24_radial_profiles.png")


def fig_interface_overlay():
    """Overlay peptide and water profiles for X3 and X3S to make interface
    visible.  Find the Gibbs dividing surface from the water profile."""
    df = pd.read_csv(os.path.join(DATA, "24_radial_profiles.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, sharey=False)
    for ax, sys in zip(axes, ["X3", "X3S"]):
        sub = df[df["system"] == sys].sort_values("r")
        r = sub["r"].values
        rho_p = sub["rho_pep_heavy"].values
        rho_w = sub["rho_water_O"].values
        ax_p = ax
        ax_w = ax.twinx()
        ax_p.plot(r, rho_p, color=COLOR[sys], lw=2.5,
                  label="peptide heavy atoms")
        ax_w.plot(r, rho_w, color="#1f77b4", lw=2.0, ls="--",
                  label="water O")
        # estimate Gibbs dividing surface as half of peptide max
        if rho_p.max() > 0:
            half = rho_p.max() / 2
            i = np.argmin(np.abs(rho_p - half))
            ax.axvline(r[i], color="grey", lw=0.7, ls=":")
            ax.text(r[i] + 0.4, rho_p.max() * 0.85,
                    f"R_gds≈{r[i]:.1f} Å", color="grey", fontsize=9)
        ax_p.set_xlabel(r"$r$ (Å)")
        ax_p.set_ylabel(r"$\rho_{\rm pep}$ (Å$^{-3}$)",
                         color=COLOR[sys])
        ax_w.set_ylabel(r"$\rho_{\rm water}$ (Å$^{-3}$)", color="#1f77b4")
        ax.set_title(sys, fontsize=11, color=COLOR[sys], fontweight="bold")
        ax_p.tick_params(axis="y", labelcolor=COLOR[sys])
        ax_w.tick_params(axis="y", labelcolor="#1f77b4")
    fig.suptitle("Peptide vs water density across the condensate interface",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "25_interface_X3_X3S.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("saved 25_interface_X3_X3S.png")


if __name__ == "__main__":
    fig_pep_vs_water()
    fig_interface_overlay()
    print("Done.")
