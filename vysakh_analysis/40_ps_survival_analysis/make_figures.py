#!/usr/bin/env python
"""Regenerate the water-dynamics figures from the corrected common-axis results.

Palette note. The manuscript's existing class colors (roughly #3b76af blue /
#3f9b52 green / #d1352b red) FAIL a colour-vision-deficiency check: the green-red
pair separates by only dE 5.3 under deuteranopia, below even the conditional 6-8
floor. The set used here keeps the same blue / green / red semantics and reads
almost identically to full-colour vision, but separates by dE 16.9 (deutan) and
28.7 (normal vision). Class is additionally encoded by x-axis grouping, dashed
separators and the tick labels, so identity never rests on colour alone.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.systems import SYSTEMS, DT_PS, PUBLISHED_INNER, PUBLISHED_OUTER

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "results", "ckpt")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)
DT_NS = DT_PS / 1000.0

# CVD-validated categorical palette, fixed order, never cycled
CLASS_COLOR = {"3-residue": "#2166ac", "9-residue": "#4daf4a", "patterned": "#b2182b"}
INK, INK2, GRID = "#1a1a1a", "#555555", "#d8d8d8"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ORDER = [s[0] for s in SYSTEMS]
KLASS = {s[0]: s[3] for s in SYSTEMS}


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def class_separators(ax, labels):
    """Dashed lines between classes: secondary encoding of class identity."""
    for i in range(1, len(labels)):
        if KLASS[labels[i]] != KLASS[labels[i - 1]]:
            ax.axvline(i - 0.5, color=INK2, ls=(0, (4, 3)), lw=0.8, zorder=1)


def bars(ax, df, layer, title):
    s = df[df.layer == layer].set_index("system").reindex(ORDER).dropna(subset=["tau_int_ns"])
    labels = list(s.index)
    x = np.arange(len(labels))
    cols = [CLASS_COLOR[KLASS[l]] for l in labels]
    # width 0.72 leaves a clear surface gap between adjacent bars
    ax.bar(x, s.tau_int_ns, width=0.72, color=cols, zorder=3,
           yerr=s.tau_int_sd, error_kw=dict(ecolor=INK2, lw=0.9, capsize=2.5))
    # open marker flags a curve that had not decayed inside the window
    nd = ~s.decayed.astype(bool)
    if nd.any():
        ax.scatter(x[nd.values], (s.tau_int_ns + s.tau_int_sd)[nd] * 1.06,
                   marker="^", s=18, facecolor="none", edgecolor=INK2, lw=0.9, zorder=4)
    class_separators(ax, labels)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"$\int S\,\mathrm{d}\tau$  (ns)")
    ax.set_title(title, loc="left")
    style(ax)
    return labels


def fig_residence(df):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    bars(axes[0], df, "inner", "A   inner core")
    bars(axes[1], df, "outer", "B   outer interfacial layer")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOR.values()]
    axes[0].legend(handles, list(CLASS_COLOR), frameon=False, fontsize=8,
                   loc="upper left", ncol=1)
    axes[1].scatter([], [], marker="^", facecolor="none", edgecolor=INK2,
                    label="S had not decayed below 1/e\nin the 32 ns window\n(value is a lower bound)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Water residence on a common 40 ps time axis, all fourteen systems",
                 x=0.005, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = f"{FIGS}/fig_residence_corrected.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    t = np.arange(800) * DT_NS
    for ax, layer, ttl in zip(axes, ("inner", "outer"),
                              ("A   inner core", "B   outer interfacial layer")):
        seen = set()
        for lbl in ORDER:
            fs = sorted(glob.glob(f"{CKPT}/{lbl}_*_{layer}.npy"))
            if not fs:
                continue
            S = np.mean([np.load(f) for f in fs], axis=0)
            k = KLASS[lbl]
            ax.plot(t, np.maximum(S, 1e-4), lw=1.3,
                    color=CLASS_COLOR[k], alpha=0.75,
                    label=k if k not in seen else None)
            seen.add(k)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.4)
        ax.set_xlabel(r"$\tau$  (ns)")
        ax.set_ylabel(r"$S(\tau)$")
        ax.set_title(ttl, loc="left")
        ax.axhline(1 / np.e, color=INK2, ls=":", lw=0.8)
        ax.text(t[-1], 1 / np.e, " 1/e", va="center", fontsize=7, color=INK2)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Continuous survival probability, identical 40 ps sampling for every system",
                 x=0.005, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = f"{FIGS}/fig_survival_curves.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_before_after(df):
    """Class means, published axis vs corrected axis, showing the inversion."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, layer, pub in zip(axes, ("inner", "outer"),
                              (PUBLISHED_INNER, PUBLISHED_OUTER)):
        s = df[df.layer == layer].set_index("system")
        classes = ["3-residue", "9-residue", "patterned"]
        pub_mean = [np.mean([pub[l] for l in ORDER if KLASS[l] == k]) for k in classes]
        cor_mean = [s.reindex([l for l in ORDER if KLASS[l] == k]).tau_int_ns.mean()
                    for k in classes]
        # normalise each series to its own 3-residue value: the CLAIM is about the
        # ratio between classes, and the two series carry different units
        pn = [v / pub_mean[0] for v in pub_mean]
        cn = [v / cor_mean[0] for v in cor_mean]
        for i, k in enumerate(classes):
            ax.plot([0, 1], [pn[i], cn[i]], "-o", lw=1.6, ms=7,
                    color=CLASS_COLOR[k], label=k, zorder=3)
            ax.annotate(f"{cn[i]:.1f}x", (1, cn[i]), textcoords="offset points",
                        xytext=(8, 0), va="center", fontsize=8, color=INK)
        ax.axhline(1.0, color=GRID, lw=0.8, zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["as published\n(mixed 1 ps / 40 ps)",
                            "corrected\n(common 40 ps)"])
        ax.set_yscale("log")
        ax.set_ylabel("class mean, relative to 3-residue")
        ax.set_title(f"{'AB'[0 if layer=='inner' else 1]}   {layer} layer", loc="left")
        ax.set_xlim(-0.25, 1.35)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Effect of putting every system on one time axis",
                 x=0.005, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = f"{FIGS}/fig_published_vs_corrected.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_water_structure():
    f = os.path.join(HERE, "results", "water_structure.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f).set_index("system").reindex(ORDER).dropna(how="all")
    labels = list(df.index)
    x = np.arange(len(labels))
    regions = ["inner", "outer", "bulk"]
    # sequential-by-depth: one hue, light to dark, since region is ordinal
    depth = {"bulk": "#c6dbef", "outer": "#6baed6", "inner": "#08519c"}
    specs = [("n_HB", "hydrogen bonds per water"),
             ("n_coord", r"OW neighbours within 3.5 $\AA$"),
             ("q", "tetrahedral order parameter $q$")]
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 8.0), sharex=True)
    w = 0.27
    for ax, (key, ylab) in zip(axes, specs):
        for i, r in enumerate(regions):
            ax.bar(x + (i - 1) * w, df[f"{key}_{r}"], width=w * 0.88,
                   yerr=df[f"{key}_{r}_sd"], color=depth[r], label=r, zorder=3,
                   error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        class_separators(ax, labels)
        ax.set_ylabel(ylab)
        style(ax)
        lo = np.nanmin([df[f"{key}_{r}"].min() for r in regions])
        ax.set_ylim(max(0, lo * 0.85), None)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=45, ha="right")
    fig.suptitle("Per-molecule water structure by region: replacement for the "
                 "retracted dielectric panel", x=0.005, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = f"{FIGS}/fig_water_structure.png"
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(HERE, "results", "survival_summary.csv"))
    for p in (fig_residence(d), fig_curves(), fig_before_after(d), fig_water_structure()):
        print("wrote", p if p else "(water_structure.csv absent, panel skipped)")
