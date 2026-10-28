"""Composite Figure 1: system design overview.

Layout:
  (A)  Chain-topology schematics for the 14 systems, grouped by axis
       (composition, chain-length, sticker pattern). G = grey beads,
       X = colored bead by chemistry class.
  (B)  Systems table (taken from intro.tex but redrawn for visual clarity).
  (C)  Three representative condensate snapshots (3C, 9C, X3S) showing
       composition, chain-length, and patterning axes.

Snapshots are produced by src/render_snapshots.py via PyMOL and saved to
manuscript_figs/snapshots/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle
import matplotlib.image as mpimg

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots"
SNAP = os.path.join(OUT, "manuscript_figs", "snapshots")

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 11, "axes.titlesize": 11,
})

# colors per chemistry class (matching residue groups elsewhere)
CLASS_FILL = {
    "E": "#888888",   # mixed/equimolar
    "C": "#3a6ba6",   # charged
    "A": "#c0504d",   # aromatic
    "P": "#4f8c4a",   # polar
    "N": "#d68433",   # non-polar
}

# system definitions: each is (label, chain layout, class)
# chain layout = list of residues, "G" or "X" (X colored by class)
SYSTEMS = [
    # 3-residue single-sticker
    ("3E",  list("GXG"), "E"),
    ("3C",  list("GXG"), "C"),
    ("3A",  list("GXG"), "A"),
    ("3P",  list("GXG"), "P"),
    ("3N",  list("GXG"), "N"),
    # 9-residue single-sticker
    ("9E",  list("GGGGXGGGG"), "E"),
    ("9C",  list("GGGGXGGGG"), "C"),
    ("9A",  list("GGGGXGGGG"), "A"),
    ("9P",  list("GGGGXGGGG"), "P"),
    ("9N",  list("GGGGXGGGG"), "N"),
    # patterned (equimolar over all 20 X identities)
    ("X2",  list("GGGXGXGGG"), "E"),
    ("X2S", list("GXGGGGGXG"), "E"),
    ("X3",  list("GGGXXXGGG"), "E"),
    ("X3S", list("GXGGXGGXG"), "E"),
]


def draw_chain(ax, x0, y0, layout, cls, bead_d=0.6, gap=0.10):
    """Draw a chain as a row of beads at (x0, y0) baseline.
    The bead is drawn as a scatter point so it stays circular regardless
    of axis aspect ratio (matplotlib Circle distorts when aspect != equal)."""
    for i, r in enumerate(layout):
        cx = x0 + i * (bead_d + gap)
        cy = y0
        if r == "X":
            fc = CLASS_FILL[cls]
            ec = "k"
            lw = 0.8
        else:
            fc = "#dadada"
            ec = "#888"
            lw = 0.3
        ax.scatter(cx, cy, s=180, facecolor=fc, edgecolor=ec,
                   linewidths=lw, zorder=2)


def panel_a_schematics(ax):
    """Draw chain schematics for all 14 systems in 3 column groups."""
    # axis coordinate system: 30 wide; do NOT force equal aspect — let the
    # panel fill its allotted width.
    ax.set_xlim(0, 30)
    ax.set_ylim(2.0, 16.0)
    ax.axis("off")
    # column starting positions for the 3 groups (beads start here)
    col_x = {"3res": 2.5, "9res": 11.5, "pat": 22.0}
    # group headers — centered above the BEAD area for that group
    bead_d = 0.6; gap = 0.05
    # 3-res chain spans 3 beads, 9-res chain spans 9 beads, patterned 9
    def chain_width(n):
        return n * bead_d + (n - 1) * gap
    # column headers (slightly lower than before so they sit clear of title)
    ax.text(col_x["3res"] + chain_width(3) / 2, 13.6,
            "single-sticker · 3-residue",
            ha="center", fontsize=11, fontweight="bold", color="#444")
    ax.text(col_x["9res"] + chain_width(9) / 2, 13.6,
            "single-sticker · 9-residue",
            ha="center", fontsize=11, fontweight="bold", color="#444")
    ax.text(col_x["pat"] + chain_width(9) / 2, 13.6,
            "patterned · 9-residue",
            ha="center", fontsize=11, fontweight="bold", color="#444")

    # 5 rows of chains, packed tightly
    row_top = 12.4
    row_gap = 1.4
    for i, (lbl, layout, cls) in enumerate(SYSTEMS[:5]):
        y = row_top - row_gap * i
        ax.text(col_x["3res"] - 0.5, y, lbl, ha="right", va="center",
                fontsize=11, fontweight="bold", color=CLASS_FILL[cls])
        draw_chain(ax, col_x["3res"], y, layout, cls)
    for i, (lbl, layout, cls) in enumerate(SYSTEMS[5:10]):
        y = row_top - row_gap * i
        ax.text(col_x["9res"] - 0.5, y, lbl, ha="right", va="center",
                fontsize=11, fontweight="bold", color=CLASS_FILL[cls])
        draw_chain(ax, col_x["9res"], y, layout, cls)
    for i, (lbl, layout, cls) in enumerate(SYSTEMS[10:]):
        y = row_top - row_gap * i
        ax.text(col_x["pat"] - 0.5, y, lbl, ha="right", va="center",
                fontsize=11, fontweight="bold", color="#444")
        draw_chain(ax, col_x["pat"], y, layout, "E")

    # chemistry-class legend at bottom (across full width)
    legend_items = [
        ("G  glycine spacer",            "#dadada"),
        ("E  equimolar 20-AA",           CLASS_FILL["E"]),
        ("C  charged (R,K,D,E,H)",       CLASS_FILL["C"]),
        ("A  aromatic (F,Y,W)",          CLASS_FILL["A"]),
        ("P  polar (S,T,N,Q,C)",         CLASS_FILL["P"]),
        ("N  non-polar (A,V,L,I,M,G,P)", CLASS_FILL["N"]),
    ]
    # legend placed close to bottom of chain rows (was a big white gap)
    leg_y = 4.4
    for i, (lbl, color) in enumerate(legend_items):
        row, col = i // 3, i % 3
        cx = 1.0 + col * 10.0
        cy = leg_y - row * 1.1
        ax.scatter(cx, cy, s=160, facecolor=color, edgecolor="k",
                   linewidths=0.5)
        ax.text(cx + 0.6, cy, lbl, ha="left", va="center", fontsize=10)
    # panel title — sits well above headers
    ax.text(0, 15.4,
            "(A)  Chain designs for the 14 simulated peptide systems",
            ha="left", fontsize=12, fontweight="normal")


def panel_b_snapshots(axes, system_labels):
    """Mount three rendered snapshots as subplots."""
    for ax, sys in zip(axes, system_labels):
        png = os.path.join(SNAP, f"snap_{sys}.png")
        if os.path.exists(png):
            img = mpimg.imread(png)
            ax.imshow(img)
            ax.set_title(f"{sys}", fontsize=12, fontweight="bold",
                         color=CLASS_FILL.get(sys[1] if sys.startswith("3")
                                              or sys.startswith("9") else "E",
                                              "#444"))
        else:
            ax.text(0.5, 0.5, f"snapshot missing\n{sys}",
                    ha="center", va="center", transform=ax.transAxes,
                    color="grey")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].text(-0.05, 1.15,
                 "(B)  Representative condensate snapshots",
                 transform=axes[0].transAxes,
                 fontsize=12, ha="left")


def main():
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0],
                          hspace=0.30, wspace=0.10)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = [fig.add_subplot(gs[1, j]) for j in range(3)]
    panel_a_schematics(ax_a)
    panel_b_snapshots(ax_b, ["3C", "9C", "X3S"])
    out = os.path.join(OUT, "manuscript_figs", "Fig1.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
