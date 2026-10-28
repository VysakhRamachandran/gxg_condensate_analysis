"""Position-resolved sticker engagement.

For each system, for each (peptide, residue-position-within-chain) pair,
count the number of *distinct partner peptides* the residue makes inter-
chain heavy-atom contacts with (cutoff 4.5 Å, ≥3 atom-pair support).

Aggregates over chains and frames; writes per-system per-position summary.

Answers: in X3S (stickers at positions 2,5,8) are all three equally engaged,
or does one dominate? In X3 (positions 4,5,6) is the middle sticker buried?
"""

import os
import warnings
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import distances

warnings.filterwarnings("ignore")

GXG_ROOT = "/Volumes/potoyan-lab/will/gxg-project/gxg_comps"
OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots/data"

CUTOFF = 4.5
MIN_EDGE_WEIGHT = 3
N_FRAMES = 30

# we only need 9-residue systems (single sticker at 5, plus the patterned
# multi-sticker ones). The 3-residue systems have a single sticker so per-
# position analysis is uninteresting; we skip them here.
SYSTEMS = {
    "9E":  (f"{GXG_ROOT}/9_res/equal",            (5,)),
    "9C":  (f"{GXG_ROOT}/9_res/charged",          (5,)),
    "9A":  (f"{GXG_ROOT}/9_res/aromatic",         (5,)),
    "9P":  (f"{GXG_ROOT}/9_res/polar",            (5,)),
    "9N":  (f"{GXG_ROOT}/9_res/nonpolar",         (5,)),
    "X2":  (f"{GXG_ROOT}/9_res/equal/2_close",    (4, 6)),
    "X2S": (f"{GXG_ROOT}/9_res/equal/2_spread",   (2, 8)),
    "X3":  (f"{GXG_ROOT}/9_res/equal/3_close",    (4, 5, 6)),
    "X3S": (f"{GXG_ROOT}/9_res/equal/3_spread",   (2, 5, 8)),
}


def analyze_system(label, base, sticker_positions, n_frames=N_FRAMES):
    print(f"  {label}: loading…", flush=True)
    u = mda.Universe(f"{base}/output-nw.pdb", f"{base}/traj-nw.dcd")
    heavy = u.select_atoms(
        "not name H* and not resname HOH SOL TIP3 WAT TIP")
    n_atoms = heavy.n_atoms
    seg_for_atom = heavy.segindices.copy()
    # residue position within chain — use resid from the PDB (1-indexed)
    resid_for_atom = heavy.resids.copy()
    n_pep = int(np.unique(seg_for_atom).size)

    start = u.trajectory.n_frames // 2
    frames = np.linspace(start, u.trajectory.n_frames - 1, n_frames,
                         dtype=int)

    # accumulators
    per_frame_counts = {p: {pos: [] for pos in sticker_positions}
                        for p in range(n_pep)}
    # union counts: distinct partner peptides across ALL stickers of this peptide
    per_frame_union = {p: [] for p in range(n_pep)}
    # pairwise overlap: avg |A ∩ B| / |A ∪ B| Jaccard across sticker pairs
    per_frame_jaccard = {p: [] for p in range(n_pep)}

    for frame in frames:
        u.trajectory[int(frame)]
        pairs, _ = distances.capped_distance(
            heavy.positions, heavy.positions, CUTOFF, box=u.dimensions)
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        seg_i = seg_for_atom[pairs[:, 0]]
        seg_j = seg_for_atom[pairs[:, 1]]
        inter = seg_i != seg_j
        pairs = pairs[inter]; seg_i = seg_i[inter]; seg_j = seg_j[inter]
        # peptide-pair weight: how many atom-pair contacts support each
        # (peptide_a, peptide_b) edge in this frame. Drop pairs whose
        # backing peptide-edge has < MIN_EDGE_WEIGHT.
        df = pd.DataFrame({"i": seg_i, "j": seg_j,
                           "ai": pairs[:, 0], "aj": pairs[:, 1]})
        df["pep_pair"] = np.minimum(df.i, df.j).astype(int) * 10000 + \
                         np.maximum(df.i, df.j).astype(int)
        weights = df.groupby("pep_pair").size()
        ok_edges = set(weights[weights >= MIN_EDGE_WEIGHT].index.tolist())
        df = df[df["pep_pair"].isin(ok_edges)]
        if df.empty:
            for p in range(n_pep):
                for pos in sticker_positions:
                    per_frame_counts[p][pos].append(0)
                per_frame_union[p].append(0)
                per_frame_jaccard[p].append(np.nan)
            continue
        ri = resid_for_atom[df.ai.values]
        rj = resid_for_atom[df.aj.values]
        long_a = pd.DataFrame({"pep": df.i.values, "pos": ri,
                               "partner": df.j.values})
        long_b = pd.DataFrame({"pep": df.j.values, "pos": rj,
                               "partner": df.i.values})
        long_all = pd.concat([long_a, long_b], ignore_index=True)
        long_all = long_all[long_all["pos"].isin(sticker_positions)]
        # distinct partner per (pep, pos)
        partner_counts = (long_all.groupby(["pep", "pos"])["partner"]
                          .nunique()).to_dict()
        # also: full partner SET per (pep, pos) — for union/Jaccard
        partner_sets = (long_all.groupby(["pep", "pos"])["partner"]
                        .apply(lambda s: set(s.tolist()))).to_dict()
        for p in range(n_pep):
            for pos in sticker_positions:
                per_frame_counts[p][pos].append(
                    partner_counts.get((p, pos), 0))
            # union of all sticker sets for this peptide
            sets_p = [partner_sets.get((p, pos), set())
                      for pos in sticker_positions]
            union_set = set().union(*sets_p) if sets_p else set()
            per_frame_union[p].append(len(union_set))
            # mean pairwise Jaccard across distinct sticker pairs
            if len(sticker_positions) >= 2:
                jacs = []
                for ai in range(len(sticker_positions)):
                    for bj in range(ai + 1, len(sticker_positions)):
                        sa, sb = sets_p[ai], sets_p[bj]
                        if not (sa or sb):
                            continue
                        jacs.append(len(sa & sb) / len(sa | sb))
                per_frame_jaccard[p].append(
                    float(np.mean(jacs)) if jacs else np.nan)
            else:
                per_frame_jaccard[p].append(np.nan)

    rows = []
    for p in range(n_pep):
        for pos in sticker_positions:
            arr = np.array(per_frame_counts[p][pos])
            rows.append({"system": label, "peptide": p, "position": pos,
                         "mean_partners": float(arr.mean()),
                         "engaged_frac": float((arr > 0).mean()),
                         "n_frames": len(arr)})
    diversification = []
    for p in range(n_pep):
        union = np.array(per_frame_union[p])
        jac = np.array(per_frame_jaccard[p])
        sum_per_sticker = np.sum([
            np.array(per_frame_counts[p][pos]) for pos in sticker_positions
        ], axis=0)
        # diversification ratio = union / sum (per frame); avg over frames
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(sum_per_sticker > 0,
                             union / sum_per_sticker, np.nan)
        diversification.append({
            "system": label, "peptide": p,
            "union_partners": float(union.mean()),
            "sum_partners":   float(sum_per_sticker.mean()),
            "diversification_ratio": float(np.nanmean(ratio))
                                       if np.any(np.isfinite(ratio)) else np.nan,
            "mean_jaccard":  float(np.nanmean(jac))
                                       if np.any(np.isfinite(jac)) else np.nan,
        })
    return pd.DataFrame(rows), pd.DataFrame(diversification)


def run_all():
    parts, div_parts = [], []
    for label, (base, positions) in SYSTEMS.items():
        df, div = analyze_system(label, base, positions)
        parts.append(df); div_parts.append(div)
    out = pd.concat(parts, ignore_index=True)
    divs = pd.concat(div_parts, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "15_sticker_engagement.csv"), index=False)
    divs.to_csv(os.path.join(OUT, "15_sticker_diversification.csv"),
                index=False)
    summary = (out.groupby(["system", "position"])
                  .agg(mean_partners=("mean_partners", "mean"),
                       sem_partners=("mean_partners",
                                     lambda v: float(v.std() / np.sqrt(len(v)))),
                       engaged_frac=("engaged_frac", "mean"),
                       n_peptides=("peptide", "nunique"))).reset_index()
    summary.to_csv(os.path.join(OUT, "15_sticker_engagement_summary.csv"),
                   index=False)
    div_summary = (divs.groupby("system")
                       .agg(union_partners=("union_partners", "mean"),
                            sum_partners=("sum_partners", "mean"),
                            diversification_ratio=("diversification_ratio", "mean"),
                            mean_jaccard=("mean_jaccard", "mean"),
                            n_peptides=("peptide", "nunique"))).reset_index()
    div_summary.to_csv(
        os.path.join(OUT, "15_sticker_diversification_summary.csv"),
        index=False)
    print("Saved sticker engagement tables to", OUT)
    return summary, div_summary


if __name__ == "__main__":
    summary, div = run_all()
    pd.set_option("display.float_format", "{:.2f}".format)
    print("\n--- per-position summary ---")
    print(summary.to_string(index=False))
    print("\n--- per-peptide diversification summary ---")
    print(div.to_string(index=False))
