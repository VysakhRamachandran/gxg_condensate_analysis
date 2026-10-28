"""Residue-pair contact enrichment matrix per system.

For each system we count inter-peptide heavy-atom contacts at 4.5 Å and
aggregate by (residue type A, residue type B). Enrichment is

    E_AB = obs_AB / expected_AB(mole-fractions, random mixing)

For random mixing, expected pair frequency is f_A * f_B for distinct
types, or f_A^2 for self-pairs (with a factor-of-2 correction so the
contact total matches).

We restrict to a single shared cutoff so all systems are comparable.
"""

import os
import warnings
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import distances
from collections import Counter
from .sequences import SYSTEMS, AA3to1
from .contact_network import system_paths

warnings.filterwarnings("ignore")

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots/data"
CUTOFF = 4.5
N_FRAMES = 30

AA20 = list("ARNDCEQGHILKMFPSTWYV")  # standard order
AA_INDEX = {a: i for i, a in enumerate(AA20)}


def mole_fractions_for_system(sys_label):
    """Residue mole fractions across the peptide mixture."""
    mix = SYSTEMS[sys_label]
    counts = Counter()
    for seq, n in mix.items():
        for r in seq:
            counts[r] += n
    total = sum(counts.values())
    return {a: counts.get(a, 0) / total for a in AA20}


def analyze_system(label, pdb, traj, n_frames=N_FRAMES):
    print(f"  {label}: loading…", flush=True)
    u = mda.Universe(pdb, traj)
    heavy = u.select_atoms(
        "not name H* and not resname HOH SOL TIP3 WAT TIP")
    # Per-atom: residue 1-letter code (skip non-canonical)
    res3 = heavy.resnames
    res1_for_atom = np.array(
        [AA3to1.get(r, "?") for r in res3])
    seg_for_atom = heavy.segindices.copy()
    type_idx_for_atom = np.array([AA_INDEX.get(r, -1)
                                   for r in res1_for_atom])

    start = u.trajectory.n_frames // 2
    frames = np.linspace(start, u.trajectory.n_frames - 1, n_frames,
                         dtype=int)
    pair_counts = np.zeros((20, 20), dtype=np.int64)
    for frame in frames:
        u.trajectory[int(frame)]
        pairs, _ = distances.capped_distance(
            heavy.positions, heavy.positions, CUTOFF, box=u.dimensions)
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        inter = seg_for_atom[pairs[:, 0]] != seg_for_atom[pairs[:, 1]]
        pairs = pairs[inter]
        ti = type_idx_for_atom[pairs[:, 0]]
        tj = type_idx_for_atom[pairs[:, 1]]
        ok = (ti >= 0) & (tj >= 0)
        ti, tj = ti[ok], tj[ok]
        # symmetrize
        a = np.minimum(ti, tj); b = np.maximum(ti, tj)
        # add to upper triangle then mirror at the end
        np.add.at(pair_counts, (a, b), 1)
    # mirror upper to lower (so matrix is symmetric); diagonal counted once
    pair_counts = pair_counts + pair_counts.T - np.diag(pair_counts.diagonal())

    # expected: random mixing over mole fractions
    f = np.array([mole_fractions_for_system(label)[a] for a in AA20])
    total = pair_counts.sum() / 2  # unordered pairs
    # expected unordered pair count between i,j (i != j): 2*f_i*f_j*total
    # expected self pair: f_i^2 * total
    expected = np.outer(f, f) * total * 2
    np.fill_diagonal(expected, f**2 * total)
    # for the symmetric matrix store the observed COUNT and the expected count
    with np.errstate(divide="ignore", invalid="ignore"):
        enrichment = np.where(expected > 0, pair_counts / expected, np.nan)
    return pair_counts, expected, enrichment, f


def run_all():
    os.makedirs(OUT, exist_ok=True)
    paths = system_paths()
    all_results = {}
    for label, (pdb, traj) in paths.items():
        obs, exp, enr, f = analyze_system(label, pdb, traj)
        all_results[label] = {"obs": obs, "exp": exp,
                              "enr": enr, "f": f}
        df_enr = pd.DataFrame(enr, index=AA20, columns=AA20)
        df_obs = pd.DataFrame(obs, index=AA20, columns=AA20)
        df_enr.to_csv(os.path.join(OUT, f"resmat_enr_{label}.csv"))
        df_obs.to_csv(os.path.join(OUT, f"resmat_obs_{label}.csv"))
    # also: top-5 enriched pairs per system (excluding GG, weighted by total counts)
    top = []
    for label, d in all_results.items():
        e = d["enr"]; obs = d["obs"]
        # ignore G–G and pairs with obs < 20 (low statistics)
        mask = np.zeros_like(e, dtype=bool)
        for i in range(20):
            for j in range(i, 20):  # upper triangle including diagonal
                if AA20[i] == "G" and AA20[j] == "G":
                    continue
                if obs[i, j] < 20:
                    continue
                mask[i, j] = True
        idx = np.argwhere(mask & np.isfinite(e))
        ranks = sorted(idx.tolist(), key=lambda ij: -e[ij[0], ij[1]])
        for rank, (i, j) in enumerate(ranks[:5]):
            top.append({"system": label, "rank": rank + 1,
                        "pair": f"{AA20[i]}-{AA20[j]}",
                        "enrichment": float(e[i, j]),
                        "obs_count": int(obs[i, j])})
    pd.DataFrame(top).to_csv(
        os.path.join(OUT, "19_residue_pair_top.csv"), index=False)
    print("Saved residue-pair tables to", OUT)
    return all_results


if __name__ == "__main__":
    res = run_all()
