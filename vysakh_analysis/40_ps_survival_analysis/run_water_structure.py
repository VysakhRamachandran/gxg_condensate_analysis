#!/usr/bin/env python
"""Per-molecule water structure for all fourteen systems: the replacement panel
for the retracted dielectric constant.

These are instantaneous structural quantities, so a handful of well-separated
frames is enough and this runs in a couple of minutes, unlike the survival
analysis. No new simulations are needed.

Output: results/water_structure.csv with n_HB, n_coord and q for inner, outer and
bulk water in every system, plus the spread across frames.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import numpy as np
import pandas as pd
import parmed
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.coordinates.DCD import DCDReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.systems import SYSTEMS
from src.layers import largest_cluster, gitim_layers
from src.survival import DIST_WATER
from src.water_structure import hbonds_per_water, coordination_per_water, tetrahedral_q

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

N_FRAMES = 8          # well-separated frames per system
BULK_EXCL = 8.0       # Angstrom from any peptide atom


def region_waters(u, ts, resindices, ow):
    """OW indices within DIST_WATER of the given peptide residues."""
    ag = u.atoms[np.isin(u.atoms.resindices, resindices)]
    if len(ag) == 0:
        return np.array([], dtype=int)
    p = capped_distance(ow.positions, ag.positions, max_cutoff=DIST_WATER,
                        box=ts.dimensions, return_distances=False)
    return np.unique(p[:, 0])


def main():
    rows = []
    for lbl, d, traj, klass in SYSTEMS:
        nfr = DCDReader(os.path.join(d, traj)).n_frames
        frames = np.linspace(int(0.55 * nfr), nfr - 2, N_FRAMES).astype(int)
        stru = parmed.load_file(f"{d}/output.top", xyz=f"{d}/output.gro")
        u = mda.Universe(stru, f"{d}/{traj}")
        ow = u.select_atoms("name OW")
        hw = u.select_atoms("name HW1 or name HW2")
        prot = u.select_atoms("protein")
        acc = {k: {m: [] for m in ("n_HB", "n_coord", "q")}
               for k in ("inner", "outer", "bulk")}
        t0 = time.time()
        for f in frames:
            ts = u.trajectory[int(f)]
            cluster = largest_cluster(u, ts)
            inner, mid, outer = gitim_layers(u, ts, cluster, tmpdir=RES)
            ow_pos = ow.positions
            hw_pos = hw.positions.reshape(len(ow), 2, 3)
            idx = {"inner": region_waters(u, ts, inner, ow),
                   "outer": region_waters(u, ts, outer, ow)}
            far = capped_distance(ow_pos, prot.positions, max_cutoff=BULK_EXCL,
                                  box=ts.dimensions, return_distances=False)
            near = np.unique(far[:, 0])
            idx["bulk"] = np.setdiff1d(np.arange(len(ow)), near)
            # Subsampling the bulk set is safe here ONLY because every observable
            # searches partners among ALL water; a random subset is then an unbiased
            # sample of a per-molecule mean.
            if len(idx["bulk"]) > 2000:
                idx["bulk"] = np.random.default_rng(0).choice(idx["bulk"], 2000,
                                                              replace=False)
            for k, ii in idx.items():
                if len(ii) == 0:
                    continue
                acc[k]["n_HB"].append(hbonds_per_water(ii, ow_pos, hw_pos,
                                                       ts.dimensions))
                acc[k]["n_coord"].append(coordination_per_water(ow_pos[ii], ow_pos,
                                                                ts.dimensions))
                acc[k]["q"].append(tetrahedral_q(ow_pos[ii], ow_pos, ts.dimensions))
        row = {"system": lbl, "class": klass, "n_frames": len(frames)}
        for k in ("inner", "outer", "bulk"):
            for m in ("n_HB", "n_coord", "q"):
                v = acc[k][m]
                row[f"{m}_{k}"] = np.mean(v) if v else np.nan
                row[f"{m}_{k}_sd"] = np.std(v) if v else np.nan
        rows.append(row)
        print(f"{lbl:4s} ({time.time()-t0:.0f}s)  "
              f"nHB in/out/bulk = {row['n_HB_inner']:.2f}/{row['n_HB_outer']:.2f}/"
              f"{row['n_HB_bulk']:.2f}   q = {row['q_inner']:.3f}/"
              f"{row['q_outer']:.3f}/{row['q_bulk']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "water_structure.csv"), index=False)
    print("\nwrote results/water_structure.csv")


if __name__ == "__main__":
    main()
