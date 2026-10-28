#!/usr/bin/env python
"""OPTION 1: recompute inner/outer water survival for all fourteen systems on a
single common 40 ps time axis.

Every (system, origin) is checkpointed to results/ckpt/ so the run is resumable
and a transient network-share failure cannot lose completed work. Re-running
skips whatever is already on disk.

    PYTHONPATH=<dir with pytim> python run_survival.py
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import numpy as np
import parmed
import MDAnalysis as mda

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.systems import SYSTEMS, DT_PS, NLAGS, origins
from src.layers import largest_cluster, gitim_layers
from src.survival import survival_multi

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "results", "ckpt")
os.makedirs(CKPT, exist_ok=True)

LAYERS = ("inner", "outer")


def main():
    t_start = time.time()
    for lbl, d, traj, klass in SYSTEMS:
        print(f"\n{'='*72}\n{lbl}  ({klass})  {d}/{traj}", flush=True)
        u = None
        stru = None
        # peek at frame count without loading the topology
        from MDAnalysis.coordinates.DCD import DCDReader
        nfr = DCDReader(os.path.join(d, traj)).n_frames
        for o in origins(nfr):
            tag = f"{lbl}_{o}"
            if all(os.path.exists(f"{CKPT}/{tag}_{n}.npy") for n in LAYERS):
                print(f"  origin {o:>6}: cached", flush=True)
                continue
            if u is None:
                stru = parmed.load_file(f"{d}/output.top", xyz=f"{d}/output.gro")
                u = mda.Universe(stru, f"{d}/{traj}")
                assert abs(u.trajectory.dt - DT_PS) < 1e-6, \
                    f"{lbl}: expected {DT_PS} ps, got {u.trajectory.dt}"
            t0 = time.time()
            ts = u.trajectory[o]
            cluster = largest_cluster(u, ts)
            inner, mid, outer = gitim_layers(u, ts, cluster, tmpdir=CKPT)
            S, n0 = survival_multi(u, {"inner": inner, "outer": outer}, o, NLAGS)
            for n in LAYERS:
                np.save(f"{CKPT}/{tag}_{n}.npy", S[n])
            np.save(f"{CKPT}/{tag}_meta.npy",
                    np.array([len(cluster), len(inner), len(mid), len(outer),
                              n0["inner"], n0["outer"], nfr]))
            print(f"  origin {o:>6}: cluster={len(cluster):>4} "
                  f"inner={len(inner):>4} mid={len(mid):>4} outer={len(outer):>4} | "
                  f"n0 inner={n0['inner']:>5} outer={n0['outer']:>5} | "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\nALL DONE in {(time.time()-t_start)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
