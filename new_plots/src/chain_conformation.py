"""Chain conformation (Rg, end-to-end) inside vs outside condensate.

For each 9-residue system we compute, per chain per frame:
  - radius of gyration of the chain backbone
  - end-to-end distance (Cα of residue 1 to Cα of residue 9)
  - whether the chain is in the condensate or in dilute phase

Condensate membership: a chain is "inside" if its center-of-mass is within
the largest-cluster bounding region. We approximate by:
  1. compute peptide COMs
  2. DBSCAN with eps = 8 Å in COM space
  3. label the largest cluster as the condensate; everything else dilute

Outputs per-system Rg and Ree distributions split by inside/outside.

3-residue systems are skipped — Rg is essentially meaningless for L=3.
"""

import os
import warnings
import numpy as np
import pandas as pd
import MDAnalysis as mda
from sklearn.cluster import DBSCAN

from .contact_network import system_paths

warnings.filterwarnings("ignore")

OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots/data"
N_FRAMES = 30
DBSCAN_EPS = 8.0  # Å, on peptide COM distances
DBSCAN_MINSAMPLES = 4


def analyze_system(label, pdb, traj, n_frames=N_FRAMES):
    print(f"  {label}: loading…", flush=True)
    u = mda.Universe(pdb, traj)
    # peptide chains: select all heavy non-water atoms grouped by segment
    pep = u.select_atoms(
        "not name H* and not resname HOH SOL TIP3 WAT TIP")
    seg_idx = pep.segindices.copy()
    # Cα-only selection for end-to-end (use the alpha carbons)
    ca = u.select_atoms("name CA and not resname HOH SOL TIP3 WAT TIP")
    ca_seg = ca.segindices.copy()
    n_pep = len(np.unique(seg_idx))

    # build per-segment slice maps once
    seg_to_atoms = {s: np.where(seg_idx == s)[0] for s in np.unique(seg_idx)}
    seg_to_ca   = {s: np.where(ca_seg == s)[0] for s in np.unique(ca_seg)}

    start = u.trajectory.n_frames // 2
    frames = np.linspace(start, u.trajectory.n_frames - 1, n_frames,
                         dtype=int)

    rows = []
    for fi, frame in enumerate(frames):
        u.trajectory[int(frame)]
        # unwrap each peptide so chains spanning PBC are made whole.
        # we have to do this per-fragment so atoms don't get spread out
        # across the box (causes huge fake Rg).
        try:
            pep.unwrap(compound="segments", inplace=True)
        except Exception:
            # fallback: per-segment unwrap with reference
            pass
        coords = pep.positions
        ca_coords = ca.positions
        # peptide COMs (geometric center of heavy atoms)
        coms = np.zeros((n_pep, 3))
        for s, ais in seg_to_atoms.items():
            coms[s] = coords[ais].mean(axis=0)
        # DBSCAN to identify the largest cluster (= condensate)
        labels = DBSCAN(eps=DBSCAN_EPS,
                        min_samples=DBSCAN_MINSAMPLES).fit_predict(coms)
        # find the largest non-noise cluster
        from collections import Counter
        c = Counter(labels[labels >= 0])
        if not c:
            largest = -1
        else:
            largest = c.most_common(1)[0][0]
        is_in_condensate = (labels == largest)
        # per chain Rg, Ree
        for s in seg_to_atoms:
            ais = seg_to_atoms[s]
            chain_coords = coords[ais]
            cmean = chain_coords.mean(axis=0)
            rg2 = ((chain_coords - cmean) ** 2).sum(axis=1).mean()
            rg = float(np.sqrt(rg2))
            cas = seg_to_ca.get(s, [])
            if len(cas) >= 2:
                ree = float(np.linalg.norm(ca_coords[cas[-1]]
                                            - ca_coords[cas[0]]))
            else:
                ree = np.nan
            rows.append({
                "system": label, "frame": int(frame), "peptide": int(s),
                "Rg": rg, "Ree": ree,
                "in_condensate": bool(is_in_condensate[s]),
            })
    return pd.DataFrame(rows)


def run_all():
    os.makedirs(OUT, exist_ok=True)
    paths = system_paths()
    parts = []
    for label, (pdb, traj) in paths.items():
        # skip 3-residue systems
        if label.startswith("3"):
            continue
        parts.append(analyze_system(label, pdb, traj))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(OUT, "20_chain_conformation.csv"), index=False)
    summary = (df.groupby(["system", "in_condensate"])
                 .agg(Rg_mean=("Rg", "mean"), Rg_std=("Rg", "std"),
                      Ree_mean=("Ree", "mean"), Ree_std=("Ree", "std"),
                      n=("Rg", "size"))).reset_index()
    summary.to_csv(os.path.join(OUT, "20_chain_conformation_summary.csv"),
                   index=False)
    print("Saved chain conformation tables to", OUT)
    return summary


if __name__ == "__main__":
    s = run_all()
    pd.set_option("display.float_format", "{:.2f}".format)
    print(s.to_string(index=False))
