"""Continuous radial density profiles of peptide and water from condensate COM.

For each frame:
  1. Identify the largest peptide cluster via DBSCAN on peptide COMs.
  2. Compute that cluster's center of mass (cluster_COM).
  3. Bin peptide heavy-atom and water-O positions by distance r from cluster_COM.
  4. Convert counts to spherical-shell densities ρ(r) = N / (4π r² Δr).

Aggregates over N_FRAMES frames per system and stores the resulting
profiles for plotting.
"""

import os
import warnings
import numpy as np
import pandas as pd
import MDAnalysis as mda
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

GXG_ROOT = "/Volumes/potoyan-lab/will/gxg-project/gxg_comps"
OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots/data"
N_FRAMES = 10
DBSCAN_EPS = 8.0
DBSCAN_MINSAMPLES = 4
R_MAX = 40.0   # Å, radial profile extent
DR = 1.0       # Å, radial bin width

SYSTEMS = {
    "9E":  f"{GXG_ROOT}/9_res/equal",
    "9C":  f"{GXG_ROOT}/9_res/charged",
    "9A":  f"{GXG_ROOT}/9_res/aromatic",
    "9P":  f"{GXG_ROOT}/9_res/polar",
    "9N":  f"{GXG_ROOT}/9_res/nonpolar",
    "X2":  f"{GXG_ROOT}/9_res/equal/2_close",
    "X2S": f"{GXG_ROOT}/9_res/equal/2_spread",
    "X3":  f"{GXG_ROOT}/9_res/equal/3_close",
    "X3S": f"{GXG_ROOT}/9_res/equal/3_spread",
}


def _shell_volumes(edges):
    return (4 / 3) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def analyze_system(label, base, n_frames=N_FRAMES):
    print(f"  {label}: loading…", flush=True)
    u = mda.Universe(f"{base}/output.pdb", f"{base}/traj.dcd")
    pep_heavy = u.select_atoms(
        "not name H* and not resname HOH SOL TIP3 WAT TIP")
    water_o = u.select_atoms("name OW or (resname HOH and name O)")
    if water_o.n_atoms == 0:
        water_o = u.select_atoms("resname HOH SOL TIP3 WAT TIP and name O")
    seg_idx = pep_heavy.segindices.copy()
    uniq_segs = np.unique(seg_idx)
    n_pep = len(uniq_segs)
    # remap arbitrary segindex values to 0..n_pep-1
    seg_remap = {s: i for i, s in enumerate(uniq_segs)}
    seg_idx_compact = np.array([seg_remap[s] for s in seg_idx])
    seg_to_atoms = {i: np.where(seg_idx_compact == i)[0]
                    for i in range(n_pep)}

    start = u.trajectory.n_frames // 2
    frames = np.linspace(start, u.trajectory.n_frames - 1, n_frames,
                         dtype=int)
    edges = np.arange(0, R_MAX + DR, DR)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shells = _shell_volumes(edges)

    pep_density_acc = np.zeros_like(centers)
    water_density_acc = np.zeros_like(centers)
    frames_used = 0

    for frame in frames:
        u.trajectory[int(frame)]
        pep_pos = pep_heavy.positions
        water_pos = water_o.positions
        # peptide COMs for clustering
        coms = np.zeros((n_pep, 3))
        for s, ais in seg_to_atoms.items():
            coms[s] = pep_pos[ais].mean(axis=0)
        labels = DBSCAN(eps=DBSCAN_EPS,
                        min_samples=DBSCAN_MINSAMPLES).fit_predict(coms)
        from collections import Counter
        c = Counter(labels[labels >= 0])
        if not c:
            continue
        largest = c.most_common(1)[0][0]
        cluster_pep = coms[labels == largest]
        cluster_com = cluster_pep.mean(axis=0)
        # distances
        r_pep = np.linalg.norm(pep_pos - cluster_com, axis=1)
        r_water = np.linalg.norm(water_pos - cluster_com, axis=1)
        pep_counts, _ = np.histogram(r_pep, bins=edges)
        water_counts, _ = np.histogram(r_water, bins=edges)
        pep_density_acc += pep_counts / shells
        water_density_acc += water_counts / shells
        frames_used += 1

    if frames_used == 0:
        return None
    pep_density_acc /= frames_used
    water_density_acc /= frames_used
    return pd.DataFrame({
        "r": centers,
        "rho_pep_heavy": pep_density_acc,   # atoms / Å³
        "rho_water_O":   water_density_acc, # waters / Å³
    })


def run_all():
    os.makedirs(OUT, exist_ok=True)
    all_dfs = {}
    for label, base in SYSTEMS.items():
        df = analyze_system(label, base)
        if df is not None:
            df["system"] = label
            all_dfs[label] = df
    combined = pd.concat(all_dfs.values(), ignore_index=True)
    combined.to_csv(os.path.join(OUT, "24_radial_profiles.csv"), index=False)
    print("Saved radial profiles to", OUT)
    return combined


if __name__ == "__main__":
    df = run_all()
    print(df.head(20))
