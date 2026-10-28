"""Water structure inside vs outside condensate.

For each system, per frame, classify each water O as:
    'inner'      — within R_INNER of cluster COM
    'interface'  — R_INNER ≤ r < R_OUTER
    'bulk'       — r ≥ R_OUTER   (i.e. comfortably outside condensate)

For each class we compute:
    g_OO(r)            — water-water radial distribution function
    tetrahedral q      — Errington-Debenedetti order parameter
                         q = 1 - (3/8) Σ_{j<k}(cos θ_jk + 1/3)²
                         taken over the 4 nearest water-O neighbors.

Tip: q ≈ 0.95 for perfect ice; bulk liquid water at 300 K is ~0.5–0.55
for TIP3P. q < bulk indicates LESS tetrahedral structure (more "broken"
H-bond network).
"""

import os
import warnings
import numpy as np
import pandas as pd
import MDAnalysis as mda
from scipy.spatial import cKDTree
from MDAnalysis.lib import distances
from sklearn.cluster import DBSCAN
from collections import Counter

warnings.filterwarnings("ignore")

GXG_ROOT = "/Volumes/potoyan-lab/will/gxg-project/gxg_comps"
OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots/data"
N_FRAMES = 5  # water structure is expensive; fewer frames suffice
DBSCAN_EPS = 8.0
DBSCAN_MINSAMPLES = 4
R_INNER = 15.0   # Å, "inner core" radius from cluster COM
R_OUTER = 30.0   # Å, "interface" outer edge
R_BULK_MIN = 40.0  # Å, waters at r ≥ this are considered bulk reference
G_R_MAX = 8.0
DR_G = 0.1

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


def tetrahedral_q(positions, indices, tree, box_dim=None):
    """Compute q for each water O specified by `indices`.
    `positions` is the full water-O array; `tree` is a KDTree on it."""
    qs = np.full(len(indices), np.nan)
    # find 5 nearest neighbors (including self); skip self
    dists, neigh = tree.query(positions[indices], k=5)
    for i, nbs in enumerate(neigh):
        ref = positions[indices[i]]
        # drop self (first entry should be self with d=0)
        nbs4 = nbs[1:5]
        vecs = positions[nbs4] - ref  # (4,3)
        # MIC: shift each component to nearest image
        if box_dim is not None:
            L = box_dim[:3]
            vecs -= L * np.round(vecs / L)
        norms = np.linalg.norm(vecs, axis=1)
        u = vecs / norms[:, None]
        s = 0.0
        for j in range(3):
            for k in range(j + 1, 4):
                cos = float(np.dot(u[j], u[k]))
                s += (cos + 1.0 / 3.0) ** 2
        qs[i] = 1.0 - (3.0 / 8.0) * s
    return qs


def rdf_OO(positions, sample_indices, tree, box_dim, edges):
    """Partial RDF: O-O around the sample_indices subset."""
    counts = np.zeros(len(edges) - 1)
    for i in sample_indices:
        ref = positions[i]
        d, _ = tree.query(ref, k=200)  # sufficient for 8 Å cutoff
        d = d[1:]  # skip self
        d = d[d < edges[-1]]
        h, _ = np.histogram(d, bins=edges)
        counts += h
    return counts


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
    seg_remap = {s: i for i, s in enumerate(uniq_segs)}
    seg_idx_compact = np.array([seg_remap[s] for s in seg_idx])
    seg_to_atoms = {i: np.where(seg_idx_compact == i)[0]
                    for i in range(n_pep)}

    start = u.trajectory.n_frames // 2
    frames = np.linspace(start, u.trajectory.n_frames - 1, n_frames,
                         dtype=int)

    qs_by_class = {"inner": [], "interface": [], "bulk": []}
    g_edges = np.arange(0.0, G_R_MAX + DR_G, DR_G)
    g_counts = {"inner": np.zeros(len(g_edges) - 1),
                "interface": np.zeros(len(g_edges) - 1),
                "bulk": np.zeros(len(g_edges) - 1)}
    g_nref = {"inner": 0, "interface": 0, "bulk": 0}
    bulk_density = []

    for frame in frames:
        u.trajectory[int(frame)]
        pep_pos = pep_heavy.positions
        water_pos = water_o.positions
        box = u.dimensions.copy() if u.dimensions is not None else None
        # cluster COM
        coms = np.zeros((n_pep, 3))
        for s, ais in seg_to_atoms.items():
            coms[s] = pep_pos[ais].mean(axis=0)
        labels = DBSCAN(eps=DBSCAN_EPS,
                        min_samples=DBSCAN_MINSAMPLES).fit_predict(coms)
        c = Counter(labels[labels >= 0])
        if not c:
            continue
        largest = c.most_common(1)[0][0]
        cluster_com = coms[labels == largest].mean(axis=0)
        # classify waters
        r_water = np.linalg.norm(water_pos - cluster_com, axis=1)
        idx_inner = np.where(r_water < R_INNER)[0]
        idx_inter = np.where((r_water >= R_INNER) & (r_water < R_OUTER))[0]
        idx_bulk  = np.where(r_water >= R_BULK_MIN)[0]
        tree = cKDTree(water_pos)
        # tetrahedral q
        for name, idx in [("inner", idx_inner),
                          ("interface", idx_inter),
                          ("bulk", idx_bulk)]:
            if len(idx) == 0:
                continue
            # subsample for speed if very many
            if len(idx) > 1500:
                rng = np.random.default_rng(frame)
                idx = rng.choice(idx, 1500, replace=False)
            q = tetrahedral_q(water_pos, idx, tree, box_dim=box)
            qs_by_class[name].extend(q[~np.isnan(q)].tolist())
            # also: g_OO for a smaller sub-sample to limit O(N^2) cost
            n_for_g = min(200, len(idx))
            sub = idx[:n_for_g]
            g_counts[name] += rdf_OO(water_pos, sub, tree, box, g_edges)
            g_nref[name] += n_for_g
        # bulk density estimate (waters / Å^3) from outer region
        if box is not None:
            V_box = float(box[0] * box[1] * box[2])
            bulk_density.append(water_o.n_atoms / V_box)

    bulk_rho = float(np.mean(bulk_density)) if bulk_density else np.nan
    # convert RDF counts to g(r): counts in shell normalised by (4πr²Δr * ρ * N_ref)
    g_centers = 0.5 * (g_edges[:-1] + g_edges[1:])
    shells = 4 * np.pi * g_centers ** 2 * DR_G
    g_results = {}
    for name in g_counts:
        if g_nref[name] > 0 and bulk_rho > 0:
            g = g_counts[name] / (shells * bulk_rho * g_nref[name])
            g_results[name] = g
    return {
        "label": label,
        "qs": qs_by_class,
        "g_centers": g_centers,
        "g_results": g_results,
        "bulk_rho": bulk_rho,
    }


def run_all():
    os.makedirs(OUT, exist_ok=True)
    summary_rows = []
    q_rows = []
    g_rows = []
    for label, base in SYSTEMS.items():
        res = analyze_system(label, base)
        if res is None:
            continue
        for cls, vals in res["qs"].items():
            if vals:
                summary_rows.append({
                    "system": label, "region": cls,
                    "q_mean": float(np.mean(vals)),
                    "q_std":  float(np.std(vals)),
                    "n": len(vals),
                })
                for q in vals:
                    q_rows.append({"system": label, "region": cls, "q": q})
        for cls, g in res["g_results"].items():
            for r, gr in zip(res["g_centers"], g):
                g_rows.append({"system": label, "region": cls,
                               "r": float(r), "g": float(gr)})
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT, "25_water_q_summary.csv"), index=False)
    pd.DataFrame(q_rows).to_csv(
        os.path.join(OUT, "25_water_q_raw.csv"), index=False)
    pd.DataFrame(g_rows).to_csv(
        os.path.join(OUT, "25_water_gOO.csv"), index=False)
    print("Saved water structure tables to", OUT)


if __name__ == "__main__":
    run_all()
