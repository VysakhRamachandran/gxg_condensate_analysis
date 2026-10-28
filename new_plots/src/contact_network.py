"""Peptide-peptide contact network analysis.

For each system, load the (peptide-only) trajectory, subsample to ~30 frames,
build per-frame contact graphs at a heavy-atom cutoff, and compute:

    mean_degree         — avg # of partner peptides per peptide
    sticker_mean_degree — same, restricted to atoms from non-G residues
    lcc_fraction        — size of largest connected component / N_peptides
    clustering          — networkx average clustering coefficient
    percolation_pc      — edge-removal fraction at which LCC drops below 0.5*N
    degree_dist         — pooled per-peptide degree histogram

Outputs per-frame averages and a CSV of summary metrics.
"""

import os
import warnings
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib import distances
import networkx as nx

warnings.filterwarnings("ignore")

GXG_ROOT = "/Volumes/potoyan-lab/will/gxg-project/gxg_comps"

# system -> (pdb, traj) pair
def system_paths():
    paths = {}
    # 3-residue compositions
    for comp, lbl in [("equal", "3E"), ("charged", "3C"),
                      ("aromatic", "3A"), ("polar", "3P"),
                      ("nonpolar", "3N")]:
        base = f"{GXG_ROOT}/3_res/{comp}"
        paths[lbl] = (f"{base}/output-nw.pdb", f"{base}/traj-nw.dcd")
    # 9-residue compositions
    for comp, lbl in [("equal", "9E"), ("charged", "9C"),
                      ("aromatic", "9A"), ("polar", "9P"),
                      ("nonpolar", "9N")]:
        base = f"{GXG_ROOT}/9_res/{comp}"
        paths[lbl] = (f"{base}/output-nw.pdb", f"{base}/traj-nw.dcd")
    # patterned (under 9_res/equal/{2_close,...})
    for key, lbl in [("2_close", "X2"), ("2_spread", "X2S"),
                     ("3_close", "X3"), ("3_spread", "X3S")]:
        base = f"{GXG_ROOT}/9_res/equal/{key}"
        paths[lbl] = (f"{base}/output-nw.pdb", f"{base}/traj-nw.dcd")
    return paths


CUTOFF = 4.5  # Å, heavy-atom contact cutoff
N_FRAMES = 30  # subsample target
MIN_EDGE_WEIGHT = 3  # treat an edge as "real" only if ≥ this many atom pairs


def build_graph_for_frame(u, heavy, seg_idx_for_atom, sticker_mask):
    """Return (G_all, G_sticker) for current frame."""
    pairs, _ = distances.capped_distance(heavy.positions, heavy.positions,
                                         CUTOFF, box=u.dimensions)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    seg_pairs = np.column_stack([seg_idx_for_atom[pairs[:, 0]],
                                  seg_idx_for_atom[pairs[:, 1]]])
    # remove intra-segment
    inter = seg_pairs[:, 0] != seg_pairs[:, 1]
    inter_pairs = seg_pairs[inter]
    atom_pairs_inter = pairs[inter]
    # build weighted graph (weight = # atom pairs)
    G_all = _graph_from_pairs(inter_pairs)
    # sticker subgraph: both atoms must be in sticker_mask
    s_mask = sticker_mask[atom_pairs_inter[:, 0]] & sticker_mask[atom_pairs_inter[:, 1]]
    stick_pairs = inter_pairs[s_mask]
    G_stk = _graph_from_pairs(stick_pairs)
    return G_all, G_stk


def _graph_from_pairs(seg_pairs):
    G = nx.Graph()
    if len(seg_pairs) == 0:
        return G
    # collapse to (i,j,weight)
    df = pd.DataFrame(seg_pairs, columns=["i", "j"])
    df["pair"] = df["i"].astype(str) + "_" + df["j"].astype(str)
    w = df.groupby(["i", "j"]).size().reset_index(name="w")
    for _, r in w.iterrows():
        if r["w"] >= MIN_EDGE_WEIGHT:
            G.add_edge(int(r["i"]), int(r["j"]), weight=int(r["w"]))
    return G


def percolation_pc(G, n_total, n_realizations=20, rng=None):
    """Mean fraction of edges that must be removed before LCC < 0.5 * n_total."""
    if rng is None:
        rng = np.random.default_rng(42)
    if G.number_of_edges() == 0:
        return 0.0
    edges = list(G.edges())
    n_edges = len(edges)
    pc_values = []
    for _ in range(n_realizations):
        order = rng.permutation(n_edges)
        Gw = G.copy()
        threshold = 0.5 * n_total
        pc = 1.0  # if removing all doesn't break it (only possible if n_total tiny)
        for idx, ei in enumerate(order):
            u, v = edges[ei]
            Gw.remove_edge(u, v)
            lcc = max((len(c) for c in nx.connected_components(Gw)),
                      default=0)
            if lcc < threshold:
                pc = (idx + 1) / n_edges
                break
        pc_values.append(pc)
    return float(np.mean(pc_values))


def analyze_system(label, pdb, traj, n_frames=N_FRAMES):
    print(f"  {label}: loading…", flush=True)
    u = mda.Universe(pdb, traj)
    heavy = u.select_atoms("not name H* and not resname HOH SOL TIP3 WAT TIP")
    seg_idx_for_atom = heavy.segindices.copy()
    n_pep = len(np.unique(seg_idx_for_atom))
    # sticker mask: True if the atom's residue is non-glycine
    sticker_mask = np.array([r != "GLY" for r in heavy.resnames])
    n_total_frames = u.trajectory.n_frames
    # use the second half of the trajectory (assume first half is approach to LLPS)
    start = n_total_frames // 2
    stride_frames = np.linspace(start, n_total_frames - 1, n_frames,
                                 dtype=int)
    degrees_pool, stick_degrees_pool = [], []
    metrics = []
    rng = np.random.default_rng(0)
    for fi, frame in enumerate(stride_frames):
        u.trajectory[int(frame)]
        G_all, G_stk = build_graph_for_frame(u, heavy, seg_idx_for_atom,
                                              sticker_mask)
        # ensure all nodes present (peptides with no contacts have degree 0)
        for s in range(n_pep):
            G_all.add_node(int(s))
            G_stk.add_node(int(s))
        deg_all = np.array([d for _, d in G_all.degree()])
        deg_stk = np.array([d for _, d in G_stk.degree()])
        degrees_pool.extend(deg_all.tolist())
        stick_degrees_pool.extend(deg_stk.tolist())
        lcc_size = max((len(c) for c in nx.connected_components(G_all)),
                       default=0)
        clust = nx.average_clustering(G_all)
        pc = percolation_pc(G_all, n_pep, n_realizations=5, rng=rng)
        metrics.append({
            "frame": int(frame),
            "mean_degree": float(deg_all.mean()),
            "sticker_mean_degree": float(deg_stk.mean()),
            "max_degree": int(deg_all.max() if deg_all.size else 0),
            "lcc_fraction": lcc_size / n_pep,
            "clustering": float(clust),
            "percolation_pc": pc,
            "n_edges": G_all.number_of_edges(),
            "n_edges_sticker": G_stk.number_of_edges(),
        })
    mdf = pd.DataFrame(metrics)
    summary = mdf.mean(numeric_only=True).to_dict()
    summary["system"] = label
    summary["n_peptides"] = n_pep
    return summary, mdf, np.array(degrees_pool), np.array(stick_degrees_pool)


def run_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    paths = system_paths()
    summaries = []
    degree_dists = {}
    sticker_dists = {}
    for lbl, (pdb, traj) in paths.items():
        s, mdf, deg_pool, stk_pool = analyze_system(lbl, pdb, traj)
        summaries.append(s)
        degree_dists[lbl] = deg_pool
        sticker_dists[lbl] = stk_pool
        mdf.to_csv(os.path.join(outdir, f"net_metrics_{lbl}.csv"),
                   index=False)
    df = pd.DataFrame(summaries).set_index("system")
    df.to_csv(os.path.join(outdir, "10_network_metrics.csv"))
    # save pooled degree dists for plotting
    np.savez(os.path.join(outdir, "10_degree_distributions.npz"),
             **{f"deg_{k}": v for k, v in degree_dists.items()},
             **{f"stk_{k}": v for k, v in sticker_dists.items()})
    print("Saved summary to", outdir)
    return df


if __name__ == "__main__":
    OUT = "/Volumes/potoyan-lab/will/gxg-project/new_plots/data"
    os.makedirs(OUT, exist_ok=True)
    df = run_all(OUT)
    print(df.round(3))
