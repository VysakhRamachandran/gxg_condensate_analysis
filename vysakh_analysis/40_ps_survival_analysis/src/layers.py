"""Condensate identification and inner/outer partition.

ONE canonical setting is used everywhere in this directory. The original
notebooks contained four different GITIM settings for the same core/shell split
(alpha 4.5/max_layers 3, alpha 4.0/max_layers 3, alpha 4.2/max_layers 2, and
alpha 3.9/max_layers 2 plus a setdiff), which meant different figures rested on
different partitions. The setting below is the one water_dynamics.ipynb cell 2
used for the published survival data, so results here remain comparable to it.

Change GITIM_ALPHA / GITIM_MAXLAYERS in one place if you want to test sensitivity.
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import self_distance_array
from sklearn.cluster import DBSCAN
from collections import Counter
import pytim

DBSCAN_EPS = 9.0        # Angstrom, on CA; matches the manuscript
DBSCAN_MIN_SAMPLES = 1  # matches the notebooks (the manuscript says 2; see HANDOFF)
GITIM_ALPHA = 4.0
GITIM_MAXLAYERS = 3


def largest_cluster(u, ts):
    """DBSCAN on CA positions; return resindices of the largest cluster."""
    ca = u.select_atoms("name CA")
    n = len(ca)
    d = self_distance_array(ca.positions, box=ts.dimensions)
    sq = np.zeros((n, n))
    triu = np.triu_indices(n, k=1)
    sq[triu] = d
    sq.T[triu] = d
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                    metric="precomputed").fit_predict(sq)
    biggest = Counter(labels).most_common(1)[0][0]
    return np.asarray(ca.residues.resindices)[labels == biggest]


def gitim_layers(u, ts, cluster_resindices, tmpdir="."):
    """GITIM partition of the clustered peptides.

    Returns (inner, mid, outer) arrays of parent-universe resindices, made
    mutually disjoint with outer taking precedence.

    Mapping back to the parent universe uses ATOM ORDER, not the PDB `resid`
    field. The original code round-tripped through spatial.pdb and then matched
    on `resid`, which silently breaks past 9999 residues because the PDB resid
    field is only four characters wide.
    """
    sel = u.atoms[np.isin(u.atoms.resindices, cluster_resindices)]
    orig_resindex_of_atom = np.asarray(sel.atoms.resindices)
    tmp = os.path.join(tmpdir, "_spatial.pdb")
    with mda.Writer(tmp) as w:
        w.write(sel)
    sub = mda.Universe(tmp)
    sub.dimensions = ts.dimensions
    inter = pytim.GITIM(sub, group=sub.select_atoms("name CA"),
                        alpha=GITIM_ALPHA, max_layers=GITIM_MAXLAYERS,
                        molecular=False, centered="middle")
    out = []
    for k in range(GITIM_MAXLAYERS):
        idx = np.unique(inter.layers[k].indices)
        out.append(np.unique(orig_resindex_of_atom[idx]))
    outer, mid, inner = out[0], out[1], out[2]
    mid = np.setdiff1d(mid, outer)
    inner = np.setdiff1d(np.setdiff1d(inner, outer), mid)
    return inner, mid, outer


def safe_seek(u, frame, tries=6):
    """Seek to a frame, retrying through transient network-share read failures."""
    for attempt in range(tries):
        try:
            return u.trajectory[frame]
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
            try:
                u.trajectory._reopen()
            except Exception:
                pass
    raise RuntimeError("unreachable")
