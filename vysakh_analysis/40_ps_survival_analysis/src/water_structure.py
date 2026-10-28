"""Per-molecule water structure observables: the replacement for the retracted
dielectric-constant panel.

Why these three. The dielectric analysis failed because a fluctuation formula on
an open sub-volume is not a local dielectric constant, and because the answer
scaled with the number of molecules in the region (a control on homogeneous bulk
water returned eps from 1.15 at R=8 A to 7.42 at R=26 A). Every quantity below is
defined PER MOLECULE and needs no volume, no density, and no ideal-gas reference,
so none of that failure mode can occur:

  n_HB     hydrogen bonds per water (Luzar-Chandler geometry: r_OO < 3.5 A and
           H-O...O angle < 30 deg), counting donated plus accepted
  n_coord  first-shell coordination: OW neighbours within 3.5 A
  q        tetrahedral order parameter (same definition as the manuscript)

These also fix reviewer 2's point 14 about the g(r)_OO peak heights, which suffer
the same normalisation ambiguity: in the published Fig. 5D the bulk curve tends to
1 while the inner and outer curves decay below it, precisely because a sub-region
RDF has no well-defined local density reference. n_coord carries the same
information with no normalisation at all.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from MDAnalysis.lib.distances import capped_distance, minimize_vectors

R_HB = 3.5          # Angstrom, O-O
ANG_HB = 30.0       # degrees, H-O...O
R_COORD = 3.5       # Angstrom, first-shell OW-OW
R_TET = 3.9         # Angstrom, neighbour search for q


def _pairs(pos_a, pos_b, cutoff, box, same=False):
    p = capped_distance(pos_a, pos_b, max_cutoff=cutoff, box=box,
                        return_distances=False)
    if same:
        p = p[p[:, 0] < p[:, 1]]
    return p


def _cos_ang(v1, v2):
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.einsum("ij,ij->i", v1, v2) / n


def hbonds_per_water(sub_idx, ow_pos, hw_pos, box):
    """Mean H-bonds per water for the waters in `sub_idx`.

    Partners are searched among ALL water, never only within the subset. Counting
    partners inside the subset only would make the answer depend on how the subset
    was carved out, which is the same failure that invalidated the dielectric
    analysis: a sparse or small region would report artificially few neighbours.

    sub_idx : indices into ow_pos
    ow_pos  : (n_all, 3) all water oxygens
    hw_pos  : (n_all, 2, 3) the two hydrogens of every water
    """
    sub_idx = np.asarray(sub_idx)
    if len(sub_idx) == 0:
        return float("nan")
    p = _pairs(ow_pos[sub_idx], ow_pos, R_HB, box)
    gi = sub_idx[p[:, 0]]          # global index of the subset water
    gj = p[:, 1]                   # global index of the partner
    keep = gi != gj
    gi, gj, loc = gi[keep], gj[keep], p[keep, 0]
    if len(gi) == 0:
        return 0.0
    oo = minimize_vectors(ow_pos[gj] - ow_pos[gi], box)
    hit = np.zeros(len(gi), dtype=bool)
    # subset water donates
    for h in range(2):
        oh = minimize_vectors(hw_pos[gi, h] - ow_pos[gi], box)
        hit |= _cos_ang(oh, oo) > np.cos(np.radians(ANG_HB))
    # partner donates (subset water accepts)
    for h in range(2):
        oh = minimize_vectors(hw_pos[gj, h] - ow_pos[gj], box)
        hit |= _cos_ang(oh, -oo) > np.cos(np.radians(ANG_HB))
    counts = np.bincount(loc, weights=hit.astype(float), minlength=len(sub_idx))
    return float(counts.mean())


def coordination_per_water(sub_pos, all_ow_pos, box):
    """Mean number of OW within R_COORD of each water in the subset.

    Neighbours are searched against ALL water, not just the subset, which is what
    makes this a genuine local coordination number rather than a property of how
    the subset was carved out.
    """
    if len(sub_pos) == 0:
        return 0.0
    p = _pairs(sub_pos, all_ow_pos, R_COORD, box)
    # remove self matches (distance 0)
    d = np.linalg.norm(sub_pos[p[:, 0]] - all_ow_pos[p[:, 1]], axis=1)
    p = p[d > 1e-6]
    return float(len(p) / len(sub_pos))


def tetrahedral_q(sub_pos, all_ow_pos, box):
    """Mean tetrahedral order parameter q over the subset."""
    if len(sub_pos) == 0:
        return float("nan")
    p = _pairs(sub_pos, all_ow_pos, R_TET, box)
    d = np.linalg.norm(sub_pos[p[:, 0]] - all_ow_pos[p[:, 1]], axis=1)
    keep = d > 1e-6
    p, d = p[keep], d[keep]
    qs = []
    for i in np.unique(p[:, 0]):
        m = p[:, 0] == i
        if m.sum() < 4:
            continue
        nbr = all_ow_pos[p[m, 1]][np.argsort(d[m])[:4]]
        v = nbr - sub_pos[i]
        v /= np.linalg.norm(v, axis=1)[:, None]
        s = 0.0
        for a in range(3):
            for b in range(a + 1, 4):
                s += (np.dot(v[a], v[b]) + 1.0 / 3.0) ** 2
        qs.append(1.0 - 3.0 / 8.0 * s)
    return float(np.mean(qs)) if qs else float("nan")
