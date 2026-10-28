"""OPTION 2, step 2: water reorientation time tau_2, resolved by region.

This is the observable that should carry the "core water is dynamically
restricted" argument in place of the retracted dielectric constant. It is:

  * defined per molecule, so it has none of the sub-volume normalisation problems
    that sank the dielectric analysis;
  * the quantity actually measured by 2D IR and by NMR relaxation, so it can be
    compared against sheehan2026chain, lorenz2023ultrafast and lorenz2025beyond,
    all of which are already cited in the manuscript. That answers reviewer 1's
    point 3 with no new references;
  * unresolvable at 40 ps output, which is why it needs the reruns.

C_2(t) = <P_2( u(0) . u(t) )>, P_2(x) = (3x^2 - 1)/2, u = unit OH bond vector.
tau_2 is the integral of C_2, reported with the fraction of the decay that is
actually resolved inside the correlation window.

Report the inner/bulk RATIO as the primary result. TIP3P reorients roughly 2-3x
faster than real water, so absolute values should not be compared to experiment
directly, but the ratio between regions is far more transferable.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def oh_vectors(ow_pos, hw_pos):
    """Unit OH bond vectors: (n_water, 2, 3) -> (2*n_water, 3)."""
    v = hw_pos - ow_pos[:, None, :]
    v = v.reshape(-1, 3)
    return v / np.linalg.norm(v, axis=1)[:, None]


def c2_from_vectors(vec_series):
    """Second-order Legendre autocorrelation.

    vec_series : (n_frames, n_vectors, 3) unit vectors for a FIXED set of OH bonds
    returns C_2 of length n_frames, averaged over vectors and time origins.
    """
    nf, nv, _ = vec_series.shape
    c2 = np.zeros(nf)
    norm = np.zeros(nf)
    # FFT would be faster, but P_2 is not a simple dot product so do it directly
    # over a bounded set of origins.
    n_orig = max(1, nf // 4)
    stride = max(1, n_orig // 50)
    for t0 in range(0, n_orig, stride):
        dots = np.einsum("ij,tij->ti", vec_series[t0], vec_series[t0:])
        p2 = 1.5 * dots ** 2 - 0.5
        m = p2.mean(axis=1)
        c2[: len(m)] += m
        norm[: len(m)] += 1
    good = norm > 0
    c2[good] /= norm[good]
    return c2


def tau2(c2, dt_ps):
    """Integral of C_2, plus how much of the decay the window actually resolved."""
    t = np.arange(len(c2)) * dt_ps
    return {"tau2_ps": float(np.trapezoid(c2, t)),
            "C2_end": float(c2[-1]),
            "resolved": bool(c2[-1] < 1.0 / np.e),
            "window_ps": float(t[-1])}
