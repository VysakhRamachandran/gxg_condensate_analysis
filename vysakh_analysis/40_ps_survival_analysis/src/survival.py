"""Continuous water survival probability, and window-safe residence estimates.

S(tau) = fraction of the waters present in a layer's hydration shell at t0 that
have been present at EVERY sampled frame from t0 to t0+tau.

Two departures from the original implementation, both deliberate:

  * all layers are done in a single pass over the frames, which halves the I/O;
  * the per-frame `around 5.0` string selection is replaced by capped_distance,
    which is equivalent and about 100x faster.

Residence is reported primarily as the WINDOW INTEGRAL of S, which needs no
fitting. Both reviewers criticised the multi-exponential fits as unstable, and
they are right: the individual amplitudes and time constants are ill-conditioned,
whereas the integral is not. A two-exponential fit is also returned for
comparison, clearly labelled, along with a flag for whether S actually decayed
inside the window (if it did not, the integral is a LOWER BOUND, not an estimate).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from MDAnalysis.lib.distances import capped_distance
from scipy.optimize import curve_fit

from .layers import safe_seek

DIST_WATER = 5.0      # Angstrom, matches SP() in water_dynamics.ipynb cell 2


def survival_multi(u, layers, start, nlags, stride=1):
    """Continuous survival for several layers in one pass.

    layers : dict name -> resindices
    returns (dict name -> S array of length nlags, dict name -> n0)
    """
    ow = u.select_atoms("name OW")
    atoms = {k: u.atoms[np.isin(u.atoms.resindices, v)] for k, v in layers.items()}

    def shell(ts, ag):
        if len(ag) == 0:
            return set()
        pairs = capped_distance(ow.positions, ag.positions,
                               max_cutoff=DIST_WATER, box=ts.dimensions,
                               return_distances=False)
        return set(np.unique(pairs[:, 0]))

    ts = safe_seek(u, start)
    alive = {k: shell(ts, ag) for k, ag in atoms.items()}
    n0 = {k: len(v) for k, v in alive.items()}
    S = {k: np.full(nlags, np.nan) for k in layers}
    for k in layers:
        if n0[k]:
            S[k][0] = 1.0
    for j in range(1, nlags):
        ts = safe_seek(u, start + j * stride)
        for k, ag in atoms.items():
            if not n0[k]:
                continue
            alive[k] &= shell(ts, ag)
            S[k][j] = len(alive[k]) / n0[k]
    return S, n0


def _biexp(t, A, t1, t2):
    return A * np.exp(-t / t1) + (1 - A) * np.exp(-t / t2)


def residence(S, dt_ns):
    """Window integral, 1/e crossing, and a labelled two-exponential fit.

    Returns dict with:
      tau_int    : trapezoidal integral of S over the sampled window (ns)
      decayed    : True if S fell below 1/e inside the window
      tau_1e     : first lag at which S < 1/e, else nan
      S_end      : S at the last lag (how much is unresolved)
      tau_biexp  : sum A_k tau_k from a 2-exponential fit, or nan if it failed
    """
    S = np.asarray(S, dtype=float)
    t = np.arange(len(S)) * dt_ns
    out = {"tau_int": float(np.trapezoid(S, t)), "S_end": float(S[-1])}
    below = np.nonzero(S < 1.0 / np.e)[0]
    out["decayed"] = bool(len(below))
    out["tau_1e"] = float(t[below[0]]) if len(below) else float("nan")
    try:
        p, _ = curve_fit(_biexp, t, S, p0=[0.5, 0.1 * t[-1], t[-1]],
                         bounds=([0, 1e-6, 1e-6], [1, np.inf, np.inf]),
                         maxfev=20000)
        A, t1, t2 = p
        out["tau_biexp"] = float(A * t1 + (1 - A) * t2)
    except Exception:
        out["tau_biexp"] = float("nan")
    return out
