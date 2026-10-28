"""Single source of truth for the fourteen systems.

Every trajectory listed here has a TRUE frame spacing of 40 ps.

Do not read frame spacing off the DCD headers in this project: they are not
reliable. 3_res/<sys>/trajectory.dcd reports 1 ps and every water-stripped `-nw`
file reports 1 ps, all incorrectly. trajectory.dcd is in fact byte-identical to
traj.dcd (50000 frames) followed by traj_2.dcd (75000 frames), i.e. the full 5 us
production run at 40 ps; that was confirmed by direct coordinate comparison and by
RMS water displacement per frame (9.86 A, against 9.99 A for the 9-residue runs
whose 40 ps spacing is independently known).

traj_2.dcd rather than trajectory.dcd is used for the tripeptides only so that the
window matches the one the protein contact analysis used (traj_2-nw.dcd). Either
choice is valid; both are 40 ps.
"""
import os

ROOT = "/Volumes/potoyan-lab/will/gxg-project/gxg_comps"

# label, directory, trajectory, class
SYSTEMS = [
    # tripeptides: traj_2.dcd is the later 3 us segment (also the one used for the
    # protein contact analysis, via traj_2-nw.dcd), so we stay consistent with that.
    ("3E",  f"{ROOT}/3_res/equal",          "traj_2.dcd", "3-residue"),
    ("3C",  f"{ROOT}/3_res/charged",        "traj_2.dcd", "3-residue"),
    ("3A",  f"{ROOT}/3_res/aromatic",       "traj_2.dcd", "3-residue"),
    ("3P",  f"{ROOT}/3_res/polar",          "traj_2.dcd", "3-residue"),
    ("3N",  f"{ROOT}/3_res/nonpolar",       "traj_2.dcd", "3-residue"),
    ("9E",  f"{ROOT}/9_res/equal",          "traj.dcd",   "9-residue"),
    ("9C",  f"{ROOT}/9_res/charged",        "traj.dcd",   "9-residue"),
    ("9A",  f"{ROOT}/9_res/aromatic",       "traj.dcd",   "9-residue"),
    ("9P",  f"{ROOT}/9_res/polar",          "traj.dcd",   "9-residue"),
    ("9N",  f"{ROOT}/9_res/nonpolar",       "traj.dcd",   "9-residue"),
    ("X2",  f"{ROOT}/9_res/equal/2_close",  "traj.dcd",   "patterned"),
    ("X2S", f"{ROOT}/9_res/equal/2_spread", "traj.dcd",   "patterned"),
    ("X3",  f"{ROOT}/9_res/equal/3_close",  "traj.dcd",   "patterned"),
    ("X3S", f"{ROOT}/9_res/equal/3_spread", "traj.dcd",   "patterned"),
]

DT_PS = 40.0          # true frame spacing of every trajectory above
NLAGS = 800           # 800 * 40 ps = 32 ns survival window
ORIGIN_FRACTIONS = (0.60, 0.78, 0.95)   # three time origins per system

CLASS_COLOR = {"3-residue": "#3b76af", "9-residue": "#3f9b52", "patterned": "#d1352b"}

# Published values from Fig. 5A/5B of the JPCB submission, for the before/after
# comparison only. Units as plotted (the axis carried no unit label).
PUBLISHED_INNER = {"3E": 23.1, "3C": 24.2, "3A": 15.3, "3P": 15.1, "3N": 6.0,
                   "9E": 21.4, "9C": 19.5, "9A": 15.1, "9P": 28.4, "9N": 9.9,
                   "X2": 9.3, "X2S": 5.0, "X3": 21.1, "X3S": 16.0}
PUBLISHED_OUTER = {"3E": 0.52, "3C": 0.83, "3A": 0.69, "3P": 0.65, "3N": 0.30,
                   "9E": 0.51, "9C": 0.29, "9A": 0.33, "9P": 0.29, "9N": 0.31,
                   "X2": 0.76, "X2S": 0.24, "X3": 0.93, "X3S": 0.70}


def origins(n_frames, nlags=NLAGS):
    """Three equilibrated time origins that leave room for a full lag window."""
    out = []
    for f in ORIGIN_FRACTIONS:
        o = int(f * n_frames)
        o = min(o, n_frames - nlags - 1)
        out.append(max(o, 0))
    return sorted(set(out))


def check():
    missing = []
    for lbl, d, traj, _ in SYSTEMS:
        for f in ("output.top", "output.gro", traj):
            if not os.path.exists(os.path.join(d, f)):
                missing.append(f"{lbl}: {os.path.join(d, f)}")
    return missing


if __name__ == "__main__":
    m = check()
    print("all inputs present" if not m else "MISSING:\n  " + "\n  ".join(m))
    print(f"{len(SYSTEMS)} systems, dt = {DT_PS} ps, window = {NLAGS*DT_PS/1000:g} ns")
