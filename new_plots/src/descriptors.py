"""Sequence descriptors for peptide mixtures.

For each system (a dict of sequence -> count) we compute:
    FCR        — fraction of charged residues (R,K,D,E,H)
    NCPR_abs   — |net charge per residue|
    f_aromatic — fraction of aromatic residues (F,Y,W)
    f_sticker  — fraction of non-G residues (Pappu-style sticker fraction)
    hydropathy — mean Kyte–Doolittle hydropathy
    sc_volume  — mean side-chain volume (Å^3, Zamyatnin 1972)
    kappa_xg   — Das–Pappu κ patterning, treating X (non-G) vs G as the two classes
    sticker_spread — std-dev of non-G positions / chain length (positional dispersion)
    n_stickers — mean number of stickers (non-G) per chain

All system-level descriptors are mole-fraction-weighted averages over
the per-chain values.
"""

import numpy as np
from .sequences import SYSTEMS, total_peptides

# Kyte–Doolittle hydropathy (1982)
KD = {"A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
      "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
      "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
      "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2}

# side-chain volume (Å^3), Zamyatnin 1972 / Pommié 2004
SC_VOL = {"A":  88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
          "Q": 143.8, "E": 138.4, "G":  60.1, "H": 153.2, "I": 166.7,
          "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
          "S":  89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}

POS = {"R", "K"}            # treat H as neutral at pH 7 (consistent with sim)
NEG = {"D", "E"}
CHARGED = POS | NEG
AROMATIC = {"F", "Y", "W"}


# ---------- per-chain primitives ------------------------------------------
def chain_f_pos(seq):  return sum(r in POS for r in seq) / len(seq)
def chain_f_neg(seq):  return sum(r in NEG for r in seq) / len(seq)
def chain_FCR(seq):    return chain_f_pos(seq) + chain_f_neg(seq)
def chain_NCPR(seq):   return chain_f_pos(seq) - chain_f_neg(seq)
def chain_f_aro(seq):  return sum(r in AROMATIC for r in seq) / len(seq)
def chain_f_sticker(seq): return sum(r != "G" for r in seq) / len(seq)
def chain_KD(seq):     return np.mean([KD[r] for r in seq])
def chain_volume(seq): return np.mean([SC_VOL[r] for r in seq])


def chain_sticker_spread(seq):
    """Std-dev of sticker (non-G) positions, normalized by chain length.
    Returns 0 if 0 or 1 stickers."""
    positions = [i for i, r in enumerate(seq) if r != "G"]
    if len(positions) < 2:
        return 0.0
    return float(np.std(positions)) / len(seq)


def chain_kappa_xg(seq, blob=4):
    """Das–Pappu κ patterning with two classes: X (non-G) and G.

    σ(blob) = (f_X - f_G)^2 / (f_X + f_G)  per blob (always = 1 when only one class)
    κ = <σ_blob> / σ_max   where σ_max = (f_X - f_G)^2 / (f_X + f_G) of the whole.
    Returns 0 if the whole sequence has only one class (cannot pattern).
    """
    L = len(seq)
    is_x = np.array([r != "G" for r in seq], dtype=float)
    fx = is_x.mean()
    fg = 1 - fx
    if fx == 0 or fg == 0:
        return 0.0
    sigma_total = (fx - fg) ** 2  # /(fx+fg)=1
    if L < blob:
        return 0.0
    sigmas = []
    for i in range(L - blob + 1):
        win = is_x[i:i + blob]
        a = win.mean()
        b = 1 - a
        if (a + b) == 0:
            continue
        sigmas.append((a - b) ** 2 / (a + b))
    if not sigmas:
        return 0.0
    return float(np.mean(sigmas) / sigma_total)


# ---------- system-level (mole-fraction weighted) -------------------------
PER_CHAIN_FUNCS = {
    "FCR":            chain_FCR,
    "NCPR_abs":       lambda s: abs(chain_NCPR(s)),
    "f_aromatic":     chain_f_aro,
    "f_sticker":      chain_f_sticker,
    "hydropathy":     chain_KD,
    "sc_volume":      chain_volume,
    "kappa_xg":       chain_kappa_xg,
    "sticker_spread": chain_sticker_spread,
}


def system_descriptors(sysname):
    """Return dict of mole-fraction-weighted system descriptors."""
    mix = SYSTEMS[sysname]
    N = total_peptides(sysname)
    out = {}
    for name, f in PER_CHAIN_FUNCS.items():
        out[name] = sum(count * f(seq) for seq, count in mix.items()) / N
    # also: mean # of stickers per chain (== f_sticker * chain length)
    out["n_stickers"] = sum(count * sum(r != "G" for r in seq)
                            for seq, count in mix.items()) / N
    return out


def all_systems_descriptor_table():
    import pandas as pd
    rows = []
    for s in SYSTEMS:
        d = system_descriptors(s)
        d["system"] = s
        rows.append(d)
    df = pd.DataFrame(rows).set_index("system")
    cols = ["FCR", "NCPR_abs", "f_aromatic", "f_sticker", "n_stickers",
            "hydropathy", "sc_volume", "kappa_xg", "sticker_spread"]
    return df[cols]


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.float_format", "{:.3f}".format)
    print(all_systems_descriptor_table())
