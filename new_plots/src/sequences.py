"""Peptide sequences and stoichiometries for the 14 GXG systems.

Stoichiometries extracted from the packmol mixture.inp files in
gxg_comps/{3_res,9_res}/{equal,charged,aromatic,polar,nonpolar}/ and
gxg_comps/9_res/equal/{2_close,2_spread,3_close,3_spread}/.

Single-letter codes are used internally. Each system is a dict
sequence -> count.
"""

from collections import OrderedDict

# canonical amino acids
AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
          "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
          "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
          "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

AA_ALL = list("AGRNDCEQHILKMFPSTWYV")

# --- 3-residue systems ------------------------------------------------------
def _gxg(x): return f"G{x}G"

SYSTEMS = OrderedDict()

# 3E: 20 GXG variants equimolar (5 each used in equal/3res mixture)
SYSTEMS["3E"] = {_gxg(x): 8 for x in AA_ALL}

# 3C: enriched charged (R, K, D, E, H) at count 28; others at 10
charged_set = {"R", "K", "D", "E", "H"}
SYSTEMS["3C"] = {_gxg(x): (28 if x in charged_set else 10) for x in AA_ALL}

# 3A: enriched aromatic (F, Y, W) at 33; others at 12
aromatic_set = {"F", "Y", "W"}
SYSTEMS["3A"] = {_gxg(x): (33 if x in aromatic_set else 12) for x in AA_ALL}

# 3P: enriched polar (C, Q, N, S, T) at 28; others at 10
polar_set = {"C", "Q", "N", "S", "T"}
SYSTEMS["3P"] = {_gxg(x): (28 if x in polar_set else 10) for x in AA_ALL}

# 3N: enriched non-polar (G, P, A, V, L, I, M) at 25; others at 9
nonpolar_set = {"G", "P", "A", "V", "L", "I", "M"}
SYSTEMS["3N"] = {_gxg(x): (25 if x in nonpolar_set else 9) for x in AA_ALL}


# --- 9-residue single-sticker systems --------------------------------------
def _g4xg4(x): return f"GGGG{x}GGGG"

# 9E: equimolar over 20 G4XG4 variants
SYSTEMS["9E"] = {_g4xg4(x): 5 for x in AA_ALL}

# 9C: R, D, E, H, K at 10; others at 3
SYSTEMS["9C"] = {_g4xg4(x): (10 if x in charged_set else 3) for x in AA_ALL}

# 9A: F, W, Y at 11; others at 3
SYSTEMS["9A"] = {_g4xg4(x): (11 if x in aromatic_set else 3) for x in AA_ALL}

# 9P: N, C, Q, S, T at 10; others at 3
SYSTEMS["9P"] = {_g4xg4(x): (10 if x in polar_set else 3) for x in AA_ALL}

# 9N: G, A, I, L, M, P, V at 8; others at 2
SYSTEMS["9N"] = {_g4xg4(x): (8 if x in nonpolar_set else 2) for x in AA_ALL}


# --- patterned multi-sticker (9-residue) -----------------------------------
# Each system: equimolar over 20 X identities at count 4
# X2  (close, 2 stickers at positions 4,6)   = GGGXGXGGG
# X2S (spread, 2 stickers at positions 2,8)  = GXGGGGGXG
# X3  (close, 3 stickers at positions 4,5,6) = GGGXXXGGG
# X3S (spread, 3 stickers at 2,5,8)          = GXGGXGGXG
def _pattern(xpositions, x):
    """positions are 1-indexed in a 9-residue chain."""
    seq = list("GGGGGGGGG")
    for p in xpositions:
        seq[p - 1] = x
    return "".join(seq)

PATTERN_POSITIONS = {
    "X2":  (4, 6),
    "X2S": (2, 8),
    "X3":  (4, 5, 6),
    "X3S": (2, 5, 8),
}
for label, positions in PATTERN_POSITIONS.items():
    SYSTEMS[label] = {_pattern(positions, x): 4 for x in AA_ALL}


# --- helpers ---------------------------------------------------------------
def chain_length(sysname):
    """Length of peptide chains in a system (all chains have the same length)."""
    return len(next(iter(SYSTEMS[sysname])))


def x_positions(sysname):
    """1-indexed positions of non-G residues (sticker positions)."""
    if sysname.startswith("3"):
        return (2,)
    if sysname in PATTERN_POSITIONS:
        return PATTERN_POSITIONS[sysname]
    # other 9-residue single-sticker
    return (5,)


def n_stickers(sysname):
    return len(x_positions(sysname))


def total_peptides(sysname):
    return sum(SYSTEMS[sysname].values())


def total_residues(sysname):
    return sum(len(seq) * n for seq, n in SYSTEMS[sysname].items())


if __name__ == "__main__":
    for s in SYSTEMS:
        print(f"{s:>4}  N={total_peptides(s):>4}  L={chain_length(s)}"
              f"  stickers={n_stickers(s)} at {x_positions(s)}"
              f"  total_res={total_residues(s)}")
