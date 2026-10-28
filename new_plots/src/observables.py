"""Aggregate condensate observables per system from the existing CSVs and pickles.

Per system we report:
    water_per_peptide  (cluster_water CSV)
    waterbridges_per_res (waterbridges CSV / cluster_size)
    sasa_per_atom     (sasa_peratom_*.csv)
    tau_pw_eff        (peptide-water contact lifetime, from corr pickles)
    tau_in            (inner-core water survival lifetime via biexp fit of sp_*_new.csv)
    tau_out           (outer-layer water survival lifetime)
    tau_all           (overall water survival lifetime via biexp fit of sp_all_*_new.csv)
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/data"
CORR = "/Volumes/potoyan-lab/will/gxg-project/vysakh_analysis/corr_results"

# system -> (comp_key, chain, pattern_key)
# comp_key matches the CSV filename root used by Vysakh
# For patterned systems pattern_key is used (e.g. "2_close").
COMP_FOR = {"3E": "equal", "3C": "charged", "3A": "aromatic",
            "3P": "polar", "3N": "nonpolar",
            "9E": "equal", "9C": "charged", "9A": "aromatic",
            "9P": "polar", "9N": "nonpolar"}
CHAIN_FOR = {**{k: 3 for k in ["3E", "3C", "3A", "3P", "3N"]},
             **{k: 9 for k in ["9E", "9C", "9A", "9P", "9N",
                               "X2", "X2S", "X3", "X3S"]}}
PATTERN_KEY = {"X2": "2_close", "X2S": "2_spread",
               "X3": "3_close", "X3S": "3_spread"}

ALL_SYSTEMS = ["3E", "3C", "3A", "3P", "3N",
               "9E", "9C", "9A", "9P", "9N",
               "X2", "X2S", "X3", "X3S"]


# ---------- fit helpers ---------------------------------------------------
def biexp(t, A, t1, t2):
    return A * np.exp(-t / t1) + (1 - A) * np.exp(-t / t2)


def fit_teff_biexp(y, dt=0.005):
    t = np.arange(len(y)) * dt
    try:
        p, _ = curve_fit(biexp, t, y, p0=(0.5, 0.01, 0.5),
                         bounds=([0, 1e-5, 1e-3], [1, 5, 50]),
                         maxfev=20000)
        A, t1, t2 = p
        return A * t1 + (1 - A) * t2
    except Exception:
        return np.nan


# ---------- per-system observable readers ---------------------------------
def _cluster_water_path(sys):
    chain = CHAIN_FOR[sys]
    if sys in PATTERN_KEY:
        return os.path.join(DATA, f"{PATTERN_KEY[sys]}_cluster_water_9res.csv")
    suf = "" if chain == 3 else "_9res"
    return os.path.join(DATA, f"{COMP_FOR[sys]}_cluster_water{suf}.csv")


def _waterbridges_path(sys):
    chain = CHAIN_FOR[sys]
    if sys in PATTERN_KEY:
        return os.path.join(DATA, f"{PATTERN_KEY[sys]}_waterbridges_9res.csv")
    return os.path.join(DATA, f"{COMP_FOR[sys]}_waterbridges_{chain}res.csv")


def _sp_path(sys, kind):
    """kind in {'inner_outer', 'all'}"""
    chain = CHAIN_FOR[sys]
    pkey = PATTERN_KEY.get(sys, COMP_FOR.get(sys))
    prefix = "sp" if kind == "inner_outer" else "sp_all"
    if sys in PATTERN_KEY:
        return os.path.join(DATA, f"{prefix}_9res_{pkey}_new.csv")
    return os.path.join(DATA, f"{prefix}_{chain}res_{pkey}_new.csv")


def _sasa_csv_for_chain(chain, is_pattern):
    if is_pattern:
        return os.path.join(DATA, "sasa_peratom_9res_spacers.csv")
    return os.path.join(DATA, f"sasa_peratom_{chain}res.csv")


def _sasa_col_for_sys(sys):
    if sys in PATTERN_KEY:
        return PATTERN_KEY[sys]
    return COMP_FOR[sys]


# ---------- correlation pickle access -------------------------------------
def _corr_teff(sys):
    """Return t_eff from peptide-water contact correlation pickle."""
    if CHAIN_FOR[sys] == 3:
        path = os.path.join(CORR, "3res_corr_results.pkl")
        key = COMP_FOR[sys]
    elif sys in PATTERN_KEY:
        path = os.path.join(CORR, "9res_corr_results_spacers.pkl")
        key = PATTERN_KEY[sys]
    else:
        path = os.path.join(CORR, "9_res_corr_results.pkl")
        key = COMP_FOR[sys]
    if not os.path.exists(path):
        return np.nan
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return float(d[key]["t_eff"]) if key in d else np.nan


# ---------- main builder --------------------------------------------------
def all_systems_observables():
    rows = []
    for sys in ALL_SYSTEMS:
        row = {"system": sys, "chain": CHAIN_FOR[sys],
               "is_pattern": sys in PATTERN_KEY}
        # water per peptide and # waterbridges per residue (mean over trajectory)
        cw = _cluster_water_path(sys)
        if os.path.exists(cw):
            df = pd.read_csv(cw)
            row["water_per_peptide"] = float((df["water_mass"]
                                              / df["protein_mass"]).mean())
            n_res = df["cluster_size"].mean()
        else:
            row["water_per_peptide"] = np.nan
            n_res = np.nan
        wb = _waterbridges_path(sys)
        if os.path.exists(wb) and np.isfinite(n_res):
            wbdf = pd.read_csv(wb)
            col = "WaterBridgeCount" if "WaterBridgeCount" in wbdf else wbdf.columns[0]
            row["waterbridges_per_res"] = float(wbdf[col].mean() / n_res)
        else:
            row["waterbridges_per_res"] = np.nan
        # SASA per atom
        sasa_csv = _sasa_csv_for_chain(CHAIN_FOR[sys], sys in PATTERN_KEY)
        if os.path.exists(sasa_csv):
            sdf = pd.read_csv(sasa_csv)
            col = _sasa_col_for_sys(sys)
            if col in sdf.columns:
                row["sasa_per_atom"] = float(sdf[col].mean())
            else:
                row["sasa_per_atom"] = np.nan
        # tau peptide-water (from pickle, biexp fit already done)
        row["tau_pw_eff"] = _corr_teff(sys)
        # tau water survival, inner / outer / overall (refit biexp)
        sp_io = _sp_path(sys, "inner_outer")
        if os.path.exists(sp_io):
            d = pd.read_csv(sp_io)
            row["tau_in"]  = fit_teff_biexp(d["Inner"].values)
            row["tau_out"] = fit_teff_biexp(d["Outer"].values)
        sp_all = _sp_path(sys, "all")
        if os.path.exists(sp_all):
            d = pd.read_csv(sp_all)
            col = "SP_all" if "SP_all" in d.columns else d.columns[0]
            row["tau_all"] = fit_teff_biexp(d[col].values)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("system")
    return df


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.3f}".format)
    print(all_systems_observables())
