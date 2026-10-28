#!/usr/bin/env python
"""Turn the checkpointed survival curves into a results table.

Primary number is tau_int, the window integral of S. It requires no fitting, which
matters because both reviewers flagged the multi-exponential fits as unstable and
they were right. A two-exponential fit is reported alongside for comparison only.

The `decayed` column is the one to read before quoting any number: where S has not
fallen below 1/e inside the 32 ns window, tau_int is a LOWER BOUND on residence,
not an estimate of it.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.systems import SYSTEMS, DT_PS, PUBLISHED_INNER, PUBLISHED_OUTER
from src.survival import residence

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "results", "ckpt")
DT_NS = DT_PS / 1000.0


def main():
    rows = []
    for lbl, d, traj, klass in SYSTEMS:
        for layer in ("inner", "outer"):
            fs = sorted(glob.glob(f"{CKPT}/{lbl}_*_{layer}.npy"))
            if not fs:
                continue
            per = [residence(np.load(f), DT_NS) for f in fs]
            rows.append({
                "system": lbl, "class": klass, "layer": layer,
                "n_origins": len(per),
                "tau_int_ns": np.mean([p["tau_int"] for p in per]),
                "tau_int_sd": np.std([p["tau_int"] for p in per]),
                "tau_biexp_ns": np.nanmean([p["tau_biexp"] for p in per]),
                "S_end": np.mean([p["S_end"] for p in per]),
                "decayed": all(p["decayed"] for p in per),
                "published": (PUBLISHED_INNER if layer == "inner"
                              else PUBLISHED_OUTER).get(lbl, np.nan),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "survival_summary.csv"), index=False)

    for layer in ("inner", "outer"):
        s = df[df.layer == layer]
        if s.empty:
            continue
        print(f"\n{'='*74}\n{layer.upper()} layer, common 40 ps axis, 32 ns window\n{'='*74}")
        print(f"{'sys':>5}{'class':>12}{'tau_int (ns)':>16}{'decayed':>9}"
              f"{'S(32ns)':>9}{'published':>11}")
        for _, r in s.iterrows():
            print(f"{r.system:>5}{r['class']:>12}"
                  f"{r.tau_int_ns:>10.3f} +/-{r.tau_int_sd:<5.3f}"
                  f"{str(r.decayed):>9}{r.S_end:>9.3f}{r.published:>11.2f}")
        for k in ("3-residue", "9-residue", "patterned"):
            v = s[s['class'] == k].tau_int_ns
            if len(v):
                print(f"  class mean {k:>11}: {v.mean():.3f} ns")
        a = s[s['class'] == "3-residue"].tau_int_ns.mean()
        b = s[s['class'] == "9-residue"].tau_int_ns.mean()
        if a and b:
            print(f"  --> 9-residue / 3-residue = {b/a:.1f}x  "
                  f"(paper reported this ratio as {np.mean(list(PUBLISHED_INNER.values())[5:10])/np.mean(list(PUBLISHED_INNER.values())[:5]):.2f}x for inner)"
                  if layer == "inner" else f"  --> 9-residue / 3-residue = {b/a:.1f}x")
    print("\nwrote results/survival_summary.csv")


if __name__ == "__main__":
    main()
