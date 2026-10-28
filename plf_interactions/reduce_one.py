"""Low-memory version of reduce_system() from plf_interactions.ipynb.

Streams each <RES>_plf.csv in chunks instead of loading it whole, and writes the
same cache file the notebook reads. Run one system per process so memory is
returned to the OS when it exits:

    python plf_interactions/reduce_one.py 3_res aromatic
"""
import csv, glob, os, sys
import numpy as np
import pandas as pd

BASE = 'gxg_comps'
CACHE_DIR = 'plf_interactions/cache'
LENGTHS = {'3_res': 3, '9_res': 9}
CHUNK = 500  # frames per chunk


def reduce_file(fn):
    with open(fn, newline='') as f:
        r = csv.reader(f)
        ligs, prots, inters = next(r)[1:], next(r)[1:], next(r)[1:]
    n_cols = len(ligs)
    totals = np.zeros(n_cols, dtype=np.int64)
    n_frames = 0
    # 3 header rows + the 'Frame' row; column 0 is the frame index
    for chunk in pd.read_csv(fn, header=None, skiprows=4, usecols=range(1, n_cols + 1),
                             chunksize=CHUNK, true_values=['True'], false_values=['False'], dtype=bool):
        totals += chunk.values.sum(axis=0)
        n_frames += len(chunk)
    return ligs, prots, inters, totals, n_frames


COLUMNS = ['length', 'comp', 'lig_res', 'lig_id', 'prot_res', 'prot_id',
           'interaction', 'count', 'n_frames', 'same_chain']


def reduce_system(length, comp):
    """Each input file's result is kept in cache/partial/ so a dropped network mount only loses one file."""
    folder = f'{BASE}/{length}/{"old_equal" if (length, comp) == ("9_res", "equal") else comp}'
    part_dir = f'{CACHE_DIR}/partial/{length}_{comp}'
    os.makedirs(part_dir, exist_ok=True)
    per_chain = LENGTHS[length]
    parts = []
    for fn in sorted(glob.glob(f'{folder}/plf_files/*_plf.csv')):
        part = f'{part_dir}/{os.path.basename(fn)}'
        if not os.path.exists(part):
            ligs, prots, inters, totals, n_frames = reduce_file(fn)
            rows = []
            for lig, prot, inter, c in zip(ligs, prots, inters, totals):
                lig_i, prot_i = int(lig[3:]), int(prot[3:])
                rows.append((length, comp, lig[:3], lig_i, prot[:3], prot_i, inter, int(c), n_frames,
                             lig_i // per_chain == prot_i // per_chain))
            pd.DataFrame(rows, columns=COLUMNS).to_csv(part + '.tmp', index=False)
            os.replace(part + '.tmp', part)
        parts.append(pd.read_csv(part))
        print(length, comp, os.path.basename(fn), parts[-1].n_frames.iloc[0], flush=True)
    return pd.concat(parts, ignore_index=True)


if __name__ == '__main__':
    length, comp = sys.argv[1], sys.argv[2]
    out = f'{CACHE_DIR}/{length}_{comp}.csv'
    tmp = out + '.tmp'
    reduce_system(length, comp).to_csv(tmp, index=False)
    os.replace(tmp, out)  # only a complete file ever appears under the cache name
    print('wrote', out)
