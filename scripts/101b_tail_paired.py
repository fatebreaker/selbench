"""Round 4 (R2, editor item a6): paired interval for 'the high-dose tail's axis agreed with the target's at least as
well as the common window's did'.

Reconstructs 85_harmonization.py part (iii) for GDSC1 and GDSC2 exactly (same drugs, cells, versions and two-way
draws, seed 85.SEED + k) and reports, on the same draws, the difference between corr(CTRP v2 high-tail axis, target
window axis) and corr(CTRP v2 window axis, target window axis), with its 95% percentile interval. The point estimates
of both correlations are checked against harmonization/range_axes_correlations.csv.
Output: results/revision_round4/tail_paired/tail_minus_window.csv
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('lib77', HERE / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)
h85 = lib.load_module('h85', '85_harmonization.py')

OUT = lib.RES / 'revision_round4' / 'tail_paired'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hr = lib.load_module('hr66', '66_harmonized_range.py')
    S = hr.curves('CTRPv2')
    rS = S.groupby('pubchem_id')[['lo', 'hi']].median()
    ref = pd.read_csv(lib.OUT2 / 'harmonization' / 'range_axes_correlations.csv')
    rows = []
    for k, T in enumerate(['GDSC1', 'GDSC2', 'CCLE']):
        if T == 'CCLE':
            continue
        D = hr.curves(T)
        rT = D.groupby('pubchem_id')[['lo', 'hi']].median()
        sh = rS.index.intersection(rT.index)
        lo_c = np.maximum(rS.loc[sh, 'lo'], rT.loc[sh, 'lo'])
        hi_c = np.minimum(rS.loc[sh, 'hi'], rT.loc[sh, 'hi'])
        keep = sh[(hi_c - lo_c) >= 1.0]
        lo_c, hi_c = lo_c[keep], hi_c[keep]
        drugs = keep[(rS.loc[keep, 'hi'] - hi_c) >= h85.TAIL_MIN]
        vers = {'CTRPv2 window': h85.summaries(S, drugs, lo_c, hi_c),
                'CTRPv2 high tail': h85.summaries(S, drugs, hi_c, rS['hi']),
                'CTRPv2 full range': h85.own_auc(S, drugs),
                f'{T} window': h85.summaries(D, drugs, lo_c, hi_c),
                f'{T} full range': h85.own_auc(D, drugs)}
        dec = {v: lib.selbench.decompose(x) for v, x in vers.items()}
        cells = sorted(set.intersection(*[set(d[0].columns) for d in dec.values()]))
        dl = sorted(set.intersection(*[set(dec[v][0].index) for v in dec]))
        W, Vd = lib.draws(len(cells), len(dl), h85.SEED + k)
        AX = {v: h85.axis_reps(dec[v][0].reindex(index=dl), cells, W * 0 + 1, Vd) for v in dec}
        c_tail = h85.wcorr(AX['CTRPv2 high tail'], AX[f'{T} window'], W)
        c_win = h85.wcorr(AX['CTRPv2 window'], AX[f'{T} window'], W)
        d = c_tail - c_win
        r_t = ref[(ref.target == T) & (ref.axis_a == 'CTRPv2 high tail') & (ref.axis_b == f'{T} window')].r.iloc[0]
        r_w = ref[(ref.target == T) & (ref.axis_a == 'CTRPv2 window') & (ref.axis_b == f'{T} window')].r.iloc[0]
        rows.append({'target': T, 'drugs': len(dl), 'cells': len(cells), 'r_tail': c_tail[0], 'r_window': c_win[0],
                     'diff': d[0], 'diff_lo': np.percentile(d[1:], 2.5), 'diff_hi': np.percentile(d[1:], 97.5),
                     'share_draws_diff_gt0': float(np.mean(d[1:] > 0)),
                     'ref_r_tail': r_t, 'ref_r_window': r_w,
                     'reproduces_85': bool(abs(c_tail[0] - r_t) < 5e-5 and abs(c_win[0] - r_w) < 5e-5)})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'tail_minus_window.csv', index=False, float_format='%.5g')
    print(R.round(4).to_string())


if __name__ == '__main__':
    main()
