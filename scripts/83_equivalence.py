"""Round 2, task 5 (R3-M4, R1-M3e): an informative equivalence test for the selective-both change.

The 34 versions of the paper (Methods 'Intervals and tests'; results/revision_crosslab/tost_selectivity.csv and
harmonized_range/tost_selectivity.csv): CTRPv2-trained leaderboard predictions, own matched cells, all drugs, every
non-CTRPv2 target scored against the ln IC50 reference (14) or the AUC reference (4), plus the 16 harmonized versions
(shared panel, common range, ln IC50 shared panel and range-censored). Pooled over the 21 published models, two-way
bootstrap (42's draws, B = 1,000; 80_paired_engine.py), for selective-both as in the paper (within r(p, z_S | g_S))
and symmetrized (within r(p, z_S | g_S, g_T); task 4):
  retained fraction f = mean cross / mean within; one-sided non-inferiority H0: f <= 0.8, rejected when the one-sided
  95% lower bound (5th percentile) exceeds 0.8; bootstrap P = share of draws with f <= 0.8; 90% and 95% intervals.
  TOST of the difference d = cross - within at margins +-0.03 and +-0.05 (equivalent when the 90% interval lies
  inside the margins), and one-sided non-inferiority at -0.03 and -0.05.
Round 3 (editor item 2; R1-M2, R3-M3): the same tests for every leaderboard pair in pairs.csv (the 34, the
CTRPv2-internal re-processing rows, the drug-matched variants and the endpoint-matched provider-AUC pairings of 80's
'leaderboard_matched' part) -> equivalence_leaderboard_all.csv; the endpoint classification and counts are made by
99a_endpoint_matched.py.
Outputs (results/revision_round2/equivalence/): equivalence_34.csv, summary.json, equivalence_leaderboard_all.csv.
"""
import sys

sys.dont_write_bytecode = True
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

PAIRED = lib.OUT2 / 'paired'
OUT = lib.out_dir('equivalence')
F0 = 0.8
MARGINS = [0.03, 0.05]


def the_34(ix):
    lb = ix[ix.benchmark == 'leaderboard']
    own = lb[lb.family.isin(['lnIC50-reference', 'AUC-reference']) & ~lb.target.str.startswith('CTRPv2')]
    harm = lb[lb.family.isin(['shared panel, common range AUC', 'shared panel, own range AUC', 'shared panel ln IC50',
                              'shared panel + range ln IC50'])]
    sel = pd.concat([own, harm])
    assert len(sel) == 34, len(sel)
    return sel


def main():
    ix = pd.read_csv(PAIRED / 'pairs.csv')
    sel = the_34(ix)
    old = pd.concat([pd.read_csv(lib.RES / 'revision_crosslab' / 'tost_selectivity.csv'),
                     pd.read_csv(lib.RES / 'revision_crosslab' / 'harmonized_range' / 'tost_selectivity.csv')])
    old = old[(old.metric == 'selBOTH') & (old.subset == 'all') & (old.cellset == 'own')].set_index('pair')
    rows = equivalence_rows(sel, old)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'equivalence_34.csv', index=False, float_format='%.5g')
    # round 3: every leaderboard pair (non-identity)
    lb = ix[(ix.benchmark == 'leaderboard') & (ix.source != ix.target) & (ix.drugs_used > 0)]
    A = pd.DataFrame(equivalence_rows(lb, old))
    A = A.merge(lb[['pair', 'k', 'source', 'target']], on='pair', how='left')
    A.to_csv(OUT / 'equivalence_leaderboard_all.csv', index=False, float_format='%.5g')
    summarize(T)


def equivalence_rows(sel, old):
    rows = []
    for _, r in sel.iterrows():
        models, meta, S = lib.load_paired(PAIRED / 'arrays' / f'leaderboard__{r.k}.npz')
        gm = r.pool_main.split(';')
        for version, stat in [('paper (within r(p,z_S|g_S))', 'selBOTH_asym'), ('symmetric', 'selBOTH_sym')]:
            sw, sc, n = (lib.pooled(S, models, gm, stat, k) for k in ('sum_w', 'sum_c', 'n'))
            with np.errstate(invalid='ignore', divide='ignore'):
                f = sc / sw
                d = (sc - sw) / n
            fb = f[1:][np.isfinite(f[1:])]
            db = d[1:][np.isfinite(d[1:])]
            lo90, hi90 = lib.ci(d, 90)
            flo90, fhi90 = lib.ci(f, 90)
            row = {'pair': r.pair, 'family': r.family, 'selective_both': version, 'mean_within': float(sw[0] / n[0]),
                   'mean_cross': float(sc[0] / n[0]), **lib.summ(f, 'retained_fraction'),
                   'retained_fraction_lo90': flo90, 'retained_fraction_hi90': fhi90,
                   'retained_fraction_lower95_onesided': float(np.percentile(fb, 5)),
                   'p_boot_f_le_0.8': float((fb <= F0).mean()),
                   'noninferior_f_0.8': bool(np.percentile(fb, 5) > F0),
                   'within_ci95_includes_0': bool(lib.ci(sw / n)[0] <= 0 <= lib.ci(sw / n)[1]),
                   **lib.summ(d, 'difference'), 'difference_lo90': lo90, 'difference_hi90': hi90}
            for mg in MARGINS:
                row[f'tost_equivalent_{mg:.2f}'] = bool(lo90 > -mg and hi90 < mg)
                row[f'tost_p_{mg:.2f}'] = float(max((db <= -mg).mean(), (db >= mg).mean()))
                row[f'noninferior_diff_{mg:.2f}'] = bool(lo90 > -mg)
            if version.startswith('paper') and r.pair in old.index:
                o = old.loc[r.pair]
                row['check_paper_diff'] = float(o.diff_cross_minus_within)
                row['check_paper_equiv_0.05'] = bool(o['equivalent_margin_0.05'])
            rows.append(row)
    return rows


def summarize(T):
    summ = {}
    for v, g in T.groupby('selective_both'):
        summ[v] = {'versions': int(len(g)), 'within_min': float(g.mean_within.min()), 'within_max': float(g.mean_within.max()),
                   'within_median': float(g.mean_within.median()),
                   'n_noninferior_f_0.8': int(g['noninferior_f_0.8'].sum()),
                   'pairs_noninferior_f_0.8': list(g[g['noninferior_f_0.8']].pair),
                   'n_tost_0.03': int(g['tost_equivalent_0.03'].sum()), 'pairs_tost_0.03': list(g[g['tost_equivalent_0.03']].pair),
                   'n_tost_0.05': int(g['tost_equivalent_0.05'].sum()),
                   'n_noninferior_diff_0.05': int(g['noninferior_diff_0.05'].sum()),
                   'n_noninferior_diff_0.03': int(g['noninferior_diff_0.03'].sum()),
                   'retained_fraction_min': float(g.retained_fraction.min()),
                   'retained_fraction_max': float(g.retained_fraction.max()),
                   'retained_fraction_median': float(g.retained_fraction.median()),
                   'n_within_ci_includes_0': int(g.within_ci95_includes_0.sum())}
    lib.write_json(OUT / 'summary.json', summ)
    pd.set_option('display.width', 250)
    cols = ['pair', 'selective_both', 'mean_within', 'mean_cross', 'retained_fraction', 'retained_fraction_lo',
            'retained_fraction_hi', 'retained_fraction_lower95_onesided', 'noninferior_f_0.8', 'difference',
            'difference_lo90', 'difference_hi90', 'tost_equivalent_0.03', 'tost_equivalent_0.05']
    print(T[cols].round(3).to_string(index=False))
    print(summ)


if __name__ == '__main__':
    main()
