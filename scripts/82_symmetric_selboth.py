"""Round 2, task 4 (R3-M3d): symmetrized selective-both.

The paper's leaderboard analysis (42/43) compared cross r(p, z_T | g_T, g_S) with within r(p, z_S | g_S), so the
change mixed the change of target with the addition of a covariate. Symmetric version: within r(p, z_S | g_S, g_T)
on the same matched cells. Note that the retraining (59: sel_BOTH_s) and IMPROVE (68: paired_scores_matrix) analyses
already used the symmetric within value; their asymmetric versions are given for comparison only.
For every paired target (80_paired_engine.py), pooled over the benchmark's model group (two-way bootstrap, B = 1,000):
  mean within (asym), mean within (sym), mean cross; change asym = cross - within(asym); change sym = cross -
  within(sym); the effect of symmetrizing = change sym - change asym = within(asym) - within(sym); retained fractions.
Whether the asymmetry matters: whether the change's 95% (and 90%) interval excluding 0, or its sign, differs.
Outputs (results/revision_round2/symmetric_selboth/): selboth_by_target.csv, summary.json.
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
OUT = lib.out_dir('symmetric_selboth')


def main():
    ix = pd.read_csv(PAIRED / 'pairs.csv')
    rows = []
    for _, r in ix.iterrows():
        f = PAIRED / 'arrays' / f'{r.benchmark}__{r.k}.npz'
        if not f.exists() or r.drugs_used == 0 or r.source == r.target:
            continue
        models, meta, S = lib.load_paired(f)
        groups = {}
        if isinstance(r.pool_main, str) and r.pool_main:
            groups['main'] = r.pool_main.split(';')
        if r.benchmark == 'retrain' and isinstance(r.pool_alt, str) and r.pool_alt:
            groups['ridges'] = r.pool_alt.split(';')
        for gk, gm in groups.items():
            gm = [m for m in gm if m in models]
            if not gm:
                continue
            g = lambda st, kind: lib.pooled(S, models, gm, st, kind)  # noqa: E731
            aw, ac, an = g('selBOTH_asym', 'sum_w'), g('selBOTH_asym', 'sum_c'), g('selBOTH_asym', 'n')
            sw, sc, sn = g('selBOTH_sym', 'sum_w'), g('selBOTH_sym', 'sum_c'), g('selBOTH_sym', 'n')
            with np.errstate(invalid='ignore', divide='ignore'):
                st = {'within_asym': aw / an, 'within_sym': sw / sn, 'cross': sc / sn,
                      'change_asym': (ac - aw) / an, 'change_sym': (sc - sw) / sn,
                      'symmetrizing_effect': (sc - sw) / sn - (ac - aw) / an,
                      'retained_asym': ac / aw, 'retained_sym': sc / sw}
            row = {'benchmark': r.benchmark, 'pair': r.pair, 'family': r.family, 'group': gk, 'models': len(gm),
                   'model_drug_pairs': int(round(sn[0]))}
            for k, v in st.items():
                row.update(lib.summ(v, k))
                if k.startswith('change'):
                    lo90, hi90 = lib.ci(v, 90)
                    row[f'{k}_lo90'], row[f'{k}_hi90'] = lo90, hi90
            for k in ['change_asym', 'change_sym']:
                row[f'{k}_95_excludes_0'] = bool(row[f'{k}_lo'] > 0 or row[f'{k}_hi'] < 0)
                row[f'{k}_90_below_0'] = bool(row[f'{k}_hi90'] < 0)
            row['conclusion_differs_95'] = row['change_asym_95_excludes_0'] != row['change_sym_95_excludes_0']
            row['conclusion_differs_90_below0'] = row['change_asym_90_below_0'] != row['change_sym_90_below_0']
            row['sign_differs'] = bool(np.sign(row['change_asym']) != np.sign(row['change_sym']))
            rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'selboth_by_target.csv', index=False, float_format='%.5g')
    main_ = T[T.group == 'main'].copy()
    # round 3: the endpoint-matched provider-AUC pairings (80 'leaderboard_matched', k >= 100) are summarised on their
    # own, so that the round-2 leaderboard summary (44 targets) is unchanged
    new_fam = main_.family.isin(['AUC-reference (provider-AUC targets)', 'CTRP provider AUC reference'])
    main_.loc[new_fam, 'benchmark'] = 'leaderboard_matched'
    summ = {}
    for bm, g in main_.groupby('benchmark'):
        summ[bm] = {'targets': int(len(g)),
                    'symmetrizing_effect_min': float(g.symmetrizing_effect.min()),
                    'symmetrizing_effect_max': float(g.symmetrizing_effect.max()),
                    'symmetrizing_effect_median': float(g.symmetrizing_effect.median()),
                    'within_sym_minus_asym_median': float((g.within_sym - g.within_asym).median()),
                    'n_conclusion_differs_95': int(g.conclusion_differs_95.sum()),
                    'pairs_conclusion_differs_95': list(g[g.conclusion_differs_95].pair),
                    'n_conclusion_differs_90_below0': int(g.conclusion_differs_90_below0.sum()),
                    'pairs_conclusion_differs_90_below0': list(g[g.conclusion_differs_90_below0].pair),
                    'n_sign_differs': int(g.sign_differs.sum())}
    lib.write_json(OUT / 'summary.json', summ)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 300)
    cols = ['benchmark', 'pair', 'group', 'within_asym', 'within_sym', 'cross', 'change_asym', 'change_asym_lo',
            'change_asym_hi', 'change_sym', 'change_sym_lo', 'change_sym_hi', 'symmetrizing_effect',
            'conclusion_differs_95']
    print(T[cols].round(3).to_string(index=False))
    print(summ)


if __name__ == '__main__':
    main()
