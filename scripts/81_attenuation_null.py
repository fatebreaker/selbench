"""Round 2, task 3 (R3-M3a, editor item 2): test cross-dataset attributions against proportional attenuation.

If a target is simply noisier, or scales both terms of the decomposition by one factor k, the axis term's share of
the raw loss equals its within-dataset share and the two terms are retained equally. For every paired target
(80_paired_engine.py; pooled sums over the benchmark's model group, two-way bootstrap, B = 1,000):
  within share       w = sum mediated_w / sum raw_w
  share of the loss  s = (sum mediated_w - sum mediated_c) / (sum raw_w - sum raw_c)
  excess             s - w     (> 0: the axis term lost proportionally more than the selective term)
  retention          a = sum mediated_c / sum mediated_w,  t = sum selterm_c / sum selterm_w,  difference a - t
  linear contrast    C = (mbar_w tbar_c - tbar_w mbar_c) / rbar_w^2   (round 3, R3-M3e; means over model-drug pairs of
                     the axis term m, the selective term t and raw r). C = excess x relative raw loss = -w (1 - w) (a - t):
                     zero under proportional attenuation, positive when the axis term lost proportionally more, and free
                     of the near-zero denominators of the share of the loss (raw loss) and of a - t (t_w).
Rejection of proportional attenuation: the 95% interval of the excess excludes 0 (reported where the raw loss is
clear, i.e. the lower end of its 95% interval is above 0), and separately the 95% interval of a - t excludes 0, and
(round 3) the 95% interval of the linear contrast C excludes 0. The axis "carries" a
loss only where the excess is clearly positive; a clearly negative excess means the loss runs disproportionately
through the selective term.
Groups: leaderboard = the 21 published models (as the paper; 42's draws, so values reproduce
results/revision_crosslab/bootstrap_pooled.parquet); retraining = DrEval's four learned models (GDSC2 AUC), the
published leaderboard models (CTRPv2 ln IC50 rerun, as 59) and, per model, EnsembleMF and the per-drug ridges;
IMPROVE (replicate-free) = the seven models.
Outputs (results/revision_round2/attenuation/): attenuation_by_target.csv (pooled), attenuation_by_model.csv
(retraining and IMPROVE single models), loss_share_range.csv, run.json.
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
OUT = lib.out_dir('attenuation')


def contrasts(S, models, group):
    g = lambda st, kind: lib.pooled(S, models, group, st, kind)  # noqa: E731
    rw, rc, n = g('raw', 'sum_w'), g('raw', 'sum_c'), g('raw', 'n')
    mw, mc = g('mediated', 'sum_w'), g('mediated', 'sum_c')
    tw, tc = g('selective_term', 'sum_w'), g('selective_term', 'sum_c')
    sbw, sbc, nb = g('selBOTH_sym', 'sum_w'), g('selBOTH_sym', 'sum_c'), g('selBOTH_sym', 'n')
    with np.errstate(invalid='ignore', divide='ignore'):
        w = mw / rw
        loss = (rw - rc) / n
        s = np.where(np.abs(rw - rc) > 1e-9, (mw - mc) / (rw - rc), np.nan)
        a, t = mc / mw, tc / tw
        out = {'raw_mean_within': rw / n, 'raw_mean_cross': rc / n, 'raw_loss': loss, 'within_share': w,
               'cross_share': mc / rc, 'share_of_loss': s, 'excess': s - w, 'axis_retained': a,
               'selterm_retained': t, 'retention_difference': a - t, 'd_mediated': (mc - mw) / n,
               'd_selective_term': (tc - tw) / n, 'd_selBOTH_sym': (sbc - sbw) / nb,
               'linear_contrast': (mw * tc - tw * mc) / (rw * rw)}
    return out, int(round(n[0]))


def row_for(r, S, models, group, gname):
    st, n = contrasts(S, models, group)
    row = {'benchmark': r.benchmark, 'pair': r.pair, 'source': r.source, 'target': r.target, 'family': r.family,
           'group': gname, 'models': len([m for m in group if m in models]), 'model_drug_pairs': n}
    for k, v in st.items():
        row.update(lib.summ(v, k))
    row['loss_clear'] = bool(row['raw_loss_lo'] > 0)            # a clear (positive) loss of raw accuracy
    row['gain_clear'] = bool(row['raw_loss_hi'] < 0)
    row['excess_ci_excludes_0'] = bool(row['excess_lo'] > 0 or row['excess_hi'] < 0)
    row['retdiff_ci_excludes_0'] = bool(row['retention_difference_lo'] > 0 or row['retention_difference_hi'] < 0)
    row['rejects_proportional_excess'] = bool(row['loss_clear'] and row['excess_ci_excludes_0'])
    row['rejects_proportional_retention'] = row['retdiff_ci_excludes_0']
    row['linear_ci_excludes_0'] = bool(row['linear_contrast_lo'] > 0 or row['linear_contrast_hi'] < 0)
    if row['rejects_proportional_excess'] and row['excess'] > 0:
        row['reading'] = 'axis lost disproportionately (through the axis)'
    elif row['rejects_proportional_excess'] and row['excess'] < 0:
        row['reading'] = 'selective term lost disproportionately'
    elif row['loss_clear']:
        row['reading'] = 'clear loss, not distinguishable from proportional attenuation'
    elif row['gain_clear']:
        row['reading'] = 'clear gain in raw accuracy'
    else:
        row['reading'] = 'no clear raw loss'
    return row


def main():
    ix = pd.read_csv(PAIRED / 'pairs.csv')
    rows, mrows = [], []
    for _, r in ix.iterrows():
        f = PAIRED / 'arrays' / f'{r.benchmark}__{r.k}.npz'
        if not f.exists() or r.drugs_used == 0:
            continue
        models, meta, S = lib.load_paired(f)
        if r.source == r.target:
            continue
        groups = {}
        if isinstance(r.pool_main, str) and r.pool_main:
            groups['main'] = r.pool_main.split(';')
        if isinstance(r.pool_alt, str) and r.pool_alt:
            groups['alt'] = r.pool_alt.split(';')
        labels = {'leaderboard': {'main': 'published 21', 'alt': 'published 21 minus drug-agnostic (18)'},
                  'retrain': {'main': 'DrEval learned (retrained) or published leaderboard', 'alt': 'per-drug ridges'},
                  'improve': {'main': 'IMPROVE 7 models'}}[r.benchmark]
        for gk, gm in groups.items():
            if not [m for m in gm if m in models]:
                continue
            rows.append(row_for(r, S, models, gm, labels.get(gk, gk)))
        if r.benchmark in ('retrain', 'improve'):
            for m in models:
                if m.startswith('ref') or m == 'source_measurement':
                    continue
                if r.benchmark == 'retrain' and m.startswith('LB:') and m != 'LB:EnsembleMF':
                    continue
                mrows.append(row_for(r, S, models, [m], m))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'attenuation_by_target.csv', index=False, float_format='%.5g')
    Mo = pd.DataFrame(mrows)
    Mo.to_csv(OUT / 'attenuation_by_model.csv', index=False, float_format='%.5g')
    # range of the axis term's share of the loss over targets with a clear loss (pooled main groups, non-identity,
    # own-panel IMPROVE, excluding CTRPv2 re-scored against itself unless stated)
    main = T[T.group.isin(['published 21', 'DrEval learned (retrained) or published leaderboard', 'IMPROVE 7 models'])]
    main = main[~main.family.isin(['shared_panel'])]
    rng = []
    for (bm, fam), g in main.groupby(['benchmark', 'family']):
        c = g[g.loss_clear]
        subsets = [('all clear-loss targets', c)]
        if bm == 'leaderboard':
            subsets.append(('excluding CTRPv2 re-processed', c[~c.target.astype(str).str.startswith('CTRPv2')]))
        for subset, cc in subsets:
            if not len(cc):
                continue
            rng.append({'benchmark': bm, 'family': fam, 'subset': subset, 'targets_with_clear_loss': len(cc),
                        'share_of_loss_min': cc.share_of_loss.min(), 'share_of_loss_max': cc.share_of_loss.max(),
                        'targets_min_max': f"{cc.loc[cc.share_of_loss.idxmin(), 'pair']} | {cc.loc[cc.share_of_loss.idxmax(), 'pair']}",
                        'n_reject_excess_positive': int((cc.rejects_proportional_excess & (cc.excess > 0)).sum()),
                        'n_reject_excess_negative': int((cc.rejects_proportional_excess & (cc.excess < 0)).sum()),
                        'n_not_rejected': int((~cc.rejects_proportional_excess).sum())})
    R = pd.DataFrame(rng)
    R.to_csv(OUT / 'loss_share_range.csv', index=False, float_format='%.4g')
    lib.write_json(OUT / 'run.json', {'inputs': str(PAIRED), 'B': lib.B, 'pairs': int(len(T))})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 300)
    cols = ['benchmark', 'pair', 'group', 'raw_loss', 'raw_loss_lo', 'raw_loss_hi', 'within_share', 'share_of_loss',
            'share_of_loss_lo', 'share_of_loss_hi', 'excess', 'excess_lo', 'excess_hi', 'retention_difference',
            'retention_difference_lo', 'retention_difference_hi', 'reading']
    print(T[cols].round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
