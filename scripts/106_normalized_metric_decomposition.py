"""How much of DrEval's headline metric is general sensitivity? (round 5; the 'increment over DrEval' question.)

DrEval's "normalized Pearson" subtracts the NaiveMeanEffectsPredictor prediction (drug mean plus, for the unseen cell
lines of leave-cell-line-out splits, a tissue fallback) from measured and predicted ln IC50 and computes Pearson over
all pairs of a split, averaged over the ten splits (36_dreval_normalized.py re-implements it to 1e-15). The naive
prediction cannot know an unseen cell line's general sensitivity, so the residuals keep it. Here the same residuals are
also adjusted for the general-sensitivity axis: within each split and drug, both residuals are regressed on the
evaluation dataset's leave-drug-out axis g (selbench.decompose of the measured responses on the common cells, as in 36)
and the pooled Pearson of what remains is the "axis-adjusted normalized Pearson". The axis-carried fraction is
1 - adjusted / normalized.

Models: the 22 released leaderboard models (as 36), and this study's predictors placed on DrEval's scale exactly as in
36 (fitness_only: training-cell drug mean + per-drug slope on the omics-predicted axis, i.e. no drug-specific cell
ranking; the per-drug ridges). Read-only: imports 28/36 (hashed in the Amendment 3 manifest; not modified).
Outputs: results/revision_round5/normalized_decomposition/{normalized_decomposition.csv, by_split.csv, summary.json}.
Usage (from scripts/): ../../.venv/bin/python 106_normalized_metric_decomposition.py
"""
import importlib.util
import json
import os
import sys

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS  # noqa: E402

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location('n36', HERE / '36_dreval_normalized.py')
n36 = importlib.util.module_from_spec(_s); _s.loader.exec_module(n36)
rs = n36.rs
OUT = RESULTS / 'revision_round5' / 'normalized_decomposition'
NAIVES = ('NaivePredictor', 'NaiveDrugMeanPredictor', 'NaiveMeanEffectsPredictor')


def resid(v, g):
    """Residual of v on [1, g] (least squares) where both are finite; NaN elsewhere."""
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v) & np.isfinite(g)
    if ok.sum() < 5:
        return out
    X = np.c_[np.ones(ok.sum()), g[ok]]
    b = np.linalg.lstsq(X, v[ok], rcond=None)[0]
    out[ok] = v[ok] - X @ b
    return out


def scores(d, naive, G, increment=False):
    """Per split: DrEval's normalized Pearson and the axis-adjusted version (pairs with a finite axis value)."""
    x = d.merge(naive, on=['drug', 'cell', 'split'], how='inner')
    x['g'] = [G.at[a, b] if (a in G.index and b in G.columns) else np.nan for a, b in zip(x.drug, x.cell)]
    rows = []
    for k, t in x.groupby('split'):
        ez = (t.y - t.naive).values
        ep = (t.pred if increment else t.pred - t.naive).values
        ok = np.isfinite(ez) & np.isfinite(ep)
        full = np.corrcoef(ez[ok], ep[ok])[0, 1]
        t = t.assign(ez=ez, ep=ep)[ok & np.isfinite(t.g.values)]
        rz, rp = np.full(len(t), np.nan), np.full(len(t), np.nan)
        for _, ix in t.groupby('drug').indices.items():
            gv = t.g.values[ix]
            rz[ix] = resid(t.ez.values[ix], gv)
            rp[ix] = resid(t.ep.values[ix], gv)
        m = np.isfinite(rz) & np.isfinite(rp)
        same = np.corrcoef(t.ez.values, t.ep.values)[0, 1]      # unadjusted on the pairs with a finite axis
        rows.append({'split': k, 'normalized': full, 'normalized_axis_pairs': same,
                     'normalized_axis_adjusted': np.corrcoef(rz[m], rp[m])[0, 1], 'pairs': int(m.sum())})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = sorted(m for m in os.listdir(rs.LB / 'LCO') if (rs.LB / 'LCO' / m / 'predictions').exists())
    lb = {m: n36.load_raw(m) for m in models}
    naive = lb['NaiveMeanEffectsPredictor'][['drug', 'cell', 'split', 'pred']].rename(columns={'pred': 'naive'})
    obs = pd.concat([d[['drug', 'cell', 'y']] for d in lb.values()]).groupby(['drug', 'cell'], as_index=False).y.mean()
    obs_s = obs.rename(columns={'y': 'response'}).assign(response=lambda t: -t.response)
    sp = pd.concat([d[['cell', 'split']] for m, d in lb.items() if m not in rs.NAIVE and not m.startswith('MultiView')])
    split_of = sp.drop_duplicates().set_index('cell').split
    refs = pd.read_csv(n36.OUT / 'reference_predictions.csv.gz', dtype={'drug': str})
    common = sorted(set.intersection(*[set(d.cell) for m, d in lb.items() if m not in rs.NAIVE]) & set(refs.cell))
    Z, G, _, _ = rs.selbench.decompose(obs_s[obs_s.cell.isin(common)])
    drugs = set(Z.index)

    parts = []
    for m, d in lb.items():
        if m in NAIVES:
            continue
        s = scores(d[d.cell.isin(common) & d.drug.isin(drugs)], naive, G).assign(model=m, kind=rs.model_class(m))
        parts.append(s)

    # this study's predictors on DrEval's scale, exactly as 36 builds them
    tissue = pd.concat([d[['cell', 'tissue']] for d in lb.values()]).drop_duplicates('cell').set_index('cell').tissue
    base = lb['NaiveMeanEffectsPredictor']
    base = base[base.cell.isin(common) & base.drug.isin(drugs)][['drug', 'cell', 'split', 'y', 'pred']].rename(
        columns={'pred': 'naive_pred'})
    fo = refs[refs.model == 'fitness_only'].drop_duplicates('cell').set_index('cell').prediction
    fo_tc = fo - fo.groupby(tissue.reindex(fo.index).values).transform('mean')
    ridge = {m: refs[refs.model == m].set_index(['drug', 'cell']).prediction for m in rs.OURS if m != 'fitness_only'}
    T = []
    for k in range(10):
        tr = [c for c in common if split_of.get(c) != k]
        _, Gtr, _, _ = rs.selbench.decompose(obs_s[obs_s.cell.isin(tr)])
        Y = obs_s[obs_s.cell.isin(tr)].pivot_table(index='drug', columns='cell', values='response')
        a, s_, mu = {}, {}, {}
        for dd in Y.index.intersection(Gtr.index):
            y, g = Y.loc[dd].reindex(Gtr.columns).values, Gtr.loc[dd].values
            ok = np.isfinite(y) & np.isfinite(g)
            if ok.sum() > 25:
                a[dd] = np.cov(y[ok], g[ok])[0, 1] / np.var(g[ok], ddof=1)
                s_[dd], mu[dd] = np.std(y[ok], ddof=1), np.mean(y[ok])
        t = base[base.split == k].copy()
        t['a'], t['s'], t['mu'] = t.drug.map(a), t.drug.map(s_), t.drug.map(mu)
        key = pd.Series(list(zip(t.drug, t.cell)), index=t.index)
        t['fitness_only'] = -(t.mu + t.a * t.cell.map(fo))
        t['fitness_increment'] = -(t.a * t.cell.map(fo))
        t['fitness_increment_tissue_centred'] = -(t.a * t.cell.map(fo_tc))
        for m, ser in ridge.items():
            t[m] = -(t.mu + t.s * key.map(ser).values)
        T.append(t)
    T = pd.concat(T)
    for col in ['fitness_only'] + list(ridge):
        d = T.rename(columns={col: 'pred'})[['drug', 'cell', 'split', 'y', 'pred']].dropna()
        parts.append(scores(d, naive, G).assign(model=col, kind='this study'))
    for col in ['fitness_increment', 'fitness_increment_tissue_centred']:
        d = T.rename(columns={col: 'pred'})[['drug', 'cell', 'split', 'y', 'pred']].dropna()
        parts.append(scores(d, naive, G, increment=True).assign(model=col, kind='this study'))
    S = pd.concat(parts)
    S.to_csv(OUT / 'by_split.csv', index=False, float_format='%.6g')

    def agg(g):
        diff = g.normalized_axis_pairs - g.normalized_axis_adjusted
        return pd.Series({'normalized': g.normalized.mean(), 'normalized_sd': g.normalized.std(ddof=1),
                          'normalized_axis_pairs': g.normalized_axis_pairs.mean(),
                          'axis_adjusted': g.normalized_axis_adjusted.mean(),
                          'axis_adjusted_sd': g.normalized_axis_adjusted.std(ddof=1),
                          'axis_carried': diff.mean(), 'axis_carried_sd': diff.std(ddof=1),
                          'axis_fraction': 1 - g.normalized_axis_adjusted.mean() / g.normalized_axis_pairs.mean(),
                          'pairs': g.pairs.mean()})
    R = S.groupby(['model', 'kind']).apply(agg).reset_index()
    pub = R[R.kind.isin(['classical', 'neural/factorization'])]
    R['rank_normalized_among_published'] = R.normalized.apply(lambda v: int((pub.normalized > v).sum()) + 1)
    R['rank_adjusted_among_published'] = R.axis_adjusted.apply(lambda v: int((pub.axis_adjusted > v).sum()) + 1)
    R = R.sort_values('normalized', ascending=False)
    R.to_csv(OUT / 'normalized_decomposition.csv', index=False, float_format='%.6g')

    fo_r = R.set_index('model').loc['fitness_only']
    summ = {'published_models': int(len(pub)),
            'published_axis_fraction_median': float(pub.axis_fraction.median()),
            'published_axis_fraction_range': [float(pub.axis_fraction.min()), float(pub.axis_fraction.max())],
            'pooled_published_axis_fraction': float(1 - pub.axis_adjusted.sum() / pub.normalized_axis_pairs.sum()),
            'fitness_only': {k: float(fo_r[k]) for k in ['normalized', 'axis_adjusted', 'axis_fraction']},
            'fitness_only_exceeds_published_on_normalized': int((pub.normalized < fo_r.normalized).sum()),
            'fitness_only_exceeds_published_on_adjusted': int((pub.axis_adjusted < fo_r.axis_adjusted).sum()),
            'spearman_normalized_vs_adjusted_published': float(pub[['normalized', 'axis_adjusted']].corr('spearman').iloc[0, 1]),
            'max_abs_check_vs_36': None}
    ref36 = pd.read_csv(n36.OUT / 'normalized_metric.csv').set_index('model').normalized_pearson
    both = R.set_index('model').normalized.reindex(ref36.index).dropna()
    summ['max_abs_check_vs_36'] = float((both - ref36.reindex(both.index)).abs().max())
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(R[['model', 'kind', 'normalized', 'axis_adjusted', 'axis_fraction', 'rank_normalized_among_published',
             'rank_adjusted_among_published']].round(3).to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
