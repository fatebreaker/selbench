"""Score the DrEval leaderboard models and this study's references on DrEval's OWN headline metric.

DrEval's "normalized Pearson" (drevalpy visualization.utils._normalize_metrics_by_mean_effects) subtracts the
NaiveMeanEffectsPredictor prediction (drug mean + tissue fallback for unseen cell lines in LCO) from both the observed
and the predicted response, then computes Pearson over all pairs of a CV split, averaged over the ten splits. The
re-implementation is validated against the leaderboard's reported values.

This study's references predict on the sensitivity scale (rank-normal z of 28_dreval_rescore.py). They are placed on
DrEval's ln IC50 scale with training-cell statistics only: per split and drug, the training-cell mean (mu), SD (s) and
the slope (a) of the response on the training cells' leave-drug-out fitness axis. Ridge models: -(mu + s * z_hat);
fitness-only: -(mu + a * f_hat) and, as a stricter variant, the fitness increment alone (-a * f_hat after the naive
prediction) and that increment centred within tissue.
Outputs: results/dreval/normalized_metric.csv, normalized_metric_validation.csv.
"""
import importlib.util
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS  # noqa: E402

spec = importlib.util.spec_from_file_location('rs', Path(__file__).resolve().parent / '28_dreval_rescore.py')
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
OUT = RESULTS / 'dreval'


def load_raw(m):
    """Leaderboard predictions on DrEval's own scale (ln IC50), with split labels."""
    fs = sorted((rs.LB / 'LCO' / m / 'predictions').glob('predictions_split_*.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'pubchem_id': str}).assign(split=int(f.stem.rsplit('_', 1)[1])) for f in fs])
    return d.rename(columns={'pubchem_id': 'drug', 'cell_line_name': 'cell', 'predictions': 'pred', 'response': 'y'})


def norm_pearson(d, naive, col='pred', increment=False):
    x = d.merge(naive, on=['drug', 'cell', 'split'], how='inner')
    per = []
    for _, g in x.groupby('split'):
        yt = (g.y - g.naive).values
        yp = g[col].values if increment else (g[col] - g.naive).values
        ok = np.isfinite(yt) & np.isfinite(yp)
        per.append(np.corrcoef(yt[ok], yp[ok])[0, 1])
    return float(np.mean(per)), float(np.std(per, ddof=1))


def main():
    models = sorted(m for m in os.listdir(rs.LB / 'LCO') if (rs.LB / 'LCO' / m / 'predictions').exists())
    lb = {m: load_raw(m) for m in models}
    naive = lb['NaiveMeanEffectsPredictor'][['drug', 'cell', 'split', 'pred']].rename(columns={'pred': 'naive'})
    ev = pd.read_csv(rs.LB / 'evaluation_results.csv')
    ev['model'] = ev.algorithm.str.replace(r' \(.*\)$', '', regex=True)
    reported = ev.groupby('model')['Pearson: normalized'].mean()
    val = pd.DataFrame([{'model': m, 'reported': reported.get(m), 'reimplemented': norm_pearson(lb[m], naive)[0]}
                        for m in models if m not in ('NaivePredictor', 'NaiveDrugMeanPredictor', 'NaiveMeanEffectsPredictor')])
    val.to_csv(OUT / 'normalized_metric_validation.csv', index=False)
    assert (val.reported - val.reimplemented).abs().max() < 1e-3, val

    # common evaluation set of 28_dreval_rescore.py
    obs = pd.concat([d[['drug', 'cell', 'y']] for d in lb.values()]).groupby(['drug', 'cell'], as_index=False).y.mean()
    obs_s = obs.rename(columns={'y': 'response'}).assign(response=lambda t: -t.response)
    sp = pd.concat([d[['cell', 'split']] for m, d in lb.items() if m not in rs.NAIVE and not m.startswith('MultiView')])
    split_of = sp.drop_duplicates().set_index('cell').split
    refs = pd.read_csv(OUT / 'reference_predictions.csv.gz', dtype={'drug': str})
    common = sorted(set.intersection(*[set(d.cell) for m, d in lb.items() if m not in rs.NAIVE]) & set(refs.cell))
    Z, _, _, _ = rs.selbench.decompose(obs_s[obs_s.cell.isin(common)])
    drugs = set(Z.index)
    tissue = pd.concat([d[['cell', 'tissue']] for d in lb.values()]).drop_duplicates('cell').set_index('cell').tissue
    rows = []
    for m, d in lb.items():
        if m in ('NaivePredictor', 'NaiveDrugMeanPredictor', 'NaiveMeanEffectsPredictor'):
            continue
        mu_, sd_ = norm_pearson(d[d.cell.isin(common) & d.drug.isin(drugs)], naive)
        rows.append({'model': m, 'class': rs.model_class(m), 'normalized_pearson': mu_, 'split_sd': sd_})
    base = lb['NaiveMeanEffectsPredictor']
    base = base[base.cell.isin(common) & base.drug.isin(drugs)][['drug', 'cell', 'split', 'y', 'pred']].rename(
        columns={'pred': 'naive_pred'})
    fo = refs[refs.model == 'fitness_only'].drop_duplicates('cell').set_index('cell').prediction
    fo_tc = fo - fo.groupby(tissue.reindex(fo.index).values).transform('mean')
    ridge = {m: refs[refs.model == m].set_index(['drug', 'cell']).prediction for m in rs.OURS if m != 'fitness_only'}
    parts = []
    for k in range(10):
        tr = [c for c in common if split_of.get(c) != k]
        _, Gtr, _, _ = rs.selbench.decompose(obs_s[obs_s.cell.isin(tr)])
        Y = obs_s[obs_s.cell.isin(tr)].pivot_table(index='drug', columns='cell', values='response')
        a, s, mu = {}, {}, {}
        for dd in Y.index.intersection(Gtr.index):
            y, g = Y.loc[dd].reindex(Gtr.columns).values, Gtr.loc[dd].values
            ok = np.isfinite(y) & np.isfinite(g)
            if ok.sum() > 25:
                a[dd] = np.cov(y[ok], g[ok])[0, 1] / np.var(g[ok], ddof=1)
                s[dd], mu[dd] = np.std(y[ok], ddof=1), np.mean(y[ok])
        t = base[base.split == k].copy()
        t['a'], t['s'], t['mu'] = t.drug.map(a), t.drug.map(s), t.drug.map(mu)
        key = pd.Series(list(zip(t.drug, t.cell)), index=t.index)
        t['fitness_only'] = -(t.mu + t.a * t.cell.map(fo))
        t['fitness_increment'] = -(t.a * t.cell.map(fo))
        t['fitness_increment_tissue_centred'] = -(t.a * t.cell.map(fo_tc))
        for m, ser in ridge.items():
            t[m] = -(t.mu + t.s * key.map(ser).values)
        parts.append(t)
    T = pd.concat(parts).rename(columns={'naive_pred': 'naive_x'})
    for col in ['fitness_only'] + list(ridge):
        mu_, sd_ = norm_pearson(T.rename(columns={col: 'pred'})[['drug', 'cell', 'split', 'y', 'pred']].dropna(), naive)
        rows.append({'model': col, 'class': 'this study', 'normalized_pearson': mu_, 'split_sd': sd_})
    for col in ['fitness_increment', 'fitness_increment_tissue_centred']:
        mu_, sd_ = norm_pearson(T.rename(columns={col: 'pred'})[['drug', 'cell', 'split', 'y', 'pred']].dropna(), naive,
                                increment=True)
        rows.append({'model': col, 'class': 'this study', 'normalized_pearson': mu_, 'split_sd': sd_})
    R = pd.DataFrame(rows).sort_values('normalized_pearson', ascending=False)
    R.to_csv(OUT / 'normalized_metric.csv', index=False)
    print(R.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
