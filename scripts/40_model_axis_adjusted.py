"""Is a raw-trained model's low selective score lack of drug-specific signal, or dilution by its own fitness estimate?

The selective score partials the observed fitness axis g out of a model's predictions. If a model predicts
p = f_hat + s_hat with an imperfect fitness estimate f_hat, the error f_hat - g stays in the residual and dilutes the
selective score even when s_hat is informative. Here each model's own predicted fitness axis g_hat (the leave-drug-out
mean of its rank-normal predicted profiles) is partialled out as well: the model-axis-adjusted selective score is the
partial correlation of p with z given g and g_hat. This isolates the drug-specific part of each prediction.
Within CTRPv2 (leaderboard, leave-cell-line-out, common drugs and cell lines) and, with identical predictions, in GDSC1
and GDSC2. Output: results/dreval/model_axis_adjusted.csv.
"""
import importlib.util
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import PROJECT, RESULTS  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


rs = load('rs', '28_dreval_rescore.py')
cs = load('cs', '35_dreval_crossstudy.py')


def model_axis(P):
    """Leave-drug-out mean of the rank-normal predicted profiles (the model's own fitness axis)."""
    Zp = pd.DataFrame(np.vstack([rs.selbench.rank_normal(r) for r in P.values]), index=P.index, columns=P.columns)
    v = Zp.values; ok = np.isfinite(v)
    tot, cnt = np.nansum(v, axis=0), ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return pd.DataFrame((tot[None] - np.where(ok, v, 0)) / (cnt[None] - ok), index=P.index, columns=P.columns)


def partial(x, y, covs):
    m = np.isfinite(x) & np.isfinite(y) & np.all([np.isfinite(c) for c in covs], axis=0)
    if m.sum() < 30 or np.std(x[m]) < 1e-12:
        return np.nan
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in covs])
    rx = x[m] - X @ np.linalg.lstsq(X, x[m], rcond=None)[0]
    ry = y[m] - X @ np.linalg.lstsq(X, y[m], rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def score(P, Z, G, cells):
    P = P.reindex(columns=cells)
    Gh = model_axis(P.loc[P.notna().sum(axis=1) >= 30])
    rows = []
    for d in Gh.index.intersection(Z.index):
        p = P.loc[d].values.astype(float)
        z = Z.loc[d].reindex(cells).values.astype(float)
        g = G.loc[d].reindex(cells).values.astype(float)
        gh = Gh.loc[d].values.astype(float)
        rows.append({'drug': d, 'selective': partial(p, z, [g]), 'adjusted': partial(p, z, [g, gh])})
    return pd.DataFrame(rows)


def main():
    lb = rs.load_leaderboard()
    refs = pd.read_csv(RESULTS / 'dreval' / 'reference_predictions.csv.gz', dtype={'drug': str})
    models = {m: d[['drug', 'cell', 'prediction']] for m, d in lb.items() if m not in rs.CONSTANT}
    for m in ['ridge_raw_lm', 'ridge_sel_lm', 'ridge_raw_expr', 'ridge_sel_expr']:
        models[m] = refs[refs.model == m][['drug', 'cell', 'prediction']]
    obs = pd.concat([d[['drug', 'cell', 'response']] for d in lb.values()]).groupby(['drug', 'cell'], as_index=False).response.mean()
    pred_cells = [set(d.cell) for m, d in lb.items() if m not in rs.NAIVE]
    common = sorted(set.intersection(*pred_cells) & set(refs.cell))
    Z, G, _, _ = rs.selbench.decompose(obs[obs.cell.isin(common)])
    rows = []
    for m, d in models.items():
        P = d[d.cell.isin(common)].pivot_table(index='drug', columns='cell', values='prediction')
        T = score(P, Z, G, pd.Index(common))
        rows.append({'model': m, 'class': rs.model_class(m), 'where': 'CTRPv2', 'drugs': int(T.adjusted.notna().sum()),
                     'selective': float(T.selective.median()), 'adjusted': float(T.adjusted.median())})
    cid = pd.read_csv(PROJECT / 'data' / 'dreval' / 'cell_ids.csv', dtype=str).set_index('cell_line_name').cellosaurus_id
    Zs, Gs, _, _ = cs.selbench.decompose(cs.observed('predictions'))
    for t in ['GDSC1', 'GDSC2']:
        Zt, Gt, _, _ = cs.selbench.decompose(cs.observed(f'cross-study-{t}'))
        drugs = Zs.index.intersection(Zt.index)
        Ps = {m: d.assign(cell=d.cell.map(cid)).dropna(subset=['cell']).groupby(['drug', 'cell']).prediction.mean().unstack()
              for m, d in models.items()}
        cells = Zs.columns.intersection(Zt.columns)
        for P in Ps.values():
            cells = cells.intersection(P.columns)
        Ztm = Zt.reindex(index=drugs, columns=cells).where(Zs.reindex(index=drugs, columns=cells).notna())
        for m, P in Ps.items():
            T = score(P.reindex(index=drugs.intersection(P.index)), Ztm, Gt, cells)
            rows.append({'model': m, 'class': rs.model_class(m), 'where': t, 'drugs': int(T.adjusted.notna().sum()),
                         'selective': float(T.selective.median()), 'adjusted': float(T.adjusted.median())})
    S = pd.DataFrame(rows)
    S.to_csv(RESULTS / 'dreval' / 'model_axis_adjusted.csv', index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
    for w in ['CTRPv2', 'GDSC1', 'GDSC2']:
        print('==', w)
        print(S[S['where'] == w].sort_values('adjusted', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
