"""Two-part model: omics-predicted fitness axis plus a per-drug selective model, on the DrEval leaderboard splits.

For each split, the training cell lines' decomposition (selbench.decompose) gives each drug's slope b_d of its
rank-normal profile z on the leave-drug-out fitness axis. The two-part prediction for a held-out cell line is
z_hat = b_d * f_hat_c + r_hat_dc, where f_hat is the fitness-only ridge (one omics-predicted value per cell line) and
r_hat the per-drug kernel ridge trained on the selective residual (28_dreval_rescore.py reference predictions).
The prediction is scored like the leaderboard models: within CTRPv2 (raw, selective, fitness-mediated share), in other
laboratories with identical predictions (as in 37_leaderboard_crosslab.py) and on DrEval's normalized Pearson
(placed on the ln IC50 scale as in 36_dreval_normalized.py).
Outputs: results/dreval/two_part_predictions.csv.gz, two_part_summary.csv.
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
nm = load('nm', '36_dreval_normalized.py')
OUT = RESULTS / 'dreval'
PARTS = {'two_part_lm': 'ridge_sel_lm', 'two_part_expr': 'ridge_sel_expr'}


def main():
    lb = rs.load_leaderboard()
    obs = pd.concat([d[['drug', 'cell', 'response']] for d in lb.values()]).groupby(['drug', 'cell'], as_index=False).response.mean()
    sp = pd.concat([d[['cell', 'split']] for m, d in lb.items() if m not in rs.NAIVE and not m.startswith('MultiView')])
    split_of = sp.drop_duplicates().set_index('cell').split
    refs = pd.read_csv(OUT / 'reference_predictions.csv.gz', dtype={'drug': str})
    fo = refs[refs.model == 'fitness_only'].drop_duplicates('cell').set_index('cell').prediction
    cells = sorted(set(refs.cell))
    out = []
    for k in range(10):
        tr = [c for c in cells if split_of.get(c) != k]
        te = [c for c in cells if split_of.get(c) == k]
        Ztr, Gtr, _, _ = rs.selbench.decompose(obs[obs.cell.isin(tr)])
        slope = {}
        for d in Ztr.index:
            z, g = Ztr.loc[d].values, Gtr.loc[d].values
            m = np.isfinite(z) & np.isfinite(g)
            if m.sum() > 25:
                slope[d] = np.cov(z[m], g[m])[0, 1] / np.var(g[m], ddof=1)
        for name, src in PARTS.items():
            r = refs[(refs.model == src) & refs.cell.isin(te)]
            r = r.assign(b=r.drug.map(slope), f=r.cell.map(fo)).dropna(subset=['b', 'f', 'prediction'])
            out.append(r.assign(model=name, prediction=r.b * r.f + r.prediction)[['drug', 'cell', 'prediction', 'model']])
    P = pd.concat(out)
    P.to_csv(OUT / 'two_part_predictions.csv.gz', index=False, float_format='%.5g')

    # within CTRPv2 on the leaderboard's common cells (as 28_dreval_rescore.py)
    pred_cells = [set(d.cell) for m, d in lb.items() if m not in rs.NAIVE]
    common = sorted(set.intersection(*pred_cells) & set(cells))
    Z, Gm, _, _ = rs.selbench.decompose(obs[obs.cell.isin(common)])
    rows = []
    for name in list(PARTS) + ['ridge_sel_lm', 'ridge_raw_lm', 'ridge_raw_expr', 'ridge_sel_expr']:
        src = P[P.model == name] if name in PARTS else refs[refs.model == name]
        M = src[src.cell.isin(common)].pivot_table(index='drug', columns='cell', values='prediction')
        T = rs.score(M, Z, Gm)
        ok = T.raw.notna() & T.selective.notna()
        rows.append({'model': name, 'where': 'CTRPv2', 'drugs': int(ok.sum()), 'raw': float(T.raw.median()),
                     'selective': float(T.selective.median()),
                     'mediated_share': float(T.mediated[ok].mean() / T.raw[ok].mean())})
    # cross-laboratory, identical predictions (cell names -> Cellosaurus)
    cid = pd.read_csv(PROJECT / 'data' / 'dreval' / 'cell_ids.csv', dtype=str).set_index('cell_line_name').cellosaurus_id
    Zs, Gs, _, _ = cs.selbench.decompose(cs.observed('predictions'))
    for t in ['GDSC1', 'GDSC2', 'CTRPv1']:
        Zt, Gt, _, _ = cs.selbench.decompose(cs.observed(f'cross-study-{t}'))
        drugs = Zs.index.intersection(Zt.index)
        for name in list(PARTS) + ['ridge_sel_lm', 'ridge_raw_lm']:
            src = P[P.model == name] if name in PARTS else refs[refs.model == name]
            M = src.assign(cell=src.cell.map(cid)).dropna(subset=['cell']).groupby(['drug', 'cell']).prediction.mean().unstack()
            cc = Zs.columns.intersection(Zt.columns).intersection(M.columns)
            tw, tc = cs.score_pair(M.reindex(index=drugs), Zs, Gs, Zt, Gt, cc)
            for where, T in [(f'CTRPv2 paired with {t}', tw), (t, tc)]:
                rows.append({'model': name, 'where': where,
                             'drugs': int(T.raw.notna().sum()), 'raw': float(T.raw.median()),
                             'selective': float(T.selective.median()),
                             'mediated_share': float(T.mediated.mean() / T.raw.mean())})
    # DrEval's own normalized Pearson, with ln IC50 placement from training-cell statistics
    raw = {m: nm.load_raw(m) for m in ['NaiveMeanEffectsPredictor', 'EnsembleMF']}
    naive = raw['NaiveMeanEffectsPredictor'][['drug', 'cell', 'split', 'pred']].rename(columns={'pred': 'naive'})
    base = raw['NaiveMeanEffectsPredictor']
    base = base[base.cell.isin(common) & base.drug.isin(set(Z.index))][['drug', 'cell', 'split', 'y']]
    obs_s = obs  # sensitivity scale (higher = more sensitive)
    parts = []
    for k in range(10):
        tr = [c for c in common if split_of.get(c) != k]
        Y = obs_s[obs_s.cell.isin(tr)].pivot_table(index='drug', columns='cell', values='response')
        mu, sd = Y.mean(axis=1), Y.std(axis=1, ddof=1)
        t = base[base.split == k].copy()
        t['mu'], t['s'] = t.drug.map(mu), t.drug.map(sd)
        for name in PARTS:
            q = P[P.model == name].set_index(['drug', 'cell']).prediction
            t[name] = -(t.mu + t.s * pd.Series(list(zip(t.drug, t.cell)), index=t.index).map(q).values)
        parts.append(t)
    T = pd.concat(parts)
    for name in PARTS:
        m_, s_ = nm.norm_pearson(T.rename(columns={name: 'pred'})[['drug', 'cell', 'split', 'y', 'pred']].dropna(), naive)
        rows.append({'model': name, 'where': 'DrEval normalized Pearson', 'drugs': None, 'raw': m_, 'selective': None,
                     'mediated_share': None, 'split_sd': s_})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'two_part_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(S.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
