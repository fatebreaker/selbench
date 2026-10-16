"""Round 2, task 10 (R1-M5; editor item 8): two-way bootstrap rank intervals for the DrEval leaderboard, replacing
the drug-blocked critical difference (which treats drugs that share cell lines as independent blocks).

The 22 non-naive leaderboard models (44's 'published' set, incl. SVR) are ranked, as in 44, on the drugs that all of
them predict within CTRP v2 (CurveCurator ln IC50, the 739 cells predicted by every model), by
  raw          median over drugs of the per-drug correlation r(p, z)
  selective    median over drugs of the per-drug partial correlation r(p, z | g)
  normalized   DrEval's normalized Pearson: per CV split, Pearson over all test pairs of (y - naive) and
               (prediction - naive), naive = NaiveMeanEffectsPredictor, mean over the ten splits (36/44 estimator,
               common cells and common drugs).
Two-way bootstrap with 42's draws (seed 20261014; B = 1,000): cell-line weights enter every per-drug correlation and
every per-split Pearson (pair weight = cell weight x drug multiplicity); drug multiplicities enter the medians. The
same draws for every model and metric, so ranks and their differences are paired. Reported: point rank, 95% rank
interval, the share of model pairs whose order is resolved (95% interval of the score difference excludes 0), and the
Spearman correlation between rankings with its interval.
Outputs (results/revision_round2/rankings/): rank_intervals.csv, pairwise_resolution.csv, ranking_correlations.csv,
summary.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.out_dir('rankings')


def within_arrays(L, models):
    cd = L['cd']
    Z, Gm = L['dec'][cd.REFERENCE]
    U = L['U']
    Zu, Gu = Z.reindex(columns=U), Gm.reindex(columns=U)
    drugs = [d for d in Z.index if d in L['drug_ix']]
    R = L['W'].shape[0]
    arrs = {s: np.full((R, len(models), len(drugs)), np.nan, np.float32) for s in ('raw', 'sel')}
    for j, d in enumerate(drugs):
        z, g = Zu.loc[d].values.astype(float), Gu.loc[d].values.astype(float)
        m0 = np.isfinite(z) & np.isfinite(g)
        for m in L['lbm']:
            if d in L['P'][m].index:
                m0 &= np.isfinite(L['P'][m].loc[d].values)
        if m0.sum() < lib.MIN_CELLS:
            continue
        idx = np.where(m0)[0]
        use, pc = [], []
        for i, m in enumerate(models):
            if d not in L['P'][m].index:
                continue
            p = L['P'][m].loc[d].values[idx].astype(float)
            if not np.all(np.isfinite(p)) or np.std(p) < 1e-12:
                continue
            use.append(i)
            pc.append(p)
        if not use:
            continue
        st = lib.within_stats(z[idx], g[idx], np.column_stack(pc), L['W'][:, idx])
        arrs['raw'][:, use, j] = st['raw']
        arrs['sel'][:, use, j] = st['sel']
    return arrs, drugs


def normalized_reps(L, models, common_drugs):
    nm = lib.load_module('nm36', '36_dreval_normalized.py')
    rs = nm.rs
    cid = pd.read_csv(lib.PROJECT / 'data' / 'dreval' / 'cell_ids.csv', dtype=str).set_index('cell_line_name').cellosaurus_id
    lbm = sorted(m for m in os.listdir(rs.LB / 'LCO') if (rs.LB / 'LCO' / m / 'predictions').exists())
    raw = {m: nm.load_raw(m) for m in lbm}
    naive = raw['NaiveMeanEffectsPredictor'][['drug', 'cell', 'split', 'pred']].rename(columns={'pred': 'naive'})
    refs = pd.read_csv(lib.RES / 'dreval' / 'reference_predictions.csv.gz', dtype={'drug': str})
    common_cells = sorted(set.intersection(*[set(d.cell) for m, d in raw.items() if m not in rs.NAIVE]) & set(refs.cell))
    uix = {c: i for i, c in enumerate(L['U'])}
    W, V = L['W'], L['V']
    R = W.shape[0]
    out = np.full((R, len(models)), np.nan)
    dropped = 0
    for i, m in enumerate(models):
        x = raw[m]
        x = x[x.cell.isin(common_cells) & x.drug.isin(common_drugs)].merge(naive, on=['drug', 'cell', 'split'], how='inner')
        x = x.assign(cu=x.cell.map(cid).map(uix), dv=x.drug.map(L['drug_ix']))
        dropped += int(x.cu.isna().sum() + x.dv.isna().sum())
        x = x.dropna(subset=['cu', 'dv'])
        per = []
        for k, g in x.groupby('split'):
            a, b = (g.y - g.naive).values, (g.pred - g.naive).values
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            w = W[:, g.cu.values[ok].astype(int)] * V[:, g.dv.values[ok].astype(int)]
            F = np.column_stack([np.ones_like(a), a, b, a * a, b * b, a * b])
            S = w @ F
            n = S[:, 0]
            ma, mb = S[:, 1] / n, S[:, 2] / n
            va, vb, cab = S[:, 3] / n - ma ** 2, S[:, 4] / n - mb ** 2, S[:, 5] / n - ma * mb
            per.append(cab / np.sqrt(va * vb))
        out[:, i] = np.mean(np.vstack(per), axis=0)
    return out, dropped, len(common_cells)


def ranks_of(S):
    """S: (R, M) scores (higher = better) -> ranks (1 = best), NaN-safe per row."""
    R = np.full(S.shape, np.nan)
    for b in range(S.shape[0]):
        ok = np.isfinite(S[b])
        R[b, ok] = rankdata(-S[b, ok])
    return R


def main():
    t0 = time.time()
    L = lib.leaderboard_setup()
    models = sorted(m for m in L['P'] if L['mclass'][m] in ('classical', 'neural/factorization'))
    assert len(models) == 22, models
    arrs, drugs = within_arrays(L, models)
    common = np.all(np.isfinite(arrs['raw'][0]) & np.isfinite(arrs['sel'][0]), axis=0)
    cd_idx = np.where(common)[0]
    cdrugs = [drugs[j] for j in cd_idx]
    Vd = L['V'][:, [L['drug_ix'][d] for d in cdrugs]]
    scores = {}
    for name, s in [('raw', 'raw'), ('selective', 'sel')]:
        X = arrs[s][:, :, cd_idx].astype(float)
        scores[name] = np.column_stack([lib.wmedian_rows(X[:, i], Vd) for i in range(len(models))])
    scores['dreval_normalized'], dropped, ncells = normalized_reps(L, models, set(cdrugs))
    print('scores', f'{time.time() - t0:.0f}s', 'common drugs', len(cdrugs), 'normalized pairs dropped', dropped, flush=True)
    rows, prow, crow = [], [], []
    paper = pd.read_csv(lib.RES / 'revision_crosslab' / 'rankings.csv')
    pmap = {'raw': 'raw_within', 'selective': 'selT_within', 'dreval_normalized': 'dreval_normalized_common_drugs'}
    RK = {}
    for name, S in scores.items():
        Rk = ranks_of(S)
        RK[name] = Rk
        pr = paper[paper.ranking == pmap[name]].set_index('model')
        for i, m in enumerate(models):
            lo, hi = np.percentile(Rk[1:, i], [2.5, 97.5])
            rows.append({'metric': name, 'model': m, 'class': L['mclass'][m], 'score': float(S[0, i]),
                         **{k: v for k, v in lib.summ(S[:, i], 'score').items() if k != 'score'},
                         'rank': float(Rk[0, i]), 'rank_lo': float(lo), 'rank_hi': float(hi),
                         'rank_interval_width': float(hi - lo),
                         'paper_score': float(pr.score.get(m, np.nan)), 'paper_rank': float(pr['rank'].get(m, np.nan))})
        res = 0
        npairs = 0
        for a, b in itertools.combinations(range(len(models)), 2):
            d = S[:, a] - S[:, b]
            lo, hi = lib.ci(d)
            resolved = bool(lo > 0 or hi < 0)
            res += resolved
            npairs += 1
            prow.append({'metric': name, 'model_a': models[a], 'model_b': models[b], 'diff': float(d[0]), 'lo': lo,
                         'hi': hi, 'resolved': resolved})
        crow.append({'metric': name, 'pairs': npairs, 'resolved_pairs': res, 'share_resolved': res / npairs,
                     'median_rank_interval_width': float(np.median([r['rank_interval_width'] for r in rows
                                                                    if r['metric'] == name]))})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'rank_intervals.csv', index=False, float_format='%.5g')
    pd.DataFrame(prow).to_csv(OUT / 'pairwise_resolution.csv', index=False, float_format='%.5g')
    cr = []
    for a, b in itertools.combinations(list(scores), 2):
        rho = np.array([np.corrcoef(RK[a][k], RK[b][k])[0, 1] for k in range(RK[a].shape[0])])
        cr.append({'ranking_a': a, 'ranking_b': b, **lib.summ(rho, 'spearman')})
    C = pd.DataFrame(cr)
    C.to_csv(OUT / 'ranking_correlations.csv', index=False, float_format='%.5g')
    R = pd.DataFrame(crow)
    lib.write_json(OUT / 'summary.json', {'models': len(models), 'common_drugs': len(cdrugs), 'normalized_common_cells': ncells,
                                          'normalized_pairs_dropped_outside_draw_universe': dropped, 'B': lib.B,
                                          'resolution': R.to_dict(orient='records'), 'seconds': round(time.time() - t0)})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 100)
    print(T[['metric', 'model', 'score', 'rank', 'rank_lo', 'rank_hi', 'paper_rank', 'paper_score']].round(3).to_string(index=False))
    print(R.to_string(index=False))
    print(C.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
