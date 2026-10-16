"""Axis and selective capture relative to a perfect predictor (revision round 2; Reviewer 1 M1).

The per-drug correlation splits exactly into an axis term r_pg * r_zg and a selective term
sqrt((1 - r_pg^2)(1 - r_zg^2)) * r_pz.g. For the perfect predictor p = z the two terms are r_zg^2 and 1 - r_zg^2.
Each model's terms are therefore read as fractions of the perfect predictor's, summed over drugs:
  axis capture       A = sum_d r_pg r_zg / sum_d r_zg^2
  selective capture  S = sum_d selective term / sum_d (1 - r_zg^2)
  axis share         = sum_d r_pg r_zg / sum_d r_pz (as in the paper);  perfect-predictor share = mean r_zg^2
so that share / perfect share = A / (A m + S (1 - m)) with m the mean r_zg^2: the ratio exceeds 1 whenever S < A, and
is bounded by 1 / m. A drug-agnostic predictor has S near 0; a predictor that is right about the axis but not about
selectivity has A near 1 and S small.

Measurement-error sensitivity: measurement error is drug-specific and falls into the selective part. A predictor of the
reproducible profile (axis taken as noise-free) would have share m_d / (m_d + (1 - m_d) R_sel) per drug, with R_sel the
reliability of the selective residual. R_sel is unknown within a run; between GDSC1's two sites it is about 0.39
(results/revision_axis/gdsc1_sites_perdrug.csv), which is used with 0.55 and 0.70 as a sensitivity range.

Inputs: results/revision_retrain/within_per_drug.csv.gz (DrEval's released CTRP v2 ln IC50 leaderboard predictions,
'LB:' models, and the drevalpy retraining), results/revision_improve/within_per_drug.csv.gz (IMPROVE).
Output: results/revision_capture/{capture_by_model.csv, capture_by_setting.csv, run.json}. Drug bootstrap, B = 1,000.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RES = PROJECT / 'results'
OUT = RES / 'revision_capture'
B = 1000
SEED = 7
R_SEL = [0.39, 0.55, 0.70]
DRUG_AGNOSTIC = {'LB:MultiViewRandomForest', 'FitnessOnlyRidge', 'floor_cell_mean', 'LB:NaiveCellLineMeanPredictor'}
NAIVE = ('Naive',)


def load():
    r = pd.read_csv(RES / 'revision_retrain' / 'within_per_drug.csv.gz')
    r = r[r.panel == 'main'].rename(columns={'setting': 'setting'})
    r['benchmark'] = np.where(r.model.str.startswith('LB:'), 'DrEval released', 'retrained (drevalpy)')
    r = r[['benchmark', 'setting', 'model', 'drug', 'raw', 'r_pg', 'r_zg', 'mediated', 'selective_term', 'selective']]
    im = pd.read_csv(RES / 'revision_improve' / 'within_per_drug.csv.gz')
    im = im.rename(columns={'alignment': 'r_pg', 'loading': 'r_zg'})
    im['benchmark'] = 'IMPROVE (random pairs)'
    im['setting'] = im.dataset.map(lambda d: f'{d}_AUC')
    im = im[['benchmark', 'setting', 'model', 'drug', 'raw', 'r_pg', 'r_zg', 'mediated', 'selective_term', 'selective']]
    d = pd.concat([r, im], ignore_index=True)
    return d.dropna(subset=['raw', 'r_pg', 'r_zg', 'mediated', 'selective_term'])


def stats(x):
    m = (x.r_zg ** 2).mean()
    A = x.mediated.sum() / (x.r_zg ** 2).sum()
    S = x.selective_term.sum() / (1 - x.r_zg ** 2).sum()
    share = x.mediated.sum() / x.raw.sum()
    out = {'axis_capture': A, 'selective_capture': S, 'axis_share': share, 'perfect_share': m,
           'ratio': share / m, 'ratio_bound': 1 / m, 'mean_r_pg': x.r_pg.mean(), 'mean_r_zg': x.r_zg.mean(),
           'mean_abs_r_zg': x.r_zg.abs().mean(), 'median_raw': x.raw.median(), 'median_selective': x.selective.median()}
    for R in R_SEL:
        ref = (x.r_zg ** 2 / (x.r_zg ** 2 + (1 - x.r_zg ** 2) * R)).mean()
        out[f'reproducible_share_R{R:.2f}'] = ref
        out[f'ratio_R{R:.2f}'] = share / ref
    return out


def boot(x, rng):
    keys = ['axis_capture', 'selective_capture', 'ratio', 'ratio_R0.39']
    vals = {k: [] for k in keys}
    arr = x.reset_index(drop=True)
    n = len(arr)
    for _ in range(B):
        s = stats(arr.iloc[rng.integers(0, n, n)])
        for k in keys:
            vals[k].append(s[k])
    return {f'{k}_{q}': np.nanpercentile(v, p) for k, v in vals.items() for q, p in [('lo', 2.5), ('hi', 97.5)]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load()
    rows = []
    for (bm, st, mdl), x in d.groupby(['benchmark', 'setting', 'model']):
        s = {'benchmark': bm, 'setting': st, 'model': mdl, 'drugs': len(x)}
        s.update(stats(x))
        s.update(boot(x, rng))
        s['drug_agnostic'] = mdl in DRUG_AGNOSTIC
        s['naive'] = any(t in mdl for t in NAIVE)
        rows.append(s)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / 'capture_by_model.csv', index=False)
    learned = M[~M.drug_agnostic & ~M.naive & (M.median_raw > 0.03)]
    S = learned.groupby(['benchmark', 'setting']).agg(
        models=('model', 'size'), axis_capture_median=('axis_capture', 'median'),
        axis_capture_min=('axis_capture', 'min'), axis_capture_max=('axis_capture', 'max'),
        selective_capture_median=('selective_capture', 'median'), selective_capture_min=('selective_capture', 'min'),
        selective_capture_max=('selective_capture', 'max'), ratio_min=('ratio', 'min'), ratio_max=('ratio', 'max'),
        ratio_R039_min=('ratio_R0.39', 'min'), ratio_R039_max=('ratio_R0.39', 'max'),
        perfect_share=('perfect_share', 'median'), reproducible_share_R039=('reproducible_share_R0.39', 'median'),
        reproducible_share_R070=('reproducible_share_R0.70', 'median'), ratio_bound=('ratio_bound', 'median'),
        mean_r_pg_median=('mean_r_pg', 'median'), mean_r_zg=('mean_r_zg', 'median')).reset_index()
    S.to_csv(OUT / 'capture_by_setting.csv', index=False)
    json.dump({'B': B, 'seed': SEED, 'R_sel': R_SEL, 'models': len(M),
               'learned_filter': 'not drug-agnostic, not naive, median raw > 0.03'}, open(OUT / 'run.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    print(S.round(3).to_string())


if __name__ == '__main__':
    main()
