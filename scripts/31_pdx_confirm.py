"""Amendment 2 confirmatory evaluation in PDX_Bruna patient-derived breast-cancer models (frozen before retrieval).

Input: data/processed/pdx_bruna/pdx_long.csv.gz (compound, sample, sens; higher = more sensitive) written by
32_load_pdx.py. Frozen transfer predictions: results/pdx_transfer/pred_{sel,raw,sel_cn,raw_cn}.parquet.
Processing is that of the frozen 07_confirm.build with the registered PDX threshold: drugs measured in >= 20 of the
37 models (instead of 30) and >= 10 distinct values; rank-normal within drug; leave-drug-out fitness axis; selective
transfer = partial correlation of a frozen prediction with the drug's PDX profile given the PDX fitness axis, over
models with both (>= 15). Implements H8 and H9 of protocol/amendment2_pdx_confirmation.md.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from common import PROCESSED, RESULTS, write_json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('confirm', HERE / '07_confirm.py')
confirm = importlib.util.module_from_spec(spec); spec.loader.exec_module(confirm)
REPRODUCIBLE, NONREPRODUCIBLE = 0.30, 0.15
MIN_MODELS_PER_DRUG = 20
MIN_PAIRS = 15
MIN_DRUGS_H8 = 8


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_PAIRS or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def partial(x, y, g):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(g)
    if m.sum() < MIN_PAIRS:
        return np.nan, int(m.sum())
    X = np.c_[np.ones(m.sum()), g[m]]
    rx = x[m] - X @ np.linalg.lstsq(X, x[m], rcond=None)[0]
    ry = y[m] - X @ np.linalg.lstsq(X, y[m], rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


def evaluate(long_path, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(confirm.SEED)
    confirm.MIN_CELLS = MIN_MODELS_PER_DRUG
    long = pd.read_csv(long_path).rename(columns={'sample': 'depmap_id'})
    M, Z, Gl, R = confirm.build(long)
    res = {'drugs': int(Z.shape[0]), 'models': int(Z.shape[1]), 'split_half_reliability': confirm.split_half(Z, rng)}
    res['median_fitness_r2'] = float(np.nanmedian([corr(Z.loc[d].values, Gl.loc[d].values)[0] ** 2 for d in Z.index]))
    dpairs = pd.read_csv(RESULTS / 'decomposition' / 'cross_screen_pairs.csv')
    rep = dpairs.groupby('compound').rho_sel.mean()
    cls = pd.Series(np.where(rep >= REPRODUCIBLE, 'reproducible',
                             np.where(rep < NONREPRODUCIBLE, 'non_reproducible', 'intermediate')), index=rep.index)
    rows = []
    for fam in ['sel', 'raw', 'sel_cn', 'raw_cn']:
        P = pd.read_parquet(RESULTS / 'pdx_transfer' / f'pred_{fam}.parquet')
        for d in Z.index.intersection(P.index):
            p = P.loc[d].reindex(Z.columns).values.astype(float)
            sel, n = partial(p, Z.loc[d].values, Gl.loc[d].values)
            rows.append({'compound': d, 'family': fam, 'n': n, 'raw': corr(p, Z.loc[d].values)[0], 'sel': sel,
                         'class': cls.get(d, 'single_screen')})
    sc = pd.DataFrame(rows)
    sc.to_csv(out / 'scores.csv', index=False)
    s = sc[sc.family == 'sel']
    a = s[s['class'] == 'reproducible'].sel.dropna(); b = s[s['class'] == 'non_reproducible'].sel.dropna()
    h8 = {'drugs': len(a), 'median_sel': float(a.median()) if len(a) else None}
    if len(a) >= MIN_DRUGS_H8:
        h8['wilcoxon_p_one_sided'] = float(wilcoxon(a, alternative='greater').pvalue)
        h8['pass'] = bool(h8['wilcoxon_p_one_sided'] < 0.05)
    else:
        h8['estimable'] = False
    h9 = {'n_reproducible': len(a), 'n_non_reproducible': len(b),
          'median_reproducible': float(a.median()) if len(a) else None,
          'median_non_reproducible': float(b.median()) if len(b) else None}
    if len(a) >= 3 and len(b) >= 3:
        h9['mwu_p_one_sided'] = float(mannwhitneyu(a, b, alternative='greater').pvalue)
        h9['pass'] = bool(h9['mwu_p_one_sided'] < 0.05)
    else:
        h9['estimable'] = False
    res['H8'], res['H9'] = h8, h9
    res['secondary'] = sc.groupby(['family', 'class']).agg(n=('sel', 'size'), raw=('raw', 'median'),
                                                           sel=('sel', 'median')).reset_index().to_dict(orient='records')
    write_json(out / 'hypotheses.json', res)
    print(res)
    return res


def main():
    base = PROCESSED / 'pdx_bruna'
    evaluate(base / 'pdx_long.csv.gz', RESULTS / 'confirmation_PDX_Bruna')
    if (base / 'pdx_lnic50_long.csv.gz').exists():
        evaluate(base / 'pdx_lnic50_long.csv.gz', RESULTS / 'confirmation_PDX_Bruna_lnIC50')


if __name__ == '__main__':
    sys.exit(main())
