"""Amendment 1 confirmatory evaluation in BeatAML primary AML samples (frozen before retrieval).

Input: data/processed/beataml/beataml_long.csv.gz with columns compound (discovery key or 'BA:<name>'), sample
(dbgap RNA-seq sample id) and sens (higher = more sensitive). Frozen transfer predictions:
results/beataml_transfer/pred_{sel,raw}.parquet. Implements H6 and H7 of protocol/amendment1_extended_confirmation.md.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from common import PROCESSED, RESULTS, write_json
from modeling import corr, partial_corr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('confirm', HERE / '07_confirm.py')
confirm = importlib.util.module_from_spec(spec); spec.loader.exec_module(confirm)
REPRODUCIBLE, NONREPRODUCIBLE = 0.30, 0.15


def main(long_path=PROCESSED / 'beataml' / 'beataml_long.csv.gz', out=RESULTS / 'confirmation_BeatAML'):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(confirm.SEED)
    long = pd.read_csv(long_path).rename(columns={'sample': 'depmap_id'})
    M, Z, Gl, R = confirm.build(long)
    res = {'drugs': int(Z.shape[0]), 'samples': int(Z.shape[1]), 'split_half_reliability': confirm.split_half(Z, rng)}
    r2 = []
    for d in Z.index:
        r2.append(corr(Z.loc[d].values, Gl.loc[d].values) ** 2)
    res['median_fitness_r2'] = float(np.nanmedian(r2))
    dpairs = pd.read_csv(RESULTS / 'decomposition' / 'cross_screen_pairs.csv')
    rep = dpairs.groupby('compound').rho_sel.mean()
    cls = pd.Series(np.where(rep >= REPRODUCIBLE, 'reproducible', np.where(rep < NONREPRODUCIBLE, 'non_reproducible', 'intermediate')),
                    index=rep.index)
    rows = []
    for fam in ['sel', 'raw']:
        P = pd.read_parquet(RESULTS / 'beataml_transfer' / f'pred_{fam}.parquet')
        for d in Z.index.intersection(P.index):
            p = P.loc[d].reindex(Z.columns).values
            sel, n = partial_corr(p, Z.loc[d].values, Gl.loc[d].values)
            rows.append({'compound': d, 'family': fam, 'n': n, 'raw': corr(p, Z.loc[d].values), 'sel': sel,
                         'class': cls.get(d, 'single_screen')})
    sc = pd.DataFrame(rows)
    sc.to_csv(out / 'scores.csv', index=False)
    s = sc[sc.family == 'sel']
    a = s[s['class'] == 'reproducible'].sel.dropna(); b = s[s['class'] == 'non_reproducible'].sel.dropna()
    h6 = {'drugs': len(a), 'median_sel': float(a.median()) if len(a) else None}
    if len(a) >= 8:
        h6['wilcoxon_p_one_sided'] = float(wilcoxon(a, alternative='greater').pvalue)
        h6['pass'] = bool(h6['wilcoxon_p_one_sided'] < 0.05)
    h7 = {'n_reproducible': len(a), 'n_non_reproducible': len(b),
          'median_reproducible': float(a.median()) if len(a) else None, 'median_non_reproducible': float(b.median()) if len(b) else None}
    if len(a) >= 3 and len(b) >= 3:
        h7['mwu_p_one_sided'] = float(mannwhitneyu(a, b, alternative='greater').pvalue)
        h7['pass'] = bool(h7['mwu_p_one_sided'] < 0.05)
    res['H6'], res['H7'] = h6, h7
    res['secondary'] = sc.groupby(['family', 'class']).agg(n=('sel', 'size'), raw=('raw', 'median'), sel=('sel', 'median')).reset_index().to_dict(orient='records')
    write_json(out / 'hypotheses.json', res)
    print(res)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        main()
