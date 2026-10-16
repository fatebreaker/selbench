"""Within-laboratory stability of the fitness axis: PRISM compounds screened in the original HTS002 run and
again in the MTS010 technical redo. Fitness axes are built from the same compounds and cell lines in each run.
Output: results/prism_rescreen/summary.json.
"""
import numpy as np
import pandas as pd

from common import RAW, RESULTS, rank_normal, write_json


def zmat(sub):
    M = -sub.pivot_table(index='short', columns='depmap_id', values='auc', aggfunc='mean')
    M = M.loc[M.notna().sum(axis=1) >= 30]
    return pd.DataFrame(np.vstack([rank_normal(r) for r in M.values]), index=M.index, columns=M.columns)


def split_half(Z, rng, n=200):
    out = []
    for _ in range(n):
        q = rng.permutation(Z.index); h = len(q) // 2
        a, b = Z.loc[q[:h]].mean(), Z.loc[q[h:]].mean(); m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1]; out.append(2 * r / (1 + r))
    return float(np.median(out))


def main():
    rng = np.random.default_rng(4)
    p = pd.read_csv(RAW / 'prism' / 'secondary-screen-dose-response-curve-parameters.csv',
                    usecols=['broad_id', 'depmap_id', 'screen_id', 'auc', 'passed_str_profiling'])
    p = p[p.passed_str_profiling.astype(str).str.upper() == 'TRUE']
    p['short'] = p.broad_id.str[:13]
    both = set(p[p.screen_id == 'MTS010'].short) & set(p[p.screen_id == 'HTS002'].short)
    A = zmat(p[(p.screen_id == 'HTS002') & p.short.isin(both)])
    B = zmat(p[(p.screen_id == 'MTS010') & p.short.isin(both)])
    d = A.index.intersection(B.index); c = A.columns.intersection(B.columns)
    A, B = A.loc[d, c], B.loc[d, c]
    fa, fb = A.mean(), B.mean()
    rel_a, rel_b = split_half(A, rng), split_half(B, rng)
    r = float(np.corrcoef(fa, fb)[0, 1])
    raw = []
    for k in d:
        x, y = A.loc[k].values, B.loc[k].values; m = np.isfinite(x) & np.isfinite(y)
        raw.append(np.corrcoef(x[m], y[m])[0, 1])
    res = {'compounds': len(d), 'cells': len(c), 'split_half_HTS002': rel_a, 'split_half_MTS010': rel_b,
           'fitness_axis_corr': r, 'disattenuated': r / np.sqrt(rel_a * rel_b),
           'median_per_drug_raw_agreement': float(np.nanmedian(raw))}
    write_json(RESULTS / 'prism_rescreen' / 'summary.json', res)
    print(res)


if __name__ == '__main__':
    main()
