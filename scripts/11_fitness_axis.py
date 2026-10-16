"""Descriptive characterisation of the drug-agnostic fitness axis across all six screens (four
discovery, two confirmation; run after the confirmatory analysis).

For each screen: split-half reliability (random drug halves, Spearman-Brown), association with
CRISPR-inferred growth rate, suspension growth and lineage. For each pair of screens: correlation of
the all-drug axes over shared cell lines (bootstrap CI over cells), disattenuated correlation, and
the same restricted to drugs measured in both screens. Output: results/fitness_axis/.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import PROCESSED, RESULTS, load_models, rank_normal, write_json

DEC = RESULTS / 'decomposition'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2', 'gCSI', 'NCI60']
MIN_CELLS, N_SPLITS, N_BOOT = 30, 200, 1000


def confirmation_z(name):
    long = pd.read_csv(PROCESSED / 'confirmation' / f'{name}_long.csv.gz')
    M = long.pivot_table(index='compound', columns='depmap_id', values='sens', aggfunc='mean')
    M = M.loc[M.notna().sum(axis=1) >= MIN_CELLS]
    M = M.loc[[d for d in M.index if M.loc[d].dropna().nunique() >= 10]]
    return pd.DataFrame(np.vstack([rank_normal(r) for r in M.values]), index=M.index, columns=M.columns)


def split_half(Z, rng):
    drugs = np.array(Z.index); h = len(drugs) // 2; vals = []
    for _ in range(N_SPLITS):
        p = rng.permutation(drugs)
        a, b = Z.loc[p[:h]].mean(), Z.loc[p[h:]].mean()
        m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1]
        vals.append(2 * r / (1 + r))
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    out = RESULTS / 'fitness_axis'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in SCREENS[:4]}
    Z['gCSI'] = confirmation_z('gCSI')
    Z['NCI60'] = confirmation_z('NCI60')
    fit = {s: Z[s].mean(axis=0) for s in SCREENS}
    pd.DataFrame(fit).to_csv(out / 'fitness_axes.csv')
    growth = pd.read_csv(PROCESSED / 'features' / 'growth_rate.csv', index_col=0).growth_rate.dropna()
    models = load_models().set_index('ModelID')
    per_screen = {}
    for s in SCREENS:
        f = fit[s]
        rel = split_half(Z[s], rng)
        g = f.index.intersection(growth.index)
        susp = models.GrowthPattern.reindex(f.index).fillna('').str.contains('Suspension')
        lin = models.OncotreeLineage.reindex(f.index).fillna('Unknown')
        lin_means = f.groupby(lin).transform('mean')
        per_screen[s] = {'drugs': int(Z[s].shape[0]), 'cells': int(Z[s].shape[1]), 'split_half': rel,
                         'growth_rate_spearman': float(spearmanr(f[g], growth[g]).correlation), 'growth_n': len(g),
                         'suspension_minus_adherent': float(f[susp].mean() - f[~susp].mean()), 'suspension_n': int(susp.sum()),
                         'lineage_r2': float(np.corrcoef(f, lin_means)[0, 1] ** 2)}
    pairs = []
    for a, b in itertools.combinations(SCREENS, 2):
        cells = fit[a].index.intersection(fit[b].index)
        if len(cells) < 20:
            continue
        x, y = fit[a][cells].values, fit[b][cells].values
        r = float(np.corrcoef(x, y)[0, 1])
        boots = [np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, len(x), len(x)) for _ in range(N_BOOT))]
        common = Z[a].index.intersection(Z[b].index)
        rc = np.nan
        if len(common) >= 10:
            fa, fb = Z[a].loc[common, cells].mean(), Z[b].loc[common, cells].mean()
            m = fa.notna() & fb.notna()
            rc = float(np.corrcoef(fa[m], fb[m])[0, 1])
        rel = np.sqrt(per_screen[a]['split_half'][0] * per_screen[b]['split_half'][0])
        pairs.append({'a': a, 'b': b, 'cells': len(cells), 'corr': r, 'ci95': [float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5))], 'disattenuated': r / rel, 'common_drugs': len(common),
                      'corr_common_drugs': rc})
    pd.DataFrame(pairs).to_csv(out / 'pairs.csv', index=False)
    write_json(out / 'summary.json', {'per_screen': per_screen, 'pairs': pairs})
    print(pd.DataFrame(per_screen).T.to_string())
    print(pd.DataFrame(pairs).round(3).to_string())


if __name__ == '__main__':
    main()
