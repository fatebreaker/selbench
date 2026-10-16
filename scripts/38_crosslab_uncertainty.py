"""Drug-bootstrap uncertainty for the cross-laboratory decomposition of the leaderboard models.

For each published model and evaluation screen (results/dreval_crosslab_leaderboard/per_drug.csv.gz), on drugs scored
in both laboratories: share of the loss in mean per-drug correlation carried by the fitness-mediated term,
(sum mediated_within - sum mediated_cross) / (sum raw_within - sum raw_cross), and the retained fractions of the
fitness-mediated and selective terms, with 2,000 drug-bootstrap draws. A pooled estimate sums over the published
models (drugs resampled jointly across models).
Output: results/dreval_crosslab_leaderboard/uncertainty.csv.
"""
import numpy as np
import pandas as pd

from common import RESULTS

B = 2000
SRC = RESULTS / 'dreval_crosslab_leaderboard'


def stats(w):
    dr = w.raw_within.sum() - w.raw_cross.sum()
    return {'share': (w.mediated_within.sum() - w.mediated_cross.sum()) / dr if abs(dr) > 1e-9 else np.nan,
            'mediated_retained': w.mediated_cross.sum() / w.mediated_within.sum(),
            'selterm_retained': w.selective_term_cross.sum() / w.selective_term_within.sum()
            if w.selective_term_within.sum() > 0 else np.nan,
            'raw_loss': dr / len(w)}


def main():
    rng = np.random.default_rng(20261014)
    per = pd.read_csv(SRC / 'per_drug.csv.gz', dtype={'drug': str})
    w = per.pivot_table(index=['model', 'class', 'target', 'drug'], columns='evaluated_in',
                        values=['raw', 'mediated', 'selective_term']).dropna()
    w.columns = [f'{v}_{e}' for v, e in w.columns]
    w = w.reset_index()
    pub = w[w['class'].isin(['classical', 'neural/factorization']) & (w.model != 'SVR')]
    rows = []
    for t in ['GDSC1', 'GDSC2', 'CTRPv1', 'CCLE']:
        d = pub[pub.target == t]
        drugs = d.drug.unique()
        groups = {'pooled published models': d}
        groups.update({m: g for m, g in d.groupby('model')})
        for name, g in groups.items():
            point = stats(g)
            by_drug = {k: v for k, v in g.groupby('drug')}
            boot = []
            for _ in range(B):
                pick = rng.choice(drugs, len(drugs))
                boot.append(stats(pd.concat([by_drug[x] for x in pick if x in by_drug])))
            boot = pd.DataFrame(boot)
            row = {'target': t, 'model': name, 'drugs': int(g.drug.nunique())}
            for k, v in point.items():
                lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
                row.update({k: v, f'{k}_lo': lo, f'{k}_hi': hi})
            rows.append(row)
        print(t, 'done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(SRC / 'uncertainty.csv', index=False)
    pd.set_option('display.width', 220)
    print(out[out.model == 'pooled published models'].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
