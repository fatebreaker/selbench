"""Tahoe-100M follow-up to 10_value_of_information.py, matched to its L1000 analysis: pooled
within-drug correlations, and drug-blind ridge maps from the measured response (top-dose pseudobulk
over the 2,000 most variable genes, 50 principal components) to each screen's selective residual.
Reference rows score basal expression of the same pairs and another screen's selective residual.
Output: results/value_of_information/tahoe_pooled.csv.
"""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from common import PROCESSED, RAW, RESULTS

DEC = RESULTS / 'decomposition'
OUT = RESULTS / 'value_of_information'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']


def pooled_within(x, y, groups, min_n=5):
    df = pd.DataFrame({'x': x, 'y': y, 'g': groups}).dropna()
    df = df[df.groupby('g').g.transform('size') >= min_n]
    xc = df.x - df.groupby('g').x.transform('mean'); yc = df.y - df.groupby('g').y.transform('mean')
    return float(np.corrcoef(xc, yc)[0, 1]), int(df.g.nunique()), len(df)


def main():
    A = RAW / 'arc'
    feats = pd.read_csv(OUT / 'tahoe_pair_features.csv.gz')
    feats['row'] = np.arange(len(feats))
    delta = np.load(A / 'delta_p.npy', mmap_mode='r')
    top = feats[(feats.dose == '5.0uM') & feats.compound.notna() & feats.depmap_id.notna()]
    top = top.drop_duplicates(['compound', 'depmap_id'])
    rng = np.random.default_rng(3)
    sample = np.sort(rng.choice(len(feats), 4000, replace=False))
    hvg = np.argsort(np.asarray(delta[sample]).var(axis=0))[::-1][:2000]
    X = np.asarray(delta[np.sort(top.row.values)][:, hvg], dtype=np.float32)
    order = np.argsort(np.argsort(top.row.values))
    X = X[order]
    Xp = PCA(50, random_state=0).fit_transform(X)
    expr = pd.read_parquet(PROCESSED / 'features' / 'expression_top.parquet')
    B = expr.reindex(top.depmap_id).values
    Bp = np.full((len(top), 50), np.nan)
    okb = np.isfinite(B).all(axis=1)
    Bp[okb] = PCA(50, random_state=0).fit_transform((B[okb] - B[okb].mean(0)) / (B[okb].std(0) + 1e-6))
    rows = []
    for E in SCREENS:
        R = pd.read_parquet(DEC / f'resid_{E}.parquet')
        y = np.array([R.at[c, d] if (c in R.index and d in R.columns) else np.nan for c, d in zip(top.compound, top.depmap_id)])
        groups = top.compound.values
        ok = np.isfinite(y)
        for name, F in [('measured_response', Xp), ('basal_expression', Bp)]:
            have = ok & np.isfinite(F).all(axis=1)
            idx = np.where(have)[0]
            pred = np.full(len(top), np.nan)
            for tr, te in GroupKFold(5).split(idx, groups=groups[idx]):
                m = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(F[idx[tr]], y[idx[tr]])
                pred[idx[te]] = m.predict(F[idx[te]])
            r, nd, npair = pooled_within(pred, y, groups)
            rows.append({'eval': E, 'features': name, 'pairs': npair, 'drugs': nd, 'ridge_map_vs_selective': r})
        r, nd, npair = pooled_within(np.where(ok, top.trs.values, np.nan), y, groups)
        rows.append({'eval': E, 'features': 'response_strength', 'pairs': npair, 'drugs': nd, 'ridge_map_vs_selective': r})
        r, nd, npair = pooled_within(np.where(ok, -top.log_abundance.values, np.nan), y, groups)
        rows.append({'eval': E, 'features': 'abundance_24h', 'pairs': npair, 'drugs': nd, 'ridge_map_vs_selective': r})
        for T in SCREENS:
            if T == E:
                continue
            RT = pd.read_parquet(DEC / f'resid_{T}.parquet')
            rep = np.array([RT.at[c, d] if (c in RT.index and d in RT.columns) else np.nan for c, d in zip(top.compound, top.depmap_id)])
            r, nd, npair = pooled_within(np.where(ok, rep, np.nan), y, groups)
            rows.append({'eval': E, 'features': f'replicate_{T}', 'pairs': npair, 'drugs': nd, 'ridge_map_vs_selective': r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'tahoe_pooled.csv', index=False)
    print(res.round(3).to_string())


if __name__ == '__main__':
    main()
