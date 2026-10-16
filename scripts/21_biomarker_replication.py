"""Do drug-biomarker associations replicate better across screens after removing the fitness axis?

For every drug measured in >=2 discovery screens, Spearman associations are computed in each screen
between the drug's response (raw within-drug profile z, or selective residual r) and basal features:
mutations (genes mutated in >=3% of screened lines), the 1,000 most variable genes' expression, and
lineage indicators (lineages with >=10 lines). Discoveries are BH-significant (q<0.05 within screen,
response type and feature type). A discovery replicates in another screen if the association has the
same sign with one-sided P<0.05 there; a sensitivity analysis uses only cell lines absent from the
discovery screen. Equal-size comparisons use each screen's top-K associations.
Output: results/biomarkers/.
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as tdist

from common import PROCESSED, RESULTS, load_models, write_json

DEC = RESULTS / 'decomposition'
FEAT = PROCESSED / 'features'
DISC = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
MIN_N, MIN_MUT = 30, 3
G = {}


def init(payload):
    G.update(payload)


def assoc(y, X, binary):
    """Spearman correlation of y with each column of X (rows aligned, no missing); returns r, p, n."""
    n = len(y)
    yr = rankdata(y); yr = (yr - yr.mean()) / yr.std()
    Xr = np.apply_along_axis(rankdata, 0, X)
    Xr = (Xr - Xr.mean(0)) / np.where(Xr.std(0) > 0, Xr.std(0), np.nan)
    r = (Xr * yr[:, None]).mean(0)
    if binary is not None:
        r[(X.sum(0) < MIN_MUT) | ((1 - X).sum(0) < MIN_MUT)] = np.nan
    tt = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    p = 2 * tdist.sf(np.abs(tt), n - 2)
    return r, p


def drug_task(args):
    screen, d, kind = args
    M = G[kind][screen]
    y_all = M.loc[d]
    cells = y_all.dropna().index.intersection(G['X'].index)
    if len(cells) < MIN_N:
        return []
    y = y_all[cells].values
    out = []
    for ftype in ['mutation', 'expression', 'lineage']:
        cols = G['cols'][ftype]
        X = G['X'].loc[cells, cols].values
        r, p = assoc(y, X, binary=(ftype != 'expression'))
        for f, rr, pp in zip(cols, r, p):
            if np.isfinite(rr):
                out.append((screen, d, kind, ftype, f, float(rr), float(pp), len(cells)))
    return out


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def main():
    out = RESULTS / 'biomarkers'; out.mkdir(parents=True, exist_ok=True)
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in DISC}
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in DISC}
    multi = [d for d in set().union(*[set(z.index) for z in Z.values()]) if sum(d in Z[s].index for s in DISC) >= 2]
    screened = sorted(set().union(*[set(z.columns) for z in Z.values()]))
    mut = pd.read_parquet(FEAT / 'mutation.parquet')
    expr = pd.read_parquet(FEAT / 'expression_top.parquet')
    lin = pd.read_parquet(FEAT / 'lineage.parquet')
    cells = [c for c in screened if c in mut.index and c in expr.index and c in lin.index]
    mfreq = mut.loc[cells].mean()
    mcols = list(mfreq[mfreq >= 0.03].index)
    ecols = list(expr.columns[:1000])
    lcount = lin.loc[cells].sum()
    lcols = list(lcount[lcount >= 10].index)
    X = pd.concat([mut.loc[cells, mcols].add_prefix('MUT:'), expr.loc[cells, ecols].add_prefix('EXP:'),
                   lin.loc[cells, lcols].add_prefix('LIN:')], axis=1).astype(float)
    cols = {'mutation': ['MUT:' + c for c in mcols], 'expression': ['EXP:' + c for c in ecols], 'lineage': ['LIN:' + c for c in lcols]}
    tasks = [(s, d, k) for s in DISC for d in multi if d in Z[s].index for k in ['raw', 'sel']]
    rows = []
    with ProcessPoolExecutor(32, initializer=init, initargs=({'raw': Z, 'sel': R, 'X': X, 'cols': cols},)) as ex:
        for res in ex.map(drug_task, tasks, chunksize=8):
            rows.extend(res)
    A = pd.DataFrame(rows, columns=['screen', 'compound', 'kind', 'ftype', 'feature', 'r', 'p', 'n'])
    A['q'] = np.nan
    for key, g in A.groupby(['screen', 'kind', 'ftype']):
        A.loc[g.index, 'q'] = bh(g.p.values)
    A.to_parquet(out / 'associations.parquet')

    # replication on all cell lines of the other screen
    idx = A.set_index(['compound', 'kind', 'feature', 'screen'])[['r', 'p']].sort_index()
    rep_rows = []
    for (screen, kind, ftype), g in A[A.q < 0.05].groupby(['screen', 'kind', 'ftype']):
        for r in g.itertuples():
            for other in DISC:
                if other == screen:
                    continue
                key = (r.compound, kind, r.feature, other)
                if key not in idx.index:
                    continue
                ro, po = idx.loc[key]
                rep_rows.append({'screen': screen, 'other': other, 'kind': kind, 'ftype': ftype, 'compound': r.compound,
                                 'feature': r.feature, 'r': r.r, 'r_other': ro,
                                 'replicated': bool(np.sign(ro) == np.sign(r.r) and po / 2 < 0.05)})
    REP = pd.DataFrame(rep_rows)
    REP.to_csv(out / 'replication.csv.gz', index=False)

    summ = []
    for (kind, ftype), g in REP.groupby(['kind', 'ftype']):
        disc = A[(A.kind == kind) & (A.ftype == ftype) & (A.q < 0.05)]
        summ.append({'kind': kind, 'ftype': ftype, 'discoveries': len(disc), 'tested_pairs': len(g),
                     'replication_rate': float(g.replicated.mean()), 'replicated': int(g.replicated.sum())})
    for kind, g in REP.groupby('kind'):
        disc = A[(A.kind == kind) & (A.q < 0.05)]
        summ.append({'kind': kind, 'ftype': 'all', 'discoveries': len(disc), 'tested_pairs': len(g),
                     'replication_rate': float(g.replicated.mean()), 'replicated': int(g.replicated.sum())})
    summ = pd.DataFrame(summ)

    # equal-size comparison: top-K strongest associations per screen and response type
    topk = []
    for K in [200, 1000, 5000]:
        for kind in ['raw', 'sel']:
            hits = []
            for screen in DISC:
                g = A[(A.screen == screen) & (A.kind == kind)].nsmallest(K, 'p')
                for r in g.itertuples():
                    for other in DISC:
                        key = (r.compound, kind, r.feature, other)
                        if other == screen or key not in idx.index:
                            continue
                        ro, po = idx.loc[key]
                        hits.append(bool(np.sign(ro) == np.sign(r.r) and po / 2 < 0.05))
            topk.append({'K': K, 'kind': kind, 'tested_pairs': len(hits), 'replication_rate': float(np.mean(hits))})
    topk = pd.DataFrame(topk)

    # composition of discoveries
    comp = A[A.q < 0.05].groupby(['kind', 'ftype']).size().unstack(fill_value=0)
    write_json(out / 'summary.json', {'drugs': len(multi), 'features': {k: len(v) for k, v in cols.items()},
                                      'tests': len(A), 'replication': summ.to_dict(orient='records'),
                                      'topk': topk.to_dict(orient='records'), 'composition': comp.to_dict()})
    print(summ.round(3).to_string()); print(topk.round(3).to_string()); print(comp)


if __name__ == '__main__':
    main()
