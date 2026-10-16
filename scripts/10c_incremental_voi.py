"""Incremental value of the measured L1000 response over a cell-blind basal-omics model, on pairs that
have both (two-way-centred design of 10_value_of_information.py, part B). For each evaluation screen E
and basal-model training screen T != E: correlation of the basal model with the selective residual,
correlation of the measured-response map, their partial correlations, and the correlation of their
equal-weight combination. Output: results/value_of_information/l1000_incremental.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from common import OWN_RAW, PROCESSED, RAW, RESULTS

DEC = RESULTS / 'decomposition'
MOD = RESULTS / 'models'
OUT = RESULTS / 'value_of_information'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
PROC = RAW.parent / 'processed'


def two_way(v, drugs, cells, iters=20):
    df = pd.DataFrame({'v': v, 'd': drugs, 'c': cells})
    for _ in range(iters):
        df['v'] = df.v - df.groupby('d').v.transform('mean')
        df['v'] = df.v - df.groupby('c').v.transform('mean')
    return df.v.values


def partial(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    M = PROC / 'measured_lincs_reference'
    panel = pd.read_csv(M / 'panel.csv')
    portal = pd.read_csv(OWN_RAW / 'depmap24q4' / 'PortalCompounds.csv')
    merges = pd.read_csv(PROCESSED / 'compound_merges.csv').set_index('merged').into
    brd = {}
    for r in portal.itertuples():
        for x in str(r.SampleIDs).split(';'):
            ds, _, i = x.strip().partition(':')
            if ds == 'BRD':
                brd.setdefault(i[:13], merges.get(r.CompoundID, r.CompoundID))
    panel['compound'] = panel.broad_id.str[:13].map(brd)
    X = np.load(M / 'measured_response.npy')
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in SCREENS}
    rows = []
    for E in SCREENS:
        y = np.array([R[E].at[c, d] if (c in R[E].index and d in R[E].columns) else np.nan
                      for c, d in zip(panel.compound, panel.depmap_id)])
        for T in SCREENS:
            if T == E:
                continue
            P = pd.read_parquet(MOD / f'pred_ridge_sel_{T}.parquet')
            b = np.array([P.at[c, d] if (c in P.index and d in P.columns) else np.nan
                          for c, d in zip(panel.compound, panel.depmap_id)])
            have = np.isfinite(y) & np.isfinite(b)
            sub = panel[have]
            keep = ((sub.groupby('compound').compound.transform('size') >= 3)
                    & (sub.groupby('depmap_id').depmap_id.transform('size') >= 10)).values
            idx = np.where(have)[0][keep]
            if len(idx) < 300:
                continue
            drugs, cells = panel.compound.values[idx], panel.depmap_id.values[idx]
            yt, bt = two_way(y[idx], drugs, cells), two_way(b[idx], drugs, cells)
            Xc = X[idx] - pd.DataFrame(X[idx]).groupby(cells).transform('mean').values
            pred = np.zeros(len(idx))
            for tr, te in GroupKFold(5).split(Xc, groups=drugs):
                pred[te] = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Xc[tr], yt[tr]).predict(Xc[te])
            mt = two_way(pred, drugs, cells)
            zb, zm = (bt - bt.mean()) / bt.std(), (mt - mt.mean()) / mt.std()
            rows.append({'eval': E, 'basal_train': T, 'pairs': len(idx), 'drugs': int(len(set(drugs))),
                         'basal_r': float(np.corrcoef(bt, yt)[0, 1]), 'measured_r': float(np.corrcoef(mt, yt)[0, 1]),
                         'measured_given_basal': partial(mt, yt, bt), 'basal_given_measured': partial(bt, yt, mt),
                         'combined_r': float(np.corrcoef(zb + zm, yt)[0, 1]),
                         'basal_measured_r': float(np.corrcoef(bt, mt)[0, 1])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'l1000_incremental.csv', index=False)
    print(res.round(3).to_string())


if __name__ == '__main__':
    main()
