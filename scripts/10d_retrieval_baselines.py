"""Can any profile available for unseen chemistry carry the selectivity information of the measured response?

On the L1000 reference panel, every pair (drug d, cell c) belongs to one of the three chemical-scaffold folds used to
retrain TranSiGen. For each pair, profiles are built only from measured responses in other folds:
  nearest_chemistry  the measured response of cell c to the most Tanimoto-similar training compound
  same_target        the mean measured response of cell c to training compounds sharing an annotated target with d
Measured responses with their top principal components removed test whether the signal is a generic
cell-death/stress program. Scoring follows 10_value_of_information.py (two-way-centred design, drug-blind ridge
map to each screen's selective residual), with measured responses re-scored on the same pairs.
Output: results/value_of_information/retrieval_baselines.csv.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from common import OWN_RAW, PROCESSED, RAW, RESULTS

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('voi', HERE / '10_value_of_information.py')
voi = importlib.util.module_from_spec(spec); spec.loader.exec_module(voi)
DEC = RESULTS / 'decomposition'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
PROC = RAW.parent / 'processed'


def fingerprints(smiles):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    out = {}
    for s in set(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out[s] = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    return out, DataStructs


def score(X, y_all, panel, rng):
    have = np.isfinite(y_all) & np.isfinite(X).all(axis=1)
    sub = panel[have]
    keep = ((sub.groupby('compound').compound.transform('size') >= 3)
            & (sub.groupby('depmap_id').depmap_id.transform('size') >= 10)).values
    idx = np.where(have)[0][keep]
    if len(idx) < 300:
        return None
    drugs, cells = panel.compound.values[idx], panel.depmap_id.values[idx]
    yt = voi.two_way(y_all[idx], drugs, cells)
    Xc = X[idx] - pd.DataFrame(X[idx]).groupby(cells).transform('mean').values
    pred = np.zeros(len(idx))
    for tr, te in GroupKFold(5).split(Xc, groups=drugs):
        pred[te] = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Xc[tr], yt[tr]).predict(Xc[te])
    pt = voi.two_way(pred, drugs, cells)
    r = float(np.corrcoef(pt, yt)[0, 1])
    return {'pairs': len(idx), 'drugs': int(len(set(drugs))), 'map_r': r, 'map_ci95': voi.drug_boot(pt, yt, drugs, rng)}


def main():
    rng = np.random.default_rng(23)
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
    meas = np.load(M / 'measured_response.npy')
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    targets = {d: set(str(t).split(';')) - {''} for d, t in comp.targets.dropna().items()}
    fps, DS = fingerprints(panel.smiles.tolist())
    near = np.full_like(meas, np.nan); same = np.full_like(meas, np.nan); sim_used = np.full(len(panel), np.nan)
    by_cell = {c: g for c, g in panel.groupby('depmap_id')}
    for i, row in panel.iterrows():
        g = by_cell[row.depmap_id]
        tr = g[(g.fold != row.fold) & (g.compound != row.compound)]
        if len(tr) == 0:
            continue
        fp = fps.get(row.smiles)
        if fp is not None:
            cand = [(DS.TanimotoSimilarity(fp, fps[s]), j) for j, s in zip(tr.index, tr.smiles) if s in fps]
            if cand:
                best = max(c[0] for c in cand)
                js = [j for s_, j in cand if s_ == best]
                near[i] = meas[js].mean(axis=0); sim_used[i] = best
        tg = targets.get(row.compound, set())
        if tg:
            js = [j for j, d in zip(tr.index, tr.compound) if targets.get(d, set()) & tg]
            if js:
                same[i] = meas[js].mean(axis=0)
    # measured responses with top principal components removed (computed over all pairs)
    mc = meas - meas.mean(axis=0)
    u, s, vt = np.linalg.svd(mc, full_matrices=False)
    variants = {'measured': meas, 'nearest_chemistry': near, 'same_target': same}
    for k in (1, 3, 10):
        variants[f'measured_minus_{k}pc'] = mc - (mc @ vt[:k].T) @ vt[:k]
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in SCREENS}
    rows = []
    for E in SCREENS:
        y = np.array([R[E].at[c, d] if (c in R[E].index and d in R[E].columns) else np.nan
                      for c, d in zip(panel.compound, panel.depmap_id)])
        for name, X in variants.items():
            res = score(X, y, panel, rng)
            if res:
                rows.append({'eval': E, 'features': name, **res})
        # measured re-scored on exactly the pairs covered by each retrieval baseline
        for name in ['nearest_chemistry', 'same_target']:
            Xm = np.where(np.isfinite(variants[name]).all(axis=1)[:, None], meas, np.nan)
            res = score(Xm, y, panel, rng)
            if res:
                rows.append({'eval': E, 'features': f'measured_on_{name}_pairs', **res})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'value_of_information' / 'retrieval_baselines.csv', index=False)
    print('median nearest-chemistry Tanimoto used:', float(np.nanmedian(sim_used)),
          'coverage near/same:', float(np.isfinite(near).all(1).mean()), float(np.isfinite(same).all(1).mean()))
    print(out.round(3).to_string())


if __name__ == '__main__':
    main()
