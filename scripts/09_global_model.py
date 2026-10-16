"""The evaluation trap: a conventional global drug-response model (gradient-boosted trees on drug
fingerprints/targets and cell omics, trained on raw AUC) scored under common split designs.

Splits: random pairs (mixed set), cell-blind (shared 5-fold cell partition), drug-blind (5 drug folds),
and cross-screen (cell-blind predictions from a model trained in screen T scored in screen E).
Baselines: drug mean, and drug mean + cell mean (an additive two-way model with no omics or
chemistry). Metrics: global Pearson on raw AUC; per-drug correlation with the within-drug rank-normal
profile (raw); per-drug partial correlation given the evaluation screen's leave-drug-out fitness
axis (selective). Output: results/global_model/.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

from common import PROCESSED, RESULTS, rank_normal, write_json
from modeling import corr, partial_corr

DEC = RESULTS / 'decomposition'
FEAT = PROCESSED / 'features'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
SEED = 11


def drug_features(compounds):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    fps = np.zeros((len(compounds), 256), np.float32)
    for i, d in enumerate(compounds):
        smi = comp.smiles.get(d)
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        if m is not None:
            arr = np.zeros((256,), np.float32)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, 256), arr)
            fps[i] = arr
    tg = comp.targets.reindex(compounds).fillna('').str.split(';')
    counts = pd.Series([t for ts in tg for t in ts if t]).value_counts()
    top = counts.index[:100]
    tmat = np.array([[t in set(ts) for t in top] for ts in tg], np.float32)
    return pd.DataFrame(np.hstack([fps, tmat]), index=compounds)


def cell_features(cells):
    expr = pd.read_parquet(FEAT / 'expression_top.parquet')
    mut = pd.read_parquet(FEAT / 'mutation.parquet')
    lin = pd.read_parquet(FEAT / 'lineage.parquet')
    common = [c for c in cells if c in expr.index and c in mut.index and c in lin.index]
    e = expr.loc[common]
    pcs = PCA(100, random_state=SEED).fit_transform(((e - e.mean()) / e.std().replace(0, 1)).fillna(0))
    topmut = mut.loc[common].mean().sort_values(ascending=False).index[:50]
    X = np.hstack([pcs, mut.loc[common, topmut].values, lin.loc[common].values])
    return pd.DataFrame(X.astype(np.float32), index=common)


def fit_predict(Xtr, ytr, Xte):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=SEED)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def score(df, screen, Zs, Gs, label):
    """df: compound, depmap_id, auc (observed raw), pred_* columns (higher pred = higher AUC = resistant)."""
    rows = []
    for col in [c for c in df.columns if c.startswith('pred_')]:
        ok = df[col].notna()
        g_r = corr(df.loc[ok, col].values, df.loc[ok, 'auc'].values)
        per = []
        for d, g in df[ok].groupby('compound'):
            if d not in Zs.index or len(g) < 20:
                continue
            z = Zs.loc[d].reindex(g.depmap_id).values
            gg = Gs.loc[d].reindex(g.depmap_id).values
            p = -g[col].values                      # predicted sensitivity
            raw = corr(p, z)
            sel, _ = partial_corr(p, z, gg)
            per.append((raw, sel))
        per = np.array(per, dtype=float)
        rows.append({'screen': screen, 'split': label, 'model': col[5:], 'global_r': g_r, 'drugs': len(per),
                     'per_drug_raw': float(np.nanmedian(per[:, 0])) if len(per) else np.nan,
                     'per_drug_sel': float(np.nanmedian(per[:, 1])) if len(per) else np.nan})
    return rows


def loo_fitness(Z):
    vals = Z.values; ok = np.isfinite(vals)
    tot = np.nansum(vals, axis=0); cnt = ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = (tot[None, :] - np.where(ok, vals, 0)) / (cnt[None, :] - ok)
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--screens', nargs='+', default=SCREENS)
    args = ap.parse_args()
    out = RESULTS / 'global_model'
    out.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in SCREENS}
    G = {s: loo_fitness(Z[s]) for s in SCREENS}
    part = pd.read_csv(RESULTS / 'models' / 'cell_folds.csv').set_index('depmap_id').fold
    cells_all = sorted(set(long.depmap_id))
    CF = cell_features(cells_all)
    comps = sorted(set(long.compound))
    DF = drug_features(comps)
    rng = np.random.default_rng(SEED)
    all_rows, oof_cell = [], {}
    for s in args.screens:
        d = long[(long.screen == s) & long.compound.isin(Z[s].index) & long.depmap_id.isin(Z[s].columns)
                 & long.depmap_id.isin(CF.index) & long.depmap_id.isin(part.index)].copy().reset_index(drop=True)
        X = np.hstack([DF.loc[d.compound].values, CF.loc[d.depmap_id].values])
        y = d.auc.values
        # --- random pair split ---
        fold = rng.integers(0, 5, len(d))
        pred = np.zeros(len(d)); add = np.zeros(len(d)); dm = np.zeros(len(d))
        for f in range(5):
            tr, te = fold != f, fold == f
            pred[te] = fit_predict(X[tr], y[tr], X[te])
            dtr = d[tr]
            mu = dtr.auc.mean()
            dmean = dtr.groupby('compound').auc.mean(); cdev = (dtr.auc - dtr.compound.map(dmean)).groupby(dtr.depmap_id).mean()
            dm[te] = d.loc[te, 'compound'].map(dmean).fillna(mu).values
            add[te] = dm[te] + d.loc[te, 'depmap_id'].map(cdev).fillna(0).values
        d['pred_global_gbt'], d['pred_drug_mean'], d['pred_drug_plus_cell_mean'] = pred, dm, add
        all_rows += score(d, s, Z[s], G[s], 'random_pairs')
        # --- cell-blind split (shared partition) ---
        cf = d.depmap_id.map(part).values
        predc = np.zeros(len(d)); dmc = np.zeros(len(d))
        for f in range(5):
            tr, te = cf != f, cf == f
            predc[te] = fit_predict(X[tr], y[tr], X[te])
            dmean = d[tr].groupby('compound').auc.mean()
            dmc[te] = d.loc[te, 'compound'].map(dmean).fillna(d[tr].auc.mean()).values
        dc = d[['compound', 'depmap_id', 'auc']].copy()
        dc['pred_global_gbt'], dc['pred_drug_mean'] = predc, dmc
        all_rows += score(dc, s, Z[s], G[s], 'cell_blind')
        oof_cell[s] = dc
        # --- drug-blind split ---
        drugs = np.array(sorted(d.compound.unique())); dfold = pd.Series(rng.permutation(np.arange(len(drugs)) % 5), index=drugs)
        df_ = d.compound.map(dfold).values
        predd = np.zeros(len(d))
        for f in range(5):
            tr, te = df_ != f, df_ == f
            predd[te] = fit_predict(X[tr], y[tr], X[te])
        dd = d[['compound', 'depmap_id', 'auc']].copy(); dd['pred_global_gbt'] = predd
        all_rows += score(dd, s, Z[s], G[s], 'drug_blind')
        print(s, 'done', flush=True)
    # --- cross-screen: cell-blind predictions of T scored on E for shared drug/cell pairs ---
    for T in args.screens:
        for E in args.screens:
            if T == E:
                continue
            pT = oof_cell[T][['compound', 'depmap_id', 'pred_global_gbt']]
            obs = long[(long.screen == E)][['compound', 'depmap_id', 'auc']]
            j = obs.merge(pT, on=['compound', 'depmap_id'])
            if j.compound.nunique() < 10:
                continue
            for r in score(j, E, Z[E], G[E], f'cross_from_{T}'):
                r['train'] = T
                all_rows.append(r)
    res = pd.DataFrame(all_rows)
    res.to_csv(out / 'scores.csv', index=False)
    write_json(out / 'summary.json', {'rows': res.to_dict(orient='records')})
    print(res.round(3).to_string())


if __name__ == '__main__':
    main()
