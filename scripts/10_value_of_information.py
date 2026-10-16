"""Value of information of drug-induced transcriptional responses for drug-specific selectivity.

Part A (Tahoe-100M pseudobulk, measured; 50 cell lines x 379 drugs x 3 doses): per (drug, dose, cell)
features are the transcriptional response strength (L2 norm of the treated-minus-control log1p
pseudobulk over the 2,000 most variable genes), its agreement with the drug's leave-cell-out mean
response, and the 24 h relative cell abundance (treated/DMSO cell counts) where counts exist. Within
each drug they are correlated with every discovery screen's raw profile z_E and selective residual
r_E. Cell-blind basal ridge predictions (trained in another screen) and a replicate measurement in
another screen are scored on the same cells.

Part B (L1000 reference panel, 8,058 drug-cell pairs at 10 uM): measured responses against
TranSiGen generations from the released checkpoint (training-exposed) and from retrained generators
that held out each pair's chemistry or chemistry and cell, XPert generations from its released checkpoint, plus
basal expression alone. Features:
response strength, and drug-blind ridge maps from the cell-centred 978-gene response to r_E. Both
features and targets are two-way centred (drug and cell main effects removed); intervals resample
drugs.
Outputs: results/value_of_information/.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from common import OWN_RAW, PROCESSED, RAW, RESULTS, inchikey25, norm_name, write_json

DEC = RESULTS / 'decomposition'
MOD = RESULTS / 'models'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
PROC = RAW.parent / 'processed'
OUT = RESULTS / 'value_of_information'
MIN_CELLS = 15


def compound_lookup():
    comp = pd.read_csv(PROCESSED / 'compounds.csv')
    by_key, by_name = {}, {}
    for r in comp.itertuples():
        k = inchikey25(r.smiles) if isinstance(r.smiles, str) else None
        if k:
            by_key.setdefault(k, set()).add(r.compound)
        for n in str(r.all_names).split('|') + [r.name]:
            by_name.setdefault(norm_name(n), set()).add(r.compound)
    return by_key, by_name


def resolve(smiles, name, by_key, by_name):
    k = inchikey25(smiles) if isinstance(smiles, str) else None
    c = by_key.get(k, set()) if k else set()
    if len(c) == 1:
        return next(iter(c))
    c = by_name.get(norm_name(name), set())
    return next(iter(c)) if len(c) == 1 else None


def loo_fitness(Z):
    vals = Z.values; ok = np.isfinite(vals)
    tot = np.nansum(vals, axis=0); cnt = ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = (tot[None, :] - np.where(ok, vals, 0)) / (cnt[None, :] - ok)
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def spear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_CELLS or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return spearmanr(a[ok], b[ok]).correlation


def part_a(Z, R):
    A = RAW / 'arc'
    meta = json.loads((A / 'meta.json').read_text())
    ctx = np.load(A / 'ctx_ids.npy'); pert = np.load(A / 'pert_ids.npy')
    delta = np.load(A / 'delta_p.npy', mmap_mode='r')
    id2ctx = {v: k for k, v in meta['context_to_id'].items()}
    id2pert = {v: k for k, v in meta['pert_to_id'].items()}
    cells = pd.read_parquet(RAW / 'tahoe' / 'cell_line_metadata.parquet').drop_duplicates('Cell_ID_Cellosaur')
    cvcl2ach = cells.set_index('Cell_ID_Cellosaur').Cell_ID_DepMap
    drugs = pd.read_parquet(RAW / 'tahoe' / 'drug_metadata.parquet')
    by_key, by_name = compound_lookup()
    drug2comp = {r.drug: resolve(r.canonical_smiles, r.drug, by_key, by_name) for r in drugs.itertuples()}
    rng = np.random.default_rng(3)
    sample = np.sort(rng.choice(len(ctx), 4000, replace=False))
    var = np.asarray(delta[sample]).var(axis=0)
    hvg = np.argsort(var)[::-1][:2000]
    rows = []
    for i in range(len(ctx)):
        name = id2pert[int(pert[i])]
        drug, _, dose = name.rpartition('_')
        rows.append({'row': i, 'cvcl': id2ctx[int(ctx[i])], 'drug': drug, 'dose': dose})
    info = pd.DataFrame(rows)
    info['depmap_id'] = info.cvcl.map(cvcl2ach)
    info['compound'] = info.drug.map(drug2comp)
    X = np.asarray(delta[:, hvg], dtype=np.float32)
    info['trs'] = np.linalg.norm(X, axis=1)
    # agreement with the drug's leave-cell-out mean response (same dose)
    agree = np.full(len(info), np.nan)
    for (_, _), g in info.groupby(['drug', 'dose']):
        idx = g.row.values
        tot = X[idx].sum(axis=0)
        for j in idx:
            m = (tot - X[j]) / (len(idx) - 1)
            agree[j] = np.corrcoef(X[j], m)[0, 1]
    info['agreement'] = agree
    # 24 h relative abundance from condition counts (subset of cell lines)
    ab = []
    for f in sorted((A / 'condition_counts').glob('*.csv')):
        c = pd.read_csv(f)
        c['cvcl'] = f.stem
        c['drug'] = c.target.str.extract(r"\[\('(.*)', ([0-9.]+), 'uM'\)\]")[0]
        c['dose_val'] = c.target.str.extract(r"\[\('(.*)', ([0-9.]+), 'uM'\)\]")[1].astype(float)
        ab.append(c)
    ab = pd.concat(ab, ignore_index=True)
    ab['dose'] = ab.dose_val.map(lambda v: f'{v}uM')
    ab['log_abundance'] = np.log2((ab.target_ncells + 1) / (ab.ref_ncells + 1))
    info = info.merge(ab[['cvcl', 'drug', 'dose', 'log_abundance']], on=['cvcl', 'drug', 'dose'], how='left')
    info.drop(columns=['row']).to_csv(OUT / 'tahoe_pair_features.csv.gz', index=False)

    G = {s: loo_fitness(Z[s]) for s in SCREENS}
    preds = {s: pd.read_parquet(MOD / f'pred_ridge_sel_{s}.parquet') for s in SCREENS}
    out = []
    for (comp, dose), g in info.dropna(subset=['compound', 'depmap_id']).groupby(['compound', 'dose']):
        g = g.drop_duplicates('depmap_id').set_index('depmap_id')
        for E in SCREENS:
            if comp not in Z[E].index:
                continue
            cells_e = g.index.intersection(Z[E].columns)
            if len(cells_e) < MIN_CELLS:
                continue
            z = Z[E].loc[comp, cells_e].values; r = R[E].loc[comp, cells_e].values
            rec = {'compound': comp, 'dose': dose, 'eval': E, 'n': len(cells_e)}
            # selective version of each feature: residualise on E's leave-drug-out fitness axis
            gE = G[E].loc[comp, cells_e].values
            for f in ['trs', 'agreement', 'log_abundance']:
                v = g.loc[cells_e, f].values.astype(float)
                sign = -1.0 if f == 'log_abundance' else 1.0       # fewer surviving cells = more sensitive
                rec[f'{f}_raw'] = spear(sign * v, z)
                ok = np.isfinite(v) & np.isfinite(gE)
                if ok.sum() >= MIN_CELLS:
                    Xg = np.c_[np.ones(ok.sum()), gE[ok]]
                    vr = np.full(len(v), np.nan); vr[ok] = v[ok] - Xg @ np.linalg.lstsq(Xg, v[ok], rcond=None)[0]
                    rec[f'{f}_sel'] = spear(sign * vr, r)
            for T in SCREENS:
                if T == E or comp not in Z[T].index:
                    continue
                rec[f'replicate_{T}_sel'] = spear(R[T].loc[comp].reindex(cells_e).values, r)
                if comp in preds[T].index:
                    rec[f'ridge_{T}_sel'] = spear(preds[T].loc[comp].reindex(cells_e).values, r)
            out.append(rec)
    out = pd.DataFrame(out)
    out.to_csv(OUT / 'tahoe_within_drug.csv', index=False)
    return out, int(info.compound.notna().groupby(info.drug).any().sum())


def pooled_within(x, y, groups):
    df = pd.DataFrame({'x': x, 'y': y, 'g': groups}).dropna()
    df = df[df.groupby('g').g.transform('size') >= 3]
    xc = df.x - df.groupby('g').x.transform('mean'); yc = df.y - df.groupby('g').y.transform('mean')
    return float(np.corrcoef(xc, yc)[0, 1]), int(df.g.nunique()), len(df)


def two_way(v, drugs, cells, iters=20):
    """Remove drug and cell main effects by alternating centring."""
    df = pd.DataFrame({'v': v, 'd': drugs, 'c': cells})
    for _ in range(iters):
        df['v'] = df.v - df.groupby('d').v.transform('mean')
        df['v'] = df.v - df.groupby('c').v.transform('mean')
    return df.v.values


def drug_boot(x, y, drugs, rng, n=1000):
    ud = np.unique(drugs); pos = {d: np.where(drugs == d)[0] for d in ud}
    out = []
    for _ in range(n):
        idx = np.concatenate([pos[d] for d in rng.choice(ud, len(ud))])
        out.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def part_b(Z, R):
    """Two-way-centred design: pairs from drugs with >=3 pairs and cell lines with >=10 pairs; features
    and targets are both freed of drug and cell main effects, so only drug-specific variation counts."""
    M = PROC / 'measured_lincs_reference'
    S = PROC / 'source_holdout_stable_predictions'
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
    resp = {'measured': np.load(M / 'measured_response.npy'),
            'released_generator': np.load(S / 'released' / 'global_response.npy')}
    for kind in ['chemical_only', 'joint']:
        arr = np.full_like(resp['measured'], np.nan)
        for b in sorted(S.glob(f'{kind}_*')):
            rows = np.load(b / 'functional_test_rows.npy')
            arr[rows] = np.load(b / 'global_response.npy')[rows]
        resp[f'heldout_{kind}_generator'] = arr
    # XPert generations from its released checkpoint (also LINCS-trained), aligned to the same panel rows
    X = PROC / 'xpert_reference'
    xp_rows = pd.read_csv(X / 'panel.csv').row.to_numpy(int)
    keep = np.load(X / 'common_xpert_panel_rows.npy')
    xp = np.full_like(resp['measured'], np.nan)
    xp[xp_rows[keep]] = np.load(X / 'generated_response.npy')[keep]
    resp['xpert_released_generator'] = xp
    resp['basal_only'] = np.load(M / 'global_basal.npy')
    rng = np.random.default_rng(17)
    res = []
    for E in SCREENS:
        y_all = np.array([R[E].at[c, d] if (c in R[E].index and d in R[E].columns) else np.nan
                          for c, d in zip(panel.compound, panel.depmap_id)])
        refs = [(f'replicate_{T}', None) for T in SCREENS if T != E] + \
               [(f'basal_model_{T}', None) for T in SCREENS if T != E]
        for name, X in list(resp.items()) + refs:
            if X is None:
                kind, T = name.rsplit('_', 1)
                src = R[T] if kind == 'replicate' else pd.read_parquet(MOD / f'pred_ridge_sel_{T}.parquet')
                feat = np.array([src.at[c, d] if (c in src.index and d in src.columns) else np.nan
                                 for c, d in zip(panel.compound, panel.depmap_id)])
                have = np.isfinite(y_all) & np.isfinite(feat)
            else:
                have = np.isfinite(y_all) & np.isfinite(X).all(axis=1)
            sub = panel[have]
            keep = ((sub.groupby('compound').compound.transform('size') >= 3)
                    & (sub.groupby('depmap_id').depmap_id.transform('size') >= 10)).values
            idx = np.where(have)[0][keep]
            if len(idx) < 300:
                continue
            drugs, cells = panel.compound.values[idx], panel.depmap_id.values[idx]
            yt = two_way(y_all[idx], drugs, cells)
            rec = {'eval': E, 'features': name, 'pairs': len(idx), 'drugs': int(len(set(drugs)))}
            if X is None:
                ft = two_way(feat[idx], drugs, cells)
                rec['strength_r'] = np.nan
                rec['map_r'] = float(np.corrcoef(ft, yt)[0, 1]); rec['map_ci95'] = drug_boot(ft, yt, drugs, rng)
            else:
                st = two_way(np.linalg.norm(X[idx], axis=1), drugs, cells)
                rec['strength_r'] = float(np.corrcoef(st, yt)[0, 1]); rec['strength_ci95'] = drug_boot(st, yt, drugs, rng)
                Xc = X[idx] - pd.DataFrame(X[idx]).groupby(cells).transform('mean').values
                pred = np.zeros(len(idx))
                for tr, te in GroupKFold(5).split(Xc, groups=drugs):
                    m = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Xc[tr], yt[tr])
                    pred[te] = m.predict(Xc[te])
                pt = two_way(pred, drugs, cells)
                rec['map_r'] = float(np.corrcoef(pt, yt)[0, 1]); rec['map_ci95'] = drug_boot(pt, yt, drugs, rng)
            res.append(rec)
    res = pd.DataFrame(res)
    res.to_csv(OUT / 'l1000_two_way.csv', index=False)
    agreement = {k: float(np.nanmedian([np.corrcoef(resp['measured'][i], v[i])[0, 1] for i in range(len(v))
                                         if np.isfinite(v[i]).all()]))
                 for k, v in resp.items() if k not in ('measured', 'basal_only')}
    return res, agreement


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='ab')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in SCREENS}
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in SCREENS}
    if 'a' in args.parts:
        a, n_drugs = part_a(Z, R)
    else:
        a = pd.read_csv(OUT / 'tahoe_within_drug.csv')
        n_drugs = json.loads((OUT / 'summary.json').read_text())['tahoe_mapped_drugs']
    cols = [c for c in a.columns if c.endswith('_sel') or c.endswith('_raw')]
    summ_a = {c: float(a[c].median()) for c in cols}
    print('Tahoe mapped drugs', n_drugs, 'drug-dose-screen rows', len(a))
    print(pd.Series(summ_a).round(3).to_string())
    b, agreement = part_b(Z, R)
    print(b.round(3).to_string())
    print('median profile correlation with measured:', agreement)
    write_json(OUT / 'summary.json', {'tahoe_mapped_drugs': n_drugs, 'tahoe_rows': len(a), 'tahoe_medians': summ_a,
                                      'l1000': b.to_dict(orient='records'), 'generator_agreement': agreement})


if __name__ == '__main__':
    main()
