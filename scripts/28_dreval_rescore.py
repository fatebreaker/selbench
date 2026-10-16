"""Re-score the published DrEval leaderboard with the selectivity-aware metric.

DrEval (Zenodo record 23159774, leaderboard_Oct26) released the out-of-fold predictions of 27 models for CTRPv2
ln IC50 (CurveCurator fits) under leave-cell-line-out cross-validation with 10 splits. For every model and drug
we compute, on a common set of cell lines, the per-drug correlation of predictions with the observed profile
(raw), the partial correlation given the screen's leave-drug-out fitness axis (selective) and the correlation of
the predictions with that axis (alignment). Raw per-drug correlation splits exactly into a fitness-mediated
term r_pg r_zg and a selective term sqrt((1 - r_pg^2)(1 - r_zg^2)) r_pz.g.

References trained here on DrEval's own splits with DepMap 24Q4 omics (expression, mutation, copy number,
lineage): a fitness-only ridge (one predicted number per cell line, identical for every drug) and per-drug
kernel ridge on the raw or the selective target. Each split's targets are derived from its training cells only.
The measured leave-drug-out fitness axis is added as a non-cell-blind upper reference.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from common import PROCESSED, PROJECT, RESULTS, load_models, write_json
from modeling import block_kernel, kernel_ridge_loo

sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402

DRE = PROJECT / 'data' / 'dreval'
LB = DRE / 'leaderboard_Oct26'
OUT = RESULTS / 'dreval'
FEAT = PROCESSED / 'features'
NAIVE = ['NaivePredictor', 'NaiveDrugMeanPredictor', 'NaiveMeanEffectsPredictor', 'NaiveTissueMeanPredictor',
         'NaiveTissueDrugMeanPredictor']
# constant within a drug except for split-to-split differences of training means: per-drug metrics undefined
CONSTANT = ['NaivePredictor', 'NaiveDrugMeanPredictor']
NEURAL = ['DIPK', 'DrugGNN', 'PaccMann', 'PharmaFormer', 'Precily', 'SparseGO', 'SimpleNeuralNetwork',
          'MultiViewNeuralNetwork', 'SRMF', 'EnsembleMF']
OURS = {'fitness_only': 'Fitness only (omics ridge)', 'ridge_raw': 'Kernel ridge, raw target',
        'ridge_sel': 'Kernel ridge, selective target', 'ridge_raw_expr': 'Kernel ridge (expression), raw target',
        'ridge_sel_expr': 'Kernel ridge (expression), selective target',
        'ridge_raw_lm': 'Kernel ridge (DrEval landmark genes), raw target',
        'ridge_sel_lm': 'Kernel ridge (DrEval landmark genes), selective target'}
B = 2000

G = {}


def init_worker(payload):
    G.update(payload)


def model_class(m):
    if m in NAIVE:
        return 'naive'
    if m in NEURAL:
        return 'neural/factorization'
    if m in OURS:
        return 'this study'
    if m == 'fitness_measured':
        return 'reference'
    return 'classical'


def load_leaderboard():
    out = {}
    for m in sorted(os.listdir(LB / 'LCO')):
        fs = sorted((LB / 'LCO' / m / 'predictions').glob('predictions_split_*.csv'))
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f, dtype={'pubchem_id': str}).assign(split=int(f.stem.rsplit('_', 1)[1]))
                       for f in fs])
        d = d.groupby(['pubchem_id', 'cell_line_name'], as_index=False).agg(
            response=('response', 'mean'), prediction=('predictions', 'mean'), split=('split', 'first'),
            tissue=('tissue', 'first'))
        # ln IC50: lower = more sensitive; flip both so that higher = more sensitive
        d['response'] = -d.response
        d['prediction'] = -d.prediction
        out[m] = d.rename(columns={'pubchem_id': 'drug', 'cell_line_name': 'cell'})
    return out


def id_maps():
    cells = pd.read_csv(DRE / 'cell_ids.csv', dtype=str)
    mod = load_models()[['ModelID', 'RRID']].dropna().drop_duplicates('RRID')
    cmap = cells.merge(mod, left_on='cellosaurus_id', right_on='RRID', how='left').set_index('cell_line_name').ModelID
    pc = pd.read_csv(DRE / 'pubchem_inchikeys.csv', dtype=str)
    pc['ik14'] = pc.InChIKey.str[:14]
    comp = pd.read_csv(PROCESSED / 'compounds.csv', dtype=str)
    ik = comp.dropna(subset=['inchikey14']).drop_duplicates('inchikey14').set_index('inchikey14').compound
    dmap = pc.set_index('CID').ik14.map(ik).dropna()
    return cmap.dropna(), dmap


def fit_split(args):
    k, drugs = args
    Ztr, Rtr = G['Z'][k], G['R'][k]
    cells, split = G['cells'], G['split']
    tr_all, te = np.where(split != k)[0], np.where(split == k)[0]
    out = {}
    for d in drugs:
        if d not in Ztr.index:
            continue
        for suffix, K in G['K'].items():
            for name, M in [('ridge_raw', Ztr), ('ridge_sel', Rtr)]:
                y = M.loc[d].reindex(cells[tr_all]).values
                ok = np.isfinite(y)
                if ok.sum() < 25:
                    continue
                ti = tr_all[ok]
                out[(name + suffix, d)] = (te, kernel_ridge_loo(K[np.ix_(ti, ti)], y[ok], K[np.ix_(te, ti)])[0])
    return k, out


def train_references(obs, split_of, cmap, workers):
    expr = pd.read_parquet(FEAT / 'expression_top.parquet')
    mut = pd.read_parquet(FEAT / 'mutation.parquet')
    cn = pd.read_parquet(FEAT / 'copy_number.parquet')
    lin = pd.read_parquet(FEAT / 'lineage.parquet')
    feat_ok = set(expr.index) & set(mut.index) & set(cn.index) & set(lin.index)
    cells = np.array(sorted(c for c in split_of.index if cmap.get(c) in feat_ok))
    dep = pd.Index(cmap.reindex(cells).values)
    K = block_kernel([expr.loc[dep], mut.loc[dep], cn.loc[dep], lin.loc[dep]])
    # expression of the L1000 landmark genes that most leaderboard models use (DrEval gene list)
    lm = set(pd.read_csv(DRE / 'meta' / 'gene_lists' / 'landmark_genes.csv').Symbol)
    expr_all = pd.read_parquet(FEAT / 'expression_all.parquet')
    lm_cols = [c for c in expr_all.columns if c in lm]
    kernels = {'': K, '_expr': block_kernel([expr.loc[dep]]), '_lm': block_kernel([expr_all.loc[dep, lm_cols]])}
    split = split_of.reindex(cells).values.astype(int)
    Z, R, fit = {}, {}, {}
    for k in range(10):
        tr = set(cells[split != k])
        z, g, r, _ = selbench.decompose(obs[obs.cell.isin(tr)])
        Z[k], R[k] = z, r
        f = z.mean(axis=0).reindex(cells[split != k]).values
        te = np.where(split == k)[0]
        ti = np.where(split != k)[0]
        fit[k] = (te, kernel_ridge_loo(K[np.ix_(ti, ti)], f, K[np.ix_(te, ti)])[0])
    drugs = sorted(obs.drug.unique())
    chunks = [(k, drugs[i::4]) for k in range(10) for i in range(4)]
    preds = {n + sfx: pd.DataFrame(np.nan, index=drugs, columns=cells) for n in ['ridge_raw', 'ridge_sel'] for sfx in kernels}
    payload = {'Z': Z, 'R': R, 'cells': cells, 'split': split, 'K': kernels}
    with ProcessPoolExecutor(workers, initializer=init_worker, initargs=(payload,)) as ex:
        for k, o in ex.map(fit_split, chunks):
            for (name, d), (te, p) in o.items():
                preds[name].loc[d, cells[te]] = p
    fo = np.full(len(cells), np.nan)
    for k, (te, p) in fit.items():
        fo[te] = p
    long = []
    for name, P in preds.items():
        s = P.stack().rename('prediction').reset_index().rename(columns={'level_0': 'drug', 'level_1': 'cell'})
        long.append(s.assign(model=name))
    fo_long = pd.DataFrame([(d, c, v) for d in drugs for c, v in zip(cells, fo)], columns=['drug', 'cell', 'prediction'])
    long.append(fo_long.assign(model='fitness_only'))
    return pd.concat(long), cells


def score(P, Z, Gm):
    """Per-drug raw, selective, alignment and the exact raw = mediated + selective-term split."""
    rows = []
    for d in Z.index.intersection(P.index):
        p, z, g = P.loc[d].reindex(Z.columns).values, Z.loc[d].values, Gm.loc[d].values
        m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
        if m.sum() < 30 or np.std(p[m]) < 1e-12:
            rows.append({'drug': d, 'n': int(m.sum())})
            continue
        c = np.corrcoef(np.vstack([p[m], z[m], g[m]]))
        r_pz, r_pg, r_zg = c[0, 1], c[0, 2], c[1, 2]
        den = np.sqrt(max((1 - r_pg ** 2) * (1 - r_zg ** 2), 0))
        sel = (r_pz - r_pg * r_zg) / den if den > 1e-9 else np.nan
        rows.append({'drug': d, 'n': int(m.sum()), 'raw': r_pz, 'selective': sel, 'alignment': r_pg,
                     'r_zg': r_zg, 'mediated': r_pg * r_zg, 'selective_term': r_pz - r_pg * r_zg})
    return pd.DataFrame(rows)


def boot_median(x, rng):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    bs = np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--retrain', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)

    lb = load_leaderboard()
    # observed responses must agree across models
    obs_all = pd.concat([d[['drug', 'cell', 'response']] for d in lb.values()])
    grp = obs_all.groupby(['drug', 'cell']).response
    spread = grp.max() - grp.min()
    obs = obs_all.groupby(['drug', 'cell'], as_index=False).response.mean()
    sp = pd.concat([d[['cell', 'split']] for m, d in lb.items() if m not in NAIVE and not m.startswith('MultiView')])
    sp = sp.drop_duplicates()
    assert sp.cell.is_unique, 'cell lines must belong to one split across models'
    split_of = sp.set_index('cell').split

    cmap, dmap = id_maps()
    cache = OUT / 'reference_predictions.csv.gz'
    if cache.exists() and not args.retrain:
        refs = pd.read_csv(cache, dtype={'drug': str})
        ref_cells = np.array(sorted(refs.cell.unique()))
    else:
        refs, ref_cells = train_references(obs, split_of, cmap, args.workers)
        refs.to_csv(cache, index=False, float_format='%.5g')

    pred_cells = [set(d.cell) for m, d in lb.items() if m not in NAIVE]
    common = sorted(set.intersection(*pred_cells) & set(ref_cells))
    Z, Gm, _, r2 = selbench.decompose(obs[obs.cell.isin(common)])
    models = {m: d for m, d in lb.items()}
    for name in OURS:
        models[name] = refs[refs.model == name]
    tables, summ = [], []
    for m, d in models.items():
        if m in CONSTANT:
            summ.append({'model': m, 'class': model_class(m), 'drugs': 0})
            continue
        P = d[d.cell.isin(common)].pivot_table(index='drug', columns='cell', values='prediction')
        T = score(P, Z, Gm).assign(model=m)
        pr2 = np.nan
        if P.shape[0] > 1 and m not in NAIVE[:3]:
            dd = d[d.cell.isin(common)].rename(columns={'prediction': 'resp_p'})[['drug', 'cell', 'resp_p']]
            _, _, _, pr = selbench.decompose(dd.rename(columns={'resp_p': 'response'}))
            pr2 = float(pr.median())
        tables.append(T)
        row = {'model': m, 'class': model_class(m), 'drugs': int(T.raw.notna().sum()) if 'raw' in T else 0,
               'prediction_fitness_r2': pr2}
        for k in ['raw', 'selective', 'alignment']:
            if k in T and T[k].notna().sum():
                row[k] = float(T[k].median())
                row[f'{k}_lo'], row[f'{k}_hi'] = boot_median(T[k], rng)
        if 'raw' in T and T.raw.notna().sum():
            ok = T.raw.notna() & T.selective.notna()
            row['mean_raw'] = float(T.raw[ok].mean())
            row['mean_mediated'] = float(T.mediated[ok].mean())
            row['mean_selective_term'] = float(T.selective_term[ok].mean())
            row['mediated_share'] = row['mean_mediated'] / row['mean_raw'] if row['mean_raw'] > 0 else np.nan
        summ.append(row)
    # measured fitness axis as a predictor (not cell-blind): raw only
    fm = pd.DataFrame({'drug': Z.index, 'raw': [np.corrcoef(Z.loc[d].values[np.isfinite(Z.loc[d].values)],
                                                            Gm.loc[d].values[np.isfinite(Z.loc[d].values)])[0, 1]
                                                for d in Z.index]}).assign(model='fitness_measured')
    tables.append(fm)
    lo, hi = boot_median(fm.raw, rng)
    summ.append({'model': 'fitness_measured', 'class': 'reference', 'drugs': len(fm), 'raw': float(fm.raw.median()),
                 'raw_lo': lo, 'raw_hi': hi, 'selective': 0.0, 'alignment': 1.0})
    per = pd.concat(tables, ignore_index=True)
    per['compound'] = per.drug.map(dmap)
    per['single_agent'] = per.drug.str.fullmatch(r'\d+')
    per.to_csv(OUT / 'per_drug.csv.gz', index=False)
    S = pd.DataFrame(summ)

    # DrEval's own leaderboard metrics (mean over the 10 splits)
    ev = pd.read_csv(LB / 'evaluation_results.csv')
    ev['model'] = ev.algorithm.str.replace(r' \(.*\)$', '', regex=True)
    evm = ev.groupby('model')[['Pearson', 'Spearman', 'Pearson: normalized', 'Spearman: normalized']].mean()
    evm.columns = ['dreval_pearson', 'dreval_spearman', 'dreval_pearson_norm', 'dreval_spearman_norm']
    S = S.merge(evm, left_on='model', right_index=True, how='left')

    # subsets: single agents, drugs mapped to the discovery compounds and their reproducibility class
    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0)
    repro = set(dt.index[dt.rho_sel >= 0.3])
    sub = {}
    for m in S.model:
        t = per[(per.model == m)]
        sub[m] = {'single_agent_selective': float(t[t.single_agent].selective.median()) if 'selective' in t else None,
                  'reproducible_selective': float(t[t.compound.isin(repro)].selective.median()),
                  'reproducible_raw': float(t[t.compound.isin(repro)].raw.median()),
                  'reproducible_drugs': int(t.compound.isin(repro).sum())}
    S = S.merge(pd.DataFrame(sub).T, left_on='model', right_index=True, how='left')
    S = S.sort_values('selective', ascending=False)
    S.to_csv(OUT / 'model_summary.csv', index=False)

    # paired comparisons over drugs (selective): ours vs every published model
    W = per.pivot_table(index='drug', columns='model', values='selective')
    comp = []
    for ours in ['ridge_sel_lm', 'ridge_raw_lm', 'ridge_sel', 'ridge_raw', 'ridge_sel_expr', 'ridge_raw_expr']:
        for m in [c for c in W.columns if c not in OURS]:
            x = (W[ours] - W[m]).dropna()
            if len(x) < 20:
                continue
            bs = np.median(x.values[rng.integers(0, len(x), (B, len(x)))], axis=1)
            comp.append({'ours': ours, 'model': m, 'drugs': len(x), 'median_diff': float(x.median()),
                         'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5)),
                         'frac_better': float((x > 0).mean()), 'wilcoxon_p': float(wilcoxon(x).pvalue)})
    pd.DataFrame(comp).to_csv(OUT / 'paired_vs_ours.csv', index=False)

    pub = S[S['class'].isin(['classical', 'neural/factorization'])].dropna(subset=['selective'])
    info = {
        'models': int(len(lb)), 'common_cells': len(common), 'drugs_evaluated': int(len(Z)),
        'mapped_cells': int(len(ref_cells)), 'mapped_drugs': int(dmap.reindex(Z.index).notna().sum()),
        'obs_max_spread_across_models': float(spread.max()),
        'median_fitness_r2_observed': float(r2.median()),
        'rank_spearman_raw_vs_selective_published': float(spearmanr(pub.raw, pub.selective)[0]),
        'rank_spearman_dreval_norm_vs_selective_published': float(spearmanr(pub.dreval_pearson_norm, pub.selective)[0]),
        'best_published_selective': pub.iloc[0][['model', 'selective', 'raw']].to_dict(),
        'best_published_raw': pub.sort_values('raw', ascending=False).iloc[0][['model', 'selective', 'raw']].to_dict(),
    }
    write_json(OUT / 'run.json', info)
    pd.set_option('display.width', 250)
    print(S[['model', 'class', 'drugs', 'raw', 'selective', 'alignment', 'mediated_share', 'prediction_fitness_r2',
             'dreval_pearson_norm', 'reproducible_selective']].round(3).to_string(index=False))
    print(info)


if __name__ == '__main__':
    main()
