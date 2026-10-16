"""Cell-blind per-drug predictors trained in one screen and evaluated within and across screens.

A single 5-fold partition of all cell lines with basal features is shared by every screen, so a
cell's prediction always comes from a model that never saw that cell, whichever screen evaluates it.

Predictors (per training screen T and drug d):
  fitness_omics   ridge from omics to T's leave-d-out fitness axis (same prediction for every drug)
  ridge_raw       kernel ridge, omics -> z_T[d] (within-drug rank-normal response)
  ridge_sel       kernel ridge, omics -> r_T[d] (selective residual)
  ridge_crispr    kernel ridge, omics + CRISPR gene effect -> z_T[d]
  lineage_raw     kernel ridge, lineage one-hot only -> z_T[d]
  target_prior    -mean CRISPR effect of annotated targets (no response training)
  replicate       z_T[d] measured in T for the same cell (not cell-blind; empirical reference)

Every prediction is scored in every screen E measuring d: raw = corr(pred, z_E[d]); selective =
partial corr given E's leave-d-out fitness axis. Outputs: results/models/scores.csv.gz and the
out-of-fold prediction matrices (parquet).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import PROCESSED, RESULTS, write_json
from modeling import ALPHAS, block_kernel, corr, kernel_ridge_loo, partial_corr

DEC = RESULTS / 'decomposition'
FEAT = PROCESSED / 'features'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
N_FOLDS = 5
MIN_TRAIN = 25


# ---- worker globals (filled by initializer) ----
G = {}


def init_worker(payload):
    G.update(payload)


def fit_drug(args):
    screen, drug = args
    Z, R, cells, folds = G['Z'][screen], G['R'][screen], G['cells'], G['folds']
    z = Z.loc[drug].reindex(cells).values
    r = R.loc[drug].reindex(cells).values
    out = {}
    for name, K, target in [('ridge_raw', G['K_omics'], z), ('ridge_sel', G['K_omics'], r),
                            ('ridge_crispr', G['K_crispr'], z), ('lineage_raw', G['K_lineage'], z)]:
        if K is None:
            continue
        pred = np.full(len(cells), np.nan)
        avail = np.isfinite(target) & G['has_kernel'][name]
        for f in range(N_FOLDS):
            te = (folds == f) & G['has_kernel'][name]
            tr = avail & (folds != f)
            if tr.sum() < MIN_TRAIN or te.sum() == 0:
                continue
            ti, ei = np.where(tr)[0], np.where(te)[0]
            pred[ei], _ = kernel_ridge_loo(K[np.ix_(ti, ti)], target[ti], K[np.ix_(ei, ti)])
        out[name] = pred
    return screen, drug, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--out', default='models')
    args = ap.parse_args()
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)

    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in SCREENS}
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in SCREENS}

    expr = pd.read_parquet(FEAT / 'expression_top.parquet')
    mut = pd.read_parquet(FEAT / 'mutation.parquet')
    cn = pd.read_parquet(FEAT / 'copy_number.parquet')
    lin = pd.read_parquet(FEAT / 'lineage.parquet')
    crispr = pd.read_parquet(FEAT / 'crispr_top.parquet')
    crispr_all = pd.read_parquet(FEAT / 'crispr_all.parquet')

    screened = sorted(set().union(*[set(Z[s].columns) for s in SCREENS]))
    omics_cells = sorted(set(expr.index) & set(mut.index) & set(cn.index) & set(lin.index) & set(screened))
    cells = omics_cells
    rng = np.random.default_rng(20261014)
    folds = rng.permutation(np.arange(len(cells)) % N_FOLDS)
    pd.DataFrame({'depmap_id': cells, 'fold': folds}).to_csv(out / 'cell_folds.csv', index=False)

    idx = pd.Index(cells)
    K_omics = block_kernel([expr.loc[idx], mut.loc[idx], cn.loc[idx], lin.loc[idx]])
    K_lineage = block_kernel([lin.loc[idx]])
    has_crispr = idx.isin(crispr.index)
    cr = crispr.reindex(idx)
    cr = cr.fillna(cr.mean())
    K_crispr = block_kernel([expr.loc[idx], mut.loc[idx], cn.loc[idx], lin.loc[idx], cr])
    has_kernel = {'ridge_raw': np.ones(len(idx), bool), 'ridge_sel': np.ones(len(idx), bool),
                  'lineage_raw': np.ones(len(idx), bool), 'ridge_crispr': has_crispr}

    tasks = [(s, d) for s in SCREENS for d in Z[s].index]
    payload = {'Z': Z, 'R': R, 'cells': idx, 'folds': folds, 'K_omics': K_omics, 'K_crispr': K_crispr,
               'K_lineage': K_lineage, 'has_kernel': has_kernel}
    preds = {}
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(payload,)) as ex:
        for i, (s, d, o) in enumerate(ex.map(fit_drug, tasks, chunksize=8)):
            for name, p in o.items():
                preds.setdefault((name, s), {})[d] = p
            if i % 500 == 0:
                print('fitted', i, '/', len(tasks), flush=True)

    # fitness_omics: kernel ridge from omics to T's all-drug fitness axis (cell-blind)
    for s in SCREENS:
        fit = Z[s].mean(axis=0).reindex(idx).values
        pred = np.full(len(idx), np.nan)
        for f in range(N_FOLDS):
            tr = np.isfinite(fit) & (folds != f)
            te = folds == f
            ti, ei = np.where(tr)[0], np.where(te)[0]
            pred[ei], _ = kernel_ridge_loo(K_omics[np.ix_(ti, ti)], fit[ti], K_omics[np.ix_(ei, ti)])
        preds[('fitness_omics', s)] = {d: pred for d in Z[s].index}
        # measured fitness excludes the scored drug itself (leave-one-drug-out), so that within-screen
        # scores carry no self-contribution; fitness_omics keeps the all-drug target (weight 1/N on d)
        tot, cnt = Z[s].sum(axis=0), Z[s].notna().sum(axis=0)
        preds[('fitness_measured', s)] = {
            d: ((tot - Z[s].loc[d].fillna(0)) / (cnt - Z[s].loc[d].notna().astype(int))).reindex(idx).values
            for d in Z[s].index}
        preds[('replicate', s)] = {d: Z[s].loc[d].reindex(idx).values for d in Z[s].index}

    # target-dependency prior (training-free)
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    cr_all = crispr_all.reindex(idx)
    prior = {}
    for d, tg in comp.targets.dropna().items():
        genes = [t for t in str(tg).split(';') if t in cr_all.columns]
        if genes:
            prior[d] = -cr_all[genes].mean(axis=1).values
    for s in SCREENS:
        preds[('target_prior', s)] = {d: prior[d] for d in Z[s].index if d in prior}

    for (name, s), dct in preds.items():
        pd.DataFrame(dct, index=idx).T.astype('float32').to_parquet(out / f'pred_{name}_{s}.parquet')

    # scoring within and across screens
    Gloo = {}
    for s in SCREENS:
        z = Z[s]
        tot = z.sum(axis=0); cnt = z.notna().sum(axis=0)
        Gloo[s] = (z, tot, cnt)
    rows = []
    for (name, T), dct in preds.items():
        for d, p in dct.items():
            p = np.asarray(p, dtype=float)
            for E in SCREENS:
                if d not in Z[E].index:
                    continue
                z_all, tot, cnt = Gloo[E]
                zE = z_all.loc[d].reindex(idx).values
                own = z_all.loc[d].reindex(idx)
                g = ((tot - z_all.loc[d].fillna(0)) / (cnt - z_all.loc[d].notna().astype(int))).reindex(idx).values
                raw = corr(p, zE)
                sel, n = partial_corr(p, zE, g)
                rows.append({'model': name, 'train': T, 'eval': E, 'compound': d, 'n': n, 'raw': raw, 'sel': sel})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / 'scores.csv.gz', index=False)
    summ = scores.groupby(['model', 'train', 'eval']).agg(drugs=('compound', 'size'), raw=('raw', 'median'),
                                                         sel=('sel', 'median')).reset_index()
    summ.to_csv(out / 'score_summary.csv', index=False)
    write_json(out / 'run.json', {'cells': len(idx), 'folds': N_FOLDS, 'alphas': ALPHAS.tolist(),
                                  'tasks': len(tasks)})
    print(summ.to_string())


if __name__ == '__main__':
    main()
