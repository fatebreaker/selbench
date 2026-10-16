"""Frozen, discovery-only predictions for every DepMap cell line with basal omics.

For each compound measured in the chosen training screens, consensus targets average the per-screen
rank-normal profile (raw) and selective residual (sel) over screens. Cell lines screened in any
training screen receive out-of-fold predictions under the shared 5-fold cell partition; all other
cell lines receive predictions from a model fitted on every screened cell line.

Prediction families (matrices compound x cell, parquet):
  fit_measured     consensus drug-agnostic fitness (mean of standardised screen fitness axes)
  fit_omics        kernel ridge omics -> consensus fitness (cell-blind)
  ridge_raw_cons   kernel ridge omics -> consensus raw profile
  ridge_sel_cons   kernel ridge omics -> consensus selective profile
  lineage_sel_cons kernel ridge lineage only -> consensus selective profile
  target_prior     -mean CRISPR gene effect of annotated targets (training-free)
  rep_raw_cons     measured consensus raw profile (not cell-blind; empirical reference)
  rep_sel_cons     measured consensus selective profile (not cell-blind; empirical reference)
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import PROCESSED, RESULTS, write_json
from modeling import block_kernel, kernel_ridge_loo

DEC = RESULTS / 'decomposition'
FEAT = PROCESSED / 'features'
N_FOLDS = 5
MIN_TRAIN = 25
G = {}


def init_worker(payload):
    G.update(payload)


def fit_target(y, K, folds, screened):
    """Out-of-fold predictions for screened cells; full-model predictions for unscreened cells."""
    pred = np.full(len(y), np.nan)
    avail = np.isfinite(y)
    for f in range(N_FOLDS):
        tr = avail & (folds != f) & screened
        te = (folds == f) & screened
        if tr.sum() < MIN_TRAIN or te.sum() == 0:
            continue
        ti, ei = np.where(tr)[0], np.where(te)[0]
        pred[ei], _ = kernel_ridge_loo(K[np.ix_(ti, ti)], y[ti], K[np.ix_(ei, ti)])
    tr = avail & screened
    te = ~screened
    if tr.sum() >= MIN_TRAIN and te.any():
        ti, ei = np.where(tr)[0], np.where(te)[0]
        pred[ei], _ = kernel_ridge_loo(K[np.ix_(ti, ti)], y[ti], K[np.ix_(ei, ti)])
    return pred


def fit_compound(d):
    out = {}
    for name, K, key in [('ridge_raw_cons', G['K_omics'], 'raw'), ('ridge_sel_cons', G['K_omics'], 'sel'),
                         ('lineage_sel_cons', G['K_lineage'], 'sel')]:
        y = G[key].loc[d].values if d in G[key].index else None
        if y is None or np.isfinite(y).sum() < MIN_TRAIN:
            continue
        out[name] = fit_target(y, K, G['folds'], G['screened'])
    return d, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--screens', nargs='+', default=['PRISM', 'CTRP', 'GDSC1', 'GDSC2'])
    ap.add_argument('--out', default='frozen_predictions')
    ap.add_argument('--workers', type=int, default=40)
    args = ap.parse_args()
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)

    expr = pd.read_parquet(FEAT / 'expression_top.parquet')
    mut = pd.read_parquet(FEAT / 'mutation.parquet')
    cn = pd.read_parquet(FEAT / 'copy_number.parquet')
    lin = pd.read_parquet(FEAT / 'lineage.parquet')
    crispr_all = pd.read_parquet(FEAT / 'crispr_all.parquet')
    cells = pd.Index(sorted(set(expr.index) & set(mut.index) & set(cn.index) & set(lin.index)))

    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in args.screens}
    Rr = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in args.screens}
    # consensus profiles
    compounds = sorted(set().union(*[set(Z[s].index) for s in args.screens]))
    raw_sum = pd.DataFrame(0.0, index=compounds, columns=cells); raw_n = raw_sum.copy()
    sel_sum = raw_sum.copy()
    for s in args.screens:
        z = Z[s].reindex(index=compounds, columns=cells)
        r = Rr[s].reindex(index=compounds, columns=cells)
        raw_sum += z.fillna(0); sel_sum += r.fillna(0); raw_n += z.notna()
    raw = (raw_sum / raw_n).where(raw_n > 0)
    sel = (sel_sum / raw_n).where(raw_n > 0)
    fits = []
    for s in args.screens:
        f = Z[s].mean(axis=0)
        fits.append(((f - f.mean()) / f.std()).reindex(cells))
    fit_meas = pd.concat(fits, axis=1).mean(axis=1)

    # shared partition: reuse the discovery partition for screened cells
    part = pd.read_csv(RESULTS / 'models' / 'cell_folds.csv').set_index('depmap_id').fold
    screened_ids = set().union(*[set(Z[s].columns) for s in args.screens])
    screened = np.array([c in screened_ids and c in part.index for c in cells])
    folds = np.array([part.get(c, -1) if s else -1 for c, s in zip(cells, screened)])

    K_omics = block_kernel([expr.loc[cells], mut.loc[cells], cn.loc[cells], lin.loc[cells]])
    K_lineage = block_kernel([lin.loc[cells]])

    payload = {'raw': raw, 'sel': sel, 'K_omics': K_omics, 'K_lineage': K_lineage, 'folds': folds,
               'screened': screened}
    preds = {k: {} for k in ['ridge_raw_cons', 'ridge_sel_cons', 'lineage_sel_cons']}
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(payload,)) as ex:
        for i, (d, o) in enumerate(ex.map(fit_compound, compounds, chunksize=4)):
            for k, v in o.items():
                preds[k][d] = v
            if i % 300 == 0:
                print('fitted', i, '/', len(compounds), flush=True)

    fit_omics = fit_target(fit_meas.values, K_omics, folds, screened)
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    cr = crispr_all.reindex(cells)
    prior = {}
    for d in compounds:
        tg = comp.targets.get(d)
        genes = [t for t in str(tg).split(';') if t in cr.columns] if isinstance(tg, str) else []
        if genes:
            prior[d] = -cr[genes].mean(axis=1).values

    mats = {k: pd.DataFrame(v, index=cells).T for k, v in preds.items()}
    mats['target_prior'] = pd.DataFrame(prior, index=cells).T
    mats['fit_measured'] = pd.DataFrame(np.tile(fit_meas.values, (len(compounds), 1)), index=compounds, columns=cells)
    mats['fit_omics'] = pd.DataFrame(np.tile(fit_omics, (len(compounds), 1)), index=compounds, columns=cells)
    mats['rep_raw_cons'] = raw
    mats['rep_sel_cons'] = sel
    for k, m in mats.items():
        m.astype('float32').to_parquet(out / f'{k}.parquet')
    pd.DataFrame({'depmap_id': cells, 'screened': screened, 'fold': folds}).to_csv(out / 'cells.csv', index=False)
    n_screens = pd.Series({d: sum(d in Z[s].index for s in args.screens) for d in compounds}, name='n_screens')
    n_screens.to_csv(out / 'compound_screens.csv')
    write_json(out / 'run.json', {'screens': args.screens, 'compounds': len(compounds), 'cells': len(cells),
                                  'screened_cells': int(screened.sum()),
                                  'families': {k: list(m.shape) for k, m in mats.items()}})
    print('done', {k: m.shape for k, m in mats.items()})


if __name__ == '__main__':
    main()
