"""Second benchmark: the IMPROVE cross-study analysis (CSA; Partin et al., Brief. Bioinform. 2026, bbaf667).

Inputs (data/improve/):
  csa_data.zip    Zenodo 15258883. Harmonized response data of CCLE, CTRPv2, GDSC1, GDSC2 and gCSI. Every
                  dose-response curve was refitted with one three-parameter Hill model, and AUC was computed over
                  one fixed range (1e-10 to 1e-4 M) and normalized to [0, 1]; lower AUC = more sensitive.
  test_preds.zip  Zenodo 15851723 (v4). Raw predictions of seven models (DeepCDR, DeepTTC, GraphDRP, HiDRA, LGBM,
                  tCNNS, UNO) for every source x target dataset and ten splits, files
                  test_preds/<SRC>_<TRG>_split_<k>_<model>.csv (read from the zip) with model, src, trg, set, split,
                  improve_sample_id, improve_chem_id, auc, auc_true, auc_pred. Within files hold split k's test pairs;
                  cross files hold every pair of the target.

Design. Within a source dataset S the ten test splits are a random 10-fold partition of its drug-cell pairs (models
see other responses of every test cell line, as in leave-pair-out). The split-k model also predicts every pair of
every other dataset T. A pair measured in both S and T therefore has one out-of-fold prediction, from the split in
which it was an S test pair, and that same model's prediction for the pair in T. Inputs are the same in both files
(one omics profile per cell line and one representation per drug), which is checked. These identical predictions
are scored against S's and T's measurements on matched cells per drug (selbench.paired_scores_matrix), as in the
DrEval analysis.

Outputs (results/revision_improve/):
  within_per_drug.csv.gz, within_summary.csv   per model and dataset: raw, selective, axis share (pooled), the
                                               perfect-predictor share, and a drug-agnostic floor (the training mean
                                               of the test cell line's rank-normal responses, per split)
  paired_per_drug.csv.gz, paired_summary.csv   per model, source and target (selbench.paired_summary)
  paired_pooled.csv                            per source and target, pooled over models: two-way bootstrap
                                               (selbench.two_way_bootstrap) and the change in selective-both
  identity_check.csv                           identical predictions in the within and cross files
With --exclude-split-replicates (revision round 2, Reviewer 1 M4), pairs whose replicate rows fall in different test
splits are dropped before scoring (in the split in which one row was a test pair, the model was trained on another
row of the same pair); outputs go to results/revision_improve/no_split_replicates/.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import PROJECT, RESULTS  # noqa: E402

sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402

DATA = PROJECT / 'data' / 'improve'
OUT = RESULTS / 'revision_improve'
DATASETS = ['CCLE', 'CTRPv2', 'GDSCv1', 'GDSCv2', 'gCSI']     # IMPROVE's source names
MODELS = ['deepcdr', 'deepttc', 'graphdrp', 'hidra', 'lgbm', 'tcnns', 'uno']
CELL, DRUG = 'improve_sample_id', 'improve_chem_id'
SEED = 20261015
N_BOOT = 500


def extract():
    """Extract the response table and splits (not the omics inputs); predictions are read from the zip directly."""
    if find(DATA / 'csa_data', 'response.tsv'):
        return
    with zipfile.ZipFile(DATA / 'csa_data.zip') as z:
        z.extractall(DATA, members=[m for m in z.namelist() if 'y_data/' in m or 'splits/' in m])


_ZIP = None


def preds_zip():
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(DATA / 'test_preds.zip')
    return _ZIP


def find(root, pattern):
    hits = sorted(Path(root).rglob(pattern))
    return hits


def responses():
    """Long tables per dataset: drug, cell, response = -AUC (higher = more sensitive)."""
    f = find(DATA / 'csa_data', 'response.tsv')
    assert f, 'response.tsv not found'
    y = pd.read_csv(f[0], sep='\t', low_memory=False)
    out = {}
    for ds in DATASETS:
        d = y[y.source == ds]
        assert len(d), ds
        d = d.groupby([DRUG, CELL], as_index=False).auc.mean()
        out[ds] = pd.DataFrame({'drug': d[DRUG].astype(str), 'cell': d[CELL].astype(str), 'response': -d.auc.values})
    return out


def load_preds(src, trg, model):
    z = preds_zip()
    names = set(z.namelist())
    frames = []
    for k in range(10):
        name = f'test_preds/{src}_{trg}_split_{k}_{model}.csv'
        if name not in names:
            continue
        with z.open(name) as fh:
            d = pd.read_csv(fh, usecols=[CELL, DRUG, 'split', 'auc_true', 'auc_pred'])
        assert (d.split == k).all(), name
        frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d[CELL], d[DRUG] = d[CELL].astype(str), d[DRUG].astype(str)
    return d


def drop_split_replicates(w):
    """Drop pairs whose rows are test pairs of more than one split (a replicate row was in that split's training)."""
    return w[w.groupby([DRUG, CELL]).split.transform('nunique') == 1]


def decompose(long, panel=None):
    Z, G, R, r2 = selbench.decompose(long)
    if panel is not None:
        G = selbench.fitness_axis(Z, panel=pd.Index(panel).intersection(Z.index))
    return Z, G


def floor_predictions(within):
    """Drug-agnostic predictor: per split, the mean rank-normal response of the test cell line over its training
    pairs (all other splits), identical for every drug of that cell line."""
    rows = []
    for k, test in within.groupby('split'):
        train = within[within.split != k]
        z = train.assign(s=-train.auc_true)
        z['z'] = z.groupby(DRUG).s.transform(lambda v: selbench.rank_normal(v.values))
        cm = z.groupby(CELL).z.mean()
        t = test[[DRUG, CELL]].copy()
        t['prediction'] = t[CELL].map(cm)
        rows.append(t)
    d = pd.concat(rows).dropna()
    return d.pivot_table(index=DRUG, columns=CELL, values='prediction')


def within_scores(task):
    model, src, P, Z, G = task
    rows = []
    for d in P.index.intersection(Z.index):
        p = P.loc[d].reindex(Z.columns).values.astype(float)
        z, g = Z.loc[d].values.astype(float), G.loc[d].values.astype(float)
        m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
        if m.sum() < selbench.MIN_PAIRED_CELLS or np.std(p[m]) < 1e-12:
            continue
        rows.append({'model': model, 'dataset': src, 'drug': d, 'n': int(m.sum()), **selbench.split_terms(p[m], z[m], g[m])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--boot', type=int, default=N_BOOT)
    ap.add_argument('--exclude-split-replicates', action='store_true')
    args = ap.parse_args()
    global OUT
    if args.exclude_split_replicates:
        OUT = OUT / 'no_split_replicates'
    clean = drop_split_replicates if args.exclude_split_replicates else (lambda w: w)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    extract()
    obs = responses()
    dec = {ds: decompose(obs[ds]) for ds in DATASETS}
    info = {ds: {'drugs': int(dec[ds][0].shape[0]), 'cells': int(dec[ds][0].shape[1]),
                 'pairs': int(len(obs[ds])), 'oracle_share': float(selbench.oracle_fitness_share(*dec[ds]))}
            for ds in DATASETS}
    print(json.dumps(info), flush=True)

    within_rows, paired_rows, ident, P_within, P_cross = [], [], [], {}, {}
    for model in MODELS:
        for src in DATASETS:
            w = load_preds(src, src, model)
            if w is None:
                print('missing within', model, src, flush=True)
                continue
            w = clean(w)
            dup = w.duplicated([DRUG, CELL]).sum()
            # replicated pairs (same cell line and drug, several rows) have one prediction per split in which a row was
            # a test pair; keep the lowest such split, in the within and the cross files alike
            w = w.sort_values('split', kind='stable').drop_duplicates([DRUG, CELL])
            P_within[(model, src)] = w.assign(pred=-w.auc_pred).pivot_table(index=DRUG, columns=CELL, values='pred')
            ident.append({'model': model, 'source': src, 'target': src, 'pairs': int(len(w)), 'duplicate_pairs': int(dup)})
            key = w.set_index([DRUG, CELL]).split
            for trg in DATASETS:
                if trg == src:
                    continue
                c = load_preds(src, trg, model)
                if c is None:
                    print('missing cross', model, src, trg, flush=True)
                    continue
                c = c.set_index([DRUG, CELL])
                common = c.index.intersection(key.index)
                cc = c.loc[common]
                cc = cc[cc.split.values == key.reindex(cc.index).values]  # the split in which the pair was an S test pair
                cc = cc[~cc.index.duplicated()]
                # identity check: same model, same inputs -> same prediction as in the within file
                wv = w.set_index([DRUG, CELL]).loc[cc.index, 'auc_pred']
                diff = np.abs(cc.auc_pred.values - wv.values)
                ident.append({'model': model, 'source': src, 'target': trg, 'pairs': int(len(cc)),
                              'max_abs_diff_vs_within': float(diff.max()) if len(diff) else np.nan,
                              'median_abs_diff_vs_within': float(np.median(diff)) if len(diff) else np.nan})
                P_cross[(model, src, trg)] = (cc.reset_index().assign(pred=lambda x: -x.auc_pred)
                                              .pivot_table(index=DRUG, columns=CELL, values='pred'))
            print(model, src, f'{time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(ident).to_csv(OUT / 'identity_check.csv', index=False)

    # floors (drug-agnostic, per source dataset; LPO design)
    floors = {}
    for src in DATASETS:
        w = load_preds(src, src, MODELS[0])
        if w is not None:
            floors[src] = floor_predictions(clean(w).sort_values('split', kind='stable').drop_duplicates([DRUG, CELL]))

    # within-dataset scores
    tasks = [(m, s, P, *dec[s]) for (m, s), P in P_within.items()] + \
            [('floor_cell_mean', s, P, *dec[s]) for s, P in floors.items()]
    with ProcessPoolExecutor(args.workers) as ex:
        for rows in ex.map(within_scores, tasks):
            within_rows += rows
    W = pd.DataFrame(within_rows)
    W.to_csv(OUT / 'within_per_drug.csv.gz', index=False, float_format='%.6g')
    ws = []
    for (m, s), d in W.groupby(['model', 'dataset']):
        ws.append({'model': m, 'dataset': s, 'drugs': int(len(d)), 'median_raw': float(d.raw.median()),
                   'median_selective': float(d.selective.median()), 'axis_share': float(d.mediated.sum() / d.raw.sum()),
                   'oracle_share': info[s]['oracle_share']})
    WS = pd.DataFrame(ws)
    # two-way bootstrap of within medians and axis share, per dataset pooled over the seven models
    boot = []
    for s in DATASETS:
        Pm = {m: P_within[(m, s)] for m in MODELS if (m, s) in P_within}
        if not Pm:
            continue
        summ, _ = selbench.two_way_bootstrap(Pm, *dec[s], n_boot=args.boot, seed=SEED)
        for stat, r in summ.iterrows():
            boot.append({'dataset': s, 'stat': stat, 'estimate': r.estimate, 'lo': r.lo, 'hi': r.hi})
    WS.to_csv(OUT / 'within_summary.csv', index=False)
    pd.DataFrame(boot).to_csv(OUT / 'within_pooled_bootstrap.csv', index=False)
    print('within done', f'{time.time() - t0:.0f}s', flush=True)

    # paired cross-dataset scores: own-panel axes and shared-panel axes
    pooled = []
    for src in DATASETS:
        for trg in DATASETS:
            if trg == src:
                continue
            shared = dec[src][0].index.intersection(dec[trg][0].index)
            variants = {'own_panel': (dec[src], dec[trg]),
                        'shared_panel': (decompose(obs[src], panel=shared), decompose(obs[trg], panel=shared))}
            for var, ((Zs, Gs), (Zt, Gt)) in variants.items():
                Pm = {}
                for m in MODELS:
                    if (m, src, trg) not in P_cross:
                        continue
                    P = P_cross[(m, src, trg)]
                    t = selbench.paired_scores_matrix(P, Zs, Gs, Zt, Gt)
                    if not len(t):
                        continue
                    t.insert(0, 'model', m); t.insert(1, 'source', src); t.insert(2, 'target', trg)
                    t.insert(3, 'axis_panel', var)
                    paired_rows.append(t)
                    Pm[m] = P
                if not Pm:
                    continue
                # reference: the source's own measurement scored as a prediction (how far the two datasets' profiles
                # agree on the same matched pairs; the most an identical prediction of the source could keep)
                both = obs[src].merge(obs[trg][['drug', 'cell']], on=['drug', 'cell'])
                Pref = both.pivot_table(index='drug', columns='cell', values='response')
                Pref = Pref.where(next(iter(Pm.values())).reindex(index=Pref.index, columns=Pref.columns).notna())
                t = selbench.paired_scores_matrix(Pref, Zs, Gs, Zt, Gt, prediction_scale="rank")
                if len(t):
                    t.insert(0, 'model', 'source_measurement'); t.insert(1, 'source', src); t.insert(2, 'target', trg)
                    t.insert(3, 'axis_panel', var)
                    paired_rows.append(t)
                summ, samples = selbench.two_way_bootstrap(Pm, Zs, Gs, Zt, Gt, n_boot=args.boot, seed=SEED)
                for stat, r in summ.iterrows():
                    pooled.append({'source': src, 'target': trg, 'axis_panel': var, 'stat': stat,
                                   'estimate': r.estimate, 'lo': r.lo, 'hi': r.hi, 'drugs': summ.attrs['drugs'],
                                   'models': summ.attrs['models']})
            print('paired', src, trg, f'{time.time() - t0:.0f}s', flush=True)
    Ptab = pd.concat(paired_rows, ignore_index=True)
    Ptab.to_csv(OUT / 'paired_per_drug.csv.gz', index=False, float_format='%.6g')
    selbench.paired_summary(Ptab, by=['axis_panel', 'source', 'target', 'model']).to_csv(OUT / 'paired_summary.csv',
                                                                                       index=False)
    # change in selective-both, pooled over models, drug bootstrap (per-drug mean over models first)
    sb = []
    w = Ptab.rename(columns={'source': 'src', 'target': 'trg'}).pivot_table(
        index=['axis_panel', 'src', 'trg', 'model', 'drug'], columns='evaluated_in',
        values='selective_both').dropna().reset_index()
    rng = np.random.default_rng(SEED)
    for (var, s, t), d in w.groupby(['axis_panel', 'src', 'trg']):
        ref = d[d.model == 'source_measurement']
        d = d[d.model != 'source_measurement']
        per = d.assign(diff=d.target - d.source).groupby('drug')['diff'].mean().values
        bs = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)])
        sb.append({'axis_panel': var, 'source': s, 'target': t, 'drugs': int(len(per)),
                   'delta_selective_both_mean': float(per.mean()), 'ci90_lo': float(np.percentile(bs, 5)),
                   'ci90_hi': float(np.percentile(bs, 95)), 'within_mean': float(d.source.mean()),
                   'cross_mean': float(d.target.mean()),
                   'measurement_cross_mean': float(ref.target.mean()) if len(ref) else np.nan,
                   'models_cross_mean_same_drugs': float(d[d.drug.isin(ref.drug)].groupby('drug').target.mean().mean())
                   if len(ref) else np.nan})
    pd.DataFrame(pooled).to_csv(OUT / 'paired_pooled.csv', index=False)
    pd.DataFrame(sb).to_csv(OUT / 'selective_both_change.csv', index=False)
    (OUT / 'run.json').write_text(json.dumps({'datasets': info, 'seconds': round(time.time() - t0), 'n_boot': args.boot,
                                              'seed': SEED}, indent=1))
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
