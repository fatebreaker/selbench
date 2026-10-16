"""Round 5 (second domain): CRISPR gene-dependency benchmark -- data, gene sets, axes, folds and out-of-fold predictions.

Tests whether the study's central finding (models capture the general-sensitivity axis far better than item-specific
selectivity) generalizes from drug-response prediction to gene-dependency prediction, which is scored the same way:
per-gene correlation across cell lines of predicted and measured CRISPR knockout effects ("predictability").
This script builds the benchmark and the out-of-fold (OOF) predictions; 105c_crispr_dependency_score.py scores them.

Data (DepMap 24Q4 public; fetched by 105a_fetch_depmap_crispr.py into data/raw/depmap24q4/):
  CRISPRGeneEffect.csv (Chronos), CRISPRInferredCommonEssentials.csv, AchillesNonessentialControls.csv, Model.csv
  (OncotreeLineage), OmicsExpressionProteinCodingGenesTPMLogp1.csv (indexed by ModelID: every row id is a DepMap
  ModelID; checked, with a ProfileID -> ModelID fallback through OmicsProfiles.csv/OmicsDefaultModelProfiles.csv as in
  the project's earlier scripts), CRISPRInferredModelGrowthRate.csv (mean over the screens a line has, as 03).
Items and orientation (mirroring Box 1):
  sensitivity s = -gene effect (higher = more dependent). Cell lines: those with both CRISPR and expression.
  Universe U: genes that are neither DepMap inferred common essentials nor Achilles nonessential controls, measured in
  >=30 lines with >=10 distinct values (selbench.decompose's rule).
  Active genes (analogue of active drugs): >=10% of measured lines with gene effect < -0.5. The benchmark ('active'
  setting) is the active genes (capped at the 3,000 with the largest SD if more; the cap is reported).
  Sensitivity set ('sd02'): genes of U with SD(gene effect) >= 0.2 (same cap).
  Profiles z: each gene's s rank-normalized over all benchmark cell lines (pooled over folds; selbench.rank_normal;
  identical to selbench.decompose, checked). Axis g (protocol): box1.active_axis -- for gene d the leave-gene-out mean
  of the other active genes' z (>=10 contributing genes); genes outside the active set get the full active mean.
  All-gene axis (sensitivity): selbench.fitness_axis over every gene of U (leave-gene-out).
Splits: 10-fold leave-cell-line-out (sklearn KFold, shuffle, seed 42, over the sorted cell lines). All fitting uses the
  training lines of a fold only (feature selection, scaling, hyperparameters, axis, slopes, tissue means).
Features: the 5,000 highest-variance expression genes on the training lines, standardized on the training lines.
Models (targets s; multi-output models fit per-gene standardized targets, back-transformed):
  ridge            per-gene ridge (DepMap-predictability-style linear model); alpha per gene by efficient
                   leave-one-out CV on the training lines (sklearn RidgeCV, alpha_per_target; genes grouped by their
                   missing-value pattern so each fits on its measured lines).
  random_forest    multi-output random forest (300 trees, max_features 0.1, min_samples_leaf 5; missing training
                   targets set to the training mean).
  mlp              global multi-output MLP (torch, one GPU): 5,000 -> 512 -> 256 -> genes, ReLU, dropout 0.1 (input)
                   and 0.3, AdamW (lr 1e-3, weight decay 1e-2), batch 64, masked MSE, early stopping on 10% of the
                   training lines (patience 20, max 300 epochs); mean of 3 seeds.
  knn              k-nearest-neighbour regression in the space of the top 100 principal components of the training
                   features (Euclidean); k in {5, 10, 20, 40, 80} by inner 5-fold CV on the training lines (masked MSE
                   of standardized targets); NaN-aware neighbour mean.
  pca_regression   principal-component regression: per-gene OLS on the top k components, k in
                   {5, 10, 25, 50, 100, 200} by inner 5-fold CV (masked MSE of standardized targets).
Gene-agnostic predictors (analogues of the drug-agnostic floor) and the tissue baseline:
  omics_axis       selbench.fitness_baseline: kernel ridge (LOO penalty) from the training features to each training
                   line's active-gene axis (mean z over the active genes, z rank-normalized on the training lines);
                   per gene: training mean + training slope (s on the gene's training leave-gene-out active axis) x
                   (predicted axis - the training axis mean). One value per line for every gene, scaled per gene.
  tissue_axis      selbench.tissue_mean_baseline (training-line lineage mean of the training axis; 0 for unseen
                   lineages), scaled per gene like omics_axis (the drug-agnostic tissue mean of Box 1, step 4).
  tissue_gene_mean selbench.tissue_drug_mean_baseline: mean s of the gene over training lines of the same
                   OncotreeLineage, else the gene's training mean.
  The perfect predictor is the measured profile (scored analytically in 105c).
Inputs: data/raw/depmap24q4/* (above).
Outputs (results/revision_round5/crispr_dependency/):
  inputs/   genes.csv (gene sets and activity statistics), cells.csv (fold, lineage, growth rate), sensitivity.parquet
            (cells x genes, s), Z.parquet, G_active.parquet, G_all.parquet (genes x cells), cell_axes.csv,
            expression.parquet (cells x genes, float32 cache), prepare.json
  predictions/<setting>/<model>.parquet  OOF predictions (cells x genes, float32) and, for the gene-agnostic models,
            <model>_value.parquet (the per-line predicted axis or tissue axis value); train_<setting>_<tag>.json
            (hyperparameters per fold, timings).
Usage (from scripts/, with ../../.venv/bin/python):
  105b_crispr_dependency_benchmark.py prepare
  105b_crispr_dependency_benchmark.py train --setting active --models ridge random_forest knn pca_regression \
      omics_axis tissue_axis tissue_gene_mean
  105b_crispr_dependency_benchmark.py train --setting active --models mlp --gpu 2
  (likewise --setting sd02), then 105c_crispr_dependency_score.py.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

_THREADS = os.environ.get('CRISPR_BLAS_THREADS', '8')
for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = _THREADS
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for _p in [str(HERE), str(PROJECT / 'selbench')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import selbench  # noqa: E402
import box1  # noqa: E402

DM = PROJECT / 'data' / 'raw' / 'depmap24q4'
OUT = PROJECT / 'results' / 'revision_round5' / 'crispr_dependency'
INP = OUT / 'inputs'
PRED = OUT / 'predictions'

SEED = 42
N_FOLDS = 10
N_FEATURES = 5000
MIN_LINES = 30
ACTIVE_CUT, ACTIVE_FRAC = -0.5, 0.10
SD_CUT = 0.2
MAX_GENES = 3000
MIN_AXIS_GENES = 10
RIDGE_ALPHAS = np.logspace(1, 7, 31)
KNN_KS = [5, 10, 20, 40, 80]
KNN_PCS = 100
PCR_KS = [5, 10, 25, 50, 100, 200]
INNER_FOLDS = 5
RF = dict(n_estimators=300, max_features=0.1, min_samples_leaf=5)
RF_JOBS = int(os.environ.get('CRISPR_RF_JOBS', '16'))
MLP = dict(hidden=(512, 256), dropout_in=0.1, dropout=0.3, lr=1e-3, weight_decay=1e-2, batch=64, max_epochs=300,
           patience=20, val_frac=0.1, seeds=(0, 1, 2))
LEARNED = ['ridge', 'random_forest', 'mlp', 'knn', 'pca_regression']
AGNOSTIC = ['omics_axis', 'tissue_axis']
TISSUE = ['tissue_gene_mean']
ALL_MODELS = LEARNED + AGNOSTIC + TISSUE


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def write_json(path, obj):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return [conv(v) for v in o.tolist()]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(conv(obj), indent=1) + '\n')


def symbol(g):
    return str(g).split(' (')[0]


# ----------------------------------------------------------------------------------------------------------------
# prepare
def load_expression():
    expr = pd.read_csv(DM / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv', index_col=0)
    ids = pd.Index(expr.index.astype(str))
    info = {'expression_rows': int(len(ids)), 'expression_rows_modelid': int(ids.str.startswith('ACH-').sum())}
    if not ids.str.startswith('ACH-').all():
        # ProfileID-indexed release: map default RNA profiles to ModelID (as the project's earlier loaders)
        prof = pd.read_csv(DM / 'OmicsProfiles.csv', usecols=['ProfileID', 'ModelID'])
        dflt = pd.read_csv(DM / 'OmicsDefaultModelProfiles.csv')
        dflt = dflt[dflt.ProfileType == 'rna']
        p2m = prof.set_index('ProfileID').ModelID
        keep = ids.isin(dflt.ProfileID)
        expr = expr.loc[keep]
        expr.index = p2m.reindex(expr.index).values
        info['mapped_from_profile_ids'] = True
    dflt = pd.read_csv(DM / 'OmicsDefaultModelProfiles.csv')
    info['expression_models_in_default_rna_profiles'] = int(expr.index.isin(dflt.loc[dflt.ProfileType == 'rna',
                                                                                       'ModelID']).sum())
    expr = expr.loc[~expr.index.duplicated()]
    return expr, info


def gene_list(path):
    return set(pd.read_csv(path).iloc[:, 0].astype(str))


def prepare():
    t0 = time.time()
    INP.mkdir(parents=True, exist_ok=True)
    ce = pd.read_csv(DM / 'CRISPRGeneEffect.csv', index_col=0)
    expr, info = load_expression()
    cells = sorted(set(ce.index) & set(expr.index))
    ess = gene_list(DM / 'CRISPRInferredCommonEssentials.csv')
    non = gene_list(DM / 'AchillesNonessentialControls.csv')
    E = ce.loc[cells]
    info.update({'crispr_lines': int(ce.shape[0]), 'crispr_genes': int(ce.shape[1]), 'lines_crispr_and_expression':
                 len(cells), 'common_essentials_listed': len(ess), 'common_essentials_in_crispr':
                 int(len(ess & set(ce.columns))), 'nonessential_controls_listed': len(non),
                 'nonessential_controls_in_crispr': int(len(non & set(ce.columns)))})
    keep = [g for g in E.columns if g not in ess and g not in non]
    n = E[keep].notna().sum()
    nun = E[keep].nunique()
    U = [g for g in keep if n[g] >= MIN_LINES and nun[g] >= 10]
    EU = E[U]
    n = EU.notna().sum()
    frac = (EU < ACTIVE_CUT).sum() / n
    sd = EU.std()
    active = [g for g in U if frac[g] >= ACTIVE_FRAC]
    sdset = [g for g in U if sd[g] >= SD_CUT]
    info.update({'universe_genes': len(U), 'active_genes_before_cap': len(active), 'sd02_genes_before_cap': len(sdset)})
    if len(active) > MAX_GENES:
        active = list(sd[active].sort_values(ascending=False).index[:MAX_GENES])
    if len(sdset) > MAX_GENES:
        sdset = list(sd[sdset].sort_values(ascending=False).index[:MAX_GENES])
    active, sdset = sorted(active), sorted(sdset)
    info.update({'active_genes': len(active), 'sd02_genes': len(sdset), 'active_and_sd02': len(set(active) & set(sdset)),
                 'active_cap_applied': info['active_genes_before_cap'] > MAX_GENES,
                 'sd02_cap_applied': info['sd02_genes_before_cap'] > MAX_GENES})
    S = -EU                                                    # sensitivity: higher = more dependent
    # rank-normal profiles over all benchmark lines (pooled over folds)
    ZU = pd.DataFrame(np.vstack([selbench.rank_normal(S[g].values) for g in U]), index=U, columns=cells)
    # check against selbench.decompose (the reference implementation) on the active genes
    lg = S[active].stack().rename('response').reset_index()
    lg.columns = ['cell', 'drug', 'response']
    Zd, _, _, _ = selbench.decompose(lg[['drug', 'cell', 'response']])
    Zd = Zd.reindex(index=active, columns=cells)
    info['decompose_check_max_abs_diff'] = float(np.nanmax(np.abs(Zd.values - ZU.loc[active].values)))
    info['decompose_check_genes'] = int(len(Zd.dropna(how='all')))
    ax_active, G_act = box1.active_axis(ZU, active, min_drugs=MIN_AXIS_GENES)
    G_all = selbench.fitness_axis(ZU, min_drugs=MIN_AXIS_GENES)
    genes = sorted(set(active) | set(sdset))
    # common-essential axis (descriptive; 105c's axis analysis)
    ce_genes = [g for g in E.columns if g in ess and E[g].notna().sum() >= MIN_LINES]
    Zce = np.vstack([selbench.rank_normal(-E[g].values) for g in ce_genes])
    cell_axes = pd.DataFrame({'axis_active': ax_active.reindex(cells).values,
                              'axis_all_genes': ZU.mean(axis=0).reindex(cells).values,
                              'axis_common_essentials': np.nanmean(Zce, axis=0),
                              'n_active_genes': ZU.loc[active].notna().sum(axis=0).reindex(cells).values},
                             index=pd.Index(cells, name='cell'))
    # folds, lineage and growth
    from sklearn.model_selection import KFold
    fold = np.zeros(len(cells), int)
    for k, (_, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(cells)):
        fold[te] = k
    model = pd.read_csv(DM / 'Model.csv', index_col=0)
    lin = model.OncotreeLineage.reindex(cells).fillna('Unknown')
    vc = lin.value_counts()
    lin_pooled = lin.where(lin.map(vc) >= 5, 'Other (<5 lines)')
    growth = pd.read_csv(DM / 'CRISPRInferredModelGrowthRate.csv', index_col=0).mean(axis=1)
    cel = pd.DataFrame({'fold': fold, 'lineage': lin.values, 'lineage_pooled': lin_pooled.values,
                        'growth_rate': growth.reindex(cells).values}, index=pd.Index(cells, name='cell'))
    info.update({'lineages': int(vc.size), 'lineages_ge5': int((vc >= 5).sum()), 'lines_in_small_lineages':
                 int(vc[vc < 5].sum()), 'growth_rate_lines': int(cel.growth_rate.notna().sum()),
                 'fold_sizes': cel.fold.value_counts().sort_index().tolist(),
                 'common_essential_genes_in_axis': len(ce_genes)})
    gtab = pd.DataFrame({'gene': U, 'symbol': [symbol(g) for g in U], 'n_lines': n[U].values,
                         'frac_effect_lt_-0.5': frac[U].values, 'sd_effect': sd[U].values,
                         'mean_effect': EU.mean()[U].values})
    gtab['in_active'] = gtab.gene.isin(active)
    gtab['in_sd02'] = gtab.gene.isin(sdset)
    gtab['in_universe'] = True
    gtab.to_csv(INP / 'genes.csv', index=False)
    cel.to_csv(INP / 'cells.csv')
    cell_axes.to_csv(INP / 'cell_axes.csv')
    S[genes].to_parquet(INP / 'sensitivity.parquet')
    ZU.loc[genes].to_parquet(INP / 'Z.parquet')
    G_act.loc[genes].to_parquet(INP / 'G_active.parquet')
    G_all.loc[genes].to_parquet(INP / 'G_all.parquet')
    expr.loc[cells].astype('float32').to_parquet(INP / 'expression.parquet')
    info['runtime_s'] = time.time() - t0
    write_json(INP / 'prepare.json', info)
    log('prepare', info)


# ----------------------------------------------------------------------------------------------------------------
# models
def by_pattern(mask):
    """Group target columns by their missing-value pattern over the training rows: {pattern: column indices}."""
    groups = {}
    for j in range(mask.shape[1]):
        groups.setdefault(mask[:, j].tobytes(), []).append(j)
    return [(mask[:, cols[0]], np.array(cols)) for cols in groups.values()]


def fit_ridge(Xtr, Ytr, Xte):
    from sklearn.linear_model import RidgeCV
    pred = np.full((Xte.shape[0], Ytr.shape[1]), np.nan)
    alphas = np.full(Ytr.shape[1], np.nan)
    for rows, cols in by_pattern(np.isfinite(Ytr)):
        m = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(Xtr[rows], Ytr[np.ix_(rows, cols)])
        pred[:, cols] = m.predict(Xte).reshape(len(Xte), len(cols))
        alphas[cols] = np.atleast_1d(m.alpha_)
    a = np.log10(alphas)
    return pred, {'alpha_log10_median': float(np.median(a)), 'alpha_log10_q05': float(np.quantile(a, 0.05)),
                  'alpha_log10_q95': float(np.quantile(a, 0.95)),
                  'alpha_at_lower_bound': int((alphas <= RIDGE_ALPHAS[0]).sum()),
                  'alpha_at_upper_bound': int((alphas >= RIDGE_ALPHAS[-1]).sum())}


def fit_rf(Xtr, Ys, Xte, seed):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(random_state=seed, n_jobs=RF_JOBS, **RF).fit(Xtr, np.nan_to_num(Ys))
    return m.predict(Xte), dict(RF)


def pcs(Xtr, Xte, k):
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    return U[:, :k] * s[:k], Xte @ Vt[:k].T


def inner_splits(n, seed):
    from sklearn.model_selection import KFold
    return list(KFold(INNER_FOLDS, shuffle=True, random_state=seed).split(np.arange(n)))


def masked_mse(pred, Y):
    m = np.isfinite(Y) & np.isfinite(pred)
    return float(((pred - np.where(m, Y, 0)) ** 2)[m].mean())


def knn_predict(Ptr, Ys, Pte, ks):
    """NaN-aware mean of the k nearest training lines' standardized targets for every k in ks: {k: prediction}."""
    d = ((Pte[:, None, :] - Ptr[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d, axis=1, kind='mergesort')
    fin = np.isfinite(Ys)
    Y0 = np.where(fin, Ys, 0.0)
    out = {}
    for k in ks:
        nb = order[:, :k]
        s = Y0[nb].sum(axis=1)
        c = fin[nb].sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[k] = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    return out


def fit_knn(Xtr, Ys, Xte, seed):
    Ptr, Pte = pcs(Xtr, Xte, KNN_PCS)
    err = {k: [] for k in KNN_KS}
    for a, b in inner_splits(len(Xtr), seed):
        pr = knn_predict(Ptr[a], Ys[a], Ptr[b], KNN_KS)
        for k in KNN_KS:
            err[k].append(masked_mse(pr[k], Ys[b]))
    cv = {k: float(np.mean(v)) for k, v in err.items()}
    best = min(cv, key=cv.get)
    return knn_predict(Ptr, Ys, Pte, [best])[best], {'k': best, 'cv_mse': cv, 'pcs': KNN_PCS}


def ols_predict(Ptr, Ytr, Pte):
    """Per-target OLS with intercept on the measured training rows (targets grouped by missing pattern)."""
    pred = np.full((Pte.shape[0], Ytr.shape[1]), np.nan)
    for rows, cols in by_pattern(np.isfinite(Ytr)):
        X = np.c_[np.ones(rows.sum()), Ptr[rows]]
        b = np.linalg.lstsq(X, Ytr[np.ix_(rows, cols)], rcond=None)[0]
        pred[:, cols] = np.c_[np.ones(len(Pte)), Pte] @ b
    return pred


def fit_pcr(Xtr, Ys, Xte, seed):
    kmax = max(PCR_KS)
    Ptr, Pte = pcs(Xtr, Xte, kmax)
    err = {k: [] for k in PCR_KS}
    for a, b in inner_splits(len(Xtr), seed):
        for k in PCR_KS:
            err[k].append(masked_mse(ols_predict(Ptr[a][:, :k], Ys[a], Ptr[b][:, :k]), Ys[b]))
    cv = {k: float(np.mean(v)) for k, v in err.items()}
    best = min(cv, key=cv.get)
    return ols_predict(Ptr[:, :best], Ys, Pte[:, :best]), {'k': best, 'cv_mse': cv}


def fit_mlp(Xtr, Ys, Xte, seed, device):
    import torch
    from torch import nn
    preds, epochs = [], []
    for s in MLP['seeds']:
        torch.manual_seed(1000 * seed + s)
        rng = np.random.default_rng(1000 * seed + s)
        n = len(Xtr)
        perm = rng.permutation(n)
        nval = max(int(round(MLP['val_frac'] * n)), 20)
        va, tr = perm[:nval], perm[nval:]
        X = torch.tensor(Xtr, dtype=torch.float32, device=device)
        Y = torch.tensor(np.nan_to_num(Ys), dtype=torch.float32, device=device)
        M = torch.tensor(np.isfinite(Ys), dtype=torch.float32, device=device)
        h1, h2 = MLP['hidden']
        net = nn.Sequential(nn.Dropout(MLP['dropout_in']), nn.Linear(X.shape[1], h1), nn.ReLU(), nn.Dropout(MLP['dropout']),
                            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(MLP['dropout']), nn.Linear(h2, Y.shape[1])).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=MLP['lr'], weight_decay=MLP['weight_decay'])
        best, best_state, best_ep, wait = np.inf, None, 0, 0
        vt = torch.tensor(va, device=device)
        for ep in range(MLP['max_epochs']):
            net.train()
            order = rng.permutation(tr)
            for i in range(0, len(order), MLP['batch']):
                b = torch.tensor(order[i:i + MLP['batch']], device=device)
                opt.zero_grad()
                loss = (((net(X[b]) - Y[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(((((net(X[vt]) - Y[vt]) ** 2) * M[vt]).sum() / M[vt].sum()).item())
            if vl < best - 1e-6:
                best, best_ep, wait = vl, ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= MLP['patience']:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            preds.append(net(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy().astype(float))
        epochs.append(best_ep)
    return np.mean(preds, axis=0), {**{k: v for k, v in MLP.items() if k != 'seeds'}, 'seeds': list(MLP['seeds']),
                                    'best_epochs': epochs}


def long_obs(S, cells):
    lg = S.loc[cells].stack().rename('response').reset_index()
    lg.columns = ['cell', 'drug', 'response']
    return lg[['drug', 'cell', 'response']]


def agnostic_scaling(S_set, S_act, tr):
    """Per gene of the setting: training mean, training slope of s on the gene's training leave-gene-out active axis
    (full active mean for genes outside the active set) and that axis's training mean."""
    Ztr, Gtr, _, _ = selbench.decompose(long_obs(S_act, tr))
    full = Ztr.mean(axis=0).reindex(tr)
    rows = {}
    for g in S_set.columns:
        y = S_set.loc[tr, g].values.astype(float)
        x = (Gtr.loc[g].reindex(tr) if g in Gtr.index else full).values.astype(float)
        m = np.isfinite(y) & np.isfinite(x)
        xm, ym = x[m].mean(), y[m].mean()
        b = float(((x[m] - xm) * (y[m] - ym)).sum() / ((x[m] - xm) ** 2).sum())
        rows[g] = (ym, b, xm)
    return pd.DataFrame(rows, index=['mean', 'slope', 'axis_mean']).T


def train(setting, models, gpu):
    t_all = time.time()
    genes = pd.read_csv(INP / 'genes.csv')
    gset = list(genes.loc[genes['in_active' if setting == 'active' else 'in_sd02'], 'gene'])
    act = list(genes.loc[genes.in_active, 'gene'])
    cel = pd.read_csv(INP / 'cells.csv', index_col=0)
    S_all = pd.read_parquet(INP / 'sensitivity.parquet')
    S, S_act = S_all[gset], S_all[act]
    expr = pd.read_parquet(INP / 'expression.parquet').loc[cel.index]
    out = PRED / setting
    out.mkdir(parents=True, exist_ok=True)
    device = None
    if 'mlp' in models:
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        log('mlp device', device, os.environ.get('CUDA_VISIBLE_DEVICES'))
    P = {m: pd.DataFrame(np.nan, index=cel.index, columns=gset) for m in models}
    V = {m: pd.Series(np.nan, index=cel.index) for m in models if m in AGNOSTIC}
    meta = {'setting': setting, 'genes': len(gset), 'cells': len(cel), 'models': models, 'folds': [],
            'n_features': N_FEATURES, 'threads': _THREADS}
    for k in range(N_FOLDS):
        tr = list(cel.index[cel.fold != k])
        te = list(cel.index[cel.fold == k])
        fm = {'fold': k, 'train': len(tr), 'test': len(te), 'models': {}}
        feats = expr.loc[tr].var().sort_values(ascending=False).index[:N_FEATURES]
        Xtr_raw, Xte_raw = expr.loc[tr, feats].values.astype(float), expr.loc[te, feats].values.astype(float)
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
        sd[sd < 1e-8] = 1.0
        Xtr, Xte = (Xtr_raw - mu) / sd, (Xte_raw - mu) / sd
        Ytr = S.loc[tr].values.astype(float)
        ymu, ysd = np.nanmean(Ytr, 0), np.nanstd(Ytr, 0)
        ysd[ysd < 1e-12] = 1.0
        Ys = (Ytr - ymu) / ysd
        scal = None
        for m in models:
            t0 = time.time()
            if m == 'ridge':
                p, h = fit_ridge(Xtr, Ytr, Xte)
            elif m == 'random_forest':
                p, h = fit_rf(Xtr.astype(np.float32), Ys, Xte.astype(np.float32), SEED + k)
                p = p * ysd + ymu
            elif m == 'knn':
                p, h = fit_knn(Xtr, Ys, Xte, SEED + k)
                p = p * ysd + ymu
            elif m == 'pca_regression':
                p, h = fit_pcr(Xtr, Ys, Xte, SEED + k)
                p = p * ysd + ymu
            elif m == 'mlp':
                p, h = fit_mlp(Xtr, Ys, Xte, SEED + k, device)
                p = p * ysd + ymu
            elif m in AGNOSTIC:
                if scal is None:
                    scal = agnostic_scaling(S, S_act, tr)
                obs_tr = long_obs(S_act, tr)
                if m == 'omics_axis':
                    feat = pd.DataFrame(np.vstack([Xtr_raw, Xte_raw]), index=tr + te, columns=feats)
                    lg =selbench.fitness_baseline(obs_tr, feat, cells=te, drugs=['__axis__'])
                    val = lg.set_index('cell').prediction.reindex(te)
                else:
                    val = selbench.tissue_mean_baseline(obs_tr, cel.lineage, te)
                V[m].loc[te] = val.values
                sc = scal.loc[gset]
                p = sc['mean'].values[None, :] + sc['slope'].values[None, :] * (val.values[:, None] - sc['axis_mean'].values[None, :])
                h = {'slope_median': float(sc.slope.median()), 'slope_negative': int((sc.slope < 0).sum())}
            elif m == 'tissue_gene_mean':
                lg = selbench.tissue_drug_mean_baseline(long_obs(S, tr), cel.lineage, te, drugs=gset)
                p = lg.pivot_table(index='cell', columns='drug', values='prediction').reindex(index=te, columns=gset).values
                h = {}
            else:
                raise ValueError(m)
            P[m].loc[te] = p
            fm['models'][m] = {'seconds': round(time.time() - t0, 1), **h}
            log(setting, 'fold', k, m, f'{time.time() - t0:.1f}s', {kk: vv for kk, vv in h.items() if kk != 'cv_mse'})
        meta['folds'].append(fm)
    for m in models:
        P[m].astype('float32').to_parquet(out / f'{m}.parquet')
        if m in V:
            V[m].rename('value').to_frame().to_parquet(out / f'{m}_value.parquet')
    meta['runtime_s'] = time.time() - t_all
    tag = '_'.join(models) if len(models) <= 2 else 'cpu'
    write_json(out / f'train_{setting}_{tag}.json', meta)
    log('train done', setting, models, f'{meta["runtime_s"]:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'train'])
    ap.add_argument('--setting', choices=['active', 'sd02'], default='active')
    ap.add_argument('--models', nargs='+', default=[m for m in ALL_MODELS if m != 'mlp'])
    ap.add_argument('--gpu', default=None, help='single CUDA device index for the MLP')
    a = ap.parse_args()
    if a.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    if a.stage == 'prepare':
        prepare()
    else:
        train(a.setting, a.models, a.gpu)


if __name__ == '__main__':
    main()
