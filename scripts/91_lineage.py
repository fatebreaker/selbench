"""Round 2 (WS6b), task 2 (editor item 4; R1-M2; R2-Major 5): lineage-adjusted selective scores and the
tissue-by-drug baseline beside every learned model.

Per model and drug (cells and two-way draws exactly as 78_capture.py, so the axis-only statistics reproduce 78):
  selective             r(p, z | g)                 partial correlation given the leave-drug-out axis (the paper's)
  lineage-adjusted      r(p, z | g, T)              partial correlation given the axis and tissue indicators T of
                                                     the evaluation cells (one indicator per tissue present among the
                                                     drug's cells, the most frequent as reference; every label kept)
  covariate term        S_pC S_CC^-1 S_Cz / (s_p s_z) the part of r(p, z) carried by (g, T) jointly, C = (g, T);
                                                     r(p, z) = covariate term + residual term exactly
  R2_z|C                share of z explained by (g, T) (the perfect predictor's covariate term)
  lineage-adjusted selective capture  sum (r_pz - covariate term) / sum (1 - R2_z|C)   (78's selective capture with
                                                     (g, T) in place of g)
Weighted moments under every bootstrap draw (cell weights inside each correlation, drug multiplicities when pooling);
a ridge of 1e-9 x the largest covariate variance keeps S_CC invertible when a tissue receives no weight in a draw.
The tissue-by-drug baseline is scored identically: DrEval's NaiveTissueDrugMeanPredictor (leaderboard: released;
retraining: run in drevalpy, or the released one for CTRP v2 ln IC50), and for IMPROVE a training-fold tissue x drug
mean (per split, the mean response of the drug's training pairs (other splits' pairs, as 68's floor) in the test cell's
tissue, else the drug's training mean). Its lineage-adjusted score is not meaningful: fitted in leave-cell-line-out
splits, its per-tissue means differ between splits, and the part left after the tissue indicators is that split-to-
split leave-out variation, which is negatively related to the test cells' responses (a negative score by construction).
Statistics built on R2_z|C get basic bootstrap intervals besides percentile ones (summ_basic).
Tissue labels: DrEval's tissue mapping (results/revision_crosslab/cell_tissue.csv; identical to the retraining
responses' tissue column) for Cellosaurus ids; IMPROVE's DepMap ids are mapped through DepMap Model.csv RRID to the same
labels, else DepMap OncotreeLineage.
Settings: (a) DrEval's released CTRP v2 ln IC50 leaderboard (42's 739 cells and draws), every model; (b) the drevalpy
retraining (59 main panels): GDSC2 AUC, CTRP v2 ln IC50, GDSC2 ln IC50 (learned models when their 10 splits exist);
(c) IMPROVE, replicate-free within-dataset predictions (5 datasets; each model on its own pairs).
Pools (sums over models and drugs, same draws): leaderboard published 21; retraining DrEval learned models (retrained)
and per-drug ridges (and the published 21 for CTRP v2 ln IC50); IMPROVE 7 models.
Usage: 91_lineage.py [--parts leaderboard retrain improve] [--settings GDSC2_AUC ...]
Outputs (results/revision_round2/lineage/): lineage_by_model[_tag].csv, lineage_pooled[_tag].csv,
per_drug_lineage[_tag].csv.gz, run[_tag].json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.out_dir('lineage')
WORKERS = 3
DREVAL_LEARNED = ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting']
STATS = ('raw', 'r_pg', 'r_zg', 'sel', 'sel_lin', 'cov_term', 'r2z_lin')
RIDGE = 1e-9
TISSUE_BASELINES = {'NaiveTissueDrugMeanPredictor', 'LB:NaiveTissueDrugMeanPredictor', 'tissue_drug_mean'}


# ----------------------------------------------------------------------------------------------------------------
def tissue_of_cellosaurus():
    return pd.read_csv(lib.RES / 'revision_crosslab' / 'cell_tissue.csv').set_index('cellosaurus_id').tissue


def tissue_of_depmap():
    m = pd.read_csv(lib.PROJECT.parent / 'data' / 'raw' / 'depmap24q4_metadata' / 'Model.csv',
                    usecols=['ModelID', 'RRID', 'OncotreeLineage'])
    ct = tissue_of_cellosaurus()
    lab = m.RRID.map(ct)
    src = np.where(lab.notna(), 'DrEval tissue (via RRID)', 'DepMap OncotreeLineage')
    lab = lab.fillna(m.OncotreeLineage)
    return pd.Series(lab.values, index=m.ModelID.values), pd.Series(src, index=m.ModelID.values)


def _std(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.full_like(x, np.nan)


def lineage_stats(z, g, tis, P, W):
    """One drug. z, g: (n,); tis: (n,) tissue labels; P: (n, M); W: (R, n).
    Returns {stat: (R, M)} for raw, r_pg, r_zg, sel (| g), sel_lin (| g, T), cov_term, r2z_lin."""
    z, g = _std(z), _std(g)
    P = np.column_stack([_std(P[:, j]) for j in range(P.shape[1])])
    labs, cnt = np.unique(tis.astype(str), return_counts=True)
    ref = labs[np.argmax(cnt)]
    T = np.column_stack([(tis == l).astype(float) for l in labs if l != ref]) if len(labs) > 1 else np.zeros((len(z), 0))
    C = np.column_stack([g, T])
    M, c = P.shape[1], C.shape[1]
    X = np.column_stack([z, P, C])                       # z | P (M) | C (c)
    iz, iP, iC = 0, np.arange(1, 1 + M), np.arange(1 + M, 1 + M + c)
    S0 = W.sum(1)
    E1 = (W @ X) / S0[:, None]
    prods = [X[:, [iz]] * X[:, [iz]], X[:, iP] * X[:, [iz]], X[:, iP] ** 2,
             X[:, iC] * X[:, [iz]]]
    prods += [X[:, iP] * X[:, [k]] for k in iC]          # M x c, column block per covariate
    iu = np.triu_indices(c)
    prods.append(X[:, iC][:, iu[0]] * X[:, iC][:, iu[1]])
    Q = np.concatenate(prods, axis=1)
    E2 = (W @ Q) / S0[:, None]
    R = W.shape[0]
    mz, mP, mC = E1[:, iz], E1[:, iP], E1[:, iC]
    o = 0
    vz = E2[:, o] - mz ** 2; o += 1
    cpz = E2[:, o:o + M] - mP * mz[:, None]; o += M
    vp = E2[:, o:o + M] - mP ** 2; o += M
    czC = E2[:, o:o + c] - mC * mz[:, None]; o += c
    cPC = np.empty((R, c, M))
    for k in range(c):
        cPC[:, k, :] = E2[:, o:o + M] - mP * mC[:, [k]]
        o += M
    ncc = len(iu[0])
    CC = np.empty((R, c, c))
    vals = E2[:, o:o + ncc] - mC[:, iu[0]] * mC[:, iu[1]]
    CC[:, iu[0], iu[1]] = vals
    CC[:, iu[1], iu[0]] = vals
    lam = RIDGE * np.max(np.diagonal(CC, axis1=1, axis2=2), axis=1)
    CCr = CC + lam[:, None, None] * np.eye(c)[None]
    rhs = np.concatenate([czC[:, :, None], cPC], axis=2)          # (R, c, 1 + M)
    beta = np.linalg.solve(CCr, rhs)
    bz, bP = beta[:, :, 0], beta[:, :, 1:]
    expl_zz = np.einsum('rc,rc->r', czC, bz)
    expl_pz = np.einsum('rc,rcm->rm', czC, bP)
    expl_pp = np.einsum('rcm,rcm->rm', cPC, bP)
    with np.errstate(invalid='ignore', divide='ignore'):
        pvz = vz - expl_zz
        pvp = vp - expl_pp
        pc = cpz - expl_pz
        ok = (pvz[:, None] > 1e-8 * vz[:, None]) & (pvp > 1e-8 * vp)
        sel_lin = np.where(ok, pc / np.sqrt(pvz[:, None] * pvp), np.nan)
        cov_term = expl_pz / np.sqrt(vz[:, None] * vp)
        r2z = np.broadcast_to((expl_zz / vz)[:, None], (R, M))
        # axis-only (78/77 within_stats)
        vg = CC[:, 0, 0]
        czg = czC[:, 0]
        cpg = cPC[:, 0, :]
        raw = cpz / np.sqrt(vp * vz[:, None])
        rpg = cpg / np.sqrt(vp * vg[:, None])
        rzg = np.broadcast_to((czg / np.sqrt(vz * vg))[:, None], (R, M))
        sel = lib.pcor1(vp, vz[:, None], cpz, cpg, czg[:, None], vg[:, None])
    return {'raw': raw, 'r_pg': rpg, 'r_zg': np.array(rzg), 'sel': sel, 'sel_lin': sel_lin, 'cov_term': cov_term,
            'r2z_lin': np.array(r2z)}, len(labs)


def engine(Z, G, Pm, tissue, W, V, mask='common', mask_models=None):
    """Z, G: drug x cell; Pm: {model: drug x cell (columns aligned to Z)}; tissue: Series over Z's columns; W (R, n
    cells of Z); V (R, n drugs of Z). mask: 'common' (cells where every model predicting the drug is finite, 59/78),
    'per_model' (each model on its own cells, 68/78 IMPROVE), 'subset' (cells where every model in mask_models is
    finite; other models are scored only where finite on all of them, 78 leaderboard)."""
    models = list(Pm)
    drugs = list(Z.index)
    Rr, M, D = W.shape[0], len(models), len(drugs)
    out = {s: np.full((Rr, M, D), np.nan, np.float32) for s in STATS}
    ncell = np.zeros((M, D), int)
    ntis = np.zeros(D, int)
    Zv, Gv = Z.values.astype(float), G.reindex(index=Z.index, columns=Z.columns).values.astype(float)
    tv = tissue.reindex(Z.columns).fillna('unknown').astype(str).values
    Pv = {m: Pm[m].reindex(index=Z.index, columns=Z.columns).values.astype(float) for m in models}
    for j, d in enumerate(drugs):
        z, g = Zv[j], Gv[j]
        base = np.isfinite(z) & np.isfinite(g)
        have = [m for m in models if np.isfinite(Pv[m][j]).any()]
        if not have:
            continue
        if mask == 'common':
            mk = base.copy()
            for m in have:
                mk &= np.isfinite(Pv[m][j])
            groups = [(mk, have)]
        elif mask == 'subset':
            mk = base.copy()
            for m in mask_models:
                if m in have:
                    mk &= np.isfinite(Pv[m][j])
            groups = [(mk, [m for m in have if np.isfinite(Pv[m][j][mk]).all()])]
        else:
            gd = {}
            for m in have:
                mk = base & np.isfinite(Pv[m][j])
                gd.setdefault(mk.tobytes(), (mk, []))[1].append(m)
            groups = list(gd.values())
        for mk, ms in groups:
            if mk.sum() < lib.MIN_CELLS:
                continue
            idx = np.where(mk)[0]
            keep = [m for m in ms if np.std(Pv[m][j][idx]) >= 1e-12]
            if not keep:
                continue
            st, nt = lineage_stats(z[idx], g[idx], tv[idx], np.column_stack([Pv[m][j][idx] for m in keep]), W[:, idx])
            ntis[j] = max(ntis[j], nt)
            for jj, m in enumerate(keep):
                i = models.index(m)
                for s in STATS:
                    out[s][:, i, j] = st[s][:, jj]
                ncell[i, j] = len(idx)
    return out, models, drugs, ncell, ntis


# ----------------------------------------------------------------------------------------------------------------
def reps(a, V, mask_lin=True):
    """Per-draw summaries from arrays (R, D) of one model, or (R, K*D) concatenated over models with V tiled."""
    raw, rpg, rzg, sel = (a[k].astype(float) for k in ('raw', 'r_pg', 'r_zg', 'sel'))
    sl, ct, r2 = (a[k].astype(float) for k in ('sel_lin', 'cov_term', 'r2z_lin'))
    ok = np.isfinite(raw) & np.isfinite(rpg) & np.isfinite(rzg) & np.isfinite(sel)
    okl = ok & np.isfinite(sl) if mask_lin else ok
    okc = ok & np.isfinite(ct) & np.isfinite(r2)
    out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        for lab, msk in [('', ok)]:
            w = np.where(msk, V, 0.0)
            f = lambda x: (w * np.where(msk, x, 0.0)).sum(1)  # noqa: E731
            n = w.sum(1)
            med = rpg * rzg
            out.update({'mean_raw': f(raw) / n, 'mean_selective': f(sel) / n, 'axis_share': f(med) / f(raw),
                        'perfect_share': f(rzg ** 2) / n,
                        'selective_capture': (f(raw) - f(med)) / (n - f(rzg ** 2)),
                        'axis_capture': f(med) / f(rzg ** 2)})
        w = np.where(okl, V, 0.0)
        f = lambda x: (w * np.where(okl, x, 0.0)).sum(1)  # noqa: E731
        n = w.sum(1)
        out.update({'mean_selective_lin': f(sl) / n, 'mean_selective_same_drugs': f(sel) / n,
                    'lineage_adjustment': (f(sl) - f(sel)) / n})
        w = np.where(okc, V, 0.0)
        f = lambda x: (w * np.where(okc, x, 0.0)).sum(1)  # noqa: E731
        n = w.sum(1)
        out.update({'covariate_share': f(ct) / f(raw), 'perfect_covariate_share': f(r2) / n,
                    'selective_capture_lin': (f(raw) - f(ct)) / (n - f(r2)),
                    'covariate_capture': f(ct) / f(r2)})
    out['median_raw'] = lib.wmedian_rows(np.where(ok, raw, np.nan), V)
    out['median_selective'] = lib.wmedian_rows(np.where(ok, sel, np.nan), V)
    out['median_selective_lin'] = lib.wmedian_rows(np.where(okl, sl, np.nan), V)
    return out, ok, okl


R2_TYPE = ('perfect_covariate_share', 'selective_capture_lin', 'covariate_capture', 'covariate_share')


def summ_basic(v, k):
    """lib.summ plus, for statistics built on R2_z|C (in-sample R^2 on ~20-30 tissue indicators, inflated by the
    duplicated cells of a bootstrap draw), the basic (reverse-percentile) interval 2 x point - percentiles, which
    corrects a constant bias; the percentile interval of these statistics can exclude the point estimate."""
    out = lib.summ(v, k)
    if k in R2_TYPE and np.isfinite(out[k]):
        out[f'{k}_lo_basic'] = 2 * out[k] - out[f'{k}_hi']
        out[f'{k}_hi_basic'] = 2 * out[k] - out[f'{k}_lo']
    return out


def summarize(bench, setting, arrs, models, drugs, ncell, V, groups, baseline, info_rows, pool_rows, per_rows, kind):
    sub = lambda i: {k: arrs[k][:, i] for k in STATS}  # noqa: E731
    base_rep = None
    if baseline in models:
        bi = models.index(baseline)
        base_rep, base_ok, _ = reps(sub(bi), V, mask_lin=False)
    for i, m in enumerate(models):
        if not np.isfinite(arrs['raw'][0, i]).any():
            continue
        st, ok, okl = reps(sub(i), V, mask_lin=m not in TISSUE_BASELINES)
        row = {'benchmark': bench, 'setting': setting, 'model': m, 'kind': kind(m), 'drugs': int(ok[0].sum()),
               'drugs_lineage_defined': int(okl[0].sum()), 'tissue_baseline': baseline if baseline in models else ''}
        for k, v in st.items():
            row.update(summ_basic(v, k))
        if base_rep is not None and m != baseline:
            # model minus the tissue-by-drug baseline (same draws; drugs scored for both)
            both = ok & base_ok
            pair = {k: arrs[k][:, [i, bi]] for k in STATS}
            for k in ('raw', 'sel'):
                x = pair[k].astype(float)
                with np.errstate(invalid='ignore', divide='ignore'):
                    w = np.where(both, V, 0.0)
                    dmean = (w * np.where(both, x[:, 0] - x[:, 1], 0.0)).sum(1) / w.sum(1)
                    dmed = lib.wmedian_rows(np.where(both, x[:, 0], np.nan), V) - lib.wmedian_rows(np.where(both, x[:, 1], np.nan), V)
                lab = {'raw': 'raw', 'sel': 'selective'}[k]
                row.update(lib.summ(dmean, f'{lab}_mean_minus_baseline'))
                row.update(lib.summ(dmed, f'{lab}_median_minus_baseline'))
            # the model's lineage-adjusted score against the baseline's (axis-only) selective score
            with np.errstate(invalid='ignore', divide='ignore'):
                bl = both & np.isfinite(arrs['sel_lin'][:, i])
                w = np.where(bl, V, 0.0)
                row.update(lib.summ((w * np.where(bl, arrs['sel_lin'][:, i] - arrs['sel'][:, bi], 0)).sum(1) / w.sum(1),
                                    'selective_lin_mean_minus_baseline_selective'))
        info_rows.append(row)
        for j, d in enumerate(drugs):
            if ok[0, j]:
                per_rows.append({'benchmark': bench, 'setting': setting, 'model': m, 'drug': d,
                                 'n_cells': int(ncell[i, j]), **{s: float(arrs[s][0, i, j]) for s in STATS}})
    for gname, gm in groups.items():
        ix = [models.index(m) for m in gm if m in models and np.isfinite(arrs['raw'][0, models.index(m)]).any()]
        if not ix:
            continue
        cat = {k: np.concatenate([arrs[k][:, i] for i in ix], axis=1) for k in STATS}
        Vc = np.concatenate([V] * len(ix), axis=1)
        st, ok, okl = reps(cat, Vc)
        prow = {'benchmark': bench, 'setting': setting, 'group': gname, 'models': len(ix),
                'model_list': ';'.join(models[i] for i in ix), 'model_drug_pairs': int(ok[0].sum())}
        for k, v in st.items():
            prow.update(summ_basic(v, k))
        if base_rep is not None:
            for k in ('mean_selective', 'median_selective', 'mean_raw', 'median_raw'):
                prow.update(lib.summ(base_rep[k], f'baseline_{k}'))
                prow.update(lib.summ(st[k] - base_rep[k], f'{k}_minus_baseline'))
            prow.update(lib.summ(st['mean_selective_lin'] - base_rep['mean_selective'],
                                 'mean_selective_lin_minus_baseline_selective'))
            # median over models of each model's median selective minus the baseline's
            meds = np.vstack([reps({k: arrs[k][:, i] for k in STATS}, V)[0]['median_selective'] for i in ix])
            prow.update(lib.summ(np.nanmedian(meds, 0) - base_rep['median_selective'],
                                 'median_over_models_median_selective_minus_baseline'))
            prow['n_models_median_selective_above_baseline'] = int((meds[:, 0] > base_rep['median_selective'][0]).sum())
        pool_rows.append(prow)


# ----------------------------------------------------------------------------------------------------------------
def kind_fn(m):
    mm = m.replace('LB:', '')
    if m in TISSUE_BASELINES:
        return 'tissue-by-drug baseline'
    if 'Naive' in mm:
        return 'naive'
    if mm in ('MultiViewRandomForest', 'FitnessOnlyRidge', 'FitnessOnlyRidgeDm', 'fitness_only', 'floor_cell_mean',
              'Lasso', 'ElasticNet'):
        return 'drug-agnostic'
    if mm.startswith(('PerDrugRidge', 'ridge_')):
        return 'this study (ridge)'
    return 'learned'


def run_leaderboard():
    L = lib.leaderboard_setup()
    cd = L['cd']
    Z, Gm = L['dec'][cd.REFERENCE]
    U = L['U']
    drugs = [d for d in Z.index if d in L['drug_ix']]
    Zu, Gu = Z.reindex(index=drugs, columns=U), Gm.reindex(index=drugs, columns=U)
    Pm = {m: P.reindex(index=drugs, columns=U) for m, P in L['P'].items()}
    V = L['V'][:, [L['drug_ix'][d] for d in drugs]]
    arrs, models, dl, ncell, ntis = engine(Zu, Gu, Pm, tissue_of_cellosaurus(), L['W'], V, mask='subset',
                                           mask_models=L['lbm'])
    agn = {'Lasso', 'ElasticNet', 'MultiViewRandomForest'}
    groups = {'published 21 (paper)': L['published21'],
              'published 21 minus drug-agnostic (18)': [m for m in L['published21'] if m not in agn]}
    return ('leaderboard', 'CTRPv2_lnIC50', arrs, models, dl, ncell, V, groups, 'NaiveTissueDrugMeanPredictor',
            {'cells': len(U), 'drugs': len(dl), 'median_tissues_per_drug': float(np.median(ntis[ntis > 0]))})


def run_retrain(setting):
    wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
    base = sorted(wp[(wp.setting == setting) & (wp.panel == 'main')].model.unique())
    rt = lib.load_module('rt59', '59_retrain_score.py')
    oof, _, _ = rt.load_oof(setting)
    extra = [m for m in DREVAL_LEARNED if m in oof and m not in base]
    S = lib.retrain_setup(setting, models=base + extra)
    O = S['obs'][S['obs'].cell.isin(S['cells'])][['drug', 'cell', 'response']]
    Z, G, _, _ = lib.selbench.decompose(O)
    Pm = {m: P.reindex(columns=Z.columns) for m, P in S['P'].items()}
    W, V = lib.draws(len(Z.columns), len(Z.index), 20261031 + sum(map(ord, setting)))     # 78's draws
    arrs, models, dl, ncell, ntis = engine(Z, G, Pm, tissue_of_cellosaurus(), W, V, mask='common')
    lm = [m for m in DREVAL_LEARNED if m in models]
    groups = {}
    if lm:
        groups['DrEval learned (retrained)'] = lm
    groups['per-drug ridges (this study)'] = [m for m in models if m.startswith('PerDrugRidge')]
    if setting == 'CTRPv2_LNIC50':
        groups['published 21 (released, as 59)'] = rt.published_pool(models)
    bl = 'NaiveTissueDrugMeanPredictor' if 'NaiveTissueDrugMeanPredictor' in models else 'LB:NaiveTissueDrugMeanPredictor'
    return ('retrain', setting, arrs, models, dl, ncell, V, groups, bl,
            {'cells': len(Z.columns), 'drugs': len(Z.index), 'models': models, 'new_learned': extra,
             'incomplete': {k: str(v) for k, v in S['incomplete'].items()},
             'median_tissues_per_drug': float(np.median(ntis[ntis > 0]))})


def tissue_drug_mean(within, tissue):
    """IMPROVE training-fold tissue x drug mean (per split, other splits' pairs as training, as 68's floor)."""
    DRUG, CELL = 'improve_chem_id', 'improve_sample_id'
    w = within.assign(t=within[CELL].map(tissue).fillna('unknown'), s=-within.auc_true)
    rows = []
    for k, test in w.groupby('split'):
        train = w[w.split != k]
        td = train.groupby([DRUG, 't']).s.mean()
        dm = train.groupby(DRUG).s.mean()
        t = test[[DRUG, CELL, 't']].copy()
        key = pd.MultiIndex.from_arrays([t[DRUG], t['t']])
        t['prediction'] = td.reindex(key).values
        t['prediction'] = t.prediction.fillna(t[DRUG].map(dm))
        rows.append(t)
    d = pd.concat(rows).dropna(subset=['prediction'])
    return d.pivot_table(index=DRUG, columns=CELL, values='prediction')


def run_improve(ds, k):
    I = lib.improve_setup(replicate_free=True, need_cross=False)
    im = I['im']
    tis, tsrc = tissue_of_depmap()
    Z, G = I['dec'][ds]
    Pm = {m: I['P_within'][(m, ds)] for m in I['models'] if (m, ds) in I['P_within']}
    Pm['floor_cell_mean'] = I['floors'][ds]
    # the same within table that 77.improve_setup used for the floor (first model with predictions)
    for model in im.MODELS:
        w = im.load_preds(ds, ds, model)
        if w is not None:
            break
    w = im.drop_split_replicates(w).sort_values('split', kind='stable').drop_duplicates([im.DRUG, im.CELL])
    Pm['tissue_drug_mean'] = tissue_drug_mean(w, tis)
    Pm = {m: P.reindex(columns=Z.columns) for m, P in Pm.items()}
    W, V = lib.draws(len(Z.columns), len(Z.index), 20261041 + k)                       # 78's draws
    arrs, models, dl, ncell, ntis = engine(Z, G, Pm, tis, W, V, mask='per_model')
    groups = {'IMPROVE models (7)': [m for m in I['models'] if m in models]}
    cs = pd.Series(Z.columns)
    return ('improve', f'{ds}_AUC', arrs, models, dl, ncell, V, groups, 'tissue_drug_mean',
            {'cells': len(Z.columns), 'drugs': len(Z.index),
             'cells_tissue_source': cs.map(tsrc).fillna('unmapped').value_counts().to_dict(),
             'median_tissues_per_drug': float(np.median(ntis[ntis > 0]))})


def task(t):
    kind, arg = t
    if kind == 'leaderboard':
        res = run_leaderboard()
    elif kind == 'retrain':
        res = run_retrain(arg)
    else:
        res = run_improve(*arg)
    bench, setting, arrs, models, dl, ncell, V, groups, bl, info = res
    rows, prow, per = [], [], []
    summarize(bench, setting, arrs, models, dl, ncell, V, groups, bl, rows, prow, per, kind_fn)
    return setting, rows, prow, per, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='*', default=['leaderboard', 'retrain', 'improve'])
    ap.add_argument('--settings', nargs='*', default=['GDSC2_AUC', 'CTRPv2_LNIC50', 'GDSC2_LNIC50'])
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    t0 = time.time()
    tasks = []
    if 'leaderboard' in args.parts:
        tasks.append(('leaderboard', None))
    if 'retrain' in args.parts:
        tasks += [('retrain', s) for s in args.settings]
    if 'improve' in args.parts:
        tasks += [('improve', (ds, k)) for k, ds in enumerate(['CCLE', 'CTRPv2', 'GDSCv1', 'GDSCv2', 'gCSI'])]
    rows, prow, per, info = [], [], [], {}
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context('fork')) as ex:
        for setting, r, p, pr, inf in ex.map(task, tasks):
            rows += r
            prow += p
            per += pr
            info[setting] = inf
            print('done', setting, f'{time.time() - t0:.0f}s', flush=True)
    tag = args.tag
    M = pd.DataFrame(rows)
    P = pd.DataFrame(prow)
    D = pd.DataFrame(per)
    # merge with earlier runs of other settings (so that partial reruns update in place)
    for df, fn in [(M, f'lineage_by_model{tag}.csv'), (P, f'lineage_pooled{tag}.csv'),
                   (D, f'per_drug_lineage{tag}.csv.gz')]:
        f = OUT / fn
        if f.exists() and len(df):
            old = pd.read_csv(f)
            old = old[~old.setting.isin(df.setting.unique())]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(f, index=False, float_format='%.5g' if not fn.endswith('.gz') else '%.6g')
    runf = OUT / f'run{tag}.json'
    allinfo = {}
    if runf.exists():
        import json
        allinfo = json.loads(runf.read_text()).get('info', {})
    allinfo.update(info)
    lib.write_json(runf, {'seconds_last': round(time.time() - t0), 'B': lib.B, 'ridge': RIDGE, 'info': allinfo})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 400)
    cols = ['benchmark', 'setting', 'group', 'models', 'median_selective', 'median_selective_lin', 'mean_selective',
            'mean_selective_lin', 'lineage_adjustment', 'baseline_median_selective', 'median_selective_minus_baseline',
            'selective_capture', 'selective_capture_lin', 'covariate_share', 'perfect_covariate_share']
    print(P[[c for c in cols if c in P]].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
