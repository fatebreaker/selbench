"""Revision workstream: paired per-drug scores of identical predictions under every processing of every target, and
a two-way (drug x cell) bootstrap of all summaries.

Design (R1-M1/M2/M3, R2 minor 6, R3-M4b/e). DrEval's leaderboard models were trained on CTRPv2 CurveCurator ln IC50
(leave-cell-line-out). For each pair (reference version S, target version T) and each drug, on the cell lines
measured in both S and T (matched cells, >= 30) and predicted by every model, the SAME out-of-fold prediction p is
scored against S ("within") and T ("cross"). Every version is decomposed separately with selbench.decompose
(rank-normal profiles z and leave-drug-out axis g). Per model and drug, for within and cross:

  raw        r(p, z)
  r_pg, r_zg alignment of p with the evaluation axis, loading of z on it; mediated = r_pg r_zg;
             selective term = raw - mediated (exact split)
  selT       partial r(p, z | g_T)                     (the paper's selective score)
  selBOTH    partial r(p, z | g_T, g_S)                (model-independent; equals selT when T = S)
  adj        partial r(p, z | g_T, g_hat_model)        (diagnostic; g_hat = leave-drug-out mean of the model's
             per-drug z-scored predictions; undefined for exactly drug-agnostic predictors)
  lin        partial r(p, z | g_T, tissue indicators)  (tissues with >= 5 matched cells; largest tissue = baseline)
  lcoT/lcoB  selT / selBOTH with leave-class-out axes (mechanism classes, strict mapping: id_maps + drug_table.csv)
  lcxT/lcxB  the same with the extended class mapping (name fallback + 12_drug_classes.classify)
  spr/spsel  Spearman versions of raw and selT (Pearson / partial Pearson on ranks)

Pairs: reference CTRPv2 CurveCurator ln IC50 against all 17 versions (own matched cells; and 'dataset-common': the
cells and drugs shared by all versions of a dataset, so processing is the only difference), plus reference CTRPv2
CurveCurator AUC against every dataset's CurveCurator AUC (uncensored axes; censoring control).
Pseudo-predictors: CTRPv2's measured response z_S (one-replicate reference) and its leave-drug-out axis g_S.

Two-way bootstrap: cell lines and drugs are resampled independently with replacement (multinomial counts over the
cells predicted by every model and over the CTRPv2 drug panel), B = 1,000 draws, seed 20261014. The same draws are
used for every pair, model and subset, so all differences are paired. Per-drug statistics are recomputed under
each cell resample from weighted moments (the rank-normal profiles, axes and ranks are held at their full-sample
values); summaries aggregate over the resampled drugs. Draw 0 uses unit weights and gives the point estimates.

Outputs (results/revision_crosslab/): per_drug_scores.csv.gz, pair_info.csv, bootstrap_pooled.parquet (replicates of
pooled sums over the 21 published models, excluding SVR), bootstrap_medmed.parquet (replicates of the median over
published models of per-model drug medians), model_medians.csv, paired_boot.csv, axis_diagnostics.csv.
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
from scipy.stats import rankdata  # noqa: E402

from common import write_json  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cd = load('cd41', '41_crosslab_data.py')
rs, selbench, OUT = cd.rs, cd.selbench, cd.OUT
SEED = 20261014
MIN_CELLS = 30
MIN_TISSUE = 5
TOL = 1e-10
MEASURED, FITNESS = 'ref_CTRPv2_measured', 'ref_CTRPv2_fitness_ldo'
OURS = 'ridge_sel_lm'
STAT_PAIRS = [('raw', 'raw_w', 'raw_c'), ('mediated', 'med_w', 'med_c'), ('selective_term', 'selterm_w', 'selterm_c'),
              ('r_pg', 'rpg_w', 'rpg_c'), ('r_zg', 'rzg_w', 'rzg_c'), ('selT', 'selT_w', 'selT_c'),
              ('selBOTH', 'selT_w', 'selB_c'), ('adjusted', 'adj_w', 'adj_c'), ('lineage', 'lin_w', 'lin_c'),
              ('lcoT', 'lcoT_w', 'lcoT_c'), ('lcoBOTH', 'lcoT_w', 'lcoB_c'), ('lcxT', 'lcxT_w', 'lcxT_c'),
              ('lcxBOTH', 'lcxT_w', 'lcxB_c'), ('spearman_raw', 'spr_w', 'spr_c'), ('spearman_selT', 'spsel_w', 'spsel_c')]
STATS = ['raw_w', 'raw_c', 'rpg_w', 'rpg_c', 'rzg_w', 'rzg_c', 'med_w', 'med_c', 'selterm_w', 'selterm_c',
         'selT_w', 'selT_c', 'selB_c', 'adj_w', 'adj_c', 'lin_w', 'lin_c', 'lcoT_w', 'lcoT_c', 'lcoB_c',
         'lcxT_w', 'lcxT_c', 'lcxB_c', 'spr_w', 'spr_c', 'spsel_w', 'spsel_c']
MED_STATS = [s for s in STATS if s not in ('selterm_w', 'selterm_c')]
G = {}


# ----------------------------------------------------------------------------------------------- set-up helpers
def leave_class_out(Z, Gm, classes):
    """Axis excluding every drug of the scored drug's class (unclassified drugs keep the leave-drug-out axis)."""
    v = Z.values
    ok = np.isfinite(v)
    tot, cnt = np.nansum(v, axis=0), ok.sum(axis=0)
    out = Gm.values.copy()
    lab = pd.Series(Z.index.map(classes), index=Z.index)
    for k in lab.dropna().unique():
        rows = np.where(lab.values == k)[0]
        tk, ck = np.nansum(v[rows], axis=0), ok[rows].sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            g = np.where(cnt - ck > 0, (tot - tk) / (cnt - ck), np.nan)
        out[rows] = np.where(np.isfinite(g)[None], g[None], Gm.values[rows])
    return pd.DataFrame(out, index=Z.index, columns=Z.columns), int(lab.notna().sum())


def model_axis(P):
    """Leave-drug-out mean of the per-drug z-scored predictions (the model's own fitness axis)."""
    v = P.values.astype(float)
    ok = np.isfinite(v)
    mu = np.nanmean(v, axis=1, keepdims=True)
    sd = np.nanstd(v, axis=1, ddof=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        zp = np.where(sd > 1e-12, (v - mu) / sd, np.nan)
    ok = np.isfinite(zp)
    tot, cnt = np.nansum(zp, axis=0), ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        gh = (tot[None] - np.where(ok, zp, 0)) / (cnt[None] - ok)
    return pd.DataFrame(gh, index=P.index, columns=P.columns)


def std(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.full_like(x, np.nan)


# ----------------------------------------------------------------------------------------------- algebra
def corr(c, va, vb):
    with np.errstate(invalid='ignore', divide='ignore'):
        return c / np.sqrt(va * vb)


def pcor1(vx, vy, cxy, cx, cy, vc):
    """Partial correlation of x and y given one covariate, from (co)variances (broadcasting)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        rxx, ryy, rxy = vx - cx * cx / vc, vy - cy * cy / vc, cxy - cx * cy / vc
        out = rxy / np.sqrt(rxx * ryy)
    return np.where((rxx > TOL * vx) & (ryy > TOL * vy), out, np.nan)


def pcor2(vx, vy, cxy, cx1, cx2, cy1, cy2, v1, v2, c12):
    """Partial correlation of x and y given two covariates."""
    with np.errstate(invalid='ignore', divide='ignore'):
        det = v1 * v2 - c12 * c12
        qxx = (cx1 * cx1 * v2 - 2 * cx1 * cx2 * c12 + cx2 * cx2 * v1) / det
        qyy = (cy1 * cy1 * v2 - 2 * cy1 * cy2 * c12 + cy2 * cy2 * v1) / det
        qxy = (cx1 * cy1 * v2 - (cx1 * cy2 + cx2 * cy1) * c12 + cx2 * cy2 * v1) / det
        rxx, ryy, rxy = vx - qxx, vy - qyy, cxy - qxy
        out = rxy / np.sqrt(rxx * ryy)
    return np.where((det > 1e-9 * v1 * v2) & (rxx > TOL * vx) & (ryy > TOL * vy), out, np.nan)


def pcor_k(vx, vy, cxy, cxC, cyC, SCC):
    """Partial correlation given k covariates. vx, cxy: (B, M); vy: (B,); cxC: (B, M, k); cyC: (B, k); SCC: (B, k, k)."""
    k = SCC.shape[-1]
    lam = 1e-10 * np.trace(SCC, axis1=1, axis2=2) / k
    inv = np.linalg.inv(SCC + lam[:, None, None] * np.eye(k)[None])
    a = np.einsum('bij,bj->bi', inv, cyC)
    bx = np.einsum('bij,bmj->bmi', inv, cxC)
    rxy = cxy - np.einsum('bmi,bi->bm', cxC, a)
    rxx = vx - np.einsum('bmi,bmi->bm', cxC, bx)
    ryy = (vy - np.einsum('bi,bi->b', cyC, a))[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = rxy / np.sqrt(rxx * ryy)
    return np.where((rxx > TOL * vx) & (ryy > TOL * vy[:, None]), out, np.nan)


def drug_stats(Wd, sh, D, Pm, GHm, identity):
    """All per-drug statistics for every bootstrap draw (rows of Wd) and model (columns of Pm).

    sh: dict of shared n-vectors (zs, gs, zt, gt, gsl, gtl, gsx, gtx, rzs, rgs, rzt, rgt); D: n x K tissue
    indicators; Pm, GHm: n x M standardized predictions and model axes (rank columns are derived here).
    """
    n, M = Pm.shape
    K = D.shape[1]
    RPm = np.column_stack([std(rankdata(Pm[:, j])) for j in range(M)])
    names = ['zs', 'gs', 'zt', 'gt', 'gsl', 'gtl', 'gsx', 'gtx', 'rzs', 'rgs', 'rzt', 'rgt']
    S = np.column_stack([sh[k] for k in names])
    ix = {k: i for i, k in enumerate(names)}
    sp = [('zs', 'zs'), ('gs', 'gs'), ('zs', 'gs'), ('zt', 'zt'), ('gt', 'gt'), ('zt', 'gt'), ('gs', 'zt'), ('gs', 'gt'),
          ('gsl', 'gsl'), ('zs', 'gsl'), ('gtl', 'gtl'), ('zt', 'gtl'), ('gtl', 'gsl'), ('zt', 'gsl'),
          ('gsx', 'gsx'), ('zs', 'gsx'), ('gtx', 'gtx'), ('zt', 'gtx'), ('gtx', 'gsx'), ('zt', 'gsx'),
          ('rzs', 'rzs'), ('rgs', 'rgs'), ('rzs', 'rgs'), ('rzt', 'rzt'), ('rgt', 'rgt'), ('rzt', 'rgt')]
    spi = {p: j for j, p in enumerate(sp)}
    SP = np.column_stack([S[:, ix[a]] * S[:, ix[b]] for a, b in sp])
    DZ = np.concatenate([D * S[:, [ix[k]]] for k in ('zs', 'gs', 'zt', 'gt')], axis=1)
    p8 = ['zs', 'gs', 'zt', 'gt', 'gsl', 'gtl', 'gsx', 'gtx']
    blocks = [S, D, SP, DZ, Pm, GHm, RPm, Pm * Pm, GHm * GHm, RPm * RPm, Pm * GHm]
    blocks += [Pm * S[:, [ix[k]]] for k in p8]
    blocks += [(Pm[:, :, None] * D[:, None, :]).reshape(n, M * K)]
    blocks += [GHm * S[:, [ix[k]]] for k in ('zs', 'gs', 'zt', 'gt')]
    blocks += [RPm * S[:, [ix[k]]] for k in ('rzs', 'rgs', 'rzt', 'rgt')]
    Q = np.concatenate(blocks, axis=1)
    E = (Wd @ Q) / Wd.sum(axis=1, keepdims=True)
    o = 0

    def take(w):
        nonlocal o
        out = E[:, o:o + w]
        o += w
        return out
    es, ed, esp, edz = take(12), take(K), take(len(sp)), take(4 * K)
    ep, eg, er, epp, egg, err, epg = take(M), take(M), take(M), take(M), take(M), take(M), take(M)
    epx = {k: take(M) for k in p8}
    epd = take(M * K).reshape(E.shape[0], M, K)
    egx = {k: take(M) for k in ('zs', 'gs', 'zt', 'gt')}
    erx = {k: take(M) for k in ('rzs', 'rgs', 'rzt', 'rgt')}
    m1 = {k: es[:, ix[k]] for k in names}

    def cs(a, b):  # shared covariance (B,)
        key = (a, b) if (a, b) in spi else (b, a)
        return esp[:, spi[key]] - m1[a] * m1[b]
    vp = epp - ep * ep
    vg = egg - eg * eg
    vr = err - er * er
    cpg_h = epg - ep * eg
    cp = {k: epx[k] - ep * m1[k][:, None] for k in p8}
    cg = {k: egx[k] - eg * m1[k][:, None] for k in ('zs', 'gs', 'zt', 'gt')}
    cr = {k: erx[k] - er * m1[k][:, None] for k in ('rzs', 'rgs', 'rzt', 'rgt')}
    out = {}
    for side, z, g, gl, gx, rz, rg in [('w', 'zs', 'gs', 'gsl', 'gsx', 'rzs', 'rgs'), ('c', 'zt', 'gt', 'gtl', 'gtx', 'rzt', 'rgt')]:
        vz, vgg_, czg = cs(z, z)[:, None], cs(g, g)[:, None], cs(z, g)[:, None]
        raw = corr(cp[z], vp, vz)
        rpg = corr(cp[g], vp, vgg_)
        rzg = np.broadcast_to(corr(czg, vz, vgg_), raw.shape)
        out[f'raw_{side}'], out[f'rpg_{side}'], out[f'rzg_{side}'] = raw, rpg, rzg
        out[f'med_{side}'] = rpg * rzg
        out[f'selterm_{side}'] = raw - rpg * rzg
        out[f'selT_{side}'] = pcor1(vp, vz, cp[z], cp[g], czg, vgg_)
        # model-axis adjusted: covariates (g, g_hat)
        out[f'adj_{side}'] = pcor2(vp, vz, cp[z], cp[g], cpg_h, czg, cg[z], vgg_, vg, cg[g])
        # leave-class-out axes (strict, extended)
        for tag, gg in [('lco', gl), ('lcx', gx)]:
            out[f'{tag}T_{side}'] = pcor1(vp, vz, cp[z], cp[gg], cs(z, gg)[:, None], cs(gg, gg)[:, None])
        # lineage: covariates (g, tissue indicators)
        if K:
            mD = ed
            SDD = np.einsum('bk,kl->bkl', mD, np.eye(K)) - mD[:, :, None] * mD[:, None, :]
            j = {'zs': 0, 'gs': 1, 'zt': 2, 'gt': 3}
            cDz = edz[:, j[z] * K:(j[z] + 1) * K] - mD * m1[z][:, None]
            cDg = edz[:, j[g] * K:(j[g] + 1) * K] - mD * m1[g][:, None]
            SCC = np.empty((E.shape[0], K + 1, K + 1))
            SCC[:, 0, 0] = cs(g, g)
            SCC[:, 0, 1:] = cDg
            SCC[:, 1:, 0] = cDg
            SCC[:, 1:, 1:] = SDD
            cpD = epd - ep[:, :, None] * mD[:, None, :]
            cxC = np.concatenate([cp[g][:, :, None], cpD], axis=2)
            cyC = np.concatenate([cs(z, g)[:, None], cDz], axis=1)
            out[f'lin_{side}'] = pcor_k(vp, cs(z, z), cp[z], cxC, cyC, SCC)
        else:
            out[f'lin_{side}'] = out[f'selT_{side}']
        vrz, vrg, crzg = cs(rz, rz)[:, None], cs(rg, rg)[:, None], cs(rz, rg)[:, None]
        out[f'spr_{side}'] = corr(cr[rz], vr, vrz)
        out[f'spsel_{side}'] = pcor1(vr, vrz, cr[rz], cr[rg], crzg, vrg)
    if identity:
        out['selB_c'], out['lcoB_c'], out['lcxB_c'] = out['selT_c'], out['lcoT_c'], out['lcxT_c']
    else:
        vz = cs('zt', 'zt')[:, None]
        for name, a, b in [('selB_c', 'gt', 'gs'), ('lcoB_c', 'gtl', 'gsl'), ('lcxB_c', 'gtx', 'gsx')]:
            out[name] = pcor2(vp, vz, cp['zt'], cp[a], cp[b], cs('zt', a)[:, None], cs('zt', b)[:, None],
                              cs(a, a)[:, None], cs(b, b)[:, None], cs(a, b)[:, None])
    return out


# ----------------------------------------------------------------------------------------------- aggregation
def wmedian(X, w):
    """Weighted median over the last axis with integer replicate weights (X: (B, M, D), w: (B, D)) -> (B, M)."""
    W = np.broadcast_to(w[:, None, :], X.shape)
    ok = np.isfinite(X) & (W > 0)
    Xf = np.where(ok, X, np.inf)
    order = np.argsort(Xf, axis=-1, kind='stable')
    Xs = np.take_along_axis(Xf, order, -1)
    cw = np.cumsum(np.take_along_axis(np.where(ok, W, 0), order, -1), axis=-1)
    tot = np.rint(cw[..., -1]).astype(np.int64)
    k1 = np.where(tot % 2 == 1, (tot + 1) // 2, tot // 2)
    k2 = np.where(tot % 2 == 1, (tot + 1) // 2, tot // 2 + 1)
    last = X.shape[-1] - 1
    i1 = np.minimum((cw < k1[..., None] - 0.5).sum(-1), last)
    i2 = np.minimum((cw < k2[..., None] - 0.5).sum(-1), last)
    v = 0.5 * (np.take_along_axis(Xs, i1[..., None], -1)[..., 0] + np.take_along_axis(Xs, i2[..., None], -1)[..., 0])
    return np.where(tot > 0, v, np.nan)


def wmean(X, w):
    W = np.broadcast_to(w[:, None, :], X.shape)
    ok = np.isfinite(X)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.where(ok, X, 0) * W).sum(-1) / np.where(ok, W, 0).sum(-1)


def ci(rep):
    rep = np.asarray(rep, float)
    b = rep[1:]
    b = b[np.isfinite(b)]
    if len(b) < 10:
        return np.nan, np.nan
    return tuple(np.percentile(b, [2.5, 97.5]))


def p_boot(rep):
    b = np.asarray(rep, float)[1:]
    b = b[np.isfinite(b)]
    if len(b) < 10:
        return np.nan
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))


# ----------------------------------------------------------------------------------------------- worker
def run_pair(pr):
    t0 = time.time()
    ref, tgt, cellset, name = pr['reference'], pr['target'], pr['cellset'], pr['pair']
    dec = G['dec']
    identity = ref == tgt
    U = G['cells']
    S_, T_ = dec[ref], dec[tgt]
    drugs = [d for d in S_['Z'].index.intersection(T_['Z'].index) if d in G['drug_ix']]
    common = G['common'][tgt.split('|')[0]] if cellset == 'dataset-common' else None
    models = G['models'] + [MEASURED, FITNESS]
    pub = [i for i, m in enumerate(models) if m in G['published21']]
    lbm = [m for m in G['models'] if m in G['leaderboard']]
    pseudo = {MEASURED: S_['Z'].reindex(columns=U), FITNESS: S_['G'].reindex(columns=U)}
    mats = {k: (V_[key].reindex(columns=U) if key in V_ else None)
            for k, (V_, key) in {'zs': (S_, 'Z'), 'gs': (S_, 'G'), 'gsl': (S_, 'GL'), 'gsx': (S_, 'GX'),
                                 'zt': (T_, 'Z'), 'gt': (T_, 'G'), 'gtl': (T_, 'GL'), 'gtx': (T_, 'GX')}.items()}
    if common is not None:
        common = common.reindex(columns=U).fillna(False).astype(bool)
    pseudo_axes = G['pseudo_axes'][ref]
    B1 = G['W'].shape[0]
    M = len(models)
    arrs = {s: np.full((B1, M, len(drugs)), np.nan, np.float32) for s in STATS}
    keep, rows, dropped, noaxis = [], [], 0, 0
    used_cells = np.zeros(len(U), bool)
    for di, d in enumerate(drugs):
        cols = {k: m_.loc[d].values.astype(float) for k, m_ in mats.items()}
        m0 = np.all([np.isfinite(v) for v in cols.values()], axis=0)
        if common is not None:
            m0 &= common.loc[d].values if d in common.index else False
        for m in lbm:
            P = G['P'][m]
            if d in P.index:
                m0 &= np.isfinite(P.loc[d].values)
        if m0.sum() < MIN_CELLS:
            continue
        idx = np.where(m0)[0]
        n = len(idx)
        use, pcols, gcols = [], [], []
        for j, m in enumerate(models):
            if m in pseudo:
                p = pseudo[m].loc[d].values[idx]
                gh = pseudo_axes[m].loc[d].values[idx]
            else:
                P = G['P'][m]
                if d not in P.index:
                    continue
                p = P.loc[d].values[idx]
                gh = G['axes'][m].loc[d].values[idx] if d in G['axes'][m].index else np.full(n, np.nan)
            if not np.all(np.isfinite(p)) or np.std(p) < 1e-12:
                dropped += int(np.isfinite(p).any())
                continue
            if not np.all(np.isfinite(gh)) or np.std(gh) < 1e-12:
                gh = np.full(n, np.nan)  # NaN propagates to the model-axis-adjusted score of this model only
                noaxis += 1
            use.append(j)
            pcols.append(std(p))
            gcols.append(std(gh))
        sh = {k: std(v[idx]) for k, v in cols.items()}
        for k in ['zs', 'gs', 'zt', 'gt']:
            sh['r' + k] = std(rankdata(cols[k][idx]))
        tis = G['tissue'].reindex(U[idx]).fillna('unknown').values
        vc = pd.Series(tis).value_counts()
        qual = [t for t in vc.index[vc >= MIN_TISSUE]][1:]          # drop the most frequent (baseline)
        D = np.column_stack([(tis == t).astype(float) for t in qual]) if qual else np.zeros((n, 0))
        Wd = G['W'][:, idx]
        st = drug_stats(Wd, sh, D, np.column_stack(pcols), np.column_stack(gcols), identity)
        for s in STATS:
            arrs[s][:, use, di] = st[s]
        for jj, j in enumerate(use):
            m = models[j]
            if m in G['agnostic']:
                arrs['adj_w'][:, j, di] = np.nan
                arrs['adj_c'][:, j, di] = np.nan
            row = {'pair': name, 'model': m, 'drug': d, 'n_cells': n, 'n_tissue_terms': len(qual)}
            row.update({s: float(st[s][0, jj]) for s in STATS})
            row['adj_literal_w'], row['adj_literal_c'] = row['adj_w'], row['adj_c']
            row['adj_w'] = float(arrs['adj_w'][0, j, di])
            row['adj_c'] = float(arrs['adj_c'][0, j, di])
            rows.append(row)
        keep.append(di)
        used_cells[idx] = True
    drugs = [drugs[i] for i in keep]
    arrs = {s: a[:, :, keep] for s, a in arrs.items()}
    per = pd.DataFrame(rows)
    dix = np.array([G['drug_ix'][d] for d in drugs], dtype=int)
    Vall = G['V'][:, dix]
    act = np.array([d in G['active'] for d in drugs])
    subsets = {'all': Vall, 'active': Vall * act[None, :]}
    pooled, medmed, msum, comp = [], [], [], []
    for sub, Vw in subsets.items():
        if Vw[0].sum() < 5:
            continue
        # (A) pooled sums over the 21 published models (paired mask: within and cross both defined)
        for stat, a, b in STAT_PAIRS:
            xa, xb = arrs[a][:, pub, :].astype(float), arrs[b][:, pub, :].astype(float)
            ok = np.isfinite(xa) & np.isfinite(xb)
            for kind, val in [('sum_w', np.einsum('bmd,bd->b', np.where(ok, xa, 0), Vw)),
                              ('sum_c', np.einsum('bmd,bd->b', np.where(ok, xb, 0), Vw)),
                              ('n', np.einsum('bmd,bd->b', ok.astype(float), Vw))]:
                pooled.append((name, sub, stat, kind, val))
        # model-independent oracle terms: mean over drugs of r_zg^2 (perfect predictor's mediated share)
        rz = {s: arrs[f'rzg_{s}'][:, models.index(MEASURED), :].astype(float) for s in ('w', 'c')}
        okz = np.isfinite(rz['w']) & np.isfinite(rz['c'])
        for kind, val in [('sum_w', np.einsum('bd,bd->b', np.where(okz, rz['w'] ** 2, 0), Vw)),
                          ('sum_c', np.einsum('bd,bd->b', np.where(okz, rz['c'] ** 2, 0), Vw)),
                          ('n', np.einsum('bd,bd->b', okz.astype(float), Vw))]:
            pooled.append((name, sub, 'oracle_rzg2', kind, val))
        # (B, C) per-model weighted medians and means over drugs; median over published models
        for s in MED_STATS:
            X = arrs[s]
            med = wmedian(X, Vw)
            mea = wmean(X.astype(float), Vw)
            medmed.append((name, sub, s, np.nanmedian(med[:, pub], axis=1)))
            for j, m in enumerate(models):
                if not np.isfinite(X[0, j]).any():
                    continue
                lo, hi = ci(med[:, j])
                mlo, mhi = ci(mea[:, j])
                msum.append({'pair': name, 'subset': sub, 'model': m, 'stat': s,
                             'drugs': int((np.isfinite(X[0, j]) & (Vw[0] > 0)).sum()),
                             'median': float(med[0, j]), 'median_lo': lo, 'median_hi': hi,
                             'mean': float(mea[0, j]), 'mean_lo': mlo, 'mean_hi': mhi})
        # (E) paired comparisons: weighted median over drugs of per-drug differences
        specs = [('selBOTH', OURS, 'selB_c', m, 'selB_c') for m in G['published22']]
        specs += [('adjusted', OURS, 'adj_c', m, 'adj_c') for m in G['published22']]
        specs += [('selT', OURS, 'selT_c', m, 'selT_c') for m in G['published22']]
        specs += [('own_axis_removed_raw_ridge', OURS, 'adj_c', 'ridge_raw_lm', 'adj_c'),
                  ('own_axis_removed_raw_ridge_vs_selT', OURS, 'selT_c', 'ridge_raw_lm', 'adj_c'),
                  ('own_axis_removed_raw_ridge_vs_selBOTH', OURS, 'selB_c', 'ridge_raw_lm', 'adj_c')]
        for metric, a, sa, b, sb in specs:
            if b not in models:
                continue
            ja, jb = models.index(a), models.index(b)
            diff = (arrs[sa][:, ja, :].astype(float) - arrs[sb][:, jb, :].astype(float))[:, None, :]
            if np.isfinite(diff[0, 0]).sum() < 10:
                continue
            md = wmedian(diff, Vw)[:, 0]
            mn = wmean(diff, Vw)[:, 0]
            lo, hi = ci(md)
            mlo, mhi = ci(mn)
            ok0 = np.isfinite(diff[0, 0]) & (Vw[0] > 0)
            comp.append({'pair': name, 'subset': sub, 'metric': metric, 'ours': a, 'ours_stat': sa, 'model': b,
                         'model_stat': sb, 'drugs': int(ok0.sum()), 'median_diff': float(md[0]), 'median_lo': lo,
                         'median_hi': hi, 'p_boot_median': p_boot(md), 'mean_diff': float(mn[0]), 'mean_lo': mlo,
                         'mean_hi': mhi, 'p_boot_mean': p_boot(mn), 'frac_better': float((diff[0, 0][ok0] > 0).mean())})
    info = {'pair': name, 'reference': ref, 'target': tgt, 'cellset': cellset, 'family': pr['family'],
            'drugs': len(drugs), 'active_drugs': int(act.sum()), 'cells_union': int(used_cells.sum()),
            'median_cells_per_drug': float(per.groupby('drug').n_cells.first().median()) if len(per) else np.nan,
            'model_drug_dropped_incomplete': dropped, 'model_drug_without_axis': noaxis,
            'seconds': round(time.time() - t0, 1)}
    return per, pooled, medmed, msum, comp, info


# ----------------------------------------------------------------------------------------------- main
def build_pairs(vt):
    L, A = cd.REFERENCE, cd.AUC_REFERENCE
    pairs = [{'pair': f'{L}->{v}', 'reference': L, 'target': v, 'cellset': 'own', 'family': 'lnIC50-reference'}
             for v in vt.version]
    pairs += [{'pair': f'{L}->{v}@common', 'reference': L, 'target': v, 'cellset': 'dataset-common',
               'family': 'lnIC50-reference'} for v in vt.version]
    pairs += [{'pair': f'{A}->{ds}|CC|AUC', 'reference': A, 'target': f'{ds}|CC|AUC', 'cellset': 'own',
               'family': 'AUC-reference'} for ds in cd.DATASETS]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--only', default=None, help='comma-separated pair names (testing)')
    ap.add_argument('--out', default=str(OUT), help='output directory (testing)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert args.workers <= 6
    t0 = time.time()
    vt = pd.read_csv(OUT / 'versions.csv')
    cls = pd.read_csv(OUT / 'drug_classes.csv', dtype={'drug': str})
    strict = cls.set_index('drug').lco_class_strict.dropna()
    ext = cls.set_index('drug').lco_class_extended.dropna()
    tissue = pd.read_csv(OUT / 'cell_tissue.csv').set_index('cellosaurus_id').tissue
    act = pd.read_csv(OUT / 'drug_activity_ctrpv2.csv', dtype={'drug': str})
    ag = pd.read_csv(OUT / 'drug_agnostic_check.csv')
    V = cd.versions()
    dec, diag = {}, []
    for v in vt.version:
        Z, Gm, _, r2 = selbench.decompose(V[v])
        GL, nl = leave_class_out(Z, Gm, strict)
        GX, nx = leave_class_out(Z, Gm, ext)
        dec[v] = {'Z': Z, 'G': Gm, 'GL': GL, 'GX': GX}
        row = {'version': v, 'drugs': len(Z), 'cells': Z.shape[1], 'median_fitness_r2': float(r2.median()),
               'classed_strict': nl, 'classed_extended': nx}
        for tag, GG, cmap in [('strict', GL, strict), ('extended', GX, ext)]:
            cl = []
            for d in Z.index.intersection(cmap.index):
                a, b = Gm.loc[d].values, GG.loc[d].values
                ok = np.isfinite(a) & np.isfinite(b)
                cl.append(np.corrcoef(a[ok], b[ok])[0, 1])
            row[f'median_r_ldo_vs_lco_axis_{tag}_classed'] = float(np.median(cl)) if cl else np.nan
        diag.append(row)
        print(v, Z.shape, f'{time.time() - t0:.0f}s', flush=True)
    P, mclass, lb = cd.predictions()
    cells = None
    for M_ in P.values():
        cells = M_.columns if cells is None else cells.intersection(M_.columns)
    U = np.array(sorted(cells))
    P = {m: M_.reindex(columns=U) for m, M_ in P.items()}
    axes = {m: model_axis(M_.loc[M_.notna().sum(axis=1) >= MIN_CELLS]) for m, M_ in P.items()}
    pseudo_axes = {}
    for ref in [cd.REFERENCE, cd.AUC_REFERENCE]:
        Zr, Gr = dec[ref]['Z'].reindex(columns=U), dec[ref]['G'].reindex(columns=U)
        pseudo_axes[ref] = {MEASURED: model_axis(Zr), FITNESS: model_axis(Gr)}
    # dataset-common masks: drug x cell finite in every version of a dataset
    common = {}
    for ds in cd.DATASETS:
        mk = None
        for v in vt[vt.dataset == ds].version:
            f = dec[v]['Z'].notna() & dec[v]['G'].notna()
            if mk is None:
                mk = f
            else:
                ri, ci_ = mk.index.intersection(f.index), mk.columns.intersection(f.columns)
                mk = mk.loc[ri, ci_] & f.loc[ri, ci_]
        common[ds] = mk
    all_drugs = sorted(dec[cd.REFERENCE]['Z'].index.union(dec[cd.AUC_REFERENCE]['Z'].index))
    rng = np.random.default_rng(SEED)
    B = args.draws
    Wc = np.vstack([np.ones(len(U)), rng.multinomial(len(U), np.full(len(U), 1 / len(U)), size=B)]).astype(float)
    Vd = np.vstack([np.ones(len(all_drugs)), rng.multinomial(len(all_drugs), np.full(len(all_drugs), 1 / len(all_drugs)),
                                                              size=B)]).astype(float)
    lbset = set(lb)
    published22 = [m for m in P if mclass[m] in ('classical', 'neural/factorization')]
    G.update({'dec': dec, 'cells': U, 'P': P, 'axes': axes, 'pseudo_axes': pseudo_axes, 'common': common,
              'tissue': tissue, 'W': Wc, 'V': Vd, 'drug_ix': {d: i for i, d in enumerate(all_drugs)},
              'active': set(act[act.active_ctrpv2].drug), 'models': list(P), 'leaderboard': lbset,
              'published22': published22, 'published21': [m for m in published22 if m != 'SVR'],
              'agnostic': set(ag[ag.exactly_drug_agnostic].model) | {FITNESS}})
    pairs = build_pairs(vt)
    if args.only:
        keepn = set(args.only.split(','))
        pairs = [p for p in pairs if p['pair'] in keepn]
    # heavy (CTRPv2-panel) pairs first for load balance
    pairs = sorted(pairs, key=lambda p: (not p['target'].startswith('CTRPv2'), p['pair']))
    print(f'set-up done {time.time() - t0:.0f}s; cells {len(U)}, drugs {len(all_drugs)}, pairs {len(pairs)}, '
          f'models {len(P) + 2}, draws {B}', flush=True)
    per, pooled, medmed, msum, comp, info = [], [], [], [], [], []
    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as ex:
        for r in ex.map(run_pair, pairs):
            per.append(r[0]); pooled += r[1]; medmed += r[2]; msum += r[3]; comp += r[4]; info.append(r[5])
            print(r[5], f'{time.time() - t0:.0f}s', flush=True)
    per = pd.concat(per, ignore_index=True)
    per['class'] = per.model.map(lambda m: mclass.get(m, 'reference'))
    per['active_ctrpv2'] = per.drug.isin(G['active'])
    per.to_csv(out / 'per_drug_scores.csv.gz', index=False, float_format='%.6g')
    pd.DataFrame(info).to_csv(out / 'pair_info.csv', index=False)
    nd = B + 1
    rep = pd.DataFrame({'pair': np.repeat([t[0] for t in pooled], nd), 'subset': np.repeat([t[1] for t in pooled], nd),
                        'stat': np.repeat([t[2] for t in pooled], nd), 'kind': np.repeat([t[3] for t in pooled], nd),
                        'draw': np.tile(np.arange(nd), len(pooled)), 'value': np.concatenate([t[4] for t in pooled])})
    rep.to_parquet(out / 'bootstrap_pooled.parquet', index=False)
    mm = pd.DataFrame({'pair': np.repeat([t[0] for t in medmed], nd), 'subset': np.repeat([t[1] for t in medmed], nd),
                       'stat': np.repeat([t[2] for t in medmed], nd), 'draw': np.tile(np.arange(nd), len(medmed)),
                       'value': np.concatenate([t[3] for t in medmed])})
    mm.to_parquet(out / 'bootstrap_medmed.parquet', index=False)
    pd.DataFrame(msum).to_csv(out / 'model_medians.csv', index=False, float_format='%.6g')
    pd.DataFrame(comp).to_csv(out / 'paired_boot.csv', index=False, float_format='%.6g')
    pd.DataFrame(diag).to_csv(out / 'axis_diagnostics.csv', index=False)
    write_json(out / 'scores_run.json', {'draws': B, 'seed': SEED, 'cells_universe': len(U), 'drug_universe': len(all_drugs),
                                         'models': G['models'] + [MEASURED, FITNESS], 'published21': G['published21'],
                                         'exactly_agnostic_adj_undefined': sorted(G['agnostic']),
                                         'seconds': round(time.time() - t0, 1)})
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
