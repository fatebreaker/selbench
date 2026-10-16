"""Round 4 (WS8), analysis 1: does selective capture, but not the raw score, predict which models keep drug-specific
accuracy in another dataset?

Units are models. Each model's identical out-of-fold predictions are scored within its training dataset (the
leaderboard view a practitioner has) and in another dataset on matched drugs and, per drug, matched cell lines (the
paired design of 80/82). Within-dataset statistics (predictors) are related across models to the target's
selective-both (outcome), with intervals from a joint bootstrap.

Benchmarks (existing data only)
  leaderboard  DrEval's released CTRP v2 CurveCurator ln IC50 leave-cell-line-out predictions (42's 739 cells, 42's
               two-way draws, seed 20261014, which the paired leaderboard arrays of 80 also use, so within and target
               scores are resampled jointly). Model sets: the 21 published models with per-drug signal (Lasso,
               ElasticNet and MultiViewRandomForest flagged drug-agnostic) and the 18 learned ones. Targets: primary
               = endpoint-matched CurveCurator ln IC50 of CTRP v1, GDSC1, GDSC2 and CCLE; sensitivity = every other
               non-identity version of 80 (CurveCurator AUC, provider summaries, harmonized versions, endpoint-matched
               provider-AUC pairings).
  retrain      drevalpy retraining (59/80): GDSC2 CurveCurator AUC and GDSC2 CurveCurator ln IC50 (all panel models
               except the collapsed SimpleNeuralNetworkSelective; FitnessOnlyRidge and NaiveMeanEffectsPredictor
               flagged drug-agnostic), each against the other DrEval datasets with the same endpoint. CTRP v2 ln IC50
               retraining is scored (drevalpy-run models only) but not pooled: its five drevalpy-run models are too
               few and the leaderboard covers that setting. CTRP v2 AUC (provisional) is not used.
  improve      IMPROVE cross-study analysis, replicate-free (68 --exclude-split-replicates as in 77.improve_setup),
               7 models x 5 sources x 4 targets; own-panel axes primary, shared-panel axes sensitivity; the
               drug-agnostic cell-mean floor added as an eighth unit in a sensitivity set.
Draws: for retraining and IMPROVE one set of cell weights W and drug multiplicities V is drawn per source over the
union of the source's within-dataset cells/drugs and every paired target's cells/drugs (seed 20261100 + i), and used
for both the within-dataset and the paired scores; point estimates (draw 0) reproduce 78/80/91 exactly, intervals
differ only by Monte Carlo. B = 1,000.

Within-dataset predictors (full within-dataset evaluation, each model on its own drugs as in 78/91):
  raw_median       median per-drug r(p, z) (the usual leaderboard score)            [primary comparator]
  dreval_norm      DrEval's normalized Pearson (88's estimator: per split, Pearson of (y - naive) and (p - naive) over
                   pairs, pair weight = cell weight x drug multiplicity; leaderboard on common drugs/cells as 88,
                   retraining on 59's pairs); not available for IMPROVE
  A                axis capture  sum r_pg r_zg / sum r_zg^2
  S                selective capture  sum (r_pz - r_pg r_zg) / sum (1 - r_zg^2)      [primary candidate]
  sel_mean         mean per-drug selective correlation r(p, z | g)
  S_lin, sel_lin_mean  lineage-adjusted selective capture and mean selective correlation (91: given g and tissue)
  also raw_mean, sel_median; and, as a sensitivity, the same statistics on the matched drugs and cells of each pair
  ('matched' predictors: raw_median_w, A_w, S_w, sel_mean_w).
Outcomes in the target (paired, matched cells):
  selB_c           mean over drugs of the symmetric selective-both r(p, z_T | g_S, g_T) (82's 'cross')  [primary]
  raw_median_c     median per-drug r(p, z_T) in the target (the target's leaderboard score)            [part (c)]
  also selB_median_c, raw_mean_c, A_c, S_c.

Pre-stated analysis rules (fixed before any result was computed)
  (a) Per benchmark and target, the across-model Spearman (and Pearson) correlation between a predictor and an
      outcome. Pooling: Fisher-z mean over the benchmark's primary targets with equal weights (leaderboard 4;
      retraining 8 = 2 settings x 4 targets; IMPROVE 20 own-panel source-target pairs), back-transformed; overall =
      equal-weight Fisher-z mean of the three benchmark means. Contrast: z-mean(S) - z-mean(raw_median) (and its
      back-transformed difference). Intervals: 95% percentile intervals of a joint bootstrap that resamples models
      as clusters (the same model multiplicities in every target of a benchmark), drugs and cell lines (the same
      draws for within and target scores); also conditional intervals (drugs and cells only, models fixed).
      Two-sided bootstrap P for the contrast = 2 min(P(d <= 0), P(d >= 0)). Primary model sets: leaderboard 21,
      retraining all panel models, IMPROVE 7; drug-agnostic-excluded sets reported beside them.
  (b) Decision analysis: the top-k models (k = 1, 3, 5) by a within-dataset statistic; outcome = mean target
      selective-both of the chosen set; regret = mean of the k best models by the outcome (same draw) minus that of
      the chosen set; pooled as the mean over the benchmark's primary targets; drug x cell intervals (the menu of
      models is fixed); difference in regret between choosing by raw_median and by S (paired).
      Amended after the first run, for a technical reason (not the results): the 'same draw' best set also picks
      noise, so that regret is >= 0 in every draw and its percentile interval can exclude the point estimate. The
      primary regret now fixes the benchmark set at the k best by the full-sample outcome (regret_redrawn keeps the
      rule as first stated). The contrast between two statistics (chosen_minus_raw_median) is identical under both
      definitions. Bootstrap P values use mid-P for draws exactly at 0 (identical choices).
  (c) as (a) and (b) with the target's median raw correlation as the outcome.
  (d) Patient samples: see patient_predictions_check.json (no multi-model patient predictions are released).
Outputs (results/revision_round4/transfer_prediction/): cache/*.npz (per-draw per-model summaries), units.csv,
correlations_by_target.csv, correlations_pooled.csv (with bias-corrected percentile intervals bc_lo/bc_hi beside the
percentile ones), contrast_by_target.csv, contrast_pooled.csv, decision_pooled.csv, fig_*.csv (figure-ready),
supp_sensitivity_groups.csv, verification_{paired,within,normalized}.csv, patient_predictions_check.json, run.json;
key numbers in
results/revision_round4/key_numbers_ws8.json (section 'transfer_prediction').
Usage: 100a_transfer_prediction.py [--compute] [--analyse] [--parts leaderboard retrain improve]
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
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
import box1  # noqa: E402  (selbench/box1.py, on sys.path via lib77)

OUT = lib.RES / 'revision_round4' / 'transfer_prediction'
CACHE = OUT / 'cache'
KEYS = lib.RES / 'revision_round4' / 'key_numbers_ws8.json'
WORKERS = 3
G = {}
PKEYS = ['raw_w', 'r_pg_w', 'r_zg_w', 'selT_w', 'selB_w', 'raw_c', 'r_pg_c', 'r_zg_c', 'selT_c', 'selB_c']
LB_AGNOSTIC = ['Lasso', 'ElasticNet', 'MultiViewRandomForest']
RT_AGNOSTIC = ['FitnessOnlyRidge', 'NaiveMeanEffectsPredictor']
RT_EXCLUDE = ['SimpleNeuralNetworkSelective']           # collapsed to the drug mean (paper: excluded)
RT_SETTINGS = ['GDSC2_AUC', 'GDSC2_LNIC50', 'CTRPv2_LNIC50']
RT_POOLED = ['GDSC2_AUC', 'GDSC2_LNIC50']
LB_PRIMARY_TARGETS = ['CTRPv1|CC|lnIC50', 'GDSC1|CC|lnIC50', 'GDSC2|CC|lnIC50', 'CCLE|CC|lnIC50']
PREDICTORS = ['raw_median', 'dreval_norm', 'A', 'S', 'sel_mean', 'S_lin', 'sel_lin_mean', 'raw_mean', 'sel_median']
MATCHED_PREDICTORS = ['raw_median_w', 'raw_mean_w', 'A_w', 'S_w', 'sel_mean_w', 'selB_mean_w']
OUTCOMES = ['selB_c', 'raw_median_c', 'selB_median_c', 'raw_mean_c', 'A_c', 'S_c']


# ----------------------------------------------------------------------------------------------------------------
# summaries
def within_summary(arrs, models, V):
    """Per model, per draw: within-dataset statistics from 91's engine arrays (R, M, D) (91.reps: the paper's code)."""
    lin = lib.load_module('lin91', '91_lineage.py')
    R, M = arrs['raw'].shape[:2]
    keys = {'raw_median': 'median_raw', 'raw_mean': 'mean_raw', 'A': 'axis_capture', 'S': 'selective_capture',
            'sel_mean': 'mean_selective', 'sel_median': 'median_selective', 'S_lin': 'selective_capture_lin',
            'sel_lin_mean': 'mean_selective_lin', 'axis_share': 'axis_share', 'perfect_share': 'perfect_share'}
    out = {k: np.full((R, M), np.nan) for k in keys}
    out['drugs'] = np.zeros(M)
    for i in range(M):
        a = {k: arrs[k][:, i] for k in lin.STATS}
        if not np.isfinite(a['raw'][0]).any():
            continue
        st, ok, _ = lin.reps(a, V)
        for k, s in keys.items():
            out[k][:, i] = st[s]
        out['drugs'][i] = ok[0].sum()
    return out


def paired_summary(A, V):
    """Per model, per draw: matched within-dataset and target statistics from per-drug arrays A[k] (R, M, D)."""
    R, M, D = A['raw_w'].shape
    names = MATCHED_PREDICTORS + OUTCOMES + ['drugs_selB']
    out = {k: np.full((R, M), np.nan) for k in names}
    for i in range(M):
        g = {k: A[k][:, i].astype(float) for k in PKEYS}
        if not np.isfinite(g['raw_w'][0]).any():
            continue
        cw = box1.capture(g['raw_w'], g['r_pg_w'], g['r_zg_w'], sel=g['selT_w'], weights=V)
        cc = box1.capture(g['raw_c'], g['r_pg_c'], g['r_zg_c'], sel=g['selT_c'], weights=V)
        okr = np.isfinite(g['raw_w']) & np.isfinite(g['raw_c'])
        oks = okr & np.isfinite(g['selT_w']) & np.isfinite(g['selT_c'])
        okb = np.isfinite(g['selB_w']) & np.isfinite(g['selB_c'])
        with np.errstate(invalid='ignore', divide='ignore'):
            def wmean(x, ok):
                w = np.where(ok, V, 0.0)
                return (w * np.where(ok, x, 0.0)).sum(1) / w.sum(1)
            out['raw_median_w'][:, i] = lib.wmedian_rows(np.where(okr, g['raw_w'], np.nan), V)
            out['raw_mean_w'][:, i] = wmean(g['raw_w'], okr)
            out['A_w'][:, i] = cw['axis_capture']
            out['S_w'][:, i] = cw['selective_capture']
            out['sel_mean_w'][:, i] = wmean(g['selT_w'], oks)
            out['selB_mean_w'][:, i] = wmean(g['selB_w'], okb)
            out['selB_c'][:, i] = wmean(g['selB_c'], okb)
            out['selB_median_c'][:, i] = lib.wmedian_rows(np.where(okb, g['selB_c'], np.nan), V)
            out['raw_median_c'][:, i] = lib.wmedian_rows(np.where(okr, g['raw_c'], np.nan), V)
            out['raw_mean_c'][:, i] = wmean(g['raw_c'], okr)
            out['A_c'][:, i] = cc['axis_capture']
            out['S_c'][:, i] = cc['selective_capture']
            out['drugs_selB'][:, i] = np.where(okb, V, 0).sum(1)
    return out


class PerDrug:
    def __init__(self, models, R, D):
        self.models = list(models)
        self.ix = {m: i for i, m in enumerate(self.models)}
        self.A = {k: np.full((R, len(self.models), D), np.nan, np.float32) for k in PKEYS}
        self.sums = {k: {kind: np.zeros((R, len(self.models))) for kind in ('sum_w', 'sum_c', 'n')}
                     for k in ('raw', 'mediated', 'selBOTH_sym')}

    def add(self, j, st, models_here, v):
        cols = [self.ix[m] for m in models_here]
        for k in PKEYS:
            self.A[k][:, cols, j] = st[k]
        # 80's per-stat sums (verification against results/revision_round2/paired/arrays)
        med_w, med_c = st['r_pg_w'] * st['r_zg_w'], st['r_pg_c'] * st['r_zg_c']
        for k, (a, b) in {'raw': (st['raw_w'], st['raw_c']), 'mediated': (med_w, med_c),
                          'selBOTH_sym': (st['selB_w'], st['selB_c'])}.items():
            ok = np.isfinite(a) & np.isfinite(b)
            self.sums[k]['sum_w'][:, cols] += np.where(ok, a, 0.0) * v[:, None]
            self.sums[k]['sum_c'][:, cols] += np.where(ok, b, 0.0) * v[:, None]
            self.sums[k]['n'][:, cols] += ok * v[:, None]


def save_cache(name, models, stats, meta):
    np.savez_compressed(CACHE / f'{name}.npz', models=np.array(models), meta=json.dumps(meta),
                        **{k: np.asarray(v, np.float64) for k, v in stats.items()})


def load_cache(name):
    z = np.load(CACHE / f'{name}.npz', allow_pickle=False)
    return list(z['models']), json.loads(str(z['meta'])), {k: z[k] for k in z.files if k not in ('models', 'meta')}


def normalized_draws(rows, naive, cell_ix, drug_ix, W, V, models):
    """DrEval's normalized Pearson per draw (88.normalized_reps generalised): rows {model: drug, cell, split, pred, y}
    on DrEval's scale; naive: drug, cell, split, naive; pair weight W[cell] x V[drug]; mean over splits."""
    R = W.shape[0]
    out = np.full((R, len(models)), np.nan)
    for i, m in enumerate(models):
        if m not in rows:
            continue
        x = rows[m].merge(naive, on=['drug', 'cell', 'split'], how='inner')
        x = x.assign(cu=x.cell.map(cell_ix), dv=x.drug.map(drug_ix)).dropna(subset=['cu', 'dv'])
        per = []
        for _, g in x.groupby('split'):
            a, b = (g.y - g.naive).values, (g.pred - g.naive).values
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            w = W[:, g.cu.values[ok].astype(int)] * V[:, g.dv.values[ok].astype(int)]
            F = np.column_stack([np.ones_like(a), a, b, a * a, b * b, a * b])
            S = w @ F
            n = S[:, 0]
            with np.errstate(invalid='ignore', divide='ignore'):
                ma, mb = S[:, 1] / n, S[:, 2] / n
                va, vb, cab = S[:, 3] / n - ma ** 2, S[:, 4] / n - mb ** 2, S[:, 5] / n - ma * mb
                per.append(cab / np.sqrt(va * vb))
        out[:, i] = np.mean(np.vstack(per), axis=0)
    return out


# ----------------------------------------------------------------------------------------------------------------
# leaderboard
def lb_versions():
    vt = pd.read_csv(lib.RES / 'revision_crosslab' / 'versions.csv')
    TG = ['CTRPv1', 'GDSC1', 'GDSC2', 'CCLE']
    harm = [f'CTRPv2|{f}-{T}|AUC' for T in TG for f in ('HR', 'SP')] + [f'{T}|{f}|AUC' for T in TG for f in ('HR', 'SP')]
    harm += [f'CTRPv2|{f}-{T}|lnIC50' for T in TG for f in ('SL', 'RL')] + [f'{T}|{f}|lnIC50' for T in TG for f in ('SL', 'RL')]
    matched = ['CTRPv2|CTRP|AUC', 'CTRPv1|CTRP|AUC', 'GDSC1|Sanger|AUC', 'GDSC2|Sanger|AUC', 'CCLE|CCLE|ActArea']
    return sorted(set(vt.version) | set(harm) | set(matched))


def lb_tasks():
    """The 80 leaderboard pairs (k < 38 and k >= 100) except identity and CTRP v2 re-processed targets."""
    ix = pd.read_csv(lib.OUT2 / 'paired' / 'pairs.csv')
    ix = ix[(ix.benchmark == 'leaderboard') & ~ix.pair.str.contains('@RLdrugs')]
    ix = ix[~ix.target.str.startswith('CTRPv2')]
    return [{'k': int(r.k), 'pair': r.pair, 'reference': r.source, 'target': r.target, 'family': r.family}
            for _, r in ix.iterrows()]


def lb_within_task(_):
    L = G['L']
    lin = lib.load_module('lin91', '91_lineage.py')
    cd = L['cd']
    Z, Gm = L['dec'][cd.REFERENCE]
    U = L['U']
    drugs = [d for d in Z.index if d in L['drug_ix']]
    Zu, Gu = Z.reindex(index=drugs, columns=U), Gm.reindex(index=drugs, columns=U)
    Pm = {m: P.reindex(index=drugs, columns=U) for m, P in L['P'].items()}
    V = L['V'][:, [L['drug_ix'][d] for d in drugs]]
    arrs, models, dl, ncell, ntis = lin.engine(Zu, Gu, Pm, lin.tissue_of_cellosaurus(), L['W'], V, mask='subset',
                                               mask_models=L['lbm'])
    st = within_summary(arrs, models, V)
    # DrEval's normalized Pearson (88: published models, drugs common to the 22 published models)
    rk = lib.load_module('rk88', '88_rank_intervals.py')
    pub = sorted(m for m in L['P'] if L['mclass'][m] in ('classical', 'neural/factorization'))
    ok = np.isfinite(arrs['raw'][0][[models.index(m) for m in pub]]) & np.isfinite(arrs['sel'][0][[models.index(m) for m in pub]])
    common = set(np.array(dl)[ok.all(0)])
    nrm, dropped, ncells = rk.normalized_reps(L, pub, common)
    st['dreval_norm'] = np.full_like(st['A'], np.nan)
    for j, m in enumerate(pub):
        st['dreval_norm'][:, models.index(m)] = nrm[:, j]
    save_cache('leaderboard__within', models, st, {'drugs': len(dl), 'cells': len(U), 'normalized_common_drugs': len(common),
                                                   'normalized_dropped': dropped})
    return 'leaderboard__within'


def lb_pair_task(t):
    L = G['L']
    Zs, Gs = L['dec'][t['reference']]
    Zt, Gt = L['dec'][t['target']]
    U = L['U']
    models = list(L['P'])
    mats = {k: M.reindex(columns=U) for k, M in [('zs', Zs), ('gs', Gs), ('zt', Zt), ('gt', Gt)]}
    drugs = [d for d in Zs.index.intersection(Zt.index) if d in L['drug_ix']]
    R = L['W'].shape[0]
    acc = PerDrug(models, R, len(drugs))
    used = []
    for j, d in enumerate(drugs):
        cols = {k: mats[k].loc[d].values.astype(float) for k in mats}
        m0 = np.all([np.isfinite(v) for v in cols.values()], axis=0)
        for m in L['lbm']:
            Pm = L['P'][m]
            if d in Pm.index:
                m0 &= np.isfinite(Pm.loc[d].values)
        if m0.sum() < lib.MIN_CELLS:
            continue
        idx = np.where(m0)[0]
        use, pc = [], []
        for m in models:
            if d not in L['P'][m].index:
                continue
            p = L['P'][m].loc[d].values[idx].astype(float)
            if not np.all(np.isfinite(p)) or np.std(p) < 1e-12:
                continue
            use.append(m)
            pc.append(p)
        st = lib.paired_stats(cols['zs'][idx], cols['gs'][idx], cols['zt'][idx], cols['gt'][idx], np.column_stack(pc),
                              L['W'][:, idx], identity=False)
        v = L['V'][:, L['drug_ix'][d]]
        acc.add(j, st, use, v)
        used.append(j)
    V = L['V'][:, [L['drug_ix'][d] for d in drugs]]
    keep = np.array(used, int)
    A = {k: a[:, :, keep] for k, a in acc.A.items()}
    out = paired_summary(A, V[:, keep])
    for k, d in acc.sums.items():
        for kind, a in d.items():
            out[f'sum|{k}|{kind}'] = a
    name = f"leaderboard__pair{t['k']}"
    save_cache(name, models, out, {**t, 'drugs_used': len(keep)})
    return name


# ----------------------------------------------------------------------------------------------------------------
# retraining
def rt_prepare(setting, seed):
    lin = lib.load_module('lin91', '91_lineage.py')
    wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
    base = sorted(wp[(wp.setting == setting) & (wp.panel == 'main')].model.unique())
    rt = lib.load_module('rt59', '59_retrain_score.py')
    oof, rawscale, _ = rt.load_oof(setting)
    extra = [m for m in ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting'] if m in oof and m not in base]
    S = lib.retrain_setup(setting, models=base + extra)
    O = S['obs'][S['obs'].cell.isin(S['cells'])][['drug', 'cell', 'response']]
    Zw, Gw, _, _ = lib.selbench.decompose(O)                              # 78/91 within decomposition
    Zs, Gs, _, _ = lib.selbench.decompose(S['obs'][['drug', 'cell', 'response']])   # 80's paired source
    targets = {}
    for T in [t for t in rt.DATASETS if t != S['dataset']]:
        obs_t = S['resp'][(S['resp'].dataset == T) & (S['resp'].endpoint == S['endpoint'])][['drug', 'cell', 'response']]
        Zt, Gt, _, _ = lib.selbench.decompose(obs_t)
        pdrugs = sorted(set(Zs.index) & set(Zt.index) & set().union(*[set(S['P'][m].index) for m in S['models']]))
        pcells = sorted(set(Zs.columns) & set(Zt.columns) & set(S['cells']))
        if len(pdrugs) < 5 or len(pcells) < lib.MIN_CELLS:
            continue
        targets[T] = (Zt, Gt, pdrugs, pcells)
    cells = sorted(set(Zw.columns) | set().union(*[set(v[3]) for v in targets.values()]))
    drugs = sorted(set(Zw.index) | set().union(*[set(v[2]) for v in targets.values()]))
    W, V = lib.draws(len(cells), len(drugs), seed)
    models = [m for m in S['models']]
    if setting == 'CTRPv2_LNIC50':
        models = [m for m in models if not m.startswith('LB:')]
    S.update({'Zw': Zw, 'Gw': Gw, 'Zs': Zs, 'Gs': Gs, 'targets': targets, 'cells_master': cells, 'drugs_master': drugs,
              'W': W, 'V': V, 'cix': {c: i for i, c in enumerate(cells)}, 'dix': {d: i for i, d in enumerate(drugs)},
              'models_used': models, 'rawscale': rawscale, 'tissue': lin.tissue_of_cellosaurus()})
    return S


def rt_within_task(setting):
    S = G['retrain'][setting]
    lin = lib.load_module('lin91', '91_lineage.py')
    Zw, Gw = S['Zw'], S['Gw']
    ci = [S['cix'][c] for c in Zw.columns]
    di = [S['dix'][d] for d in Zw.index]
    W, V = S['W'][:, ci], S['V'][:, di]
    Pm = {m: S['P'][m].reindex(columns=Zw.columns) for m in S['P']}       # every panel model (91's mask)
    arrs, models, dl, ncell, ntis = lin.engine(Zw, Gw, Pm, S['tissue'], W, V, mask='common')
    st = within_summary(arrs, models, V)
    # DrEval's normalized Pearson (59's pairs: cells of the panel, drugs of the within decomposition)
    nv_name = 'NaiveMeanEffectsPredictor' if 'NaiveMeanEffectsPredictor' in S['rawscale'] else 'LB:NaiveMeanEffectsPredictor'
    st['dreval_norm'] = np.full_like(st['A'], np.nan)
    if nv_name in S['rawscale']:
        nv = S['rawscale'][nv_name][['drug', 'cell', 'pred', 'split']].rename(columns={'pred': 'naive'})
        rows = {m: S['rawscale'][m][['drug', 'cell', 'pred', 'y', 'split']] for m in models
                if m in S['rawscale'] and m != nv_name}
        rows = {m: x[x.cell.isin(S['cells']) & x.drug.isin(set(Zw.index))] for m, x in rows.items()}
        cix = {c: i for i, c in enumerate(Zw.columns)}
        dix = {d: i for i, d in enumerate(Zw.index)}
        nrm = normalized_draws(rows, nv, cix, dix, W, V, models)
        st['dreval_norm'] = nrm
        if nv_name in models:
            st['dreval_norm'][:, models.index(nv_name)] = 0.0      # 0 by construction
    save_cache(f'retrain_{setting}__within', models, st, {'drugs': len(dl), 'cells': len(Zw.columns)})
    return f'retrain_{setting}__within'


def rt_pair_task(arg):
    setting, T = arg
    S = G['retrain'][setting]
    Zs, Gs = S['Zs'], S['Gs']
    Zt, Gt, pdrugs, pcells = S['targets'][T]
    models = S['models']                                  # 80's panel (masking uses every model)
    Pm = {m: S['P'][m].reindex(index=[d for d in pdrugs if d in S['P'][m].index], columns=pcells) for m in models}
    ci = np.array([S['cix'][c] for c in pcells])
    W = S['W'][:, ci]
    mats = [M.reindex(index=pdrugs, columns=pcells).values.astype(float) for M in (Zs, Gs, Zt, Gt)]
    Pv = {m: Pm[m].reindex(index=pdrugs, columns=pcells).values.astype(float) for m in models}
    acc = PerDrug(models, W.shape[0], len(pdrugs))
    used = []
    for j, d in enumerate(pdrugs):
        zs, gs, zt, gt = (M[j] for M in mats)
        have = [m for m in models if d in Pm[m].index]
        ok = np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        for m in have:
            ok &= np.isfinite(Pv[m][j])
        if ok.sum() < lib.MIN_CELLS:
            continue
        idx = np.where(ok)[0]
        use = [m for m in have if np.std(Pv[m][j][idx]) >= 1e-12]
        st = lib.paired_stats(zs[idx], gs[idx], zt[idx], gt[idx], np.column_stack([Pv[m][j][idx] for m in use]), W[:, idx])
        acc.add(j, st, use, S['V'][:, S['dix'][d]])
        used.append(j)
    keep = np.array(used, int)
    V = S['V'][:, [S['dix'][pdrugs[j]] for j in keep]]
    out = paired_summary({k: a[:, :, keep] for k, a in acc.A.items()}, V)
    for k, dd in acc.sums.items():
        for kind, a in dd.items():
            out[f'sum|{k}|{kind}'] = a
    name = f'retrain_{setting}__pair_{T}'
    save_cache(name, models, out, {'setting': setting, 'target': T, 'drugs_used': len(keep), 'cells': len(pcells)})
    return name


# ----------------------------------------------------------------------------------------------------------------
# IMPROVE
def im_prepare(I, src, seed):
    im = I['im']
    Zw, Gw = I['dec'][src]
    targets = {}
    for var in ['own_panel', 'shared_panel']:
        for trg in I['datasets']:
            if trg == src:
                continue
            if var == 'own_panel':
                (Zs, Gs), (Zt, Gt) = I['dec'][src], I['dec'][trg]
            else:
                shared = I['dec'][src][0].index.intersection(I['dec'][trg][0].index)
                Zs, Gs = im.decompose(I['obs'][src], panel=shared)
                Zt, Gt = im.decompose(I['obs'][trg], panel=shared)
            targets[(trg, var)] = (Zs, Gs, Zt, Gt)
    cells = sorted(set(Zw.columns) | set().union(*[set(v[0].columns) for v in targets.values()]))
    drugs = sorted(set(Zw.index) | set().union(*[set(v[0].index) for v in targets.values()]))
    W, V = lib.draws(len(cells), len(drugs), seed)
    return {'Zw': Zw, 'Gw': Gw, 'targets': targets, 'W': W, 'V': V, 'cix': {c: i for i, c in enumerate(cells)},
            'dix': {d: i for i, d in enumerate(drugs)}}


def im_within_task(src):
    I = G['improve']
    S = I['src'][src]
    lin = lib.load_module('lin91', '91_lineage.py')
    tis, _ = lin.tissue_of_depmap()
    Zw, Gw = S['Zw'], S['Gw']
    Pm = {m: I['P_within'][(m, src)] for m in I['models'] if (m, src) in I['P_within']}
    Pm['floor_cell_mean'] = I['floors'][src]
    Pm = {m: P.reindex(columns=Zw.columns) for m, P in Pm.items()}
    W = S['W'][:, [S['cix'][c] for c in Zw.columns]]
    V = S['V'][:, [S['dix'][d] for d in Zw.index]]
    arrs, models, dl, ncell, ntis = lin.engine(Zw, Gw, Pm, tis, W, V, mask='per_model')
    st = within_summary(arrs, models, V)
    st['dreval_norm'] = np.full_like(st['A'], np.nan)
    save_cache(f'improve_{src}__within', models, st, {'drugs': len(dl), 'cells': len(Zw.columns)})
    return f'improve_{src}__within'


def im_pair_task(arg):
    src, trg, var = arg
    I = G['improve']
    S = I['src'][src]
    Zs, Gs, Zt, Gt = S['targets'][(trg, var)]
    Pm = {m: I['P_cross'][(m, src, trg)] for m in I['models'] if (m, src, trg) in I['P_cross']}
    if not Pm:
        return None
    # drug-agnostic floor: the identical cell-mean prediction of the source split (sensitivity unit)
    Pm['floor_cell_mean'] = I['floors'][src]
    models = list(Pm)
    cells = Zs.columns.intersection(Zt.columns)
    drugs = Zs.index.intersection(Zt.index).intersection(pd.Index(sorted(set().union(
        *[set(P.index) for m, P in Pm.items() if m != 'floor_cell_mean']))))
    W = S['W'][:, [S['cix'][c] for c in cells]]
    mats = [M.reindex(index=drugs, columns=cells).values.astype(float) for M in (Zs, Gs, Zt, Gt)]
    Pv = {m: Pm[m].reindex(index=drugs, columns=cells).values.astype(float) for m in models}
    # the floor is restricted to the pairs the models predicted (pairs measured in both datasets)
    anym = np.zeros((len(drugs), len(cells)), bool)
    for m in models:
        if m != 'floor_cell_mean':
            anym |= np.isfinite(Pv[m])
    Pv['floor_cell_mean'] = np.where(anym, Pv['floor_cell_mean'], np.nan)
    acc = PerDrug(models, W.shape[0], len(drugs))
    used = []
    for j, d in enumerate(drugs):
        zs, gs, zt, gt = (M[j] for M in mats)
        base = np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        groups = {}
        for m in models:
            mk = base & np.isfinite(Pv[m][j])
            if mk.sum() < lib.MIN_CELLS:
                continue
            groups.setdefault(mk.tobytes(), (mk, []))[1].append(m)
        u = False
        for mk, ms in groups.values():
            idx = np.where(mk)[0]
            use = [m for m in ms if np.std(Pv[m][j][idx]) >= 1e-12]
            if not use:
                continue
            st = lib.paired_stats(zs[idx], gs[idx], zt[idx], gt[idx], np.column_stack([Pv[m][j][idx] for m in use]),
                                  W[:, idx])
            acc.add(j, st, use, S['V'][:, S['dix'][d]])
            u = True
        if u:
            used.append(j)
    keep = np.array(used, int)
    V = S['V'][:, [S['dix'][drugs[j]] for j in keep]]
    out = paired_summary({k: a[:, :, keep] for k, a in acc.A.items()}, V)
    for k, dd in acc.sums.items():
        for kind, a in dd.items():
            out[f'sum|{k}|{kind}'] = a
    name = f'improve_{src}__pair_{trg}_{var}'
    save_cache(name, models, out, {'source': src, 'target': trg, 'variant': var, 'drugs_used': len(keep),
                                   'cells': len(cells)})
    return name


def run_pool(fn, tasks, tag, t0):
    names = []
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context('fork')) as ex:
        for n in ex.map(fn, tasks):
            names.append(n)
            print(tag, n, f'{time.time() - t0:.0f}s', flush=True)
    return names


def compute(parts, t0):
    CACHE.mkdir(parents=True, exist_ok=True)
    if 'leaderboard' in parts:
        L = lib.leaderboard_setup(need_versions=lb_versions(), harmonized=True)
        G['L'] = L
        print('leaderboard set-up', f'{time.time() - t0:.0f}s', flush=True)
        run_pool(lb_within_task, [None], 'leaderboard', t0)
        run_pool(lb_pair_task, lb_tasks(), 'leaderboard', t0)
        G.pop('L')
    if 'retrain' in parts:
        G['retrain'] = {}
        for i, s in enumerate(RT_SETTINGS):
            G['retrain'][s] = rt_prepare(s, 20261100 + i)
            print('retrain set-up', s, len(G['retrain'][s]['models']), 'models', f'{time.time() - t0:.0f}s', flush=True)
        run_pool(rt_within_task, RT_SETTINGS, 'retrain', t0)
        run_pool(rt_pair_task, [(s, T) for s in RT_SETTINGS for T in G['retrain'][s]['targets']], 'retrain', t0)
        G.pop('retrain')
    if 'improve' in parts:
        I = lib.improve_setup(replicate_free=True)
        I['src'] = {src: im_prepare(I, src, 20261110 + i) for i, src in enumerate(I['datasets'])}
        G['improve'] = I
        print('IMPROVE set-up', f'{time.time() - t0:.0f}s', flush=True)
        run_pool(im_within_task, I['datasets'], 'improve', t0)
        tasks = [(s, t, v) for s in I['datasets'] for t in I['datasets'] if t != s for v in ['own_panel', 'shared_panel']]
        run_pool(im_pair_task, tasks, 'improve', t0)
        G.pop('improve')


# ----------------------------------------------------------------------------------------------------------------
# analysis
def weighted_pearson(x, y, w):
    """Row-wise weighted Pearson. x, y, w: (R, M); invalid entries carry weight 0."""
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    w = np.where(ok, w, 0.0)
    x, y = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        n = w.sum(1)
        mx, my = (w * x).sum(1) / n, (w * y).sum(1) / n
        cxy = (w * (x - mx[:, None]) * (y - my[:, None])).sum(1)
        vx, vy = (w * (x - mx[:, None]) ** 2).sum(1), (w * (y - my[:, None]) ** 2).sum(1)
        r = cxy / np.sqrt(vx * vy)
    distinct = (ok & (w > 0)).sum(1)
    return np.where(distinct >= 3, r, np.nan)


def weighted_midranks(x, w):
    """Mid-ranks of the expanded sample (model i repeated w_i times); x, w: (R, M); NaN/zero-weight -> NaN."""
    ok = np.isfinite(x) & (w > 0)
    ww = np.where(ok, w, 0.0)
    xx = np.where(ok, x, np.inf)
    less = (xx[:, None, :] < xx[:, :, None]) & ok[:, None, :]
    tie = (xx[:, None, :] == xx[:, :, None]) & ok[:, None, :]
    L_ = (less * ww[:, None, :]).sum(2)
    T_ = (tie * ww[:, None, :]).sum(2)
    return np.where(ok, L_ + (T_ + 1) / 2, np.nan)


def weighted_spearman(x, y, w):
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    w = np.where(ok, w, 0.0)
    return weighted_pearson(weighted_midranks(np.where(ok, x, np.nan), w), weighted_midranks(np.where(ok, y, np.nan), w), w)


def fz(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def pboot(d):
    """Two-sided bootstrap P (mid-P for draws exactly at 0, e.g. identical top-k choices)."""
    b = np.asarray(d, float)[1:]
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan
    z = 0.5 * (b == 0).mean()
    return float(min(1.0, 2 * min((b < 0).mean() + z, (b > 0).mean() + z)))


def build_units():
    """Pair records: per benchmark/setting/target, model names, predictors (R, M) and outcomes (R, M)."""
    recs = []
    # leaderboard
    if (CACHE / 'leaderboard__within.npz').exists():
        mw, metaw, W_ = load_cache('leaderboard__within')
        L_pub = pd.read_csv(lib.RES / 'revision_round2' / 'capture' / 'capture_by_model.csv')
        for f in sorted(CACHE.glob('leaderboard__pair*.npz')):
            mp_, meta, P_ = load_cache(f.stem)
            recs.append({'benchmark': 'leaderboard', 'setting': 'CTRPv2_lnIC50', 'source': meta['reference'],
                         'target': meta['target'], 'variant': meta['family'], 'pair': meta['pair'], 'k': meta['k'],
                         'primary': meta['reference'] == 'CTRPv2|CC|lnIC50' and meta['target'] in LB_PRIMARY_TARGETS,
                         'drugs_used': meta['drugs_used'], 'models_w': mw, 'W': W_, 'models_p': mp_, 'P': P_})
    for s in RT_SETTINGS:
        if not (CACHE / f'retrain_{s}__within.npz').exists():
            continue
        mw, _, W_ = load_cache(f'retrain_{s}__within')
        for f in sorted(CACHE.glob(f'retrain_{s}__pair_*.npz')):
            mp_, meta, P_ = load_cache(f.stem)
            recs.append({'benchmark': 'retrain', 'setting': s, 'source': s, 'target': meta['target'],
                         'variant': 'same endpoint', 'pair': f"{s}->{meta['target']}", 'k': None,
                         'primary': s in RT_POOLED, 'drugs_used': meta['drugs_used'], 'models_w': mw, 'W': W_,
                         'models_p': mp_, 'P': P_})
    for src in ['CCLE', 'CTRPv2', 'GDSCv1', 'GDSCv2', 'gCSI']:
        if not (CACHE / f'improve_{src}__within.npz').exists():
            continue
        mw, _, W_ = load_cache(f'improve_{src}__within')
        for f in sorted(CACHE.glob(f'improve_{src}__pair_*.npz')):
            mp_, meta, P_ = load_cache(f.stem)
            recs.append({'benchmark': 'improve', 'setting': f'{src}_AUC', 'source': src, 'target': meta['target'],
                         'variant': meta['variant'], 'pair': f"{src}->{meta['target']}|{meta['variant']}", 'k': None,
                         'primary': meta['variant'] == 'own_panel', 'drugs_used': meta['drugs_used'], 'models_w': mw,
                         'W': W_, 'models_p': mp_, 'P': P_})
    return recs


def model_sets(rec):
    """Named model sets for a record -> list of model names."""
    b = rec['benchmark']
    if b == 'leaderboard':
        if 'pub21' not in G:
            cap = pd.read_csv(lib.RES / 'revision_round2' / 'capture' / 'capture_pooled.csv')
            G['pub21'] = cap[(cap.benchmark == 'leaderboard') & (cap.group == 'published 21 (paper)')].model_list.iloc[0].split(';')
        pub21 = G['pub21']
        return {'all': pub21, 'learned': [m for m in pub21 if m not in LB_AGNOSTIC]}
    if b == 'retrain':
        ms = [m for m in rec['models_w'] if m in rec['models_p'] and m not in RT_EXCLUDE and not m.startswith('LB:')
              and not m.startswith('ref')]
        return {'all': ms, 'learned': [m for m in ms if m not in RT_AGNOSTIC]}
    seven = ['deepcdr', 'deepttc', 'graphdrp', 'hidra', 'lgbm', 'tcnns', 'uno']
    return {'all': seven, 'all_plus_floor': seven + ['floor_cell_mean']}


def matrices(rec, models):
    """Predictors and outcomes (R, len(models)) for the model list (NaN where absent)."""
    X = {}
    for k in PREDICTORS:
        arr = rec['W'].get(k)
        X[k] = np.column_stack([arr[:, rec['models_w'].index(m)] if (arr is not None and m in rec['models_w'])
                                else np.full(arr.shape[0] if arr is not None else 1, np.nan) for m in models])
    R = X['A'].shape[0]
    for k in MATCHED_PREDICTORS + OUTCOMES:
        arr = rec['P'][k]
        X[k] = np.column_stack([arr[:, rec['models_p'].index(m)] if m in rec['models_p'] else np.full(R, np.nan)
                                for m in models])
    return X


def analyse(B_model_seed=20261120):
    t0 = time.time()
    recs = build_units()
    print('records', len(recs), f'{time.time() - t0:.0f}s', flush=True)
    R = recs[0]['W']['A'].shape[0]
    # model-cluster multiplicities per benchmark (row 0 = ones)
    rng = np.random.default_rng(B_model_seed)
    clusters = {}
    for b in ['leaderboard', 'retrain', 'improve']:
        names = sorted(set().union(*[set(sum(model_sets(r).values(), [])) for r in recs if r['benchmark'] == b]))
        if not names:
            continue
        U = np.vstack([np.ones(len(names)), rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=R - 1)])
        clusters[b] = (names, U)
    preds = PREDICTORS + MATCHED_PREDICTORS
    rows, unit_rows = [], []
    store = {}       # (benchmark, set, pair, predictor, outcome, method, cond) -> (R,) correlations
    dec_store = {}   # (benchmark, set, pair, predictor, outcome, k) -> dict of (R,) arrays
    for rec in recs:
        for sname, ms in model_sets(rec).items():
            ms = [m for m in ms if m in rec['models_w'] and m in rec['models_p']]
            if len(ms) < 4:
                continue
            X = matrices(rec, ms)
            names, U = clusters[rec['benchmark']]
            Wm = U[:, [names.index(m) for m in ms]]
            ones = np.ones_like(Wm)
            for i, m in enumerate(ms):
                unit_rows.append({'benchmark': rec['benchmark'], 'setting': rec['setting'], 'pair': rec['pair'],
                                  'model_set': sname, 'model': m, 'primary': rec['primary'],
                                  **{k: float(X[k][0, i]) for k in preds + OUTCOMES}})
            for out_k in ['selB_c', 'raw_median_c']:
                Y = X[out_k]
                for p in preds:
                    if not np.isfinite(X[p][0]).sum() >= 4:
                        continue
                    for meth, fn in [('spearman', weighted_spearman), ('pearson', weighted_pearson)]:
                        r_full = fn(X[p], Y, Wm)
                        r_cond = fn(X[p], Y, ones)
                        store[(rec['benchmark'], sname, rec['pair'], p, out_k, meth, 'full')] = r_full
                        store[(rec['benchmark'], sname, rec['pair'], p, out_k, meth, 'cond')] = r_cond
                        rows.append({'benchmark': rec['benchmark'], 'setting': rec['setting'], 'pair': rec['pair'],
                                     'source': rec['source'], 'target': rec['target'], 'variant': rec['variant'],
                                     'primary': rec['primary'], 'model_set': sname, 'models': len(ms),
                                     'drugs_used': rec['drugs_used'], 'predictor': p, 'outcome': out_k, 'method': meth,
                                     'r': float(r_cond[0]), 'lo_cond': lib.ci(r_cond)[0], 'hi_cond': lib.ci(r_cond)[1],
                                     'lo_full': lib.ci(r_full)[0], 'hi_full': lib.ci(r_full)[1]})
                    # decision analysis (models fixed; drug x cell draws)
                    for k in (1, 3, 5):
                        if k >= len(ms):
                            continue
                        dec_store[(rec['benchmark'], sname, rec['pair'], p, out_k, k)] = decision(X[p], Y, k)
    C = pd.DataFrame(rows)
    C.to_csv(OUT / 'correlations_by_target.csv', index=False, float_format='%.5g')
    pd.DataFrame(unit_rows).to_csv(OUT / 'units.csv', index=False, float_format='%.5g')
    return recs, store, dec_store, C, clusters


def decision(x, y, k):
    """Top-k models by a within-dataset statistic x, re-chosen in every draw; outcome y (R, M).
    regret (primary): mean y of the k best models by the full-sample (point) outcome minus mean y of the chosen set,
      both evaluated in the same draw (the benchmark set is fixed, so the regret can be negative in a draw);
    regret_redrawn (the rule first stated): the k best by y in the same draw minus the chosen set (>= 0; its
      percentile interval is biased upward because the oracle also picks noise);
    gain_share: (chosen - mean of all models) / (best - mean of all models), the share of the achievable gain over a
      random pick. The difference in regret between two statistics does not depend on the benchmark set."""
    R, M = x.shape
    ok = np.isfinite(x) & np.isfinite(y)
    xs = np.where(ok, x, -np.inf)
    ys = np.where(ok, y, -np.inf)
    ch = np.argsort(-xs, axis=1, kind='stable')[:, :k]
    orc = np.argsort(-ys, axis=1, kind='stable')[:, :k]
    best0 = np.argsort(-ys[0], kind='stable')[:k]
    rr = np.arange(R)[:, None]
    with np.errstate(invalid='ignore'):
        chosen = np.where(np.isfinite(ys[rr, ch]), ys[rr, ch], np.nan).mean(1)
        oracle_redrawn = ys[rr, orc].mean(1)
        best = np.where(np.isfinite(ys[:, best0]), ys[:, best0], np.nan).mean(1)
        mean_all = np.nanmean(np.where(ok, y, np.nan), 1)
        gain = (chosen - mean_all) / (best - mean_all)
    return {'chosen': chosen, 'best': best, 'regret': best - chosen, 'oracle_redrawn': oracle_redrawn,
            'regret_redrawn': oracle_redrawn - chosen, 'mean_all': mean_all, 'regret_random': best - mean_all,
            'gain_share': gain, 'chosen_set_point': ';'.join(map(str, ch[0]))}


def bc_interval(rep, level=95):
    """Bias-corrected percentile interval (Efron): z0 = Phi^-1(share of draws below the point)."""
    from scipy.stats import norm
    rep = np.asarray(rep, float)
    b = rep[1:][np.isfinite(rep[1:])]
    if len(b) < 10 or not np.isfinite(rep[0]):
        return np.nan, np.nan
    p0 = np.clip(((b < rep[0]).sum() + 0.5 * (b == rep[0]).sum()) / len(b), 1 / len(b), 1 - 1 / len(b))
    z0 = norm.ppf(p0)
    za = norm.ppf(1 - (100 - level) / 200)
    lo, hi = norm.cdf(2 * z0 - za), norm.cdf(2 * z0 + za)
    return float(np.percentile(b, 100 * lo)), float(np.percentile(b, 100 * hi))


def groups_of(recs):
    """Pooling groups -> (benchmark whose model clusters apply, [pairs]). Primary groups first (pre-stated)."""
    g = {}
    for r in recs:
        b = r['benchmark']
        if r['primary']:
            g.setdefault(b, (b, []))[1].append(r['pair'])
        if b == 'retrain':
            g.setdefault(f"retrain:{r['setting']}", (b, []))[1].append(r['pair'])
        if b == 'leaderboard':
            fam = 'primary (CC ln IC50 targets)' if r['primary'] else r['variant']
            g.setdefault(f'leaderboard:{fam}', (b, []))[1].append(r['pair'])
            if not r['primary']:
                g.setdefault('leaderboard:all non-primary versions', (b, []))[1].append(r['pair'])
        if b == 'improve':
            g.setdefault(f"improve:{r['variant']}", (b, []))[1].append(r['pair'])
            g.setdefault(f"improve:source {r['source']}|{r['variant']}", (b, []))[1].append(r['pair'])
    return g


def pool(store, dec_store, recs, clusters):
    """Pre-stated pooling (Fisher-z mean over a group's targets with equal weights; primary groups = the benchmark
    names; overall = equal-weight mean of the three benchmark means) and decision pooling (mean over targets)."""
    groups = groups_of(recs)
    rows, crow, drow = [], [], []
    bench_z = {}
    preds = PREDICTORS + MATCHED_PREDICTORS
    import warnings

    def zmean(arrs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(fz(np.vstack(arrs)), axis=0)

    def contrast_rows(gname, sname, out_k, meth, cond, Z, ntarg):
        for ref in ['raw_median', 'dreval_norm']:
            for p in preds:
                if p == ref or p not in Z or ref not in Z:
                    continue
                d = Z[p] - Z[ref]
                dr = np.tanh(Z[p]) - np.tanh(Z[ref])
                lo, hi = lib.ci(d)
                lo2, hi2 = lib.ci(dr)
                blo, bhi = bc_interval(dr)
                crow.append({'group': gname, 'model_set': sname, 'outcome': out_k, 'method': meth, 'interval': cond,
                             'predictor': p, 'vs': ref, 'targets': ntarg, 'dz': float(d[0]), 'dz_lo': lo, 'dz_hi': hi,
                             'dr': float(dr[0]), 'dr_lo': lo2, 'dr_hi': hi2, 'dr_bc_lo': blo, 'dr_bc_hi': bhi,
                             'p': pboot(d), 'share_draws_gt0': float(np.mean(d[1:][np.isfinite(d[1:])] > 0))})

    for gname, (b, pairs) in groups.items():
        for sname in ['all', 'learned', 'all_plus_floor']:
            for out_k in ['selB_c', 'raw_median_c']:
                for meth in ['spearman', 'pearson']:
                    for cond in ['full', 'cond']:
                        Z = {}
                        for p in preds:
                            arrs = [store[(b, sname, pr, p, out_k, meth, cond)] for pr in pairs
                                    if (b, sname, pr, p, out_k, meth, cond) in store]
                            if len(arrs) < len(pairs) or not arrs:
                                continue
                            z = zmean(arrs)
                            Z[p] = z
                            lo, hi = lib.ci(np.tanh(z))
                            blo, bhi = bc_interval(np.tanh(z))
                            rows.append({'group': gname, 'benchmark': b, 'model_set': sname, 'outcome': out_k,
                                         'method': meth, 'interval': cond, 'predictor': p, 'targets': len(arrs),
                                         'r_pooled': float(np.tanh(z[0])), 'lo': lo, 'hi': hi, 'bc_lo': blo,
                                         'bc_hi': bhi, 'boot_median': float(np.nanmedian(np.tanh(z[1:]))),
                                         'p_gt0': pboot(z)})
                            if gname == b:
                                bench_z[(b, sname, out_k, meth, cond, p)] = z
                        contrast_rows(gname, sname, out_k, meth, cond, Z, len(pairs))
                for p in preds:
                    for k in (1, 3, 5):
                        ds = [dec_store[(b, sname, pr, p, out_k, k)] for pr in pairs if (b, sname, pr, p, out_k, k) in dec_store]
                        if len(ds) < len(pairs) or not ds:
                            continue
                        agg = {q: np.nanmean(np.vstack([d[q] for d in ds]), 0) for q in ds[0] if q != 'chosen_set_point'}
                        row = {'group': gname, 'benchmark': b, 'model_set': sname, 'outcome': out_k, 'predictor': p,
                               'k': k, 'targets': len(ds)}
                        for q, v in agg.items():
                            row.update(lib.summ(v, q))
                        for ref in ['raw_median', 'dreval_norm']:
                            rd = [dec_store[(b, sname, pr, ref, out_k, k)] for pr in pairs
                                  if (b, sname, pr, ref, out_k, k) in dec_store]
                            if p != ref and len(rd) == len(ds):
                                dd = np.nanmean(np.vstack([r_['chosen'] for r_ in ds]), 0) - \
                                    np.nanmean(np.vstack([r_['chosen'] for r_ in rd]), 0)
                                row.update(lib.summ(dd, f'chosen_minus_{ref}'))
                                row[f'p_chosen_minus_{ref}'] = pboot(dd)
                        drow.append(row)
    for sname in ['all', 'learned']:
        for out_k in ['selB_c', 'raw_median_c']:
            for meth in ['spearman', 'pearson']:
                for cond in ['full', 'cond']:
                    Z = {}
                    for p in preds:
                        zs = []
                        for b in ['leaderboard', 'retrain', 'improve']:
                            key = (b, sname if not (b == 'improve' and sname == 'learned') else 'all', out_k, meth, cond, p)
                            if key in bench_z:
                                zs.append(bench_z[key])
                        if len(zs) < 3:
                            continue                     # overall only where all three benchmarks have the statistic
                        z = np.nanmean(np.vstack(zs), 0)
                        Z[p] = z
                        lo, hi = lib.ci(np.tanh(z))
                        blo, bhi = bc_interval(np.tanh(z))
                        rows.append({'group': 'overall', 'benchmark': 'overall', 'model_set': sname, 'outcome': out_k,
                                     'method': meth, 'interval': cond, 'predictor': p, 'targets': len(zs),
                                     'r_pooled': float(np.tanh(z[0])), 'lo': lo, 'hi': hi, 'bc_lo': blo, 'bc_hi': bhi,
                                     'boot_median': float(np.nanmedian(np.tanh(z[1:]))), 'p_gt0': pboot(z)})
                    contrast_rows('overall', sname, out_k, meth, cond, Z, 3)
                    # overall over the two benchmarks with DrEval's metric
                    Z2 = {}
                    for p in ['raw_median', 'dreval_norm', 'S', 'sel_mean', 'S_lin', 'A']:
                        zs = [bench_z.get((b, sname, out_k, meth, cond, p)) for b in ['leaderboard', 'retrain']]
                        if any(z_ is None for z_ in zs):
                            continue
                        Z2[p] = np.nanmean(np.vstack(zs), 0)
                        lo, hi = lib.ci(np.tanh(Z2[p]))
                        rows.append({'group': 'overall (DrEval benchmarks)', 'benchmark': 'overall', 'model_set': sname,
                                     'outcome': out_k, 'method': meth, 'interval': cond, 'predictor': p, 'targets': 2,
                                     'r_pooled': float(np.tanh(Z2[p][0])), 'lo': lo, 'hi': hi,
                                     'p_gt0': pboot(Z2[p])})
                    contrast_rows('overall (DrEval benchmarks)', sname, out_k, meth, cond, Z2, 2)
    return pd.DataFrame(rows), pd.DataFrame(crow), pd.DataFrame(drow)


def per_target_contrasts(store, recs):
    """Per target: S minus raw_median correlation (Fisher z), paired bootstrap."""
    rows = []
    for r in recs:
        for sname in ['all', 'learned', 'all_plus_floor']:
            for out_k in ['selB_c', 'raw_median_c']:
                for meth in ['spearman', 'pearson']:
                    for cond in ['full', 'cond']:
                        a = store.get((r['benchmark'], sname, r['pair'], 'S', out_k, meth, cond))
                        b = store.get((r['benchmark'], sname, r['pair'], 'raw_median', out_k, meth, cond))
                        if a is None or b is None:
                            continue
                        d = fz(a) - fz(b)
                        dr = a - b
                        rows.append({'benchmark': r['benchmark'], 'pair': r['pair'], 'primary': r['primary'],
                                     'model_set': sname, 'outcome': out_k, 'method': meth, 'interval': cond,
                                     'r_S': float(a[0]), 'r_raw': float(b[0]), 'dr': float(dr[0]),
                                     'dr_lo': lib.ci(dr)[0], 'dr_hi': lib.ci(dr)[1], 'dz': float(d[0]),
                                     'dz_lo': lib.ci(d)[0], 'dz_hi': lib.ci(d)[1], 'p': pboot(d)})
    return pd.DataFrame(rows)


def verify(recs):
    """Point values (and, for the leaderboard, all draws) against 80's paired arrays, and within statistics against
    78/91 (capture_by_model.csv, lineage_by_model.csv)."""
    rows = []
    ix = pd.read_csv(lib.OUT2 / 'paired' / 'pairs.csv')
    for r in recs:
        if r['benchmark'] == 'leaderboard':
            f = lib.OUT2 / 'paired' / 'arrays' / f"leaderboard__{r['k']}.npz"
        elif r['benchmark'] == 'retrain':
            k = ix[(ix.benchmark == 'retrain') & (ix.pair == r['pair'])].k
            f = lib.OUT2 / 'paired' / 'arrays' / f'retrain__{int(k.iloc[0])}.npz' if len(k) else None
        else:
            k = ix[(ix.benchmark == 'improve') & (ix.pair == r['pair'])].k
            f = lib.OUT2 / 'paired' / 'arrays' / f'improve__{int(k.iloc[0])}.npz' if len(k) else None
        if f is None or not f.exists():
            continue
        models80, _, S80 = lib.load_paired(f)
        for m in r['models_p']:
            if m not in models80:
                continue
            i, j = r['models_p'].index(m), models80.index(m)
            for st in ['raw', 'mediated', 'selBOTH_sym']:
                for kind in ['sum_w', 'sum_c', 'n']:
                    a = r['P'][f'sum|{st}|{kind}'][:, i]
                    b = S80[st][kind][:, j]
                    rows.append({'benchmark': r['benchmark'], 'pair': r['pair'], 'model': m, 'stat': f'{st}|{kind}',
                                 'point_absdiff': float(abs(a[0] - b[0])),
                                 'all_draws_maxabsdiff': float(np.nanmax(np.abs(a - b))) if r['benchmark'] == 'leaderboard' else np.nan})
    V = pd.DataFrame(rows)
    # within statistics against 78 (capture) and 91 (lineage)
    cap = pd.read_csv(lib.OUT2 / 'capture' / 'capture_by_model.csv')
    lin = pd.read_csv(lib.OUT2 / 'lineage' / 'lineage_by_model.csv')
    wrows = []
    seen = set()
    for r in recs:
        key = (r['benchmark'], r['setting'])
        if key in seen:
            continue
        seen.add(key)
        for m in r['models_w']:
            i = r['models_w'].index(m)
            c = cap[(cap.benchmark == r['benchmark']) & (cap.setting == r['setting']) & (cap.model == m)]
            l_ = lin[(lin.benchmark == r['benchmark']) & (lin.setting == r['setting']) & (lin.model == m)]
            if len(c):
                for a, b in [('A', 'axis_capture'), ('S', 'selective_capture'), ('raw_median', 'median_raw')]:
                    wrows.append({'benchmark': r['benchmark'], 'setting': r['setting'], 'model': m, 'stat': a,
                                  'ours': float(r['W'][a][0, i]), 'reference': float(c[b].iloc[0]), 'source': '78'})
            if len(l_):
                for a, b in [('S_lin', 'selective_capture_lin'), ('sel_lin_mean', 'mean_selective_lin')]:
                    wrows.append({'benchmark': r['benchmark'], 'setting': r['setting'], 'model': m, 'stat': a,
                                  'ours': float(r['W'][a][0, i]), 'reference': float(l_[b].iloc[0]), 'source': '91'})
    Wv = pd.DataFrame(wrows)
    if len(Wv):
        Wv['absdiff'] = (Wv.ours - Wv.reference).abs()
    # DrEval normalized point values against 88 (leaderboard) and 59 (retraining)
    nrows = []
    rk = pd.read_csv(lib.OUT2 / 'rankings' / 'rank_intervals.csv')
    rk = rk[rk.metric == 'dreval_normalized'].set_index('model').score
    npr = pd.read_csv(lib.RES / 'revision_retrain' / 'normalized_pearson.csv')
    seen = set()
    for r in recs:
        key = (r['benchmark'], r['setting'])
        if key in seen or r['benchmark'] == 'improve':
            continue
        seen.add(key)
        for m in r['models_w']:
            v = float(r['W']['dreval_norm'][0, r['models_w'].index(m)])
            if not np.isfinite(v):
                continue
            ref = rk.get(m, np.nan) if r['benchmark'] == 'leaderboard' else \
                npr[(npr.setting == r['setting']) & (npr.model == m)].normalized_pearson
            ref = float(ref.iloc[0]) if isinstance(ref, pd.Series) and len(ref) else (float(ref) if not isinstance(ref, pd.Series) else np.nan)
            nrows.append({'benchmark': r['benchmark'], 'setting': r['setting'], 'model': m, 'ours': v, 'reference': ref,
                          'absdiff': abs(v - ref)})
    return V, Wv, pd.DataFrame(nrows)


def figure_csvs(Cp, Cc, D, Ctg, recs):
    """One CSV per intended panel (with intervals)."""
    # panel a: per-model scatter, leaderboard primary targets: within raw_median, S, DrEval's metric vs target selB
    rows = []
    for r in recs:
        if not (r['benchmark'] == 'leaderboard' and r['primary']) and not (r['benchmark'] == 'retrain' and r['primary']):
            continue
        for m in model_sets(r)['all']:
            if m not in r['models_w'] or m not in r['models_p']:
                continue
            i, j = r['models_w'].index(m), r['models_p'].index(m)
            ag = m in (LB_AGNOSTIC if r['benchmark'] == 'leaderboard' else RT_AGNOSTIC)
            row = {'benchmark': r['benchmark'], 'setting': r['setting'], 'pair': r['pair'], 'target': r['target'],
                   'model': m, 'drug_agnostic': ag}
            for k in ['raw_median', 'S', 'A', 'sel_mean', 'dreval_norm', 'S_lin']:
                row.update(lib.summ(r['W'][k][:, i], k))
            for k in ['selB_c', 'raw_median_c']:
                row.update(lib.summ(r['P'][k][:, j], k))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'fig_a_model_scatter.csv', index=False, float_format='%.5g')
    # panel b: pooled Spearman correlation of each within-dataset statistic with target selective-both and with the
    # target's raw score, per benchmark and overall (joint model x drug x cell intervals)
    keep = ['leaderboard', 'retrain', 'improve', 'overall', 'retrain:GDSC2_AUC', 'retrain:GDSC2_LNIC50']
    Cp[(Cp.method == 'spearman') & (Cp.interval == 'full') & Cp.predictor.isin(PREDICTORS) & Cp.group.isin(keep)].to_csv(
        OUT / 'fig_b_pooled_correlations.csv', index=False, float_format='%.5g')
    # panel c: S (and the others) minus the raw score, per target and pooled
    Ctg[(Ctg.method == 'spearman')].to_csv(OUT / 'fig_c_contrast_by_target.csv', index=False, float_format='%.5g')
    Cc[(Cc.method == 'spearman') & (Cc.interval == 'full')].to_csv(OUT / 'fig_c_contrast_pooled.csv', index=False,
                                                                   float_format='%.5g')
    # panel d: decision regret (k = 1, 3, 5) by statistic
    D[D.predictor.isin(['raw_median', 'dreval_norm', 'S', 'sel_mean', 'S_lin', 'A']) & D.group.isin(keep)].to_csv(
        OUT / 'fig_d_decision_regret.csv', index=False, float_format='%.5g')
    # sensitivity: every pooling group (leaderboard target families, IMPROVE shared-panel axes, per source)
    Cp[(Cp.method == 'spearman') & (Cp.interval == 'full') & Cp.predictor.isin(['raw_median', 'dreval_norm', 'S',
                                                                                'sel_mean', 'S_lin', 'A'])].to_csv(
        OUT / 'supp_sensitivity_groups.csv', index=False, float_format='%.5g')


def key_numbers(Cp, Cc, D, Ctg, V, Wv, Nv):
    kn = {}
    sub = Cp[(Cp.method == 'spearman')]
    for r in sub.itertuples():
        if r.predictor not in ('raw_median', 'dreval_norm', 'A', 'S', 'sel_mean', 'S_lin', 'raw_median_w', 'S_w'):
            continue
        key = f'corr|{r.group}|{r.model_set}|{r.outcome}|{r.predictor}|{r.interval}'
        kn[key] = {'r': r.r_pooled, 'lo': r.lo, 'hi': r.hi, 'targets': int(r.targets)}
        if hasattr(r, 'bc_lo') and np.isfinite(getattr(r, 'bc_lo', np.nan)):
            kn[key].update({'bc_lo': r.bc_lo, 'bc_hi': r.bc_hi})
    for r in Cc[(Cc.method == 'spearman')].itertuples():
        if r.predictor not in ('dreval_norm', 'A', 'S', 'sel_mean', 'S_lin', 'S_w'):
            continue
        kn[f'contrast|{r.group}|{r.model_set}|{r.outcome}|{r.predictor}-{r.vs}|{r.interval}'] = {
            'dr': r.dr, 'dr_lo': r.dr_lo, 'dr_hi': r.dr_hi, 'dz': r.dz, 'dz_lo': r.dz_lo, 'dz_hi': r.dz_hi, 'p': r.p}
    cols = ['chosen', 'chosen_lo', 'chosen_hi', 'best', 'regret', 'regret_lo', 'regret_hi', 'regret_redrawn',
            'regret_redrawn_lo', 'regret_redrawn_hi', 'mean_all', 'gain_share', 'gain_share_lo', 'gain_share_hi',
            'chosen_minus_raw_median', 'chosen_minus_raw_median_lo', 'chosen_minus_raw_median_hi',
            'p_chosen_minus_raw_median']
    for _, r in D.iterrows():
        if r.predictor in ('raw_median', 'S', 'sel_mean', 'S_lin', 'dreval_norm') and r.model_set in ('all', 'learned'):
            kn[f'decision|{r.group}|{r.model_set}|{r.outcome}|{r.predictor}|k{r.k}'] = {
                k: (None if pd.isna(r.get(k)) else float(r.get(k))) for k in cols if k in r.index}
    # headline numbers (pre-stated primary estimands; Spearman, joint model x drug x cell intervals unless 'cond')
    H = {}
    for g in ['leaderboard', 'retrain', 'improve', 'overall', 'retrain:GDSC2_AUC', 'retrain:GDSC2_LNIC50']:
        for sname in ['all', 'learned']:
            for out_k in ['selB_c', 'raw_median_c']:
                for p_ in ['raw_median', 'dreval_norm', 'A', 'S', 'sel_mean', 'S_lin']:
                    k = f'corr|{g}|{sname}|{out_k}|{p_}|full'
                    if k in kn:
                        H[f'{g}|{sname}|{out_k}|{p_}'] = kn[k]
                for p_ in ['S', 'dreval_norm', 'sel_mean']:
                    for ref in ['raw_median', 'dreval_norm']:
                        k = f'contrast|{g}|{sname}|{out_k}|{p_}-{ref}|full'
                        if k in kn:
                            H[f'{g}|{sname}|{out_k}|{p_}-{ref}'] = kn[k]
    for k, v in kn.items():
        if k.startswith('decision|') and k.split('|')[1] in ('leaderboard', 'retrain', 'improve', 'retrain:GDSC2_AUC',
                                                            'retrain:GDSC2_LNIC50') and k.split('|')[4] in ('raw_median', 'S', 'dreval_norm'):
            H[k] = v
    kn = {'headline': H, **kn}
    kn['verification'] = {'paired_point_max_absdiff': float(V.point_absdiff.max()) if len(V) else None,
                          'paired_leaderboard_all_draws_max_absdiff': float(V.all_draws_maxabsdiff.max()) if len(V) else None,
                          'within_point_max_absdiff_vs_78_91_csv': float(Wv.absdiff.max()) if len(Wv) else None,
                          'normalized_point_max_absdiff_vs_88_59': float(Nv.absdiff.max()) if len(Nv) else None}
    return kn

# ----------------------------------------------------------------------------------------------------------------
# (d) patient samples: are multi-model predictions on BeatAML2 or PDX_Bruna released?
def patient_check():
    """Inspect DrEval's local release (all_results.zip member list; evaluation_results.csv rand_setting values) and the
    public Zenodo listings of DrEval's dataset record versions (file names only; nothing is downloaded)."""
    import urllib.request
    import zipfile
    dre = lib.PROJECT / 'data' / 'dreval'
    out = {'checked_utc': pd.Timestamp.utcnow().isoformat()}
    z = zipfile.ZipFile(dre / 'all_results.zip')
    names = [n for n in z.namelist() if not n.startswith('__MACOSX')]
    out['all_results_members'] = names
    ev = pd.read_csv(z.open('all_results/evaluation_results.csv'), usecols=['algorithm', 'rand_setting', 'test_mode'])
    out['evaluation_results_rand_settings'] = sorted(ev.rand_setting.unique())
    out['evaluation_results_algorithms'] = sorted(ev.algorithm.unique())
    out['local_parquet_targets'] = sorted(p.stem for p in (dre / 'all_results_parquet').glob('*.parquet'))
    rec = json.loads((dre / 'zenodo_record.json').read_text())
    out['local_record'] = {'doi': rec.get('doi'), 'files': [(f['key'], f['size']) for f in rec['files']]}
    remote = {}
    for rid in ['23159774', '20624451', '17611663', '17600096', '17153749']:
        try:
            d = json.loads(urllib.request.urlopen(f'https://zenodo.org/api/records/{rid}', timeout=60).read())
            remote[rid] = {'created': d.get('created'), 'files': [(f['key'], f['size']) for f in d.get('files', [])]}
        except Exception as e:  # noqa: BLE001
            remote[rid] = {'error': repr(e)}
    out['zenodo_dreval_dataset_versions'] = remote
    sizes = {tuple(x) for v in remote.values() for x in v.get('files', []) if x[0] == 'all_results.zip'}
    out['distinct_all_results_zip_sizes'] = sorted(s for _, s in sizes)
    out['patient_zip_members_checked'] = {
        'BeatAML2.zip': 'responses (BeatAML2.csv), gene expression, pathway features, drug features only '
                        '(central directory read by HTTP range request on 2026-10-15; 349 members, no predictions)',
        'PDX_Bruna.zip': 'responses (PDX_Bruna.csv), expression, copy number, drug features only (222 members, no '
                         'predictions)'}
    out['improve'] = 'IMPROVE CSA test predictions (Zenodo 15258883/15851723) cover CCLE, CTRPv2, GDSCv1, GDSCv2 and gCSI only'
    out['conclusion'] = ('No public DrEval or IMPROVE release contains per-model predictions on BeatAML2 or PDX_Bruna: '
                         "DrEval's all_results (identical in every dataset-record version) holds cross-study predictions "
                         'for CCLE, CTRPv1, GDSC1 and GDSC2 only, and the BeatAML2/PDX_Bruna archives hold responses and '
                         'features. Part (d) is therefore not done and nothing is substituted.')
    lib.write_json(OUT / 'patient_predictions_check.json', out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compute', action='store_true')
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--patient-check', action='store_true')
    ap.add_argument('--parts', nargs='*', default=['leaderboard', 'retrain', 'improve'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.patient_check:
        print(patient_check()['conclusion'], flush=True)
    if args.compute:
        compute(args.parts, t0)
    if args.analyse:
        recs, store, dec_store, C, clusters = analyse()
        Cp, Cc, D = pool(store, dec_store, recs, clusters)
        Ctg = per_target_contrasts(store, recs)
        Cp.to_csv(OUT / 'correlations_pooled.csv', index=False, float_format='%.5g')
        Cc.to_csv(OUT / 'contrast_pooled.csv', index=False, float_format='%.5g')
        D.to_csv(OUT / 'decision_pooled.csv', index=False, float_format='%.5g')
        Ctg.to_csv(OUT / 'contrast_by_target.csv', index=False, float_format='%.5g')
        V, Wv, Nv = verify(recs)
        V.to_csv(OUT / 'verification_paired.csv', index=False, float_format='%.6g')
        Wv.to_csv(OUT / 'verification_within.csv', index=False, float_format='%.6g')
        Nv.to_csv(OUT / 'verification_normalized.csv', index=False, float_format='%.6g')
        figure_csvs(Cp, Cc, D, Ctg, recs)
        kn = key_numbers(Cp, Cc, D, Ctg, V, Wv, Nv)
        allk = json.loads(KEYS.read_text()) if KEYS.exists() else {}
        allk['transfer_prediction'] = kn
        pc = OUT / 'patient_predictions_check.json'
        if pc.exists():
            allk['transfer_prediction_patient_check'] = json.loads(pc.read_text())
        lib.write_json(KEYS, allk)
        lib.write_json(OUT / 'run.json', {'seconds': round(time.time() - t0), 'B': lib.B, 'records': len(recs),
                                          'clusters': {b: v[0] for b, v in clusters.items()}})
        pd.set_option('display.width', 250)
        pd.set_option('display.max_rows', 500)
        keep = ['leaderboard', 'retrain', 'improve', 'overall', 'retrain:GDSC2_AUC', 'retrain:GDSC2_LNIC50']
        sel = Cp[(Cp.method == 'spearman') & (Cp.interval == 'full') & Cp.group.isin(keep) &
                 Cp.predictor.isin(['raw_median', 'dreval_norm', 'A', 'S', 'sel_mean', 'S_lin'])]
        print(sel.drop(columns=['benchmark', 'method', 'interval']).round(3).to_string(index=False))
        print(Cc[(Cc.method == 'spearman') & (Cc.interval == 'full') & Cc.group.isin(keep + ['overall (DrEval benchmarks)']) &
                 (Cc.predictor.isin(['S', 'S_lin', 'sel_mean', 'dreval_norm']))].drop(columns=['method', 'interval']).round(3).to_string(index=False))
        print(D[(D.predictor.isin(['raw_median', 'S', 'dreval_norm'])) & D.model_set.isin(['all', 'learned']) & D.group.isin(keep)][
            ['group', 'model_set', 'outcome', 'predictor', 'k', 'chosen', 'best', 'regret', 'regret_lo', 'regret_hi',
             'mean_all', 'gain_share', 'chosen_minus_raw_median', 'chosen_minus_raw_median_lo',
             'chosen_minus_raw_median_hi']].round(3).to_string(index=False))
        print('verification', kn['verification'])
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
