"""Score the drevalpy retraining runs (58) like the paper's Fig. 2: within the training dataset and, with identical
out-of-fold predictions on matched cells, in the other DrEval datasets with the same endpoint.

For each setting (training dataset x endpoint) and panel of models (cells common to every model of the panel):
  within  (leave-cell-line-out, out-of-fold predictions; decomposition of the training dataset on the panel cells):
          per drug raw r, selective-T (partial given the screen's leave-drug-out axis g; equals selective-BOTH within
          the training dataset), alignment r_pg, target loading r_zg, mediated term r_pg r_zg, selective term
          raw - r_pg r_zg, and the model-axis-adjusted score (partial given g and the model's own leave-drug-out axis
          of per-drug z-scored predictions). Fitness-mediated share = mean mediated / mean raw; the oracle (p = z)
          reference is mean r_zg^2.
  normalized
          DrEval's normalized Pearson: per CV split, Pearson over all test pairs of (y - naive) and (prediction -
          naive), naive = the NaiveMeanEffectsPredictor of the same splits (drug mean + tissue effect fitted on the
          training cells); mean (and SD) over the 10 splits. Validated against DrEval's leaderboard values.
  paired  for every target dataset T: the same out-of-fold predictions scored against the training dataset S and
          against T on, for each drug, the identical cells measured in both (35_dreval_crossstudy.score_pair logic;
          only cells with out-of-fold predictions; S and T axes from the full datasets as in 35/37). Per drug and for
          S ("within") and T ("cross"): raw, selective-T (given that screen's axis), selective-BOTH (given g_S and
          g_T), model-axis-adjusted (given that screen's axis and the model's own axis over the paired drugs and
          cells), r_pg, r_zg, mediated and selective terms. Reference 'ref:source_measurement': the training dataset's
          own measured profile (rank-normal z) scored the same way (one replicate measurement in another screen); each
          model's cross selective-BOTH is also expressed as a fraction of it (ratio of drug means and of drug medians,
          on the drugs where both are defined, with two-way bootstrap intervals).
The model-axis-adjusted score is set to undefined for drug-agnostic predictors (FitnessOnlyRidge, NaiveMeanEffects-
Predictor, NaiveTissueMeanPredictor): their predictions equal their own axis up to per-drug affine maps, so the residual
left after partialling it is numerical noise from per-drug z-scoring over slightly different measured-cell subsets.
Uncertainty: two-way bootstrap (B = 1000) resampling cell lines (one multinomial draw over the cell universe,
applied to every drug, so drugs sharing cells stay dependent) and drugs independently; percentile 95% intervals,
bootstrap SE and Wald P values; Holm adjustment within families defined by one panel and one metric (target effect
within model class; feature source; model class; and, separately for each of this study's ridges, its comparisons with
every other model across evaluation datasets, as in R1-M4), per setting and over all settings.
Also: Wilcoxon signed-rank P over drugs for every contrast (drug-level, ignores shared cells), the provenance of
target-dataset predictions (out-of-fold vs fold-averaged drevalpy cross-study predictions for cells absent from the
training dataset; the latter use the target dataset's expression file and are never used in the paired analysis) and
an unpaired score of the fold-averaged predictions on those new cells (supplementary).

Outputs (results/revision_retrain/): within_per_drug.csv.gz, within_summary.csv, normalized_pearson.csv,
normalized_validation.csv, paired_per_drug.csv.gz, paired_summary.csv, paired_pooled.csv, contrasts.csv,
cell_provenance.csv, new_cells_unpaired.csv, dataset_fitness.csv, score_run.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402

OUT = PROJECT / 'results' / 'revision_retrain'
RUNS = OUT / 'drevalpy_runs'
INP = OUT / 'inputs'
LB = PROJECT / 'data' / 'dreval' / 'leaderboard_Oct26'
DATASETS = ['CTRPv2', 'CTRPv1', 'GDSC1', 'GDSC2', 'CCLE']
SETTINGS = {'GDSC2_AUC': ('GDSC2', 'auc'), 'GDSC2_LNIC50': ('GDSC2', 'lnic50'), 'CTRPv2_LNIC50': ('CTRPv2', 'lnic50'),
            'CTRPv2_AUC': ('CTRPv2', 'auc'), 'GDSC1_AUC': ('GDSC1', 'auc')}
B = 1000
MIN_CELLS = 30
SEED = 20261014

# model -> (class, training target, features)
DL = 'DrEval landmark expression'
DLF = 'DrEval landmark expression + Morgan fingerprints'
DM = 'DepMap 24Q4 expression (top 5,000)'
MODEL_INFO = {
    'NaiveMeanEffectsPredictor': ('naive (drug-agnostic within drug)', 'raw response', 'tissue labels'),
    'NaiveTissueDrugMeanPredictor': ('naive', 'raw response', 'tissue labels'),
    'FitnessOnlyRidge': ('drug-agnostic (fitness only)', 'cell mean z', DL),
    'FitnessOnlyRidgeDm': ('drug-agnostic (fitness only)', 'cell mean z', DM),
    'PerDrugRidgeRaw': ('per-drug kernel ridge', 'raw (z)', DL),
    'PerDrugRidgeSelective': ('per-drug kernel ridge', 'selective', DL),
    'PerDrugRidgeRawDm': ('per-drug kernel ridge', 'raw (z)', DM),
    'PerDrugRidgeSelectiveDm': ('per-drug kernel ridge', 'selective', DM),
    'GradientBoosting': ('global gradient boosting', 'raw response', DLF),
    'GradientBoostingZ': ('global gradient boosting', 'raw (z)', DLF),
    'GradientBoostingSelective': ('global gradient boosting', 'selective', DLF),
    'RandomForest': ('global random forest', 'raw response', DLF),
    'SimpleNeuralNetwork': ('global neural network', 'raw response', DLF),
    'SimpleNeuralNetworkSelective': ('global neural network', 'selective', DLF),
    'EnsembleMF': ('factorization ensemble', 'raw response', 'DrEval landmark expression + Morgan + drug id'),
    'EnsembleMFSelective': ('factorization ensemble', 'selective', 'DrEval landmark expression + Morgan + drug id'),
    'ref:source_measurement': ("reference: training screen's own measurement", 'measured (rank-normal z)',
                               'none (not a model)'),
}
DREVAL_LEARNED = ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting']
REF = 'ref:source_measurement'
DRUG_AGNOSTIC = ['FitnessOnlyRidge', 'FitnessOnlyRidgeDm', 'NaiveMeanEffectsPredictor', 'LB:NaiveMeanEffectsPredictor',
                 'LB:NaiveTissueMeanPredictor']
FEATSRC = ['FitnessOnlyRidge', 'PerDrugRidgeRaw', 'PerDrugRidgeSelective', 'FitnessOnlyRidgeDm', 'PerDrugRidgeRawDm',
           'PerDrugRidgeSelectiveDm']


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------
# data
def responses():
    return pd.read_parquet(INP / 'responses.parquet')


def cell_map(ds):
    n = pd.read_csv(INP / 'cell_names.csv', dtype=str)
    return n[n.dataset == ds].set_index('cell_line_name').cellosaurus_id


def read_split_files(files, cmap):
    parts = []
    for f in files:
        k = int(f.stem.rsplit('_', 1)[1])
        d = pd.read_csv(f, dtype={'pubchem_id': str, 'cell_line_name': str})
        parts.append(d.assign(split=k))
    d = pd.concat(parts, ignore_index=True)
    d['cell'] = d.cell_line_name.map(cmap)
    return d.dropna(subset=['cell']).rename(columns={'pubchem_id': 'drug'})


def load_oof(setting):
    """Out-of-fold predictions of every complete model of a drevalpy run (+ the leaderboard for CTRPv2 ln IC50)."""
    ds, ep = SETTINGS[setting]
    base = RUNS / setting / ds / 'LCO'
    cmap = cell_map(ds)
    oof, raw_rows, incomplete = {}, {}, {}
    sources = []
    if base.exists():
        sources += [(md.name, md) for md in sorted(base.iterdir()) if md.is_dir() and md.name != 'splits']
    if setting == 'CTRPv2_LNIC50':
        # NaivePredictor and NaiveDrugMeanPredictor are constant within a drug (per-drug metrics undefined; as 28)
        sources += [(f'LB:{md.name}', md) for md in sorted((LB / 'LCO').iterdir()) if md.is_dir()
                    and md.name not in ('NaivePredictor', 'NaiveDrugMeanPredictor')]
    for name, md in sources:
        fs = sorted((md / 'predictions').glob('predictions_split_*.csv'))
        if len(fs) < 10:
            incomplete[name] = len(fs)
            continue
        d = read_split_files(fs, cmap)
        assert (d.groupby('cell').split.nunique() == 1).all(), name
        g = d.groupby(['drug', 'cell'], as_index=False).agg(pred=('predictions', 'mean'), split=('split', 'first'))
        if g.pred.groupby(g.drug).std().fillna(0).max() < 1e-12:      # constant within every drug
            incomplete[name] = 'constant'
            continue
        # rows as written by drevalpy (duplicate measurements kept), for DrEval's normalized metric
        raw_rows[name] = d[['drug', 'cell', 'predictions', 'response', 'split']].rename(
            columns={'predictions': 'pred', 'response': 'y'})
        oof[name] = g.assign(prediction=-g.pred)[['drug', 'cell', 'prediction', 'split']]
    return oof, raw_rows, incomplete


def load_cross(setting, model, target, source_cells):
    """drevalpy cross-study predictions of one model for a target dataset, averaged over the 10 split models for cells
    absent from the training dataset; target cells that are training-dataset cells are only flagged."""
    ds, ep = SETTINGS[setting]
    md = RUNS / setting / ds / 'LCO' / model / 'cross_study'
    fs = sorted(md.glob(f'cross_study_{target}_split_*.csv'))
    if not fs:
        return None, {}
    d = read_split_files(fs, cell_map(target))
    d = d[np.isfinite(d.predictions)]
    src = d.cell.isin(source_cells)
    new = d[~src].groupby(['drug', 'cell'], as_index=False).agg(prediction=('predictions', 'mean'),
                                                                n_models=('split', 'nunique'))
    new['prediction'] = -new.prediction
    info = {'cross_files': len(fs), 'target_cells_fold_averaged': int(new.cell.nunique()),
            'source_cells_repredicted_with_target_features_unused': int(d[src].cell.nunique()),
            'fold_models_per_new_cell_min': int(new.n_models.min()) if len(new) else 0}
    return new, info


# ----------------------------------------------------------------------------------------------------------------
# bootstrap engine: weighted (co)variances for many replicates at once
def weighted_cov(X, W, pairs):
    """X: n x V, W: R x n non-negative weights, pairs: P x 2 column indices -> R x P weighted covariances."""
    S0 = W.sum(1)
    M1 = (W @ X) / S0[:, None]
    prod = X[:, pairs[:, 0]] * X[:, pairs[:, 1]]
    M2 = (W @ prod) / S0[:, None]
    return M2 - M1[:, pairs[:, 0]] * M1[:, pairs[:, 1]]


class Cov:
    def __init__(self, names):
        self.names = list(names)
        self.ix = {n: i for i, n in enumerate(self.names)}
        self.pairs = []
        self.pix = {}

    def need(self, a, b):
        key = tuple(sorted((a, b)))
        if key not in self.pix:
            self.pix[key] = len(self.pairs)
            self.pairs.append((self.ix[key[0]], self.ix[key[1]]))

    def compute(self, X, W):
        self.C = weighted_cov(X, W, np.array(self.pairs, dtype=int))

    def c(self, a, b):
        return self.C[:, self.pix[tuple(sorted((a, b)))]]

    def corr(self, a, b):
        with np.errstate(invalid='ignore', divide='ignore'):
            return self.c(a, b) / np.sqrt(self.c(a, a) * self.c(b, b))

    def pcorr(self, a, b, covs):
        """Partial correlation of a and b given covs (Schur complement); NaN if a or b is (near) collinear."""
        k = len(covs)
        S = np.empty((self.C.shape[0], k, k))
        for i, u in enumerate(covs):
            for j, v in enumerate(covs):
                S[:, i, j] = self.c(u, v)
        sa = np.stack([self.c(a, u) for u in covs], 1)
        sb = np.stack([self.c(b, u) for u in covs], 1)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        ab = self.c(a, b) - np.einsum('ri,rij,rj->r', sa, Sinv, sb)
        aa = self.c(a, a) - np.einsum('ri,rij,rj->r', sa, Sinv, sa)
        bb = self.c(b, b) - np.einsum('ri,rij,rj->r', sb, Sinv, sb)
        bad = (aa <= 1e-10 * self.c(a, a)) | (bb <= 1e-10 * self.c(b, b))
        with np.errstate(invalid='ignore', divide='ignore'):
            out = ab / np.sqrt(aa * bb)
        out[bad] = np.nan
        return out


def zscore_axis(P):
    """Model's own leave-drug-out axis: per-drug z-scored predictions, mean over all other drugs (per cell)."""
    v = P.values.astype(float)
    mu = np.nanmean(v, axis=1, keepdims=True)
    sd = np.nanstd(v, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = (v - mu) / np.where(sd > 1e-12, sd, np.nan)
    ok = np.isfinite(z)
    tot, cnt = np.nansum(z, axis=0), ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        h = (tot[None] - np.where(ok, z, 0)) / (cnt[None] - ok)
    return pd.DataFrame(h, index=P.index, columns=P.columns)


def draw_weights(n_cells, n_drugs, rng):
    Wc = np.ones((B + 1, n_cells))
    Wc[1:] = rng.multinomial(n_cells, np.full(n_cells, 1.0 / n_cells), size=B)
    Vd = np.ones((B + 1, n_drugs))
    Vd[1:] = rng.multinomial(n_drugs, np.full(n_drugs, 1.0 / n_drugs), size=B)
    return Wc, Vd


# ----------------------------------------------------------------------------------------------------------------
def score_within(models, Z, G, Pm, Hm, cells, rng):
    """Per-drug within-dataset statistics for R = B + 1 replicates. Returns {model: {stat: R x D}}, drugs, Vd."""
    drugs = list(Z.index)
    cidx = {c: i for i, c in enumerate(cells)}
    Wc, Vd = draw_weights(len(cells), len(drugs), rng)
    stats = ['raw', 'r_pg', 'r_zg', 'mediated', 'selective_term', 'selective', 'adjusted']
    res = {m: {s: np.full((B + 1, len(drugs)), np.nan, dtype=np.float32) for s in stats} for m in models}
    ncell = np.zeros(len(drugs), int)
    for j, d in enumerate(drugs):
        z, g = Z.loc[d].reindex(cells).values, G.loc[d].reindex(cells).values
        have = [m for m in models if d in Pm[m].index]
        cols = {'z': z, 'g': g}
        for m in have:
            cols[f'p|{m}'] = Pm[m].loc[d].reindex(cells).values
            cols[f'h|{m}'] = Hm[m].loc[d].reindex(cells).values if d in Hm[m].index else np.full(len(cells), np.nan)
        X = np.column_stack(list(cols.values()))
        ok = np.isfinite(X[:, :2]).all(1)
        pcols = [i for i, n in enumerate(cols) if n.startswith('p|')]
        ok &= np.isfinite(X[:, pcols]).all(1) if pcols else ok
        hm = [m for m in have if np.isfinite(cols[f'h|{m}'][ok]).all()]
        if ok.sum() < MIN_CELLS:
            continue
        ncell[j] = ok.sum()
        names = list(cols)
        X = X[ok]
        X[~np.isfinite(X)] = 0.0
        cv = Cov(names)
        for a, b in [('z', 'z'), ('g', 'g'), ('z', 'g')]:
            cv.need(a, b)
        for m in have:
            p, h = f'p|{m}', f'h|{m}'
            for a, b in [(p, p), (p, 'z'), (p, 'g'), (h, h), (h, 'z'), (h, 'g'), (p, h)]:
                cv.need(a, b)
        W = Wc[:, [cidx[c] for c in np.asarray(cells)[ok]]]
        cv.compute(X, W)
        rzg = cv.corr('z', 'g')
        for m in have:
            p, h = f'p|{m}', f'h|{m}'
            if np.std(cols[p][ok]) < 1e-12:
                continue
            raw, rpg = cv.corr(p, 'z'), cv.corr(p, 'g')
            r = res[m]
            r['raw'][:, j], r['r_pg'][:, j], r['r_zg'][:, j] = raw, rpg, rzg
            r['mediated'][:, j] = rpg * rzg
            r['selective_term'][:, j] = raw - rpg * rzg
            r['selective'][:, j] = cv.pcorr(p, 'z', ['g'])
            if m in hm and m not in DRUG_AGNOSTIC:      # undefined for drug-agnostic predictors (see docstring)
                r['adjusted'][:, j] = cv.pcorr(p, 'z', ['g', h])
    return res, drugs, Vd, ncell


def score_paired(models, Zs, Gs, Zt, Gt, Pm, Hm, drugs, cells, rng):
    """Per-drug within-S ('_s') and cross-T ('_t') statistics on matched cells for R = B + 1 replicates."""
    cidx = {c: i for i, c in enumerate(cells)}
    Wc, Vd = draw_weights(len(cells), len(drugs), rng)
    base = ['raw', 'r_pg', 'r_zg', 'mediated', 'selective_term', 'sel_T', 'sel_BOTH', 'adjusted']
    stats = [f'{s}_{w}' for s in base for w in ('s', 't')]
    res = {m: {s: np.full((B + 1, len(drugs)), np.nan, dtype=np.float32) for s in stats} for m in models}
    ncell = np.zeros(len(drugs), int)
    for j, d in enumerate(drugs):
        cols = {'zs': Zs.loc[d].reindex(cells).values, 'gs': Gs.loc[d].reindex(cells).values,
                'zt': Zt.loc[d].reindex(cells).values, 'gt': Gt.loc[d].reindex(cells).values}
        have = [m for m in models if d in Pm[m].index]
        for m in have:
            cols[f'p|{m}'] = Pm[m].loc[d].reindex(cells).values
            cols[f'h|{m}'] = Hm[m].loc[d].reindex(cells).values if d in Hm[m].index else np.full(len(cells), np.nan)
        names = list(cols)
        X = np.column_stack(list(cols.values()))
        ok = np.isfinite(X[:, :4]).all(1)
        pcols = [i for i, n in enumerate(names) if n.startswith('p|')]
        if pcols:
            ok &= np.isfinite(X[:, pcols]).all(1)
        if ok.sum() < MIN_CELLS:
            continue
        ncell[j] = ok.sum()
        hm = [m for m in have if np.isfinite(cols[f'h|{m}'][ok]).all()]
        X = X[ok]
        X[~np.isfinite(X)] = 0.0
        cv = Cov(names)
        for a, b in [('zs', 'zs'), ('gs', 'gs'), ('zt', 'zt'), ('gt', 'gt'), ('zs', 'gs'), ('zt', 'gt'), ('zs', 'gt'),
                     ('zt', 'gs'), ('gs', 'gt')]:
            cv.need(a, b)
        for m in have:
            p, h = f'p|{m}', f'h|{m}'
            for a, b in [(p, p), (p, 'zs'), (p, 'gs'), (p, 'zt'), (p, 'gt'), (h, h), (h, 'zs'), (h, 'gs'),
                         (h, 'zt'), (h, 'gt'), (p, h)]:
                cv.need(a, b)
        W = Wc[:, [cidx[c] for c in np.asarray(cells)[ok]]]
        cv.compute(X, W)
        for m in have:
            p, h = f'p|{m}', f'h|{m}'
            if np.std(cols[p][ok]) < 1e-12:
                continue
            r = res[m]
            for w in ('s', 't'):
                z, g = f'z{w}', f'g{w}'
                raw, rpg, rzg = cv.corr(p, z), cv.corr(p, g), cv.corr(z, g)
                r[f'raw_{w}'][:, j], r[f'r_pg_{w}'][:, j], r[f'r_zg_{w}'][:, j] = raw, rpg, rzg
                r[f'mediated_{w}'][:, j] = rpg * rzg
                r[f'selective_term_{w}'][:, j] = raw - rpg * rzg
                r[f'sel_T_{w}'][:, j] = cv.pcorr(p, z, [g])
                r[f'sel_BOTH_{w}'][:, j] = cv.pcorr(p, z, ['gs', 'gt'])
                if m in hm and m not in DRUG_AGNOSTIC:
                    r[f'adjusted_{w}'][:, j] = cv.pcorr(p, z, [g, h])
    return res, Vd, ncell


# ----------------------------------------------------------------------------------------------------------------
# aggregation over drugs with drug-bootstrap weights
def wsum(S, V, mask):
    return np.nansum(np.where(mask, S, 0.0) * V, axis=1)


def wmean(S, V, mask=None):
    m = np.isfinite(S) if mask is None else (mask & np.isfinite(S))
    with np.errstate(invalid='ignore', divide='ignore'):
        return wsum(S, V, m) / (m * V).sum(1)


def wmedian(S, V, mask=None):
    """Per replicate (row): median of S over drugs with multiplicities V (drug bootstrap), finite values only."""
    m = np.isfinite(S) if mask is None else (mask & np.isfinite(S))
    out = np.full(S.shape[0], np.nan)
    for b in range(S.shape[0]):
        x, w = S[b][m[b]], V[b][m[b]]
        if not len(x) or w.sum() <= 0:
            continue
        o = np.argsort(x)
        cw = np.cumsum(w[o])
        half = cw[-1] / 2.0
        i = np.searchsorted(cw, half)
        out[b] = x[o][i] if cw[i] > half or i + 1 >= len(x) else 0.5 * (x[o][i] + x[o][i + 1])
    return out


def summarise(x, prefix):
    """Point estimate (replicate 0), bootstrap SE, 95% percentile interval, Wald P (two-sided, H0: 0)."""
    pt, bs = float(x[0]), x[1:][np.isfinite(x[1:])]
    if not np.isfinite(pt) or len(bs) < 0.9 * B:
        return {prefix: pt, f'{prefix}_lo': np.nan, f'{prefix}_hi': np.nan, f'{prefix}_se': np.nan}
    lo, hi = np.percentile(bs, [2.5, 97.5])
    lo90, hi90 = np.percentile(bs, [5, 95])
    se = float(np.std(bs, ddof=1))
    return {prefix: pt, f'{prefix}_lo': float(lo), f'{prefix}_hi': float(hi), f'{prefix}_lo90': float(lo90),
            f'{prefix}_hi90': float(hi90), f'{prefix}_se': se,
            f'{prefix}_p': float(2 * norm.sf(abs(pt) / se)) if se > 0 else np.nan}


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


# ----------------------------------------------------------------------------------------------------------------
def contrast_pairs(models, setting):
    pairs = [
        ('target: per-drug ridge, DrEval features (selective - raw z)', 'PerDrugRidgeSelective', 'PerDrugRidgeRaw'),
        ('target: per-drug ridge, DepMap features (selective - raw z)', 'PerDrugRidgeSelectiveDm', 'PerDrugRidgeRawDm'),
        ('target: global GBT (selective - raw z)', 'GradientBoostingSelective', 'GradientBoostingZ'),
        ('target: global GBT (selective - raw response)', 'GradientBoostingSelective', 'GradientBoosting'),
        ('target: global GBT (raw z - raw response)', 'GradientBoostingZ', 'GradientBoosting'),
        ('target: global NN (selective - raw response)', 'SimpleNeuralNetworkSelective', 'SimpleNeuralNetwork'),
        ('target: EnsembleMF (selective - raw response)', 'EnsembleMFSelective', 'EnsembleMF'),
        ('target: EnsembleMF (selective - leaderboard raw response)', 'EnsembleMFSelective', 'LB:EnsembleMF'),
        ('features: raw ridge (DepMap - DrEval)', 'PerDrugRidgeRawDm', 'PerDrugRidgeRaw'),
        ('features: selective ridge (DepMap - DrEval)', 'PerDrugRidgeSelectiveDm', 'PerDrugRidgeSelective'),
        ('class: raw z target (per-drug ridge - global GBT)', 'PerDrugRidgeRaw', 'GradientBoostingZ'),
        ('class: selective target (per-drug ridge - global GBT)', 'PerDrugRidgeSelective', 'GradientBoostingSelective'),
    ]
    others = [m for m in models if m not in ('PerDrugRidgeSelective', 'PerDrugRidgeRaw') and not m.endswith('Dm')
              and m not in ('GradientBoostingZ', 'GradientBoostingSelective', 'SimpleNeuralNetworkSelective',
                            'EnsembleMFSelective')]
    for m in others:
        pairs.append((f'selective ridge vs {m}', 'PerDrugRidgeSelective', m))
        pairs.append((f'raw ridge vs {m}', 'PerDrugRidgeRaw', m))
    return [(n, a, b) for n, a, b in pairs if a in models and b in models]


_RS = {}


def rs28():
    if 'm' not in _RS:
        _RS['m'] = load_module('rs28', '28_dreval_rescore.py')
    return _RS['m']


def model_meta(m):
    if m.startswith('LB:'):
        rs = rs28()
        return {'model_class': 'DrEval leaderboard (' + rs.model_class(m[3:]) + ')', 'training_target': 'raw response',
                'features': 'as released', 'source': 'DrEval leaderboard_Oct26 predictions'}
    c, t, f = MODEL_INFO.get(m, ('other', '', ''))
    return {'model_class': c, 'training_target': t, 'features': f, 'source': 'drevalpy 1.6.0 run (58)'}


def published_pool(models):
    rs = rs28()
    return [m for m in models if m.startswith('LB:') and rs.model_class(m[3:]) in ('classical', 'neural/factorization')
            and m != 'LB:SVR']


def run_setting(setting, rng, log):
    ds, ep = SETTINGS[setting]
    t0 = time.time()
    resp = responses()
    oof, rawscale, incomplete = load_oof(setting)    # rawscale: drevalpy rows (original scale)
    if not oof:
        log[setting] = {'skipped': 'no complete models', 'incomplete': incomplete}
        return {}
    models_all = sorted(oof)
    obs_s_full = resp[(resp.dataset == ds) & (resp.endpoint == ep)][['drug', 'cell', 'tissue', 'response']]
    panels = {'main': [m for m in models_all if not m.endswith('Dm')],
              'feature_source': [m for m in FEATSRC if m in oof]}
    if len(panels['feature_source']) < 2:
        panels.pop('feature_source')
    out = {k: [] for k in ['within_per_drug', 'within_summary', 'normalized', 'paired_per_drug', 'paired_summary',
                           'paired_pooled', 'contrasts', 'provenance', 'new_cells', 'norm_validation']}
    info = {'models': models_all, 'incomplete_models': incomplete, 'panels': {}}
    for panel, models in panels.items():
        cells = sorted(set.intersection(*[set(oof[m].cell) for m in models]))
        Pfull = {m: oof[m][oof[m].cell.isin(cells)].pivot_table(index='drug', columns='cell', values='prediction')
                 for m in models}
        # ---------------- within the training dataset
        O = obs_s_full[obs_s_full.cell.isin(cells)][['drug', 'cell', 'response']]
        Z, G, _, r2 = selbench.decompose(O)
        Pm = {m: P.reindex(columns=Z.columns) for m, P in Pfull.items()}
        Hm = {m: zscore_axis(P.loc[P.notna().sum(axis=1) >= MIN_CELLS]) for m, P in Pm.items()}
        res, drugs, Vd, ncell = score_within(models, Z, G, Pm, Hm, list(Z.columns), rng)
        rzg2 = None
        for m in models:
            r = res[m]
            pt = pd.DataFrame({s: r[s][0] for s in r}, index=drugs)
            pt['n_cells'] = ncell
            out['within_per_drug'].append(pt.reset_index().rename(columns={'index': 'drug'}).assign(
                setting=setting, panel=panel, model=m))
            ok = np.isfinite(r['raw']) & np.isfinite(r['selective'])
            row = {'setting': setting, 'panel': panel, 'model': m, **model_meta(m),
                   'drugs': int(np.isfinite(r['raw'][0]).sum()), 'cells': len(Z.columns),
                   'median_raw': float(np.nanmedian(r['raw'][0])), 'median_selective': float(np.nanmedian(r['selective'][0])),
                   'median_adjusted': float(np.nanmedian(r['adjusted'][0])) if np.isfinite(r['adjusted'][0]).any() else np.nan,
                   'median_alignment': float(np.nanmedian(r['r_pg'][0]))}
            for stat in ['raw', 'selective', 'adjusted']:
                row.update(summarise(wmedian(r[stat], Vd), f'median_{stat}_boot'))
            row.update(summarise(wmean(r['raw'], Vd, ok), 'mean_raw'))
            row.update(summarise(wmean(r['mediated'], Vd, ok), 'mean_mediated'))
            row.update(summarise(wmean(r['selective_term'], Vd, ok), 'mean_selective_term'))
            row.update(summarise(wmean(r['selective'], Vd, ok), 'mean_selective'))
            row.update(summarise(wmean(r['adjusted'], Vd), 'mean_adjusted'))
            with np.errstate(invalid='ignore', divide='ignore'):
                share = wsum(r['mediated'], Vd, ok) / wsum(r['raw'], Vd, ok)
            row.update(summarise(share, 'mediated_share'))
            rzg2_m = wmean(r['r_zg'] ** 2, Vd, ok)
            row.update(summarise(rzg2_m, 'oracle_mean_rzg2'))
            out['within_summary'].append(row)
        # contrasts within the training dataset, on the drugs scored for every model of the panel
        common = np.all([np.isfinite(res[m]['raw'][0]) for m in models], axis=0)
        info.setdefault('common_drugs', {})[f'{panel}:within'] = int(common.sum())
        for name, a, b in contrast_pairs(models, setting):
            for stat, label in [('selective', 'selective (= selective-BOTH within)'), ('adjusted', 'model-axis-adjusted'),
                                ('raw', 'raw')]:
                D = res[a][stat] - res[b][stat]
                ok0 = np.isfinite(D[0]) & common
                if ok0.sum() < 10:
                    continue
                row = {'setting': setting, 'panel': panel, 'where': f'within {ds}', 'contrast': name, 'model_a': a,
                       'model_b': b, 'metric': label, 'drugs': int(ok0.sum()),
                       'median_diff': float(np.median(D[0][ok0])), 'frac_a_better': float((D[0][ok0] > 0).mean()),
                       'wilcoxon_p': float(wilcoxon(D[0][ok0]).pvalue) if np.any(D[0][ok0] != 0) else np.nan}
                row.update(summarise(wmean(D, Vd, np.broadcast_to(ok0, D.shape)), 'mean_diff'))
                out['contrasts'].append(row)
        # ---------------- DrEval normalized Pearson (main panel only)
        if panel == 'main':
            naive_name = 'NaiveMeanEffectsPredictor' if 'NaiveMeanEffectsPredictor' in rawscale else \
                'LB:NaiveMeanEffectsPredictor'
            if naive_name in rawscale:
                nv = rawscale[naive_name][['drug', 'cell', 'pred', 'split']].rename(columns={'pred': 'naive'})
                for m in models:
                    if m == naive_name:
                        continue
                    x = rawscale[m][['drug', 'cell', 'pred', 'y', 'split']].merge(nv, on=['drug', 'cell', 'split'])
                    x = x[x.cell.isin(cells) & x.drug.isin(set(Z.index))]
                    per = [np.corrcoef(g.y - g.naive, g.pred - g.naive)[0, 1] for _, g in x.groupby('split')]
                    out['normalized'].append({'setting': setting, 'model': m, **model_meta(m),
                                              'normalized_pearson': float(np.mean(per)),
                                              'split_sd': float(np.std(per, ddof=1)), 'splits': len(per),
                                              'naive_baseline': naive_name, 'pairs': len(x)})
                if setting == 'CTRPv2_LNIC50':      # validation against DrEval's reported leaderboard values
                    ev = pd.read_csv(LB / 'evaluation_results.csv')
                    ev['model'] = 'LB:' + ev.algorithm.str.replace(r' \(.*\)$', '', regex=True)
                    rep = ev.groupby('model')['Pearson: normalized'].mean()
                    for m in [m for m in models if m in rep.index and m != naive_name]:
                        x = rawscale[m][['drug', 'cell', 'pred', 'y', 'split']].merge(nv, on=['drug', 'cell', 'split'])
                        per = [np.corrcoef(g.y - g.naive, g.pred - g.naive)[0, 1] for _, g in x.groupby('split')]
                        out['norm_validation'].append({'model': m, 'reported': float(rep[m]),
                                                       'reimplemented_all_pairs': float(np.mean(per))})
        # ---------------- paired cross-dataset
        Zs_full, Gs_full, _, _ = selbench.decompose(obs_s_full[['drug', 'cell', 'response']])
        pooled_groups = {'DrEval learned models (raw response)': [m for m in DREVAL_LEARNED if m in models],
                         'published leaderboard models (21, as 38)': published_pool(models)}
        for T in [t for t in DATASETS if t != ds]:
            obs_t = resp[(resp.dataset == T) & (resp.endpoint == ep)][['drug', 'cell', 'response']]
            Zt, Gt, _, _ = selbench.decompose(obs_t)
            pdrugs = sorted(set(Zs_full.index) & set(Zt.index) & set().union(*[set(Pfull[m].index) for m in models]))
            pcells = sorted(set(Zs_full.columns) & set(Zt.columns) & set(cells))
            if len(pdrugs) < 5 or len(pcells) < MIN_CELLS:
                continue
            Pm_t = {m: Pfull[m].reindex(index=[d for d in pdrugs if d in Pfull[m].index], columns=pcells)
                    for m in models}
            # reference: the training dataset's own measured profile (rank-normal z) used as the prediction, scored on
            # the same matched drugs and cells (one replicate measurement; cf. ref_CTRPv2_measured in 42/43)
            Pm_t[REF] = Zs_full.reindex(index=pdrugs, columns=pcells)
            models_t = models + [REF]
            Hm_t = {m: zscore_axis(P.loc[P.notna().sum(axis=1) >= MIN_CELLS]) for m, P in Pm_t.items()}
            res_t, Vd_t, ncell_t = score_paired(models_t, Zs_full, Gs_full, Zt, Gt, Pm_t, Hm_t, pdrugs, pcells, rng)
            ref = res_t[REF]
            for m in models_t:
                r = res_t[m]
                pt = pd.DataFrame({s: r[s][0] for s in r}, index=pdrugs)
                pt['n_cells'] = ncell_t
                out['paired_per_drug'].append(pt.reset_index().rename(columns={'index': 'drug'}).assign(
                    setting=setting, panel=panel, target=T, model=m))
                ok = np.ones_like(r['raw_s'], dtype=bool)
                for s in ['raw_s', 'raw_t', 'mediated_s', 'mediated_t', 'selective_term_s', 'selective_term_t']:
                    ok &= np.isfinite(r[s])
                ok0 = ok[0]
                if ok0.sum() < 5:
                    continue
                okb = np.broadcast_to(ok0, ok.shape) & ok
                row = {'setting': setting, 'panel': panel, 'target': T, 'model': m, **model_meta(m),
                       'drugs': int(ok0.sum()), 'median_cells_per_drug': float(np.median(ncell_t[ok0])),
                       'median_raw_s': float(np.median(r['raw_s'][0][ok0])), 'median_raw_t': float(np.median(r['raw_t'][0][ok0]))}
                for s in ['raw', 'mediated', 'selective_term', 'r_pg', 'r_zg', 'sel_T', 'sel_BOTH', 'adjusted']:
                    for w in ('s', 't'):
                        row.update(summarise(wmean(r[f'{s}_{w}'], Vd_t, okb if s != 'adjusted' else None), f'mean_{s}_{w}'))
                    D = r[f'{s}_t'] - r[f'{s}_s']
                    row.update(summarise(wmean(D, Vd_t, okb if s != 'adjusted' else None), f'delta_{s}'))
                with np.errstate(invalid='ignore', divide='ignore'):
                    drop = wsum(r['raw_s'], Vd_t, okb) - wsum(r['raw_t'], Vd_t, okb)
                    row.update(summarise((wsum(r['mediated_s'], Vd_t, okb) - wsum(r['mediated_t'], Vd_t, okb)) / drop,
                                         'share_of_drop_from_fitness'))
                    row.update(summarise(wsum(r['mediated_t'], Vd_t, okb) / wsum(r['mediated_s'], Vd_t, okb),
                                         'mediated_retained'))
                    row.update(summarise(wsum(r['selective_term_t'], Vd_t, okb) / wsum(r['selective_term_s'], Vd_t, okb),
                                         'selterm_retained'))
                    row.update(summarise(wsum(r['r_pg_t'], Vd_t, okb) / wsum(r['r_pg_s'], Vd_t, okb), 'alignment_ratio'))
                    row.update(summarise(wsum(r['r_zg_t'], Vd_t, okb) / wsum(r['r_zg_s'], Vd_t, okb), 'loading_ratio'))
                    sb = wmean(r['sel_BOTH_t'], Vd_t, okb) / wmean(r['sel_BOTH_s'], Vd_t, okb)
                    row.update(summarise(sb, 'sel_BOTH_retained'))
                    if m != REF:
                        # cross selective-BOTH as a fraction of the training screen's own measurement (same drugs/cells)
                        both = okb & np.isfinite(r['sel_BOTH_t']) & np.isfinite(ref['sel_BOTH_t'])
                        both &= np.broadcast_to(both[0], both.shape)
                        row.update(summarise(wsum(r['sel_BOTH_t'], Vd_t, both) / wsum(ref['sel_BOTH_t'], Vd_t, both),
                                             'sel_BOTH_t_frac_of_measured_mean'))
                        row.update(summarise(wmedian(r['sel_BOTH_t'], Vd_t, both) / wmedian(ref['sel_BOTH_t'], Vd_t, both),
                                             'sel_BOTH_t_frac_of_measured_median'))
                        row.update(summarise(wsum(r['raw_t'], Vd_t, both) / wsum(ref['raw_t'], Vd_t, both),
                                             'raw_t_frac_of_measured_mean'))
                for st in ['raw_t', 'sel_BOTH_t', 'sel_T_t', 'mediated_t', 'adjusted_t']:
                    row[f'median_{st}'] = float(np.nanmedian(r[st][0][ok0])) if np.isfinite(r[st][0][ok0]).any() else np.nan
                out['paired_summary'].append(row)
            # pooled over groups of models (drugs resampled jointly across models)
            for gname, gmodels in pooled_groups.items():
                gm = [m for m in gmodels if m in res_t]
                if len(gm) < 2:
                    continue
                acc = {}
                for m in gm:
                    r = res_t[m]
                    ok = np.ones_like(r['raw_s'], dtype=bool)
                    for s in ['raw_s', 'raw_t', 'mediated_s', 'mediated_t', 'selective_term_s', 'selective_term_t',
                              'r_pg_s', 'r_pg_t', 'r_zg_s', 'r_zg_t', 'sel_BOTH_s', 'sel_BOTH_t']:
                        ok &= np.isfinite(r[s])
                    ok &= np.broadcast_to(ok[0], ok.shape)
                    for s in ['raw_s', 'raw_t', 'mediated_s', 'mediated_t', 'selective_term_s', 'selective_term_t',
                              'r_pg_s', 'r_pg_t', 'r_zg_s', 'r_zg_t', 'sel_BOTH_s', 'sel_BOTH_t']:
                        acc[s] = acc.get(s, 0) + wsum(r[s], Vd_t, ok)
                    acc['n'] = acc.get('n', 0) + (ok * Vd_t).sum(1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    row = {'setting': setting, 'panel': panel, 'target': T, 'group': gname, 'models': len(gm),
                           'model_list': ';'.join(gm)}
                    row.update(summarise((acc['mediated_s'] - acc['mediated_t']) / (acc['raw_s'] - acc['raw_t']),
                                         'share_of_drop_from_fitness'))
                    row.update(summarise(acc['mediated_t'] / acc['mediated_s'], 'mediated_retained'))
                    row.update(summarise(acc['selective_term_t'] / acc['selective_term_s'], 'selterm_retained'))
                    row.update(summarise(acc['r_pg_t'] / acc['r_pg_s'], 'alignment_ratio'))
                    row.update(summarise(acc['r_zg_t'] / acc['r_zg_s'], 'loading_ratio'))
                    for s in ['raw', 'mediated', 'selective_term', 'sel_BOTH']:
                        row.update(summarise(acc[f'{s}_s'] / acc['n'], f'mean_{s}_s'))
                        row.update(summarise(acc[f'{s}_t'] / acc['n'], f'mean_{s}_t'))
                        row.update(summarise((acc[f'{s}_t'] - acc[f'{s}_s']) / acc['n'], f'delta_{s}'))
                out['paired_pooled'].append(row)
            # contrasts in the target dataset, on the paired drugs scored for every model of the panel
            common_t = np.all([np.isfinite(res_t[m]['raw_t'][0]) & np.isfinite(res_t[m]['raw_s'][0]) for m in models],
                              axis=0)
            info.setdefault('common_drugs', {})[f'{panel}:{T}'] = int(common_t.sum())
            for name, a, b in contrast_pairs(models, setting):
                for stat, label in [('sel_BOTH_t', 'selective-BOTH'), ('adjusted_t', 'model-axis-adjusted'),
                                    ('sel_T_t', 'selective-T'), ('raw_t', 'raw')]:
                    D = res_t[a][stat] - res_t[b][stat]
                    ok0 = np.isfinite(D[0]) & common_t
                    if ok0.sum() < 10:
                        continue
                    row = {'setting': setting, 'panel': panel, 'where': f'cross {T}', 'contrast': name, 'model_a': a,
                           'model_b': b, 'metric': label, 'drugs': int(ok0.sum()),
                           'median_diff': float(np.median(D[0][ok0])), 'frac_a_better': float((D[0][ok0] > 0).mean()),
                           'wilcoxon_p': float(wilcoxon(D[0][ok0]).pvalue) if np.any(D[0][ok0] != 0) else np.nan}
                    row.update(summarise(wmean(D, Vd_t, np.broadcast_to(ok0, D.shape)), 'mean_diff'))
                    out['contrasts'].append(row)
            # provenance of target-dataset predictions and an unpaired score on new cells (supplementary)
            if panel == 'main':
                tcells = set(obs_t.cell)
                for m in models:
                    if m.startswith('LB:'):
                        continue
                    new, pinfo = load_cross(setting, m, T, set(oof[m].cell))
                    prov = {'setting': setting, 'target': T, 'model': m, 'target_cells': len(tcells),
                            'target_cells_with_oof_prediction': len(tcells & set(oof[m].cell)),
                            'paired_cells_used': len(pcells), **pinfo}
                    out['provenance'].append(prov)
                    if new is None or not len(new):
                        continue
                    Pn = new.pivot_table(index='drug', columns='cell', values='prediction')
                    for d in sorted(set(Pn.index) & set(Zt.index) & set(Zs_full.index)):
                        p = Pn.loc[d].reindex(Zt.columns).values
                        z, g = Zt.loc[d].values, Gt.loc[d].values
                        okn = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
                        if okn.sum() < MIN_CELLS or np.std(p[okn]) < 1e-12:
                            continue
                        c = np.corrcoef(np.vstack([p[okn], z[okn], g[okn]]))
                        den = np.sqrt(max((1 - c[0, 2] ** 2) * (1 - c[1, 2] ** 2), 0))
                        out['new_cells'].append({'setting': setting, 'target': T, 'model': m, 'drug': d,
                                                 'n_new_cells': int(okn.sum()), 'raw': c[0, 1],
                                                 'sel_T': (c[0, 1] - c[0, 2] * c[1, 2]) / den if den > 1e-9 else np.nan})
            info['panels'][f'{panel}:{T}'] = {'paired_drugs': len(pdrugs), 'paired_cells': len(pcells)}
        info['panels'][panel] = {'models': models, 'cells': len(cells), 'drugs_within': len(Z.index),
                                 'median_fitness_r2_within': float(r2.median())}
    info['seconds'] = time.time() - t0
    log[setting] = info
    return out


def dataset_fitness_table():
    """How fitness-dominated each DrEval dataset and endpoint is (full dataset, all cells and drugs)."""
    resp = responses()
    rows = []
    for (ds, ep), o in resp.groupby(['dataset', 'endpoint']):
        Z, G, R, r2 = selbench.decompose(o[['drug', 'cell', 'response']])
        rows.append({'dataset': ds, 'endpoint': ep, 'drugs': len(Z), 'cells': Z.shape[1],
                     'median_fitness_r2': float(r2.median()), 'mean_fitness_r2': float(r2.mean()),
                     'q25_fitness_r2': float(r2.quantile(0.25)), 'q75_fitness_r2': float(r2.quantile(0.75))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', nargs='*', default=list(SETTINGS))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    log, frames = {}, {}
    for setting in args.settings:
        print('scoring', setting, flush=True)
        out = run_setting(setting, rng, log)
        for k, v in out.items():
            frames.setdefault(k, []).extend(v)
        print(setting, json.dumps({k: (v if not isinstance(v, dict) else '...') for k, v in log[setting].items()},
                                  default=str)[:500], flush=True)
    files = {'within_per_drug': 'within_per_drug.csv.gz', 'within_summary': 'within_summary.csv',
             'normalized': 'normalized_pearson.csv', 'norm_validation': 'normalized_validation.csv',
             'paired_per_drug': 'paired_per_drug.csv.gz', 'paired_summary': 'paired_summary.csv',
             'paired_pooled': 'paired_pooled.csv', 'contrasts': 'contrasts.csv', 'provenance': 'cell_provenance.csv',
             'new_cells': 'new_cells_unpaired.csv'}
    for k, fn in files.items():
        v = frames.get(k, [])
        if not v:
            continue
        df = pd.concat(v, ignore_index=True) if isinstance(v[0], pd.DataFrame) else pd.DataFrame(v)
        if k == 'contrasts':
            # Holm within pre-specified families: per setting, and per family over all settings
            prefix = {'target: ': 'target effect within model class', 'features: ': 'feature source',
                      'class: ': 'model class'}
            df['family'] = [next((v for k, v in prefix.items() if c.startswith(k)), 'vs other models')
                            for c in df.contrast]
            # Holm family: one metric, one panel; for comparisons with other models also one reference model
            # (e.g. selective ridge vs every other model x every evaluation dataset on one metric, as in R1-M4)
            df['holm_family'] = df.family + ' | ' + df.metric + np.where(df.family == 'vs other models',
                                                                         ' | ' + df.model_a, '')
            for col, src in [('holm_p_family_setting', 'mean_diff_p'), ('wilcoxon_holm_family_setting', 'wilcoxon_p')]:
                df[col] = np.nan
                for _, idx in df.groupby(['setting', 'panel', 'holm_family']).groups.items():
                    df.loc[idx, col] = holm(df.loc[idx, src].values)
            df['holm_p_family_all_settings'] = np.nan
            for _, idx in df.groupby(['panel', 'holm_family']).groups.items():
                df.loc[idx, 'holm_p_family_all_settings'] = holm(df.loc[idx, 'mean_diff_p'].values)
        if k == 'new_cells':
            df = df.groupby(['setting', 'target', 'model']).agg(drugs=('drug', 'size'),
                                                               median_new_cells=('n_new_cells', 'median'),
                                                               median_raw=('raw', 'median'), mean_raw=('raw', 'mean'),
                                                               median_sel_T=('sel_T', 'median'),
                                                               mean_sel_T=('sel_T', 'mean')).reset_index()
        df.to_csv(OUT / fn, index=False, float_format='%.6g')
    dataset_fitness_table().to_csv(OUT / 'dataset_fitness.csv', index=False, float_format='%.6g')
    (OUT / 'score_run.json').write_text(json.dumps(log, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
