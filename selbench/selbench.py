"""selbench: selectivity-aware evaluation of drug-response predictions.

Minimal dependencies (numpy, pandas, scipy). Typical use:

    import pandas as pd, selbench
    obs = pd.read_csv('observed.csv')        # columns: drug, cell, response (higher = more sensitive)
    pred = pd.read_csv('predictions.csv')    # columns: drug, cell, prediction (higher = more sensitive)
    table, summary = selbench.evaluate(pred, obs)

`evaluate` decomposes the observed screen into a drug-agnostic fitness axis and drug-specific selective residuals
(as in the accompanying paper) and reports, per drug, the usual per-drug correlation ("raw") and the partial
correlation given the screen's leave-drug-out fitness axis ("selective"). With `reliability` (a per-drug
reliability of the evaluation screen, e.g. from `release/drug_reproducibility.csv`), selective correlations are
also divided by the square root of the reliability (ceiling-normalized).

Revision 1 adds the components of the recommended evaluation protocol (Box; maintainer version in
results/revision_repro/evaluation_protocol.md):

  Axis           fitness_axis (leave-drug-out or leave-class-out, drug panel, minimum drugs per cell line),
                 selective_residuals, axis_reliability, axis_agreement
  Primitives     partial_corr (any number of covariates), split_terms (exact fitness-mediated/selective split)
  Scores         selective_scores (evaluation axis; evaluation + training axes), paired_scores /
                 paired_scores_matrix (identical predictions, matched cells, two screens), paired_summary,
                 model_axis / model_axis_scores / model_axis_diagnostic (model's own axis; a diagnostic),
                 leakage_floor (drug-agnostic predictors), oracle_fitness_share, summarize (raw and selective
                 side by side)
  Baselines      fitness_baseline (omics-predicted drug-agnostic baseline), tissue_drug_mean_baseline,
                 tissue_mean_baseline, drug_agnostic_predictions
  Uncertainty    two_way_bootstrap (drug x cell), weighted_median, paired_comparison, significant
  Reliability    shrink_reliability, normalize_by_reliability (shrinkage, minimum, ReliabilityWarning)

Conventions: long tables have columns drug, cell and response (observed) or prediction; matrices are DataFrames
with drugs as rows and cell lines as columns; higher values mean more sensitive in observations and predictions.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

__all__ = ['rank_normal', 'decompose', 'evaluate', 'MIN_CELLS',
           # revision 1
           'MIN_PAIRED_CELLS', 'MIN_DRUGS_PER_CELL', 'RELIABILITY_FLOOR', 'MIN_RELIABILITY', 'DEFAULT_SHRINKAGE',
           'ReliabilityWarning', 'as_matrix', 'fitness_axis', 'selective_residuals', 'axis_reliability',
           'axis_agreement', 'partial_corr', 'split_terms', 'selective_scores', 'paired_scores',
           'paired_scores_matrix', 'paired_summary', 'model_axis', 'model_axis_scores', 'model_axis_diagnostic',
           'drug_agnostic_predictions', 'leakage_floor', 'oracle_fitness_share', 'summarize', 'fitness_baseline',
           'tissue_drug_mean_baseline', 'tissue_mean_baseline', 'two_way_bootstrap', 'weighted_median',
           'paired_comparison', 'significant', 'shrink_reliability', 'normalize_by_reliability']
MIN_CELLS = 30
MIN_PAIRED_CELLS = 30     # per drug: cells (matched in both screens for paired scoring) needed for a per-drug score
MIN_DRUGS_PER_CELL = 10   # recommended minimum number of other drugs behind a cell line's axis value (protocol)
RELIABILITY_FLOOR = 0.1   # per-drug reliabilities below this raise a ReliabilityWarning
MIN_RELIABILITY = 0.1     # shrunken reliabilities below this are not used (caps the inflation at 1/sqrt(0.1) = 3.2)
DEFAULT_SHRINKAGE = 0.5   # shrinkage weight toward the screen median when no standard errors are available


class ReliabilityWarning(UserWarning):
    """Per-drug reliabilities are too low for per-drug ceiling normalisation."""


def rank_normal(x):
    """Rank-based inverse normal transform; NaN preserved, ties averaged."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum():
        out[ok] = norm.ppf((rankdata(x[ok]) - 0.5) / ok.sum())
    return out


def decompose(obs, min_cells=MIN_CELLS):
    """Return (z, fitness, residual, fitness_r2) for a long table with columns drug, cell, response.

    z: drug x cell rank-normal profiles; fitness: drug x cell leave-drug-out mean of z over all other drugs;
    residual: OLS residual of z on fitness (selective profile); fitness_r2: per-drug share explained.
    """
    M = obs.pivot_table(index='drug', columns='cell', values='response', aggfunc='mean')
    M = M.loc[M.notna().sum(axis=1) >= min_cells]
    M = M.loc[[d for d in M.index if M.loc[d].dropna().nunique() >= 10]]
    Z = pd.DataFrame(np.vstack([rank_normal(r) for r in M.values]), index=M.index, columns=M.columns)
    v = Z.values; ok = np.isfinite(v)
    tot, cnt = np.nansum(v, axis=0), ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = pd.DataFrame((tot[None] - np.where(ok, v, 0)) / (cnt[None] - ok), index=Z.index, columns=Z.columns)
    R = Z * np.nan
    r2 = {}
    for d in Z.index:
        z, g = Z.loc[d].values, G.loc[d].values
        m = np.isfinite(z) & np.isfinite(g)
        X = np.c_[np.ones(m.sum()), g[m]]
        b = np.linalg.lstsq(X, z[m], rcond=None)[0]
        r = np.full(len(z), np.nan); r[m] = z[m] - X @ b
        R.loc[d] = r
        r2[d] = float(np.corrcoef(z[m], g[m])[0, 1] ** 2)
    return Z, G, R, pd.Series(r2, name='fitness_r2')


def _partial(x, y, g):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(g)
    if m.sum() < 20:
        return np.nan
    X = np.c_[np.ones(m.sum()), g[m]]
    rx = x[m] - X @ np.linalg.lstsq(X, x[m], rcond=None)[0]
    ry = y[m] - X @ np.linalg.lstsq(X, y[m], rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def evaluate(pred, obs, reliability=None, min_cells=MIN_CELLS, shrinkage=None, reliability_se=None,
             min_reliability=None, warn_below=RELIABILITY_FLOOR):
    """Per-drug raw and selective correlations of predictions with an observed screen.

    pred: long table (drug, cell, prediction); obs: long table (drug, cell, response). Higher = more sensitive in
    both. reliability: optional mapping drug -> reliability of the observed screen's selective profile.
    Returns (per-drug table, summary dict of medians).

    `selective_ceiling_normalized` (selective / sqrt(reliability), the original behaviour) is unchanged, but a
    ReliabilityWarning is now raised when any per-drug reliability is below `warn_below`: near-zero reliabilities
    inflate that column without bound. Passing `shrinkage`, `reliability_se` or `min_reliability` adds the
    recommended column `selective_reliability_normalized` (see normalize_by_reliability).
    """
    Z, G, R, r2 = decompose(obs, min_cells)
    P = pred.pivot_table(index='drug', columns='cell', values='prediction', aggfunc='mean')
    rows = []
    for d in Z.index.intersection(P.index):
        p = P.loc[d].reindex(Z.columns).values
        rows.append({'drug': d, 'n': int(np.isfinite(p).sum()), 'raw': _corr(p, Z.loc[d].values),
                     'selective': _partial(p, Z.loc[d].values, G.loc[d].values), 'fitness_r2': r2[d]})
    T = pd.DataFrame(rows)
    if reliability is not None and len(T):
        rel = T.drug.map(reliability)
        T['selective_ceiling_normalized'] = T.selective / np.sqrt(rel.where(rel > 0))
        if warn_below is not None:
            _warn_low_reliability(rel, warn_below)
        if shrinkage is not None or reliability_se is not None or min_reliability is not None:
            se = None if reliability_se is None else T.drug.map(reliability_se).values
            N = normalize_by_reliability(T.selective.values, rel.values, weight=shrinkage, se=se,
                                         min_reliability=MIN_RELIABILITY if min_reliability is None else min_reliability,
                                         warn_below=None)
            T['selective_reliability_normalized'] = N['normalized'].values
    summary = {k: float(T[k].median()) for k in T.columns if k not in ('drug', 'n')} if len(T) else {}
    summary['drugs'] = int(len(T))
    return T, summary


# ----------------------------------------------------------------------------------------------------------------
# Matrices and the fitness axis
# ----------------------------------------------------------------------------------------------------------------

def as_matrix(long, value='prediction'):
    """drug x cell matrix (mean over duplicated drug-cell pairs) from a long table with columns drug, cell, value."""
    return long.pivot_table(index='drug', columns='cell', values=value, aggfunc='mean')


def fitness_axis(Z, groups=None, panel=None, min_drugs=1):
    """Leave-drug-out (or leave-class-out) fitness axis of a drug x cell matrix of standardised profiles.

    G[d, c] is the mean of Z[d', c] over the drugs d' of the axis panel measured on cell line c, excluding d itself
    (leave-drug-out) or, with `groups` (mapping drug -> class, e.g. target or mechanism), excluding every drug of
    d's class (leave-class-out; drugs without a class are their own class). `panel` restricts the drugs that define
    the axis (e.g. the drugs shared by two screens, which makes axes comparable across screens); drugs outside the
    panel still receive an axis (the mean over the panel). Cell lines with fewer than `min_drugs` contributing drugs
    get NaN. With the defaults the result is identical (same arithmetic) to the axis returned by `decompose`.
    """
    v = Z.values.astype(float)
    ok = np.isfinite(v)
    if groups is None and panel is None and min_drugs <= 1:
        tot, cnt = np.nansum(v, axis=0), ok.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            return pd.DataFrame((tot[None] - np.where(ok, v, 0)) / (cnt[None] - ok), index=Z.index, columns=Z.columns)
    inp = np.ones(len(Z), bool) if panel is None else np.asarray(Z.index.isin(list(panel)))
    vv = np.where(ok & inp[:, None], v, 0.0)
    oo = (ok & inp[:, None]).astype(float)
    tot, cnt = vv.sum(axis=0), oo.sum(axis=0)
    if groups is None:
        own_sum, own_cnt = vv, oo
    else:
        lab = pd.Series([groups.get(d, None) if hasattr(groups, 'get') else None for d in Z.index], index=Z.index)
        lab = lab.where(lab.notna(), pd.Series([f'__own__{d}' for d in Z.index], index=Z.index)).astype(str)
        own_sum, own_cnt = np.zeros_like(vv), np.zeros_like(oo)
        for _, idx in lab.groupby(lab).indices.items():
            own_sum[idx] = vv[idx].sum(axis=0)
            own_cnt[idx] = oo[idx].sum(axis=0)
    n_used = cnt[None] - own_cnt
    with np.errstate(invalid='ignore', divide='ignore'):
        G = (tot[None] - own_sum) / n_used
    G[n_used < max(min_drugs, 1)] = np.nan
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def selective_residuals(Z, G):
    """Per-drug OLS residual of each profile on its axis (the selective profile) and the fitness R^2.

    Same computation as in `decompose`, for any axis (e.g. leave-class-out or panel-restricted)."""
    R = Z * np.nan
    r2 = {}
    for d in Z.index:
        z, g = Z.loc[d].values.astype(float), G.loc[d].reindex(Z.columns).values.astype(float)
        m = np.isfinite(z) & np.isfinite(g)
        if m.sum() < 3 or np.std(g[m]) < 1e-12:
            r2[d] = np.nan
            continue
        X = np.c_[np.ones(m.sum()), g[m]]
        b = np.linalg.lstsq(X, z[m], rcond=None)[0]
        r = np.full(len(z), np.nan); r[m] = z[m] - X @ b
        R.loc[d] = r
        r2[d] = float(np.corrcoef(z[m], g[m])[0, 1] ** 2)
    return R, pd.Series(r2, name='fitness_r2')


def _screen(obs, min_cells=MIN_CELLS, groups=None, panel=None, min_drugs=1):
    """Rank-normal profiles and leave-drug-out axis of a screen given as a long table."""
    Z, G, _, _ = decompose(obs, min_cells)
    if groups is not None or panel is not None or min_drugs > 1:
        G = fitness_axis(Z, groups=groups, panel=panel, min_drugs=min_drugs)
    return Z, G


def axis_reliability(Z, n_splits=100, seed=0, min_drugs=1):
    """Split-half reliability of a screen's fitness axis over random halves of its drugs (Spearman-Brown corrected).

    Returns the median over `n_splits` random drug splits of 2r/(1+r), where r is the correlation across cell lines
    of the two half-panel mean profiles. This is reliability with respect to drug sampling only (it does not
    include run-to-run or between-laboratory error)."""
    rng = np.random.default_rng(seed)
    v = Z.values.astype(float)
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(len(v))
        a, b = perm[: len(v) // 2], perm[len(v) // 2:]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ga, gb = np.nanmean(v[a], axis=0), np.nanmean(v[b], axis=0)
        na, nb = np.isfinite(v[a]).sum(axis=0), np.isfinite(v[b]).sum(axis=0)
        m = np.isfinite(ga) & np.isfinite(gb) & (na >= min_drugs) & (nb >= min_drugs)
        if m.sum() < 3:
            continue
        r = np.corrcoef(ga[m], gb[m])[0, 1]
        out.append(2 * r / (1 + r))
    return float(np.median(out)) if out else np.nan


def axis_agreement(obs_a, obs_b, shared_drugs=False, min_cells=MIN_CELLS, min_drugs=1):
    """Correlation, across the cell lines of both screens, of two screens' fitness axes (mean rank-normal profile).

    With shared_drugs=True each axis is built from the drugs measured in both screens only, which removes the
    dependence of the axis on the drug panel. Returns dict(r, cells, drugs_a, drugs_b)."""
    Za, _, _, _ = decompose(obs_a, min_cells)
    Zb, _, _, _ = decompose(obs_b, min_cells)
    if shared_drugs:
        common = Za.index.intersection(Zb.index)
        Za, Zb = Za.loc[common], Zb.loc[common]
    cells = Za.columns.intersection(Zb.columns)
    ga, gb = Za[cells].mean(axis=0), Zb[cells].mean(axis=0)
    na, nb = Za[cells].notna().sum(axis=0), Zb[cells].notna().sum(axis=0)
    m = ga.notna() & gb.notna() & (na >= min_drugs) & (nb >= min_drugs)
    r = float(np.corrcoef(ga[m], gb[m])[0, 1]) if m.sum() >= 3 else np.nan
    return {'r': r, 'cells': int(m.sum()), 'drugs_a': int(len(Za)), 'drugs_b': int(len(Zb))}


# ----------------------------------------------------------------------------------------------------------------
# Per-drug primitives
# ----------------------------------------------------------------------------------------------------------------

def partial_corr(x, y, covariates=(), min_n=20, degenerate=np.nan, rtol=1e-9):
    """Partial correlation of x and y given any number of covariates (OLS residualisation on [1, covariates]).

    x, y and each covariate are 1-D arrays of equal length; only complete cases are used. Returns NaN with fewer
    than `min_n` complete cases or when y has no variance left; returns `degenerate` when x has no variance left
    (x is a linear function of the covariates; relative tolerance `rtol`). Use degenerate=0.0 when such a predictor
    should count as carrying no information beyond the covariates (model_axis_scores does)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    covs = [np.asarray(c, float) for c in covariates]
    m = np.isfinite(x) & np.isfinite(y)
    for c in covs:
        m &= np.isfinite(c)
    if m.sum() < min_n:
        return np.nan
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in covs])
    rx = x[m] - X @ np.linalg.lstsq(X, x[m], rcond=None)[0]
    ry = y[m] - X @ np.linalg.lstsq(X, y[m], rcond=None)[0]
    sx, sy = x[m].std(), y[m].std()
    if sy < 1e-12 or ry.std() <= rtol * sy:
        return np.nan
    if sx < 1e-12 or rx.std() <= rtol * sx:
        return degenerate
    return float(np.corrcoef(rx, ry)[0, 1])


def split_terms(p, z, g):
    """Exact split of a per-drug correlation into a fitness-mediated and a selective term.

        r_pz = r_pg * r_zg + sqrt((1 - r_pg^2) (1 - r_zg^2)) * r_pz.g

    p: predictions, z: observed rank-normal profile, g: the evaluation screen's leave-drug-out axis (complete,
    equal-length arrays). Returns raw (r_pz), alignment (r_pg: how much the predictions follow the axis), loading
    (r_zg: how much the observed profile loads on its own axis), mediated (r_pg r_zg), selective_term
    (r_pz - r_pg r_zg) and selective (the partial correlation r_pz.g)."""
    c = np.corrcoef(np.vstack([p, z, g]))
    r_pz, r_pg, r_zg = c[0, 1], c[0, 2], c[1, 2]
    den = np.sqrt(max((1 - r_pg ** 2) * (1 - r_zg ** 2), 0))
    return {'raw': r_pz, 'selective': (r_pz - r_pg * r_zg) / den if den > 1e-9 else np.nan,
            'alignment': r_pg, 'loading': r_zg, 'mediated': r_pg * r_zg, 'selective_term': r_pz - r_pg * r_zg}


def _vec(M, d, cells):
    return M.loc[d].reindex(cells).values.astype(float) if d in M.index else np.full(len(cells), np.nan)


def _axis_rows(Z, G, drugs):
    """Axis rows for `drugs`: the leave-drug-out axis if the drug is in the screen, else the all-drug mean."""
    mean = Z.mean(axis=0)
    return pd.DataFrame([G.loc[d] if d in G.index else mean for d in drugs], index=pd.Index(drugs))


def _scale(p, prediction_scale):
    if prediction_scale == 'native':
        return p
    if prediction_scale == 'rank':
        return rank_normal(p)
    raise ValueError("prediction_scale must be 'native' or 'rank'")


# ----------------------------------------------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------------------------------------------

def selective_scores(pred, obs, train_obs=None, min_cells=MIN_CELLS, min_paired=MIN_PAIRED_CELLS, cells=None,
                     groups=None, min_drugs=1, prediction_scale='native'):
    """Per-drug raw and selective scores of predictions against an evaluation screen.

    selective: partial correlation of the predictions with the observed rank-normal profile given the evaluation
    screen's leave-drug-out axis g_T (the metric of the paper). With `train_obs` (the screen the model was trained
    on), selective_both is the partial correlation given g_T and the training screen's leave-drug-out axis g_S, which
    removes any training-screen axis carried by the predictions; both columns are then computed on the same cells
    (those with a defined g_S), so they can be compared drug for drug. A prediction that is an exact linear function
    of g_T and g_S scores selective_both = 0. Also returns the split_terms columns (selective is NaN, as in the
    published scripts, when the prediction is collinear with g_T).
    cells: optional cell set to score on (e.g. cells predicted by every model); prediction_scale='rank' scores
    rank-normal predictions (Spearman-type, robust to outlying predictions)."""
    Zt, Gt = _screen(obs, min_cells, groups, None, min_drugs)
    P = as_matrix(pred, 'prediction')
    cols = Zt.columns.intersection(P.columns)
    if cells is not None:
        cols = cols.intersection(pd.Index(cells))
    Gs = None
    if train_obs is not None:
        Zs, Gs0 = _screen(train_obs, min_cells, groups, None, min_drugs)
        cols = cols.intersection(Zs.columns)
        Gs = _axis_rows(Zs, Gs0, list(Zt.index.intersection(P.index)))
    rows = []
    for d in Zt.index.intersection(P.index):
        p, z, g = _vec(P, d, cols), _vec(Zt, d, cols), _vec(Gt, d, cols)
        m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
        gs = None
        if Gs is not None:
            gs = Gs.loc[d].reindex(cols).values.astype(float)
            m &= np.isfinite(gs)
        if m.sum() < min_paired or np.std(p[m]) < 1e-12:
            continue
        pm = _scale(p[m], prediction_scale)
        row = {'drug': d, 'n': int(m.sum()), **split_terms(pm, z[m], g[m])}
        if gs is not None:
            row['selective_both'] = partial_corr(pm, z[m], [g[m], gs[m]], min_n=min_paired, degenerate=0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_scores_matrix(P, Zs, Gs, Zt, Gt, cells=None, min_paired=MIN_PAIRED_CELLS, prediction_scale='native'):
    """Score one prediction matrix against two screens on matched cells (matrix version of paired_scores).

    For each drug in P and in both screens, the cells used are those (within `cells`, default: cell lines of both
    screens) with a finite prediction and a finite observed profile and axis in BOTH screens. Rows with fewer than
    `min_paired` matched cells or constant predictions are skipped. Returns a long table: drug, n, evaluated_in
    ('source' or 'target'), split_terms columns, and selective_both (given both screens' axes). With the defaults
    this reproduces scripts/35_dreval_crossstudy.score_pair exactly (plus the loading and selective_both columns)."""
    cells = Zs.columns.intersection(Zt.columns) if cells is None else pd.Index(cells)
    rows = []
    for d in P.index.intersection(Zs.index).intersection(Zt.index):
        p = P.loc[d].reindex(cells).values.astype(float)
        zs, gs = Zs.loc[d].reindex(cells).values.astype(float), Gs.loc[d].reindex(cells).values.astype(float)
        zt, gt = Zt.loc[d].reindex(cells).values.astype(float), Gt.loc[d].reindex(cells).values.astype(float)
        m = np.isfinite(p) & np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        if m.sum() < min_paired or np.std(p[m]) < 1e-12:
            continue
        pm = _scale(p[m], prediction_scale)
        for where, z, g in [('source', zs[m], gs[m]), ('target', zt[m], gt[m])]:
            rows.append({'drug': d, 'n': int(m.sum()), 'evaluated_in': where, **split_terms(pm, z, g),
                         'selective_both': partial_corr(pm, z, [gs[m], gt[m]], min_n=min_paired, degenerate=0.0)})
    return pd.DataFrame(rows)


def paired_scores(pred, obs_source, obs_target, min_cells=MIN_CELLS, min_paired=MIN_PAIRED_CELLS, cells=None,
                  groups=None, min_drugs=1, prediction_scale='native'):
    """Paired cross-dataset scoring: identical predictions scored in two screens on matched cells, per drug.

    pred: predictions of a model trained on the source screen (long table); obs_source, obs_target: long tables of
    the two screens. Each screen is decomposed on its own (rank-normal profiles, leave-drug-out axis); per drug the
    same predictions are scored on the same cell lines against both screens, so the difference between the two
    evaluations is due to the evaluation dataset (laboratory, protocol, curve processing, drug panel), not to the
    model, the cells or the predictions. Returns the table of paired_scores_matrix; summarise it with
    paired_summary and two_way_bootstrap."""
    Zs, Gs = _screen(obs_source, min_cells, groups, None, min_drugs)
    Zt, Gt = _screen(obs_target, min_cells, groups, None, min_drugs)
    return paired_scores_matrix(as_matrix(pred, 'prediction'), Zs, Gs, Zt, Gt, cells, min_paired, prediction_scale)


def _wide(table, by):
    key = list(by) + ['drug']
    vals = ['raw', 'selective', 'selective_both', 'alignment', 'loading', 'mediated', 'selective_term']
    vals = [v for v in vals if v in table.columns]
    w = table.pivot_table(index=key, columns='evaluated_in', values=vals)
    need = [(v, e) for v in ['raw', 'selective', 'mediated', 'selective_term'] for e in ('source', 'target')]
    w = w.dropna(subset=[c for c in need if c in w.columns])
    w.columns = [f'{v}_{e}' for v, e in w.columns]
    return w.reset_index()


def paired_summary(table, by=()):
    """Summaries of a paired table over drugs scored in both evaluations (optionally per group, e.g. by='model').

    Medians over drugs: raw and selective (and selective_both) in each evaluation, and the median paired
    difference of the selective score. Means over drugs: split terms, alignment r_pg and loading r_zg. Pooled ratios
    (sums over drugs): share of the loss carried by the fitness-mediated term,
    (sum mediated_s - sum mediated_t) / (sum raw_s - sum raw_t); retained fractions of the mediated and selective
    terms, of the alignment and of the loading (target / source). Absolute differences (target - source, per drug
    averaged) of raw, mediated and selective terms. Ratios with small denominators (small losses) are unstable:
    report them with two_way_bootstrap intervals and prefer the absolute differences."""
    by = [by] if isinstance(by, str) else list(by)
    w = _wide(table, by)

    def one(x):
        out = {'drugs': int(len(x))}
        for e in ('source', 'target'):
            out[f'raw_{e}'] = float(x[f'raw_{e}'].median())
            out[f'sel_{e}'] = float(x[f'selective_{e}'].median())
            if f'selective_both_{e}' in x:
                out[f'sel_both_{e}'] = float(x[f'selective_both_{e}'].median())
            for t in ['raw', 'mediated', 'selective_term', 'alignment', 'loading']:
                if f'{t}_{e}' in x:
                    out[f'mean_{t}_{e}'] = float(x[f'{t}_{e}'].mean())
        S = {t: (x[f'{t}_source'].sum(), x[f'{t}_target'].sum()) for t in
             ['raw', 'mediated', 'selective_term', 'alignment', 'loading'] if f'{t}_source' in x}
        loss = S['raw'][0] - S['raw'][1]
        out['share_of_loss_fitness'] = (S['mediated'][0] - S['mediated'][1]) / loss if abs(loss) > 1e-9 else np.nan
        for t, name in [('mediated', 'mediated_retained'), ('selective_term', 'selterm_retained'),
                        ('alignment', 'alignment_retained'), ('loading', 'loading_retained')]:
            if t in S:
                out[name] = S[t][1] / S[t][0] if abs(S[t][0]) > 1e-12 else np.nan
        n = max(len(x), 1)
        for t in ['raw', 'mediated', 'selective_term']:
            out[f'delta_{t}'] = (S[t][1] - S[t][0]) / n
        out['delta_selective_median'] = float((x['selective_target'] - x['selective_source']).median())
        return out

    if not by:
        return pd.DataFrame([one(w)])
    return pd.DataFrame([{**dict(zip(by, k if isinstance(k, tuple) else (k,))), **one(g)}
                         for k, g in w.groupby(by)])


def model_axis(P, scale='zscore', min_cells=MIN_PAIRED_CELLS, groups=None, min_drugs=1):
    """The model's own leave-drug-out fitness axis.

    Each drug's predicted profile is standardised across cell lines (scale='zscore', default; or 'rank_normal' as in
    scripts/40_model_axis_adjusted.py), and for drug d and cell line c the axis is the mean of the other drugs'
    standardised predictions on c (fitness_axis; leave-class-out with `groups`). Drugs with fewer than `min_cells`
    predictions or constant predictions are not used. With z-scoring and a complete prediction matrix, a drug-agnostic
    predictor (the same value for every drug) is an exact linear function of its own axis."""
    P = P.loc[P.notna().sum(axis=1) >= min_cells]
    if scale == 'zscore':
        mu, sd = P.mean(axis=1), P.std(axis=1, ddof=0)
        sd = sd.where(sd > 1e-12)
        S = P.sub(mu, axis=0).div(sd, axis=0)
    elif scale == 'rank_normal':
        S = pd.DataFrame(np.vstack([rank_normal(r) for r in P.values]) if len(P) else np.zeros((0, P.shape[1])),
                         index=P.index, columns=P.columns)
    else:
        raise ValueError("scale must be 'zscore' or 'rank_normal'")
    S = S.loc[S.notna().any(axis=1)]
    return fitness_axis(S, groups=groups, min_drugs=min_drugs)


def model_axis_scores(P, Z, G, cells=None, extra_axes=None, scale='zscore', min_paired=MIN_PAIRED_CELLS,
                      prediction_scale='native'):
    """Model-axis diagnostic, per drug (matrix version).

    selective: partial correlation of predictions and observed profile given the evaluation axis G (and the
    `extra_axes`, a dict name -> drug x cell matrix, e.g. the training screen's axis). adjusted: additionally given
    the model's own leave-drug-out axis (model_axis of P on the same cells). A prediction that is a linear function of
    its own axis (e.g. any drug-agnostic predictor with a complete prediction matrix) scores exactly 0. Because the
    model axis differs between models, `adjusted` is a diagnostic of dilution by (or leakage from) a model's own
    fitness estimate, not a common benchmark metric. Also returns model_alignment (correlation of p with its axis)."""
    cells = Z.columns.intersection(P.columns) if cells is None else pd.Index(cells)
    P = P.reindex(columns=cells)
    Gh = model_axis(P, scale=scale, min_cells=min_paired)
    extra = extra_axes or {}
    rows = []
    for d in Gh.index.intersection(Z.index):
        p, z, g, gh = _vec(P, d, cells), _vec(Z, d, cells), _vec(G, d, cells), _vec(Gh, d, cells)
        xs = [_vec(E, d, cells) for E in extra.values()]
        m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g) & np.isfinite(gh)
        for x in xs:
            m &= np.isfinite(x)
        if m.sum() < min_paired or np.std(p[m]) < 1e-12:
            continue
        pm = _scale(p[m], prediction_scale)
        base = [g[m]] + [x[m] for x in xs]
        rows.append({'drug': d, 'n': int(m.sum()),
                     'selective': partial_corr(pm, z[m], base, min_n=min_paired),
                     'adjusted': partial_corr(pm, z[m], base + [gh[m]], min_n=min_paired, degenerate=0.0),
                     'model_alignment': float(np.corrcoef(pm, gh[m])[0, 1]) if np.std(gh[m]) > 1e-12 else np.nan})
    return pd.DataFrame(rows)


def model_axis_diagnostic(pred, obs, train_obs=None, min_cells=MIN_CELLS, min_paired=MIN_PAIRED_CELLS, cells=None,
                          scale='zscore', min_drugs=1, prediction_scale='native'):
    """Model-axis diagnostic from long tables: selective score given the evaluation axis (and the training screen's
    axis if `train_obs` is given), and the same additionally given the model's own leave-drug-out axis built from
    per-drug z-scored predictions. See model_axis_scores."""
    Zt, Gt = _screen(obs, min_cells, None, None, min_drugs)
    P = as_matrix(pred, 'prediction')
    cols = Zt.columns.intersection(P.columns)
    if cells is not None:
        cols = cols.intersection(pd.Index(cells))
    extra = None
    if train_obs is not None:
        Zs, Gs0 = _screen(train_obs, min_cells, None, None, min_drugs)
        cols = cols.intersection(Zs.columns)
        extra = {'training_axis': _axis_rows(Zs, Gs0, list(Zt.index))}
    return model_axis_scores(P, Zt, Gt, cells=cols, extra_axes=extra, scale=scale, min_paired=min_paired,
                             prediction_scale=prediction_scale)


def _matrix_to_long(M, name='prediction'):
    v = M.values.astype(float)
    i, j = np.where(np.isfinite(v))
    return pd.DataFrame({'drug': M.index.values[i], 'cell': M.columns.values[j], name: v[i, j]})


def drug_agnostic_predictions(values, drugs):
    """Long table (drug, cell, prediction) repeating one value per cell line (a Series indexed by cell) for every drug."""
    v = pd.Series(values).dropna()
    return pd.DataFrame({'drug': np.repeat(np.asarray(list(drugs), dtype=object), len(v)),
                         'cell': np.tile(v.index.values, len(drugs)), 'prediction': np.tile(v.values, len(drugs))})


def leakage_floor(obs, agnostic=None, train_obs=None, min_cells=MIN_CELLS, min_paired=MIN_PAIRED_CELLS, cells=None,
                  min_drugs=1):
    """Selective scores of drug-agnostic predictors: the floor that a model's selective score must clear.

    agnostic: dict name -> Series (one value per cell line), e.g. an omics-predicted fitness (fitness_baseline),
    a drug-agnostic tissue mean (tissue_mean_baseline) or any naive cell-line mean; a drug x cell DataFrame is also
    accepted. With `train_obs`, the training screen's measured leave-drug-out axis is added as 'training_screen_axis'
    (drug-blind: it never contains the scored drug's own response; an all-drug mean would, see R1 minor 1).
    Each predictor is repeated for every drug of the evaluation screen and scored like a model: selective_scores
    (selective, and selective_both if train_obs) and the model-axis diagnostic (adjusted; exactly 0 for drug-agnostic
    predictors with complete coverage). Within the screen that defines the axis such predictors score close to 0
    (exactly 0 only if the axis were noise-free); across screens they usually do not, because the transferred axis is
    not collinear with the evaluation axis. Returns (per_drug, summary); summary has one row per predictor and
    `floor`, the largest median selective score."""
    agnostic = dict(agnostic or {})
    Zt, Gt = _screen(obs, min_cells, None, None, min_drugs)
    if train_obs is not None:
        Zs, Gs0 = _screen(train_obs, min_cells, None, None, min_drugs)
        agnostic.setdefault('training_screen_axis', _axis_rows(Zs, Gs0, list(Zt.index)))
    per, summ = [], []
    for name, values in agnostic.items():
        pr = (_matrix_to_long(values.reindex(index=Zt.index)) if isinstance(values, pd.DataFrame)
              else drug_agnostic_predictions(values, Zt.index))
        T = selective_scores(pr, obs, train_obs=train_obs, min_cells=min_cells, min_paired=min_paired, cells=cells,
                             min_drugs=min_drugs)
        if not len(T):
            continue
        A = model_axis_diagnostic(pr, obs, train_obs=train_obs, min_cells=min_cells, min_paired=min_paired,
                                  cells=cells, min_drugs=min_drugs)
        if len(A):
            T = T.merge(A[['drug', 'adjusted']], on='drug', how='left')
        T.insert(0, 'predictor', name)
        per.append(T)
        summ.append({'predictor': name, 'drugs': int(len(T)), 'raw': float(T.raw.median()),
                     'selective': float(T.selective.median()),
                     'selective_both': float(T.selective_both.median()) if 'selective_both' in T else np.nan,
                     'adjusted': float(T.adjusted.median()) if 'adjusted' in T else np.nan})
    S = pd.DataFrame(summ)
    if len(S):
        S['floor'] = float(S.selective.max())
    return (pd.concat(per, ignore_index=True) if per else pd.DataFrame()), S


def oracle_fitness_share(Z, G, min_cells=MIN_PAIRED_CELLS):
    """Fitness-mediated share of a perfect predictor (p = z): mean over drugs of r_zg^2.

    The reference against which a model's fitness-mediated share (sum mediated / sum raw) should be read."""
    vals = []
    for d in Z.index.intersection(G.index):
        z, g = Z.loc[d].values.astype(float), G.loc[d].reindex(Z.columns).values.astype(float)
        m = np.isfinite(z) & np.isfinite(g)
        if m.sum() >= min_cells and np.std(g[m]) > 1e-12:
            vals.append(np.corrcoef(z[m], g[m])[0, 1] ** 2)
    return float(np.mean(vals)) if vals else np.nan


def summarize(tables, floor=None, oracle_share=None):
    """Raw and selective scores side by side, one row per model.

    tables: dict model -> per-drug table from selective_scores, model_axis_scores or one evaluation of a paired
    table. Columns: drugs, raw and selective medians (and selective_both, adjusted when present), fitness_share
    (sum mediated / sum raw: the share of per-drug accuracy carried by the fitness axis), and, if given, the leakage
    floor (selective_above_floor = selective - floor) and the perfect-predictor share (oracle_fitness_share)."""
    rows = []
    for name, T in tables.items():
        if not len(T):
            continue
        r = {'model': name, 'drugs': int(T.drug.nunique())}
        for c in ['raw', 'selective', 'selective_both', 'adjusted']:
            if c in T:
                r[c] = float(T[c].median())
        if 'mediated' in T and 'raw' in T:
            r['fitness_share'] = float(T.mediated.sum() / T.raw.sum()) if abs(T.raw.sum()) > 1e-12 else np.nan
        if floor is not None:
            r['leakage_floor'] = float(floor)
            r['selective_above_floor'] = r.get('selective', np.nan) - float(floor)
        if oracle_share is not None:
            r['oracle_fitness_share'] = float(oracle_share)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------------------------------------------

_ALPHAS = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])


def _kernel_ridge_loo(K_tr, y, K_te, alphas=_ALPHAS):
    """Kernel ridge with intercept; penalty (scaled by the mean kernel diagonal) chosen by closed-form LOO error."""
    ym = y.mean()
    yc = y - ym
    w, V = np.linalg.eigh(K_tr)
    w = np.clip(w, 0, None)
    Vy = V.T @ yc
    scale = np.mean(np.diag(K_tr))
    best = None
    for a in np.asarray(alphas) * scale:
        shrink = w / (w + a)
        hdiag = (V ** 2) @ shrink
        loo = (yc - V @ (shrink * Vy)) / np.clip(1 - hdiag - 1.0 / len(y), 1e-6, None)
        err = np.mean(loo ** 2)
        if best is None or err < best[0]:
            best = (err, a)
    return ym + K_te @ (V @ (Vy / (w + best[1])))


def fitness_baseline(train_obs, features, cells=None, drugs=None, min_cells=MIN_CELLS, alphas=_ALPHAS):
    """Omics-predicted drug-agnostic baseline (mandatory for leave-cell-line-out and leave-tissue-out benchmarks).

    The training screen's fitness value of each training cell line (its mean rank-normal profile over all drugs) is
    predicted from the cell lines' features (DataFrame indexed by cell) by ridge regression (linear kernel; features
    standardised on training cell lines only; penalty by closed-form leave-one-out error) for `cells` (default: every
    feature row that is not a training cell line). The single predicted value is repeated for every drug (`drugs`,
    default: the training screen's drugs), so the baseline has no drug-specific information: any per-drug accuracy
    it attains is fitness-mediated. Returns a long table (drug, cell, prediction)."""
    Z, _, _, _ = decompose(train_obs, min_cells)
    f = Z.mean(axis=0)
    tr = [c for c in f.index if c in features.index and np.isfinite(f[c])]
    trs = set(tr)
    cells = [c for c in features.index if c not in trs] if cells is None else [c for c in cells if c in features.index]
    X = features.loc[tr].astype(float)
    mu, sd = X.mean(), X.std().replace(0, 1).fillna(1)
    Xtr = ((X - mu) / sd).fillna(0).values / np.sqrt(X.shape[1])
    Xte = ((features.loc[cells].astype(float) - mu) / sd).fillna(0).values / np.sqrt(X.shape[1])
    pred = _kernel_ridge_loo(Xtr @ Xtr.T, f[tr].values, Xte @ Xtr.T, alphas)
    return drug_agnostic_predictions(pd.Series(pred, index=cells), Z.index if drugs is None else drugs)


def tissue_drug_mean_baseline(train_obs, tissue, cells, drugs=None, min_per_tissue=1):
    """Tissue x drug mean baseline (analogue of DrEval's NaiveTissueDrugMeanPredictor).

    For each drug, a cell line's prediction is the drug's mean training response over training cell lines of the
    same tissue (`tissue`: mapping cell -> tissue), or the drug's overall training mean if that tissue has fewer
    than `min_per_tissue` training cell lines. It uses tissue labels (not feature-free): selectivity it captures is
    tissue-of-origin selectivity. Returns a long table (drug, cell, prediction)."""
    tissue = pd.Series(tissue)
    o = train_obs[['drug', 'cell', 'response']].copy()
    o['tissue'] = o.cell.map(tissue)
    dm = o.groupby('drug').response.mean()
    dt = o.dropna(subset=['tissue']).groupby(['drug', 'tissue']).response.agg(['mean', 'size'])
    dt = dt[dt['size'] >= min_per_tissue]['mean']
    drugs = list(dm.index) if drugs is None else [d for d in drugs if d in dm.index]
    cells = list(cells)
    grid = pd.DataFrame({'drug': np.repeat(np.asarray(drugs, dtype=object), len(cells)),
                         'cell': np.tile(np.asarray(cells, dtype=object), len(drugs))})
    grid['tissue'] = grid.cell.map(tissue)
    key = pd.MultiIndex.from_arrays([grid.drug, grid.tissue])
    grid['prediction'] = dt.reindex(key).values
    miss = grid.prediction.isna()
    grid.loc[miss, 'prediction'] = grid.loc[miss, 'drug'].map(dm).values
    return grid[['drug', 'cell', 'prediction']]


def tissue_mean_baseline(train_obs, tissue, cells, min_cells=MIN_CELLS):
    """Drug-agnostic tissue mean (analogue of DrEval's NaiveTissueMeanPredictor): the mean, over the training cell
    lines of a cell line's tissue, of their fitness value (mean rank-normal profile over all drugs); 0 (the overall
    mean) for unseen tissues. Returns a Series indexed by cell, for leakage_floor or drug_agnostic_predictions."""
    tissue = pd.Series(tissue)
    Z, _, _, _ = decompose(train_obs, min_cells)
    f = Z.mean(axis=0)
    tm = f.groupby(f.index.map(tissue)).mean()
    cells = list(cells)
    return pd.Series([float(tm.get(tissue.get(c), 0.0)) for c in cells], index=cells)


# ----------------------------------------------------------------------------------------------------------------
# Uncertainty and decisions
# ----------------------------------------------------------------------------------------------------------------

def weighted_median(values, counts):
    """Median of a multiset given by values and integer multiplicities (columns are separate replicates).

    values, counts: arrays of shape (n,) or (n, B). Entries with count 0 or non-finite values are ignored. Equals
    np.median(np.repeat(values, counts)) for each column."""
    v, c = np.asarray(values, float), np.asarray(counts)
    one = v.ndim == 1
    if one:
        v, c = v[:, None], c[:, None]
    out = np.full(v.shape[1], np.nan)
    for b in range(v.shape[1]):
        ok = (c[:, b] > 0) & np.isfinite(v[:, b])
        if not ok.any():
            continue
        vv, cc = v[ok, b], c[ok, b]
        o = np.argsort(vv, kind='mergesort')
        vv, cum = vv[o], np.cumsum(cc[o])
        n = cum[-1]
        lo = np.searchsorted(cum, (n - 1) // 2, side='right')
        hi = np.searchsorted(cum, n // 2, side='right')
        out[b] = 0.5 * (vv[lo] + vv[hi])
    return out[0] if one else out


def _prep_eval(p, z, g, mask):
    """Row-centred, masked arrays and their products for weighted per-drug correlations."""
    def c(x):
        x = np.where(mask, x, 0.0)
        n = mask.sum(axis=1, keepdims=True)
        return np.where(mask, x - x.sum(axis=1, keepdims=True) / np.maximum(n, 1), 0.0)
    p, z, g = c(p), c(z), c(g)
    one = mask.astype(float)
    return [one, p, z, g, p * p, z * z, g * g, p * z, p * g, z * g]


def _eval_terms(mats, W):
    """Per-drug correlations (r_pz, r_pg, r_zg) under cell weights W (cells x B); returns dict of (drugs x B)."""
    S = [M @ W for M in mats]
    n = S[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        mp, mz, mg = S[1] / n, S[2] / n, S[3] / n
        vp, vz, vg = S[4] / n - mp ** 2, S[5] / n - mz ** 2, S[6] / n - mg ** 2
        cpz, cpg, czg = S[7] / n - mp * mz, S[8] / n - mp * mg, S[9] / n - mz * mg
        r_pz, r_pg, r_zg = cpz / np.sqrt(vp * vz), cpg / np.sqrt(vp * vg), czg / np.sqrt(vz * vg)
        den = np.sqrt(np.clip((1 - r_pg ** 2) * (1 - r_zg ** 2), 0, None))
        sel = np.where(den > 1e-9, (r_pz - r_pg * r_zg) / den, np.nan)
    bad = n < 3
    out = {'raw': r_pz, 'alignment': r_pg, 'loading': r_zg, 'mediated': r_pg * r_zg,
           'selective_term': r_pz - r_pg * r_zg, 'selective': sel}
    for k in out:
        out[k] = np.where(bad, np.nan, out[k])
    return out


def two_way_bootstrap(P, Zs, Gs, Zt=None, Gt=None, n_boot=1000, seed=0, cells=None, min_paired=MIN_PAIRED_CELLS,
                      level=0.95, drug_only=False, batch=250):
    """Two-way (drug x cell) bootstrap of medians and pooled ratios of per-drug scores.

    P: prediction matrix (drug x cell) or dict name -> matrix (models pooled: the same drug and cell resamples are
    applied to every model, so the interval is over drugs and cells, not over a population of methods).
    Single-screen mode (Zt is None): scores against (Zs, Gs). Paired mode: identical predictions against the source
    (Zs, Gs) and target (Zt, Gt) screens on matched cells, as in paired_scores_matrix.
    Each replicate resamples drugs with replacement (multiplicities) and, unless drug_only=True, cell lines with
    replacement (integer weights in every per-drug correlation); the screens' profiles and axes are held fixed.
    Statistics (pooled over models when P is a dict): median raw and selective per evaluation, fitness_share
    (sum mediated / sum raw); in paired mode also share_of_loss_fitness, mediated_retained, selterm_retained,
    alignment_retained, loading_retained, delta_raw, delta_mediated, delta_selective_term and
    delta_selective_median (target - source). Returns (summary, samples): summary has estimate (no resampling),
    lo, hi (percentile interval at `level`) and se per statistic; samples is an n_boot x statistics DataFrame.
    The same seed gives identical results, whatever `batch` (which only bounds memory)."""
    models = P if isinstance(P, dict) else {'model': P}
    paired = Zt is not None
    if cells is None:
        cells = Zs.columns.intersection(Zt.columns) if paired else Zs.columns
    cells = pd.Index(cells)
    drugs = Zs.index if not paired else Zs.index.intersection(Zt.index)
    drugs = drugs.intersection(pd.Index(sorted(set().union(*[set(M.index) for M in models.values()]))))
    evals = ['source', 'target'] if paired else ['source']
    screens = {'source': (Zs.reindex(index=drugs, columns=cells).values.astype(float),
                          Gs.reindex(index=drugs, columns=cells).values.astype(float))}
    if paired:
        screens['target'] = (Zt.reindex(index=drugs, columns=cells).values.astype(float),
                             Gt.reindex(index=drugs, columns=cells).values.astype(float))
    base = np.ones((len(drugs), len(cells)), bool)
    for z, g in screens.values():
        base &= np.isfinite(z) & np.isfinite(g)
    prepared = []                              # per model: predictions and mask only (products built per batch)
    for M in models.values():
        p = M.reindex(index=drugs, columns=cells).values.astype(float)
        mask = base & np.isfinite(p)
        sd = np.array([np.std(p[i, mask[i]]) if mask[i].sum() else 0.0 for i in range(len(drugs))])
        mask &= ((mask.sum(axis=1) >= min_paired) & (sd >= 1e-12))[:, None]
        prepared.append((p, mask))
    keys = ['raw', 'mediated', 'selective_term', 'alignment', 'loading']

    def stats(W, Cd):
        """W: cells x B weights; Cd: drugs x B drug multiplicities -> dict of B-vectors."""
        B = W.shape[1]
        sums = {e: {k: np.zeros(B) for k in keys} for e in evals}
        med = {e: {'raw': [], 'selective': []} for e in evals}
        dsel, cnts, nn = [], [], np.zeros(B)
        for p, mask in prepared:
            T = {e: _eval_terms(_prep_eval(p, screens[e][0], screens[e][1], mask), W) for e in evals}
            ok = np.all([np.isfinite(T[e]['raw']) & np.isfinite(T[e]['selective']) for e in evals], axis=0)
            c = np.where(ok, Cd, 0)
            cnts.append(c)
            nn += c.sum(axis=0)
            for e in evals:
                for k in keys:
                    sums[e][k] += (np.where(c > 0, T[e][k], 0.0) * c).sum(axis=0)
                med[e]['raw'].append(T[e]['raw'])
                med[e]['selective'].append(T[e]['selective'])
            if paired:
                dsel.append(T['target']['selective'] - T['source']['selective'])
        cc = np.vstack(cnts)
        out = {}
        for e in evals:
            out[f'median_raw_{e}'] = weighted_median(np.vstack(med[e]['raw']), cc)
            out[f'median_selective_{e}'] = weighted_median(np.vstack(med[e]['selective']), cc)
        with np.errstate(invalid='ignore', divide='ignore'):
            for e in evals:
                out[f'fitness_share_{e}'] = sums[e]['mediated'] / sums[e]['raw']
            if paired:
                s, t = sums['source'], sums['target']
                loss = s['raw'] - t['raw']
                out['share_of_loss_fitness'] = np.where(np.abs(loss) > 1e-9, (s['mediated'] - t['mediated']) / loss,
                                                        np.nan)
                out['mediated_retained'] = t['mediated'] / s['mediated']
                out['selterm_retained'] = t['selective_term'] / s['selective_term']
                out['alignment_retained'] = t['alignment'] / s['alignment']
                out['loading_retained'] = t['loading'] / s['loading']
                for k in ['raw', 'mediated', 'selective_term']:
                    out[f'delta_{k}'] = (t[k] - s[k]) / nn
                out['delta_selective_median'] = weighted_median(np.vstack(dsel), cc)
        return out

    nd, nc = len(drugs), len(cells)
    point = {k: float(v[0]) for k, v in stats(np.ones((nc, 1)), np.ones((nd, 1), dtype=int)).items()}
    rng = np.random.default_rng(seed)        # all resamples drawn up front: results do not depend on `batch`
    Cd_all = rng.multinomial(nd, np.full(nd, 1.0 / nd), size=n_boot)
    W_all = None if drug_only else rng.multinomial(nc, np.full(nc, 1.0 / nc), size=n_boot)
    chunks = []
    for start in range(0, n_boot, batch):
        sl = slice(start, min(start + batch, n_boot))
        Cd = Cd_all[sl].T
        W = np.ones((nc, Cd.shape[1])) if drug_only else W_all[sl].T.astype(float)
        chunks.append(pd.DataFrame(stats(W, Cd)))
    samples = pd.concat(chunks, ignore_index=True)[list(point)]
    a = (1 - level) / 2
    summary = pd.DataFrame({'estimate': pd.Series(point), 'lo': samples.quantile(a), 'hi': samples.quantile(1 - a),
                            'se': samples.std()})
    summary.attrs.update({'n_boot': n_boot, 'seed': seed, 'drugs': nd, 'cells': nc, 'models': len(models),
                          'drug_only': drug_only})
    return summary, samples


def significant(p, alpha=0.05):
    """Decision rule for a p-value: True if and only if p is a finite number and p < alpha.

    A p-value that underflows to 0.0 is the strongest evidence and must count as significant. The idiom
    `(p or 1) < alpha` turns p = 0.0 into 1 (0.0 is falsy); it caused the H4 flag error in scripts/07_confirm.py
    (NCI-60 P stored as 0.0, flag False). Missing p-values (None, NaN) are not significant."""
    if p is None:
        return False
    try:
        p = float(p)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(p) and p < alpha)


def paired_comparison(a, b, alternative='two-sided'):
    """Drug-for-drug comparison of two per-drug score Series (e.g. selective scores of two models).

    Uses the drugs present (finite) in both. Returns drugs, median_diff (a - b), frac_better (a > b) and the Wilcoxon
    signed-rank P (zero differences dropped). Decide with significant(p), never with `(p or 1) < alpha`. The test
    treats drugs as independent; drugs share cell lines, so pair it with two_way_bootstrap intervals."""
    from scipy.stats import wilcoxon
    x = (pd.Series(a) - pd.Series(b)).dropna()
    x = x[np.isfinite(x)]
    out = {'drugs': int(len(x)), 'median_diff': float(x.median()) if len(x) else np.nan,
           'frac_better': float((x > 0).mean()) if len(x) else np.nan, 'p': np.nan}
    if len(x) and (x != 0).any():
        out['p'] = float(wilcoxon(x.values, alternative=alternative).pvalue)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Reliability normalisation
# ----------------------------------------------------------------------------------------------------------------

def _warn_low_reliability(rel, warn_below):
    r = pd.Series(np.asarray(rel, float))
    low = r.notna() & (r < warn_below)
    if low.any():
        worst = max(float(r[low].min()), 1e-12)
        warnings.warn(f'{int(low.sum())} of {int(r.notna().sum())} per-drug reliabilities are below {warn_below} '
                      f'(minimum {float(r[low].min()):.3f}); dividing by sqrt(reliability) would inflate those '
                      f'selective scores up to {1 / np.sqrt(worst):.1f}-fold. Use normalize_by_reliability (shrinkage '
                      f'and a minimum reliability) or an aggregate reliability; reliabilities are specific to the '
                      f'screen and endpoint they were estimated on.', ReliabilityWarning, stacklevel=3)


def shrink_reliability(rel, weight=None, se=None, target=None, bounds=(0.0, 1.0)):
    """Shrink per-drug reliabilities toward a common value.

    rel: per-drug reliabilities (Series, dict or array; NaN kept). weight: fixed shrinkage weight in [0, 1]
    (0 = per-drug values, 1 = the common value for every drug, i.e. aggregate normalisation). se: per-drug standard
    errors; when given and weight is None, empirical-Bayes weights se_d^2 / (se_d^2 + tau^2) are used, with
    tau^2 = max(var(rel) - mean(se^2), 0) (method of moments) and the precision-weighted mean as target. With neither,
    weight = DEFAULT_SHRINKAGE. target: the common value (default: median of the finite reliabilities, or the
    precision-weighted mean in the empirical-Bayes case). Results are clipped to `bounds`."""
    idx = rel.index if isinstance(rel, pd.Series) else (list(rel) if isinstance(rel, dict) else None)
    r = pd.Series(rel, dtype=float) if isinstance(rel, (pd.Series, dict)) else pd.Series(np.asarray(rel, float))
    fin = r[np.isfinite(r)]
    if not len(fin):
        return r
    if weight is None and se is not None:
        if isinstance(se, (pd.Series, dict)):
            s2 = pd.Series(se, dtype=float).reindex(r.index) ** 2
        else:
            s2 = pd.Series(np.asarray(se, float) ** 2, index=r.index)
        ok = np.isfinite(r) & np.isfinite(s2)
        tau2 = max(float(r[ok].var(ddof=1) - s2[ok].mean()), 0.0) if ok.sum() > 1 else 0.0
        if target is None:
            wt = 1.0 / (s2[ok] + tau2) if (tau2 > 0 or (s2[ok] > 0).all()) else pd.Series(1.0, index=r[ok].index)
            t = float((wt * r[ok]).sum() / wt.sum())
        else:
            t = float(target)
        lam = (s2 / (s2 + tau2)) if tau2 > 0 else pd.Series(1.0, index=r.index)
        lam = lam.fillna(1.0)          # no standard error: shrink fully to the common value
    else:
        t = float(np.median(fin)) if target is None else float(target)
        lam = float(DEFAULT_SHRINKAGE if weight is None else np.clip(weight, 0.0, 1.0))
    out = ((1 - lam) * r + lam * t).clip(*bounds)
    out[~np.isfinite(r)] = np.nan
    if idx is not None and not isinstance(rel, pd.Series):
        out.index = idx
    return out


def normalize_by_reliability(scores, rel, weight=None, se=None, target=None, min_reliability=MIN_RELIABILITY,
                             warn_below=RELIABILITY_FLOOR):
    """Ceiling-normalised selective scores: score / sqrt(shrunken reliability), with a minimum reliability.

    scores, rel: aligned Series (by drug) or arrays. Reliabilities are shrunk with shrink_reliability (weight / se /
    target); drugs whose shrunken reliability is below `min_reliability` are not normalised (NaN, 'not
    estimable'), which caps the inflation at 1/sqrt(min_reliability). A ReliabilityWarning is raised when any raw
    per-drug reliability is below `warn_below` (None disables it). Per-drug reliabilities are specific to the screen
    and endpoint on which they were estimated; prefer aggregate normalisation (weight=1) when they come from another
    endpoint. Returns a DataFrame with columns score, reliability, reliability_shrunk, normalized."""
    if isinstance(scores, pd.Series):
        s = scores.astype(float)
        r = pd.Series(rel, dtype=float).reindex(s.index) if isinstance(rel, (pd.Series, dict)) else \
            pd.Series(np.asarray(rel, float), index=s.index)
    else:
        s = pd.Series(np.asarray(scores, float))
        r = pd.Series(np.asarray(rel, float), index=s.index)
    if warn_below is not None:
        _warn_low_reliability(r.values, warn_below)
    sev = None if se is None else (pd.Series(se, dtype=float).reindex(s.index) if isinstance(se, (pd.Series, dict))
                                   else pd.Series(np.asarray(se, float), index=s.index))
    rs = shrink_reliability(r, weight=weight, se=sev, target=target)
    norm_ = s / np.sqrt(rs.where(rs >= min_reliability))
    return pd.DataFrame({'score': s, 'reliability': r, 'reliability_shrunk': rs, 'normalized': norm_})
