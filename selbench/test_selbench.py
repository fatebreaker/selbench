"""Tests for selbench on synthetic screens with a known fitness axis and known drug-specific selectivity."""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import selbench


def synthetic(seed=0, n_drugs=80, n_cells=150):
    rng = np.random.default_rng(seed)
    fitness = rng.normal(size=n_cells)
    selectivity = rng.normal(size=(n_drugs, n_cells))
    loading = rng.uniform(0.5, 1.5, n_drugs)
    resp = loading[:, None] * fitness[None] + selectivity + rng.normal(scale=0.5, size=(n_drugs, n_cells))
    drugs, cells = [f'd{i}' for i in range(n_drugs)], [f'c{j}' for j in range(n_cells)]
    obs = pd.DataFrame([(d, c, resp[i, j]) for i, d in enumerate(drugs) for j, c in enumerate(cells)],
                       columns=['drug', 'cell', 'response'])
    return obs, fitness, selectivity, drugs, cells


def test_rank_normal_preserves_nan_and_order():
    x = np.array([3.0, np.nan, 1.0, 2.0])
    z = selbench.rank_normal(x)
    assert np.isnan(z[1]) and z[2] < z[3] < z[0]


def test_fitness_only_predictor_scores_raw_but_not_selective():
    obs, fitness, selectivity, drugs, cells = synthetic()
    pred = pd.DataFrame([(d, c, fitness[j]) for d in drugs for j, c in enumerate(cells)], columns=['drug', 'cell', 'prediction'])
    T, s = selbench.evaluate(pred, obs)
    assert s['raw'] > 0.4 and abs(s['selective']) < 0.15


def test_selectivity_predictor_scores_selective():
    obs, fitness, selectivity, drugs, cells = synthetic()
    pred = pd.DataFrame([(d, c, selectivity[i, j]) for i, d in enumerate(drugs) for j, c in enumerate(cells)],
                        columns=['drug', 'cell', 'prediction'])
    T, s = selbench.evaluate(pred, obs, reliability={d: 0.64 for d in drugs})
    assert s['selective'] > 0.6
    assert abs(s['selective_ceiling_normalized'] - s['selective'] / 0.8) < 1e-9


# ----------------------------------------------------------------------------------------------------------------
# Revision 1: helpers and reference implementations
# ----------------------------------------------------------------------------------------------------------------

def _long(M, drugs, cells, name):
    i, j = np.where(np.isfinite(M))
    return pd.DataFrame({'drug': np.asarray(drugs, dtype=object)[i], 'cell': np.asarray(cells, dtype=object)[j],
                         name: M[i, j]})


def two_screens(seed=1, n_drugs=60, n_cells=220, missing=0.1):
    """A training (source) and an evaluation (target) screen of the same drugs and cell lines.

    Both share drug-specific selectivity and a fitness axis f1; in the target screen a class of 30% of the drugs also
    loads on a second, class-specific axis f2, which tilts the target's leave-drug-out axis away from f1 (so that a
    drug-agnostic predictor of f1 acquires a non-zero 'selective' score there). Missingness differs between screens:
    the first 20 cell lines are not measured in the target, and drug d1 is measured on cell lines 0-40 only in the
    source, so it has too few matched cells for paired scoring."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n_cells), rng.normal(size=n_cells)
    sel = rng.normal(size=(n_drugs, n_cells))
    cls = np.arange(n_drugs) < int(0.3 * n_drugs)
    src = rng.uniform(0.5, 1.5, n_drugs)[:, None] * f1 + sel + rng.normal(scale=0.7, size=(n_drugs, n_cells))
    tgt = (rng.uniform(0.5, 1.5, n_drugs)[:, None] * f1 + 2.0 * cls[:, None] * f2 + sel
           + rng.normal(scale=0.7, size=(n_drugs, n_cells)))
    src[rng.random(src.shape) < missing] = np.nan
    tgt[rng.random(tgt.shape) < missing] = np.nan
    tgt[:, :20] = np.nan
    src[1, 41:] = np.nan
    pred = 0.8 * f1[None] + 0.5 * sel + rng.normal(scale=0.8, size=(n_drugs, n_cells))
    pred[rng.random(pred.shape) < 0.05] = np.nan
    drugs, cells = [f'd{i}' for i in range(n_drugs)], [f'c{j}' for j in range(n_cells)]
    return {'obs_s': _long(src, drugs, cells, 'response'), 'obs_t': _long(tgt, drugs, cells, 'response'),
            'pred': _long(pred, drugs, cells, 'prediction'), 'f1': pd.Series(f1, index=cells), 'sel': sel,
            'drugs': drugs, 'cells': cells}


def _rank_normal_v0(x):
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    if ok.sum():
        out[ok] = norm.ppf((rankdata(x[ok]) - 0.5) / ok.sum())
    return out


def _decompose_v0(obs, min_cells=30):
    """Verbatim copy of the original (pre-revision) selbench.decompose."""
    M = obs.pivot_table(index='drug', columns='cell', values='response', aggfunc='mean')
    M = M.loc[M.notna().sum(axis=1) >= min_cells]
    M = M.loc[[d for d in M.index if M.loc[d].dropna().nunique() >= 10]]
    Z = pd.DataFrame(np.vstack([_rank_normal_v0(r) for r in M.values]), index=M.index, columns=M.columns)
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


def _terms_35(p, z, g):
    """Verbatim copy of scripts/35_dreval_crossstudy._terms."""
    c = np.corrcoef(np.vstack([p, z, g]))
    r_pz, r_pg, r_zg = c[0, 1], c[0, 2], c[1, 2]
    den = np.sqrt(max((1 - r_pg ** 2) * (1 - r_zg ** 2), 0))
    return {'raw': r_pz, 'selective': (r_pz - r_pg * r_zg) / den if den > 1e-9 else np.nan,
            'alignment': r_pg, 'mediated': r_pg * r_zg, 'selective_term': r_pz - r_pg * r_zg}


def _score_pair_35(P, Zs, Gs, Zt, Gt, cells):
    """Verbatim copy of scripts/35_dreval_crossstudy.score_pair (used by scripts 37, 39 and 40)."""
    within, cross = [], []
    for d in P.index.intersection(Zs.index).intersection(Zt.index):
        p = P.loc[d].reindex(cells).values.astype(float)
        zs, gs = Zs.loc[d].reindex(cells).values.astype(float), Gs.loc[d].reindex(cells).values.astype(float)
        zt, gt = Zt.loc[d].reindex(cells).values.astype(float), Gt.loc[d].reindex(cells).values.astype(float)
        m = np.isfinite(p) & np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        if m.sum() < 30 or np.std(p[m]) < 1e-12:
            continue
        within.append({'drug': d, 'n': int(m.sum()), **_terms_35(p[m], zs[m], gs[m])})
        cross.append({'drug': d, 'n': int(m.sum()), **_terms_35(p[m], zt[m], gt[m])})
    return pd.DataFrame(within), pd.DataFrame(cross)


def _model_axis_40(P):
    """Verbatim copy of scripts/40_model_axis_adjusted.model_axis (rank-normal predicted profiles)."""
    Zp = pd.DataFrame(np.vstack([selbench.rank_normal(r) for r in P.values]), index=P.index, columns=P.columns)
    v = Zp.values; ok = np.isfinite(v)
    tot, cnt = np.nansum(v, axis=0), ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return pd.DataFrame((tot[None] - np.where(ok, v, 0)) / (cnt[None] - ok), index=P.index, columns=P.columns)


def _partial_by_regression(x, y, covs):
    """Partial correlation from the t statistic of x in the OLS fit y ~ 1 + x + covariates: t / sqrt(t^2 + df)."""
    n = len(x)
    X = np.column_stack([np.ones(n), x] + list(covs))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    df = n - X.shape[1]
    cov = (resid @ resid / df) * np.linalg.inv(X.T @ X)
    t = beta[1] / np.sqrt(cov[1, 1])
    return t / np.sqrt(t ** 2 + df)


def _partial_by_precision(x, y, covs):
    """Partial correlation from the inverse correlation matrix: -W_xy / sqrt(W_xx W_yy)."""
    W = np.linalg.inv(np.corrcoef(np.vstack([x, y] + list(covs))))
    return -W[0, 1] / np.sqrt(W[0, 0] * W[1, 1])


# ----------------------------------------------------------------------------------------------------------------
# Revision 1: tests
# ----------------------------------------------------------------------------------------------------------------

def test_decompose_and_evaluate_backward_compatible():
    obs, fitness, selectivity, drugs, cells = synthetic(seed=13, n_drugs=40, n_cells=120)
    obs = obs[np.random.default_rng(13).random(len(obs)) > 0.1]          # with missing values
    new, old = selbench.decompose(obs), _decompose_v0(obs)
    for a, b in zip(new, old):
        assert a.equals(b)
    G = selbench.fitness_axis(new[0])                                   # default axis: identical arithmetic
    assert np.array_equal(G.values, new[1].values, equal_nan=True)
    R, r2 = selbench.selective_residuals(new[0], new[1])
    assert np.allclose(R.values, new[2].values, equal_nan=True, atol=1e-12) and np.allclose(r2, new[3], atol=1e-12)
    pred = pd.DataFrame([(d, c, fitness[j]) for d in drugs for j, c in enumerate(cells)],
                        columns=['drug', 'cell', 'prediction'])
    T, s = selbench.evaluate(pred, obs)
    assert list(T.columns) == ['drug', 'n', 'raw', 'selective', 'fitness_r2']
    assert set(s) == {'raw', 'selective', 'fitness_r2', 'drugs'}


def test_split_terms_identity():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(30, 300))
        g = rng.normal(size=n)
        z = rng.uniform(-1, 1) * g + rng.normal(size=n)
        p = rng.uniform(-1, 1) * g + rng.uniform(-1, 1) * z + rng.normal(size=n)
        t = selbench.split_terms(p, z, g)
        assert abs(t['raw'] - (t['mediated'] + t['selective_term'])) < 1e-12
        f = np.sqrt((1 - t['alignment'] ** 2) * (1 - t['loading'] ** 2))
        assert abs(t['selective_term'] - f * t['selective']) < 1e-12
        assert abs(t['raw'] - np.corrcoef(p, z)[0, 1]) < 1e-12
        assert abs(t['loading'] - np.corrcoef(z, g)[0, 1]) < 1e-12
        assert abs(t['alignment'] - np.corrcoef(p, g)[0, 1]) < 1e-12
    S = two_screens()
    T = selbench.paired_scores(S['pred'], S['obs_s'], S['obs_t'])        # every row of a paired table
    f = np.sqrt((1 - T.alignment ** 2) * (1 - T.loading ** 2))
    assert len(T) > 50
    assert np.allclose(T.raw, T.mediated + T.selective_term, atol=1e-12, rtol=0)
    assert np.allclose(T.selective_term, f * T.selective, atol=1e-12, rtol=0)


def test_partial_correlations_match_regression():
    rng = np.random.default_rng(4)
    for k in [1, 2, 3]:
        for _ in range(20):
            n = 120
            covs = [rng.normal(size=n) for _ in range(k)]
            x = sum(rng.normal() * c for c in covs) + rng.normal(size=n)
            y = sum(rng.normal() * c for c in covs) + rng.uniform(-1, 1) * x + rng.normal(size=n)
            r = selbench.partial_corr(x, y, covs)
            assert abs(r - _partial_by_regression(x, y, covs)) < 1e-10
            assert abs(r - _partial_by_precision(x, y, covs)) < 1e-10
            if k == 1:
                assert abs(r - selbench._partial(x, y, covs[0])) < 1e-12
                assert abs(r - selbench.split_terms(x, y, covs[0])['selective']) < 1e-10
    # complete cases only; too few cases -> NaN; x explained by the covariates -> `degenerate`
    x[:5], y[5:8] = np.nan, np.nan
    m = np.isfinite(x) & np.isfinite(y)
    assert abs(selbench.partial_corr(x, y, covs) - _partial_by_regression(x[m], y[m], [c[m] for c in covs])) < 1e-10
    assert np.isnan(selbench.partial_corr(x[:15], y[:15], [c[:15] for c in covs]))
    assert np.isnan(selbench.partial_corr(covs[0], y, covs))
    assert selbench.partial_corr(2 * covs[0] - covs[1], y, covs, degenerate=0.0) == 0.0


def test_model_axis_diagnostic_scores_drug_agnostic_predictors_zero():
    obs, fitness, selectivity, drugs, cells = synthetic(seed=5)
    rng = np.random.default_rng(5)
    Z, G, _, _ = selbench.decompose(obs)
    tissue = rng.integers(0, 6, len(cells))
    agnostic = {'true_fitness': fitness, 'noisy_fitness': fitness + rng.normal(size=len(cells)),
                'tissue_mean': np.array([fitness[tissue == t].mean() for t in tissue]),
                'unrelated': rng.normal(size=len(cells))}
    for name, v in agnostic.items():
        P = pd.DataFrame(np.tile(v, (len(drugs), 1)), index=drugs, columns=cells)
        T = selbench.model_axis_scores(P, Z, G)
        assert len(T) == len(drugs), name
        assert (T.adjusted == 0.0).all(), name
        assert np.allclose(T.model_alignment, 1.0)
    # the same through the long-table API, with another screen's axis as an extra covariate
    S = two_screens(seed=6)
    pr = selbench.drug_agnostic_predictions(S['f1'], S['drugs'])
    D = selbench.model_axis_diagnostic(pr, S['obs_t'], train_obs=S['obs_s'])
    assert len(D) > 50 and (D.adjusted == 0.0).all()
    # genuinely drug-specific predictions keep their score
    P = pd.DataFrame(selectivity, index=drugs, columns=cells)
    T = selbench.model_axis_scores(P, Z, G)
    assert T.adjusted.median() > 0.6 and abs(T.adjusted.median() - T.selective.median()) < 0.05
    # rank-normal model axis reproduces scripts/40_model_axis_adjusted.model_axis
    Pn = P + rng.normal(scale=0.1, size=P.shape)
    Pn.iloc[:, :5] = np.nan
    assert np.allclose(selbench.model_axis(Pn, scale='rank_normal').values, _model_axis_40(Pn).values,
                       equal_nan=True, atol=1e-14)


def test_paired_scoring_uses_matched_cells():
    S = two_screens()
    T = selbench.paired_scores(S['pred'], S['obs_s'], S['obs_t'])
    Zs, Gs, _, _ = selbench.decompose(S['obs_s'])
    Zt, Gt, _, _ = selbench.decompose(S['obs_t'])
    P = selbench.as_matrix(S['pred'])
    cells = Zs.columns.intersection(Zt.columns)
    assert not set(S['cells'][:20]) & set(cells)                           # target-only gaps are not used
    for d, g in T.groupby('drug'):
        assert sorted(g.evaluated_in) == ['source', 'target'] and g.n.nunique() == 1
        v = [M.loc[d].reindex(cells).values.astype(float) for M in (P, Zs, Gs, Zt, Gt)]
        m = np.all([np.isfinite(x) for x in v], axis=0)
        assert g.n.iloc[0] == m.sum()
        for where, z, gg in [('source', v[1], v[2]), ('target', v[3], v[4])]:
            ref = selbench.split_terms(v[0][m], z[m], gg[m])
            row = g[g.evaluated_in == where].iloc[0]
            assert all(abs(row[k] - ref[k]) < 1e-12 for k in ref)
            assert abs(row.selective_both - selbench.partial_corr(v[0][m], z[m], [v[2][m], v[4][m]])) < 1e-12
    assert 'd1' not in set(T.drug)                                         # too few matched cells
    assert 'd1' in set(Zs.index) and 'd1' in set(Zt.index)
    # unpaired scoring in the source screen uses more cells for the same drugs
    U = selbench.selective_scores(S['pred'], S['obs_s']).set_index('drug').n
    src = T[T.evaluated_in == 'source'].set_index('drug').n
    assert (U.reindex(src.index) >= src).all() and (U.reindex(src.index) > src).any()
    # identical to the published implementation (scripts/35_dreval_crossstudy.score_pair)
    w, c = _score_pair_35(P, Zs, Gs, Zt, Gt, cells)
    for where, ref in [('source', w), ('target', c)]:
        new = T[T.evaluated_in == where].reset_index(drop=True)
        assert list(new.drug) == list(ref.drug)
        for k in ['n', 'raw', 'selective', 'alignment', 'mediated', 'selective_term']:
            assert np.array_equal(new[k].values, ref[k].values, equal_nan=True), k
    # pooled summary: share of the loss and retained fractions are consistent with the means
    s = selbench.paired_summary(T).iloc[0]
    assert abs(s.share_of_loss_fitness - (s.mean_mediated_source - s.mean_mediated_target)
               / (s.mean_raw_source - s.mean_raw_target)) < 1e-12
    assert abs(s.delta_raw - (s.delta_mediated + s.delta_selective_term)) < 1e-12
    assert abs(s.alignment_retained - s.mean_alignment_target / s.mean_alignment_source) < 1e-12
    by = selbench.paired_summary(pd.concat([T.assign(model='a'), T.assign(model='b')]), by='model')
    assert list(by.model) == ['a', 'b'] and np.allclose(by.share_of_loss_fitness, s.share_of_loss_fitness)


def test_selective_scores_both_axes_and_leakage_floor():
    S = two_screens(seed=2)
    T = selbench.selective_scores(S['pred'], S['obs_t'], train_obs=S['obs_s'])
    assert {'selective', 'selective_both', 'mediated', 'loading'} <= set(T.columns)
    per, floor = selbench.leakage_floor(S['obs_t'], {'true_f1': S['f1']}, train_obs=S['obs_s'])
    f = floor.set_index('predictor')
    assert set(f.index) == {'true_f1', 'training_screen_axis'}
    assert (per.adjusted == 0.0).all()                                     # model-axis diagnostic: exactly 0
    # across screens a drug-agnostic predictor acquires a 'selective' score (the leakage floor) ...
    assert f.loc['true_f1', 'selective'] > 0.1 and f.loc['training_screen_axis', 'selective'] > 0.1
    # ... which partialling the training screen's axis as well removes or reduces
    assert f.loc['training_screen_axis', 'selective_both'] == 0.0
    assert f.loc['true_f1', 'selective_both'] < f.loc['true_f1', 'selective']
    assert np.isclose(floor.floor.iloc[0], f.selective.max())
    # within the screen that defines the axis the floor is smaller
    _, within = selbench.leakage_floor(S['obs_s'], {'true_f1': S['f1']})
    assert within.selective.iloc[0] < f.loc['true_f1', 'selective']
    # side-by-side report against the floor
    rep = selbench.summarize({'model': T}, floor=floor.floor.iloc[0], oracle_share=0.3)
    assert {'raw', 'selective', 'selective_both', 'fitness_share', 'selective_above_floor'} <= set(rep.columns)
    # rank-scale scores are invariant to a monotone transformation of the predictions
    A = selbench.selective_scores(S['pred'], S['obs_t'], prediction_scale='rank')
    B = selbench.selective_scores(S['pred'].assign(prediction=np.exp(S['pred'].prediction)), S['obs_t'],
                                  prediction_scale='rank')
    assert np.allclose(A.selective, B.selective, atol=1e-12)


def test_two_way_bootstrap_shapes_and_reproducibility():
    S = two_screens(seed=3)
    Zs, Gs, _, _ = selbench.decompose(S['obs_s'])
    Zt, Gt, _, _ = selbench.decompose(S['obs_t'])
    P = selbench.as_matrix(S['pred'])
    s1, b1 = selbench.two_way_bootstrap(P, Zs, Gs, Zt, Gt, n_boot=60, seed=7, batch=25)
    s2, b2 = selbench.two_way_bootstrap(P, Zs, Gs, Zt, Gt, n_boot=60, seed=7, batch=25)
    s3, b3 = selbench.two_way_bootstrap(P, Zs, Gs, Zt, Gt, n_boot=60, seed=8, batch=25)
    assert b1.shape == (60, len(s1)) and list(b1.columns) == list(s1.index)
    assert list(s1.columns) == ['estimate', 'lo', 'hi', 'se']
    assert {'share_of_loss_fitness', 'mediated_retained', 'selterm_retained', 'alignment_retained',
            'loading_retained', 'median_selective_target', 'delta_selective_median'} <= set(s1.index)
    pd.testing.assert_frame_equal(b1, b2)
    pd.testing.assert_frame_equal(s1, s2)
    _, b1b = selbench.two_way_bootstrap(P, Zs, Gs, Zt, Gt, n_boot=60, seed=7, batch=60)     # batch-independent
    pd.testing.assert_frame_equal(b1, b1b)
    assert not b1.equals(b3)
    assert (s1.lo <= s1.hi).all() and (s1.se >= 0).all()
    # the point estimates are the paired summary (no resampling)
    ps = selbench.paired_summary(selbench.paired_scores_matrix(P, Zs, Gs, Zt, Gt)).iloc[0]
    for k, v in [('share_of_loss_fitness', ps.share_of_loss_fitness), ('mediated_retained', ps.mediated_retained),
                 ('selterm_retained', ps.selterm_retained), ('median_selective_target', ps.sel_target),
                 ('median_raw_source', ps.raw_source), ('delta_raw', ps.delta_raw)]:
        assert abs(s1.loc[k, 'estimate'] - v) < 1e-9, k
    # cell resampling widens the interval relative to a drug-only bootstrap
    s4, _ = selbench.two_way_bootstrap(P, Zs, Gs, Zt, Gt, n_boot=200, seed=7, drug_only=True)
    s5, _ = selbench.two_way_bootstrap(P, Zs, Gs, Zt, Gt, n_boot=200, seed=7)
    assert s5.loc['median_selective_target', 'se'] > s4.loc['median_selective_target', 'se']
    # pooled models and single-screen mode
    s6, b6 = selbench.two_way_bootstrap({'a': P, 'b': 2 * P + 1}, Zs, Gs, n_boot=20, seed=1)
    assert b6.shape == (20, len(s6)) and 'median_selective_source' in s6.index
    assert 'share_of_loss_fitness' not in s6.index and s6.attrs['models'] == 2
    # weighted median equals the median of the expanded multiset
    rng = np.random.default_rng(0)
    v, c = rng.normal(size=(9, 30)), rng.integers(0, 4, size=(9, 30))
    c[0] += 1
    ref = [np.median(np.repeat(v[:, b], c[:, b])) for b in range(30)]
    assert np.allclose(selbench.weighted_median(v, c), ref)


def test_reliability_shrinkage_and_minimum():
    rel = pd.Series({'a': 0.0, 'b': 0.02, 'c': 0.3, 'd': 0.5, 'e': 0.6})
    assert np.allclose(selbench.shrink_reliability(rel, weight=0), rel)
    assert np.allclose(selbench.shrink_reliability(rel, weight=1), rel.median())
    assert np.allclose(selbench.shrink_reliability(rel, weight=0.5), (rel + rel.median()) / 2)
    assert np.allclose(selbench.shrink_reliability(rel), (rel + rel.median()) / 2)    # DEFAULT_SHRINKAGE = 0.5
    # empirical Bayes: the noisier a reliability, the more it is shrunk
    se = pd.Series({'a': 0.3, 'b': 0.3, 'c': 0.05, 'd': 0.05, 'e': 0.05})
    eb = selbench.shrink_reliability(rel, se=se)
    lam = (eb - rel).abs() / (eb - rel).abs().max()
    assert lam[['a', 'b']].min() > lam[['c', 'd', 'e']].max()
    assert (eb >= 0).all() and (eb <= 1).all()
    big = selbench.shrink_reliability(rel, se=pd.Series(1.0, index=rel.index))      # noise only: full pooling
    assert np.allclose(big, big.iloc[0])
    # normalisation: warning below the floor, minimum reliability, bounded inflation
    scores = pd.Series(0.2, index=rel.index)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        N = selbench.normalize_by_reliability(scores, rel, weight=0)
    assert any(issubclass(x.category, selbench.ReliabilityWarning) for x in w)
    assert N.loc[['a', 'b'], 'normalized'].isna().all()
    assert np.isclose(N.loc['c', 'normalized'], 0.2 / np.sqrt(0.3))
    assert (N.normalized.dropna() <= 0.2 / np.sqrt(selbench.MIN_RELIABILITY) + 1e-12).all()
    N2 = selbench.normalize_by_reliability(scores, rel, weight=0.5, warn_below=None)
    assert N2.normalized.notna().all() and np.isclose(N2.loc['a', 'reliability_shrunk'], 0.15)
    assert N2.normalized.max() < N.normalized.max() * 1.5
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        selbench.normalize_by_reliability(scores, rel.clip(lower=0.2))
    assert not any(issubclass(x.category, selbench.ReliabilityWarning) for x in w)
    # evaluate(): legacy column unchanged but warned about; opt-in shrunken column
    obs, fitness, selectivity, drugs, cells = synthetic(seed=1, n_drugs=30, n_cells=100)
    pred = pd.DataFrame([(d, c, selectivity[i, j]) for i, d in enumerate(drugs) for j, c in enumerate(cells)],
                        columns=['drug', 'cell', 'prediction'])
    relmap = {d: (0.001 if i < 3 else 0.5) for i, d in enumerate(drugs)}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        T, s = selbench.evaluate(pred, obs, reliability=relmap, shrinkage=0.5)
    assert any(issubclass(x.category, selbench.ReliabilityWarning) for x in w)
    leg = T.selective / np.sqrt(T.drug.map(relmap))
    assert np.allclose(T.selective_ceiling_normalized, leg)
    assert T.selective_ceiling_normalized.max() > 10 * T.selective.max()      # the inflation R3 reported
    assert T.selective_reliability_normalized.max() <= T.selective.max() / np.sqrt(0.1) + 1e-12


def test_decision_uses_p_less_than_alpha_when_p_underflows():
    rng = np.random.default_rng(9)
    a = pd.Series(rng.normal(1.0, 0.1, 20000))
    b = pd.Series(np.zeros(20000))
    res = selbench.paired_comparison(a, b, alternative='greater')
    p = res['p']
    assert res['drugs'] == 20000 and res['frac_better'] == 1.0
    assert p == 0.0                          # underflow: the exact P is far below the smallest double
    assert selbench.significant(p)           # p < alpha
    assert not ((p or 1) < 0.05)             # the idiom behind the H4 flag error calls it non-significant
    assert selbench.significant(np.float64(0.0)) and selbench.significant(0.0499)
    assert not selbench.significant(0.05) and not selbench.significant(None) and not selbench.significant(np.nan)
    assert not selbench.significant(float('inf'))


def test_fitness_axis_variants_and_axis_diagnostics():
    obs, fitness, selectivity, drugs, cells = synthetic(seed=12, n_drugs=40, n_cells=120)
    Z, G, R, r2 = selbench.decompose(obs)
    groups = {d: f'k{i % 4}' for i, d in enumerate(drugs)}
    Gc = selbench.fitness_axis(Z, groups=groups)
    d = drugs[0]
    others = [x for x in drugs if groups[x] != groups[d]]
    assert np.allclose(Gc.loc[d], Z.loc[others].mean(axis=0))
    panel = drugs[:10]
    Gp = selbench.fitness_axis(Z, panel=panel)
    assert np.allclose(Gp.loc[drugs[20]], Z.loc[panel].mean(axis=0))
    assert np.allclose(Gp.loc[drugs[0]], Z.loc[panel[1:]].mean(axis=0))
    Zm = Z.copy()
    Zm.iloc[5:, 0] = np.nan                                                # cell line c0: only 5 drugs measured
    Gm = selbench.fitness_axis(Zm, min_drugs=10)
    assert Gm.iloc[:, 0].isna().all() and Gm.iloc[:, 1:].notna().all().all()
    assert selbench.axis_reliability(Z, n_splits=20) > 0.9
    assert abs(selbench.axis_agreement(obs, obs)['r'] - 1) < 1e-12
    share = selbench.oracle_fitness_share(Z, G)
    assert abs(share - r2.mean()) < 1e-9


def test_baselines():
    obs, fitness, selectivity, drugs, cells = synthetic(seed=11, n_drugs=40, n_cells=200)
    rng = np.random.default_rng(11)
    feats = pd.DataFrame(np.c_[fitness + 0.3 * rng.normal(size=200), rng.normal(size=(200, 20))], index=cells)
    train, test = cells[:150], cells[150:]
    tr_obs = obs[obs.cell.isin(train)]
    B = selbench.fitness_baseline(tr_obs, feats)
    assert set(B.cell) == set(test) and B.drug.nunique() == 40
    M = selbench.as_matrix(B)
    assert np.allclose(M.std(axis=0), 0)                                   # drug-agnostic
    assert np.corrcoef(M.iloc[0].reindex(test), fitness[150:])[0, 1] > 0.8
    tissue = pd.Series(rng.choice(list('ABCDE'), 200), index=cells)
    TD = selbench.tissue_drug_mean_baseline(tr_obs, tissue, test)
    assert len(TD) == 40 * 50 and TD.prediction.notna().all()
    c0 = test[0]
    ref = tr_obs[(tr_obs.drug == 'd3') & tr_obs.cell.map(tissue).eq(tissue[c0])].response.mean()
    assert np.isclose(TD[(TD.drug == 'd3') & (TD.cell == c0)].prediction.iloc[0], ref)
    TM = selbench.tissue_mean_baseline(tr_obs, tissue, test)
    assert list(TM.index) == list(test) and TM.nunique() <= 5


if __name__ == '__main__':
    import sys
    tests = [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print('ok', name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print('FAILED', name, type(e).__name__, e)
    print(f'{len(tests) - failed} passed, {failed} failed')
    sys.exit(1 if failed else 0)
