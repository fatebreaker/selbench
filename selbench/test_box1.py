"""Tests for selbench.box1 on synthetic screens with a known axis, known selectivity and known tissue effects."""
import numpy as np
import pandas as pd

import box1
import selbench


def screen(seed=0, n_drugs=60, n_cells=200, tissue_effect=0.0, n_tissues=5):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=n_cells)
    tissue = rng.integers(0, n_tissues, n_cells)
    teff = rng.normal(size=(n_drugs, n_tissues)) * tissue_effect
    sel = rng.normal(size=(n_drugs, n_cells))
    load = rng.uniform(0.5, 1.5, n_drugs)
    resp = load[:, None] * axis[None] + sel + teff[:, tissue] + rng.normal(scale=0.5, size=(n_drugs, n_cells))
    drugs, cells = [f'd{i}' for i in range(n_drugs)], [f'c{j}' for j in range(n_cells)]
    obs = pd.DataFrame([(d, c, resp[i, j]) for i, d in enumerate(drugs) for j, c in enumerate(cells)],
                       columns=['drug', 'cell', 'response'])
    Z, G, R, _ = selbench.decompose(obs)
    S = pd.DataFrame(sel, index=drugs, columns=cells).reindex(index=Z.index, columns=Z.columns)
    return Z, G, R, S, pd.Series(tissue.astype(str), index=cells).reindex(Z.columns), teff


def per_drug_terms(P, Z, G):
    rows = []
    for d in Z.index:
        p, z, g = P.loc[d].values, Z.loc[d].values, G.loc[d].values
        rows.append((np.corrcoef(p, z)[0, 1], np.corrcoef(p, g)[0, 1], np.corrcoef(z, g)[0, 1],
                     selbench.partial_corr(p, z, [g])))
    return [np.array(x) for x in zip(*rows)]


def test_capture_perfect_and_agnostic_predictors():
    Z, G, _, _, _, _ = screen()
    raw, rpg, rzg, sel = per_drug_terms(Z, Z, G)                  # perfect predictor p = z
    c = box1.capture(raw, rpg, rzg, sel)
    assert abs(c['axis_capture'] - 1) < 1e-12 and abs(c['selective_capture'] - 1) < 1e-12
    assert abs(c['share_ratio'] - 1) < 1e-12 and abs(c['perfect_share'] - np.mean(rzg ** 2)) < 1e-12
    raw, rpg, rzg, sel = per_drug_terms(G, Z, G)                  # drug-agnostic predictor p = g
    c = box1.capture(raw, rpg, rzg)
    assert abs(c['selective_capture']) < 1e-12                    # no selective term at all
    assert c['share_ratio'] <= c['bound'] + 1e-12
    # bootstrap-shaped input: row 0 equals the point estimate; weights act as drug multiplicities
    W = np.vstack([np.ones(len(raw)), np.r_[np.full(len(raw) // 2, 2.0), np.zeros(len(raw) - len(raw) // 2)]])
    cb = box1.capture(np.vstack([raw, raw]), np.vstack([rpg, rpg]), np.vstack([rzg, rzg]), weights=W)
    h = len(raw) // 2
    c2 = box1.capture(raw[:h], rpg[:h], rzg[:h])
    assert abs(cb['axis_capture'][0] - c['axis_capture']) < 1e-12
    assert abs(cb['axis_capture'][1] - c2['axis_capture']) < 1e-12


def test_corrected_reference_and_capture():
    Z, G, _, _, _, _ = screen(1)
    raw, rpg, rzg, _ = per_drug_terms(Z, Z, G)
    assert abs(box1.corrected_reference(rzg, 1.0) - np.mean(rzg ** 2)) < 1e-12     # R_sel = 1: no correction
    ref = box1.corrected_reference(rzg, 0.4)
    assert ref > np.mean(rzg ** 2)                                                   # noisier selective part
    assert box1.corrected_reference(rzg, 0.4, kappa=0.55) < ref
    cc = box1.corrected_capture(raw, rpg, rzg, 1.0)
    assert abs(cc['axis_capture_tau'] - 1) < 1e-12 and abs(cc['selective_capture_tau'] - 1) < 1e-12
    draws = box1.corrected_capture(raw, rpg, rzg, np.array([0.5, 0.6, 0.7]))
    assert draws['corrected_reference'].shape == (3,)


def test_selective_both_symmetric_and_identity():
    Z, G, _, S, _, _ = screen(2)
    rng = np.random.default_rng(3)
    d = Z.index[0]
    p = S.loc[d].values + rng.normal(size=Z.shape[1])
    gt = G.loc[d].values + rng.normal(scale=0.5, size=Z.shape[1])
    zt = Z.loc[d].values + rng.normal(scale=0.7, size=Z.shape[1])
    r = box1.selective_both(p, Z.loc[d].values, G.loc[d].values, zt, gt)
    assert abs(r['within_sym'] - selbench.partial_corr(p, Z.loc[d].values, [G.loc[d].values, gt])) < 1e-12
    assert abs(r['within_asym'] - selbench.partial_corr(p, Z.loc[d].values, [G.loc[d].values])) < 1e-12
    same = box1.selective_both(p, Z.loc[d].values, G.loc[d].values, Z.loc[d].values, G.loc[d].values)
    assert abs(same['change_sym']) < 1e-12 and abs(same['within_sym'] - same['within_asym']) < 1e-9
    pc = box1.pooled_change(np.array([0.2, 0.4]), np.array([0.1, 0.3]))
    assert abs(pc['change'] + 0.1) < 1e-12 and abs(pc['retained'] - 4 / 6) < 1e-12


def test_attenuation_proportional_is_null():
    rng = np.random.default_rng(4)
    rpg, rzg = rng.uniform(0.3, 0.9, 50), rng.uniform(0.2, 0.8, 50)
    sel_term = rng.uniform(0.0, 0.3, 50)
    raw = rpg * rzg + sel_term
    k = 0.6                                         # both terms scaled by one factor: proportional attenuation
    a = box1.attenuation(raw, rpg, rzg, k * raw, rpg, k * rzg)
    assert abs(a['excess']) < 1e-12 and abs(a['retention_difference']) < 1e-12
    assert abs(a['axis_retained'] - k) < 1e-12 and abs(a['raw_loss'] - (1 - k) * raw.mean()) < 1e-12
    b = box1.attenuation(raw, rpg, rzg, raw - rpg * rzg * 0.5, rpg, rzg * 0.5)   # only the axis term lost
    assert b['excess'] > 0 and abs(b['selterm_retained'] - 1) < 1e-12


def test_mean_viability_analytic_matches_trapezoid_and_limits():
    rng = np.random.default_rng(5)
    n = 200
    pec, s = rng.uniform(4, 9, n), rng.uniform(0.3, 3, n)
    f, b = rng.uniform(0.8, 1.1, n), rng.uniform(0, 0.6, n)
    lo = rng.uniform(-9, -7, n)
    hi = lo + rng.uniform(1, 4, n)
    a = box1.mean_viability(pec, s, f, b, lo, hi)
    t = box1.mean_viability(pec, s, f, b, lo, hi, method='trapezoid', n_grid=20001)
    assert np.max(np.abs(a - t)) < 1e-7
    flat = box1.mean_viability(pec, s, b, b, lo, hi)                              # front = back: constant curve
    assert np.max(np.abs(flat - b)) < 1e-12
    far = box1.mean_viability(np.zeros(n), np.full(n, 3.0), f, b, lo, hi)        # EC50 (1 M) far above the window
    assert np.max(np.abs(far - f)) < 1e-6


def test_common_window_and_harmonize_pair():
    rng = np.random.default_rng(6)
    rows = []
    for ds, shift in [('a', 0.0), ('b', 0.7)]:
        for d in range(6):
            for c in range(40):
                lo = -9 + shift + (1.5 if (ds == 'b' and d == 5) else 0)
                hi = lo + (2.0 if d == 5 else 3.5)
                rows.append({'ds': ds, 'drug': f'd{d}', 'cell': f'c{c}', 'pec50': rng.uniform(5, 8),
                             'slope': rng.uniform(0.5, 2), 'front': 1.0, 'back': rng.uniform(0, 0.5), 'lo': lo, 'hi': hi})
    cur = pd.DataFrame(rows)
    A, B = cur[cur.ds == 'a'].drop(columns='ds'), cur[cur.ds == 'b'].drop(columns='ds')
    w = box1.common_window(box1.tested_ranges(A), box1.tested_ranges(B), 1.0)
    assert abs(w.loc['d0', 'lo'] + 8.3) < 1e-12 and abs(w.loc['d0', 'hi'] + 5.5) < 1e-12
    assert not w.loc['d5', 'kept']                                                  # window 0.5 log10 units
    a, b, _ = box1.harmonize_pair(A, B, 'common_range')
    assert set(a.drug) == set(b.drug) == {f'd{i}' for i in range(5)}
    a2, b2, _ = box1.harmonize_pair(A, B, 'shared_panel')
    assert set(a2.drug) == {f'd{i}' for i in range(5)}
    x = A[A.drug == 'd0'].iloc[0]
    v = box1.mean_viability(x.pec50, x.slope, x.front, x.back, -8.3, -5.5)
    assert abs(-a[(a.drug == 'd0') & (a.cell == x.cell)].response.iloc[0] - v) < 1e-12
    o1, o2 = box1.shared_panel(a.assign(drug=a.drug.replace('d1', 'x')), b)
    assert set(o1.drug) == set(o2.drug) and 'd1' not in set(o1.drug)


def test_retained_fraction_test_and_bootstrap():
    sw = np.r_[1.0, np.full(999, 1.0)]
    sc = np.r_[0.9, np.linspace(0.85, 0.95, 999)]
    t = box1.retained_fraction_test(sw, sc, margin=0.8)
    assert t['noninferior'] and t['p_boot'] == 0 and abs(t['retained_fraction'] - 0.9) < 1e-12
    t2 = box1.retained_fraction_test(sw, np.r_[0.9, np.linspace(0.5, 1.2, 999)], margin=0.8)
    assert not t2['noninferior'] and 0 < t2['p_boot'] < 1
    Z, G, _, S, _, _ = screen(7, n_drugs=20, n_cells=120)
    rng = np.random.default_rng(8)
    Zt = Z + rng.normal(scale=0.3, size=Z.shape)
    P = S + rng.normal(size=S.shape)
    res, per = box1.retained_fraction_bootstrap(P, Z, G, Zt, G, n_boot=200, seed=1)
    d = Z.index[0]
    ref = box1.selective_both(P.loc[d].values, Z.loc[d].values, G.loc[d].values, Zt.loc[d].values, G.loc[d].values)
    assert abs(per.set_index('drug').loc[d, 'within_sym'] - ref['within_sym']) < 1e-9
    assert abs(per.set_index('drug').loc[d, 'cross'] - ref['cross']) < 1e-9
    assert 0.3 < res['retained_fraction'] <= 1.2


def test_holm_matches_definition():
    p = np.array([0.01, 0.04, np.nan, 0.03, 0.005])
    h = box1.holm(p)
    assert np.isnan(h[2])
    assert np.allclose(h[[4, 0, 3, 1]], [0.02, 0.03, 0.06, 0.06])
    assert np.all(box1.holm(np.array([0.5, 0.9])) <= 1)


def test_biomarker_scores_selective_feature():
    Z, G, R, S, _, _ = screen(9)
    d = Z.index[3]
    feat = pd.Series(S.loc[d].values, index=Z.columns)        # the drug's own selective signal as a biomarker
    out = box1.biomarker_scores(feat, Z, G, drugs=[d, Z.index[4]]).set_index('drug')
    assert out.loc[d, 'selective'] > 0.6 and abs(out.loc[Z.index[4], 'selective']) < 0.3
    axis_feat = pd.Series(G.loc[d].values, index=Z.columns)
    assert abs(box1.biomarker_scores(axis_feat, Z, G, drugs=[d]).selective.iloc[0]) < 0.05


def test_lineage_adjusted_scores():
    Z, G, _, S, tis, _ = screen(10, tissue_effect=1.0)
    # a tissue-by-drug mean fitted on all lines lies in the span of the tissue indicators: undefined lineage score
    tdm = Z.T.groupby(tis.reindex(Z.columns).values).transform('mean').T
    L = box1.lineage_adjusted_scores(tdm, Z, G, tis)
    assert L.selective_lineage.isna().all() and (L.selective > 0.2).mean() > 0.8
    # the selective signal itself keeps a lineage-adjusted score; the covariate split is exact
    L2 = box1.lineage_adjusted_scores(S, Z, G, tis).set_index('drug')
    assert (L2.selective_lineage > 0.3).mean() > 0.9
    d = Z.index[0]
    r = L2.loc[d]
    t = box1.tissue_indicators(tis.values)
    assert abs(r.selective_lineage - selbench.partial_corr(S.loc[d].values, Z.loc[d].values,
                                                           [G.loc[d].values] + list(t.T))) < 1e-9
    assert abs(r.raw - np.corrcoef(S.loc[d].values, Z.loc[d].values)[0, 1]) < 1e-12
    cap = box1.lineage_capture(L2.raw.values, L2.covariate_term.values, L2.r2_z_covariates.values)
    assert 0 < cap['perfect_covariate_share'] < 1 and cap['selective_capture_lineage'] > 0
    zz = box1.lineage_adjusted_scores(Z, Z, G, tis)                       # perfect predictor
    capz = box1.lineage_capture(zz.raw.values, zz.covariate_term.values, zz.r2_z_covariates.values)
    assert abs(capz['selective_capture_lineage'] - 1) < 1e-9


def test_active_drugs_axis_and_split_half():
    rng = np.random.default_rng(11)
    n_d, n_c = 40, 150
    auc = pd.DataFrame(rng.uniform(0.85, 1.0, (n_d, n_c)), index=[f'd{i}' for i in range(n_d)],
                       columns=[f'c{j}' for j in range(n_c)])
    auc.iloc[:15, :30] = 0.5                                           # 15 drugs active in 20% of lines
    auc.iloc[15, :10] = 0.5                                            # 6.7%: inactive
    act = box1.active_drugs(auc)
    assert list(act) == [f'd{i}' for i in range(15)]
    Z, G, _, _, _, _ = screen(12, n_drugs=n_d, n_cells=n_c)
    Z.index = auc.index
    ax, Ga = box1.active_axis(Z, act, min_drugs=10)
    assert np.allclose(ax.values, Z.loc[act].mean(axis=0).values)
    d = act[0]
    assert np.allclose(Ga.loc[d].values, Z.loc[act.drop(d)].mean(axis=0).values)
    med, lo, hi = box1.axis_split_half(Z, act, n_splits=50)
    assert 0.8 < med <= 1 and lo <= med <= hi


# ---------------------------------------------------------------------------------------------------- round 3
def test_censored_rank_normal_ties_at_resistant_end():
    x = np.array([0.3, np.nan, 1.2, -0.5, np.nan, 2.0])
    r = box1.censored_rank_normal(x)                                     # NaN taken as censored
    assert np.isfinite(r).all()
    assert r[1] == r[4] and r[1] < np.nanmin(r[[0, 2, 3, 5]])            # tied, below every observed value
    # ties averaged: the two censored entries share ranks 1 and 2 of 6
    from scipy.stats import norm
    assert abs(r[1] - norm.ppf((1.5 - 0.5) / 6)) < 1e-12
    # the observed entries keep their order and their ranks 3..6
    assert np.allclose(r[[3, 0, 2, 5]], norm.ppf((np.arange(3, 7) - 0.5) / 6))
    # explicit flags: a value beyond the tested range is censored too; NaN not flagged = not screened (stays NaN)
    c = np.array([False, True, False, True, False, False])
    r2 = box1.censored_rank_normal(x, c)
    assert np.isnan(r2[4]) and r2[1] == r2[3] and r2[1] < min(r2[0], r2[2], r2[5])
    assert np.isnan(box1.censored_rank_normal([np.nan, np.nan])).all()


def test_censor_resistant_screened_pairs_and_limit():
    rng = np.random.default_rng(21)
    drugs, cells = [f'd{i}' for i in range(12)], [f'c{j}' for j in range(80)]
    full = pd.DataFrame([(d, c, rng.normal()) for d in drugs for c in cells], columns=['drug', 'cell', 'response'])
    # undefined (resistant) values: absent from the observed table, or NaN in it
    resistant = full.response < -0.6
    obs = full[~resistant | (rng.uniform(size=len(full)) < 0.3)].copy()
    obs.loc[obs.response < -0.6, 'response'] = np.nan
    obs = pd.concat([obs, pd.DataFrame({'drug': ['d0'], 'cell': ['not_screened'], 'response': [3.0]})])
    out = box1.censor_resistant(obs, screened=full[['drug', 'cell']])
    assert len(out) == len(full) and 'not_screened' not in set(out.cell)       # restricted to screened pairs
    assert out.censored.sum() == resistant.sum()
    for d, g in out.groupby('drug'):
        cz = g[g.censored].response
        assert cz.nunique() == 1 and cz.iloc[0] < g[~g.censored].response.min()
    Z, G, R, r2 = selbench.decompose(out)
    zc = Z.values[out.pivot_table(index='drug', columns='cell', values='censored').reindex(
        index=Z.index, columns=Z.columns).values.astype(bool)]
    assert np.isfinite(Z.values).all()                                  # every screened pair ranked
    # equals 98_endpoint_cell_selection's construction (left merge, lowest defined value - 1)
    t = full[['drug', 'cell']].merge(obs, on=['drug', 'cell'], how='left')
    t['response'] = t.response.fillna(t.groupby('drug').response.transform('min') - 1.0)
    Z98, _, _, r298 = selbench.decompose(t)
    assert np.allclose(Z98.values, Z.values) and np.allclose(r298.values, r2.values)
    # resistant_limit: values at or below the limit are censored as well (e.g. IC50 beyond the highest dose)
    lim = pd.Series(-0.2, index=drugs)
    out2 = box1.censor_resistant(obs, screened=full[['drug', 'cell']], resistant_limit=lim)
    assert out2.censored.sum() == (full.response <= -0.2).sum()
    out3 = box1.censor_resistant(obs.assign(flag=obs.response.fillna(0) > 1.5), censored='flag')
    assert out3.censored.sum() == ((obs.response > 1.5) | obs.response.isna()).sum()
    assert len(zc) == resistant.sum()


def test_activity_rules_endpoint_defaults():
    rng = np.random.default_rng(31)
    n_d, n_c = 30, 100
    idx, cols = [f'd{i}' for i in range(n_d)], [f'c{j}' for j in range(n_c)]
    auc = pd.DataFrame(rng.uniform(0.85, 1.0, (n_d, n_c)), index=idx, columns=cols)
    auc.iloc[:10, :20] = 0.5                                           # active (20% of lines)
    auc.iloc[10, :5] = 0.5                                             # 5%: inactive
    paper = box1.active_drugs(auc)
    assert list(box1.active_drugs_rule(auc)) == list(paper) == idx[:10]          # the paper's rule = AUC default
    assert list(box1.active_drugs_rule(1 - auc, 'aac')) == idx[:10]             # AAC = 1 - AUC, > 0.2
    assert list(box1.active_drugs_rule(-auc, 'generic', threshold=-0.8, direction='above')) == idx[:10]
    gr = (1 - auc) * 1.0
    assert list(box1.active_drugs_rule(gr, 'gr_aoc')) == idx[:10]
    # ln IC50: a line responds when its IC50 lies below the highest tested concentration; undefined IC50s (NaN) on
    # screened pairs count as non-responding
    ln_max = pd.Series(np.log(10.0), index=idx)                        # 10 uM top dose, IC50 in uM
    ic = pd.DataFrame(np.nan, index=idx, columns=cols)
    ic.iloc[:10, :20] = np.log(1.0)                                    # 1 uM: responds
    ic.iloc[10:20, :40] = np.log(50.0)                                 # beyond the tested range: does not respond
    ic.iloc[20, :5] = np.log(2.0)                                      # 5% responding
    screened = pd.DataFrame(True, index=idx, columns=cols)
    act = box1.active_drugs_rule(ic, 'ln_ic50', max_tested=ln_max, screened=screened)
    assert list(act) == idx[:10]
    f = box1.activity_fraction(ic, 'ln_ic50', max_tested=ln_max, screened=screened)
    assert abs(f['d0'] - 0.2) < 1e-12 and abs(f['d20'] - 0.05) < 1e-12 and f['d15'] == 0
    # without `screened` the undefined IC50s leave the denominator and activity is overstated (documented)
    f_naive = box1.activity_fraction(ic, 'ln_ic50', max_tested=ln_max)
    assert f_naive['d20'] == 1.0
    # per-pair maximum concentrations
    mx = pd.DataFrame(np.log(10.0), index=idx, columns=cols)
    mx.iloc[0, :20] = np.log(0.5)                                      # d0's lines tested only to 0.5 uM
    assert 'd0' not in box1.active_drugs_rule(ic, 'ln_ic50', max_tested=mx, screened=screened)
    for bad in [dict(endpoint='generic', threshold=0.1), dict(endpoint='ln_ic50')]:
        try:
            box1.activity_fraction(auc, **bad)
            raise AssertionError('expected ValueError')
        except ValueError:
            pass
    assert 'auc' in box1.PAPER_ACTIVITY_RULE.lower() and set(box1.ACTIVITY_RULES) >= {'auc', 'aac', 'gr_aoc', 'ln_ic50'}


def test_rank_intervals_orders_and_resolves():
    rng = np.random.default_rng(5)
    true = np.array([0.30, 0.29, 0.10, 0.05])
    S = np.vstack([true, true + rng.normal(scale=0.01, size=(400, 4))])
    T = box1.rank_intervals(S, ['a', 'b', 'c', 'd'])
    assert list(T['rank']) == [1.0, 2.0, 3.0, 4.0]
    a, c = T.set_index('model').loc['a'], T.set_index('model').loc['c']
    assert a.rank_lo == 1 and a.rank_hi == 2          # a and b overlap
    assert c.rank_lo == c.rank_hi == 3                 # c is separated from every other model
    assert a.resolved_vs_others == 2 and c.resolved_vs_others == 3


def test_triad_reliability_and_expected_retention():
    # shared signal t, dataset noise u_X with reliabilities R = 1 / (1 + s_X^2)
    rng = np.random.default_rng(9)
    n, sd = 4000, {'f': 0.6, 'a': 1.0, 'b': 0.8, 'c': 1.2}
    drug, rfa, rfb, rab, nn = [], [], [], [], []
    for d in range(30):
        t = rng.normal(size=n)
        z = {k: t + rng.normal(scale=s, size=n) for k, s in sd.items()}
        for a, b in [('a', 'b'), ('a', 'c'), ('b', 'c')]:
            drug.append(d); nn.append(n)
            rfa.append(np.corrcoef(z['f'], z[a])[0, 1]); rfb.append(np.corrcoef(z['f'], z[b])[0, 1])
            rab.append(np.corrcoef(z[a], z[b])[0, 1])
    R_f, nd = box1.triad_reliability(drug, rfa, rfb, rab, nn)
    assert nd == 30 and abs(R_f - 1 / (1 + 0.6 ** 2)) < 0.02
    # bias correction: psi_overlap is n x the sampling covariance of two correlations sharing a variable
    assert abs(box1.psi_overlap(0.0, 0.0, 0.5) - 0.5) < 1e-12
    # expected retention: sqrt(R_T / R_S); scalars and matched draws
    assert abs(box1.expected_retention(0.5, 0.32) - 0.8) < 1e-12
    assert np.allclose(box1.expected_retention([0.5, 0.4], [0.5, 0.1]), [1.0, 0.5])


def test_attenuation_reliability_adjustment():
    rng = np.random.default_rng(3)
    D = 40
    rzw = rng.uniform(0.3, 0.6, D); pgw = rng.uniform(0.5, 0.9, D); raw_w = pgw * rzw + 0.1
    rzc = rzw * 0.9; pgc = pgw; raw_c = pgc * rzc + 0.08
    a0 = box1.attenuation(raw_w, pgw, rzw, raw_c, pgc, rzc)
    a1 = box1.attenuation(raw_w, pgw, rzw, raw_c, pgc, rzc, rho_within=1.0, rho_cross=1.0)
    assert abs(a0['excess'] - a1['excess']) < 1e-12               # reliability 1: unchanged
    a2 = box1.attenuation(raw_w, pgw, rzw, raw_c, pgc, rzc, rho_within=0.98, rho_cross=0.9)
    a3 = box1.attenuation(raw_w, pgw, rzw / 0.98, raw_c, pgc, rzc / 0.9)
    assert abs(a2['excess'] - a3['excess']) < 1e-12               # same as dividing the loadings by hand
    # a noisier target (lower axis reliability) inflates the unadjusted excess; the adjustment removes it
    assert a2['excess'] < a0['excess']
    # bootstrap draws: per-draw reliabilities broadcast over drugs
    R = np.vstack([raw_w, raw_w * 1.01]); P = np.vstack([pgw, pgw]); Z = np.vstack([rzw, rzw])
    Rc = np.vstack([raw_c, raw_c]); Pc = np.vstack([pgc, pgc]); Zc = np.vstack([rzc, rzc])
    a4 = box1.attenuation(R, P, Z, Rc, Pc, Zc, rho_within=np.array([0.98, 0.97]), rho_cross=np.array([0.9, 0.91]))
    assert a4['excess'].shape == (2,)


def _classical_terms(rng, D=40, rho=0.8, kind='balanced'):
    """Per-drug latent correlations and their observed versions under classical axis error (r_xg = sqrt(rho) r_xa)."""
    r_za = rng.uniform(0.3, 0.6, D)
    if kind == 'agnostic':                       # p = a: r_pa = 1, r_pz = r_za (no drug-specific signal)
        r_pa = np.ones(D)
        r_pz = r_za.copy()
    elif kind == 'selective':                    # p independent of a: r_pa = 0
        r_pa = np.zeros(D)
        r_pz = rng.uniform(0.1, 0.4, D)
    else:
        r_pa = rng.uniform(0.4, 0.8, D)
        r_pz = r_pa * r_za + rng.uniform(0.05, 0.2, D)
    lat = box1.capture(r_pz, r_pa, r_za)
    obs = box1.capture(r_pz, np.sqrt(rho) * r_pa, np.sqrt(rho) * r_za)
    return lat, obs


def test_axis_leakage_bound_exact_bound_and_inversion():
    rng = np.random.default_rng(11)
    for rho in (0.6, 0.9, 0.98):
        for kind in ('agnostic', 'balanced', 'selective'):
            lat, obs = _classical_terms(rng, rho=rho, kind=kind)
            A, m, S = obs['axis_capture'], obs['perfect_share'], obs['selective_capture']
            assert abs(A - lat['axis_capture']) < 1e-12 and abs(m - rho * lat['perfect_share']) < 1e-12
            bound = box1.axis_leakage_bound(A, m, rho)
            leak = S - lat['selective_capture']
            # the full two-term leakage, and the exact inversion
            two_term = bound - lat['selective_capture'] * lat['perfect_share'] * (1 - rho) / (1 - m)
            assert abs(leak - two_term) < 1e-12
            assert abs(box1.corrected_selective_capture(S, A, m, rho) - lat['selective_capture']) < 1e-12
            if kind == 'agnostic':                                     # exact for drug-agnostic predictors
                assert abs(lat['selective_capture']) < 1e-12 and abs(leak - bound) < 1e-12 and bound > 0
            elif kind == 'balanced':                                   # an upper bound otherwise
                assert lat['selective_capture'] > 0 and leak < bound
            else:                                                      # purely selective: no axis term, leakage < 0
                assert abs(bound) < 1e-12 and leak < 0
    assert box1.axis_leakage_bound(1.0, 0.2, 1.0) == 0.0               # a perfectly reliable axis leaks nothing
    assert abs(box1.axis_leakage_bound(1.0, 0.2, 0.9) - 0.2 * 0.1 / (0.8 * 0.9)) < 1e-15
    # matched bootstrap draws
    b = box1.axis_leakage_bound(np.array([1.0, 1.1]), np.array([0.2, 0.25]), np.array([0.9, 0.85]))
    assert b.shape == (2,) and np.all(b > 0)
    c = box1.corrected_selective_capture(np.array([0.05, 0.06]), np.array([1.0, 1.1]), np.array([0.2, 0.25]),
                                         np.array([0.9, 0.85]))
    assert c.shape == (2,)


def test_axis_leakage_refuses_pair_and_drug_splits():
    for split in ('pair', 'drug'):
        for f, args in [(box1.axis_leakage_bound, (1.0, 0.2, 0.9)), (box1.corrected_selective_capture, (0.05, 1.0, 0.2, 0.9))]:
            try:
                f(*args, split=split)
            except ValueError as e:
                assert 'matched_baselines' in str(e)
            else:
                raise AssertionError(f'{f.__name__} accepted split={split}')
    try:
        box1.axis_leakage_bound(1.0, 0.2, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError('rho = 0 accepted')
    assert box1.axis_leakage_bound(1.0, 0.2, 0.9, split='leave-cell-line-out') == box1.axis_leakage_bound(1.0, 0.2, 0.9)


def test_axis_leakage_simulated_screen():
    """Drug-agnostic predictor p = a on a simulated leave-cell-line-out panel: its observed S is the axis's leakage,
    which the formula (with the latent reliability) reproduces and the correction removes."""
    rng = np.random.default_rng(5)
    D, N = 12, 6000
    a = rng.normal(size=N)
    load = rng.uniform(0.3, 0.65, D)
    Y = load[:, None] * a[None] + rng.normal(scale=0.65, size=(D, N)) + rng.normal(scale=0.6, size=(D, N))
    Z = pd.DataFrame(np.vstack([selbench.rank_normal(r) for r in Y]))
    G = box1.axis(Z).values
    P = np.tile(a, (D, 1))
    terms = [selbench.split_terms(P[d], Z.values[d], G[d]) for d in range(D)]
    T = pd.DataFrame(terms)
    c = box1.capture(T.raw.values, T.alignment.values, T.loading.values, sel=T.selective.values)
    rho = float(np.mean([np.corrcoef(G[d], a)[0, 1] ** 2 for d in range(D)]))
    bound = box1.axis_leakage_bound(c['axis_capture'], c['perfect_share'], rho)
    assert c['selective_capture'] > 0.05                                   # leakage is visible with 12 drugs
    assert abs(c['selective_capture'] - bound) < 0.15 * bound
    assert abs(box1.corrected_selective_capture(c['selective_capture'], c['axis_capture'], c['perfect_share'], rho)) < 0.02


def test_general_sensitivity_aliases():
    assert box1.axis is selbench.fitness_axis and box1.general_sensitivity_axis is selbench.fitness_axis
    assert box1.axis_baseline is selbench.fitness_baseline
    assert box1.oracle_axis_share is selbench.oracle_fitness_share
    Z, G, _, _, _, _ = screen(7, n_drugs=20, n_cells=80)
    pd.testing.assert_frame_equal(box1.axis(Z), selbench.fitness_axis(Z))
    pd.testing.assert_frame_equal(box1.axis(Z), G)
    s1, s2 = box1.oracle_axis_share(Z, G), selbench.oracle_fitness_share(Z, G)
    assert (s1 == s2) if np.isscalar(s1) else np.allclose(np.asarray(s1, float), np.asarray(s2, float), equal_nan=True)


def _terms_table(drugs, r_pg, r_zg, sel):
    """Per-drug split_terms rows for given alignment, loading and selective correlation."""
    r_pg, r_zg, sel = (np.broadcast_to(np.asarray(x, float), (len(drugs),)) for x in (r_pg, r_zg, sel))
    raw = r_pg * r_zg + np.sqrt((1 - r_pg ** 2) * (1 - r_zg ** 2)) * sel
    return pd.DataFrame({'drug': drugs, 'raw': raw, 'selective': sel, 'alignment': r_pg, 'loading': r_zg})


def test_matched_baselines_compare_like_with_like():
    # z = 0.6 g + 0.8 e (external review's example): p = e and p = 2g + 0.8e both have selective correlation 1, but
    # selective capture 1.25 and 0.46, so a correlation floor cannot be read as a capture floor
    drugs = [f'd{i}' for i in range(5)]
    on_axis = _terms_table(drugs, 2 / np.sqrt(4.64), 0.6, 1.0)
    off_axis = _terms_table(drugs, 0.0, 0.6, 1.0)
    R = box1.matched_baselines(on_axis, {'off_axis': off_axis})
    m, b = R.iloc[0], R.iloc[1]
    assert list(R.predictor) == ['model', 'off_axis'] and (R.drugs == 5).all()
    assert abs(m.selective_capture - (1.84 / np.sqrt(4.64) - 0.6 * 2 / np.sqrt(4.64)) / 0.64) < 1e-12
    assert abs(m.selective_capture - 0.4642) < 1e-4 and abs(b.selective_capture - 1.25) < 1e-12
    assert m.margin_median_selective == 0 and m.margin_selective_capture < -0.78
    # only drugs finite for the model and every baseline are pooled; the strongest baseline sets each margin
    weak = _terms_table(drugs, 0.0, 0.6, 0.2)
    weak.loc[0, 'selective'] = np.nan
    R = box1.matched_baselines(off_axis, {'on_axis': on_axis, 'weak': weak})
    assert (R.drugs == 4).all()
    assert abs(R.loc[0, 'margin_selective_capture'] - (1.25 - 0.4642)) < 1e-4
    assert abs(R.loc[0, 'margin_median_selective']) < 1e-12
    # project-style column names are accepted; long tables are refused
    alt = on_axis.rename(columns={'alignment': 'r_pg', 'loading': 'r_zg'})
    pd.testing.assert_frame_equal(box1.matched_baselines(alt, {'off_axis': off_axis}),
                                  box1.matched_baselines(on_axis, {'off_axis': off_axis}))
    for bad in (pd.concat([on_axis, on_axis]), on_axis.drop(columns='loading')):
        try:
            box1.matched_baselines(bad, {'off_axis': off_axis})
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError')


def test_matched_baselines_with_leakage_floor_output():
    # end to end: a drug-agnostic predictor scored by selbench.leakage_floor, compared on both metrics with a model
    Z, G, _, _, _, _ = screen(3, n_drugs=30, n_cells=120)
    rng = np.random.default_rng(5)
    agn = pd.Series(G.mean(0).values + rng.normal(0, 0.3, Z.shape[1]), index=Z.columns)
    obs = Z.stack().rename('response').reset_index()
    obs.columns = ['drug', 'cell', 'response']
    per, summ = selbench.leakage_floor(obs, agnostic={'agnostic': agn})
    base = per[per.predictor == 'agnostic'].drop(columns='predictor')
    pred = (Z + rng.normal(0, 1.0, Z.shape)).stack().rename('prediction').reset_index()
    pred.columns = ['drug', 'cell', 'prediction']
    model = selbench.selective_scores(pred, obs)
    R = box1.matched_baselines(model, {'agnostic': base})
    assert R.loc[1, 'drugs'] <= len(base) and abs(R.loc[1, 'median_selective'] - summ.loc[0, 'selective']) < 0.05
    assert R.loc[0, 'margin_selective_capture'] > 0 and R.loc[0, 'margin_median_selective'] > 0
    assert abs(R.loc[1, 'selective_capture']) < 0.1


def test_matched_baselines_weights_are_multiplicities():
    # external review round 3: zero-weight drugs must not move any metric (the median ignored the weights before)
    drugs = ['d0', 'd1', 'd2']
    model = _terms_table(drugs, 0.0, 0.6, [0.0, 0.9, 0.9])
    base = _terms_table(drugs, 0.0, 0.6, 0.2)
    w = pd.Series([1.0, 0.0, 0.0], index=drugs)
    R = box1.matched_baselines(model, {'b': base}, weights=w)
    R1 = box1.matched_baselines(model.iloc[:1], {'b': base.iloc[:1]})
    for col in ('selective_capture', 'median_selective', 'mean_selective', 'margin_selective_capture',
                'margin_median_selective', 'margin_mean_selective', 'drugs', 'weight_total'):
        assert abs(R.loc[0, col] - R1.loc[0, col]) < 1e-12, col
    assert R.loc[0, 'median_selective'] == 0.0 and abs(R.loc[0, 'margin_median_selective'] + 0.2) < 1e-12
    # integer multiplicities (one bootstrap draw) give the same metrics as the explicitly repeated rows
    rng = np.random.default_rng(11)
    D = [f'd{i}' for i in range(9)]
    m = _terms_table(D, rng.uniform(-0.3, 0.6, 9), rng.uniform(0.2, 0.7, 9), rng.uniform(-0.2, 0.5, 9))
    b = _terms_table(D, rng.uniform(0.8, 0.95, 9), m.loading.values, rng.uniform(-0.05, 0.1, 9))
    for counts in ([2, 0, 1, 3, 0, 1, 1, 2, 0], [1, 1, 1, 1, 1, 1, 1, 1, 2]):
        R = box1.matched_baselines(m, {'b': b}, weights=pd.Series(counts, index=D, dtype=float))
        rep = np.repeat(np.arange(9), counts)
        for row, T in ((0, m), (1, b)):
            c = box1.capture(T.raw.values[rep], T.alignment.values[rep], T.loading.values[rep])
            assert abs(R.loc[row, 'selective_capture'] - c['selective_capture']) < 1e-12
            assert abs(R.loc[row, 'axis_capture'] - c['axis_capture']) < 1e-12
            assert abs(R.loc[row, 'median_selective'] - np.median(T.selective.values[rep])) < 1e-12
            assert abs(R.loc[row, 'mean_selective'] - np.mean(T.selective.values[rep])) < 1e-12
        assert R.loc[0, 'drugs'] == int(np.count_nonzero(counts)) and R.loc[0, 'weight_total'] == sum(counts)
    # invalid weights are refused
    for bad in ([-1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]):
        try:
            box1.matched_baselines(model, {'b': base}, weights=pd.Series(bad, index=drugs))
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected ValueError for weights {bad}')

