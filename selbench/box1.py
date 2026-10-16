"""selbench.box1: the components of Box 1 that selbench.py (hashed; unchanged) does not implement.

Round-2 revision (editor item 8; R1-M5; R2-Major 4). Each function reproduces the computation of a project script and
is validated against that script's results by scripts/95_validate_box1.py (results/revision_round2/box1_validation/).

  (a) capture             capture, corrected_capture, corrected_reference         (78_capture.py)
  (b) selective-both      selective_both, pooled_change                           (77/80/82, symmetric R3-M3d)
  (c) attenuation         attenuation (optionally reliability-adjusted, 100b)     (81_attenuation_null.py)
  (d) harmonization       mean_viability, tested_ranges, common_window, harmonize_pair, shared_panel
                                                                                  (66_harmonized_range.py, 90)
  (e) equivalence         retained_fraction_test, retained_fraction_bootstrap     (83_equivalence.py)
  (f) multiplicity        holm                                                    (77/86)
  (g) biomarker baseline  biomarker_scores                                        (52_biomarker_baselines.py)
  (h) lineage             lineage_adjusted_scores, lineage_capture                (91_lineage.py)
  (i) active-drug axis    active_drugs, active_axis, axis_split_half              (45_axis_activity_lineage.py)
  Round 3 (editor item 8; R2-M3, R1-m6, R3-m12):
  (j) censoring           censored_rank_normal, censor_resistant                  (98_endpoint_cell_selection.py's
                          tied reference; Box 1: censored or undefined values ranked as tied at the resistant end)
  (k) portable activity   ACTIVITY_RULES, PAPER_ACTIVITY_RULE, activity_fraction, active_drugs_rule
                          (endpoint-specific defaults; the paper's rule is the AUC default)
  Validated by scripts/99d_validate_box1_r3.py (results/revision_round2/box1_validation/validation_r3.csv).
  (l) rank intervals      rank_intervals                                          (88_rank_intervals.py)
  (m) expected retention  psi_overlap, triad_reliability, expected_retention      (84_source_reference.py)
  Validated by scripts/99g_validate_box1_rank_retention.py (box1_validation/validation_r3b.csv).
  Round 4 (editor item 8; R1-M3, R3-M3c, R1-m13):
  (n) axis leakage        axis_leakage_bound, corrected_selective_capture          (100b_known_answer_simulation.py)
                          (classical-error model: exact for drug-agnostic predictors and an upper bound otherwise
                          only under its conditions; invalid for pair-split and drug-blind predictions; prefer the
                          metric-matched empirical baselines of (p))
  (o) general-sensitivity names for selbench's 'fitness' functions (aliases; selbench.py is hashed and unchanged):
                          axis, general_sensitivity_axis (= fitness_axis), axis_baseline (= fitness_baseline),
                          oracle_axis_share (= oracle_fitness_share)
  Validated by scripts/102b_validate_box1_r4.py (box1_validation/validation_r4.csv).
  Round 7 (external review, Box 1 step 4):
  (p) matched baselines   matched_baselines: each metric of a model against the same metric of drug-agnostic
                          predictors on the same drugs and aggregation (selbench.leakage_floor's `floor` is a median
                          selective correlation, not a selective capture); weights are drug multiplicities applied
                          to every metric (weighted median). Tested in test_box1.py.

Conventions as in selbench: per-drug vectors over cell lines; higher = more sensitive; drug x cell matrices are
DataFrames. Functions that summarise over drugs accept arrays of shape (D,) or bootstrap draws of shape (R, D) with
matching drug multiplicities (R, D) (row 0 = the point estimate), as produced by a two-way (cell x drug) bootstrap.
"""
import numpy as np
import pandas as pd

import selbench

LN10 = np.log(10.0)


# ----------------------------------------------------------------------------------------------------------------
# helpers
def _as2d(*arrays):
    out = [np.asarray(a, float) for a in arrays]
    one = out[0].ndim == 1
    return [a[None, :] if a.ndim == 1 else a for a in out], one


def _weights(shape, weights):
    if weights is None:
        return np.ones(shape)
    w = np.asarray(weights, float)
    return np.broadcast_to(w[None, :] if w.ndim == 1 else w, shape)


def _ret(d, one):
    return {k: (float(v[0]) if one and np.ndim(v) else v) for k, v in d.items()}


def _std(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.full_like(x, np.nan)


# ----------------------------------------------------------------------------------------------------------------
# (a) axis and selective capture against a perfect predictor (and a reliability-corrected one)
def capture(raw, r_pg, r_zg, sel=None, weights=None):
    """Capture of the axis-mediated and selective parts of per-drug accuracy, relative to a perfect predictor (p = z,
    whose terms are r_zg^2 and 1 - r_zg^2).

    raw = r(p, z), r_pg = r(p, g), r_zg = r(z, g) per drug (and optionally sel = r(p, z | g)); arrays (D,) or (R, D)
    with drug multiplicities `weights`. Drugs with any non-finite input are dropped. Returns
      axis_capture        sum r_pg r_zg / sum r_zg^2
      selective_capture   sum (r_pz - r_pg r_zg) / sum (1 - r_zg^2)
      axis_share          sum r_pg r_zg / sum r_pz       perfect_share  mean r_zg^2     share_ratio  axis/perfect
      bound               1 / perfect_share (largest share ratio with a non-negative selective term)
      alignment_ratio     mean r_pg / mean r_zg          mean_raw
      cf_share_ratio      (with sel) the share ratio of a predictor aligned like the profile (r_pg = r_zg) that keeps
                          the model's selective correlation: raw_cf = r_zg^2 + (1 - r_zg^2) sel (R3-M1)."""
    (raw, rpg, rzg), one = _as2d(raw, r_pg, r_zg)
    ok = np.isfinite(raw) & np.isfinite(rpg) & np.isfinite(rzg)
    if sel is not None:
        (s_,), _ = _as2d(sel)
        ok &= np.isfinite(s_)
    w = np.where(ok, _weights(raw.shape, weights), 0.0)

    def f(x):
        return (w * np.where(ok, x, 0.0)).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        n = w.sum(1)
        s_med, s_r2, s_raw = f(rpg * rzg), f(rzg ** 2), f(raw)
        m = s_r2 / n
        out = {'axis_capture': s_med / s_r2, 'selective_capture': (s_raw - s_med) / (n - s_r2),
               'axis_share': s_med / s_raw, 'perfect_share': m, 'share_ratio': s_med / s_raw / m, 'bound': 1 / m,
               'alignment_ratio': f(rpg) / f(rzg), 'mean_raw': s_raw / n, 'drugs': n}
        if sel is not None:
            out['cf_share_ratio'] = (s_r2 / f(rzg ** 2 + (1 - rzg ** 2) * s_)) / m
    return _ret(out, one)


def corrected_reference(r_zg, R_sel, kappa=1.0, weights=None, floor=0.02):
    """Perfect-predictor axis share for the reproducible profile tau (error model: the axis is noise-free, drug-
    specific noise is independent of p and g, so rho_d = r_zg^2 + (1 - r_zg^2) R_sel): mean_d r_zg^2 / rho_d.
    With kappa < 1 a fraction 1 - kappa of the axis component is treated as non-reproducible:
    mean kappa r_zg^2 / (kappa r_zg^2 + (1 - r_zg^2) R_sel). R_sel: scalar or (R,) draws (floored at `floor`)."""
    (rzg,), one = _as2d(r_zg)
    Rs = np.clip(np.atleast_1d(np.asarray(R_sel, float)), floor, 1.0)[:, None]
    ok = np.isfinite(rzg)
    w = np.where(ok, _weights(rzg.shape, weights), 0.0)
    rho = kappa * rzg ** 2 + (1 - rzg ** 2) * Rs
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = (w * np.where(ok, kappa * rzg ** 2 / rho, 0.0)).sum(1) / w.sum(1)
    return float(ref[0]) if one and ref.size == 1 else ref


def corrected_capture(raw, r_pg, r_zg, R_sel, weights=None, floor=0.02):
    """Capture relative to a perfect predictor of the reproducible profile (78 corrected_reps, kappa = 1): the model's
    terms are disattenuated by sqrt(rho_d) and compared with r_zg^2 / rho_d (axis) and 1 - r_zg^2 / rho_d
    (selective). Returns corrected_reference, axis_capture_tau, selective_capture_tau, axis_share_tau,
    share_ratio_tau, bound_tau."""
    (raw, rpg, rzg), one = _as2d(raw, r_pg, r_zg)
    Rs = np.clip(np.atleast_1d(np.asarray(R_sel, float)), floor, 1.0)[:, None]
    ok = np.isfinite(raw) & np.isfinite(rpg) & np.isfinite(rzg)
    w = np.where(ok, _weights(raw.shape, weights), 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = rzg ** 2 + (1 - rzg ** 2) * Rs
        z = lambda x: np.where(ok, x, 0.0)  # noqa: E731
        n = w.sum(1)
        s_ref = (w * z(rzg ** 2 / rho)).sum(1)
        inv = z(1 / np.sqrt(rho))
        s_med = (w * z(rpg * rzg) * inv).sum(1)
        s_raw = (w * z(raw) * inv).sum(1)
        share = s_med / s_raw
        ref = s_ref / n
        out = {'corrected_reference': ref, 'axis_capture_tau': s_med / s_ref,
               'selective_capture_tau': (s_raw - s_med) / (n - s_ref), 'axis_share_tau': share,
               'share_ratio_tau': share / ref, 'bound_tau': 1 / ref}
    return _ret(out, one and Rs.shape[0] == 1)


# ----------------------------------------------------------------------------------------------------------------
# (b) symmetric selective-both
def selective_both(p, z_s, g_s, z_t, g_t, min_n=20):
    """Identical predictions p scored on matched cells against a source screen S and a target screen T.

    Returns within_sym r(p, z_S | g_S, g_T), within_asym r(p, z_S | g_S) (the paper's round-1 within value),
    cross r(p, z_T | g_S, g_T), cross_own r(p, z_T | g_T), and the symmetric change cross - within_sym."""
    pc = selbench.partial_corr
    out = {'within_sym': pc(p, z_s, [g_s, g_t], min_n=min_n), 'within_asym': pc(p, z_s, [g_s], min_n=min_n),
           'cross': pc(p, z_t, [g_s, g_t], min_n=min_n), 'cross_own': pc(p, z_t, [g_t], min_n=min_n)}
    out['change_sym'] = out['cross'] - out['within_sym']
    out['change_asym'] = out['cross'] - out['within_asym']
    return out


def pooled_change(within, cross, weights=None):
    """Pooled (mean over drugs, or over model-drug pairs) change and retained fraction of a paired score.
    within, cross: (D,) or (R, D); drugs with either value non-finite are dropped."""
    (wi, cr), one = _as2d(within, cross)
    ok = np.isfinite(wi) & np.isfinite(cr)
    w = np.where(ok, _weights(wi.shape, weights), 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        sw, sc, n = (w * np.where(ok, wi, 0)).sum(1), (w * np.where(ok, cr, 0)).sum(1), w.sum(1)
        out = {'mean_within': sw / n, 'mean_cross': sc / n, 'change': (sc - sw) / n, 'retained': sc / sw}
    return _ret(out, one)


# ----------------------------------------------------------------------------------------------------------------
# (c) proportional attenuation
def attenuation(raw_w, r_pg_w, r_zg_w, raw_c, r_pg_c, r_zg_c, weights=None, rho_within=None, rho_cross=None):
    """Test of a cross-dataset loss against proportional attenuation (R3-M3a; 81). Per drug within (S) and cross (T)
    raw r and axis terms r_pg r_zg; arrays (D,) or (R, D) (pool models by concatenating along the drug axis).
      within_share        w = sum med_w / sum raw_w
      share_of_loss       s = (sum med_w - sum med_c) / (sum raw_w - sum raw_c)
      excess              s - w  (> 0: the axis term lost proportionally more than the selective term)
      axis_retained a = sum med_c / sum med_w; selterm_retained t = sum (raw - med)_c / sum (raw - med)_w;
      retention_difference a - t;  raw_loss = mean raw_w - mean raw_c;  d_axis_term, d_selective_term.
    Under proportional attenuation (both terms scaled by one factor) the excess and the retention difference are 0;
    decide with two-way bootstrap intervals (row 0 = point).
    rho_within, rho_cross: split-half reliabilities of the two datasets' axes (axis_split_half; scalars or (R,) draws).
    When given, each loading r_zg is divided by its axis's reliability before the shares are compared (the
    reliability-adjusted test, 100b_known_answer_simulation.py): measurement noise in an axis moves part of the axis
    term into the selective term, and without the adjustment a target that is only noisier rejects proportional
    attenuation (about 90% of simulated replicates; adjusted, none). Not valid for pair-split predictions, which share
    the evaluation axis's measurement noise."""
    if rho_within is not None:
        r_zg_w = np.asarray(r_zg_w, float) / (np.asarray(rho_within, float)[..., None] if np.ndim(rho_within) else rho_within)
    if rho_cross is not None:
        r_zg_c = np.asarray(r_zg_c, float) / (np.asarray(rho_cross, float)[..., None] if np.ndim(rho_cross) else rho_cross)
    (rw, pw, zw, rc, pcc, zc), one = _as2d(raw_w, r_pg_w, r_zg_w, raw_c, r_pg_c, r_zg_c)
    mw, mc = pw * zw, pcc * zc
    ok = np.isfinite(rw) & np.isfinite(rc) & np.isfinite(mw) & np.isfinite(mc)
    w = np.where(ok, _weights(rw.shape, weights), 0.0)

    def f(x):
        return (w * np.where(ok, x, 0.0)).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        n = w.sum(1)
        Rw, Rc, Mw, Mc = f(rw), f(rc), f(mw), f(mc)
        Tw, Tc = Rw - Mw, Rc - Mc
        s = np.where(np.abs(Rw - Rc) > 1e-9, (Mw - Mc) / (Rw - Rc), np.nan)
        out = {'raw_mean_within': Rw / n, 'raw_mean_cross': Rc / n, 'raw_loss': (Rw - Rc) / n,
               'within_share': Mw / Rw, 'share_of_loss': s, 'excess': s - Mw / Rw, 'axis_retained': Mc / Mw,
               'selterm_retained': Tc / Tw, 'retention_difference': Mc / Mw - Tc / Tw, 'd_axis_term': (Mc - Mw) / n,
               'd_selective_term': (Tc - Tw) / n}
    return _ret(out, one)


def interval(draws, level=95):
    """Percentile interval of bootstrap draws (row 0 = point estimate, excluded)."""
    b = np.asarray(draws, float)[1:]
    b = b[np.isfinite(b)]
    if len(b) < 10:
        return np.nan, np.nan
    a = (100 - level) / 2
    return float(np.percentile(b, a)), float(np.percentile(b, 100 - a))


# ----------------------------------------------------------------------------------------------------------------
# (d) harmonization: one fitter, one concentration window, one drug panel (a diagnostic; needs curve-level data)
def mean_viability(pec50, slope, front, back, lo, hi, method='analytic', n_grid=101):
    """Mean of a four-parameter log-logistic curve f(x) = (front - back) / (1 + 10^(slope (x + pEC50))) + back over
    [lo, hi] (x = log10 molar dose; CurveCurator's parametrisation, pEC50 = -log10 EC50). DrEval's AUC_curvecurator is
    this mean over the tested range. method='analytic' integrates exactly (int 1/(1+10^(s(x+p))) dx =
    x - log10(1 + 10^(s(x+p)))/s); method='trapezoid' reproduces 66_harmonized_range.mean_viability (101-point grid)."""
    pec, s, f, b, lo, hi = (np.asarray(v, float) for v in (pec50, slope, front, back, lo, hi))
    if method == 'trapezoid':
        grid = np.linspace(0, 1, n_grid)
        X = lo[..., None] + (hi - lo)[..., None] * grid
        with np.errstate(over='ignore'):
            Y = (f - b)[..., None] / (1 + 10 ** (s[..., None] * (X + pec[..., None]))) + b[..., None]
        trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        return trap(Y, grid, axis=-1)
    small = np.abs(s) < 1e-12

    def prim(x):
        u = s * (x + pec) * LN10
        with np.errstate(divide='ignore', invalid='ignore'):
            return x - np.logaddexp(0.0, u) / (LN10 * np.where(small, 1.0, s))
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(small, 0.5, (prim(hi) - prim(lo)) / (hi - lo))
    return b + (f - b) * frac


def tested_ranges(curves, drug='drug', lo='lo', hi='hi'):
    """Per-drug median lowest and highest tested log10 dose (66: the per-drug median over measurements)."""
    return curves.groupby(drug)[[lo, hi]].median().rename(columns={lo: 'lo', hi: 'hi'})


def common_window(range_a, range_b, min_width=1.0):
    """Per shared drug, the concentration window tested in both datasets [max(lo_a, lo_b), min(hi_a, hi_b)] and
    whether it spans >= min_width log10 units (66). range_*: DataFrames indexed by drug with columns lo, hi."""
    shared = range_a.index.intersection(range_b.index)
    w = pd.DataFrame({'lo': np.maximum(range_a.loc[shared, 'lo'], range_b.loc[shared, 'lo']),
                      'hi': np.minimum(range_a.loc[shared, 'hi'], range_b.loc[shared, 'hi'])})
    w['width'] = w.hi - w.lo
    w['kept'] = w.width >= min_width
    return w


def curve_frame(df, drug='pubchem_id', cell='cellosaurus_id', pec50='pEC50_curvecurator', slope='Slope',
                front='Front', back='Back', min_dose='min_dose_M', max_dose='max_dose_M', auc='AUC_curvecurator'):
    """Standardise a curve table (default: DrEval's curated columns) to drug, cell, pec50, slope, front, back, lo, hi
    (log10 molar) and auc (if present); rows without a complete fit or with an invalid dose range are dropped."""
    cols = {drug: 'drug', cell: 'cell', pec50: 'pec50', slope: 'slope', front: 'front', back: 'back'}
    out = df.rename(columns=cols)
    out = out.dropna(subset=list(cols.values()) + [min_dose, max_dose])
    out = out[(out[min_dose] > 0) & (out[max_dose] > out[min_dose])].copy()
    out['lo'], out['hi'] = np.log10(out[min_dose]), np.log10(out[max_dose])
    keep = ['drug', 'cell', 'pec50', 'slope', 'front', 'back', 'lo', 'hi'] + (['auc'] if auc in df.columns else [])
    if auc in df.columns:
        out = out.rename(columns={auc: 'auc'})
    return out[keep]


def summarize_curves(curves, window=None, method='analytic'):
    """Long table (drug, cell, response = -mean viability) from a curve_frame, over each drug's `window` (DataFrame
    lo, hi by drug; drugs outside it dropped) or over each curve's own tested range; duplicates averaged."""
    x = curves if window is None else curves[curves.drug.isin(window.index)]
    if window is None:
        lo, hi = x.lo.values, x.hi.values
    else:
        lo, hi = window.loc[x.drug, 'lo'].values, window.loc[x.drug, 'hi'].values
    v = mean_viability(x.pec50.values, x.slope.values, x.front.values, x.back.values, lo, hi, method=method)
    o = pd.DataFrame({'drug': x.drug.values, 'cell': x.cell.values, 'v': v}).groupby(['drug', 'cell'], as_index=False).v.mean()
    return pd.DataFrame({'drug': o.drug.values, 'cell': o.cell.values, 'response': -o.v.values})


def shared_panel(obs_a, obs_b, drugs=None):
    """Restrict two long tables to the drugs measured in both (or to `drugs`), so that both axes use one panel."""
    keep = set(obs_a.drug) & set(obs_b.drug)
    if drugs is not None:
        keep &= set(drugs)
    return obs_a[obs_a.drug.isin(keep)], obs_b[obs_b.drug.isin(keep)]


def harmonize_pair(curves_a, curves_b, step='common_range', min_width=1.0, method='analytic'):
    """The harmonization ladder for one pair of datasets (66/90), from curve_frame tables.
    step: 'shared_panel' (each dataset's own-range mean viability on the drugs whose common window spans >= min_width;
    66's SP) or 'common_range' (both datasets summarised over the common window; 66's HR). Returns (obs_a, obs_b,
    window). The harmonized summaries are diagnostics: neither is what the models were trained on."""
    win = common_window(tested_ranges(curves_a), tested_ranges(curves_b), min_width)
    kept = win[win.kept]
    if step == 'shared_panel':
        a = summarize_curves(curves_a[curves_a.drug.isin(kept.index)], None, method)
        b = summarize_curves(curves_b[curves_b.drug.isin(kept.index)], None, method)
    elif step == 'common_range':
        a, b = summarize_curves(curves_a, kept, method), summarize_curves(curves_b, kept, method)
    else:
        raise ValueError(step)
    return a, b, win


# ----------------------------------------------------------------------------------------------------------------
# (e) equivalence on the retained fraction
def retained_fraction_test(sum_within, sum_cross, margin=0.8, alpha=0.05):
    """Non-inferiority of a retained fraction f = cross / within (R1-M3e, R3-M4; 83). sum_within, sum_cross:
    bootstrap draws (R,) of the pooled within and cross scores (row 0 = point). H0: f <= margin, rejected when the
    one-sided (1 - alpha) lower bound (the alpha percentile) exceeds the margin; bootstrap P = share of draws with
    f <= margin. Returns point, two-sided 95% and (1 - 2 alpha) intervals, the lower bound, P and the decision."""
    f = np.asarray(sum_cross, float) / np.asarray(sum_within, float)
    b = f[1:][np.isfinite(f[1:])]
    lo95, hi95 = interval(f, 95)
    lo90, hi90 = interval(f, 100 - 200 * alpha)
    lower = float(np.percentile(b, 100 * alpha))
    return {'retained_fraction': float(f[0]), 'lo95': lo95, 'hi95': hi95, 'lo90': lo90, 'hi90': hi90,
            'lower_onesided': lower, 'p_boot': float(np.mean(b <= margin)), 'noninferior': bool(lower > margin),
            'margin': margin}


def two_way_draws(n_cells, n_drugs, n_boot=1000, seed=0):
    """Cell weights W (n_boot+1, n_cells) and drug multiplicities V (n_boot+1, n_drugs); row 0 = ones (77.draws)."""
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones(n_cells), rng.multinomial(n_cells, np.full(n_cells, 1.0 / n_cells), size=n_boot)])
    V = np.vstack([np.ones(n_drugs), rng.multinomial(n_drugs, np.full(n_drugs, 1.0 / n_drugs), size=n_boot)])
    return W.astype(float), V.astype(float)


def _wpartial2(x, y, c1, c2, w):
    """Weighted partial correlation of x and y given (c1, c2) for every row of w (R, n)."""
    X = np.column_stack([x, y, c1, c2])
    s = w.sum(1)
    m = (w @ X) / s[:, None]
    C = np.einsum('rn,ni,nj->rij', w, X, X) / s[:, None, None] - m[:, :, None] * m[:, None, :]
    A, Bm = C[:, :2, :2], C[:, :2, 2:]
    Cc = C[:, 2:, 2:]
    with np.errstate(invalid='ignore', divide='ignore'):
        P = A - Bm @ np.linalg.pinv(Cc, rcond=1e-10) @ np.transpose(Bm, (0, 2, 1))   # pinv: g_S = g_T allowed
        return P[:, 0, 1] / np.sqrt(P[:, 0, 0] * P[:, 1, 1])


def retained_fraction_bootstrap(P, Zs, Gs, Zt, Gt, margin=0.8, n_boot=1000, seed=0, min_cells=30, alpha=0.05):
    """Symmetric selective-both within (r(p, z_S | g_S, g_T)) and cross (r(p, z_T | g_S, g_T)) per drug on matched
    cells, a two-way bootstrap (cell weights inside every correlation, drug multiplicities when pooling), and
    retained_fraction_test on the pooled means. P, Zs, Gs, Zt, Gt: drug x cell DataFrames (one model).
    Returns (test dict, per-drug DataFrame of point values)."""
    drugs = [d for d in P.index if d in Zs.index and d in Zt.index]
    cells = P.columns.intersection(Zs.columns).intersection(Zt.columns)
    W, V = two_way_draws(len(cells), len(drugs), n_boot, seed)
    sw, sc = np.zeros(W.shape[0]), np.zeros(W.shape[0])
    rows = []
    for j, d in enumerate(drugs):
        v = [M.loc[d].reindex(cells).values.astype(float) for M in (P, Zs, Gs, Zt, Gt)]
        ok = np.all([np.isfinite(x) for x in v], axis=0)
        if ok.sum() < min_cells or np.std(v[0][ok]) < 1e-12:
            continue
        p, zs, gs, zt, gt = (_std(x[ok]) for x in v)
        wi = _wpartial2(p, zs, gs, gt, W[:, ok])
        cr = _wpartial2(p, zt, gs, gt, W[:, ok])
        good = np.isfinite(wi) & np.isfinite(cr)
        sw += np.where(good, wi, 0) * V[:, j]
        sc += np.where(good, cr, 0) * V[:, j]
        rows.append({'drug': d, 'n_cells': int(ok.sum()), 'within_sym': wi[0], 'cross': cr[0]})
    return retained_fraction_test(sw, sc, margin, alpha), pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# (f) multiplicity
def holm(p):
    """Holm step-down adjusted P values (NaN preserved; ties in input order)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok], kind='mergesort')]
    m = len(order)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


# ----------------------------------------------------------------------------------------------------------------
# (g) canonical-biomarker baseline
def biomarker_scores(feature, Z, G, drugs=None, sign=1.0, min_n=30):
    """A training-free, cell-blind single-feature predictor f = sign * feature (e.g. SLFN11 expression for TOP1
    poisons, TP53 wild type for MDM2 antagonists; 52's pre-specified catalogue) scored like a model: per drug, raw
    r(f, z) and selective r(f, z | g) over the cell lines with f, z and g (0 when f is a linear function of g).
    feature: Series indexed by cell line."""
    drugs = list(Z.index) if drugs is None else [d for d in drugs if d in Z.index]
    f = sign * pd.Series(feature).reindex(Z.columns).values.astype(float)
    rows = []
    for d in drugs:
        z, g = Z.loc[d].values.astype(float), G.loc[d].reindex(Z.columns).values.astype(float)
        m = np.isfinite(f) & np.isfinite(z) & np.isfinite(g)
        if m.sum() < min_n or np.std(f[m]) < 1e-12:
            continue
        rows.append({'drug': d, 'n': int(m.sum()), 'raw': float(np.corrcoef(f[m], z[m])[0, 1]),
                     'selective': selbench.partial_corr(f[m], z[m], [g[m]], min_n=min_n, degenerate=0.0)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# (h) lineage-adjusted selective score
def tissue_indicators(tissue):
    """0/1 indicators of every tissue label but the most frequent (reference); (n, k)."""
    lab = np.asarray(tissue).astype(str)
    u, cnt = np.unique(lab, return_counts=True)
    ref = u[np.argmax(cnt)]
    cols = [(lab == x).astype(float) for x in u if x != ref]
    return np.column_stack(cols) if cols else np.zeros((len(lab), 0))


def lineage_terms(p, z, g, tissue, min_n=30):
    """One drug: raw r(p, z), selective r(p, z | g), lineage-adjusted r(p, z | g, tissue indicators), the covariate
    term (the part of r(p, z) carried by (g, tissue): S_pC S_CC^-1 S_Cz / (s_p s_z), so raw = covariate term +
    residual term exactly) and R2_z|C (the perfect predictor's covariate term). Unweighted (91 adds bootstrap
    weights). A predictor in the span of (g, tissue), such as a tissue-by-drug mean fitted on all lines, has an
    undefined lineage-adjusted score (NaN)."""
    p, z, g = (np.asarray(x, float) for x in (p, z, g))
    tis = np.asarray(tissue)
    m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g) & pd.notna(tis)
    if m.sum() < min_n or np.std(p[m]) < 1e-12:
        return None
    p, z, g, tis = _std(p[m]), _std(z[m]), _std(g[m]), tis[m]
    Cm = np.column_stack([g, tissue_indicators(tis)])
    X = np.column_stack([np.ones(len(z)), Cm])
    bp, bz = (np.linalg.lstsq(X, v, rcond=None)[0] for v in (p, z))
    ep, ez = p - X @ bp, z - X @ bz
    n = len(z)
    resid_term = (ep @ ez) / n                        # p, z standardised: residual covariance in correlation units
    raw = (p @ z) / n
    lin = np.nan if (ep @ ep) / n <= 1e-8 or (ez @ ez) / n <= 1e-8 else resid_term / np.sqrt((ep @ ep) * (ez @ ez) / n ** 2)
    return {'n': int(n), 'tissues': int(len(np.unique(tis))), 'raw': float(raw),
            'selective': selbench.partial_corr(p, z, [g], min_n=min_n),
            'selective_lineage': float(lin), 'covariate_term': float(raw - resid_term),
            'r2_z_covariates': float(1 - (ez @ ez) / (z @ z))}


def lineage_adjusted_scores(P, Z, G, tissue, drugs=None, min_n=30):
    """Per drug lineage_terms for a model's predictions P (drug x cell DataFrame) against Z with axis G; tissue: Series
    cell -> label (e.g. DrEval's tissue mapping or DepMap OncotreeLineage)."""
    drugs = [d for d in (P.index if drugs is None else drugs) if d in Z.index and d in P.index]
    cells = P.columns.intersection(Z.columns)
    t = pd.Series(tissue).reindex(cells).values
    rows = []
    for d in drugs:
        r = lineage_terms(P.loc[d, cells].values, Z.loc[d, cells].values, G.loc[d].reindex(cells).values, t, min_n)
        if r is not None:
            rows.append({'drug': d, **r})
    return pd.DataFrame(rows)


def lineage_capture(raw, covariate_term, r2_z_covariates, weights=None):
    """Lineage-adjusted selective capture: sum (raw - covariate term) / sum (1 - R2_z|C), i.e. capture's selective
    term with (g, tissue) in place of g; and the covariate share sum covariate term / sum raw."""
    (raw, ct, r2), one = _as2d(raw, covariate_term, r2_z_covariates)
    ok = np.isfinite(raw) & np.isfinite(ct) & np.isfinite(r2)
    w = np.where(ok, _weights(raw.shape, weights), 0.0)

    def f(x):
        return (w * np.where(ok, x, 0.0)).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        n = w.sum(1)
        out = {'selective_capture_lineage': (f(raw) - f(ct)) / (n - f(r2)), 'covariate_share': f(ct) / f(raw),
               'perfect_covariate_share': f(r2) / n}
    return _ret(out, one)


# ----------------------------------------------------------------------------------------------------------------
# (i) active-drug axis with split-half reliability
def active_drugs(auc, frac=0.1, cut=0.8):
    """Drugs active in a screen (45 / 02_decompose.activity_stats): at least `frac` of the measured cell lines have
    AUC < `cut` (AUC on the 0-1 mean-viability scale; lower = more sensitive). auc: drug x cell DataFrame."""
    a = auc.values.astype(float)
    n = np.isfinite(a).sum(axis=1)
    with np.errstate(invalid='ignore'):
        f = (a < cut).sum(axis=1) / np.where(n > 0, n, np.nan)
    return pd.Index(auc.index[f >= frac])


def active_axis(Z, active, min_drugs=10):
    """Active-drug axis: (cell-level axis, per-drug leave-drug-out axis matrix). The cell-level axis is the mean z over
    the active drugs (45's axis_of; cells with fewer than min_drugs active drugs -> NaN); the per-drug axis for
    scoring is selbench.fitness_axis restricted to the active panel (inactive drugs receive the full active mean)."""
    S = Z.loc[Z.index.intersection(pd.Index(active))]
    ax = S.mean(axis=0)
    ax[S.notna().sum(axis=0) < min_drugs] = np.nan
    G = selbench.fitness_axis(Z, panel=pd.Index(active), min_drugs=min_drugs)
    return ax, G


def axis_split_half(Z, drugs=None, n_splits=200, seed=0, min_n=20):
    """Split-half reliability of an axis over random halves of its drug set (Spearman-Brown; 45's split_half):
    (median, 2.5th, 97.5th percentile) over n_splits splits. Report it beside every selective score: an unreliable
    axis under-adjusts the selective score (gCSI's 44 drugs give 0.73)."""
    rng = np.random.default_rng(seed)
    S = Z if drugs is None else Z.loc[Z.index.intersection(pd.Index(drugs))]
    V = S.values.astype(float)
    if len(V) < 4:
        return np.nan, np.nan, np.nan
    h = len(V) // 2
    out = []
    for _ in range(n_splits):
        pm = rng.permutation(len(V))
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                a, b = np.nanmean(V[pm[:h]], axis=0), np.nanmean(V[pm[h:]], axis=0)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_n:
            continue
        r = np.corrcoef(a[m], b[m])[0, 1]
        out.append(2 * r / (1 + r))
    out = np.array(out)
    if not len(out):
        return np.nan, np.nan, np.nan
    return float(np.median(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ----------------------------------------------------------------------------------------------------------------
# (j) censoring: undefined or out-of-range values ranked as tied at the resistant end
def censored_rank_normal(x, censored=None):
    """Rank-based inverse normal transform of one drug's profile (higher = more sensitive) with censored entries tied
    at the resistant end. x: (n,) values; censored: (n,) booleans (True = censored or undefined, e.g. an ln IC50 beyond
    the tested range or a curve that never crosses 50%; the value in x is ignored). Without `censored`, every NaN in x
    is taken as censored, so pass `censored` explicitly when NaN can also mean 'not screened' (those stay NaN).
    The censored entries share the average of the lowest ranks (selbench.rank_normal's tie rule), below every
    uncensored value. If no entry is uncensored the profile is undefined (all NaN)."""
    x = np.asarray(x, float)
    c = ~np.isfinite(x) if censored is None else np.asarray(censored, bool)
    obs = ~c & np.isfinite(x)
    y = np.full(x.shape, np.nan)
    if not obs.any():
        return y
    y[obs] = x[obs]
    y[c] = x[obs].min() - 1.0
    return selbench.rank_normal(y)


def censor_resistant(obs, screened=None, censored=None, resistant_limit=None, offset=1.0):
    """Long table in which censored or undefined responses are tied at the resistant end, ready for
    selbench.decompose (whose rank-normal transform averages ties). Box 1's censoring rule.

    obs: long table drug, cell, response (higher = more sensitive, e.g. -ln IC50); NaN = undefined.
    screened: optional long table with columns drug, cell listing every screened pair (e.g. the pairs with an AUC):
        screened pairs absent from obs, or NaN there, are censored; obs pairs outside `screened` are dropped.
    censored: optional boolean flags aligned with obs's rows (array/Series) or the name of a boolean column of obs
        (True = censored by the fitter).
    resistant_limit: optional resistant-end limit on the response scale: a scalar, a Series indexed by drug, or the
        name of a column of obs; responses <= limit are censored. For -ln IC50 use -ln(highest tested concentration)
        in the IC50's units, so that an IC50 beyond the tested range is tied with the undefined ones.
    offset: the tie value is the drug's lowest uncensored response minus `offset` (98_endpoint_cell_selection.py
        uses 1.0); any positive value gives the same ranks.
    Returns drug, cell, response, censored (bool). Drugs without an uncensored response keep NaN responses."""
    o = obs.copy()
    flag = np.zeros(len(o), bool)
    if censored is not None:
        flag |= (o[censored].values if isinstance(censored, str) else np.asarray(censored)).astype(bool)
    if resistant_limit is not None:
        if isinstance(resistant_limit, str):
            lim = o[resistant_limit].values.astype(float)
        elif isinstance(resistant_limit, pd.Series):
            lim = o.drug.map(resistant_limit).values.astype(float)
        else:
            lim = np.full(len(o), float(resistant_limit))
        with np.errstate(invalid='ignore'):
            flag |= np.isfinite(lim) & (o.response.values.astype(float) <= lim)
    o['censored'] = flag | ~np.isfinite(o.response.values.astype(float))
    o = o[['drug', 'cell', 'response', 'censored']]
    if screened is not None:
        o = screened[['drug', 'cell']].drop_duplicates().merge(o, on=['drug', 'cell'], how='left')
        o['censored'] = np.where(o.censored.isna(), True, o.censored).astype(bool)
    resp = o.response.where(~o.censored)
    lowest = resp.groupby(o.drug).transform('min')
    o['response'] = np.where(o.censored, lowest - offset, resp)
    return o.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# (k) portable activity rule with endpoint-specific defaults
ACTIVITY_RULES = {
    'auc': {'direction': 'below', 'threshold': 0.8,
            'scale': 'AUC as mean viability over the tested range (0-1; lower = more sensitive), e.g. DrEval '
                     'CurveCurator AUC, CTRP and GDSC provider AUC, gCSI mean viability'},
    'aac': {'direction': 'above', 'threshold': 0.2,
            'scale': 'activity area / AAC = 1 - normalized AUC (0-1; higher = more sensitive), e.g. PharmacoGx AAC, '
                     'CCLE ActArea divided by its number of doses'},
    'gr_aoc': {'direction': 'above', 'threshold': 0.2,
               'scale': 'GR area over the curve (higher = more sensitive)'},
    'ln_ic50': {'direction': 'below', 'threshold': 'max_tested',
                'scale': 'ln IC50 in the units of max_tested (ln of the highest tested concentration): a line responds '
                         'when its IC50 lies below the highest tested concentration; screened lines without an IC50 '
                         '(the curve never crosses 50%) count as non-responding'},
    'generic': {'direction': None, 'threshold': None,
                'scale': 'any endpoint: a line responds when its value lies beyond a stated threshold in the stated '
                         'direction (use direction="above" on an oriented scale, higher = more sensitive)'},
}
PAPER_ACTIVITY_RULE = ('The paper used the AUC default: a drug is active in a screen when at least 10% of its cell lines '
                       'have AUC < 0.8 on the 0-1 mean-viability scale (discovery screens PRISM, CTRP, GDSC1, GDSC2 and '
                       'gCSI; 02_decompose.activity_stats, 45_axis_activity_lineage.py), and the equivalent AAC form '
                       '(1 - AAC < 0.8, i.e. AAC > 0.2) for CCLE, FIMM and GRAY. The AUC threshold depends on the '
                       'tested range (a drug tested to 66 uM looks more active than one tested to 5 uM); the ln IC50 '
                       'default (IC50 below the highest tested concentration) is range-aware.')


def activity_fraction(M, endpoint='auc', threshold=None, direction=None, max_tested=None, screened=None):
    """Per drug, the fraction of screened cell lines that respond (lie beyond the endpoint's activity threshold).

    M: drug x cell DataFrame of the endpoint on its native scale. endpoint: a key of ACTIVITY_RULES. `threshold` and
    `direction` ('below' or 'above') override the defaults and are required for 'generic'.
    max_tested (ln_ic50): the highest tested concentration on the IC50's ln scale: a scalar, a Series indexed by drug,
    or a drug x cell DataFrame (per pair).
    screened: optional boolean drug x cell DataFrame of the screened pairs (default: the non-missing entries of M).
    Screened pairs with a missing value count as non-responding lines (an undefined IC50). For ln IC50 pass it:
    otherwise undefined IC50s are dropped from the denominator and activity is overstated."""
    rule = ACTIVITY_RULES[endpoint]
    direction = direction or rule['direction']
    if direction not in ('below', 'above'):
        raise ValueError('direction must be "below" or "above" (required for a generic rule)')
    A = M.astype(float)
    if endpoint == 'ln_ic50' and threshold is None:
        if max_tested is None:
            raise ValueError('ln_ic50 needs max_tested (ln of the highest tested concentration, IC50 units)')
        if isinstance(max_tested, pd.DataFrame):
            thr = max_tested.reindex(index=A.index, columns=A.columns).values.astype(float)
        elif isinstance(max_tested, pd.Series):
            thr = max_tested.reindex(A.index).values.astype(float)[:, None]
        else:
            thr = float(max_tested)
    else:
        thr = rule['threshold'] if threshold is None else threshold
        if thr is None or isinstance(thr, str):
            raise ValueError('a numeric threshold is required for this endpoint')
    v = A.values
    with np.errstate(invalid='ignore'):
        hit = (v < thr) if direction == 'below' else (v > thr)
    hit &= np.isfinite(v)
    scr = np.isfinite(v) if screened is None else screened.reindex(index=A.index, columns=A.columns).fillna(False).values.astype(bool)
    hit &= scr
    n = scr.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        f = hit.sum(axis=1) / np.where(n > 0, n, np.nan)
    return pd.Series(f, index=A.index, name='active_fraction')


def active_drugs_rule(M, endpoint='auc', frac=0.1, threshold=None, direction=None, max_tested=None, screened=None):
    """Drugs active in a screen under a portable rule: at least `frac` (default 10%) of the screened cell lines
    respond (activity_fraction). With the defaults (endpoint='auc') this is the paper's rule (PAPER_ACTIVITY_RULE)
    and equals active_drugs(M)."""
    f = activity_fraction(M, endpoint, threshold, direction, max_tested, screened)
    return pd.Index(f.index[f.values >= frac])


# ----------------------------------------------------------------------------------------------------------------
# (l) rank intervals (88_rank_intervals.py)
def rank_intervals(scores, models=None, level=95):
    """Bootstrap rank intervals for a leaderboard (Box 1, step 6). scores: (R, M) draws of each model's summary score
    (higher = better; row 0 = the point estimate), every model scored on the same two-way draws so that ranks and score
    differences are paired. Returns one row per model: point rank (1 = best; ties averaged), its `level`% rank interval,
    and the number of other models whose order relative to it is resolved (the interval of the score difference
    excludes 0)."""
    from scipy.stats import rankdata
    S = np.asarray(scores, float)
    if S.ndim != 2:
        raise ValueError('scores must be (R, M) draws with row 0 = the point estimate')
    models = list(models) if models is not None else [f'model{i}' for i in range(S.shape[1])]
    Rk = np.full(S.shape, np.nan)
    for b in range(S.shape[0]):
        ok = np.isfinite(S[b])
        Rk[b, ok] = rankdata(-S[b, ok])
    rows = []
    for i, m in enumerate(models):
        lo, hi = interval(Rk[:, i], level)
        resolved = 0
        for j in range(S.shape[1]):
            if j != i:
                dlo, dhi = interval(S[:, i] - S[:, j], level)
                resolved += bool(dlo > 0 or dhi < 0)
        rows.append({'model': m, 'score': float(S[0, i]), 'rank': float(Rk[0, i]), 'rank_lo': lo, 'rank_hi': hi,
                     'resolved_vs_others': resolved})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# (m) expected retention from screen triads (84_source_reference.py; 49's estimator)
def psi_overlap(r_fa, r_fb, r_ab):
    """n x Cov(r_fa, r_fb) for two correlations that share variable f (Pearson-Filon / Olkin-Siotani)."""
    return r_ab * (1 - r_fa ** 2 - r_fb ** 2) - 0.5 * r_fa * r_fb * (1 - r_fa ** 2 - r_fb ** 2 - r_ab ** 2)


def triad_reliability(drug, r_fa, r_fb, r_ab, n, drug_weights=None):
    """Reproducible share R_f of a focal dataset f's selective variance, from screen triads.

    Under z_X = t + u_X with errors independent between datasets, r_fa r_fb / r_ab = R_f for any two other datasets
    a, b measuring the drug. Inputs are one entry per (drug, a, b) triad: the correlations of the selective residuals
    on common cells (f-a, f-b, a-b) and n, the smallest of the three cell counts. The products are bias-corrected
    (r_fa r_fb - psi/n) and the estimate is pooled as a ratio of sums over drugs of the per-drug mean over triads,
    sum_d mean_k (r_fa r_fb - psi/n) / sum_d mean_k r_ab (84, as 49's primary estimator). drug_weights: optional
    {drug: multiplicity} (a drug-bootstrap draw). Returns (R_f, number of drugs)."""
    r_fa, r_fb, r_ab, n = (np.asarray(x, float) for x in (r_fa, r_fb, r_ab, n))
    T = pd.DataFrame({'drug': np.asarray(drug), 'num': r_fa * r_fb - psi_overlap(r_fa, r_fb, r_ab) / n, 'den': r_ab})
    g = T.groupby('drug')[['num', 'den']].mean()
    w = np.ones(len(g)) if drug_weights is None else pd.Series(drug_weights).reindex(g.index).fillna(0).values
    with np.errstate(invalid='ignore', divide='ignore'):
        return float((w * g.num.values).sum() / (w * g.den.values).sum()), int(len(g))


def expected_retention(R_source, R_target):
    """Retention of selective-both expected for a predictor of the selectivity two datasets share (Box 1, step 3;
    84): sqrt(R_target / R_source). Under z_X = t + u_X a predictor p of t has corr(p, z_X) = corr(p, t) sqrt(R_X),
    so its cross-dataset over within-dataset score is sqrt(R_T / R_S). Observed retention below it means the
    predictions carried selectivity specific to the source dataset (or, if the tissue-by-drug mean falls as far short,
    tissue-level effects that differ between datasets). Accepts scalars or matched bootstrap draws."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(np.asarray(R_target, float) / np.asarray(R_source, float))


# ----------------------------------------------------------------------------------------------------------------
# (n) leakage of an unreliable axis into selective capture (Box 1, step 4; 100b_known_answer_simulation.py)
_LEAKAGE_SPLITS = {'cell': 'leave-cell-line-out', 'leave-cell-line-out': 'leave-cell-line-out'}


def _leakage_args(A, m, rho, split):
    if split not in _LEAKAGE_SPLITS:
        raise ValueError(
            f"split={split!r}: the leakage formula assumes that the axis's measurement error is independent of the "
            'predictions, which holds only when the predictions were made without the evaluation cell lines\' measured '
            'responses (leave-cell-line-out). In pair splits (and drug-blind splits) the predictions are trained on the '
            "same cell lines' responses to other drugs and share the axis's error; the formula then over-corrects "
            "(IMPROVE's cell-mean floor: S 0.002-0.003 observed, 0.08-0.12 predicted). Use empirical drug-agnostic "
            'baselines on the same metric (matched_baselines) instead.')
    A, m, rho = (np.asarray(x, float) for x in (A, m, rho))
    if np.any((rho <= 0) | (rho > 1)):
        raise ValueError('rho must lie in (0, 1]')
    return A, m, rho


def axis_leakage_bound(A, m, rho, split='cell'):
    """Selective capture that an unreliable leave-drug-out axis adds to a predictor's: A m (1 - rho) / [(1 - m) rho].

    A: axis capture and m: perfect-predictor axis share (mean r_zg^2), both as estimated with the observed axis
    (capture()['axis_capture'], ['perfect_share']); rho: the axis's reliability as a measure of the latent axis
    (corr(g, a)^2; estimate it with axis_split_half). Scalars, arrays or matched bootstrap draws.

    Derivation (classical error): if the axis's error is independent of the predictions p and of the profile z,
    r_zg = sqrt(rho) r_za and r_pg = sqrt(rho) r_pa, so A is unchanged, m = rho m_lat, and
        S_obs - S_lat = A m (1 - rho) / [(1 - m) rho]  -  S_lat m_lat (1 - rho) / (1 - m).
    Hence the value returned is
      * exact for drug-agnostic predictors that relate to the responses only through the axis (S_lat = 0): their
        whole observed S is leakage;
      * an upper bound on the leakage of predictors with positive drug-specific signal (S_lat > 0), increasingly
        conservative as S_lat grows (a purely selective predictor's leakage is negative);
      * invalid for pair-split predictions (shown for IMPROVE's cell-mean floor) and, by the same argument, for
        drug-blind ones: the predictions are then built from the same cell lines' responses to other drugs and share
        the axis's measurement error (ValueError for split='pair' or 'drug').
    Conditions: (i) axis error independent of p and z, (ii) rho is the reliability with respect to the latent axis.
    The split-half estimate (axis_split_half) overstates it for small drug panels (simulation, 100b: 0.59 against
    0.54 at 6 drugs, 0.88 against 0.88 at 30), and the formula then understates the leakage (drug-agnostic
    predictor: 0.17 against 0.20 at 6 drugs, 0.057 against 0.059 at 30), so with few drugs it is not a bound.
    Drug agnosticism alone does not give condition (i): noise shared by all drugs of a cell line, or signal of related
    drugs in the axis, can correlate the axis's error with the profile. Where a drug-agnostic predictor can be scored
    in the same evaluation, prefer its observed selective capture to this formula, compared on the same drugs with the
    same pooling (matched_baselines). selbench.leakage_floor's `floor` is a median selective correlation and must be
    compared with a model's median selective correlation, not with S."""
    A, m, rho = _leakage_args(A, m, rho, split)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = A * m * (1 - rho) / ((1 - m) * rho)
    return float(out) if out.ndim == 0 else out


def corrected_selective_capture(S, A, m, rho, split='cell'):
    """Latent selective capture S_lat implied by the observed S under the classical-error model of
    axis_leakage_bound (100b corrected_S): S_lat = [S (1 - m) - (1 - rho) A m / rho] / (1 - m / rho).

    The inversion is exact under conditions (i)-(ii) of axis_leakage_bound, for any predictor (it uses the full
    two-term leakage, not the bound); it inherits their failure in pair-split and drug-blind designs (ValueError for
    split='pair' or 'drug'). With rho estimated by split-half reliability, it removes most but not all of the
    leakage (simulation, 100b: drug-agnostic S at 30 drugs x 100 cells 0.059 -> 0.002). Prefer metric-matched
    empirical baselines (matched_baselines) where drug-agnostic predictors can be scored. S, A, m: capture() estimates
    with the observed axis; rho: axis reliability (scalars, arrays or matched draws)."""
    A, m, rho = _leakage_args(A, m, rho, split)
    S = np.asarray(S, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (S * (1 - m) - (1 - rho) * A * m / rho) / (1 - m / rho)
    return float(out) if out.ndim == 0 else out


# ----------------------------------------------------------------------------------------------------------------
# (o) general-sensitivity names for selbench's 'fitness' functions (selbench.py keeps the study's early name)
axis = selbench.fitness_axis                      # leave-drug-out general-sensitivity axis per drug
general_sensitivity_axis = selbench.fitness_axis
axis_baseline = selbench.fitness_baseline         # omics-predicted axis baseline (drug-agnostic, cell-blind)
oracle_axis_share = selbench.oracle_fitness_share  # perfect-predictor axis share


# ----------------------------------------------------------------------------------------------------------------
# (p) metric-matched drug-agnostic baselines
def _per_drug_terms(T):
    """drug-indexed raw, selective, alignment (r_pg) and loading (r_zg) from a selbench per-drug table (split_terms
    columns; 'r_pg' and 'r_zg' are accepted for 'alignment' and 'loading')."""
    T = T.rename(columns={'r_zg': 'loading', 'r_pg': 'alignment'})
    missing = {'drug', 'raw', 'selective', 'alignment', 'loading'} - set(T.columns)
    if missing:
        raise ValueError(f'per-drug table lacks {sorted(missing)}')
    if T.drug.duplicated().any():
        raise ValueError('one row per drug expected (filter a long table by predictor first)')
    return T.set_index('drug')[['raw', 'selective', 'alignment', 'loading']].astype(float)


def _weighted_median(x, w):
    """Median of x with drug multiplicities w (> 0): the median of the rows repeated w times when w are integers (the
    mean of the two middle values when the total is even), and the analogous midpoint for non-integer weights."""
    o = np.argsort(x, kind='mergesort')
    x, c = x[o], np.cumsum(w[o])
    half = c[-1] / 2.0
    lo = x[np.searchsorted(c, half - 1e-12 * c[-1], side='left')]
    hi = x[np.searchsorted(c, half + 1e-12 * c[-1], side='right')]
    return float((lo + hi) / 2.0)


def matched_baselines(model, baselines, weights=None):
    """A model's metrics beside the same metrics of drug-agnostic predictors, on the same drugs and aggregation.

    model: one model's per-drug table (selbench.selective_scores: drug, raw, selective, alignment, loading);
    baselines: dict name -> the same table for a drug-agnostic predictor scored on the same evaluation (e.g. the
    per_drug output of selbench.leakage_floor, one predictor at a time). Only drugs finite for the model and for every
    baseline are kept, so all rows pool the same drugs.

    weights: optional Series of drug multiplicities indexed by drug (as in capture(), e.g. the drug counts of one
    bootstrap draw). They must be finite and non-negative with a positive total; drugs missing from the Series get
    weight 0. Every metric uses the same weighted distribution of drugs, so the result equals that of the table with
    each drug repeated `weight` times: pooled captures and the mean are weighted, the median is the weighted median,
    zero-weight drugs are dropped, `drugs` counts the distinct drugs with positive weight and `weight_total` is the sum
    of their weights (the number of rows of the repeated table).

    Returns one row per predictor (the model first): pooled axis_capture and selective_capture (capture()), median and
    mean selective correlation, the drug count and weight total. The model row adds, per metric, the margin over the
    strongest baseline (margin_selective_capture, margin_median_selective, margin_mean_selective). Compare like with
    like: per drug S = sqrt((1 - r_pg^2) / (1 - r_zg^2)) r_pz.g, so a floor on the selective correlation
    (selbench.leakage_floor's `floor`) does not convert into a floor on selective capture. The baselines are empirical
    references, not mathematical lower bounds."""
    tables = {'model': _per_drug_terms(model)}
    tables.update({k: _per_drug_terms(v) for k, v in baselines.items()})
    if len(tables) < 2:
        raise ValueError('at least one baseline is needed')
    common = None
    for T in tables.values():
        ok = T.index[np.isfinite(T.values).all(1)]
        common = ok if common is None else common.intersection(ok)
    if not len(common):
        raise ValueError('no drug is finite for the model and every baseline')
    common = common.sort_values()
    if weights is None:
        w = np.ones(len(common))
    else:
        ws = pd.Series(weights, dtype=float)
        if not np.all(np.isfinite(ws.values)) or np.any(ws.values < 0):
            raise ValueError('weights must be finite and non-negative')
        w = ws.reindex(common).fillna(0.0).values        # drugs missing from the Series get weight 0
        if not w.sum() > 0:
            raise ValueError('weights must have a positive total over the drugs kept')
        keep = w > 0
        common, w = common[keep], w[keep]
    rows = []
    for name, T in tables.items():
        T = T.loc[common]
        c = capture(T.raw.values, T.alignment.values, T.loading.values, weights=w)
        sel = T.selective.values
        rows.append({'predictor': name, 'drugs': int(len(common)), 'weight_total': float(w.sum()),
                     'axis_capture': float(c['axis_capture']), 'selective_capture': float(c['selective_capture']),
                     'median_selective': _weighted_median(sel, w), 'mean_selective': float(np.average(sel, weights=w))})
    R = pd.DataFrame(rows)
    for col in ('selective_capture', 'median_selective', 'mean_selective'):
        R.loc[0, f'margin_{col}'] = R.loc[0, col] - R[col].iloc[1:].max()
    return R

