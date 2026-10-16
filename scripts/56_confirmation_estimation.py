"""Registered confirmation recast as estimation (R3 M2, M6, M7; R3 minors 11, 13; R2 minor 3).

For every untouched cell-line screen (gCSI, NCI-60, CCLE, FIMM, GRAY) the frozen 07_confirm.build is re-run in memory
(read-only import) and every registered quantity is given with a confidence interval, next to its discovery value,
the GDSC2 rehearsal value and the registered threshold:
  H1  disattenuated fitness-axis correlation r / sqrt(rel_C rel_s). Bootstrap of the whole estimator: Poisson(1)
      weights per cell line shared by the confirmation and discovery screens, and (two-way) Poisson(1) weights per
      drug of each screen; both axes and both split-half reliabilities (20 random half-splits by drug identity per
      replicate, Spearman-Brown, median) are recomputed in every replicate, so the uncertainty of the reliabilities is
      propagated. The registered interval (cells resampled, reliabilities fixed) is shown alongside.
  H2  median selective / median raw cross-screen correlation; drug bootstrap (all pairs of a drug move together)
      and two-way (cell x drug) bootstrap.
  H3  medians of the confirmation-consensus selective correlation for reproducible and non-reproducible drugs and
      their difference; drug bootstrap stratified by class, and two-way bootstrap.
  H4  median own-minus-discovery fitness correlation over all drugs; cell bootstrap (per-drug correlations recomputed
      with cell weights) and two-way.
  H5a median selective correlation of the frozen ridge_sel_cons model (reproducible drugs); H5b median paired
      difference consensus-measurement minus model; drug bootstrap and two-way.
  Floor: the frozen drug-agnostic discovery fitness axis (fit_measured, identical for every drug) scored exactly as
      H5a; its median, one-sided Wilcoxon P (the registered H5a test), and the paired model-minus-floor difference.
Also: Holm adjustment across the 23 registered tests that carry P values (with fragility flags and
clustering-robust substitutes from 54_patient_derived_inference.py); the misoriented first gCSI run; gCSI H3's
non-reproducible drugs; NCI-60 reported as effect sizes only (its P values are not interpretable).
Outputs: results/revision_transfer/confirmation_*.csv, holm_registered_tests.csv, 56_summary.json.
"""
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import PROCESSED, RESULTS, write_json

HERE = Path(__file__).resolve().parent
OUT = RESULTS / 'revision_transfer'
DEC = RESULTS / 'decomposition'
SCREENS = ['gCSI', 'NCI60', 'CCLE', 'FIMM', 'GRAY']
DISCOVERY = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
SEED = 20261017
B_H1 = 1000
N_SPLITS = 20
B_CELL = 1000
B_DRUG = 10000
H4_CHUNK = 4000
MIN_DISTINCT = 10


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CONFIRM = _load('confirm_frozen', '07_confirm.py')
FAMS = ['fit_measured', 'ridge_sel_cons', 'rep_sel_cons']


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else np.nan


def ci(a):
    return [pct(a, 2.5), pct(a, 97.5)]


# ------------------------------------------------------------------ weighted statistics
def wcorr_cols(W, X, Y, M):
    """Weighted Pearson correlation per column. W: (B x N) cell weights; X, Y: (N x D) zero-filled; M: (N x D) mask."""
    S0 = W @ M
    Sx, Sy = W @ (M * X), W @ (M * Y)
    Sxx, Syy, Sxy = W @ (M * X * X), W @ (M * Y * Y), W @ (M * X * Y)
    distinct = (W > 0).astype(float) @ M
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my = Sx / S0, Sy / S0
        r = (Sxy / S0 - mx * my) / np.sqrt((Sxx / S0 - mx ** 2) * (Syy / S0 - my ** 2))
    r[distinct < MIN_DISTINCT] = np.nan
    return r


def wpartial_cols(W, P, Z, G, M):
    """Weighted partial correlation r(p, z | g) per column (drug); arrays (N x D), zero-filled, mask M."""
    S0 = W @ M

    def mean(a):
        return (W @ (M * a)) / S0
    with np.errstate(invalid='ignore', divide='ignore'):
        mp, mz, mg = mean(P), mean(Z), mean(G)
        cpz = mean(P * Z) - mp * mz; cpg = mean(P * G) - mp * mg; czg = mean(Z * G) - mz * mg
        vp = mean(P * P) - mp ** 2; vz = mean(Z * Z) - mz ** 2; vg = mean(G * G) - mg ** 2
        rpz = cpz / np.sqrt(vp * vz); rpg = cpg / np.sqrt(vp * vg); rzg = czg / np.sqrt(vz * vg)
        r = (rpz - rpg * rzg) / np.sqrt((1 - rpg ** 2) * (1 - rzg ** 2))
    distinct = (W > 0).astype(float) @ M
    r[distinct < MIN_DISTINCT] = np.nan
    return r


def stack(vectors):
    """List of 1-D arrays (same length) -> zero-filled (N x D) matrix and mask."""
    A = np.vstack(vectors).T.astype(float)
    M = np.isfinite(A).astype(float)
    return np.where(M > 0, A, 0.0), M


def drug_resample_median(rb, rng, groups=None):
    """Median over drugs (columns) after resampling drugs with replacement in every replicate (row)."""
    B, D = rb.shape
    pick = rng.integers(0, D, size=(B, D))
    return np.nanmedian(np.take_along_axis(rb, pick, axis=1), axis=1)


# ------------------------------------------------------------------ H1
def axis_and_reliability(Zf, Mk, wd, wc, splits):
    """Weighted all-drug axis over cells and split-half reliability (median Spearman-Brown over splits)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        f = (wd @ Zf) / (wd @ Mk)
        A = splits * wd[None, :]; Bm = (~splits) * wd[None, :]
        a = (A @ Zf) / (A @ Mk); b = (Bm @ Zf) / (Bm @ Mk)
    ok = np.isfinite(a) & np.isfinite(b)
    w = wc[None, :] * ok
    S0 = w.sum(axis=1)
    a0, b0 = np.where(ok, a, 0), np.where(ok, b, 0)
    ma, mb = (w * a0).sum(1) / S0, (w * b0).sum(1) / S0
    cov = (w * a0 * b0).sum(1) / S0 - ma * mb
    va = (w * a0 * a0).sum(1) / S0 - ma ** 2; vb = (w * b0 * b0).sum(1) / S0 - mb ** 2
    r = cov / np.sqrt(va * vb)
    return f, float(np.median(2 * r / (1 + r)))


def random_splits(D, rng, S=N_SPLITS):
    h = D // 2
    out = np.zeros((S, D), dtype=bool)
    for s in range(S):
        out[s, rng.permutation(D)[:h]] = True
    return out


def h1_bootstrap(screen, Z, hyp, rng):
    """Joint bootstrap of the registered H1 estimator for every discovery screen."""
    Zs = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in DISCOVERY}
    fC = Z.mean(axis=0)
    tested = [x for x in hyp['H1']['by_discovery'] if 'pass' in x]
    disc = [x['discovery'] for x in tested]
    union = sorted(set(Z.columns).union(*[set(Zs[s].columns) for s in disc]))
    upos = {c: i for i, c in enumerate(union)}
    ZCf = np.nan_to_num(Z.values); MC = np.isfinite(Z.values).astype(float)
    cC = np.array([upos[c] for c in Z.columns])
    dd = {}
    for s in disc:
        Zv = Zs[s].values
        common = fC.index.intersection(Zs[s].columns)
        dd[s] = {'Zf': np.nan_to_num(Zv), 'M': np.isfinite(Zv).astype(float), 'cells': np.array([upos[c] for c in Zs[s].columns]),
                 'iC': np.array([Z.columns.get_loc(c) for c in common]), 'iS': np.array([Zs[s].columns.get_loc(c) for c in common])}
    res = {s: {'cells_only': [], 'two_way': []} for s in disc}
    res_max = {'cells_only': [], 'two_way': []}
    for b in range(B_H1):
        wc_all = rng.poisson(1.0, len(union)).astype(float)
        for mode in ['cells_only', 'two_way']:
            wdC = rng.poisson(1.0, Z.shape[0]).astype(float) if mode == 'two_way' else np.ones(Z.shape[0])
            wcC = wc_all[cC]
            fCb, relC = axis_and_reliability(ZCf, MC, wdC, wcC, random_splits(Z.shape[0], rng))
            vals = []
            for s in disc:
                q = dd[s]
                wdS = rng.poisson(1.0, q['Zf'].shape[0]).astype(float) if mode == 'two_way' else np.ones(q['Zf'].shape[0])
                fSb, relS = axis_and_reliability(q['Zf'], q['M'], wdS, wc_all[q['cells']], random_splits(q['Zf'].shape[0], rng))
                x, y, w = fCb[q['iC']], fSb[q['iS']], wcC[q['iC']]
                ok = np.isfinite(x) & np.isfinite(y)
                x, y, w = x[ok], y[ok], w[ok]
                mx, my = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
                r = ((w * (x - mx) * (y - my)).sum()) / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())
                dis = r / np.sqrt(relC * relS)
                res[s][mode].append(dis); vals.append(dis)
            res_max[mode].append(max(vals))
    rows = []
    for x in tested:
        s = x['discovery']
        rows.append({'screen': screen, 'discovery': s, 'common_cells': x['common_cells'], 'corr': x['corr'],
                     'rel_confirm': x['rel_confirm'], 'rel_discovery': x['rel_discovery'], 'disattenuated': x['disattenuated'],
                     'ci95_registered_fixed_reliability': x['dis_ci95'],
                     'ci95_cells_reliability_propagated': ci(res[s]['cells_only']),
                     'ci95_cells_x_drugs_reliability_propagated': ci(res[s]['two_way']),
                     'p_boot_ge_0p70_two_way': float(np.mean(np.array(res[s]['two_way']) >= 0.70)),
                     'boot_upper_lt_1_two_way': pct(res[s]['two_way'], 97.5) < 1.0})
    mx = max(tested, key=lambda x: x['disattenuated'])
    summ = {'max_disattenuated': mx['disattenuated'], 'max_discovery': mx['discovery'],
            'max_ci95_cells': ci(res_max['cells_only']), 'max_ci95_two_way': ci(res_max['two_way']),
            'rel_confirm_two_way_note': 'reliabilities recomputed in every replicate'}
    return rows, summ


# ------------------------------------------------------------------ per-screen analysis
def analyse_screen(screen):
    rng = np.random.default_rng([SEED, SCREENS.index(screen)])
    hyp = json.loads((RESULTS / f'confirmation_{screen}' / 'hypotheses.json').read_text())
    long = pd.read_csv(PROCESSED / 'confirmation' / f'{screen}_long.csv.gz')
    M, Z, Gl, R = CONFIRM.build(long)
    assert Z.shape == (hyp['drugs'], hyp['cells']), (screen, Z.shape)
    sc = pd.read_csv(RESULTS / f'confirmation_{screen}' / 'scores.csv')
    P = {f: pd.read_parquet(RESULTS / 'frozen_predictions' / f'{f}.parquet') for f in FAMS}
    cells = Z.columns
    out = {'screen': screen, 'rows': []}

    def row(hyp_id, quantity, est, ci_drug=None, ci_cell=None, ci_two=None, n=None, extra=None):
        r = {'screen': screen, 'hypothesis': hyp_id, 'quantity': quantity, 'estimate': est,
             'ci95_drug_boot': ci_drug, 'ci95_cell_boot': ci_cell, 'ci95_two_way_boot': ci_two, 'n_units': n}
        r.update(extra or {})
        out['rows'].append(r)

    # ---------------- per-drug vectors for the selective scores (verified against scores.csv)
    def scores_for(fam, drugs):
        vec = {}
        for d in drugs:
            p = P[fam].loc[d].reindex(cells).values.astype(float)
            vec[d] = (p, Z.loc[d].values.astype(float), Gl.loc[d].values.astype(float))
        return vec

    scf = {f: sc[sc.family == f].set_index('compound') for f in FAMS + ['own_loo_fitness']}
    rep = scf['rep_sel_cons']
    a_dr = list(rep[rep['class'] == 'reproducible'].sel.dropna().index)
    b_dr = list(rep[rep['class'] == 'non_reproducible'].sel.dropna().index)
    mod = scf['ridge_sel_cons'][scf['ridge_sel_cons']['class'] == 'reproducible'].sel
    rp = rep[rep['class'] == 'reproducible'].sel
    cm = list(mod.index.intersection(rp.index))
    h5_drugs = list(mod[cm].dropna().index)
    W = rng.poisson(1.0, size=(B_CELL, len(cells))).astype(float)

    def boot_partials(fam, drugs):
        v = scores_for(fam, drugs)
        Pm, Mp = stack([v[d][0] for d in drugs]); Zm, Mz = stack([v[d][1] for d in drugs]); Gm, Mg = stack([v[d][2] for d in drugs])
        Mk = Mp * Mz * Mg
        r0 = wpartial_cols(np.ones((1, len(cells))), Pm, Zm, Gm, Mk)[0]
        ref = scf[fam].sel.reindex(drugs).values
        assert np.nanmax(np.abs(r0 - ref)) < 1e-8, (screen, fam, np.nanmax(np.abs(r0 - ref)))
        return wpartial_cols(W, Pm, Zm, Gm, Mk), r0

    # ---------------- H3
    if len(a_dr) and len(b_dr):
        ra_b, ra = boot_partials('rep_sel_cons', a_dr)
        rb_b, rb = boot_partials('rep_sel_cons', b_dr)
        ma, mb = np.median(ra), np.median(rb)
        dboot_a = np.median(rng.choice(ra, (B_DRUG, len(ra))), axis=1); dboot_b = np.median(rng.choice(rb, (B_DRUG, len(rb))), axis=1)
        tw_a = drug_resample_median(ra_b, rng); tw_b = drug_resample_median(rb_b, rng)
        row('H3', 'median reproducible', ma, ci(dboot_a), ci(np.nanmedian(ra_b, axis=1)), ci(tw_a), len(ra))
        row('H3', 'median non-reproducible', mb, ci(dboot_b), ci(np.nanmedian(rb_b, axis=1)), ci(tw_b), len(rb))
        row('H3', 'difference (reproducible - non-reproducible)', ma - mb, ci(dboot_a - dboot_b),
            ci(np.nanmedian(ra_b, axis=1) - np.nanmedian(rb_b, axis=1)), ci(tw_a - tw_b), f'{len(ra)} vs {len(rb)}',
            {'registered_p': hyp['H3'].get('mwu_p_one_sided'), 'p_boot_diff_le_0_two_way': float(np.mean((tw_a - tw_b) <= 0))})
        if screen == 'gCSI':
            nm = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound').name
            out['gcsi_h3_non_reproducible'] = [{'compound': d, 'name': nm.get(d), 'sel': float(rep.loc[d, 'sel']),
                                                'n': int(rep.loc[d, 'n'])} for d in b_dr]

    # ---------------- H5a, H5b and the drug-agnostic floor
    if len(h5_drugs) >= 8:
        m_b, m0 = boot_partials('ridge_sel_cons', h5_drugs)
        r_b, r0 = boot_partials('rep_sel_cons', h5_drugs)
        f_b, f0 = boot_partials('fit_measured', h5_drugs)
        D = len(h5_drugs)
        pick = rng.integers(0, D, size=(B_DRUG, D))
        pick2 = rng.integers(0, D, size=(B_CELL, D))

        def both(x0, xb, label, hyp_id, extra=None):
            dboot = np.median(x0[pick], axis=1)
            tw = np.nanmedian(np.take_along_axis(xb, pick2, axis=1), axis=1)
            row(hyp_id, label, float(np.median(x0)), ci(dboot), ci(np.nanmedian(xb, axis=1)), ci(tw), D, extra)
            return tw

        both(m0, m_b, 'median model selective correlation (ridge_sel_cons)', 'H5a',
             {'registered_p': hyp['H5'].get('model_gt0_p')})
        both(r0, r_b, 'median consensus-measurement selective correlation (rep_sel_cons)', 'H5b')
        tw = both(r0 - m0, r_b - m_b, 'median paired difference measurement - model', 'H5b',
                  {'registered_p': hyp['H5'].get('replicate_gt_model_p')})
        both(f0, f_b, 'FLOOR: median selective correlation of drug-agnostic fit_measured', 'H5a-floor',
             {'wilcoxon_p_one_sided_floor_gt0': float(wilcoxon(f0, alternative='greater').pvalue),
              'floor_n_positive': int((f0 > 0).sum())})
        both(m0 - f0, m_b - f_b, 'median paired difference model - floor', 'H5a-floor',
             {'wilcoxon_p_one_sided_model_gt_floor': float(wilcoxon(m0 - f0, alternative='greater').pvalue)})
        out['h5_p_boot_diff_le_0_two_way'] = float(np.mean(tw <= 0))

    # ---------------- H4 (all confirmation drugs; cell-level bootstrap)
    own = scf['own_loo_fitness'].raw
    fm = P['fit_measured'].iloc[0].reindex(cells).values.astype(float)
    drugs4 = list(Z.index)
    z_all = Z.values.astype(float); g_all = Gl.values.astype(float)
    diffs = np.empty((B_CELL, len(drugs4)), dtype=np.float32)
    own_hat = np.empty(len(drugs4)); disc_hat = np.empty(len(drugs4))
    one = np.ones((1, len(cells)))
    for s0 in range(0, len(drugs4), H4_CHUNK):
        sl = slice(s0, min(s0 + H4_CHUNK, len(drugs4)))
        Zc = z_all[sl].T; Gc = g_all[sl].T
        Fc = np.repeat(fm[:, None], Zc.shape[1], axis=1)
        Mzg = (np.isfinite(Zc) & np.isfinite(Gc)).astype(float)
        Mzf = (np.isfinite(Zc) & np.isfinite(Fc)).astype(float)
        Zf, Gf, Ff = np.nan_to_num(Zc), np.nan_to_num(Gc), np.nan_to_num(Fc)
        own_hat[sl] = wcorr_cols(one, Zf, Gf, Mzg)[0]
        disc_hat[sl] = wcorr_cols(one, Zf, Ff, Mzf)[0]
        diffs[:, sl] = (wcorr_cols(W, Zf, Gf, Mzg) - wcorr_cols(W, Zf, Ff, Mzf)).astype(np.float32)
    # 07_confirm scores the drug-agnostic fit_measured on every confirmation drug: gain = r(z, own axis) - r(z, fm)
    assert np.nanmax(np.abs(own_hat - own.reindex(drugs4).values)) < 1e-8, screen
    point = own_hat - disc_hat
    med4 = float(np.nanmedian(point))
    assert abs(med4 - hyp['H4']['median_difference']) < 1e-8, (screen, med4, hyp['H4']['median_difference'])
    cell_meds = np.nanmedian(diffs, axis=1)
    pick = rng.integers(0, len(drugs4), size=(B_CELL, len(drugs4)))
    two = np.nanmedian(np.take_along_axis(diffs, pick, axis=1), axis=1)
    dboot = np.array([np.nanmedian(point[rng.integers(0, len(point), len(point))]) for _ in range(2000)])
    row('H4', 'median own-minus-discovery fitness correlation (gain)', med4, ci(dboot), ci(cell_meds), ci(two), len(drugs4),
        {'registered_p': hyp['H4'].get('wilcoxon_p_one_sided'), 'p_boot_gain_lt_0p15_cell': float(np.mean(cell_meds < 0.15))})
    del diffs

    # ---------------- H2 (pairs; drug and two-way bootstrap)
    pairs = pd.read_csv(RESULTS / f'confirmation_{screen}' / 'cross_screen_pairs.csv')
    if len(pairs):
        ratio0 = pairs.rho_sel.median() / pairs.rho_raw.median()
        drug_ids = pairs.compound.values
        ud = pd.unique(drug_ids)
        idx_by = {d: np.where(drug_ids == d)[0] for d in ud}
        dboot = []
        for _ in range(B_DRUG // 5):
            sel = rng.integers(0, len(ud), len(ud))
            ii = np.concatenate([idx_by[ud[k]] for k in sel])
            dboot.append(np.median(pairs.rho_sel.values[ii]) / np.median(pairs.rho_raw.values[ii]))
        # cell-weighted re-computation of every pair
        Rs = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in pairs.discovery.unique()}
        Zs = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in pairs.discovery.unique()}
        raw_b = np.full((B_CELL, len(pairs)), np.nan); sel_b = np.full((B_CELL, len(pairs)), np.nan)
        for s, g in pairs.groupby('discovery'):
            cc = Z.columns.intersection(Zs[s].columns)
            wi = np.array([cells.get_loc(c) for c in cc])
            dr = list(g.compound)
            X, Mx = stack([Z.loc[d, cc].values.astype(float) for d in dr]); Y, My = stack([Zs[s].loc[d, cc].values.astype(float) for d in dr])
            Xr, Mxr = stack([R.loc[d, cc].values.astype(float) for d in dr]); Yr, Myr = stack([Rs[s].loc[d, cc].values.astype(float) for d in dr])
            r_raw0 = wcorr_cols(np.ones((1, len(cc))), X, Y, Mx * My)[0]
            assert np.nanmax(np.abs(r_raw0 - g.rho_raw.values)) < 1e-8
            raw_b[:, g.index.values] = wcorr_cols(W[:, wi], X, Y, Mx * My)
            sel_b[:, g.index.values] = wcorr_cols(W[:, wi], Xr, Yr, Mxr * Myr)
        cell_ratio = np.nanmedian(sel_b, axis=1) / np.nanmedian(raw_b, axis=1)
        two = []
        for b in range(B_CELL):
            sel = rng.integers(0, len(ud), len(ud))
            ii = np.concatenate([idx_by[ud[k]] for k in sel])
            two.append(np.nanmedian(sel_b[b, ii]) / np.nanmedian(raw_b[b, ii]))
        row('H2', 'median selective / median raw cross-screen correlation', float(ratio0), ci(dboot), ci(cell_ratio), ci(two),
            f'{len(pairs)} pairs, {len(ud)} drugs', {'p_boot_ratio_lt_0p8_two_way': float(np.mean(np.array(two) < 0.8))})

    # ---------------- H1
    h1_rows, h1_summ = h1_bootstrap(screen, Z, hyp, rng)
    out['h1_rows'] = h1_rows
    row('H1', 'largest disattenuated fitness-axis correlation', h1_summ['max_disattenuated'], None,
        h1_summ['max_ci95_cells'], h1_summ['max_ci95_two_way'], h1_summ['max_discovery'],
        {'registered_ci95_of_that_pair': [x for x in hyp['H1']['by_discovery'] if x.get('discovery') == h1_summ['max_discovery']][0]['dis_ci95']})
    return out


# ------------------------------------------------------------------ references
def discovery_and_rehearsal():
    pn = json.loads((RESULTS / 'paper_numbers.json').read_text())
    disc_pairs = [p for p in pn['fitness_axis']['pairs'] if p['a'] in DISCOVERY and p['b'] in DISCOVERY]
    h1 = [min(p['disattenuated'] for p in disc_pairs), max(p['disattenuated'] for p in disc_pairs)]
    dp = pd.read_csv(DEC / 'cross_screen_pairs.csv')
    h2 = float(dp.rho_sel.median() / dp.rho_raw.median())
    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0)
    h3 = [float(dt.loc[dt.rho_sel >= 0.30, 'rho_sel'].median()), float(dt.loc[dt.rho_sel < 0.15, 'rho_sel'].median())]
    within = pn['models_within_raw']['fitness_measured']
    cross = pn['models_cross_median']['fitness_measured']['raw']
    h4 = [min(within.values()) - cross, max(within.values()) - cross]
    terms = pd.read_csv(RESULTS / 'ceiling' / 'disattenuation_terms.csv')
    long = pd.concat([terms[['model', 'compound', 'pc1']].rename(columns={'pc1': 'pc'}),
                      terms[['model', 'compound', 'pc2']].rename(columns={'pc2': 'pc'})])
    per = long.groupby(['model', 'compound']).pc.mean().unstack(0)
    repd = dt.index[dt.rho_sel >= 0.30]
    per = per.reindex(per.index.intersection(repd))
    h5a = float(per['ridge_sel'].median()); h5r = float(per['replicate'].median())
    h5d = float((per['replicate'] - per['ridge_sel']).median())
    disc = {'H1': h1, 'H2': h2, 'H3': h3, 'H4': h4, 'H5a': h5a, 'H5b_measurement': h5r, 'H5b_difference': h5d,
            'notes': {'H1': 'range of the six discovery-pair disattenuated fitness-axis correlations',
                      'H2': 'median selective / median raw over all discovery cross-screen pairs',
                      'H3': 'discovery class medians of mean selective agreement (by construction: classes are defined by it)',
                      'H4': 'within-screen measured-fitness predictor (raw) minus cross-screen, range over the four screens',
                      'H5': 'cross-screen selective correlation of the per-screen selective ridge / of another screen\'s '
                            'measurement, reproducible drugs (per-drug mean over training x evaluation terms)'}}
    rh = json.loads((RESULTS / 'confirmation_rehearsal_GDSC2' / 'hypotheses.json').read_text())
    rv = [x['disattenuated'] for x in rh['H1']['by_discovery'] if 'disattenuated' in x]
    rehearsal = {'H1': [min(rv), max(rv)], 'H2': rh['H2']['ratio'], 'H3': [rh['H3']['median_reproducible'], rh['H3']['median_non_reproducible']],
                 'H4': rh['H4']['median_difference'], 'H5a': rh['H5']['median_model_sel'], 'H5b_measurement': rh['H5']['median_replicate_sel']}
    sc = pd.read_csv(RESULTS / 'confirmation_rehearsal_GDSC2' / 'scores.csv')
    a = sc[(sc.family == 'ridge_sel_cons') & (sc['class'] == 'reproducible')].set_index('compound').sel
    b = sc[(sc.family == 'rep_sel_cons') & (sc['class'] == 'reproducible')].set_index('compound').sel
    f = sc[(sc.family == 'fit_measured') & (sc['class'] == 'reproducible')].set_index('compound').sel
    c = a.index.intersection(b.index)
    rehearsal['H5b_difference'] = float((b[c] - a[c]).median())
    rehearsal['floor'] = float(f.reindex(a[c].dropna().index).median())
    return disc, rehearsal


THRESH = {'H1': '< 0.70 for every discovery screen and 95% cell-bootstrap upper bound < 1',
          'H2': '>= 0.80', 'H3': 'one-sided Mann-Whitney P < 0.05 (no effect-size threshold)',
          'H4': 'median gain >= 0.15 and one-sided Wilcoxon P < 0.05', 'H5a': 'one-sided Wilcoxon P < 0.05 (model > 0)',
          'H5b': 'paired one-sided Wilcoxon P < 0.05 (measurement > model)', 'H5a-floor': 'not registered (reference)'}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


def scipy_method(test, n_a, n_b=None):
    """Method scipy 1.8 used in the frozen scripts (continuous correlations: no zeros, no ties)."""
    if test == 'wilcoxon':
        return 'exact' if n_a <= 25 else 'normal approximation, no continuity correction'
    return 'exact' if min(n_a, n_b) <= 8 else 'asymptotic normal with continuity correction'


def registered_tests():
    rows = []
    for s in SCREENS:
        h = json.loads((RESULTS / f'confirmation_{s}' / 'hypotheses.json').read_text())
        for k, p, test, na, nb in [('H3', h['H3'].get('mwu_p_one_sided'), 'mannwhitneyu', h['H3']['n_reproducible'], h['H3']['n_non_reproducible']),
                                   ('H4', h['H4'].get('wilcoxon_p_one_sided'), 'wilcoxon', h['H4']['drugs'], None),
                                   ('H5a', h['H5'].get('model_gt0_p'), 'wilcoxon', h['H5']['drugs'], None),
                                   ('H5b', h['H5'].get('replicate_gt_model_p'), 'wilcoxon (paired)', h['H5']['drugs'], None)]:
            if p is not None:
                rows.append({'screen': s, 'hypothesis': k, 'registered_p': p, 'test': test, 'n': na, 'n_b': nb})
    h = json.loads((RESULTS / 'confirmation_BeatAML' / 'hypotheses.json').read_text())
    rows += [{'screen': 'BeatAML', 'hypothesis': 'H6', 'registered_p': h['H6']['wilcoxon_p_one_sided'], 'test': 'wilcoxon', 'n': h['H6']['drugs'], 'n_b': None},
             {'screen': 'BeatAML', 'hypothesis': 'H7', 'registered_p': h['H7']['mwu_p_one_sided'], 'test': 'mannwhitneyu',
              'n': h['H7']['n_reproducible'], 'n_b': h['H7']['n_non_reproducible']}]
    h = json.loads((RESULTS / 'confirmation_PDX_Bruna' / 'hypotheses.json').read_text())
    rows += [{'screen': 'PDX_Bruna', 'hypothesis': 'H8', 'registered_p': h['H8']['wilcoxon_p_one_sided'], 'test': 'wilcoxon', 'n': h['H8']['drugs'], 'n_b': None},
             {'screen': 'PDX_Bruna', 'hypothesis': 'H9', 'registered_p': h['H9']['mwu_p_one_sided'], 'test': 'mannwhitneyu',
              'n': h['H9']['n_reproducible'], 'n_b': h['H9']['n_non_reproducible']}]
    t = pd.DataFrame(rows)
    t['scipy_1.8_method'] = [scipy_method('wilcoxon' if x.startswith('wilcoxon') else 'mwu', a, b) for x, a, b in zip(t.test, t.n, t.n_b)]
    t['p_for_holm'] = t.registered_p.clip(lower=1e-300)  # NCI-60 H4 underflowed to 0.0 (P < 1e-300)
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(analyse_screen, SCREENS))
    disc, rehearsal = discovery_and_rehearsal()
    est = pd.DataFrame([r for res in results for r in res['rows']])

    def ref(h, q):
        """(discovery value, rehearsal value) matching an estimate row."""
        if h == 'H3':
            if q.startswith('median reproducible'):
                return disc['H3'][0], rehearsal['H3'][0]
            if q.startswith('median non'):
                return disc['H3'][1], rehearsal['H3'][1]
            return disc['H3'][0] - disc['H3'][1], rehearsal['H3'][0] - rehearsal['H3'][1]
        if h == 'H5b':
            if 'difference' in q:
                return disc['H5b_difference'], rehearsal['H5b_difference']
            return disc['H5b_measurement'], rehearsal['H5b_measurement']
        if h == 'H5a-floor':
            return (None, rehearsal['floor']) if 'FLOOR' in q else (None, None)
        return disc.get(h), rehearsal.get(h)

    est['discovery_value'] = [json.dumps(ref(h, q)[0]) for h, q in zip(est.hypothesis, est.quantity)]
    est['rehearsal_GDSC2_value'] = [json.dumps(ref(h, q)[1]) for h, q in zip(est.hypothesis, est.quantity)]
    est['registered_threshold'] = est.hypothesis.map(THRESH)
    est['nci60_effect_size_only'] = est.screen == 'NCI60'
    est.loc[est.screen == 'NCI60', 'registered_p'] = np.nan  # effect sizes only (R2 minor 3; R3 M6)
    for c in ['ci95_drug_boot', 'ci95_cell_boot', 'ci95_two_way_boot']:
        est[c] = est[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else json.dumps([round(x, 4) for x in v]))
    est.to_csv(OUT / 'confirmation_estimates.csv', index=False)
    h1 = pd.DataFrame([r for res in results for r in res['h1_rows']])
    h1.to_csv(OUT / 'confirmation_H1_bootstrap.csv', index=False)

    # ---------------- Holm across the registered tests that carry a P value
    t = registered_tests()
    t['holm_adjusted_p'] = holm(t.p_for_holm)
    s54 = json.loads((OUT / '54_summary.json').read_text())
    pdx = pd.DataFrame(s54['pdx']['tests']); bea = pd.DataFrame(s54['beataml']['tests'])
    t1 = pdx[pdx.analysis.str.startswith('T1') & (pdx.grouping == 'G1')].iloc[0]
    t1b = pdx[pdx.analysis.str.startswith('T1') & (pdx.grouping == 'G2')].iloc[0]
    regb = bea.iloc[0]
    cls = s54['beataml']['class_level']
    robust = {('PDX_Bruna', 'H8'): ('tumour level T1 (G1/G2): Wilcoxon P = %.3g/%.3g; mean-z permutation P = %.3g/%.3g' %
                                    (t1.H8_wilcoxon_p, t1b.H8_wilcoxon_p, t1.H8_perm_p_meanz, t1b.H8_perm_p_meanz),
                                    max(t1.H8_wilcoxon_p, t1b.H8_wilcoxon_p, t1.H8_perm_p_meanz, t1b.H8_perm_p_meanz)),
              ('PDX_Bruna', 'H9'): ('tumour level T1 (G1/G2): Mann-Whitney P = %.3g/%.3g; mean-z permutation P = %.3g/%.3g' %
                                    (t1.H9_mwu_p, t1b.H9_mwu_p, t1.H9_perm_p_meanz, t1b.H9_perm_p_meanz),
                                    max(t1.H9_mwu_p, t1b.H9_mwu_p, t1.H9_perm_p_meanz, t1b.H9_perm_p_meanz)),
              ('BeatAML', 'H6'): ('patient-block permutation P = %.2g; class-level Wilcoxon P = %.3g (old classes), %.3g (curated)' %
                                  (regb.H6_perm_p_patient_blocks, cls['old_class']['wilcoxon_p_classes'], cls['curated_class']['wilcoxon_p_classes']),
                                  regb.H6_perm_p_patient_blocks),
              ('BeatAML', 'H7'): ('patient-block permutation P = %.2g; class-level Mann-Whitney P = %.3g (old), %.3g (curated)' %
                                  (regb.H7_perm_p_patient_blocks, cls['old_class']['mwu_p_classes'], cls['curated_class']['mwu_p_classes']),
                                  regb.H7_perm_p_patient_blocks)}
    for res in results:
        s = res['screen']
        e = est[est.screen == s]
        h3 = e[(e.hypothesis == 'H3') & e.quantity.str.startswith('difference')]
        if len(h3):
            two = json.loads(h3.ci95_two_way_boot.iloc[0])
            robust[(s, 'H3')] = (f'two-way (cell x drug) bootstrap 95% CI of the median difference {two}', float(h3.p_boot_diff_le_0_two_way.iloc[0]))
        h5 = e[(e.hypothesis == 'H5b') & e.quantity.str.contains('difference')]
        if len(h5):
            two = json.loads(h5.ci95_two_way_boot.iloc[0])
            robust[(s, 'H5b')] = (f'two-way bootstrap 95% CI of the median paired difference {two}', res.get('h5_p_boot_diff_le_0_two_way'))
    pcl = s54['pdx']['class_level']
    class_p = {('BeatAML', 'H6'): (cls['old_class']['wilcoxon_p_classes'], cls['curated_class']['wilcoxon_p_classes']),
               ('BeatAML', 'H7'): (cls['old_class']['mwu_p_classes'], cls['curated_class']['mwu_p_classes']),
               ('PDX_Bruna', 'H8'): (pcl['old_class|r_model']['wilcoxon_p_classes'], pcl['curated_class|r_model']['wilcoxon_p_classes']),
               ('PDX_Bruna', 'H9'): (pcl['old_class|r_model']['mwu_p_classes'], pcl['curated_class|r_model']['mwu_p_classes'])}
    t['class_level_p_old_curated'] = [json.dumps([round(x, 4) for x in class_p[(a, b)]]) if (a, b) in class_p else None
                                      for a, b in zip(t.screen, t.hypothesis)]
    t['robust_reanalysis'] = [robust.get((a, b), (None, None))[0] for a, b in zip(t.screen, t.hypothesis)]
    t['robust_p_or_boot_tail'] = [robust.get((a, b), (None, None))[1] for a, b in zip(t.screen, t.hypothesis)]
    t['fragile_holm_0p01_to_0p05'] = (t.holm_adjusted_p >= 0.01) & (t.holm_adjusted_p < 0.05)
    t['fails_robust_reanalysis'] = t.robust_p_or_boot_tail.map(lambda v: bool(v is not None and np.isfinite(v) and v >= 0.05))
    t['fragile'] = t.fragile_holm_0p01_to_0p05 | t.fails_robust_reanalysis
    t['nci60_p_not_interpretable'] = t.screen == 'NCI60'
    sub = t[t.screen != 'NCI60'].copy()
    t['holm_adjusted_p_excluding_nci60'] = pd.Series(holm(sub.p_for_holm), index=sub.index).reindex(t.index)
    rob = t.copy()
    for (a, b), (txt, p) in robust.items():
        if b in ('H6', 'H7', 'H8', 'H9') and p is not None:
            rob.loc[(rob.screen == a) & (rob.hypothesis == b), 'p_for_holm'] = max(p, float(rob.loc[(rob.screen == a) & (rob.hypothesis == b), 'p_for_holm'].iloc[0]))
    t['holm_adjusted_p_with_clustered_substitutes'] = holm(rob.p_for_holm)  # least favourable unit-level P (max over variants)
    # primary unit-level reanalyses: PDX tumour level (T1, G1) and BeatAML patient level, same rank tests as registered
    patl = bea[bea.analysis.str.startswith('patient level')].iloc[0]
    unit = {('PDX_Bruna', 'H8'): t1.H8_wilcoxon_p, ('PDX_Bruna', 'H9'): t1.H9_mwu_p,
            ('BeatAML', 'H6'): patl.H6_wilcoxon_p, ('BeatAML', 'H7'): patl.H7_mwu_p}
    up = t.p_for_holm.copy()
    for (a, b), p in unit.items():
        up[(t.screen == a) & (t.hypothesis == b)] = p
    t['unit_level_p'] = [unit.get((a, b)) for a, b in zip(t.screen, t.hypothesis)]
    t['holm_adjusted_p_with_unit_level_p'] = holm(up)
    t.to_csv(OUT / 'holm_registered_tests.csv', index=False)

    # ---------------- PDX versus BeatAML pooled transfer (the manuscript's "twice as large")
    from scipy.stats import norm
    pm = pd.DataFrame(s54['pdx']['meta']); bm = pd.DataFrame(s54['beataml']['meta'])
    b0 = bm[(bm.analysis == 'registered sample level') & (bm['class'] == 'reproducible')].iloc[0]
    cmp_rows = []
    for lab, q in [('PDX registered model level', pm.analysis == 'registered model level'),
                   ('PDX T1 tumour level G1', (pm.analysis == 'T1 tumour level') & (pm.grouping == 'G1')),
                   ('PDX T1 tumour level G2', (pm.analysis == 'T1 tumour level') & (pm.grouping == 'G2'))]:
        r0 = pm[q & (pm['class'] == 'reproducible')].iloc[0]
        d = r0.mu_z - b0.mu_z; se = float(np.sqrt(r0.se_z ** 2 + b0.se_z ** 2))
        cmp_rows.append({'pdx_analysis': lab, 'pdx_pooled_r': float(np.tanh(r0.mu_z)), 'beataml_pooled_r': float(np.tanh(b0.mu_z)),
                         'ratio': float(np.tanh(r0.mu_z) / np.tanh(b0.mu_z)), 'diff_fisher_z': float(d),
                         'diff_ci95': [float(d - 1.96 * se), float(d + 1.96 * se)], 'p_two_sided': float(2 * norm.sf(abs(d) / se))})
    pd.DataFrame(cmp_rows).to_csv(OUT / 'pdx_vs_beataml_pooled.csv', index=False)

    # ---------------- misoriented first gCSI run
    mis = json.loads((RESULTS / 'confirmation_gCSI_misoriented_first_run' / 'hypotheses.json').read_text())
    mrows = [{'hypothesis': 'H1', 'statistic': 'disattenuated r range', 'value': json.dumps([round(x['disattenuated'], 3) for x in mis['H1']['by_discovery'] if 'disattenuated' in x]),
              'passed_as_registered': mis['H1']['pass'], 'passes_with_abs_value': all(abs(x['disattenuated']) < 0.7 for x in mis['H1']['by_discovery'] if 'disattenuated' in x)},
             {'hypothesis': 'H2', 'statistic': 'median sel / median raw', 'value': f"{mis['H2']['median_sel']:.3f} / {mis['H2']['median_raw']:.3f}", 'passed_as_registered': mis['H2']['pass']},
             {'hypothesis': 'H3', 'statistic': 'medians; MWU P', 'value': f"{mis['H3']['median_reproducible']:.3f} vs {mis['H3']['median_non_reproducible']:.3f}; P={mis['H3']['mwu_p_one_sided']:.3f}", 'passed_as_registered': mis['H3']['pass']},
             {'hypothesis': 'H4', 'statistic': 'median gain; Wilcoxon P', 'value': f"{mis['H4']['median_difference']:.3f}; P={mis['H4']['wilcoxon_p_one_sided']:.2g}",
              'passed_as_registered': bool(mis['H4']['median_difference'] >= 0.15 and mis['H4']['wilcoxon_p_one_sided'] < 0.05)},
             {'hypothesis': 'H5a', 'statistic': 'median model; P', 'value': f"{mis['H5']['median_model_sel']:.3f}; P={mis['H5']['model_gt0_p']:.3g}", 'passed_as_registered': mis['H5']['pass_a']},
             {'hypothesis': 'H5b', 'statistic': 'median measurement vs model; P', 'value': f"{mis['H5']['median_replicate_sel']:.3f} vs {mis['H5']['median_model_sel']:.3f}; P={mis['H5']['replicate_gt_model_p']:.3g}", 'passed_as_registered': mis['H5']['pass_b']}]
    pd.DataFrame(mrows).to_csv(OUT / 'misoriented_first_run.csv', index=False)

    summary = {'discovery_reference': disc, 'rehearsal_reference': rehearsal,
               'estimates': est.to_dict(orient='records'), 'h1': h1.to_dict(orient='records'),
               'holm': t.to_dict(orient='records'), 'misoriented_first_run': mrows, 'pdx_vs_beataml': cmp_rows,
               'gcsi_h3_non_reproducible': [r for res in results for r in res.get('gcsi_h3_non_reproducible', [])],
               'settings': {'seed': SEED, 'B_H1': B_H1, 'splits_per_replicate': N_SPLITS, 'B_cell': B_CELL, 'B_drug': B_DRUG,
                            'weights': 'Poisson(1) per cell line (shared across screens) and per drug'}}
    write_json(OUT / '56_summary.json', summary)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
    print(est[['screen', 'hypothesis', 'quantity', 'estimate', 'ci95_drug_boot', 'ci95_cell_boot', 'ci95_two_way_boot']].to_string())
    print(t[['screen', 'hypothesis', 'registered_p', 'holm_adjusted_p', 'fragile', 'robust_p_or_boot_tail']].to_string())


if __name__ == '__main__':
    main()
