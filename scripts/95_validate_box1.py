"""Round 2 (WS6b), task 5: validate selbench/box1.py against the project's existing results (editor item 8; R1-M5;
R2-Major 4), and run the selbench unit tests (the 14 original tests and the box1 tests).

Each row of validation.csv recomputes a published quantity with a box1 function from the same inputs and reports the
largest absolute difference against the stored value and the tolerance it must meet (stored CSVs keep 5-6
significant digits, hence tolerances of 1e-4 to 1e-5 where values pass through a CSV; 1e-9 where the stored values
are full precision; random split-half reliabilities within 0.005):
  a  capture / corrected_capture / corrected_reference   vs results/revision_round2/capture/capture_by_model.csv and
     reliability/corrected_capture_by_model.csv (78), per-drug inputs from lineage/per_drug_lineage.csv.gz (91, whose
     axis-only statistics are identical to 78's)
  b  selective_both          vs paired/per_drug_retrain.csv.gz (80), pair GDSC2_AUC -> GDSC1, every model and drug
  c  attenuation             vs attenuation/attenuation_by_target.csv (81), every retraining pair, pooled group
  d  mean_viability / harmonize_pair   vs DrEval's AUC_curvecurator (GDSC2, every curve) and 66's mean_viability on the
     GDSC2-GDSC1 common window (90's ranges.csv kept count)
  e  retained_fraction_test  vs equivalence/equivalence_34.csv (83), 34 versions x 2 selective-both variants
  f  holm                    vs holm/holm_25.csv (86), every variant
  g  biomarker_scores        vs revision_ceiling/52_biomarker_c2_perdrug.csv within-screen pc_<screen> (52)
  h  lineage_adjusted_scores vs lineage/per_drug_lineage.csv.gz (91), GDSC2 AUC retraining, three models
  i  active_drugs / active_axis / axis_split_half  vs decomposition/drug_screen_stats.csv, revision_axis/axes_native.csv
     and axis_variants.csv (45)
Outputs (results/revision_round2/box1_validation/): validation.csv, tests.txt, summary.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)
import box1  # noqa: E402  (selbench/ is on sys.path via lib77)

OUT = lib.out_dir('box1_validation')
O2 = lib.OUT2
ROWS = []


def record(component, function, reference, diffs, tol, note=''):
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    mx = float(np.max(np.abs(d))) if len(d) else np.nan
    ROWS.append({'component': component, 'function': function, 'reference': reference, 'n_compared': int(len(d)),
                 'max_abs_diff': mx, 'tolerance': tol, 'passed': bool(len(d) and mx <= tol), 'note': note})
    print(f'{component:3s} {function:45s} n={len(d):6d} max|diff|={mx:.2e} tol={tol:.0e} {"PASS" if ROWS[-1]["passed"] else "FAIL"}',
          flush=True)


# ----------------------------------------------------------------------------------------------------------------
def v_capture():
    per = pd.read_csv(O2 / 'lineage' / 'per_drug_lineage.csv.gz', dtype={'drug': str})
    C = pd.read_csv(O2 / 'capture' / 'capture_by_model.csv')
    keys = ['axis_capture', 'selective_capture', 'axis_share', 'perfect_share', 'share_ratio', 'cf_share_ratio',
            'mean_raw', 'alignment_ratio']
    diffs = {k: [] for k in keys}
    groups = per.groupby(['benchmark', 'setting', 'model'])
    for _, r in C.iterrows():
        k = (r.benchmark, r.setting, r.model)
        if k not in groups.groups:
            continue
        g = groups.get_group(k)
        c = box1.capture(g.raw.values, g.r_pg.values, g.r_zg.values, g.sel.values)
        for kk in keys:
            if np.isfinite(r[kk]) and abs(r[kk]) < 1e3:
                diffs[kk].append((c[kk] - r[kk]) / max(1.0, abs(r[kk])))
    for kk in keys:
        record('a', f'capture: {kk}', 'capture/capture_by_model.csv (78)', diffs[kk], 1e-4,
               'relative to max(1, |value|); per-drug inputs stored with 6 significant digits')
    # reliability-corrected
    rz = dict(np.load(O2 / 'reliability' / 'reliability_draws.npz'))
    tp = O2 / 'reliability' / 'triad_R_draws.npz'
    if tp.exists():
        rz.update(dict(np.load(tp)))
    CC = pd.read_csv(O2 / 'reliability' / 'corrected_capture_by_model.csv')
    dd = {k: [] for k in ['corrected_reference', 'axis_capture_tau', 'selective_capture_tau', 'kappa_reference']}
    for _, r in CC.iterrows():
        k = (r.benchmark, r.setting, r.model)
        if k not in groups.groups or r.reliability_source not in rz:
            continue
        g = groups.get_group(k)
        Rs = float(rz[r.reliability_source][0])
        if r.kappa == 1.0:
            c = box1.corrected_capture(g.raw.values, g.r_pg.values, g.r_zg.values, Rs)
            for kk in ['corrected_reference', 'axis_capture_tau', 'selective_capture_tau']:
                if np.isfinite(r[kk]):
                    dd[kk].append((c[kk] - r[kk]) / max(1.0, abs(r[kk])))
        else:
            ok = np.isfinite(g.raw.values) & np.isfinite(g.r_pg.values) & np.isfinite(g.r_zg.values) & np.isfinite(g.sel.values)
            ref = box1.corrected_reference(g.r_zg.values[ok], Rs, kappa=r.kappa)
            dd['kappa_reference'].append(ref - r.corrected_reference)
    for kk, v in dd.items():
        record('a', f'corrected_capture: {kk}', 'reliability/corrected_capture_by_model.csv (78)', v, 1e-4,
               'R_sel point estimate = draw 0 of 79/84 reliability draws')


def v_selective_both():
    rt = lib.load_module('rt59', '59_retrain_score.py')
    wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
    base = sorted(wp[(wp.setting == 'GDSC2_AUC') & (wp.panel == 'main')].model.unique())
    S = lib.retrain_setup('GDSC2_AUC', models=base + [m for m in ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF',
                                                                   'GradientBoosting'] if m not in base])
    Zs, Gs, _, _ = lib.selbench.decompose(S['obs'][['drug', 'cell', 'response']])
    resp = S['resp']
    Zt, Gt, _, _ = lib.selbench.decompose(resp[(resp.dataset == 'GDSC1') & (resp.endpoint == 'auc')][['drug', 'cell', 'response']])
    per = pd.read_csv(O2 / 'paired' / 'per_drug_retrain.csv.gz', dtype={'drug': str})
    per = per[per.pair == 'GDSC2_AUC->GDSC1'].set_index(['model', 'drug'])
    models = S['models']
    REF = 'ref:source_measurement'
    Pm = dict(S['P'])
    Pm[REF] = Zs
    allm = models + [REF]
    pdrugs = sorted(set(Zs.index) & set(Zt.index) & set().union(*[set(S['P'][m].index) for m in models]))
    pcells = sorted(set(Zs.columns) & set(Zt.columns) & set(S['cells']))
    mats = [M.reindex(index=pdrugs, columns=pcells).values.astype(float) for M in (Zs, Gs, Zt, Gt)]
    Pv = {m: Pm[m].reindex(index=pdrugs, columns=pcells).values.astype(float) for m in allm}
    dw, dc, da = [], [], []
    for j, d in enumerate(pdrugs):
        zs, gs, zt, gt = (M[j] for M in mats)
        have = [m for m in allm if d in Pm[m].index]
        ok = np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        for m in have:
            ok &= np.isfinite(Pv[m][j])
        if ok.sum() < lib.MIN_CELLS:
            continue
        for m in have:
            if (m, d) not in per.index:
                continue
            r = box1.selective_both(Pv[m][j][ok], zs[ok], gs[ok], zt[ok], gt[ok])
            ref = per.loc[(m, d)]
            dw.append(r['within_sym'] - ref.selB_w)
            dc.append(r['cross'] - ref.selB_c)
            da.append(r['within_asym'] - ref.selT_w)
    note = 'GDSC2 AUC retraining -> GDSC1, every model and drug; stored values have 6 significant digits'
    record('b', 'selective_both: within_sym r(p,z_S|g_S,g_T)', 'paired/per_drug_retrain.csv.gz (80)', dw, 1e-5, note)
    record('b', 'selective_both: cross r(p,z_T|g_S,g_T)', 'paired/per_drug_retrain.csv.gz (80)', dc, 1e-5, note)
    record('b', 'selective_both: within_asym r(p,z_S|g_S)', 'paired/per_drug_retrain.csv.gz (80)', da, 1e-5, note)


def v_attenuation():
    T = pd.read_csv(O2 / 'attenuation' / 'attenuation_by_target.csv')
    ix = pd.read_csv(O2 / 'paired' / 'pairs.csv')
    per = pd.read_csv(O2 / 'paired' / 'per_drug_retrain.csv.gz', dtype={'drug': str})
    keys = ['raw_loss', 'within_share', 'share_of_loss', 'excess', 'axis_retained', 'selterm_retained',
            'retention_difference']
    diffs = {k: [] for k in keys}
    for _, r in ix[ix.benchmark == 'retrain'].iterrows():
        if not isinstance(r.pool_main, str):
            continue
        t = T[(T.benchmark == 'retrain') & (T.pair == r.pair) & (T.group == 'DrEval learned (retrained) or published leaderboard')]
        if not len(t):
            continue
        t = t.iloc[0]
        g = per[(per.pair == r.pair) & per.model.isin(r.pool_main.split(';'))]
        a = box1.attenuation(g.raw_w.values, g.r_pg_w.values, g.r_zg_w.values, g.raw_c.values, g.r_pg_c.values,
                             g.r_zg_c.values)
        for k in keys:
            if np.isfinite(t[k]) and abs(t[k]) < 50:
                diffs[k].append((a[k] - t[k]) / max(1.0, abs(t[k])))
    for k in keys:
        record('c', f'attenuation: {k}', 'attenuation/attenuation_by_target.csv (81)', diffs[k], 1e-4,
               'every retraining pair, pooled main group; relative to max(1, |value|)')


def v_harmonization():
    hr = lib.load_module('hr66', '66_harmonized_range.py')
    rawS = hr.curves('GDSC2')
    rawT = hr.curves('GDSC1')
    S = box1.curve_frame(rawS)
    T = box1.curve_frame(rawT)
    a = box1.mean_viability(S.pec50, S.slope, S.front, S.back, S.lo, S.hi)
    err = np.abs(a - S.auc.values)
    ROWS.append({'component': 'd', 'function': 'mean_viability (analytic) vs AUC_curvecurator: median |diff|',
                 'reference': 'DrEval GDSC2 curated AUC_curvecurator (every curve)', 'n_compared': int(len(err)),
                 'max_abs_diff': float(np.median(err)), 'tolerance': 1e-4, 'passed': bool(np.median(err) < 1e-4),
                 'note': f'99th percentile {np.percentile(err, 99):.1e} (66 requires < 1e-2; step-like curves); max {err.max():.1e}'})
    print('d median |diff| vs AUC_curvecurator', np.median(err), 'q99', np.percentile(err, 99), flush=True)
    sub = rawS.sample(5000, random_state=2)
    t66 = hr.mean_viability(sub, sub.lo.values, sub.hi.values)
    sb = box1.curve_frame(sub)
    tb = box1.mean_viability(sb.pec50, sb.slope, sb.front, sb.back, sb.lo, sb.hi, method='trapezoid')
    record('d', "mean_viability(method='trapezoid') vs 66.mean_viability", '66_harmonized_range.py', tb - t66, 1e-12,
           '5,000 GDSC2 curves')
    # the GDSC2-GDSC1 common window and the HR summaries (as 90 builds them with 66's functions)
    a_b, b_b, win = box1.harmonize_pair(S, T, 'common_range', method='trapezoid')
    rng90 = pd.read_csv(O2 / 'harmonization_gdsc2' / 'ranges.csv', dtype={'drug': str})
    k90 = int(rng90[(rng90.target == 'GDSC1') & rng90.kept].shape[0])
    ROWS.append({'component': 'd', 'function': 'common_window: drugs kept (GDSC2-GDSC1)',
                 'reference': 'harmonization_gdsc2/ranges.csv (90)', 'n_compared': 1,
                 'max_abs_diff': float(abs(int(win.kept.sum()) - k90)), 'tolerance': 0.0,
                 'passed': int(win.kept.sum()) == k90, 'note': f'{int(win.kept.sum())} vs {k90}'})
    w90 = rng90[(rng90.target == 'GDSC1')].set_index('drug')
    record('d', 'common_window: lo/hi vs stored ranges', 'harmonization_gdsc2/ranges.csv (90)',
           np.r_[win.lo.values - w90.reindex(win.index).lo_common.values, win.hi.values - w90.reindex(win.index).hi_common.values],
           1e-3, 'ranges.csv stores 5 significant digits of values near -10 (rounding up to 5e-4)')
    rS, rT = rawS.groupby('pubchem_id')[['lo', 'hi']].median(), rawT.groupby('pubchem_id')[['lo', 'hi']].median()
    sh = rS.index.intersection(rT.index)
    lo66, hi66 = np.maximum(rS.loc[sh, 'lo'], rT.loc[sh, 'lo']), np.minimum(rS.loc[sh, 'hi'], rT.loc[sh, 'hi'])
    record('d', 'common_window: lo/hi vs 66/90 logic (full precision)', '66_harmonized_range.build_versions logic',
           np.r_[win.loc[sh, 'lo'].values - lo66.values, win.loc[sh, 'hi'].values - hi66.values], 1e-12, '')
    keep = win[win.kept]
    x = rawS[rawS.pubchem_id.isin(keep.index)].copy()
    x['v'] = hr.mean_viability(x, keep.loc[x.pubchem_id, 'lo'].values, keep.loc[x.pubchem_id, 'hi'].values)
    x = x.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).v.mean()
    ref = pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values, 'response': -x.v.values})
    m = a_b.merge(ref, on=['drug', 'cell'], suffixes=('', '_66'))
    record('d', "harmonize_pair('common_range', trapezoid) vs 66 logic (90 versions)", '66/90 GDSC2|HR-GDSC1|AUC',
           m.response - m.response_66, 1e-12, f'{len(m)} of {len(ref)} measurements matched')
    a_an, _, _ = box1.harmonize_pair(S, T, 'common_range', method='analytic')
    mm = a_an.merge(a_b, on=['drug', 'cell'])
    ROWS.append({'component': 'd', 'function': 'analytic vs 101-point trapezoid (common window)', 'reference': '(information)',
                 'n_compared': len(mm), 'max_abs_diff': float(np.max(np.abs(mm.response_x - mm.response_y))),
                 'tolerance': np.nan, 'passed': True,
                 'note': f'median {np.median(np.abs(mm.response_x - mm.response_y)):.1e}: the 101-point grid of 66 is accurate'})


def v_equivalence():
    E = pd.read_csv(O2 / 'equivalence' / 'equivalence_34.csv')
    ix = pd.read_csv(O2 / 'paired' / 'pairs.csv').set_index(['benchmark', 'pair'])
    keys = {'retained_fraction': 'retained_fraction', 'lo90': 'retained_fraction_lo90',
            'hi90': 'retained_fraction_hi90', 'lower_onesided': 'retained_fraction_lower95_onesided',
            'p_boot': 'p_boot_f_le_0.8', 'noninferior': 'noninferior_f_0.8'}
    diffs = {k: [] for k in keys}
    for _, r in E.iterrows():
        k = ix.loc[('leaderboard', r.pair)]
        models, meta, S = lib.load_paired(O2 / 'paired' / 'arrays' / f'leaderboard__{k.k}.npz')
        stat = 'selBOTH_asym' if r.selective_both.startswith('paper') else 'selBOTH_sym'
        gm = k.pool_main.split(';')
        t = box1.retained_fraction_test(lib.pooled(S, models, gm, stat, 'sum_w'), lib.pooled(S, models, gm, stat, 'sum_c'),
                                        margin=0.8)
        for a, b in keys.items():
            v = float(t[a]) - float(r[b])
            diffs[a].append(v / max(1.0, abs(float(r[b]))))
    for a in keys:
        record('e', f'retained_fraction_test: {a}', 'equivalence/equivalence_34.csv (83)', diffs[a], 1e-4,
               '34 versions x 2 selective-both variants; margin 0.8')


def v_holm():
    H = pd.read_csv(O2 / 'holm' / 'holm_25.csv')
    diffs = []
    n = 0
    for c in H.columns:
        if not c.startswith('p_used | '):
            continue
        v = c[len('p_used | '):]
        h = box1.holm(H[c].values)
        ref = H[f'holm | {v}'].values
        ok = np.isfinite(ref)
        diffs += list((h[ok] - ref[ok]) / np.maximum(ref[ok], 1e-300))
        n += 1
    record('f', 'holm', 'holm/holm_25.csv (86)', diffs, 1e-5, f'{n} variants; relative difference (6 significant digits)')


def v_biomarker():
    bm = lib.load_module('bm52', '52_biomarker_baselines.py')
    cat = bm.catalog_frame()
    feats = bm.cell_features(cat)
    DEC = lib.RES / 'decomposition'
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in bm.SCREENS}
    G = {s: lib.selbench.fitness_axis(Z[s]) for s in bm.SCREENS}
    ref = pd.read_csv(lib.RES / 'revision_ceiling' / '52_biomarker_c2_perdrug.csv', dtype={'compound': str})
    diffs = []
    for _, r in ref.iterrows():
        f = feats[r.biomarker]
        for s in bm.SCREENS:
            col = f'pc_{s}'
            if col not in r.index or not np.isfinite(r[col]) or r.compound not in Z[s].index:
                continue
            out = box1.biomarker_scores(f, Z[s], G[s], drugs=[r.compound], min_n=20)
            if len(out):
                diffs.append(out.selective.iloc[0] - r[col])
    record('g', 'biomarker_scores: selective r(f,z|g)', 'revision_ceiling/52_biomarker_c2_perdrug.csv (52)', diffs, 1e-9,
           'every catalogued biomarker x drug x discovery screen; sign-oriented features of 52')


def v_lineage():
    wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
    base = sorted(wp[(wp.setting == 'GDSC2_AUC') & (wp.panel == 'main')].model.unique())
    rt = lib.load_module('rt59', '59_retrain_score.py')
    oof, _, _ = rt.load_oof('GDSC2_AUC')
    extra = [m for m in ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting'] if m in oof and m not in base]
    S = lib.retrain_setup('GDSC2_AUC', models=base + extra)
    O = S['obs'][S['obs'].cell.isin(S['cells'])][['drug', 'cell', 'response']]
    Z, G, _, _ = lib.selbench.decompose(O)
    Pm = {m: P.reindex(columns=Z.columns) for m, P in S['P'].items()}
    tissue = pd.read_csv(lib.RES / 'revision_crosslab' / 'cell_tissue.csv').set_index('cellosaurus_id').tissue
    tv = tissue.reindex(Z.columns).fillna('unknown').values
    per = pd.read_csv(O2 / 'lineage' / 'per_drug_lineage.csv.gz', dtype={'drug': str})
    per = per[(per.benchmark == 'retrain') & (per.setting == 'GDSC2_AUC')].set_index(['model', 'drug'])
    d_sel, d_lin, d_ct, d_r2 = [], [], [], []
    for d in Z.index:
        z, g = Z.loc[d].values.astype(float), G.loc[d].values.astype(float)
        mk = np.isfinite(z) & np.isfinite(g)
        have = [m for m in Pm if d in Pm[m].index and np.isfinite(Pm[m].loc[d].values).any()]
        for m in have:
            mk &= np.isfinite(Pm[m].loc[d].values.astype(float))
        if mk.sum() < lib.MIN_CELLS:
            continue
        for m in ['EnsembleMF', 'RandomForest', 'PerDrugRidgeRaw']:
            if (m, d) not in per.index:
                continue
            r = box1.lineage_terms(Pm[m].loc[d].values.astype(float)[mk], z[mk], g[mk], tv[mk], min_n=lib.MIN_CELLS)
            ref = per.loc[(m, d)]
            d_sel.append(r['selective'] - ref.sel)
            d_lin.append(r['selective_lineage'] - ref.sel_lin)
            d_ct.append(r['covariate_term'] - ref.cov_term)
            d_r2.append(r['r2_z_covariates'] - ref.r2z_lin)
    note = 'GDSC2 AUC retraining, EnsembleMF, RandomForest, PerDrugRidgeRaw, every drug; 6 significant digits stored'
    record('h', 'lineage_terms: selective r(p,z|g)', 'lineage/per_drug_lineage.csv.gz (91)', d_sel, 1e-5, note)
    record('h', 'lineage_terms: lineage-adjusted r(p,z|g,T)', 'lineage/per_drug_lineage.csv.gz (91)', d_lin, 1e-5, note)
    record('h', 'lineage_terms: covariate term', 'lineage/per_drug_lineage.csv.gz (91)', d_ct, 1e-5, note)
    record('h', 'lineage_terms: R2_z|C', 'lineage/per_drug_lineage.csv.gz (91)', d_r2, 1e-5, note)


def v_active():
    DEC = lib.RES / 'decomposition'
    st = pd.read_csv(DEC / 'drug_screen_stats.csv', dtype={'compound': str})
    A = pd.read_csv(lib.RES / 'revision_axis' / 'axes_native.csv').set_index('depmap_id')
    var = pd.read_csv(lib.RES / 'revision_axis' / 'axis_variants.csv').set_index(['screen', 'variant'])
    long = pd.read_csv(lib.PROJECT / 'data' / 'processed' / 'discovery_long.csv.gz', usecols=['screen', 'compound',
                                                                                              'depmap_id', 'auc'])
    for s in ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']:
        Z = pd.read_parquet(DEC / f'z_{s}.parquet')
        auc = long[long.screen == s].pivot_table(index='compound', columns='depmap_id', values='auc', aggfunc='mean')
        auc = auc.reindex(index=Z.index, columns=Z.columns)
        act = box1.active_drugs(auc)
        ref = set(st[(st.screen == s) & (st.frac_auc_below_0p8 >= 0.1)].compound)
        sym = len(set(act) ^ ref)
        ROWS.append({'component': 'i', 'function': f'active_drugs ({s})', 'reference': 'decomposition/drug_screen_stats.csv (02/45)',
                     'n_compared': len(ref), 'max_abs_diff': float(sym), 'tolerance': 0.0, 'passed': sym == 0,
                     'note': f'{len(act)} active drugs; symmetric difference {sym}'})
        ax, _ = box1.active_axis(Z, act, min_drugs=10)
        record('i', f'active_axis ({s})', 'revision_axis/axes_native.csv (45)',
               ax.reindex(A.index).values - A[f'{s}|active'].values, 1e-9, 'cell-level active-drug axis')
        med, lo, hi = box1.axis_split_half(Z, act, n_splits=200, seed=1)
        record('i', f'axis_split_half ({s}, active)', 'revision_axis/axis_variants.csv (45)',
               [med - var.loc[(s, 'active'), 'split_half']], 0.005, 'random drug halves (different seed): within 0.005')


def tests():
    sel = lib.PROJECT / 'selbench'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_selbench.py',
                        'test_box1.py'], cwd=str(sel), capture_output=True, text=True, env=env)
    (OUT / 'tests.txt').write_text(r.stdout + r.stderr)
    last = [line for line in (r.stdout + r.stderr).splitlines() if 'passed' in line or 'failed' in line]
    return {'returncode': r.returncode, 'summary': last[-1] if last else ''}


def main():
    t0 = time.time()
    for f in [v_capture, v_selective_both, v_attenuation, v_harmonization, v_equivalence, v_holm, v_biomarker,
              v_lineage, v_active]:
        f()
    T = pd.DataFrame(ROWS)
    T.to_csv(OUT / 'validation.csv', index=False, float_format='%.3g')
    tr = tests()
    lib.write_json(OUT / 'summary.json', {'seconds': round(time.time() - t0), 'checks': int(len(T)),
                                          'passed': int(T.passed.sum()), 'failed': list(T[~T.passed].function),
                                          'unit_tests': tr})
    print(T[['component', 'function', 'n_compared', 'max_abs_diff', 'tolerance', 'passed']].to_string(index=False))
    print(tr)


if __name__ == '__main__':
    main()
