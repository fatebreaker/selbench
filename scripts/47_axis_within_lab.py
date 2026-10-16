"""Revision (round 1), fitness-axis workstream, part 3: within-laboratory controls (R2-3, R2-4) and within- versus
between-laboratory agreement with a nested variance decomposition (R2-2, R3-M4c).

A. GDSC1 two-site control. GDSC1 compounds were screened at one of two sites (screened_compounds_rel_8.4.csv,
   SCREENING_SITE: SANGER or MGH) on the same cell-line panel. From the GDSC1 release (fitted_dose_response_24Jul22)
   we rebuild, per site, the paper's matrices (compound identity and cell mapping exactly as in 01_harmonize_discovery;
   >=30 cells per drug, >=20 drugs per cell; rank-normal; sensitivity = -AUC, secondary -ln IC50) and compare the
   site axes (all site compounds; site-exclusive compounds; only the compounds screened at both sites; active compounds).
   Compounds screened at both sites (same molecule by normalized name) are compared drug by drug.
B. PRISM runs. Compounds screened in HTS002 and again in the MTS010 technical redo (secondary screen; STR-passed lines).
   Endpoints: the released drc AUC (MTS010 AUCs are capped at 1, HTS002's are not), the AUC capped at 1 in both
   runs, and a re-processed mean viability (mean over the 8 doses of 2^log2FC from the replicate-collapsed matrix;
   identical processing in both runs). Axes are built from the same compounds and cell lines in each run.
C. CTRPv1 versus CTRPv2 (same institute, different screen), from the harmonized DrEval files (script 46).
D. All pairwise axis agreements (scripts 45, 46 and this one) classified as: same measurements with different
   processing; same screen, different run/site; same institution, different screen; different institution. Nested
   variance decomposition of the standardized axis: moment estimator from disattenuated pair correlations, and REML
   (statsmodels MixedLM: random cell intercept, cell x laboratory and, with runs, cell x screen-within-laboratory
   components; residual = the most specific level) on cell lines shared by all screens.
Outputs (results/revision_axis/): gdsc1_sites_axes.csv, gdsc1_sites_perdrug.csv, gdsc1_sites_vs_other_screens.csv,
prism_runs_axes.csv, prism_runs_perdrug.csv, within_lab_controls.csv, all_axis_pairs.csv, pair_class_summary.csv,
variance_components.csv, summary_47.json.
"""
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import importlib  # noqa: E402
import itertools  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import PROCESSED, RAW, RESULTS, load_models, norm_name, write_json  # noqa: E402

H = importlib.import_module('45_axis_activity_lineage')
OUT = RESULTS / 'revision_axis'
SEED = 4701
N_BOOT_VC = 200          # cell-bootstrap replicates for REML on shared cells (half for unbalanced fits)
INSTITUTION = {'PRISM': 'Broad', 'CTRP': 'Broad', 'CTRPv2': 'Broad', 'CTRPv1': 'Broad', 'GDSC1': 'Sanger+MGH',
               'GDSC1_SANGER': 'Sanger', 'GDSC1_MGH': 'MGH', 'GDSC2': 'Sanger', 'gCSI': 'Genentech', 'gCSI_GRAOC': 'Genentech',
               'NCI60': 'NCI', 'CCLE': 'Novartis', 'FIMM': 'FIMM', 'GRAY': 'LBNL', 'PRISM_HTS002': 'Broad',
               'PRISM_MTS010': 'Broad', 'GDSC1_lnIC50': 'Sanger+MGH', 'GDSC2_lnIC50': 'Sanger'}
SCREEN_OF = {'CTRPv2': 'CTRP', 'GDSC1_SANGER': 'GDSC1', 'GDSC1_MGH': 'GDSC1', 'PRISM_HTS002': 'PRISM',
             'PRISM_MTS010': 'PRISM', 'gCSI_GRAOC': 'gCSI', 'GDSC1_lnIC50': 'GDSC1', 'GDSC2_lnIC50': 'GDSC2'}


def screen_of(x):
    return SCREEN_OF.get(x, x)


def lab_of(x):
    """Institution; GDSC1 (two sites) counts as Sanger for the GDSC1-GDSC2 pair."""
    inst = INSTITUTION[x]
    return 'Sanger' if inst == 'Sanger+MGH' else inst


def classify(a, b, processing_only=False):
    if processing_only or screen_of(a) == screen_of(b) and a == b:
        return 'P: same measurements, different processing'
    if screen_of(a) == screen_of(b):
        return 'R: same screen, different run/site'
    if lab_of(a) == lab_of(b):
        return 'L: same institution, different screen'
    return 'D: different institution'


# --------------------------------------------------------------------------------------------- A. GDSC1 two sites
def gdsc1_long():
    g = pd.read_csv(RAW / 'gdsc1' / 'GDSC1_fitted_dose_response_24Jul22.csv',
                    usecols=['DRUG_ID', 'DRUG_NAME', 'SANGER_MODEL_ID', 'COSMIC_ID', 'AUC', 'LN_IC50'])
    site = pd.read_csv(RAW / 'gdsc' / 'screened_compounds_rel_8.4.csv').drop_duplicates('DRUG_ID').set_index('DRUG_ID').SCREENING_SITE
    mp = pd.read_csv(PROCESSED / 'gdsc_compound_mapping.csv')
    mp = mp[mp.screen == 'GDSC1'].set_index('DRUG_ID').dpc
    merges = pd.read_csv(PROCESSED / 'compound_merges.csv').set_index('merged').into
    models = load_models()
    sm = models.dropna(subset=['SangerModelID']).drop_duplicates('SangerModelID').set_index('SangerModelID').ModelID
    cm = models.dropna(subset=['COSMICID']).assign(COSMICID=lambda d: d.COSMICID.astype(int))
    cm = cm.drop_duplicates('COSMICID').set_index('COSMICID').ModelID
    g['site'] = g.DRUG_ID.map(site)
    g['compound'] = g.DRUG_ID.map(mp).map(lambda d: merges.get(d, d))
    g['depmap_id'] = g.SANGER_MODEL_ID.map(sm)
    miss = g.depmap_id.isna()
    g.loc[miss, 'depmap_id'] = g.loc[miss, 'COSMIC_ID'].map(cm)
    g = g.dropna(subset=['depmap_id', 'AUC', 'compound', 'site'])
    g['sens'] = -g.AUC
    g['neg_ln_ic50'] = -g.LN_IC50
    g['name_key'] = g.DRUG_NAME.map(norm_name)
    return g


def site_analysis(rng):
    g = gdsc1_long()
    # check: averaging both sites per compound reproduces the paper's GDSC1 long table
    disc = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    disc = disc[disc.screen == 'GDSC1'].set_index(['compound', 'depmap_id']).sens
    mine = g.groupby(['compound', 'depmap_id']).sens.mean()
    common = mine.index.intersection(disc.index)
    check = {'pairs_mine': len(mine), 'pairs_paper': len(disc), 'common': len(common),
             'max_abs_diff': float((mine[common] - disc[common]).abs().max())}
    sites = g.groupby('compound').site.agg(lambda s: '|'.join(sorted(set(s))))
    dual = sites.index[sites == 'MGH|SANGER']
    names = g[g.compound.isin(dual)].groupby(['compound', 'site']).name_key.agg(lambda s: '|'.join(sorted(set(s)))).unstack()
    same_mol = names.index[names.MGH == names.SANGER].tolist()
    diff_mol = names.index[names.MGH != names.SANGER].tolist()
    exact = g[g.compound.isin(dual)].groupby(['compound', 'site']).DRUG_NAME.first().unstack()
    exact_same = exact.index[exact.MGH == exact.SANGER].tolist()
    site_info = {'drug_ids': {s: int(g[g.site == s].DRUG_ID.nunique()) for s in ['SANGER', 'MGH']},
            'compounds': {s: int(g[g.site == s].compound.nunique()) for s in ['SANGER', 'MGH']},
            'dual_site_compounds': len(dual), 'dual_same_molecule': len(same_mol), 'dual_same_exact_name': len(exact_same),
            'dual_different_molecules': {c: f'MGH={exact.loc[c, "MGH"]}; SANGER={exact.loc[c, "SANGER"]}' for c in diff_mol},
            'check_vs_paper_long': check}
    rows, perdrug, axes, Zs, Rs = [], [], {}, {}, {}
    for value, label in [('sens', 'AUC'), ('neg_ln_ic50', 'lnIC50')]:
        for s in ['SANGER', 'MGH']:
            long = g[g.site == s].groupby(['compound', 'depmap_id'], as_index=False)[value].mean()
            M, Z = H.build_z(long, value, min_unique=1, min_drugs_per_cell=20)
            Zs[(label, s)] = Z
            Rs[(label, s)] = H.residuals(Z, H.loo_axis(Z))
            if label == 'AUC':
                act = M.index[((-M) < 0.8).sum(axis=1) / M.notna().sum(axis=1) >= 0.1]
                Zs[('active', s)] = act
        for s in ['SANGER', 'MGH']:
            Z = Zs[(label, s)]
            variants = {'all': None, 'site_exclusive': [d for d in Z.index if d not in set(dual)],
                        'dual_site_same_molecule': [d for d in Z.index if d in set(same_mol)]}
            if label == 'AUC':
                variants['active'] = list(Zs[('active', s)])
            for v, drugs in variants.items():
                n = Z.shape[0] if drugs is None else len(drugs)
                axes[(label, s, v)] = (H.axis_of(Z, drugs, None if drugs is None else min(10, max(2, n // 2))),
                                       H.split_half(Z, rng, drugs), n)
        cells = Zs[(label, 'SANGER')].columns.intersection(Zs[(label, 'MGH')].columns)
        for v in ['all', 'site_exclusive', 'dual_site_same_molecule', 'active']:
            if (label, 'SANGER', v) not in axes:
                continue
            (xa, ra, na), (xb, rb, nb) = axes[(label, 'SANGER', v)], axes[(label, 'MGH', v)]
            x, y = xa.reindex(cells), xb.reindex(cells)
            r = H.corr(x, y)
            lo, hi = H.boot_ci(x, y, rng)
            rows.append({'endpoint': label, 'variant': v, 'drugs_sanger': na, 'drugs_mgh': nb,
                         'cells': int((x.notna() & y.notna()).sum()), 'split_half_sanger': ra[0], 'split_half_mgh': rb[0],
                         'r': r, 'ci_lo': lo, 'ci_hi': hi, 'disattenuated': r / np.sqrt(ra[0] * rb[0])})
        # per-drug agreement of compounds screened at both sites
        Za, Zb = Zs[(label, 'SANGER')], Zs[(label, 'MGH')]
        raw_a = g[g.site == 'SANGER'].groupby(['compound', 'depmap_id'])[value].mean().unstack()
        raw_b = g[g.site == 'MGH'].groupby(['compound', 'depmap_id'])[value].mean().unstack()
        for d in sorted(dual):
            if d not in Za.index or d not in Zb.index:
                continue
            za, zb = Za.loc[d, cells].values, Zb.loc[d, cells].values
            m = np.isfinite(za) & np.isfinite(zb)
            ua, ub = raw_a.loc[d].reindex(cells).values, raw_b.loc[d].reindex(cells).values
            mm = np.isfinite(ua) & np.isfinite(ub)
            perdrug.append({'endpoint': label, 'compound': d, 'name_sanger': exact.loc[d, 'SANGER'], 'name_mgh': exact.loc[d, 'MGH'],
                            'same_molecule': d in same_mol, 'cells': int(m.sum()), 'pearson_z': H.corr(za, zb),
                            'pearson_raw': H.corr(ua, ub), 'spearman_raw': float(spearmanr(ua[mm], ub[mm]).correlation),
                            'selective': H.corr(Rs[(label, 'SANGER')].loc[d, cells].values, Rs[(label, 'MGH')].loc[d, cells].values)})
    site_axes = pd.DataFrame(rows)
    per = pd.DataFrame(perdrug)
    # site axes against other screens' axes (identical cells within each comparison)
    nat = pd.read_csv(OUT / 'axes_native.csv', index_col=0)
    dre = pd.read_csv(OUT / 'dreval_axes_depmap.csv', index_col=0)
    others = {k: nat[k] for k in ['GDSC2|all', 'GDSC2|active', 'CTRP|all', 'PRISM|all', 'gCSI|all', 'CCLE|all', 'GDSC2_lnIC50|all']}
    others.update({f'DrEval {k}': dre[k] for k in ['GDSC2|CC_AUC|all', 'GDSC2|CC_lnIC50|all', 'CTRPv2|CC_AUC|all', 'CTRPv2|CC_lnIC50|all',
                                                  'CTRPv1|CC_AUC|all', 'CCLE|CC_AUC|all']})
    vrows = []
    for v in ['all', 'active']:
        xs, xm = axes[('AUC', 'SANGER', v)][0], axes[('AUC', 'MGH', v)][0]
        for k, o in others.items():
            c = xs.dropna().index.intersection(xm.dropna().index).intersection(o.dropna().index)
            if len(c) < 30:
                continue
            rs, rm = H.corr(xs[c], o[c]), H.corr(xm[c], o[c])
            bd = []
            for _ in range(500):
                i = rng.integers(0, len(c), len(c))
                bd.append(H.corr(xs[c].values[i], o[c].values[i]) - H.corr(xm[c].values[i], o[c].values[i]))
            vrows.append({'site_variant': v, 'other_axis': k, 'cells': len(c), 'r_sanger_site': rs, 'r_mgh_site': rm,
                          'diff': rs - rm, 'diff_ci_lo': float(np.nanpercentile(bd, 2.5)), 'diff_ci_hi': float(np.nanpercentile(bd, 97.5))})
    # lineage and growth-pattern structure of each site axis (as in 45, lineage_by_screen)
    lineage, growth = H.cell_annotations()
    for lab in ['AUC', 'lnIC50']:
        for s in ['SANGER', 'MGH']:
            ax = axes[(lab, s, 'all')][0].dropna()
            lin = lineage.reindex(ax.index).fillna('Unknown')
            gp = growth.reindex(ax.index).fillna('Unknown')
            susp = gp == 'Suspension'
            site_info.setdefault('site_axis_structure', []).append({
                'endpoint': lab, 'site': s, 'cells': len(ax),
                'lineage_r2': H.corr(ax, ax.groupby(lin).transform('mean'), min_n=5) ** 2,
                'susp_minus_rest_sd_units': float((ax[susp].mean() - ax[~susp].mean()) / ax.std()),
                'n_suspension': int(susp.sum())})
    for lab in ['AUC', 'lnIC50']:
        xs, xm = axes[(lab, 'SANGER', 'all')][0], axes[(lab, 'MGH', 'all')][0]
        c = xs.dropna().index.intersection(xm.dropna().index)
        D = pd.get_dummies(lineage.reindex(c).fillna('Unknown'), drop_first=True).values.astype(float)
        site_info.setdefault('site_axes_lineage_partial', {})[lab] = {
            'r': H.corr(xs[c], xm[c]), 'r_partial_lineage': H.partial_corr(xs[c].values, xm[c].values, D), 'cells': len(c)}
    vs = pd.DataFrame(vrows)
    site_axis_series = {f'GDSC1_SANGER|{lab}|{v}': axes[(lab, 'SANGER', v)][0] for lab in ['AUC', 'lnIC50'] for v in ['all']}
    site_axis_series.update({f'GDSC1_MGH|{lab}|{v}': axes[(lab, 'MGH', v)][0] for lab in ['AUC', 'lnIC50'] for v in ['all']})
    site_rel = {f'GDSC1_{s}|{lab}': axes[(lab, s, 'all')][1][0] for lab in ['AUC', 'lnIC50'] for s in ['SANGER', 'MGH']}
    return site_axes, per, vs, site_info, site_axis_series, site_rel


# --------------------------------------------------------------------------------------------- B. PRISM runs
def prism_runs(rng):
    p = pd.read_csv(RAW / 'prism' / 'secondary-screen-dose-response-curve-parameters.csv',
                    usecols=['broad_id', 'depmap_id', 'screen_id', 'auc', 'name', 'passed_str_profiling'])
    p = p[p.passed_str_profiling.astype(str).str.upper() == 'TRUE'].copy()
    p['short'] = p.broad_id.str[:13]
    runs = p.groupby('screen_id').short.apply(set)
    both = runs['HTS002'] & runs['MTS010']
    p = p[p.short.isin(both) & p.screen_id.isin(['HTS002', 'MTS010'])]
    capped_frac = {r: float((p[p.screen_id == r].auc >= 1).mean()) for r in ['HTS002', 'MTS010']}
    above1 = {r: float((p[p.screen_id == r].auc > 1).mean()) for r in ['HTS002', 'MTS010']}
    long = {}
    for r in ['HTS002', 'MTS010']:
        q = p[p.screen_id == r]
        long[('drc_AUC', r)] = q.groupby(['short', 'depmap_id'], as_index=False).auc.mean().assign(sens=lambda t: -t.auc)
        long[('drc_AUC_capped', r)] = q.assign(auc=q.auc.clip(upper=1)).groupby(['short', 'depmap_id'], as_index=False).auc.mean().assign(sens=lambda t: -t.auc)
    # re-processed mean viability from the replicate-collapsed log2 fold changes (identical processing in both runs)
    ti = pd.read_csv(RAW / 'prism' / 'secondary-screen-replicate-collapsed-treatment-info.csv')
    ti['short'] = ti.broad_id.str[:13]
    ti = ti[ti.short.isin(both) & ti.screen_id.isin(['HTS002', 'MTS010'])]
    ci = pd.read_csv(RAW / 'prism' / 'secondary-screen-cell-line-info.csv')
    ok_rows = ci[ci.passed_str_profiling.astype(str).str.upper() == 'TRUE'].set_index('row_name').depmap_id
    keep_cols = set(ti.column_name)
    lfc = pd.read_csv(RAW / 'prism' / 'secondary-screen-replicate-collapsed-logfold-change.csv', index_col=0,
                      usecols=lambda c: c in ('', 'Unnamed: 0') or c in keep_cols)
    lfc = lfc.loc[lfc.index.intersection(ok_rows.index)]
    lfc.index = lfc.index.map(ok_rows)
    via = (2.0 ** lfc).T
    via['short'] = via.index.map(ti.set_index('column_name').short)
    via['run'] = via.index.map(ti.set_index('column_name').screen_id)
    for r in ['HTS002', 'MTS010']:
        sub = via[via.run == r].drop(columns='run')
        n = sub.groupby('short').count()
        mv = sub.groupby('short').mean()
        mv = mv.where(n >= 6)
        st = mv.stack().rename('mean_viability').reset_index().rename(columns={'level_1': 'depmap_id'})
        long[('mean_viability', r)] = st.assign(sens=lambda t: -t.mean_viability)
    rows, per = [], []
    run_axes = {}
    for ep in ['drc_AUC', 'drc_AUC_capped', 'mean_viability']:
        Ma, Za = H.build_z(long[(ep, 'HTS002')], 'sens', drug='short', min_unique=1)
        Mb, Zb = H.build_z(long[(ep, 'MTS010')], 'sens', drug='short', min_unique=1)
        d = Za.index.intersection(Zb.index)
        c = Za.columns.intersection(Zb.columns)
        Za, Zb = Za.loc[d, c], Zb.loc[d, c]
        Ra, Rb = H.residuals(Za, H.loo_axis(Za)), H.residuals(Zb, H.loo_axis(Zb))
        fa, fb = Za.mean(), Zb.mean()
        rel_a, rel_b = H.split_half(Za, rng), H.split_half(Zb, rng)
        r = H.corr(fa, fb)
        lo, hi = H.boot_ci(fa, fb, rng)
        raw = [H.corr(Za.loc[k].values, Zb.loc[k].values) for k in d]
        sel = [H.corr(Ra.loc[k].values, Rb.loc[k].values) for k in d]
        r2a = H.fitness_r2(Za, H.loo_axis(Za)).median()
        r2b = H.fitness_r2(Zb, H.loo_axis(Zb)).median()
        tiesb = float(np.mean([(Mb.loc[k].dropna().duplicated(keep=False)).mean() for k in d]))
        rows.append({'endpoint': ep, 'compounds': len(d), 'cells': len(c), 'split_half_HTS002': rel_a[0], 'split_half_MTS010': rel_b[0],
                     'r': r, 'ci_lo': lo, 'ci_hi': hi, 'disattenuated': r / np.sqrt(rel_a[0] * rel_b[0]),
                     'median_perdrug_raw': float(np.nanmedian(raw)), 'q25_perdrug_raw': float(np.nanpercentile(raw, 25)),
                     'q75_perdrug_raw': float(np.nanpercentile(raw, 75)), 'median_perdrug_selective': float(np.nanmedian(sel)),
                     'median_fitness_r2_HTS002': float(r2a), 'median_fitness_r2_MTS010': float(r2b),
                     'mean_frac_tied_values_MTS010': tiesb})
        for k, a, b in zip(d, raw, sel):
            per.append({'endpoint': ep, 'compound_short': k, 'raw': a, 'selective': b})
        run_axes[ep] = (fa, fb, rel_a[0], rel_b[0])
    names = p.drop_duplicates('short').set_index('short').name
    per = pd.DataFrame(per)
    per['name'] = per.compound_short.map(names)
    # HTS002 vs MTS006 (20 shared compounds): per-compound agreement only
    q = pd.read_csv(RAW / 'prism' / 'secondary-screen-dose-response-curve-parameters.csv',
                    usecols=['broad_id', 'depmap_id', 'screen_id', 'auc', 'passed_str_profiling'])
    q = q[q.passed_str_profiling.astype(str).str.upper() == 'TRUE'].copy()
    q['short'] = q.broad_id.str[:13]
    b6 = set(q[q.screen_id == 'HTS002'].short) & set(q[q.screen_id == 'MTS006'].short)
    m6 = []
    for k in b6:
        x = q[(q.short == k) & (q.screen_id == 'HTS002')].groupby('depmap_id').auc.mean()
        y = q[(q.short == k) & (q.screen_id == 'MTS006')].groupby('depmap_id').auc.mean()
        cc = x.index.intersection(y.index)
        if len(cc) >= 30:
            m6.append(spearmanr(x[cc], y[cc]).correlation)
    info = {'compounds_both_runs': len(both), 'frac_auc_ge1': capped_frac, 'frac_auc_gt1': above1,
            'hts002_vs_mts006': {'compounds': len(m6), 'median_spearman': float(np.median(m6)) if m6 else None}}
    return pd.DataFrame(rows), per, info, run_axes


# --------------------------------------------------------------------------------------------- D. assembly
def assemble(site_axes, prism_tab):
    rows = []
    a45 = pd.read_csv(OUT / 'axis_pairs_activity.csv')
    for r in a45.itertuples():
        if not np.isfinite(getattr(r, 'r', np.nan)):
            continue
        rows.append({'source': '45 discovery', 'endpoint': 'AUC (provider; native)', 'endpoint_family': 'AUC-type',
                     'harmonized_fitter': False, 'a': r.a, 'b': r.b, 'variant': r.variant, 'cells': r.cells,
                     'drugs_a': r.drugs_a, 'drugs_b': r.drugs_b, 'r': r.r, 'disattenuated': r.disattenuated})
    l45 = pd.read_csv(OUT / 'axis_pairs_lineage.csv')
    disc = {'PRISM|all', 'CTRP|all', 'GDSC1|all', 'GDSC2|all'}
    fam = {'NCI60': 'GI50', 'gCSI_GRAOC': 'GR-AOC'}
    for r in l45.itertuples():
        if r.a in disc and r.b in disc:
            continue
        a, b = r.a.split('|')[0], r.b.split('|')[0]
        fams = {('lnIC50-type' if 'lnIC50' in x else fam.get(x, 'AUC-type')) for x in (a, b)}
        rows.append({'source': '45 native', 'endpoint': 'native', 'endpoint_family': '/'.join(sorted(fams)),
                     'harmonized_fitter': False, 'a': a, 'b': b, 'variant': 'all', 'cells': r.cells,
                     'r': r.r, 'disattenuated': r.disattenuated})
    d46 = pd.read_csv(OUT / 'dreval_axis_pairs.csv')
    for r in d46.itertuples():
        if not np.isfinite(getattr(r, 'r', np.nan)):
            continue
        rows.append({'source': '46 DrEval', 'endpoint': r.endpoint, 'endpoint_family': 'lnIC50-type' if 'lnIC50' in r.endpoint else 'AUC-type',
                     'harmonized_fitter': r.endpoint.startswith('CC_'), 'a': r.a, 'b': r.b, 'variant': r.variant,
                     'cells': r.cells, 'drugs_a': r.drugs_a, 'drugs_b': r.drugs_b, 'r': r.r, 'disattenuated': r.disattenuated})
    w46 = pd.read_csv(OUT / 'dreval_within_screen.csv')
    for r in w46[w46.variant == 'all'].itertuples():
        rows.append({'source': '46 DrEval within screen', 'endpoint': f'{r.endpoint_1} vs {r.endpoint_2}', 'endpoint_family': 'mixed',
                     'harmonized_fitter': False, 'a': r.dataset, 'b': r.dataset, 'variant': 'all', 'cells': r.cells, 'r': r.r,
                     'disattenuated': np.nan, 'processing_only': True})
    for r in site_axes.itertuples():
        rows.append({'source': '47 GDSC1 sites', 'endpoint': f'{r.endpoint} (Sanger fits, one pipeline)',
                     'endpoint_family': 'lnIC50-type' if r.endpoint == 'lnIC50' else 'AUC-type', 'harmonized_fitter': True,
                     'a': 'GDSC1_SANGER', 'b': 'GDSC1_MGH', 'variant': r.variant, 'cells': r.cells, 'drugs_a': r.drugs_sanger,
                     'drugs_b': r.drugs_mgh, 'r': r.r, 'disattenuated': r.disattenuated})
    for r in prism_tab.itertuples():
        rows.append({'source': '47 PRISM runs', 'endpoint': r.endpoint, 'endpoint_family': 'AUC-type',
                     'harmonized_fitter': r.endpoint != 'drc_AUC', 'a': 'PRISM_HTS002', 'b': 'PRISM_MTS010',
                     'variant': 'shared_all', 'cells': r.cells, 'drugs_a': r.compounds, 'drugs_b': r.compounds, 'r': r.r,
                     'disattenuated': r.disattenuated})
    A = pd.DataFrame(rows)
    A['processing_only'] = A.get('processing_only', False).fillna(False).astype(bool)
    A['pair_class'] = [classify(a, b, po) for a, b, po in zip(A.a, A.b, A.processing_only)]
    A['institution_a'] = A.a.map(lambda x: INSTITUTION.get(x, '?'))
    A['institution_b'] = A.b.map(lambda x: INSTITUTION.get(x, '?'))
    small = {'NCI60', 'FIMM', 'GRAY'}
    A['small_panel'] = A.a.isin(small) | A.b.isin(small)
    return A


def class_summary(A):
    rows = []
    views = {
        'harmonized endpoint, all-drug axes': (A.variant == 'all') & (A.harmonized_fitter | A.source.str.startswith('47')) |
                                              ((A.source == '47 PRISM runs') & (A.endpoint == 'mean_viability')),
        'harmonized endpoint, shared-drug axes': A.variant.isin(['shared_all']) & A.harmonized_fitter,
        'harmonized endpoint, active-drug axes': A.variant.isin(['active']) & A.harmonized_fitter,
        'native/provider endpoint, all-drug axes (panel >= 100 lines)': (A.variant == 'all') & ~A.harmonized_fitter & ~A.small_panel,
    }
    for vname, m in views.items():
        sub = A[m & (A.pair_class != 'P: same measurements, different processing')]
        sub = sub.drop_duplicates(['a', 'b', 'endpoint', 'variant'])
        for (fam, cls), g in sub.groupby(['endpoint_family', 'pair_class']):
            dd = g.disattenuated.dropna()
            rows.append({'view': vname, 'endpoint_family': fam, 'pair_class': cls, 'pairs': len(g),
                         'median_r': float(g.r.median()), 'median_disattenuated': float(dd.median()) if len(dd) else np.nan,
                         'min_disattenuated': float(dd.min()) if len(dd) else np.nan,
                         'max_disattenuated': float(dd.max()) if len(dd) else np.nan,
                         'pairs_listed': '; '.join(f'{a}-{b} [{e}] {d:.2f}' for a, b, e, d in zip(g.a, g.b, g.endpoint, g.disattenuated.fillna(np.nan)))})
    P = A[A.pair_class == 'P: same measurements, different processing']
    for _, g in P.groupby('source'):
        rows.append({'view': 'same measurements, different processing (raw r; errors shared)', 'endpoint_family': 'mixed',
                     'pair_class': 'P: same measurements, different processing', 'pairs': len(g), 'median_r': float(g.r.median()),
                     'min_r': float(g.r.min()), 'max_r': float(g.r.max()),
                     'pairs_listed': '; '.join(f'{a} [{e}] {r:.2f}' for a, e, r in zip(g.a, g.endpoint, g.r))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- variance components
def reml_components(df, levels):
    """REML fit of y ~ run fixed effects + cell random intercept + variance components for `levels` within cell.
    Returns dict of variance estimates (cell, each level, residual)."""
    import statsmodels.formula.api as smf
    vc = {lv: f'0 + C({lv})' for lv in levels}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = smf.mixedlm('y ~ 0 + C(unit)', df, groups='cell', re_formula='1', vc_formula=vc)
        f = m.fit(reml=True, method=['lbfgs', 'powell'])
    out = {'cell': float(f.cov_re.iloc[0, 0]), 'residual': float(f.scale)}
    for name, v in zip(m.exog_vc.names, np.atleast_1d(f.vcomp)):
        out[name] = float(v)
    return out


def _boot_vc(args):
    """Cell bootstrap: resample cell lines with replacement (each draw becomes its own group)."""
    df, levels, seed, n = args
    rng = np.random.default_rng(seed)
    groups = df.groupby('cell').indices
    cells = list(groups)
    res = []
    for _ in range(n):
        pick = rng.integers(0, len(cells), len(cells))
        rows = [groups[cells[i]] for i in pick]
        d = df.iloc[np.concatenate(rows)].reset_index(drop=True)   # unique row labels (statsmodels builds vc by label)
        d['cell'] = np.repeat(np.arange(len(pick)), [len(r) for r in rows])
        try:
            res.append(reml_components(d, levels))
        except Exception:
            continue
    return res


def variance_decomposition(axes, units, levels, label, n_boot=None, balanced=True):
    """axes: dict unit -> Series (cell axis). units: dict unit -> {'lab':..., 'screen':...}. balanced: cells shared by
    all units, each axis standardized over those cells; otherwise every cell measured in >=2 units, each axis
    standardized over its own cells (unbalanced REML)."""
    S = pd.DataFrame({u: axes[u] for u in units})
    if balanced:
        S = S.dropna()
    else:
        S = S[S.notna().sum(axis=1) >= 2]
    S = (S - S.mean()) / S.std()
    S.index.name, S.columns.name = 'cell', 'unit'
    df = S.stack().rename('y').reset_index()
    for lv in levels + ['screen']:
        df[lv] = df.unit.map(lambda u: units[u].get(lv, u))
    est = reml_components(df, levels)
    n_boot = n_boot or (N_BOOT_VC if balanced else N_BOOT_VC // 2)
    jobs = [(df, levels, SEED + 100 + k, n_boot // 4) for k in range(4)]
    with ProcessPoolExecutor(4) as ex:
        boots = [b for part in ex.map(_boot_vc, jobs) for b in part]
    B = pd.DataFrame(boots)
    rows = []
    tot = sum(est.values())
    for k, v in est.items():
        rows.append({'model': label, 'component': k, 'variance': v, 'fraction': v / tot,
                     'ci_lo': float(B[k].quantile(0.025)) if k in B else np.nan,
                     'ci_hi': float(B[k].quantile(0.975)) if k in B else np.nan,
                     'cells': S.shape[0], 'observations': len(df), 'units': '|'.join(units), 'n_boot_ok': len(B)})
    return rows


def moment_components(A, label, mask):
    """Moment estimator from disattenuated all-drug axis correlations (pairs on their own shared cells)."""
    sub = A[mask].drop_duplicates(['a', 'b', 'endpoint', 'variant'])
    m = sub.groupby('pair_class').disattenuated.mean()
    D = m.get('D: different institution', np.nan)
    L = m.get('L: same institution, different screen', np.nan)
    R = m.get('R: same screen, different run/site', np.nan)
    rows = [{'model': label, 'component': 'cell (shared by all screens)', 'variance': D},
            {'model': label, 'component': 'cell x institution', 'variance': L - D if np.isfinite(L) else np.nan}]
    if np.isfinite(R):
        rows += [{'model': label, 'component': 'cell x screen within institution', 'variance': R - (L if np.isfinite(L) else D)},
                 {'model': label, 'component': 'cell x run/site (reliable, specific)', 'variance': 1 - R}]
    else:
        rows += [{'model': label, 'component': 'cell x screen within institution (incl. run/site)', 'variance': 1 - (L if np.isfinite(L) else D)}]
    for r in rows:
        r['pairs_D'] = int((sub.pair_class == 'D: different institution').sum())
        r['pairs_L'] = int((sub.pair_class == 'L: same institution, different screen').sum())
        r['pairs_R'] = int((sub.pair_class == 'R: same screen, different run/site').sum())
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    site_axes, site_per, site_vs, site_info, site_series, site_rel = site_analysis(rng)
    site_axes.to_csv(OUT / 'gdsc1_sites_axes.csv', index=False)
    site_per.to_csv(OUT / 'gdsc1_sites_perdrug.csv', index=False)
    site_vs.to_csv(OUT / 'gdsc1_sites_vs_other_screens.csv', index=False)
    pd.set_option('display.width', 250)
    print(site_info, flush=True)
    print(site_axes.round(3).to_string(index=False), flush=True)
    print(site_per.round(3).to_string(index=False), flush=True)
    print(site_vs.round(3).to_string(index=False), flush=True)

    prism_tab, prism_per, prism_info, run_axes = prism_runs(rng)
    prism_tab.to_csv(OUT / 'prism_runs_axes.csv', index=False)
    prism_per.to_csv(OUT / 'prism_runs_perdrug.csv', index=False)
    print(prism_info, flush=True)
    print(prism_tab.round(3).to_string(index=False), flush=True)

    # ---------------- within-laboratory control table ----------------
    d46 = pd.read_csv(OUT / 'dreval_axis_pairs.csv')
    p46 = pd.read_csv(OUT / 'dreval_perdrug_pairs.csv')
    wl = []
    for r in site_axes.itertuples():
        pdx = site_per[(site_per.endpoint == r.endpoint) & site_per.same_molecule]
        wl.append({'control': 'GDSC1 Sanger site vs MGH site', 'endpoint': r.endpoint, 'variant': r.variant, 'cells': r.cells,
                   'drugs': f'{r.drugs_sanger}/{r.drugs_mgh}', 'r': r.r, 'disattenuated': r.disattenuated,
                   'perdrug_n': len(pdx), 'perdrug_median_raw': float(pdx.pearson_z.median()),
                   'perdrug_median_selective': float(pdx.selective.median())})
    for r in prism_tab.itertuples():
        wl.append({'control': 'PRISM HTS002 vs MTS010', 'endpoint': r.endpoint, 'variant': 'shared_all', 'cells': r.cells,
                   'drugs': f'{r.compounds}/{r.compounds}', 'r': r.r, 'disattenuated': r.disattenuated, 'perdrug_n': r.compounds,
                   'perdrug_median_raw': r.median_perdrug_raw, 'perdrug_median_selective': r.median_perdrug_selective})
    for (a, b, lbl) in [('CTRPv2', 'CTRPv1', 'CTRPv2 vs CTRPv1 (DrEval)'), ('GDSC1', 'GDSC2', 'GDSC1 vs GDSC2 (DrEval)'),
                        ('CTRPv2', 'GDSC2', 'CTRPv2 vs GDSC2 (DrEval; different institution)'),
                        ('CTRPv2', 'GDSC1', 'CTRPv2 vs GDSC1 (DrEval; different institution)')]:
        for ep in ['CC_lnIC50', 'CC_AUC', 'prov_AUC']:
            for v in ['all', 'shared_all', 'shared_active_both']:
                q = d46[(d46.a == a) & (d46.b == b) & (d46.endpoint == ep) & (d46.variant == v)]
                if q.empty:
                    continue
                q = q.iloc[0]
                pp = p46[(p46.a == a) & (p46.b == b) & (p46.endpoint == ep)]
                wl.append({'control': lbl, 'endpoint': ep, 'variant': v, 'cells': q.cells, 'drugs': f'{q.drugs_a}/{q.drugs_b}',
                           'r': q.r, 'disattenuated': q.disattenuated, 'perdrug_n': len(pp),
                           'perdrug_median_raw': float(pp.raw.median()), 'perdrug_median_selective': float(pp.selective.median())})
    wl = pd.DataFrame(wl)
    wl.to_csv(OUT / 'within_lab_controls.csv', index=False)
    print(wl.round(3).to_string(index=False), flush=True)

    # ---------------- D. assembly and classification ----------------
    A = assemble(site_axes, prism_tab)
    A.to_csv(OUT / 'all_axis_pairs.csv', index=False)
    CS = class_summary(A)
    CS.to_csv(OUT / 'pair_class_summary.csv', index=False)
    print(CS.drop(columns=['pairs_listed']).round(3).to_string(index=False), flush=True)

    # ---------------- nested variance decomposition ----------------
    vrows = []
    harm = A.harmonized_fitter & (A.variant == 'all')
    for ep in ['CC_lnIC50', 'CC_AUC']:
        vrows += moment_components(A, f'moments, DrEval {ep} (5 screens, 3 institutions)', harm & (A.endpoint == ep))
    # AUC family with run/site level: DrEval CC_AUC pairs + GDSC1 sites (Sanger AUC) + PRISM runs (mean viability)
    auc_mask = (harm & (A.endpoint == 'CC_AUC')) | ((A.source == '47 GDSC1 sites') & (A.endpoint.str.startswith('AUC')) & (A.variant == 'all')) | \
               ((A.source == '47 PRISM runs') & (A.endpoint == 'mean_viability'))
    vrows += moment_components(A, 'moments, AUC-type: DrEval CC_AUC + GDSC1 sites + PRISM runs', auc_mask)
    nat_mask = (A.source.isin(['45 discovery', '45 native'])) & (A.variant == 'all') & (A.endpoint_family == 'AUC-type') & ~A.small_panel
    nat_mask = nat_mask | ((A.source == '47 GDSC1 sites') & (A.endpoint.str.startswith('AUC')) & (A.variant == 'all')) | \
        ((A.source == '47 PRISM runs') & (A.endpoint == 'drc_AUC'))
    vrows += moment_components(A, 'moments, native AUC-type screens (PRISM, CTRP, GDSC1, GDSC2, gCSI, CCLE) + runs/sites', nat_mask)
    dre = pd.read_csv(OUT / 'dreval_axes_cellosaurus.csv', index_col=0)
    for ep in ['CC_lnIC50', 'CC_AUC']:
        for names, tag, bal in [(['CTRPv2', 'CTRPv1', 'GDSC1', 'GDSC2', 'CCLE'], '5 screens, shared cells', True),
                                (['CTRPv2', 'GDSC1', 'GDSC2', 'CCLE'], '4 screens (no CTRPv1), shared cells', True),
                                (['CTRPv2', 'CTRPv1', 'GDSC1', 'GDSC2', 'CCLE'], '5 screens, unbalanced (cells in >=2 screens)', False)]:
            ax = {n: dre[f'{n}|{ep}|all'] for n in names}
            units = {n: {'lab': lab_of(n)} for n in ax}
            vrows += variance_decomposition(ax, units, ['lab'], f'REML, DrEval {ep}, {tag}: cell + cell x institution + residual',
                                            balanced=bal)
            print('REML', ep, tag, 'done', flush=True)
    nat = pd.read_csv(OUT / 'axes_native.csv', index_col=0)
    ax = {'PRISM_HTS002': run_axes['drc_AUC'][0], 'PRISM_MTS010': run_axes['drc_AUC'][1], 'CTRP': nat['CTRP|all'],
          'GDSC1_SANGER': site_series['GDSC1_SANGER|AUC|all'], 'GDSC1_MGH': site_series['GDSC1_MGH|AUC|all'],
          'GDSC2': nat['GDSC2|all'], 'gCSI': nat['gCSI|all'], 'CCLE': nat['CCLE|all']}
    units = {'PRISM_HTS002': {'lab': 'Broad', 'screen': 'PRISM'}, 'PRISM_MTS010': {'lab': 'Broad', 'screen': 'PRISM'},
             'CTRP': {'lab': 'Broad', 'screen': 'CTRP'}, 'GDSC1_SANGER': {'lab': 'GDSC', 'screen': 'GDSC1'},
             'GDSC1_MGH': {'lab': 'GDSC', 'screen': 'GDSC1'}, 'GDSC2': {'lab': 'GDSC', 'screen': 'GDSC2'},
             'gCSI': {'lab': 'Genentech', 'screen': 'gCSI'}, 'CCLE': {'lab': 'Novartis', 'screen': 'CCLE'}}
    vrows += variance_decomposition(ax, units, ['lab', 'screen'],
                                    'REML, native AUC-type with runs/sites, shared cells: cell + cell x lab + cell x screen(lab) + residual (run/site)')
    vrows += variance_decomposition(ax, units, ['lab', 'screen'],
                                    'REML, native AUC-type with runs/sites, unbalanced: cell + cell x lab + cell x screen(lab) + residual (run/site)',
                                    balanced=False)
    print('REML native done', flush=True)
    VC = pd.DataFrame(vrows)
    VC.to_csv(OUT / 'variance_components.csv', index=False)
    print(VC.round(3).to_string(index=False), flush=True)

    write_json(OUT / 'summary_47.json', {
        'gdsc1_sites': {'info': site_info, 'axes': site_axes.to_dict(orient='records'),
                        'perdrug': site_per.to_dict(orient='records'), 'vs_other_screens': site_vs.to_dict(orient='records'),
                        'site_axis_split_half': site_rel},
        'prism_runs': {'info': prism_info, 'axes': prism_tab.to_dict(orient='records')},
        'within_lab_controls': wl.to_dict(orient='records'),
        'class_summary': CS.to_dict(orient='records'),
        'variance_components': VC.to_dict(orient='records'),
        'settings': {'seed': SEED, 'n_boot_vc': N_BOOT_VC}})


if __name__ == '__main__':
    main()
