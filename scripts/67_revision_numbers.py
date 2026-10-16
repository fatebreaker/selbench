"""Numbers quoted in the revised manuscript that come from the round-1 revision analyses (results/revision_*).

collect() returns a dict that 14_collect_numbers.py stores under paper_numbers.json['revision'], so every number in the
revised text traces to one machine-readable file. Read-only.
"""
import json

import numpy as np
import pandas as pd

from common import RESULTS

AX, XL, TR, CE = (RESULTS / f'revision_{k}' for k in ['axis', 'crosslab', 'transfer', 'ceiling'])
HR = XL / 'harmonized_range'
PAIR_WITHIN = 'CTRPv2|CC|lnIC50->CTRPv2|CC|lnIC50'
FLOOR_MODELS = ['fitness_only', 'MultiViewRandomForest', 'NaiveTissueMeanPredictor', 'NaiveMeanEffectsPredictor']


def rec(df, cols=None, nd=4):
    d = df if cols is None else df[cols]
    return json.loads(d.round(nd).to_json(orient='records'))


def axis():
    out = {}
    out['activity_fraction'] = rec(pd.read_csv(AX / 'activity_fraction.csv'))
    t = pd.read_csv(AX / 'triad_shared_fraction.csv')
    out['shared_fraction'] = rec(t, ['axis_set', 'screen', 'lambda2', 'ci_lo', 'ci_hi', 'split_half'])
    out['pair_classes'] = rec(pd.read_csv(AX / 'pair_class_summary.csv'),
                              ['view', 'endpoint_family', 'pair_class', 'pairs', 'median_r', 'median_disattenuated',
                               'min_r', 'max_r'])
    out['within_screen_controls'] = rec(pd.read_csv(AX / 'within_lab_controls.csv'))
    v = pd.read_csv(AX / 'variance_components.csv')
    out['variance_components'] = rec(v[v.model.str.startswith('REML')], ['model', 'component', 'fraction', 'ci_lo', 'ci_hi'])
    out['lineage'] = rec(pd.read_csv(AX / 'lineage_by_screen.csv'))
    pp = pd.read_csv(AX / 'pairs_partial_growth.csv')
    d = pp.partial_both - pp.r
    out['growth_partial_change'] = {'median': float(d.median()), 'min': float(d.min()), 'pairs': int(len(pp))}
    g = json.loads((AX / 'summary_48.json').read_text())['proliferation']
    out['nci60_own_doubling_time'] = [r for r in g['nci60_own_dt'] if r['axis'] == 'NCI60|all']
    out['growth_proxies'] = g['proxies']
    out['gcsi_gr_formula_check'] = g['info']['gr_formula_check']
    out['leave_class_out'] = rec(pd.read_csv(AX / 'leave_class_out_summary.csv'))
    out['dreval_datasets'] = rec(pd.read_csv(AX / 'dreval_datasets.csv'))
    return out


def leaderboard():
    out = {}
    P = pd.read_csv(XL / 'per_drug_scores.csv.gz', usecols=['pair', 'model', 'raw_w', 'med_w', 'class', 'active_ctrpv2'],
                    low_memory=False)
    P = P[(P.pair == PAIR_WITHIN) & P['class'].isin(['classical', 'neural/factorization']) & (P.model != 'SVR')]
    sh = P.groupby('model').apply(lambda d: d.med_w.sum() / d.raw_w.sum())
    sha = P[P.active_ctrpv2].groupby('model').apply(lambda d: d.med_w.sum() / d.raw_w.sum())
    out['axis_share_published'] = {'median': float(sh.median()), 'min': float(sh.min()), 'max': float(sh.max()),
                                   'models': int(len(sh)), 'active_median': float(sha.median()),
                                   'active_min': float(sha.min()), 'active_max': float(sha.max())}
    o = pd.read_csv(XL / 'oracle_reference.csv').set_index('pair').loc[PAIR_WITHIN]
    out['oracle_share'] = [float(o.oracle_share_within_mean_rzg2), float(o.oracle_share_within_mean_rzg2_lo),
                           float(o.oracle_share_within_mean_rzg2_hi)]
    M = pd.read_csv(XL / 'model_medians.csv')
    out['within_metrics'] = rec(M[(M.pair == PAIR_WITHIN)], ['subset', 'model', 'stat', 'drugs', 'median', 'median_lo',
                                                             'median_hi'])
    out['rankings'] = rec(pd.read_csv(XL / 'rankings.csv'))
    out['ranking_correlations'] = rec(pd.read_csv(XL / 'ranking_correlations.csv'))
    out['critical_difference'] = rec(pd.read_csv(XL / 'critical_difference.csv'))
    R = pd.read_csv(RESULTS / 'dreval' / 'normalized_metric.csv')
    pub = R[R['class'].isin(['classical', 'neural/factorization'])]
    out['axis_only_on_normalized'] = {m: {'value': float(R.loc[R.model == m, 'normalized_pearson'].iloc[0]),
                                          'models_exceeded': int((pub.normalized_pearson <
                                                                  R.loc[R.model == m, 'normalized_pearson'].iloc[0]).sum()),
                                          'of': int(len(pub))}
                                      for m in ['fitness_only', 'fitness_increment', 'fitness_increment_tissue_centred']}
    return out


def crossdataset():
    out = {}
    frames = []
    for d in [XL, HR]:
        S = pd.read_csv(d / 'summary_by_target.csv')
        frames.append(S[(S.subset == 'all') & (S.cellset == 'own')])
    S = pd.concat(frames, ignore_index=True)
    cols = ['pair', 'family', 'drugs', 'raw_median_within', 'raw_median_cross', 'raw_loss', 'raw_loss_lo', 'raw_loss_hi',
            'mediated_retained', 'mediated_retained_lo', 'mediated_retained_hi', 'alignment_ratio_r_pg',
            'alignment_ratio_r_pg_lo', 'alignment_ratio_r_pg_hi', 'target_loading_ratio_r_zg',
            'target_loading_ratio_r_zg_lo', 'target_loading_ratio_r_zg_hi', 'share_of_loss_through_fitness',
            'share_of_loss_lo', 'share_of_loss_hi']
    out['targets'] = rec(S, cols)
    T = pd.concat([pd.read_csv(d / 'tost_selectivity.csv') for d in [XL, HR]], ignore_index=True)
    T = T[(T.subset == 'all') & (T.cellset == 'own') & (T.metric == 'selBOTH') & ~T.target.str.startswith('CTRPv2')]
    out['selective_both_change'] = rec(T, ['pair', 'diff_cross_minus_within', 'ci90_lo', 'ci90_hi',
                                           'equivalent_margin_0.05', 'p_noninferior_margin_0.05'])
    out['selective_both_counts'] = {'versions': int(len(T)), 'equivalent_0.05': int(T['equivalent_margin_0.05'].sum()),
                                    'noninferior_0.05': int((T['p_noninferior_margin_0.05'] < 0.05).sum())}
    L = pd.read_csv(XL / 'leakage_floors.csv')
    L = L[(L.subset == 'all') & (L.cellset == 'own') & (L.metric == 'selBOTH') & (L.side == 'cross')
          & L.model.isin(FLOOR_MODELS)].groupby('pair')['median'].max()
    hf = pd.read_csv(HR / 'leakage_floors.csv')
    L = pd.concat([L, hf[hf.model.isin(FLOOR_MODELS)].groupby('pair')['median'].max()])
    out['leakage_floor'] = {k: round(float(v), 4) for k, v in L.items()}
    out['censoring'] = rec(pd.read_csv(XL / 'censoring.csv'))
    out['censoring_effects'] = rec(pd.read_csv(XL / 'censoring_effects.csv'))
    out['harmonized_ranges_kept'] = pd.read_csv(HR / 'ranges.csv').groupby('target').kept.agg(['sum', 'size']).to_dict(orient='index')
    C = pd.read_csv(XL / 'paired_comparisons.csv')
    ln = C[(C.subset == 'all') & C.pair.str.endswith('|CC|lnIC50') & ~C.pair.str.contains('->CTRPv2')]
    sb = ln[ln.metric == 'selBOTH']
    out['ridge_vs_published_selective_both'] = {
        'min_gain': rec(sb.nsmallest(1, 'median_diff'), ['pair', 'model', 'median_diff', 'median_lo', 'median_hi',
                                                         'wilcoxon_p_holm'])[0],
        'all_holm_below_0.05': bool((sb.wilcoxon_p_holm < 0.05).all()), 'comparisons': int(len(sb))}
    out['ridge_vs_adjusted_ensemblemf'] = rec(ln[(ln.metric == 'adjusted') & (ln.model == 'EnsembleMF')],
                                              ['pair', 'median_diff', 'median_lo', 'median_hi', 'wilcoxon_p_holm'])
    out['selective_target_over_own_axis_removed_raw'] = rec(
        C[(C.subset == 'all') & (C.metric == 'own_axis_removed_raw_ridge')], ['pair', 'median_diff', 'median_lo',
                                                                              'median_hi'])
    M = pd.read_csv(XL / 'model_medians.csv')
    keep = M[(M.subset == 'all') & M.stat.isin(['selB_c', 'adj_c']) & M.model.isin(
        ['EnsembleMF', 'ridge_sel_lm', 'ridge_raw_lm', 'ref_CTRPv2_measured'])]
    out['cross_selectivity_key_models'] = rec(keep, ['pair', 'model', 'stat', 'median', 'median_lo', 'median_hi'])
    cs = pd.read_csv(RESULTS / 'dreval_crossstudy' / 'summary.csv')
    cs = cs[~cs.algorithm.str.startswith('Naive') & ~cs.algorithm.str.startswith('ref:')]
    out['test_modes'] = rec(cs.groupby(['mode', 'target'])[['raw_within', 'raw_cross', 'sel_within', 'sel_cross']]
                            .median().reset_index())
    return out


def ceiling():
    out = {}
    E = pd.read_csv(CE / '49_c2_estimands.csv')
    out['c2'] = rec(E[(E.kappa.isin(['none', '>0.1']))], ['model', 'class', 'kappa', 'aggregation', 'estimate', 'ci_lo',
                                                          'ci_hi', 'n_drugs'])
    R = pd.read_csv(CE / '49_c2_model_replicate_ratio.csv')
    out['model_replicate_ratio'] = rec(R[(R.kappa == 'none') & (R.aggregation == 'perdrug_romeans_bc')],
                                       ['model', 'class', 'c2_model', 'c2_replicate', 'ratio', 'ci_lo', 'ci_hi'])
    out['hierarchical'] = rec(pd.read_csv(CE / '49_c2_hierarchical_summary.csv'),
                              ['model', 'class', 'pop_median_c2', 'pop_median_ci_lo', 'pop_median_ci_hi', 'pop_q25', 'pop_q75'])
    out['crosslab_c2'] = rec(pd.read_csv(CE / '50_crosslab_c2.csv'))
    out['crosslab_ratio'] = rec(pd.read_csv(CE / '50_crosslab_ratio.csv'))
    p = pd.read_csv(CE / '50_cfa_parameters.csv')
    out['factor_model'] = rec(p[p.parameter.str.startswith(('theta', 'psi', 'share', 'mean_ratio'))])
    out['biomarker_pooled'] = rec(pd.read_csv(CE / '52_biomarker_c2_pooled.csv'))
    out['biomarker_transfer'] = rec(pd.read_csv(CE / '52_biomarker_transfer_summary.csv'))
    out['reliability'] = rec(pd.read_csv(CE / '49_reliability_estimands.csv').query("kappa in ['none', '>0.1']"))
    return out


def classes_and_confirmation():
    out = {}
    s53 = json.loads((TR / '53_summary.json').read_text())
    out['curation'] = {k: s53['curation'][k] for k in ['drugs_reviewed', 'reassigned', 'median_rho_sel_curated',
                                                      'frac_reproducible_curated', 'n_per_class_curated']}
    out['activity'] = {'spearman_min_frac_active': s53['activity']['spearman_mean_selective_agreement_vs']['min_frac_active'],
                       'bins': s53['activity']['bins'],
                       'class_given_activity': {k: {kk: v[kk] for kk in ['class_given_activity_p', 'top5_adjusted']}
                                                for k, v in s53['activity']['regression'].items()
                                                if 'class_given_activity_p' in v}}
    ar = pd.read_csv(TR / 'activity_regression.csv')
    out['activity_adjusted_ranks'] = rec(ar[(ar.scheme == 'curated') & (ar.spec == 'logit_quadratic')],
                                         ['class', 'drugs', 'unadjusted_median', 'adjusted_mean', 'rank_adjusted'])
    out['crosslab_reproducibility'] = s53['crosslab']
    out['confirmation_estimates'] = rec(pd.read_csv(TR / 'confirmation_estimates.csv'))
    H = pd.read_csv(TR / 'holm_registered_tests.csv')
    out['holm'] = rec(H, ['screen', 'hypothesis', 'registered_p', 'holm_adjusted_p', 'unit_level_p',
                          'holm_adjusted_p_with_unit_level_p', 'fragile'])
    out['holm_counts_23_superseded'] = {'tests': int(len(H)), 'survive_registered': int((H.holm_adjusted_p < 0.05).sum()),
                                        'survive_unit_level': int((H.holm_adjusted_p_with_unit_level_p < 0.05).sum())}
    # round 2: the registered family has 25 tests (Amendment 3 added H10 and H11; script 86)
    HS = pd.read_csv(RESULTS / 'revision_round2' / 'holm' / 'holm_summary.csv')
    out['holm_25'] = rec(pd.read_csv(RESULTS / 'revision_round2' / 'holm' / 'holm_25.csv'))
    out['holm_counts'] = {f"{r.variant} | {r.family}": {'tests': int(r.tests), 'survive': int(r.survive_holm_0_05)}
                          for r in HS.rename(columns={'survive_holm_0.05': 'survive_holm_0_05'}).itertuples()}
    out['misoriented_first_run'] = rec(pd.read_csv(TR / 'misoriented_first_run.csv'))
    out['beataml_tests'] = rec(pd.read_csv(TR / 'beataml_tests.csv'))
    out['beataml_meta'] = rec(pd.read_csv(TR / 'beataml_meta_analysis.csv'))
    out['beataml_axis_correlates'] = rec(pd.read_csv(TR / 'beataml_axis_correlates.csv'))
    out['beataml_subsets'] = rec(pd.read_csv(TR / 'beataml_sensitivity_subsets.csv'))
    out['pdx_tests'] = rec(pd.read_csv(TR / 'pdx_tumour_level_tests.csv'))
    out['pdx_meta'] = rec(pd.read_csv(TR / 'pdx_meta_analysis.csv'))
    out['power'] = json.loads((TR / '55_summary.json').read_text())
    s54 = json.loads((TR / '54_summary.json').read_text())
    out['pdx_structure'] = {k: s54['pdx_structure'][k] for k in ['models', 'models_with_expression', 'tumours_G1',
                                                                 'tumours_with_expression_G1', 'tumours_G2',
                                                                 'tumours_with_expression_G2']}
    rg = json.loads((RESULTS / 'revision_repro' / 'regenerated' / 'comparison.json').read_text()) \
        if (RESULTS / 'revision_repro' / 'regenerated' / 'comparison.json').exists() else None
    out['regeneration'] = rg if rg is None or len(json.dumps(rg)) < 20000 else {'note': 'see regenerated/comparison.json'}
    return out


def improve():
    """Second benchmark (68_improve_csa.py)."""
    IM = RESULTS / 'revision_improve' / 'no_split_replicates'   # primary since round 2
    if not (IM / 'within_summary.csv').exists():
        return {}
    W = pd.read_csv(IM / 'within_summary.csv')
    models = W[W.model != 'floor_cell_mean']
    floor = W[W.model == 'floor_cell_mean'].set_index('dataset')
    below = [(r.model, r.dataset) for r in models.itertuples() if r.median_raw <= floor.loc[r.dataset, 'median_raw'] + 0.02]
    out = {'run': json.loads((IM / 'run.json').read_text()),
           'within_axis_share_range': [float(models.axis_share.min()), float(models.axis_share.max())],
           'within_axis_share_median': float(models.axis_share.median()),
           'oracle_share_range': [float(W.oracle_share.min()), float(W.oracle_share.max())],
           'floor_median_raw_range': [float(floor.median_raw.min()), float(floor.median_raw.max())],
           'model_median_raw_range': [float(models.median_raw.min()), float(models.median_raw.max())],
           'model_median_selective_range': [float(models.median_selective.min()), float(models.median_selective.max())],
           'models_within_0.02_of_or_below_floor': below,
           'within_pooled_bootstrap': rec(pd.read_csv(IM / 'within_pooled_bootstrap.csv')),
           'identity_max_abs_diff': float(pd.read_csv(IM / 'identity_check.csv').max_abs_diff_vs_within.max())}
    P = pd.read_csv(IM / 'paired_pooled.csv')
    S = pd.read_csv(IM / 'selective_both_change.csv')
    for var in ['own_panel', 'shared_panel']:
        x = P[P.axis_panel == var].pivot_table(index=['source', 'target'], columns='stat', values='estimate')
        sb = S[S.axis_panel == var]
        out[var] = {'delta_raw_range': [float(x.delta_raw.min()), float(x.delta_raw.max())],
                    'mediated_retained_range': [float(x.mediated_retained.min()), float(x.mediated_retained.max())],
                    'share_of_loss_fitness_median': float(x.share_of_loss_fitness.median()),
                    'share_of_loss_fitness_range': [float(x.share_of_loss_fitness.min()), float(x.share_of_loss_fitness.max())],
                    'delta_selective_both_range': [float(sb.delta_selective_both_mean.min()), float(sb.delta_selective_both_mean.max())],
                    'pairs_ci90_below_zero': int((sb.ci90_hi < 0).sum()), 'pairs': int(len(sb)),
                    'selective_both_change': rec(sb)}
    return out


def organoids():
    """Amendment 3 (72_pdm_confirm.py, 74 correction, 75 estimates)."""
    out = {}
    for k, d in [('registered', 'confirmation_PDM'), ('corrected', 'confirmation_PDM_corrected')]:
        f = RESULTS / d / 'hypotheses.json'
        if f.exists():
            h = json.loads(f.read_text())
            out[k] = {'H10': h['H10'], 'H11': h['H11'], 'datasets': h['datasets'],
                      'secondary': {x: h['secondary'][x] for x in ['sel', 'raw', 'agnostic', 'raw_minus_agnostic_mean_raw',
                                                                    'per_dataset_reproducible', 'random_effects_reproducible',
                                                                    'sample_level_reproducible'] if x in h['secondary']}}
    f = RESULTS / 'revision_transfer' / 'pdm_meta_analysis.csv'
    if f.exists():
        out['meta_analysis'] = rec(pd.read_csv(f))
    return out


def round2():
    """Round-2 analyses (scripts 76-97): headline numbers with intervals, so that paper_numbers.json is the single
    verification target."""
    R2 = RESULTS / 'revision_round2'
    return {f[:-5]: json.loads((R2 / f).read_text()) for f in ['key_numbers.json', 'key_numbers_ws6b.json']
            if (R2 / f).exists()}


def collect():
    return {'axis': axis(), 'leaderboard': leaderboard(), 'crossdataset': crossdataset(), 'ceiling': ceiling(),
            'classes_and_confirmation': classes_and_confirmation(), 'improve': improve(), 'organoids': organoids(),
            'round2': round2()}


if __name__ == '__main__':
    c = collect()
    print({k: list(v) for k, v in c.items()})
