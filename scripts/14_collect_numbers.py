"""Collect every number quoted in the manuscript into results/paper_numbers.json (single source)."""
import json

import numpy as np
import pandas as pd

from common import PROCESSED, RESULTS, write_json


def j(p):
    return json.loads((RESULTS / p).read_text())


def main():
    N = {}
    long = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    dec = j('decomposition/summary.json')
    N['discovery_matrix'] = dec['screens']
    N['harmonize'] = json.loads((PROCESSED / 'harmonize_summary.json').read_text())['per_screen']
    stats = pd.read_csv(RESULTS / 'decomposition' / 'drug_screen_stats.csv')
    act = stats[stats.frac_auc_below_0p8 >= 0.1]
    N['fitness_r2_all'] = stats.groupby('screen').fitness_r2.median().round(3).to_dict()
    N['fitness_r2_active'] = act.groupby('screen').fitness_r2.median().round(3).to_dict()
    N['active_drugs'] = act.groupby('screen').size().to_dict()
    fa = j('fitness_axis/summary.json')
    N['fitness_axis'] = fa
    pairs = pd.DataFrame(fa['pairs'])
    disc = pairs[pairs.a.isin(['PRISM', 'CTRP', 'GDSC1', 'GDSC2']) & pairs.b.isin(['PRISM', 'CTRP', 'GDSC1', 'GDSC2'])]
    N['fitness_cross_discovery_range'] = [float(disc['corr'].min()), float(disc['corr'].max())]
    N['fitness_cross_all_range'] = [float(pairs['corr'].min()), float(pairs['corr'].max())]
    N['fitness_cross_common_drugs_range'] = [float(pairs.corr_common_drugs.min()), float(pairs.corr_common_drugs.max())]
    N['fitness_disattenuated_max'] = float(pairs.disattenuated.max())

    s = pd.read_csv(RESULTS / 'models' / 'score_summary.csv')
    w = s[s['train'] == s['eval']].pivot(index='model', columns='train', values='raw')
    c = s[s['train'] != s['eval']].groupby('model')[['raw', 'sel']].median()
    N['models_within_raw'] = w.round(3).to_dict(orient='index')
    N['models_cross_median'] = c.round(3).to_dict(orient='index')
    sc = pd.read_csv(RESULTS / 'models' / 'scores.csv.gz')
    cr = sc[sc['train'] != sc['eval']]
    N['models_cross_pairs'] = int(len(cr[cr.model == 'ridge_sel']))
    g = pd.read_csv(RESULTS / 'global_model' / 'scores.csv')
    g['split2'] = g.split.where(~g.split.str.startswith('cross_from'), 'cross_screen')
    N['global_model'] = g.groupby(['model', 'split2'])[['global_r', 'per_drug_raw', 'per_drug_sel']].median().round(3).reset_index().to_dict(orient='records')
    N['global_model_by_screen'] = g[g.split.isin(['random_pairs', 'cell_blind', 'drug_blind'])].round(3).to_dict(orient='records')

    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0)
    N['multi_screen_drugs'] = int(len(dt))
    N['rho_raw_median'] = float(dt.rho_raw.median()); N['rho_sel_median'] = float(dt.rho_sel.median())
    N['frac_reproducible'] = float((dt.rho_sel >= 0.3).mean()); N['frac_nonreproducible'] = float((dt.rho_sel < 0.15).mean())
    N['n_reproducible'] = int((dt.rho_sel >= 0.3).sum()); N['n_nonreproducible'] = int((dt.rho_sel < 0.15).sum())
    N['raw_sel_corr'] = float(np.corrcoef(dt.rho_raw, dt.rho_sel)[0, 1])
    N['class_summary'] = pd.read_csv(RESULTS / 'drug_classes' / 'class_summary.csv', index_col=0).round(3).to_dict(orient='index')
    N['top_drugs'] = j('drug_classes/summary.json')['top_drugs']
    pp = pd.read_csv(RESULTS / 'decomposition' / 'cross_screen_pairs.csv')
    N['screen_pairs'] = pp.groupby(['screen_a', 'screen_b']).agg(drugs=('compound', 'size'), raw=('rho_raw', 'median'),
                                                                  sel=('rho_sel', 'median')).round(3).reset_index().to_dict(orient='records')
    N['ceiling'] = j('ceiling/summary.json')
    N['voi_l1000'] = pd.read_csv(RESULTS / 'value_of_information' / 'l1000_two_way.csv').round(3).to_dict(orient='records')
    inc = pd.read_csv(RESULTS / 'value_of_information' / 'l1000_incremental.csv')
    N['voi_incremental'] = inc.round(3).to_dict(orient='records')
    N['voi_incremental_medians'] = inc[['basal_r', 'measured_r', 'measured_given_basal', 'combined_r', 'basal_measured_r']].median().round(3).to_dict()
    N['voi_tahoe'] = pd.read_csv(RESULTS / 'value_of_information' / 'tahoe_pooled.csv').round(3).to_dict(orient='records')
    N['voi_generator_agreement'] = j('value_of_information/summary.json')['generator_agreement']
    for k in ['gCSI', 'NCI60', 'CCLE', 'FIMM', 'GRAY', 'gCSI_GRAOC', 'NCI60_cellminerQC', 'rehearsal_GDSC2']:
        N[f'confirm_{k}'] = j(f'confirmation_{k}/hypotheses.json')
    N['confirm_BeatAML'] = j('confirmation_BeatAML/hypotheses.json')
    N['confirm_load'] = json.loads((PROCESSED / 'confirmation' / 'load_summary.json').read_text())
    N['robustness'] = j('robustness/summary.json')
    N['biomarkers'] = {k: v for k, v in j('biomarkers/summary.json').items() if k != 'composition'}
    N['retrieval_baselines'] = pd.read_csv(RESULTS / 'value_of_information' / 'retrieval_baselines.csv').round(3).to_dict(orient='records')
    if (RESULTS / 'dreval' / 'model_summary.csv').exists():
        D = pd.read_csv(RESULTS / 'dreval' / 'model_summary.csv')
        keep = ['model', 'class', 'drugs', 'raw', 'selective', 'selective_lo', 'selective_hi', 'alignment',
                'mediated_share', 'prediction_fitness_r2', 'reproducible_selective', 'dreval_pearson_norm']
        N['dreval'] = {'run': j('dreval/run.json'), 'models': D[keep].round(4).to_dict(orient='records')}
        pub = D[D['class'].isin(['classical', 'neural/factorization']) & (D.model != 'SVR')].dropna(subset=['selective'])
        N['dreval']['published_mediated_share_range'] = [float(pub.mediated_share.min()), float(pub.mediated_share.max())]
        N['dreval']['published_selective_range'] = [float(pub.selective.min()), float(pub.selective.max())]
        N['dreval']['n_beat_tissue_drug_mean'] = int((pub.selective > D.set_index('model').loc['NaiveTissueDrugMeanPredictor', 'selective']).sum())
    if (RESULTS / 'confirmation_BeatAML' / 'scores.csv').exists():
        N['beataml_transfer'] = j('confirmation_BeatAML/hypotheses.json')
    # cross-laboratory re-scoring of the leaderboard (same predictions scored in CTRPv2 and in other screens)
    X = pd.read_csv(RESULTS / 'dreval_crosslab_leaderboard' / 'summary.csv')
    pubx = X[X['class'].isin(['classical', 'neural/factorization'])]
    xl = {'run': j('dreval_crosslab_leaderboard/run.json'), 'n_published': int(pubx.model.nunique())}
    for t in ['GDSC1', 'GDSC2', 'CTRPv1', 'CCLE']:
        d = pubx[pubx.target == t]
        q = d[(d.raw_drop > 0.03) & (d.meanraw_within > 0.1)]
        r = d[d.selterm_within > 0.03]
        xl[t] = {'share_of_drop_from_fitness_median': float(q.share_of_drop_from_fitness.median()) if len(q) else None,
                 'share_iqr': [float(q.share_of_drop_from_fitness.quantile(.25)), float(q.share_of_drop_from_fitness.quantile(.75))] if len(q) else None,
                 'n_models_with_drop': int(len(q)),
                 'mediated_retained_median': float(d.mediated_retained.median()),
                 'mediated_retained_iqr': [float(d.mediated_retained.quantile(.25)), float(d.mediated_retained.quantile(.75))],
                 'selterm_retained_median': float(r.selterm_retained.median()), 'n_models_selterm': int(len(r)),
                 'raw_within_median': float(d.raw_within.median()), 'raw_cross_median': float(d.raw_cross.median()),
                 'best_published_sel_cross': d.sort_values('sel_cross', ascending=False).iloc[0][['model', 'sel_cross']].to_dict()}
        ours = X[(X.target == t) & X.model.isin(['ridge_sel_lm', 'ridge_sel_expr', 'ridge_raw_lm', 'fitness_only'])]
        xl[t]['ours'] = ours.set_index('model')[['raw_within', 'raw_cross', 'sel_within', 'sel_cross']].round(3).to_dict(orient='index')
    C = pd.read_csv(RESULTS / 'dreval_crosslab_leaderboard' / 'paired_selective_cross.csv')
    for t in ['GDSC1', 'GDSC2', 'CTRPv1']:
        c = C[(C.target == t) & (C.ours == 'ridge_sel_lm')]
        xl[t]['paired_ridge_sel_lm'] = {'models': int(len(c)), 'min_gain': float(c.median_diff.min()), 'max_p': float(c.wilcoxon_p.max()),
                                        'frac_better_range': [float(c.frac_better.min()), float(c.frac_better.max())]}
    U = pd.read_csv(RESULTS / 'dreval_crosslab_leaderboard' / 'uncertainty.csv')
    xl['pooled_bootstrap'] = U[U.model == 'pooled published models'].round(4).to_dict(orient='records')
    N['crosslab_leaderboard'] = xl
    M = pd.read_csv(RESULTS / 'dreval_crossstudy' / 'summary.csv')
    mods = M[~M.algorithm.str.startswith('Naive') & ~M.algorithm.str.startswith('ref:')]
    cm = {}
    for (mode, t), d in mods.groupby(['mode', 'target']):
        q = d[(d.raw_drop > 0.03) & (d.meanraw_within > 0.1)]
        cm[f'{mode}|{t}'] = {'models': int(len(d)), 'raw_within': float(d.raw_within.median()), 'raw_cross': float(d.raw_cross.median()),
                             'sel_within': float(d.sel_within.median()), 'sel_cross': float(d.sel_cross.median()),
                             'share_via_fitness': float(q.share_of_drop_from_fitness.median()) if len(q) >= 3 else None,
                             'mediated_retained': float(d.mediated_retained.median())}
    refm = M[M.algorithm.isin(['ref:CTRPv2_measured', 'ref:CTRPv2_fitness_axis']) & (M['mode'] == 'LCO')]
    cm['references'] = refm[['algorithm', 'target', 'raw_cross', 'sel_cross', 'mediated_retained']].round(3).to_dict(orient='records')
    N['crossstudy_dreval_full'] = cm
    NM = pd.read_csv(RESULTS / 'dreval' / 'normalized_metric.csv')
    N['dreval_normalized'] = NM.round(4).to_dict(orient='records')
    A = pd.read_csv(RESULTS / 'dreval' / 'model_axis_adjusted.csv')
    N['model_axis_adjusted'] = A.round(4).to_dict(orient='records')
    N['two_part'] = pd.read_csv(RESULTS / 'dreval' / 'two_part_summary.csv').round(4).to_dict(orient='records')
    N['confirm_PDX_Bruna'] = j('confirmation_PDX_Bruna/hypotheses.json')
    N['confirm_PDX_Bruna_lnIC50'] = j('confirmation_PDX_Bruna_lnIC50/hypotheses.json')
    N['pdx_transfer_run'] = j('pdx_transfer/run.json')
    N['freeze_amendment2'] = {'frozen_utc': json.loads((RESULTS.parent / 'protocol' / 'freeze_manifest_amendment2.json').read_text())['frozen_utc'],
                              'responses_retrieved_utc': json.loads((RESULTS.parent / 'data' / 'raw' / 'pdx_bruna_responses' / 'fetch_log.json').read_text())['utc']}
    N['freeze'] = {'frozen_utc': j('../protocol/freeze_manifest.json')['frozen_utc'] if False else
                   json.loads((RESULTS.parent / 'protocol' / 'freeze_manifest.json').read_text())['frozen_utc']}
    import importlib
    N['revision'] = importlib.import_module('67_revision_numbers').collect()
    N['harmonized_measurements'] = {'rows': int(len(long)), 'compounds': int(long.compound.nunique()),
                                    'cell_lines': int(long.depmap_id.nunique())}
    write_json(RESULTS / 'paper_numbers.json', N)
    print('written', len(N), 'entries')


if __name__ == '__main__':
    main()
