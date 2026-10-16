"""Round 3 (workstream 7): collect the headline numbers of tasks 1-4 into results/revision_round2/key_numbers_ws7.json
(every value with its 95% interval where one exists: est / lo / hi), with the source file of each block.

  task1_endpoint_matched      99a (on 80 leaderboard_matched, 81, 82, 83)
  task2_expected_retention    99b (baselines), 99c (calibration: third screen, simulation)
  task3_box1                  99d (selbench.box1 censoring and activity rules)
  task4_two_by_two            97 (four cells), 98 --dataset CTRPv2, 78 / 91 / 81 for the CTRP v2 AUC setting
"""
import sys

sys.dont_write_bytecode = True
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

O = lib.OUT2


def v(row, k, nd=4):
    if k not in row.index or pd.isna(row[k]):
        return None
    out = {'est': round(float(row[k]), nd)}
    for s in ('lo', 'hi'):
        if f'{k}_{s}' in row.index and pd.notna(row[f'{k}_{s}']):
            out[s] = round(float(row[f'{k}_{s}']), nd)
    return out


def task1():
    T = pd.read_csv(O / 'endpoint_matched' / 'pair_classification.csv')
    C = json.loads((O / 'endpoint_matched' / 'counts.json').read_text())
    out = {'source': 'results/revision_round2/endpoint_matched/{pair_classification.csv, counts.json} (99a)',
           'counts': C, 'pairs': {}}
    for _, r in T.iterrows():
        out['pairs'][r.pair] = {
            'endpoint': r.endpoint, 'comparison': r.comparison, 'role': r.role, 'processing': r.processing,
            'clear_raw_loss': bool(r.clear_raw_loss), 'rejects_proportional': bool(r.rejects_proportional),
            'rejection_direction': r.rejection_direction if isinstance(r.rejection_direction, str) else '',
            'keeps_80pct': bool(r.keeps_80pct), 'tost_0.03': bool(r['tost_0.03']),
            'leakage_overlapping': bool(r.leakage_overlapping),
            'within_selBOTH_ci_includes_0': bool(r.within_selBOTH_ci_includes_0),
            **{k: v(r, k) for k in ['raw_loss', 'within_share', 'share_of_loss', 'excess', 'linear_contrast',
                                     'selBOTH_change_sym', 'retained_fraction']},
            'leakage_floor': round(float(r.leakage_floor), 4) if pd.notna(r.leakage_floor) else None,
            'selBOTH_cross': round(float(r.selBOTH_cross), 4)}
    return out


def task2():
    G = pd.read_csv(O / 'retention_calibration' / 'baselines_gdsc2.csv')
    P = pd.read_csv(O / 'retention_calibration' / 'baselines_pairs.csv')
    D = pd.read_csv(O / 'retention_calibration' / 'shortfall_differences.csv')
    T3 = pd.read_csv(O / 'retention_calibration' / 'third_screen.csv')
    SM = pd.read_csv(O / 'retention_calibration' / 'simulation_summary.csv')
    chk = json.loads((O / 'retention_calibration' / 'checks.json').read_text())
    out = {'source': 'results/revision_round2/retention_calibration/ (99b baselines_*.csv, shortfall_differences.csv; '
                     '99c third_screen.csv, simulation_summary.csv)', 'checks': chk,
           'gdsc2_auc_trained': {}, 'gdsc2_shortfall_vs_tissue': {}, 'pairs': {}, 'pairs_shortfall_vs_tissue': {},
           'third_screen': {}, 'simulation': {}}
    keep = ['DrEval learned (4)', 'per-drug ridges (2)', 'EnsembleMF', 'RandomForest', 'NaiveTissueDrugMeanPredictor',
            'FitnessOnlyRidge', 'NaiveMeanEffectsPredictor']
    for _, r in G[G.group.isin(keep)].iterrows():
        out['gdsc2_auc_trained'][f'{r.target}|{r.step}:{r.step_label}|{r.group}'] = {
            **{k: v(r, k) for k in ['selBOTH_within', 'selBOTH_cross', 'observed_retention', 'expected_retention',
                                     'observed_minus_expected']},
            'within_ci_includes_0': bool(r.within_ci_includes_0)}
    for _, r in D[D.setting.notna()].iterrows():
        out['gdsc2_shortfall_vs_tissue'][f'{r.target}|{r.step}|{r.group}'] = {
            'retention_minus_tissue': v(r, 'retention_minus_tissue'),
            'excludes_0': bool(r.retention_minus_tissue_excludes_0)}
    for _, r in P.iterrows():
        out['pairs'][f'{r.benchmark}|{r.pair}|{r.group}'] = {
            **{k: v(r, k) for k in ['selBOTH_within', 'selBOTH_cross', 'observed_retention', 'expected_retention',
                                     'observed_minus_expected']}, 'within_ci_includes_0': bool(r.within_ci_includes_0)}
    dd = D[D.benchmark.notna()]
    for _, r in dd.iterrows():
        out['pairs_shortfall_vs_tissue'][f'{r.benchmark}|{r.pair}|{r.group}'] = {
            'retention_minus_tissue': v(r, 'retention_minus_tissue'),
            'excludes_0': bool(r.retention_minus_tissue_excludes_0)}
    # counts: pooled model groups against the tissue baseline, per benchmark
    pooled = {'leaderboard': 'published 21 (pooled)', 'improve': 'IMPROVE 7 (pooled)', 'retrain': 'DrEval learned (pooled)'}
    cnt = {}
    for bm, g in pooled.items():
        x = dd[(dd.benchmark == bm) & (dd.group == g)]
        cnt[bm] = {'pairs': int(len(x)), 'models_keep_more_clear': int((x.retention_minus_tissue_lo > 0).sum()),
                   'models_keep_less_clear': int((x.retention_minus_tissue_hi < 0).sum()),
                   'pairs_models_keep_less': list(x[x.retention_minus_tissue_hi < 0].pair)}
    x = D[D.setting.notna() & (D.group == 'DrEval learned (4)')]
    cnt['gdsc2_auc_trained_learned4'] = {'target_steps': int(len(x)),
                                         'keep_more_clear': int((x.retention_minus_tissue_lo > 0).sum()),
                                         'keep_less_clear': int((x.retention_minus_tissue_hi < 0).sum())}
    x = D[D.setting.notna() & (D.group == 'per-drug ridges (2)')]
    cnt['gdsc2_auc_trained_ridges'] = {'target_steps': int(len(x)),
                                       'keep_more_clear': int((x.retention_minus_tissue_lo > 0).sum()),
                                       'keep_less_clear': int((x.retention_minus_tissue_hi < 0).sum()),
                                       'clear_pairs': [f'{a}|{b}' for a, b in zip(x[x.retention_minus_tissue_lo > 0].target,
                                                                                 x[x.retention_minus_tissue_lo > 0].step)]}
    out['counts_vs_tissue'] = cnt
    for _, r in T3.iterrows():
        out['third_screen'][f'{r.target}|{r.third_screen}'] = {
            'drugs': int(r.drugs), 'shares_institution_with': r.shares_institution_with if isinstance(
                r.shares_institution_with, str) else '',
            **{k: v(r, k) for k in ['within_selBOTH', 'observed_retention', 'expected_retention',
                                     'observed_minus_expected', 'learned4_observed_minus_expected_same_drugs',
                                     'tissue_observed_minus_expected_same_drugs', 'learned4_minus_third_screen_retention',
                                     'tissue_minus_third_screen_retention']}}
    for _, r in SM.iterrows():
        out['simulation'][r.scenario] = {'description': r.description, 'reps': int(r.reps),
                                         **{k: round(float(r[k]), 4) for k in
                                            ['observed_retention_mean', 'observed_retention_sd', 'expected_retention_mean',
                                             'true_sqrt_ratio', 'observed_minus_expected_mean',
                                             'observed_minus_expected_sd', 'observed_minus_expected_se',
                                             'R_source_hat_mean', 'R_target_hat_mean']}}
    return out


def task3():
    S = json.loads((O / 'box1_validation' / 'summary_r3.json').read_text())
    V = pd.read_csv(O / 'box1_validation' / 'validation_r3.csv')
    return {'source': 'results/revision_round2/box1_validation/{validation_r3.csv, summary_r3.json, tests_r3.txt} (99d)',
            **S, 'checks_table': V[['component', 'function', 'value', 'reference_value', 'passed', 'note']].to_dict(
                orient='records')}


def task4():
    out = two_by_two('two_by_two')
    for topic, key in [('two_by_two_3models', 'three_model_sensitivity (EnsembleMF, GradientBoosting, RandomForest)'),
                       ('two_by_two_provisional_3models', 'PROVISIONAL three-model design (SimpleNeuralNetwork '
                                                          'still training when computed)')]:
        if (O / topic / 'cells.csv').exists():
            out[key] = two_by_two(topic, minimal=True)
    for sfx in ['_CTRPv2', '_CTRPv2_provisional_3models']:
        f = O / 'endpoint_cells' / f'summary{sfx}.csv'
        if f.exists():
            E = pd.read_csv(f)
            out[f'cell_selection{sfx}'] = {r.version: {'pairs': int(r.pairs), **{k: v(r, k) for k in
                                                       ['axis_share', 'perfect_share', 'axis_capture', 'selective_capture']}}
                                           for _, r in E.iterrows()}
    E = pd.read_csv(O / 'endpoint_cells' / 'summary.csv')
    out['cell_selection_GDSC2'] = {r.version: {'pairs': int(r.pairs), **{k: v(r, k) for k in
                                               ['axis_share', 'perfect_share', 'axis_capture', 'selective_capture']}}
                                   for _, r in E.iterrows()}
    return out


def two_by_two(topic, minimal=False):
    out = {'source': f'results/revision_round2/{topic}/{{cells,contrasts,models,attribution}}.csv (97); '
                     'endpoint_cells/summary_CTRPv2*.csv (98 --dataset CTRPv2); capture/, lineage/, attenuation/'}
    C = pd.read_csv(O / topic / 'cells.csv')
    K = pd.read_csv(O / topic / 'contrasts.csv')
    keys = ['axis_share', 'perfect_share', 'corrected_reference', 'axis_capture', 'selective_capture',
            'axis_capture_tau', 'selective_capture_tau', 'alignment_ratio', 'mean_raw', 'floor_median_raw',
            'best_learned_median_raw', 'best_minus_floor_median_raw', 'floor_over_best_median_raw',
            'tissue_baseline_median_selective', 'learned_median_selective_median', 'learned_median_selective_max',
            'best_selective_minus_tissue_baseline', 'ridge_median_raw']
    out['cells'] = {r.cell: {'setting': r.setting, 'models': r.models, 'cells_scored': int(r.cells_scored),
                             'drugs': int(r.drugs), 'best_learned_model': r.best_learned_model,
                             **{k: v(r, k) for k in keys},
                             'lineage_adjusted_median_selective_range': r.get('lineage_adjusted_median_selective_range')}
                    for _, r in C.iterrows()}
    out['contrasts'] = {r.contrast: {k: v(r, k) for k in keys if k in r.index and pd.notna(r[k])} for _, r in K.iterrows()}
    out['complete'] = bool('CTRP v2 AUC' in set(C.cell))
    if minimal:
        return out
    A = pd.read_csv(O / topic / 'attribution.csv')
    out['attribution'] = {f'{r.cell}|{r.pair}': {**{k: v(r, k) for k in ['raw_loss', 'within_share', 'share_of_loss',
                                                                          'excess', 'd_selBOTH_sym']},
                                                  'reading': r.reading} for _, r in A.iterrows()}
    M = pd.read_csv(O / topic / 'models.csv')
    out['models'] = {f'{r.cell}|{r.model}': {k: v(r, k) for k in ['axis_share', 'axis_capture', 'selective_capture',
                                                                  'median_raw', 'median_selective',
                                                                  'median_selective_lin']}
                     for _, r in M.iterrows()}
    return out


def main():
    K = {'note': 'Round-3 analyses (workstream 7; scripts 99a-99f and the extended 78, 80-84, 89, 91, 97, 98). est = '
                 'point estimate; lo/hi = 95% two-way (cell x drug) bootstrap interval unless the block says otherwise '
                 '(98: drug bootstrap; simulation: mean and SD over replicates).'}
    for name, fn in [('task1_endpoint_matched', task1), ('task2_expected_retention', task2), ('task3_box1', task3),
                     ('task4_two_by_two', task4)]:
        try:
            K[name] = fn()
        except FileNotFoundError as e:
            K[name] = {'missing': str(e)}
            print('missing', name, e)
    lib.write_json(O / 'key_numbers_ws7.json', K)
    print('written', O / 'key_numbers_ws7.json')


if __name__ == '__main__':
    main()
