"""Round 2 (WS6b): collect the headline numbers of scripts 90-95 and 97 into
results/revision_round2/key_numbers_ws6b.json. Every value is {'est', 'lo', 'hi'} where an interval exists
(95% two-way cell x drug bootstrap unless the entry's 'interval' says otherwise).
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
    L = pd.read_csv(O / 'harmonization_gdsc2' / 'ladder.csv')
    D = pd.read_csv(O / 'harmonization_gdsc2' / 'step_differences.csv')
    keys = ['raw_within', 'raw_cross', 'raw_loss', 'd_axis_term', 'd_selective_term', 'selBOTH_sym_within',
            'selBOTH_sym_cross', 'd_selBOTH_sym', 'within_share', 'share_of_loss', 'excess', 'retention_difference',
            'selBOTH_retained_sym', 'expected_retention', 'observed_minus_expected', 'ratio_to_measurement']
    out = {'interval': '95% two-way bootstrap (941 panel cells x 287 GDSC2 drugs, shared by all steps)', 'ladder': {},
           'step_differences': {}}
    for _, r in L[L.group.isin(['DrEval learned (4)', 'per-drug ridges (2)', 'EnsembleMF', 'NaiveTissueDrugMeanPredictor'])].iterrows():
        out['ladder'][f'{r.group}|{r.target}|{r.step}:{r.step_label}'] = {
            'drugs': int(r.drugs_scored), **{k: v(r, k) for k in keys}, 'reading': r.reading}
    for _, r in D[D.group.isin(['DrEval learned (4)', 'per-drug ridges (2)'])].iterrows():
        out['step_differences'][f'{r.group}|{r.target}|{r.contrast}'] = {
            k: v(r, f'diff_{k}') for k in ['d_selBOTH_sym', 'raw_loss', 'd_axis_term', 'd_selective_term', 'excess',
                                           'retention_difference']}
    return out


def task2():
    P = pd.read_csv(O / 'lineage' / 'lineage_pooled.csv')
    M = pd.read_csv(O / 'lineage' / 'lineage_by_model.csv')
    keys = ['median_selective', 'median_selective_lin', 'mean_selective', 'mean_selective_lin', 'lineage_adjustment',
            'baseline_median_selective', 'median_selective_minus_baseline', 'mean_selective_lin_minus_baseline_selective',
            'median_over_models_median_selective_minus_baseline', 'selective_capture', 'selective_capture_lin',
            'covariate_share', 'perfect_covariate_share', 'median_raw', 'baseline_median_raw']
    out = {'interval': '95% two-way bootstrap (78 draws)', 'pooled': {}, 'models': {}}
    for _, r in P.iterrows():
        out['pooled'][f'{r.benchmark}|{r.setting}|{r.group}'] = {
            'models': int(r.models), **{k: v(r, k) for k in keys if k in r.index},
            'n_models_median_selective_above_baseline': (int(r.n_models_median_selective_above_baseline)
                                                         if 'n_models_median_selective_above_baseline' in r.index
                                                         and pd.notna(r.n_models_median_selective_above_baseline) else None)}
    for _, r in M.iterrows():
        if r.kind not in ('learned', 'tissue-by-drug baseline', 'this study (ridge)'):
            continue
        out['models'][f'{r.benchmark}|{r.setting}|{r.model}'] = {
            k: v(r, k) for k in ['median_raw', 'median_selective', 'median_selective_lin', 'selective_median_minus_baseline',
                                 'selective_capture', 'selective_capture_lin']}
    return out


def task3():
    A = pd.read_csv(O / 'training_reproduction' / 'agreement_summary.csv')
    S = pd.read_csv(O / 'training_reproduction' / 'score_summary.csv')
    N = pd.read_csv(O / 'training_reproduction' / 'normalized_pearson.csv')
    out = {'interval': '95% two-way bootstrap', 'agreement': {}, 'scores': {}, 'normalized_pearson': {}}
    for _, r in A.iterrows():
        out['agreement'][r.comparison] = {k: (round(float(r[k]), 4) if isinstance(r[k], (float, np.floating)) else r[k])
                                          for k in ['splits', 'cells_common', 'split_assignment_agreement', 'drugs',
                                                    'pearson_all_pairs', 'median_drug_pearson', 'q10_drug_pearson',
                                                    'min_drug_pearson', 'frac_drugs_pearson_ge_0.9',
                                                    'median_rms_diff_rel_sd']}
    for _, r in S.iterrows():
        out['scores'][f'{r.comparison}|{r.statistic}'] = {'retrained_or_a': v(r, 'a'), 'released_or_b': v(r, 'b'),
                                                         'difference': v(r, 'a_minus_b'),
                                                         'across_drug_pearson': (round(float(r.across_drug_pearson), 4)
                                                                                 if pd.notna(r.get('across_drug_pearson')) else None)}
    for _, r in N.iterrows():
        out['normalized_pearson'][f'{r.comparison}|{r.model}'] = {
            'splits': int(r.splits), 'mean': round(float(r.normalized_pearson_mean), 4),
            'dreval_reported_all_splits': round(float(r.dreval_reported_all_splits), 4)}
    return out


def task4():
    d = O / 'axis_contents'
    A = pd.read_csv(d / 'marker_associations.csv')
    Vr = pd.read_csv(d / 'marker_variance.csv')
    Pa = pd.read_csv(d / 'pair_agreement.csv')
    B = pd.read_csv(d / 'beataml_flt3.csv')
    T = pd.read_csv(d / 'axis_transfer.csv')
    S = json.loads((d / 'summary.json').read_text())
    out = {'interval': 'cell-line bootstrap (A; axes fixed, B = 1,000); sample/patient/tumour-cluster bootstrap (B-D; '
                       'B = 2,000)', 'markers': {}, 'variance_explained_adjusted_R2': {}, 'pairs': {}, 'beataml_flt3': {},
           'axis_transfer': {}, 'organoid_histotype': S.get('organoid_histotype'), 'beataml_rule': S.get('flt3_rule'),
           'beataml_info': S.get('beataml')}
    for _, r in A.iterrows():
        out['markers'][f'{r.axis}|{r.marker}'] = {'cells': int(r.cells), 'spearman_all': v(r, 'spearman_all'),
                                                   'spearman_within_lineage': v(r, 'spearman_within_lineage')}
    for _, r in Vr.iterrows():
        out['variance_explained_adjusted_R2'][r.axis] = {k: v(r, k) for k in ['r2_markers', 'r2_lineage', 'r2_both',
                                                                               'r2_markers_beyond_lineage']}
    for _, r in Pa.iterrows():
        out['pairs'][f'{r.axis_a}~{r.axis_b}'] = {'cells': int(r.cells), **{k: v(r, k) for k in [
            'r', 'partial_markers', 'partial_lineage', 'partial_both', 'cov_share_markers', 'cov_share_both']}}
    for _, r in B.iterrows():
        out['beataml_flt3'][r.axis] = {'samples': (int(r.samples) if pd.notna(r.samples) else None),
                                       'flt3_itd_effect_sd': v(r, 'flt3_itd_effect_sd'),
                                       'mannwhitney_p': (float(r.mannwhitney_p) if pd.notna(r.mannwhitney_p) else None)}
    for _, r in T.iterrows():
        out['axis_transfer'][r.cohort] = {'units': r.unit, 'drugs': int(r.drugs),
                                          'r_agnostic_vs_measured_axis': v(r, 'r_agnostic_vs_measured_axis'),
                                          'mean_raw_r_agnostic_per_drug': v(r, 'mean_raw_r_agnostic_per_drug')}
    return out


def task5():
    S = json.loads((O / 'box1_validation' / 'summary.json').read_text())
    T = pd.read_csv(O / 'box1_validation' / 'validation.csv')
    return {**S, 'checks': {f'{r.component}|{r.function}': {'n': int(r.n_compared), 'max_abs_diff': r.max_abs_diff,
                                                             'tolerance': r.tolerance, 'passed': bool(r.passed)}
                            for _, r in T.iterrows()}}


def task6():
    r = json.loads((lib.RES / 'revision_repro' / 'regenerated_amendment3' / 'comparison.json').read_text())
    return {'summary': r['summary'], 'run': r['run'], 'environment': r['environment'],
            'script_checks': r['script_checks']}


def task7():
    d = O / 'two_by_two'
    if not (d / 'cells.csv').exists():
        return {'missing': 'two_by_two not run'}
    C = pd.read_csv(d / 'cells.csv')
    K = pd.read_csv(d / 'contrasts.csv')
    A = pd.read_csv(d / 'attribution.csv')
    keys = ['axis_share', 'perfect_share', 'corrected_reference', 'share_ratio', 'axis_capture', 'selective_capture',
            'axis_capture_tau', 'selective_capture_tau', 'floor_median_raw', 'best_learned_median_raw',
            'floor_over_best_median_raw', 'ridge_median_raw', 'tissue_baseline_median_selective',
            'learned_median_selective_median', 'learned_median_selective_max', 'best_selective_minus_tissue_baseline']
    out = {'interval': '95% two-way bootstrap (78 draws); contrasts from independent bootstraps paired by index',
           'cells': {}, 'contrasts': {}, 'attribution': {}}
    for _, r in C.iterrows():
        out['cells'][r.cell] = {'models': r.models, 'best_learned_model': r.best_learned_model,
                                **{k: v(r, k) for k in keys if k in r.index}}
    for _, r in K.iterrows():
        out['contrasts'][r.contrast] = {k: v(r, k) for k in keys if k in r.index and v(r, k) is not None}
    for _, r in A.iterrows():
        out['attribution'][f'{r.cell}|{r.pair}'] = {**{k: v(r, k) for k in ['raw_loss', 'within_share', 'share_of_loss',
                                                                              'excess', 'retention_difference',
                                                                              'd_selBOTH_sym']}, 'reading': r.reading}
    return out


def main():
    K = {'note': 'Round-2 WS6b analyses (scripts 90-97). est = point estimate; lo/hi = 95% interval as stated per task.',
         'files': {'task1': 'results/revision_round2/harmonization_gdsc2/', 'task2': 'results/revision_round2/lineage/',
                   'task3': 'results/revision_round2/training_reproduction/', 'task4': 'results/revision_round2/axis_contents/',
                   'task5': 'selbench/box1.py, selbench/test_box1.py, results/revision_round2/box1_validation/',
                   'task6': 'results/revision_repro/regenerated_amendment3/', 'task7': 'results/revision_round2/two_by_two/'}}
    for name, fn in [('task1_gdsc2_harmonization', task1), ('task2_lineage', task2), ('task3_training_reproduction', task3),
                     ('task4_axis_contents', task4), ('task5_box1', task5), ('task6_regeneration_amendment3', task6),
                     ('task7_two_by_two', task7)]:
        try:
            K[name] = fn()
        except FileNotFoundError as e:
            K[name] = {'missing': str(e)}
            print('missing', name, e)
    lib.write_json(O / 'key_numbers_ws6b.json', K)
    print('written', O / 'key_numbers_ws6b.json')


if __name__ == '__main__':
    main()
