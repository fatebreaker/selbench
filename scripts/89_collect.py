"""Round 2: collect the headline numbers of scripts 78-88 into results/revision_round2/key_numbers.json (every value
with its 95% interval where one exists; 'lo'/'hi' keys), and print a digest used to write SUMMARY.md.
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
    out = {'est': round(float(row[k]), nd)}
    for s in ('lo', 'hi'):
        if f'{k}_{s}' in row.index and pd.notna(row[f'{k}_{s}']):
            out[s] = round(float(row[f'{k}_{s}']), nd)
    return out


def capture():
    M = pd.read_csv(O / 'capture' / 'capture_by_model.csv')
    P = pd.read_csv(O / 'capture' / 'capture_pooled.csv')
    keys = ['axis_capture', 'selective_capture', 'alignment_ratio', 'axis_share', 'perfect_share', 'share_ratio',
            'cf_share_ratio', 'bound', 'median_raw', 'median_selective']
    out = {'pooled': {}, 'models': {}}
    for _, r in P.iterrows():
        d = {k: v(r, k) for k in keys if k in r.index and k not in ('median_raw', 'median_selective')}
        for k in ['axis_capture', 'selective_capture', 'share_ratio', 'cf_share_ratio', 'alignment_ratio']:
            d[f'across_models_{k}'] = {'median': round(float(r[f'median_over_models_{k}']), 4),
                                       'min': round(float(r[f'min_over_models_{k}']), 4),
                                       'max': round(float(r[f'max_over_models_{k}']), 4)}
        out['pooled'][f"{r.benchmark}|{r.setting}|{r.group}"] = d
    for _, r in M.iterrows():
        out['models'][f"{r.benchmark}|{r.setting}|{r.model}"] = {
            **{k: v(r, k) for k in keys if k in r.index and pd.notna(r[k])},
            'drug_agnostic': str(r.drug_agnostic) if pd.notna(r.drug_agnostic) else '', 'naive': bool(r.naive)}
    return out


def reliability():
    S = pd.read_csv(O / 'reliability' / 'reliability_summary.csv')
    R = pd.read_csv(O / 'reliability' / 'corrected_reference.csv')
    C = pd.read_csv(O / 'reliability' / 'corrected_capture_pooled.csv')
    out = {'sources': {}, 'corrected_reference': {}, 'corrected_capture_pooled': {}}
    for _, r in S.iterrows():
        out['sources'][f'{r.key}|{r.stat}'] = {'est': round(r.estimate, 4), 'lo': round(r.lo, 4) if pd.notna(r.lo) else None,
                                               'hi': round(r.hi, 4) if pd.notna(r.hi) else None, 'endpoint': r.endpoint,
                                               'drugs': int(r.drugs)}
    for _, r in R.iterrows():
        out['corrected_reference'][f'{r.benchmark}|{r.setting}|{r.reliability_source}|kappa={r.kappa}'] = {
            'R_sel': v(r, 'R_sel'), 'raw_reference': v(r, 'raw_reference'),
            'corrected_reference': v(r, 'corrected_reference'), 'corrected_bound': v(r, 'corrected_bound'),
            'primary': bool(r.primary_source)}
    for _, r in C[C.kappa == 1.0].iterrows():
        out['corrected_capture_pooled'][f'{r.benchmark}|{r.setting}|{r.group}|{r.reliability_source}'] = {
            k: v(r, k) for k in ['corrected_reference', 'axis_capture_tau', 'selective_capture_tau', 'share_ratio_tau']}
    return out


def attenuation():
    T = pd.read_csv(O / 'attenuation' / 'attenuation_by_target.csv')
    R = pd.read_csv(O / 'attenuation' / 'loss_share_range.csv')
    out = {'targets': {}, 'loss_share_range': R.to_dict(orient='records')}
    for _, r in T.iterrows():
        out['targets'][f'{r.benchmark}|{r.pair}|{r.group}'] = {
            **{k: v(r, k) for k in ['raw_loss', 'within_share', 'share_of_loss', 'excess', 'axis_retained',
                                     'selterm_retained', 'retention_difference', 'd_selBOTH_sym']},
            'loss_clear': bool(r.loss_clear), 'rejects_proportional_excess': bool(r.rejects_proportional_excess),
            'rejects_proportional_retention': bool(r.rejects_proportional_retention), 'reading': r.reading}
    return out


def symmetric():
    return json.loads((O / 'symmetric_selboth' / 'summary.json').read_text())


def equivalence():
    return json.loads((O / 'equivalence' / 'summary.json').read_text())


def source_ref():
    T = pd.read_csv(O / 'source_reference' / 'source_ratio_by_target.csv')
    E = pd.read_csv(O / 'source_reference' / 'expected_retention.csv')
    out = {'ratios': {}, 'expected_vs_observed': {}}
    for _, r in T.iterrows():
        if '@' in r.pair or r.family == 'shared_panel':
            continue
        d = {k: v(r, k) for k in ['measurement_cross_selBOTH', 'model_cross_selBOTH', 'ratio_to_measurement',
                                   'observed_retention_selBOTH_sym']}
        for c in T.columns:
            if c.startswith('fraction_of_corrected_max[') and not c.endswith(('_lo', '_hi')) and pd.notna(r[c]):
                d[c] = v(r, c)
        out['ratios'][f'{r.benchmark}|{r.pair}|{r.group}'] = d
    for _, r in E.iterrows():
        out['expected_vs_observed'][f'{r.benchmark}|{r.pair}|{r.group}'] = {
            k: v(r, k) for k in ['R_source', 'R_target', 'expected_retention', 'observed_retention', 'observed_minus_expected']}
    return out


def harmonization():
    A = pd.read_csv(O / 'harmonization' / 'lnic50_same_drugs.csv')
    B = pd.read_csv(O / 'harmonization' / 'training_side_steps.csv')
    C = pd.read_csv(O / 'harmonization' / 'range_axes_correlations.csv')
    Sp = pd.read_csv(O / 'harmonization' / 'range_specific_component.csv')
    Rp = pd.read_csv(O / 'harmonization' / 'range_tail_reproducibility.csv')
    Sc = pd.read_csv(O / 'harmonization' / 'range_model_scores.csv')
    out = {'lnic50_same_drugs': {}, 'steps': {}, 'range_axes': {}, 'range_specific': {}, 'tail_reproducibility': {},
           'range_model_scores': {}}
    for _, r in A.iterrows():
        out['lnic50_same_drugs'][f'{r.target}|{r.version}'] = {'drugs': int(r.drugs), **{k: v(r, k) for k in
                                                                ['mediated_retained', 'raw_loss', 'share_of_loss',
                                                                 'excess_over_within_share', 'd_selBOTH_paper']},
                                                                'median_raw_within': round(r.median_raw_within_point, 4),
                                                                'median_raw_cross': round(r.median_raw_cross_point, 4)}
    for _, r in B.iterrows():
        out['steps'][f'{r.target}|{r.step}'] = {'drugs': int(r.drugs),
                                                'training_side': {'est': r.training_side_median_raw, 'lo': r.training_side_lo, 'hi': r.training_side_hi},
                                                'target_side': {'est': r.target_side_median_raw, 'lo': r.target_side_lo, 'hi': r.target_side_hi},
                                                'mediated_retained': {'est': r.mediated_retained, 'lo': r.mediated_retained_lo, 'hi': r.mediated_retained_hi}}
    for _, r in C.iterrows():
        out['range_axes'][f'{r.target}|{r.axis_a}~{r.axis_b}'] = v(r, 'r')
    for _, r in Sp.iterrows():
        out['range_specific'][f'{r.target}|{r.component}~{r.with_axis}'] = v(r, 'r')
    for _, r in Rp.iterrows():
        out['tail_reproducibility'][f'{r.target_range_from}|{r.axis_a}~{r.axis_b}'] = {**v(r, 'r'), 'drugs': int(r.drugs)}
    for _, r in Sc.iterrows():
        out['range_model_scores'][f'{r.target_range_from}|{r.version}'] = {k: v(r, k) for k in
                                                                          ['median_raw', 'median_selective', 'axis_share_pooled', 'perfect_share']}
    return out


def holm():
    S = pd.read_csv(O / 'holm' / 'holm_summary.csv')
    return {f'{r.variant}|{r.family}': {'survive': int(r.survive_holm_0_05) if 'survive_holm_0_05' in r else int(r['survive_holm_0.05']),
                                         'tests': int(r.tests), 'fail': r.fail if isinstance(r.fail, str) else ''}
            for _, r in S.iterrows()}


def ceiling():
    B = pd.read_csv(O / 'ceiling' / 'cfa_implied_bias.csv')
    T = pd.read_csv(O / 'ceiling' / 'typical_drug.csv')
    H = pd.read_csv(O / 'ceiling' / 'headline.csv')
    out = {'cfa_bias': {}, 'typical_drug': {}, 'headline': {}}
    for _, r in B[B.configuration == 'all'].iterrows():
        out['cfa_bias'][f'{r.fit}|{r.quantity}'] = {**{k: v(r, k) for k in ['implied_estimate', 'general_truth', 'relative_bias']},
                                                   'cv_mse': r.cv_mse_of_matching_measurement_model}
    for _, r in T.iterrows():
        out['typical_drug'][f'{r["class"]}|{r.quantity}'] = v(r, 'value')
    for _, r in H.iterrows():
        d = {'est': round(r.estimate, 4)}
        for a in ['drug_cluster', 'two_way']:
            if pd.notna(r[f'{a}_lo']):
                d[a] = [round(r[f'{a}_lo'], 4), round(r[f'{a}_hi'], 4)]
        out['headline'][f'{r.quantity}|{r["class"]}'] = d
    return out


def rankings():
    T = pd.read_csv(O / 'rankings' / 'rank_intervals.csv')
    C = pd.read_csv(O / 'rankings' / 'ranking_correlations.csv')
    S = json.loads((O / 'rankings' / 'summary.json').read_text())
    out = {'models': {}, 'correlations': {}, 'resolution': S['resolution']}
    for _, r in T.iterrows():
        out['models'][f'{r.metric}|{r.model}'] = {'score': v(r, 'score'), 'rank': r['rank'], 'rank_95': [r.rank_lo, r.rank_hi],
                                                  'paper_rank': r.paper_rank}
    for _, r in C.iterrows():
        out['correlations'][f'{r.ranking_a}~{r.ranking_b}'] = v(r, 'spearman')
    return out


def main():
    K = {'note': 'Round-2 revision analyses (scripts 77-89). est = point estimate; lo/hi = 95% two-way (cell x drug) '
                 'bootstrap interval unless stated in SUMMARY.md.'}
    for name, fn in [('task1_capture', capture), ('task2_reliability', reliability), ('task3_attenuation', attenuation),
                     ('task4_symmetric_selboth', symmetric), ('task5_equivalence', equivalence),
                     ('task6_source_reference', source_ref), ('task7_harmonization', harmonization), ('task8_holm', holm),
                     ('task9_ceiling', ceiling), ('task10_rankings', rankings)]:
        try:
            K[name] = fn()
        except FileNotFoundError as e:
            K[name] = {'missing': str(e)}
            print('missing', name, e)
    lib.write_json(O / 'key_numbers.json', K)
    print('written', O / 'key_numbers.json', {k: len(v) if isinstance(v, dict) else 1 for k, v in K.items()})


if __name__ == '__main__':
    main()
