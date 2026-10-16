"""Round 4 (editor 5a item 4; R1-M4, R3-M3a/b): data for the supplementary known-answer table (Table S3), from the
existing outputs of 100b_known_answer_simulation.py (no simulation is re-run).

Per panel (150 drugs x 500 cells; 30 x 100) and predictor of known composition, for every statistic (axis capture A
and selective capture S are the primary ones): the panel-level truth (theta_design: capture computed with the panel's
own leave-drug-out axis from D drugs, at N = 20,000 cells pooled over panels, i.e. what the estimator targets for that
panel size) and the latent truth (theta_latent: the same with the true axis a, the predictor's true composition), the
difference between them (for S, the leakage of an imperfect axis), the bias, RMSE and two-way bootstrap 95% interval
coverage against each (300 replicates, B = 500), interval width relative to the replicate s.d., and for S the
reliability-corrected estimator's bias against the latent truth (100b's corrected_S, = box1.corrected_selective_capture).
Notes (JSON): the generating model, the estimands and what the generating model omits.
Outputs: results/revision_round4/simulation/known_answer_table.csv, known_answer_notes.json.
"""
import sys

sys.dont_write_bytecode = True
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'results' / 'revision_round4' / 'simulation'
LABEL = {'agnostic': 'drug-agnostic', 'agnostic_weak': 'drug-agnostic, weak', 'axis_heavy': 'axis-heavy',
         'balanced': 'balanced', 'selective': 'selective only', 'tissue_only': 'tissue only',
         'eq_agnostic': 'equal raw: drug-agnostic', 'eq_mixed': 'equal raw: mixed', 'eq_selective': 'equal raw: mostly selective'}


def sim_constants():
    """PAR, PRED and the run constants of 100b, read from its source without executing it."""
    import ast
    tree = ast.parse((HERE / '100b_known_answer_simulation.py').read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('PAR', 'PRED', 'N_BIG', 'B_REC', 'B_ATT', 'TARGET_RAW', 'SEED', 'LEARNED_LIKE'):
                out[name] = ast.literal_eval(node.value)
    return out


def main():
    C = sim_constants()
    eq = json.loads((SIM / 'eq_sigma.json').read_text())
    R = pd.read_csv(SIM / 'fig_1_2_recovery_coverage.csv')
    Cr = pd.read_csv(SIM / 'fig_5b_S_reliability_correction.csv')
    rows = []
    for r in R.itertuples():
        al, be, ga, sg = C['PRED'][r.predictor]
        sg = eq.get(r.predictor, sg) if sg is None else sg
        n = r.replicates
        cl = r.coverage_latent
        row = {'panel': f'{r.D} drugs x {r.N} cells', 'D': r.D, 'N': r.N, 'predictor': r.predictor,
               'predictor_label': LABEL.get(r.predictor, r.predictor), 'alpha_axis': al, 'beta_selective': be,
               'gamma_tissue_by_drug': ga, 'sigma_noise': sg, 'stat': r.stat, 'primary_stat': r.stat in ('A', 'S'),
               'replicates': n, 'bootstrap_B': C['B_REC'], 'truth_panel': r.truth_design, 'truth_latent': r.truth_latent,
               'panel_minus_latent': r.truth_design - r.truth_latent, 'mean_estimate': r.mean_estimate,
               'bias_vs_panel': r.bias_design, 'bias_vs_panel_mcse': r.bias_design_mcse, 'rmse_vs_panel': r.rmse_design,
               'bias_vs_latent': r.bias_latent, 'rmse_vs_latent': r.rmse_latent, 'sd_estimate': r.sd_estimate,
               'coverage_vs_panel': r.coverage_design, 'coverage_vs_panel_mcse': r.coverage_mcse,
               'coverage_vs_latent': cl, 'coverage_vs_latent_mcse': float(np.sqrt(cl * (1 - cl) / n)),
               'mean_ci_width': r.mean_ci_width, 'ci_width_over_nominal': r.mean_ci_width / (2 * 1.959964 * r.sd_estimate),
               'mean_axis_split_half': r.mean_axis_split_half}
        if r.stat == 'S':
            c = Cr[(Cr.D == r.D) & (Cr.predictor == r.predictor) & (Cr.estimate == 'corrected')]
            if len(c):
                c = c.iloc[0]
                row.update({'S_corrected_mean': c['mean'], 'S_corrected_bias_vs_latent': c.bias_vs_latent,
                            'S_corrected_rmse_vs_latent': c.rmse_vs_latent})
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(SIM / 'known_answer_table.csv', index=False, float_format='%.5g')

    P = T[T.primary_stat]
    S, A = P[P.stat == 'S'], P[P.stat == 'A']
    rng = lambda x: [float(x.min()), float(x.max())]  # noqa: E731
    notes = {
        'source': {'script': 'scripts/100b_known_answer_simulation.py (outputs read, not re-run)',
                   'files': ['fig_1_2_recovery_coverage.csv', 'fig_5b_S_reliability_correction.csv', 'truth.csv',
                             'eq_sigma.json'], 'table_script': 'scripts/102c_known_answer_table.py'},
        'generating_model': {
            'axis': 'a_c = sqrt(lam) tau_{t_c} + sqrt(1 - lam) eps_c (one shared general-sensitivity axis, variance 1; '
                    'tissue share lam; K tissues with fixed effects tau and Zipf-like frequencies)',
            'response': 'y_dc = l_d a_c + u_{d,t_c} + s_dc + e_dc; z = rank-normal(y) per drug',
            'parameters': C['PAR'],
            'loadings': f"l_d ~ U({C['PAR']['l_lo']}, {C['PAR']['l_hi']}) (perfect-predictor axis share about 0.2)",
            'predictors': 'p_dc = alpha a_c + beta s_dc + gamma m_dc + sigma xi_dc, m_dc = l_d sqrt(lam) tau_{t_c} + '
                          'u_{d,t_c} (the tissue-by-drug mean of the noise-free profile); xi independent N(0, 1)',
            'predictor_compositions': {k: list(v) for k, v in C['PRED'].items()}, 'equal_raw_sigmas': eq,
            'panels': ['150 drugs x 500 cells', '30 drugs x 100 cells'], 'replicates': 300,
            'bootstrap': f"two-way (cell weights inside every correlation, drug multiplicities when pooling), B = {C['B_REC']}",
            'truth_cells': C['N_BIG']},
        'estimands': {
            'panel_level': ("theta_panel (100b 'theta_design'): A and S computed with the panel's own leave-drug-out axis "
                            'built from its D drugs, at N = 20,000 cells, pooled over panels: the quantity the estimator '
                            'targets for that panel size. It includes the leakage of the imperfect axis into S.'),
            'latent': ("theta_latent: A and S computed with the true axis a: the predictor's true composition. "
                       'theta_panel - theta_latent for S is the leakage of the imperfect axis (box1.axis_leakage_bound: exact for '
                       'drug-agnostic predictors, an upper bound otherwise).'),
            'coverage_statement': 'Coverage is reported against both; the paper\'s "covered the truth" refers to the '
                                  'panel-level truth.'},
        'summary': {
            'A_coverage_vs_panel_range': rng(A.coverage_vs_panel), 'A_coverage_vs_latent_range': rng(A.coverage_vs_latent),
            'S_coverage_vs_panel_range': rng(S.coverage_vs_panel), 'S_coverage_vs_latent_range': rng(S.coverage_vs_latent),
            'S_coverage_vs_latent_by_predictor': {f'{r.predictor}|{r.D}x{r.N}': float(r.coverage_vs_latent) for r in S.itertuples()},
            'A_bias_vs_panel_max_abs': float(A.bias_vs_panel.abs().max()),
            'S_bias_vs_panel_max_abs': float(S.bias_vs_panel.abs().max()),
            'S_bias_vs_latent_max_abs': float(S.bias_vs_latent.abs().max()),
            'S_ci_width_over_nominal_range': rng(S.ci_width_over_nominal),
            'A_ci_width_over_nominal_range': rng(A.ci_width_over_nominal),
            'S_corrected_bias_vs_latent_max_abs': float(S.S_corrected_bias_vs_latent.abs().max())},
        'omissions': [
            'cell-level noise shared across drugs (plate, batch or seeding effects; plate ICC 12-34% in the real screens): '
            'e_dc is independent across drugs, so the axis error is independent of every drug\'s noise',
            'selective signals correlated within mechanism classes: s_dc is independent across drugs, so the '
            'leave-drug-out axis does not absorb class-shared selectivity (which would break the independence the '
            'leakage formula needs)',
            'activity-dependent reliability: every drug is active, with the same noise variance and loadings in '
            'U(0.30, 0.65); no inactive drugs with near-zero variance',
            'a single shared axis: no second shared factor beyond the fixed tissue structure',
            'Gaussian, homoscedastic errors; complete drug x cell matrices (no missing pairs) in the recovery part; '
            'censoring only in its own part (fig_6_censoring.csv)',
            'no curve fitting, dose-range, readout or processing differences (a monotone distortion and a capped tail '
            'appear only in the attenuation part)',
            'predictors are fixed linear combinations of the generating components plus independent noise: no model is '
            'trained, so there is no overfitting, model selection, or dependence between predictions and measurement '
            'noise (as in pair or drug-blind splits)',
            'cell lines are exchangeable draws (no lineage-structured sampling beyond tissue frequencies)'],
        'separate_parts': {'equal_raw_separation': 'fig_3_equal_raw_separation.csv', 'leakage': 'fig_5_axis_reliability_leakage.csv',
                           'censoring': 'fig_6_censoring.csv', 'attenuation_test': 'fig_7_attenuation_selboth.csv'}}
    (SIM / 'known_answer_notes.json').write_text(json.dumps(notes, indent=1))
    pd.set_option('display.width', 250)
    print(P[['panel', 'predictor', 'stat', 'truth_panel', 'truth_latent', 'bias_vs_panel', 'bias_vs_latent', 'coverage_vs_panel',
             'coverage_vs_latent', 'ci_width_over_nominal']].round(4).to_string(index=False))
    print(json.dumps(notes['summary'], indent=1))


if __name__ == '__main__':
    main()
