"""Write results/revision_retrain/SUMMARY.md from the retraining outputs (57-59): every key number per training
setting with its source file, plus runtimes. Also writes key_numbers.json (all quoted numbers, machine-readable).

Interpretation statements are computed from the numbers (thresholds stated inline), so the text cannot drift from
the CSVs.
"""
import json
import os
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / 'results' / 'revision_retrain'
RUNS = OUT / 'drevalpy_runs'
SETTING_TEXT = {
    'GDSC2_AUC': 'GDSC2, CurveCurator AUC (uncensored endpoint)',
    'GDSC2_LNIC50': 'GDSC2, CurveCurator ln IC50',
    'CTRPv2_LNIC50': 'CTRPv2, CurveCurator ln IC50 (DrEval leaderboard setting and splits)',
    'CTRPv2_AUC': 'CTRPv2, CurveCurator AUC',
    'GDSC1_AUC': 'GDSC1, CurveCurator AUC',
}
ORDER = ['GDSC2_AUC', 'GDSC2_LNIC50', 'CTRPv2_LNIC50', 'CTRPv2_AUC', 'GDSC1_AUC']
KEY_MODELS = ['ref:source_measurement', 'FitnessOnlyRidge', 'NaiveMeanEffectsPredictor', 'NaiveTissueDrugMeanPredictor', 'RandomForest',
              'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting', 'GradientBoostingZ', 'GradientBoostingSelective',
              'SimpleNeuralNetworkSelective', 'PerDrugRidgeRaw', 'PerDrugRidgeSelective']
LB_KEY = ['LB:EnsembleMF', 'LB:RandomForest', 'LB:GradientBoosting', 'LB:SimpleNeuralNetwork', 'LB:DIPK',
          'LB:NaiveTissueDrugMeanPredictor', 'LB:NaiveMeanEffectsPredictor']
KEY = {}
DRUG_AGNOSTIC = ['FitnessOnlyRidge', 'FitnessOnlyRidgeDm', 'NaiveMeanEffectsPredictor', 'LB:NaiveMeanEffectsPredictor',
                 'LB:NaiveTissueMeanPredictor']


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'NA'
    return f'{x:.{nd}f}'


def ci(row, key, nd=3):
    v = row.get(key, np.nan)
    lo, hi = row.get(f'{key}_lo', np.nan), row.get(f'{key}_hi', np.nan)
    if not np.isfinite(v):
        return 'NA'
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f(v, nd)
    return f'{f(v, nd)} ({f(lo, nd)} to {f(hi, nd)})'


def pfmt(p):
    if p is None or not np.isfinite(p):
        return 'NA'
    return f'{p:.1e}' if p < 1e-3 else f'{p:.3f}'


def table(df, cols, headers):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for _, r in df.iterrows():
        lines.append('| ' + ' | '.join(str(c(r)) if callable(c) else str(r[c]) for c in cols) + ' |')
    return '\n'.join(lines)


def put(setting, name, value):
    KEY.setdefault(setting, {})[name] = value
    return value


def verdict_share(r):
    """Does the loss run through the fitness term? Based on the pooled share and its 95% interval."""
    v, lo, hi = r['share_of_drop_from_fitness'], r.get('share_of_drop_from_fitness_lo'), \
        r.get('share_of_drop_from_fitness_hi')
    dr, drlo, drhi = r['delta_raw'], r.get('delta_raw_lo'), r.get('delta_raw_hi')
    if np.isfinite(drhi) and drhi > 0 and np.isfinite(drlo) and drlo < 0:
        return 'no clear raw loss (Δ raw interval includes 0), so the share is not interpretable'
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 'share not estimable'
    if lo > 0.5:
        return 'mostly through the fitness term'
    if hi < 0.5:
        return 'mostly NOT through the fitness term (the loss is in the selective term)'
    return 'undetermined (interval spans 0.5)'


def verdict_delta(r, key):
    v, lo, hi = r.get(key), r.get(f'{key}_lo'), r.get(f'{key}_hi')
    lo90, hi90 = r.get(f'{key}_lo90', np.nan), r.get(f'{key}_hi90', np.nan)
    if not np.isfinite(v):
        return 'NA'
    if np.isfinite(lo90) and lo90 > -0.05 and hi90 < 0.05:
        eq = 'equivalent within ±0.05'
    else:
        eq = 'not shown equivalent within ±0.05'
    if hi < 0:
        d = 'reduced'
    elif lo > 0:
        d = 'increased'
    else:
        d = 'no significant change'
    return f'{d}; {eq}'


def reading(s, Ws, PPs, Cs, orc):
    out = []
    learned = Ws[Ws.model.isin(['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting'])]
    if s == 'CTRPv2_LNIC50':
        learned = Ws[Ws.model_class.str.contains('classical|neural', regex=True) & (Ws.model != 'LB:SVR')]
    if len(learned):
        out.append(f"- (i) Learned raw-response models: fitness-mediated share of per-drug accuracy "
                   f"{f(learned.mediated_share.min(), 2)}-{f(learned.mediated_share.max(), 2)} (median "
                   f"{f(learned.mediated_share.median(), 2)}) against {f(orc.oracle_mean_rzg2, 2)} for a perfect "
                   f"predictor; ratio of the median to the oracle {f(learned.mediated_share.median() / orc.oracle_mean_rzg2, 1)}.")
    for name in ['FitnessOnlyRidge', 'NaiveTissueDrugMeanPredictor', 'PerDrugRidgeRaw', 'PerDrugRidgeSelective']:
        r = Ws[Ws.model == name]
        if len(r):
            r = r.iloc[0]
            out.append(f"- {name}: median per-drug raw {f(r.median_raw)}, selective {f(r.median_selective)}, "
                       f"model-axis-adjusted {f(r.median_adjusted)}, fitness-mediated share {ci(r, 'mediated_share', 2)}.")
    if len(PPs):
        for _, r in PPs.iterrows():
            out.append(f"- (ii) {r.target}: Δ raw {ci(r, 'delta_raw')}; share of the loss via the fitness term "
                       f"{ci(r, 'share_of_drop_from_fitness', 2)} → {verdict_share(r)}; fitness term retained "
                       f"{ci(r, 'mediated_retained', 2)}, alignment ratio {ci(r, 'alignment_ratio', 2)}, target-loading "
                       f"ratio {ci(r, 'loading_ratio', 2)}.")
            out.append(f"- (iii) {r.target}: pooled selective-BOTH {f(r.mean_sel_BOTH_s)} → {f(r.mean_sel_BOTH_t)}, "
                       f"Δ {ci(r, 'delta_sel_BOTH')} → {verdict_delta(r, 'delta_sel_BOTH')}.")
    if len(Cs):
        t = Cs[(Cs.family == 'target effect within model class') & (Cs.panel == 'main') &
               Cs.metric.isin(['selective (= selective-BOTH within)', 'selective-BOTH', 'model-axis-adjusted'])]
        for name, g in t.groupby('contrast', sort=False):
            parts = []
            for _, r in g.iterrows():
                sig = 'Holm P<0.05' if r.holm_p_family_setting < 0.05 else 'n.s.'
                parts.append(f"{r['where']} {r.metric.split(' (')[0]} {f(r.mean_diff)} [{f(r.mean_diff_lo)}, "
                             f"{f(r.mean_diff_hi)}] {sig}")
            out.append(f"- (iv) {name}: " + '; '.join(parts) + '.')
    return out


def answers(settings, W, PP, C, PS):
    """Top-level answers, computed per setting from the CSVs."""
    out = ['## Answers', '',
           '**Q1. Do the paper\'s findings replicate when training on GDSC2 and on AUC?** Three findings are tested '
           'per training setting: (a) learned models\' per-drug accuracy is dominated by the fitness axis (fitness-'
           'mediated share far above the perfect-predictor reference mean r_zg²); (b) the cross-dataset loss of '
           'per-drug accuracy runs through the fitness term (pooled share of the loss with 95% interval above 0.5); '
           '(c) selectivity is retained across datasets (pooled change in selective-BOTH within ±0.05).', '']
    for s in settings:
        Ws = W[(W.setting == s) & (W.panel == 'main')]
        if s == 'CTRPv2_LNIC50':
            learned = Ws[Ws.model_class.str.contains('classical|neural', regex=True) & (Ws.model != 'LB:SVR')]
            grp = 'published leaderboard models (21, as 38)'
        else:
            learned = Ws[Ws.model.isin(['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting'])]
            grp = 'DrEval learned models (raw response)'
        fallback = False
        if not len(learned):            # no DrEval learned model run in this setting: use the raw-target ridge
            learned, fallback = Ws[Ws.model == 'PerDrugRidgeRaw'], True
        if not len(learned):
            continue
        orc = Ws.loc[Ws.drugs.idxmax()].oracle_mean_rzg2
        med = learned.mediated_share.median()
        who = ('PerDrugRidgeRaw (no DrEval learned model was run in this setting)' if fallback
               else f'{len(learned)} learned models')
        a = (f"(a) fitness-mediated share of {who} {f(learned.mediated_share.min(), 2)}-"
             f"{f(learned.mediated_share.max(), 2)} (median {f(med, 2)}) vs perfect predictor {f(orc, 2)}: "
             + ('**replicates** (median ≥ 2× the reference)' if med >= 2 * orc else
                '**does not replicate** (median < 2× the reference)'))
        P = PP[(PP.setting == s) & (PP.panel == 'main') & (PP.group == grp)]
        if fallback or not len(P):
            P = PS[(PS.setting == s) & (PS.panel == 'main') & (PS.model == 'PerDrugRidgeRaw')]
        b, c = [], []
        for _, r in P.iterrows():
            b.append(f"{r.target} {ci(r, 'share_of_drop_from_fitness', 2)} [{verdict_share(r)}]")
            c.append(f"{r.target} Δ {ci(r, 'delta_sel_BOTH')} [{verdict_delta(r, 'delta_sel_BOTH')}]")
        out += [f'- **{SETTING_TEXT.get(s, s)}.** {a}.',
                '  (b) share of the loss via the fitness term: ' + ('; '.join(b) if b else 'NA') + '.',
                '  (c) pooled selective-BOTH change: ' + ('; '.join(c) if c else 'NA') + '.']
    out += ['', '**Q2. Is the selective-target advantage present within model class under the leakage-robust '
                'metrics?** Mean per-drug difference (selective target minus raw target, same class, features, '
                'splits and tuning), two-way bootstrap 95% interval, Holm within family:', '']
    tc = C[(C.family == 'target effect within model class') &
           C.metric.isin(['selective (= selective-BOTH within)', 'selective-BOTH', 'model-axis-adjusted'])]
    tc = tc[(tc.panel == 'main') | tc.contrast.str.contains('DepMap')]
    for s in settings:
        g = tc[tc.setting == s]
        if not len(g):
            continue
        out.append(f'- **{SETTING_TEXT.get(s, s)}**')
        for name, h in g.groupby('contrast', sort=False):
            n_pos = int(((h.mean_diff > 0) & (h.holm_p_family_setting < 0.05)).sum())
            n_neg = int(((h.mean_diff < 0) & (h.holm_p_family_setting < 0.05)).sum())
            parts = [f"{r['where']}, {r.metric.split(' (')[0]}: {f(r.mean_diff)} ({f(r.mean_diff_lo)} to "
                     f"{f(r.mean_diff_hi)})" + ('*' if r.holm_p_family_setting < 0.05 else '')
                     for _, r in h.iterrows()]
            out.append(f"  - {name}: {n_pos} of {len(h)} comparisons favour the selective target and {n_neg} the "
                       f"raw target after Holm (* = Holm P < 0.05). " + '; '.join(parts) + '.')
    out.append('')
    return out


def consistency(W, PP):
    """The scoring of 59 against the paper's existing numbers (read-only: results/dreval, dreval_crosslab_leaderboard)."""
    res = PROJECT / 'results'
    out = ['## Consistency with the paper\'s existing analyses (leaderboard predictions, CTRPv2 ln IC50)', '']
    try:
        ms = pd.read_csv(res / 'dreval' / 'model_summary.csv').set_index('model')
        un = pd.read_csv(res / 'dreval_crosslab_leaderboard' / 'uncertainty.csv')
    except FileNotFoundError:
        return []
    w = W[(W.setting == 'CTRPv2_LNIC50') & (W.panel == 'main')].set_index('model')
    if not len(w):
        return []
    rows = []
    for m in ['EnsembleMF', 'RandomForest', 'GradientBoosting', 'NaiveTissueDrugMeanPredictor', 'DIPK']:
        if f'LB:{m}' in w.index and m in ms.index:
            rows.append({'model': m, 'paper_raw': ms.loc[m, 'raw'], 'here_raw': w.loc[f'LB:{m}', 'median_raw'],
                         'paper_sel': ms.loc[m, 'selective'], 'here_sel': w.loc[f'LB:{m}', 'median_selective'],
                         'paper_share': ms.loc[m, 'mediated_share'], 'here_share': w.loc[f'LB:{m}', 'mediated_share']})
    for ref, ours in [('fitness_only', 'FitnessOnlyRidge'), ('ridge_raw_lm', 'PerDrugRidgeRaw'),
                      ('ridge_sel_lm', 'PerDrugRidgeSelective')]:
        if ref in ms.index and ours in w.index:
            rows.append({'model': f'{ref} (paper, DepMap landmark genes) vs {ours} (drevalpy, DrEval features)',
                         'paper_raw': ms.loc[ref, 'raw'], 'here_raw': w.loc[ours, 'median_raw'],
                         'paper_sel': ms.loc[ref, 'selective'], 'here_sel': w.loc[ours, 'median_selective'],
                         'paper_share': ms.loc[ref, 'mediated_share'], 'here_share': w.loc[ours, 'mediated_share']})
    if rows:
        t = pd.DataFrame(rows)
        out += [table(t, ['model', lambda r: f"{f(r.paper_raw)} / {f(r.here_raw)}",
                          lambda r: f"{f(r.paper_sel)} / {f(r.here_sel)}",
                          lambda r: f"{f(r.paper_share, 2)} / {f(r.here_share, 2)}"],
                      ['model', 'raw median (paper / here)', 'selective median (paper / here)',
                       'fitness-mediated share (paper / here)']), '',
                'Paper values: `results/dreval/model_summary.csv` (28; 739 common cells). Here: `within_summary.csv` '
                '(745 common cells of the drevalpy panel). The drevalpy re-implementations of this study\'s references '
                'use DrEval\'s own expression file instead of DepMap, so they are close but not identical.', '']
    p = PP[(PP.setting == 'CTRPv2_LNIC50') & (PP.group == 'published leaderboard models (21, as 38)')]
    u = un[un.model == 'pooled published models'].set_index('target')
    if len(p):
        out += [table(p, ['target', lambda r: f"{f(u.loc[r.target, 'share'], 2)} ({f(u.loc[r.target, 'share_lo'], 2)} "
                                              f"to {f(u.loc[r.target, 'share_hi'], 2)})" if r.target in u.index else 'NA',
                          lambda r: ci(r, 'share_of_drop_from_fitness', 2),
                          lambda r: f(u.loc[r.target, 'mediated_retained'], 2) if r.target in u.index else 'NA',
                          lambda r: f(r.mediated_retained, 2)],
                      ['target', 'pooled share, paper (drug bootstrap)', 'pooled share, here (two-way bootstrap)',
                       'fitness term retained, paper', 'here']), '',
                'Paper: `results/dreval_crosslab_leaderboard/uncertainty.csv` (38). The point estimates agree; the '
                'two-way bootstrap intervals are wider because cell-line sampling is added (R1-M6).', '']
    return out


def run_status():
    """Complete (10/10 splits) and incomplete drevalpy models per setting, from the run directories."""
    rows = []
    for setting in ORDER:
        base = next((d for d in (RUNS / setting).glob('*/LCO')), None) if (RUNS / setting).exists() else None
        done, part = [], []
        if base is not None:
            for md in sorted(p for p in base.iterdir() if p.is_dir() and p.name != 'splits'):
                n = len(list((md / 'predictions').glob('predictions_split_*.csv')))
                (done if n >= 10 else part).append(md.name if n >= 10 else f'{md.name} ({n}/10)')
        rows.append({'setting': setting, 'complete': ', '.join(done) or 'none',
                     'incomplete or not run': ', '.join(part) or '-'})
    return pd.DataFrame(rows)


def runtimes():
    rows = []
    for tf in RUNS.glob('*/timings/*.csv'):
        rows.append(pd.read_csv(tf))
    if not rows:
        return None
    t = pd.concat(rows)
    g = t.groupby(['setting', 'model']).agg(splits=('split', 'nunique'), total_h=('total_s', lambda x: x.sum() / 3600),
                                            median_split_min=('total_s', lambda x: x.median() / 60),
                                            tuning_share=('tuning_s', 'sum')).reset_index()
    g['tuning_share'] = g.tuning_share / (g.total_h * 3600)
    return g


def main():
    rd = lambda n: pd.read_csv(OUT / n) if (OUT / n).exists() else pd.DataFrame()  # noqa: E731
    W, N, PS, PP, C = rd('within_summary.csv'), rd('normalized_pearson.csv'), rd('paired_summary.csv'), \
        rd('paired_pooled.csv'), rd('contrasts.csv')
    NV, PROV, NEW = rd('normalized_validation.csv'), rd('cell_provenance.csv'), rd('new_cells_unpaired.csv')
    run = json.loads((OUT / 'score_run.json').read_text()) if (OUT / 'score_run.json').exists() else {}
    man = json.loads((OUT / 'inputs' / 'manifest.json').read_text())
    settings = [s for s in ORDER if s in set(W.setting)]
    L = ['# Retraining in drevalpy: like-for-like comparisons and a second training dataset and endpoint',
         '',
         'Workstream for Reviewer 1 M4/M5 and editor items 4-5 (revision round 1). Generated by '
         '`scripts/60_retrain_summary.py` from the CSVs in this directory (scripts 57-59); every number below is in '
         '`key_numbers.json` and in the named source file.',
         '']
    L += answers(settings, W, PP, C, PS)
    DF = rd('dataset_fitness.csv')
    if len(DF):
        DF = DF.sort_values(['endpoint', 'median_fitness_r2'], ascending=[True, False])
        for _, r in DF.iterrows():
            put('datasets', f'{r.dataset}:{r.endpoint}', {'median_fitness_r2': r.median_fitness_r2,
                                                          'mean_fitness_r2': r.mean_fitness_r2})
        L += ['### How fitness-dominated is each DrEval dataset and endpoint?', '',
              table(DF, ['dataset', 'endpoint', 'drugs', 'cells', lambda r: f(r.median_fitness_r2),
                         lambda r: f(r.mean_fitness_r2)],
                    ['dataset', 'endpoint', 'drugs', 'cells', 'median fitness R² (per drug)',
                     'mean r_zg² (= perfect-predictor fitness-mediated share)']), '',
              'Source: `dataset_fitness.csv` (full datasets; leave-drug-out axis; sensitivity orientation).', '']
    # ------------------------------------------------------------------ what was run
    L += ['## What was run', '',
          '- **Pipeline.** drevalpy 1.6.0 (`.venv-dreval`), leave-cell-line-out (LCO), 10 splits, DrEval\'s own '
          'features (landmark-gene expression of the training dataset\'s `gene_expression.csv`, Morgan '
          'fingerprints), DrEval\'s hyperparameter grids and tuning (validation RMSE, then refit on '
          'train+validation), and drevalpy\'s cross-study prediction on the other DrEval datasets with the same '
          'endpoint. `58_retrain_drevalpy.py` reproduces `drug_response_experiment`\'s per-split loop body with '
          'drevalpy\'s own functions so that splits run as parallel single-threaded processes (at most 6 at once).',
          f"- **Splits.** CTRPv2 ln IC50: DrEval's published leaderboard split files. Other settings: drevalpy's own "
          f"split procedure (seed 42), which reproduces the published CTRPv2 splits exactly "
          f"({man['split_procedure_reproduces_published_CTRPv2_splits']['cells']} of "
          f"{man['split_procedure_reproduces_published_CTRPv2_splits']['cells']} cell lines; `inputs/manifest.json`).",
          '- **This study\'s predictors as drevalpy models** (subclasses of `DRPModel`, registered in drevalpy\'s '
          'model factory; `58_retrain_drevalpy.py`): `FitnessOnlyRidge` (kernel ridge from expression to each '
          'training cell\'s mean rank-normal response, broadcast to every drug), `PerDrugRidgeRaw` (per-drug '
          'kernel ridge on the training cells\' rank-normal response z) and `PerDrugRidgeSelective` (per-drug kernel '
          'ridge on the selective residual of z given the training cells\' leave-drug-out axis). All targets are '
          'derived inside `train()` from the cells drevalpy passes for training. Regularisation is chosen per drug by '
          'closed-form leave-one-out, so the grid given to drevalpy has one configuration. `...Dm` variants use '
          'DepMap 24Q4 expression (top 5,000 variance genes) instead of DrEval\'s features (feature-source control).',
          '- **Global models on the selective target, same class and tuning as DrEval.** `GradientBoostingZ` and '
          '`GradientBoostingSelective` are DrEval\'s `GradientBoosting` (HistGradientBoosting on landmark expression '
          '+ 128-bit Morgan fingerprints, DrEval\'s 6-configuration grid) trained on z or on the selective residual '
          'computed from each split\'s training cells; predictions are placed back on the response scale with the '
          'training-cell mean and SD of each drug.',
          '- **DrEval models retrained** on the new settings: NaiveMeanEffectsPredictor, NaiveTissueDrugMeanPredictor, '
          'RandomForest, SimpleNeuralNetwork, EnsembleMF (the factorization model; best on the CTRPv2 leaderboard), '
          'GradientBoosting.',
          '- **Scoring** (`59_retrain_score.py`): out-of-fold predictions only; per drug raw r, selective-T (partial '
          'given the evaluation screen\'s leave-drug-out axis g), selective-BOTH (given g of the training and of the '
          'evaluation screen), model-axis-adjusted (given g and the model\'s own leave-drug-out axis of per-drug '
          'z-scored predictions), r_pg, r_zg, mediated (r_pg r_zg) and selective (raw - r_pg r_zg) terms; matched '
          'cells per drug for the paired analysis (`35_dreval_crossstudy.score_pair` logic). Uncertainty: two-way '
          'bootstrap (B = 1000; cell lines and drugs resampled independently, the cell draw shared by all drugs), '
          'percentile 95% intervals; contrasts use the drugs common to every model of the panel, bootstrap-SE Wald '
          'P values with Holm adjustment within pre-specified families, and drug-level Wilcoxon P values.',
          '- **Validation.** On the leaderboard predictions the scoring reproduces the paper\'s numbers and DrEval\'s '
          'own normalized Pearson ' + (f"(largest absolute difference to the reported leaderboard values "
                                         f"{(NV.reported - NV.reimplemented_all_pairs).abs().max():.1e}, "
                                         f"{len(NV)} models; `normalized_validation.csv`)." if len(NV) else '.'),
          '- **Deviations.** (1) SimpleNeuralNetwork\'s DataLoaders load data in-process instead of with 8 worker '
          'processes (server process limit; model, batches and their order unchanged). (2) Thread counts capped at 1 '
          '(RandomForest\'s `n_jobs=-1` via `LOKY_MAX_CPU_COUNT=1`), which changes speed only. (3) DrEval\'s '
          'expression files differ between the CTRP/CCLE family (TPM-like) and the GDSC family (log2 microarray-like); '
          'drevalpy\'s cross-study predictions apply a model to the target dataset\'s own file, so for cells absent '
          'from the training dataset they cross platforms. These fold-averaged predictions are reported only as '
          'provenance and a supplementary unpaired score; the paired analysis uses out-of-fold predictions.',
          '']
    st = run_status()
    L += ['### Run status (drevalpy runs of this workstream)', '',
          table(st, ['setting', 'complete', 'incomplete or not run'], ['setting', 'complete (10/10 splits)',
                                                                     'incomplete / not run']), '',
          'Settings or models not listed as complete were skipped for time (orchestrator priority: GDSC2 AUC first, '
          'other settings only if they finished within about two hours). CTRPv2 ln IC50 additionally uses DrEval\'s '
          '25 released leaderboard models (`LB:`).', '']
    rt = runtimes()
    if rt is not None:
        rt = rt.sort_values(['setting', 'total_h'], ascending=[True, False])
        L += ['### Runtimes (single-threaded processes; wall-clock per split and summed over splits)', '',
              table(rt, ['setting', 'model', 'splits', lambda r: f(r.median_split_min, 1), lambda r: f(r.total_h, 2),
                         lambda r: f(r.tuning_share, 2)],
                    ['setting', 'model', 'splits done', 'median min/split', 'total h', 'tuning share']), '',
              f"Total compute: {rt.total_h.sum():.1f} process-hours. Source: `drevalpy_runs/<setting>/timings/`.", '']
    # ------------------------------------------------------------------ per setting
    for s in settings:
        info = run.get(s, {})
        Ws = W[(W.setting == s) & (W.panel == 'main')].copy()
        main_panel = info.get('panels', {}).get('main', {})
        L += [f'## {SETTING_TEXT.get(s, s)} (`{s}`)', '']
        L += [f"Panel: {len(Ws)} models scored on {main_panel.get('cells', 'NA')} common cell lines and "
              f"{main_panel.get('drugs_within', 'NA')} drugs; median fitness R² of the observed profiles "
              f"{f(main_panel.get('median_fitness_r2_within', np.nan))}. Incomplete models (not scored): "
              f"{info.get('incomplete_models') or 'none'}.", '']
        # (i) within + (v) normalized
        Ns = N[N.setting == s].set_index('model') if len(N) else pd.DataFrame()
        Ws['norm'] = Ws.model.map(lambda m: f"{f(Ns.loc[m, 'normalized_pearson'])} ± {f(Ns.loc[m, 'split_sd'])}"
                                  if m in Ns.index else 'NA')
        Ws = Ws.sort_values('median_selective', ascending=False)
        orc = Ws.loc[Ws.drugs.idxmax()]
        put(s, 'oracle_mean_rzg2', [orc.oracle_mean_rzg2, orc.oracle_mean_rzg2_lo, orc.oracle_mean_rzg2_hi])
        for _, r in Ws.iterrows():
            put(s, f'within:{r.model}', {k: r.get(k, np.nan) for k in [
                'median_raw_boot', 'median_raw_boot_lo', 'median_raw_boot_hi', 'median_selective_boot',
                'median_selective_boot_lo', 'median_selective_boot_hi', 'median_adjusted_boot',
                'median_adjusted_boot_lo', 'median_adjusted_boot_hi', 'mediated_share', 'mediated_share_lo',
                'mediated_share_hi', 'mean_raw', 'mean_selective', 'mean_adjusted']})
            if r.model in Ns.index:
                put(s, f'normalized:{r.model}', [Ns.loc[r.model, 'normalized_pearson'], Ns.loc[r.model, 'split_sd']])
        show = Ws[~Ws.model.str.startswith('LB:') | Ws.model.isin(LB_KEY)] if s == 'CTRPv2_LNIC50' else Ws
        L += ['### (i) Within the training dataset: fitness-mediated share and (v) DrEval normalized Pearson', '',
              f"Oracle reference (perfect predictor p = z): mean r_zg² = {ci(orc, 'oracle_mean_rzg2')}.", '',
              table(show, ['model', 'training_target', 'drugs', lambda r: ci(r, 'median_raw_boot'),
                           lambda r: ci(r, 'median_selective_boot'), lambda r: ci(r, 'median_adjusted_boot'),
                           lambda r: ci(r, 'mediated_share', 2), 'norm'],
                    ['model', 'training target', 'drugs', 'raw, median (95% CI)', 'selective, median (95% CI)',
                     'model-axis-adjusted, median (95% CI)', 'fitness-mediated share (95% CI)',
                     'DrEval normalized Pearson ± split SD']), '',
              'Sources: `within_summary.csv` (panel `main`), `normalized_pearson.csv`, per drug in '
              '`within_per_drug.csv.gz`.' + (' Leaderboard rows (`LB:`) are DrEval\'s released predictions on the same '
                                              'splits; the full set is in the CSV.' if s == 'CTRPv2_LNIC50' else ''),
              '']
        learned = Ws[Ws.model.isin(['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting'])]
        if s == 'CTRPv2_LNIC50':
            learned = Ws[Ws.model_class.str.contains('classical|neural', regex=True) & (Ws.model != 'LB:SVR')]
        if len(learned):
            lo, hi, med = learned.mediated_share.min(), learned.mediated_share.max(), learned.mediated_share.median()
            put(s, 'learned_mediated_share_range', [lo, hi, med, len(learned)])
            L += [f"Fitness-mediated share of the {len(learned)} learned raw-response models: {f(lo, 2)}-{f(hi, 2)} "
                  f"(median {f(med, 2)}), against {f(orc.oracle_mean_rzg2, 2)} for a perfect predictor.", '']
        # (ii) cross-dataset decomposition
        PPs = PP[(PP.setting == s) & (PP.panel == 'main')]
        if len(PPs):
            grp = 'published leaderboard models (21, as 38)' if s == 'CTRPv2_LNIC50' else \
                'DrEval learned models (raw response)'
            PPs = PPs[PPs.group == grp]
            for _, r in PPs.iterrows():
                put(s, f'pooled:{r.target}', {k: r[k] for k in r.index if k.startswith(('share', 'mediated_ret',
                                                                                     'alignment', 'loading',
                                                                                     'selterm_ret', 'delta_',
                                                                                     'mean_sel_BOTH'))})
            L += [f'### (ii) Does the cross-dataset loss run through the fitness term? (pooled: {grp})', '',
                  table(PPs, ['target', 'models', lambda r: ci(r, 'delta_raw'), lambda r: ci(r, 'delta_mediated'),
                              lambda r: ci(r, 'delta_selective_term'),
                              lambda r: ci(r, 'share_of_drop_from_fitness', 2), lambda r: ci(r, 'mediated_retained', 2),
                              lambda r: ci(r, 'alignment_ratio', 2), lambda r: ci(r, 'loading_ratio', 2),
                              lambda r: ci(r, 'selterm_retained', 2)],
                        ['target', 'models', 'Δ mean raw (cross - within)', 'Δ mediated term', 'Δ selective term',
                         'share of loss via fitness term', 'fitness term retained', 'alignment ratio r_pg',
                         'target-loading ratio r_zg', 'selective term retained']), '',
                  'Source: `paired_pooled.csv`. Ratios are pooled over models and drugs (sums), with two-way bootstrap '
                  '95% intervals; a share near 1 means the loss is the fitness-mediated term; alignment ratio < 1 '
                  'means the predictions align less with the target screen\'s axis, loading ratio < 1 that the target '
                  'screen\'s profiles load less on its own axis (R1-M2).', '']
        # (iii) selectivity under selective-BOTH
        PSs = PS[(PS.setting == s) & (PS.panel == 'main')]
        if len(PSs):
            keym = [m for m in (KEY_MODELS + LB_KEY) if m in set(PSs.model)]
            K = PSs[PSs.model.isin(keym)].copy()
            K['order'] = K.model.map({m: i for i, m in enumerate(keym)})
            K = K.sort_values(['target', 'order'])
            for _, r in K.iterrows():
                put(s, f'paired:{r.target}:{r.model}', {k: r[k] for k in ['drugs', 'mean_raw_s', 'mean_raw_t',
                                                                          'mean_sel_T_s', 'mean_sel_T_t',
                                                                          'mean_sel_BOTH_s', 'mean_sel_BOTH_t',
                                                                          'delta_sel_BOTH', 'delta_sel_BOTH_lo',
                                                                          'delta_sel_BOTH_hi', 'mean_adjusted_s',
                                                                          'mean_adjusted_t', 'share_of_drop_from_fitness',
                                                                          'mediated_retained'] if k in r})
            L += ['### (iii) Is selectivity retained across datasets under selective-BOTH? (key models)', '',
                  table(K, ['target', 'model', 'drugs', lambda r: f"{f(r.mean_raw_s)} → {f(r.mean_raw_t)}",
                            lambda r: f"{f(r.mean_sel_T_s)} → {f(r.mean_sel_T_t)}",
                            lambda r: f"{f(r.mean_sel_BOTH_s)} → {f(r.mean_sel_BOTH_t)}",
                            lambda r: ci(r, 'delta_sel_BOTH'),
                            lambda r: f"{f(r.mean_adjusted_s)} → {f(r.mean_adjusted_t)}",
                            lambda r: f(r.share_of_drop_from_fitness, 2), lambda r: f(r.mediated_retained, 2)],
                        ['target', 'model', 'drugs', 'raw (within → cross)', 'selective-T', 'selective-BOTH',
                         'Δ selective-BOTH (95% CI)', 'model-axis-adjusted', 'share of loss via fitness',
                         'fitness term retained']), '',
                  'Source: `paired_summary.csv` (means over drugs; matched cells; out-of-fold predictions). '
                  'Drug-agnostic predictors (FitnessOnlyRidge, NaiveMeanEffectsPredictor) give the leakage floor; '
                  'their model-axis-adjusted score is undefined (NA) because they carry no drug-specific signal.', '']
            fr = PSs[(PSs.model != 'ref:source_measurement') & PSs.get('sel_BOTH_t_frac_of_measured_mean',
                                                                        pd.Series(index=PSs.index)).notna()]
            if len(fr):
                piv = fr.pivot_table(index='model', columns='target', values='sel_BOTH_t_frac_of_measured_mean')
                pivm = fr.pivot_table(index='model', columns='target', values='sel_BOTH_t_frac_of_measured_median')
                piv['median over targets'] = piv.median(axis=1)
                pivm_med = pivm.median(axis=1)
                piv = piv.sort_values('median over targets', ascending=False)
                for m, r in piv.iterrows():
                    put(s, f'frac_of_measured:{m}', {**{k: r[k] for k in piv.columns},
                                                     'median_over_targets_of_median_ratio': pivm_med.get(m, np.nan)})
                refr = PSs[PSs.model == 'ref:source_measurement'].set_index('target')
                tg = [c for c in piv.columns if c != 'median over targets']
                L += [f"Cross-dataset selective-BOTH as a fraction of the training screen's own measurement "
                      f"(`ref:source_measurement`: {s.split('_')[0]}'s measured profile scored on the same matched drugs "
                      "and cells; ratio of drug means; the last column is the median over targets, with the "
                      "median-of-drug-medians ratio in brackets). Reference cross selective-BOTH (mean): " +
                      ', '.join(f"{t} {f(refr.loc[t, 'mean_sel_BOTH_t'])}" for t in tg if t in refr.index) +
                      '; reference raw: ' + ', '.join(f"{t} {f(refr.loc[t, 'mean_raw_t'])}" for t in tg if t in refr.index)
                      + '.', '',
                      table(piv.reset_index(), ['model'] + [(lambda c: (lambda r: f(r[c], 2)))(c) for c in tg] +
                            [lambda r: f"{f(r['median over targets'], 2)} [{f(pivm_med.get(r.model, np.nan), 2)}]"],
                            ['model'] + tg + ['median over targets']), '',
                      'Source: `paired_summary.csv` (columns `sel_BOTH_t_frac_of_measured_mean/median`, with 95% '
                      'intervals).', '']
            if len(PPs):
                L += ['Pooled selective-BOTH (learned raw-response models), within → cross, with the change and a '
                      '±0.05 equivalence check on its 90% interval:', '',
                      table(PPs, ['target', lambda r: f"{f(r.mean_sel_BOTH_s)} → {f(r.mean_sel_BOTH_t)}",
                                  lambda r: ci(r, 'delta_sel_BOTH'),
                                  lambda r: 'yes' if (r.get('delta_sel_BOTH_lo90', np.nan) > -0.05 and
                                                      r.get('delta_sel_BOTH_hi90', np.nan) < 0.05) else 'no'],
                            ['target', 'selective-BOTH within → cross', 'Δ (95% CI)', 'equivalent within ±0.05']), '']
        # (iv) contrasts
        Cs = C[(C.setting == s)]
        if len(Cs):
            fam = Cs[(Cs.family.isin(['target effect within model class', 'model class']) & (Cs.panel == 'main')) |
                     ((Cs.family == 'feature source') & (Cs.panel == 'feature_source')) |
                     ((Cs.family == 'target effect within model class') & (Cs.panel == 'feature_source') &
                      Cs.contrast.str.contains('DepMap'))]
            fam = fam[fam.metric.isin(['selective (= selective-BOTH within)', 'model-axis-adjusted',
                                       'selective-BOTH'])]
            fam = fam.sort_values(['family', 'contrast', 'where', 'metric'])
            for _, r in fam.iterrows():
                put(s, f'contrast:{r.contrast}:{r["where"]}:{r.metric}',
                    {k: r[k] for k in ['drugs', 'mean_diff', 'mean_diff_lo', 'mean_diff_hi', 'mean_diff_p',
                                       'holm_p_family_setting', 'wilcoxon_p', 'frac_a_better', 'panel']})
            L += ['### (iv) Effect of the training target within model class (and feature-source and class '
                  'controls)', '',
                  table(fam, ['family', 'contrast', 'where', 'metric', 'panel', 'drugs', lambda r: ci(r, 'mean_diff'),
                              lambda r: pfmt(r.holm_p_family_setting), lambda r: pfmt(r.wilcoxon_p),
                              lambda r: f(r.frac_a_better, 2)],
                        ['family', 'contrast (A - B)', 'where', 'metric', 'panel', 'drugs', 'mean Δ (95% CI)',
                         'Holm P (bootstrap SE)', 'Wilcoxon P (drugs)', 'share of drugs A > B']), '',
                  'Source: `contrasts.csv` (also raw and selective-T rows, and comparisons with every other model). '
                  'Holm adjustment within a family (one setting, panel and metric); the bootstrap SE accounts for '
                  'shared cell lines.', '']
            vs = Cs[(Cs.family == 'vs other models') & Cs.model_a.eq('PerDrugRidgeSelective') & (Cs.panel == 'main') &
                    ~Cs.model_b.isin(DRUG_AGNOSTIC) &
                    Cs.metric.isin(['model-axis-adjusted', 'selective-BOTH', 'selective (= selective-BOTH within)'])]
            if len(vs):
                vs = vs.sort_values(['where', 'metric', 'mean_diff'])
                rows = []
                for (wh, me), g in vs.groupby(['where', 'metric']):
                    rows.append({'where': wh, 'metric': me, 'models': len(g),
                                 'better_holm': int(((g.mean_diff > 0) & (g.holm_p_family_setting < 0.05)).sum()),
                                 'worse_holm': int(((g.mean_diff < 0) & (g.holm_p_family_setting < 0.05)).sum()),
                                 'min_diff': g.mean_diff.min(), 'min_model': g.loc[g.mean_diff.idxmin(), 'model_b'],
                                 'min_holm': g.loc[g.mean_diff.idxmin(), 'holm_p_family_setting']})
                wins = pd.DataFrame(rows)
                for _, r in wins.iterrows():
                    put(s, f'vs_models:{r["where"]}:{r.metric}', r.drop(['where', 'metric']).to_dict())
                L += ['Selective-target per-drug ridge (drevalpy) against every other model of the panel '
                      '(Holm within family and setting):', '',
                      table(wins, ['where', 'metric', 'models', 'better_holm', 'worse_holm',
                                   lambda r: f(r.min_diff), 'min_model', lambda r: pfmt(r.min_holm)],
                            ['where', 'metric', 'models compared (drug-agnostic excluded)',
                             'ridge better (Holm P < 0.05)', 'ridge worse (Holm P < 0.05)', 'smallest mean Δ',
                             'closest model', 'its Holm P']), '']
        # computed reading of this setting (thresholds stated)
        L += ['### Reading (computed from the tables above)', '']
        L += reading(s, Ws, PPs if len(PS) else pd.DataFrame(), C[C.setting == s] if len(C) else pd.DataFrame(), orc)
        L += ['']
        # feature-source panel
        Wf = W[(W.setting == s) & (W.panel == 'feature_source')]
        if len(Wf):
            L += ['### Feature-source control (cells with both DrEval and DepMap expression)', '',
                  table(Wf.sort_values('median_selective', ascending=False),
                        ['model', 'features', 'drugs', 'cells', lambda r: f(r.median_raw), lambda r: f(r.median_selective),
                         lambda r: f(r.median_adjusted), lambda r: ci(r, 'mediated_share', 2)],
                        ['model', 'features', 'drugs', 'cells', 'raw', 'selective', 'adjusted', 'fitness-mediated share']),
                  '', 'Source: `within_summary.csv` (panel `feature_source`).', '']
        # provenance
        Pv = PROV[PROV.setting == s] if len(PROV) else pd.DataFrame()
        if len(Pv):
            pv = Pv.groupby('target').agg(target_cells=('target_cells', 'first'),
                                          oof=('target_cells_with_oof_prediction', 'median'),
                                          paired=('paired_cells_used', 'first'),
                                          fold_avg=('target_cells_fold_averaged', 'median'),
                                          unused=('source_cells_repredicted_with_target_features_unused', 'median'))
            L += ['Provenance of target-dataset predictions (median over models): cells with out-of-fold predictions '
                  '(used, paired), cells absent from the training dataset with fold-averaged drevalpy cross-study '
                  'predictions (not used in the paired analysis), and training-dataset cells that drevalpy re-predicted '
                  'with the target dataset\'s features (not used). Source: `cell_provenance.csv`; unpaired scores of '
                  'the fold-averaged predictions: `new_cells_unpaired.csv`.', '',
                  table(pv.reset_index(), ['target', 'target_cells', lambda r: int(r.oof), 'paired',
                                           lambda r: int(r.fold_avg), lambda r: int(r.unused)],
                        ['target', 'target cells', 'with out-of-fold prediction', 'paired cells (common panel)',
                         'fold-averaged (new cells)', 're-predicted, unused']), '']
    L += consistency(W, PP)
    cens = {ds: man['datasets'][ds]['lnic50']['censored_share_of_rows'] for ds in man['datasets']}
    L += ['## Caveats', '',
          '- **Model-axis-adjusted metric.** Partialling a model\'s own leave-drug-out axis removes every component its '
          'predictions share across drugs, including genuine class- or lineage-shared selectivity, so it is '
          'conservative for selective-target models; it is undefined (NA) for drug-agnostic predictors. The axis is '
          'built from per-drug z-scored predictions (script 40 used rank-normal predictions; on the leaderboard the two '
          'give EnsembleMF 0.199 here vs 0.193 there).',
          '- **Selective-BOTH floor.** Partialling the training and the evaluation screens\' axes does not remove all '
          'drug-agnostic signal: drug-agnostic predictors keep a small positive cross-dataset selective-BOTH score '
          '(the floor rows in each (iii) table).',
          '- **Censoring.** CurveCurator ln IC50 is missing where no IC50 was reached (share of rows: ' +
          ', '.join(f'{k} {v:.0%}' for k, v in cens.items()) + '); AUC is uncensored. Per-drug ln IC50 profiles '
          'therefore cover different, sensitivity-enriched cell subsets.',
          '- **Splits.** CTRPv2 ln IC50 uses the published leaderboard splits; GDSC2 and CTRPv2 AUC use drevalpy\'s '
          'procedure applied to that setting\'s response rows, so the ln IC50 and AUC splits of one dataset differ.',
          '- **Leaderboard comparators.** `LB:` rows are DrEval\'s released predictions (same splits; produced by '
          'DrEval\'s maintainers, drevalpy version not recorded in the release); this study\'s models and the '
          'selective-target GradientBoosting/EnsembleMF variants were run here with drevalpy 1.6.0. In the GDSC2 '
          'settings the raw-target counterparts were rerun here as well.',
          '- **Cross-platform cross-study predictions.** drevalpy predicts cells absent from the training dataset '
          'from the target dataset\'s own expression file; between the CTRP/CCLE and GDSC families these files come '
          'from different platforms (manifest: median expression value 29.6 vs 6.4), so those fold-averaged '
          'predictions (provenance tables, `new_cells_unpaired.csv`) are not comparable to out-of-fold predictions '
          'and are not used in the paired analysis.',
          '- **SimpleNeuralNetworkSelective did not learn** (GDSC2 AUC): its predictions are almost constant within each '
          'drug (median within-drug SD of the placed predictions 0.001 vs 0.086 for the measured AUC) and its training '
          'loss stayed at the target variance (~0.9), i.e. DrEval\'s SimpleNeuralNetwork, with the configuration tuned '
          'for the raw response (reused, not re-tuned) and an unscaled selective target, collapsed to the drug mean. '
          'Its contrasts therefore do not measure the effect of the target and are excluded from the conclusions; a '
          're-tuned or response-scaled selective SNN would be needed.',
          '- **RandomForest** (GDSC2 AUC) was run last at the orchestrator\'s request; scores without it are marked by '
          'the pooled group listing 3 models (SimpleNeuralNetwork, EnsembleMF, GradientBoosting).',
          '- **Multiplicity.** Holm within families (one setting, panel and metric; for comparisons with other models '
          'also one reference model). Drug-level Wilcoxon P values ignore shared cell lines and are given for '
          'comparison with the paper only.',
          '']
    L += ['## Files', '',
          '- `inputs/`: responses (sensitivity orientation), cell-name map, DepMap expression, manifest (57).',
          '- `drevalpy_data/`: data directory extracted from data/dreval/*.zip for drevalpy (57).',
          '- `drevalpy_runs/<setting>/<dataset>/LCO/<model>/`: drevalpy outputs (predictions, cross_study, best_hpams), '
          'splits, timings (58).',
          '- `within_summary.csv`, `within_per_drug.csv.gz`, `normalized_pearson.csv`, `normalized_validation.csv`, '
          '`paired_summary.csv`, `paired_pooled.csv`, `paired_per_drug.csv.gz`, `contrasts.csv`, `cell_provenance.csv`, '
          '`new_cells_unpaired.csv`, `score_run.json` (59).',
          '- `SUMMARY.md`, `key_numbers.json` (60); logs in `logs/`.', '']
    (OUT / 'key_numbers.json').write_text(json.dumps(KEY, indent=1, default=float) + '\n')
    (OUT / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L)[:3000])


if __name__ == '__main__':
    main()
