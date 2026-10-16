"""Round 2 (WS6b), task 7 (editor item 4; R1-M2; R2-M5): dataset x endpoint for DrEval's four learned models.

Cells of the design (leave-cell-line-out, drevalpy, DrEval's features and tuning; 59's main panels and 78's draws):
  GDSC2 CurveCurator AUC        retrained EnsembleMF, GradientBoosting, SimpleNeuralNetwork, RandomForest
  GDSC2 CurveCurator ln IC50    retrained, same four models (complete when each has 10 splits)
  CTRP v2 CurveCurator ln IC50  DrEval's released leaderboard predictions of the same four models (LB:; the
                                retrained EnsembleMF reproduces them, 92) on 59's panel cells
  CTRP v2 CurveCurator AUC      round 3: retrained in drevalpy (the fourth cell; editor item 1), same four models,
                                DrEval's features, tuning and drevalpy's own splits; included once 59 has scored it
Per cell, pooled over the four models (sums over models and drugs; two-way cell x drug bootstrap, B = 1,000, 78's
seeds) and per model: axis share of raw accuracy, perfect-predictor share (raw) and its reliability-corrected value
(78/79 primary source: IMPROVE GDSC2 replicated pairs for GDSC2; CTRP v1-v2 agreement for CTRP v2), axis capture and
selective capture (raw and against the corrected reference), median raw and selective per model; the drug-agnostic
floor (FitnessOnlyRidge, omics-predicted axis, run in drevalpy) against the best learned model (median raw r); the
tissue-by-drug baseline (NaiveTissueDrugMeanPredictor; median selective r) against the learned models; the lineage-
adjusted selective score (91); and the cross-dataset attribution (81's proportional-attenuation contrasts on 80's
paired arrays, pooled over the four models, every target dataset with the same endpoint).
Contrasts (independent bootstraps paired by index): endpoint effect = GDSC2 ln IC50 - GDSC2 AUC (same dataset, models,
cells); dataset effect = CTRP v2 ln IC50 - GDSC2 ln IC50 (same endpoint). Round 3, with the fourth cell: the endpoint
effect within CTRP v2 (CTRP v2 ln IC50 - CTRP v2 AUC), the dataset effect at AUC (CTRP v2 AUC - GDSC2 AUC), the main
effects (means of the two simple effects) and the interaction (endpoint effect in CTRP v2 - endpoint effect in GDSC2 =
dataset effect at ln IC50 - dataset effect at AUC). The bootstraps of different cells are independent (different seeds,
cell and drug universes), which is conservative for contrasts between cells that share cell lines.
Outputs (results/revision_round2/two_by_two/): cells.csv (pooled per cell), models.csv, contrasts.csv,
attribution.csv, summary.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.out_dir('two_by_two')
LEARNED = ['EnsembleMF', 'GradientBoosting', 'SimpleNeuralNetwork', 'RandomForest']
CELLS = {'GDSC2 AUC': ('GDSC2_AUC', LEARNED), 'GDSC2 ln IC50': ('GDSC2_LNIC50', LEARNED),
         'CTRP v2 ln IC50': ('CTRPv2_LNIC50', [f'LB:{m}' for m in LEARNED]),
         'CTRP v2 AUC': ('CTRPv2_AUC', LEARNED)}
PRIMARY = {'GDSC2_AUC': 'IMPROVE_GDSCv2_auc|Rsel_var', 'GDSC2_LNIC50': 'IMPROVE_GDSCv2_ic50|Rsel_var',
           'CTRPv2_LNIC50': 'CTRPv1v2_lnIC50|Rsel_median_corr', 'CTRPv2_AUC': 'CTRPv1v2_AUC|Rsel_median_corr'}
CONTRAST_KEYS = ['axis_share', 'perfect_share', 'corrected_reference', 'axis_capture', 'selective_capture',
                 'axis_capture_tau', 'selective_capture_tau', 'share_ratio', 'alignment_ratio', 'mean_raw',
                 'floor_median_raw', 'best_learned_median_raw', 'best_minus_floor_median_raw', 'floor_over_best_median_raw',
                 'tissue_baseline_median_selective', 'learned_median_selective_median', 'learned_median_selective_max',
                 'best_selective_minus_tissue_baseline', 'ridge_median_raw']


def within(setting):
    """78.retrain_within without dropping the released (LB:) models; identical cells, decomposition and draws."""
    wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
    base = sorted(wp[(wp.setting == setting) & (wp.panel == 'main')].model.unique())
    rt = lib.load_module('rt59', '59_retrain_score.py')
    oof, _, _ = rt.load_oof(setting)
    extra = [m for m in LEARNED if m in oof and m not in base]
    S = lib.retrain_setup(setting, models=base + extra)
    O = S['obs'][S['obs'].cell.isin(S['cells'])][['drug', 'cell', 'response']]
    Z, G, _, _ = lib.selbench.decompose(O)
    Pm = {m: P.reindex(columns=Z.columns) for m, P in S['P'].items()}
    W, V = lib.draws(len(Z.columns), len(Z.index), 20261031 + sum(map(ord, setting)))
    arrs, models, drugs, ncell = lib.within_engine(Z, G, Pm, W, V, common_mask=True)
    return arrs, models, V, {'cells': len(Z.columns), 'drugs': len(Z.index), 'models': models,
                             'incomplete': {k: str(v) for k, v in S['incomplete'].items()}}


def main():
    # round 3: --models restricts the learned models in every cell (e.g. a provisional three-model design while one
    # model of the fourth cell is still training); --out writes to another results/revision_round2/<topic>/
    import argparse
    global OUT, LEARNED, CELLS
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--out', default='two_by_two')
    a = ap.parse_args()
    OUT = lib.out_dir(a.out)
    if a.models:
        LEARNED = [m for m in LEARNED if m in a.models]
        CELLS = {c: (st, [m for m in ms if m.replace('LB:', '') in LEARNED]) for c, (st, ms) in CELLS.items()}
    t0 = time.time()
    cap = lib.load_module('cap78', '78_capture.py')
    att = lib.load_module('att81', '81_attenuation_null.py')
    rel = dict(np.load(lib.OUT2 / 'reliability' / 'reliability_draws.npz'))
    lin = pd.read_csv(lib.OUT2 / 'lineage' / 'lineage_by_model.csv')
    W59 = pd.read_csv(lib.RES / 'revision_retrain' / 'within_summary.csv')
    cell_rows, model_rows, draws = [], [], {}
    info = {}
    have_settings = set(pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting']).setting)
    for cname, (setting, learned) in CELLS.items():
        if setting not in have_settings:
            print('not scored yet:', setting, flush=True)
            continue
        arrs, models, V, inf = within(setting)
        info[cname] = inf
        have = [m for m in learned if m in models and np.isfinite(arrs['raw'][0, models.index(m)]).any()]
        info[cname]['learned_present'] = have
        if not have:
            continue
        ix = [models.index(m) for m in have]
        cat = lambda a: np.concatenate([a[:, i] for i in ix], axis=1)  # noqa: E731
        Vc = np.concatenate([V] * len(ix), axis=1)
        st, ok = cap.capture_reps(cat(arrs['raw']), cat(arrs['r_pg']), cat(arrs['r_zg']), cat(arrs['sel']), Vc)
        Rs = rel[PRIMARY[setting]]
        ct = cap.corrected_reps(cat(arrs['raw']), cat(arrs['r_pg']), cat(arrs['r_zg']), cat(arrs['sel']), Vc, Rs)
        d = {**{k: st[k] for k in ['axis_share', 'perfect_share', 'share_ratio', 'axis_capture', 'selective_capture',
                                   'mean_raw', 'alignment_ratio']},
             **{k: ct[k] for k in ['corrected_reference', 'axis_capture_tau', 'selective_capture_tau', 'share_ratio_tau']}}
        meds = {m: (lib.wmedian_rows(arrs['raw'][:, models.index(m)], V), lib.wmedian_rows(arrs['sel'][:, models.index(m)], V))
                for m in models}
        best = max(have, key=lambda m: meds[m][0][0])
        floor = 'FitnessOnlyRidge'
        tb = 'NaiveTissueDrugMeanPredictor' if 'NaiveTissueDrugMeanPredictor' in models else 'LB:NaiveTissueDrugMeanPredictor'
        if floor in meds:
            d['floor_median_raw'] = meds[floor][0]
            d['best_learned_median_raw'] = meds[best][0]
            d['best_minus_floor_median_raw'] = meds[best][0] - meds[floor][0]
            d['floor_over_best_median_raw'] = meds[floor][0] / meds[best][0]
        if tb in meds:
            d['tissue_baseline_median_selective'] = meds[tb][1]
            ms = np.vstack([meds[m][1] for m in have])
            d['learned_median_selective_median'] = np.nanmedian(ms, 0)
            d['learned_median_selective_max'] = np.nanmax(ms, 0)
            d['best_selective_minus_tissue_baseline'] = np.nanmax(ms, 0) - meds[tb][1]
        for ridge in ['PerDrugRidgeRaw']:
            if ridge in meds:
                d['ridge_median_raw'] = meds[ridge][0]
        draws[cname] = d
        row = {'cell': cname, 'setting': setting, 'models': ';'.join(have), 'n_models': len(have),
               'best_learned_model': best, 'tissue_baseline': tb, 'cells_scored': inf['cells'], 'drugs': inf['drugs']}
        for k, v in d.items():
            row.update(lib.summ(v, k))
        # lineage-adjusted selective (91), pooled over the same models when 91's pool matches; else model medians
        L = lin[(lin.benchmark == 'retrain') & (lin.setting == setting) & lin.model.isin(have)]
        if len(L):
            row['lineage_adjusted_median_selective_range'] = f'{L.median_selective_lin.min():.3f} to {L.median_selective_lin.max():.3f}'
            row['median_selective_range'] = f'{L.median_selective.min():.3f} to {L.median_selective.max():.3f}'
        cell_rows.append(row)
        for m in have + [x for x in [floor, tb, 'PerDrugRidgeRaw', 'PerDrugRidgeSelective', 'EnsembleMF'] if x in models and x not in have]:
            i = models.index(m)
            s1, _ = cap.capture_reps(arrs['raw'][:, i], arrs['r_pg'][:, i], arrs['r_zg'][:, i], arrs['sel'][:, i], V)
            r = {'cell': cname, 'setting': setting, 'model': m}
            for k in ['axis_share', 'axis_capture', 'selective_capture', 'perfect_share']:
                r.update(lib.summ(s1[k], k))
            r.update(lib.summ(meds[m][0], 'median_raw'))
            r.update(lib.summ(meds[m][1], 'median_selective'))
            l1 = lin[(lin.benchmark == 'retrain') & (lin.setting == setting) & (lin.model == m)]
            if len(l1):
                r.update({k: float(l1.iloc[0][k]) for k in ['median_selective_lin', 'median_selective_lin_lo',
                                                             'median_selective_lin_hi']})
            model_rows.append(r)
        print(cname, have, f'{time.time() - t0:.0f}s', flush=True)
    C = pd.DataFrame(cell_rows)
    C.to_csv(OUT / 'cells.csv', index=False, float_format='%.5g')
    pd.DataFrame(model_rows).to_csv(OUT / 'models.csv', index=False, float_format='%.5g')
    # contrasts: endpoint (GDSC2 ln IC50 - GDSC2 AUC) and dataset (CTRP v2 ln IC50 - GDSC2 ln IC50)
    crow = []
    # simple effects (round 2's three labels kept verbatim), then, with the fourth cell, the other simple effects,
    # main effects and the interaction; each as a linear combination of cell draws (weights per cell)
    G2A, G2I, C2I, C2A = 'GDSC2 AUC', 'GDSC2 ln IC50', 'CTRP v2 ln IC50', 'CTRP v2 AUC'
    combos = [('endpoint effect (GDSC2 ln IC50 - GDSC2 AUC)', {G2I: 1, G2A: -1}),
              ('dataset effect (CTRP v2 ln IC50 - GDSC2 ln IC50)', {C2I: 1, G2I: -1}),
              ('both (CTRP v2 ln IC50 - GDSC2 AUC)', {C2I: 1, G2A: -1}),
              ('endpoint effect in CTRP v2 (CTRP v2 ln IC50 - CTRP v2 AUC)', {C2I: 1, C2A: -1}),
              ('dataset effect at AUC (CTRP v2 AUC - GDSC2 AUC)', {C2A: 1, G2A: -1}),
              ('endpoint main effect (mean of the two endpoint effects)', {G2I: .5, G2A: -.5, C2I: .5, C2A: -.5}),
              ('dataset main effect (mean of the two dataset effects)', {C2I: .5, G2I: -.5, C2A: .5, G2A: -.5}),
              ('interaction (endpoint effect in CTRP v2 - endpoint effect in GDSC2)', {C2I: 1, C2A: -1, G2I: -1, G2A: 1})]
    for lab, wts in combos:
        if not all(c in draws for c in wts):
            continue
        r = {'contrast': lab}
        for k in CONTRAST_KEYS:
            if all(k in draws[c] for c in wts):
                val = sum(w * np.asarray(draws[c][k], float) for c, w in wts.items())
                r.update(lib.summ(val, k))
                r[f'{k}_excludes_0'] = bool(r[f'{k}_lo'] > 0 or r[f'{k}_hi'] < 0)
        crow.append(r)
    pd.DataFrame(crow).to_csv(OUT / 'contrasts.csv', index=False, float_format='%.5g')
    # cross-dataset attribution (80 arrays; 81 contrasts) pooled over the four models of each cell
    ix = pd.read_csv(lib.OUT2 / 'paired' / 'pairs.csv')
    arows = []
    for cname, (setting, learned) in CELLS.items():
        for _, r in ix[(ix.benchmark == 'retrain') & (ix.family == setting)].iterrows():
            f = lib.OUT2 / 'paired' / 'arrays' / f'retrain__{r.k}.npz'
            if not f.exists():
                continue
            models, meta, S = lib.load_paired(f)
            gm = [m for m in learned if m in models]
            if not gm:
                continue
            st, n = att.contrasts(S, models, gm)
            row = {'cell': cname, 'pair': r.pair, 'models': ';'.join(gm), 'model_drug_pairs': n}
            for k in ['raw_mean_within', 'raw_mean_cross', 'raw_loss', 'within_share', 'share_of_loss', 'excess',
                      'axis_retained', 'selterm_retained', 'retention_difference', 'd_mediated', 'd_selective_term',
                      'd_selBOTH_sym']:
                row.update(lib.summ(st[k], k))
            row['loss_clear'] = bool(row['raw_loss_lo'] > 0)
            exc = row['excess_lo'] > 0 or row['excess_hi'] < 0
            row['reading'] = ('axis lost disproportionately' if (row['loss_clear'] and exc and row['excess'] > 0) else
                              'selective term lost disproportionately' if (row['loss_clear'] and exc) else
                              'clear loss, proportional attenuation not rejected' if row['loss_clear'] else 'no clear raw loss')
            arows.append(row)
    A = pd.DataFrame(arows)
    A.to_csv(OUT / 'attribution.csv', index=False, float_format='%.5g')
    lib.write_json(OUT / 'summary.json', {'seconds': round(time.time() - t0), 'info': info, 'primary_reliability': PRIMARY})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 200)
    cols = ['cell', 'n_models', 'axis_share', 'axis_share_lo', 'axis_share_hi', 'perfect_share', 'corrected_reference',
            'axis_capture', 'selective_capture', 'selective_capture_lo', 'selective_capture_hi', 'floor_median_raw',
            'best_learned_median_raw', 'tissue_baseline_median_selective', 'learned_median_selective_max']
    print(C[[c for c in cols if c in C]].round(3).to_string(index=False))
    print(pd.DataFrame(crow).round(3).to_string(index=False))
    print(A[['cell', 'pair', 'raw_loss', 'raw_loss_lo', 'raw_loss_hi', 'within_share', 'share_of_loss', 'excess',
             'excess_lo', 'excess_hi', 'd_selBOTH_sym', 'reading']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
