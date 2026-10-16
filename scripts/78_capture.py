"""Round 2, task 1 (editor item 1; R1-M1, R3-M1) and the corrected reference of task 2 (R3-M2, R1-M1d): axis and
selective capture relative to a perfect predictor, with two-way (cell x drug) bootstrap intervals shared across models.

Per model (drugs d scored within the training dataset; the perfect predictor is p = z, whose terms are r_zg^2 and
1 - r_zg^2):
  axis capture        A = sum_d r_pg r_zg / sum_d r_zg^2
  selective capture   S = sum_d (r_pz - r_pg r_zg) / sum_d (1 - r_zg^2)
  alignment ratio     mean_d r_pg / mean_d r_zg
  axis share          sum_d r_pg r_zg / sum_d r_pz;  perfect-predictor share m = mean_d r_zg^2;  share ratio = share/m
  feasible bound      1/m (the share is at most 1 when the selective term is non-negative)
  counterfactual ratio (R3-M1) with r_pg set to r_zg drug by drug and the model's own selective (partial)
                      correlation s kept: raw_cf = r_zg^2 + (1 - r_zg^2) s, ratio_cf = (sum r_zg^2 / sum raw_cf) / m
  odds ratio          [share/(1-share)] / [m/(1-m)] (R1-M1b's unbounded scale)
  medians of the per-drug raw and selective correlations.
Pooled summaries (sums over the models of a group and their drugs, same draws) per setting for learned models.

Reliability-corrected reference (task 2; error model and sources in 79_reliability.py). With rho_d = r_zg^2 +
(1 - r_zg^2) R_sel (axis noise-free, drug-specific noise e independent of p and g), a perfect predictor of the
reproducible profile tau has axis share r_zg^2/rho_d, so the corrected reference is mean_d r_zg^2 / rho_d. Capture
relative to tau uses the disattenuated terms: axis r_pg r_zg / sqrt(rho_d) against r_zg^2/rho_d and selective
(r_pz - r_pg r_zg)/sqrt(rho_d) against 1 - r_zg^2/rho_d. Sensitivity to cell-level noise shared across drugs: a
fraction 1 - kappa of each drug's axis component is treated as non-reproducible, reference mean_d kappa r_zg^2 /
(kappa r_zg^2 + (1 - r_zg^2) R_sel) (kappa = 1, 0.8, 0.55; 0.55 is the axis agreement of PRISM's two runs). Draw b of
the within-dataset bootstrap is paired with draw b of the reliability bootstrap (independent resamples).

Settings
  leaderboard   DrEval's released CTRPv2 CurveCurator ln IC50 predictions (42's cells, models and draws; the paper's
                0.293 perfect-predictor share), incl. this study's references and the measured axis.
  retrain       drevalpy retraining (59's main panels): GDSC2 CurveCurator AUC, CTRPv2 ln IC50 (drevalpy-run
                references), GDSC2 ln IC50 (references; DrEval's learned models when their 10 splits are complete).
  improve       IMPROVE (68, replicate-free within-dataset predictions; each model on its own pairs), 5 datasets.
Outputs (results/revision_round2/capture/): capture_by_model.csv, capture_pooled.csv, capture_by_setting.csv;
(results/revision_round2/reliability/): corrected_reference.csv, corrected_capture_by_model.csv,
corrected_capture_pooled.csv; run.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.out_dir('capture')
ROUT = lib.out_dir('reliability')
DREVAL_LEARNED = ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting']
KAPPAS = [1.0, 0.8, 0.55]
R_FLOOR = 0.02
# setting -> [(reliability key in 79's draws, stat, label, primary?)]
REL_MAP = {
    'leaderboard|CTRPv2_lnIC50': [
        ('CTRPv1v2_lnIC50', 'Rsel_median_corr', 'CTRP v1-v2 CurveCurator ln IC50 selective agreement (lower bound on '
                                                'R_sel; same endpoint and fitter, another screen)', True),
        ('IMPROVE_CTRPv2_ic50', 'Rsel_var', 'IMPROVE CTRPv2 replicated pairs, IC50 field (within screen; IMPROVE fit; '
                                            'few replicated cell lines)', False),
        ('IMPROVE_CTRPv2_auc', 'Rsel_var', 'IMPROVE CTRPv2 replicated pairs, AUC (within screen; endpoint differs)', False),
        ('GDSC1_sites_lnIC50', 'Rsel_median_corr', "GDSC1's two sites, ln IC50 (within another screen)", False)],
    'retrain|GDSC2_AUC': [
        ('IMPROVE_GDSCv2_auc', 'Rsel_var', 'IMPROVE GDSC2 replicated pairs, AUC (within screen; IMPROVE fit and range '
                                           'differ from CurveCurator)', True),
        ('IMPROVE_GDSCv2_auc', 'Rsel_pooled_corr', 'IMPROVE GDSC2 replicated pairs, AUC, pooled replicate correlation', False),
        ('GDSC1_sites_AUC', 'Rsel_median_corr', "GDSC1's two sites, AUC", False)],
    'retrain|GDSC2_LNIC50': [
        ('IMPROVE_GDSCv2_ic50', 'Rsel_var', 'IMPROVE GDSC2 replicated pairs, IC50 field (within screen; IMPROVE fit; '
                                            'censored)', True),
        ('GDSC1_sites_lnIC50', 'Rsel_median_corr', "GDSC1's two sites, ln IC50", False)],
    'improve|CTRPv2_AUC': [('IMPROVE_GDSCv2_auc', 'Rsel_var', 'GDSC2 IMPROVE value borrowed: CTRPv2 has replicates in only '
                                                              'about 12 cell lines, an unstable estimate', True),
                           ('IMPROVE_CTRPv2_auc', 'Rsel_var', 'own replicated pairs (about 12 cell lines; unstable)', False),
                           ('IMPROVE_CTRPv2_auc', 'Rsel_pooled_corr', 'own replicated pairs, pooled correlation', False)],
    'improve|GDSCv1_AUC': [('IMPROVE_GDSCv1_auc', 'Rsel_var', 'own replicated pairs (same endpoint and processing)', True),
                           ('IMPROVE_GDSCv1_auc', 'Rsel_pooled_corr', 'own replicated pairs, pooled correlation', False)],
    'improve|GDSCv2_AUC': [('IMPROVE_GDSCv2_auc', 'Rsel_var', 'own replicated pairs (same endpoint and processing)', True),
                           ('IMPROVE_GDSCv2_auc', 'Rsel_pooled_corr', 'own replicated pairs, pooled correlation', False)],
    'improve|CCLE_AUC': [('IMPROVE_GDSCv2_auc', 'Rsel_var', 'no replicates; GDSC2 IMPROVE value borrowed', True),
                         ('IMPROVE_CTRPv2_auc', 'Rsel_var', 'no replicates; CTRPv2 IMPROVE value borrowed', False)],
    'improve|gCSI_AUC': [('IMPROVE_GDSCv2_auc', 'Rsel_var', 'no replicates; GDSC2 IMPROVE value borrowed', True),
                         ('IMPROVE_CTRPv2_auc', 'Rsel_var', 'no replicates; CTRPv2 IMPROVE value borrowed', False)],
}
TRI = 'cross-screen reproducible share from triads (84; drugs in >= 3 datasets): a lower bound on the within-screen R_sel'
for _k, _fam, _ds in [('leaderboard|CTRPv2_lnIC50', 'DrEval CC lnIC50', 'CTRPv2'), ('retrain|GDSC2_AUC', 'DrEval CC AUC', 'GDSC2'),
                      ('retrain|GDSC2_LNIC50', 'DrEval CC lnIC50', 'GDSC2'), ('improve|CTRPv2_AUC', 'IMPROVE AUC', 'CTRPv2'),
                      ('improve|GDSCv1_AUC', 'IMPROVE AUC', 'GDSCv1'), ('improve|GDSCv2_AUC', 'IMPROVE AUC', 'GDSCv2'),
                      ('improve|CCLE_AUC', 'IMPROVE AUC', 'CCLE'), ('improve|gCSI_AUC', 'IMPROVE AUC', 'gCSI')]:
    REL_MAP[_k].append((f'triad_{_fam}_{_ds}', 'R_triad', TRI, False))
REL_MAP['retrain|CTRPv2_LNIC50'] = REL_MAP['leaderboard|CTRPv2_lnIC50']
# round 3: CTRP v2 CurveCurator AUC retrained (fourth cell of the dataset x endpoint design); the same source types as
# CTRP v2 ln IC50, AUC versions: CTRP v1-v2 agreement (primary, as for ln IC50), CTRP v2's own replicated pairs
# (IMPROVE), and the cross-screen triads
REL_MAP['retrain|CTRPv2_AUC'] = [
    ('CTRPv1v2_AUC', 'Rsel_median_corr', 'CTRP v1-v2 CurveCurator AUC selective agreement (cross-screen; same endpoint '
                                         'and fitter, another screen)', True),
    ('IMPROVE_CTRPv2_auc', 'Rsel_var', 'IMPROVE CTRPv2 replicated pairs, AUC (within screen; IMPROVE fit; few '
                                       'replicated cell lines)', False),
    ('triad_DrEval CC AUC_CTRPv2', 'R_triad', TRI, False)]


# ----------------------------------------------------------------------------------------------------------------
def capture_reps(raw, rpg, rzg, sel, V):
    """Per-draw capture statistics for one model (or pooled: arrays concatenated over models along axis 1)."""
    ok = np.isfinite(raw) & np.isfinite(rpg) & np.isfinite(rzg) & np.isfinite(sel)
    w = np.where(ok, V, 0.0)
    z = lambda x: np.where(ok, x, 0.0)  # noqa: E731
    med = z(rpg * rzg)
    r2 = z(rzg ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        n = w.sum(1)
        s_med, s_r2, s_raw = (w * med).sum(1), (w * r2).sum(1), (w * z(raw)).sum(1)
        s_selterm = s_raw - s_med
        A = s_med / s_r2
        S = s_selterm / (n - s_r2)
        m = s_r2 / n
        share = s_med / s_raw
        cf = (w * z(rzg ** 2 + (1 - rzg ** 2) * sel)).sum(1)
        out = {'axis_capture': A, 'selective_capture': S, 'alignment_ratio': (w * z(rpg)).sum(1) / (w * z(rzg)).sum(1),
               'mean_r_pg': (w * z(rpg)).sum(1) / n, 'mean_r_zg': (w * z(rzg)).sum(1) / n,
               'axis_share': share, 'perfect_share': m, 'share_ratio': share / m, 'bound': 1 / m,
               'cf_share_ratio': (s_r2 / cf) / m, 'odds_ratio': (share / (1 - share)) / (m / (1 - m)),
               'mean_raw': s_raw / n, 'drugs_weighted': n}
    return out, ok


def corrected_reps(raw, rpg, rzg, sel, V, Rs, kappa=1.0):
    """Capture relative to the reproducible profile tau (rho_d = r_zg^2 + (1 - r_zg^2) R_sel[b]; axis noise-free).
    R_sel draws are floored at R_FLOOR (a draw <= 0 would make tau undefined). With kappa < 1 (sensitivity to
    cell-level noise shared across drugs) only the reference changes: mean kappa r_zg^2 / (kappa r_zg^2 +
    (1 - r_zg^2) R_sel), the share of the reproducible profile on the reproducible part of the axis; the capture
    terms are then not defined and the share ratio uses the kappa = 1 share."""
    ok = np.isfinite(raw) & np.isfinite(rpg) & np.isfinite(rzg) & np.isfinite(sel)
    Rs = np.clip(np.asarray(Rs, float), R_FLOOR, 1.0)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = rzg ** 2 + (1 - rzg ** 2) * Rs
        w = np.where(ok, V, 0.0)
        z = lambda x: np.where(ok, x, 0.0)  # noqa: E731
        n = w.sum(1)
        s_ref = (w * z(rzg ** 2 / rho)).sum(1)
        inv = z(1 / np.sqrt(rho))
        s_med = (w * z(rpg * rzg) * inv).sum(1)
        s_raw = (w * z(raw) * inv).sum(1)
        s_sel = s_raw - s_med
        share = s_med / s_raw
        if kappa < 1:
            rk = kappa * rzg ** 2 + (1 - rzg ** 2) * Rs
            ref = (w * z(kappa * rzg ** 2 / rk)).sum(1) / n
            return {'corrected_reference': ref, 'share_ratio_tau': share / ref, 'bound_tau': 1 / ref}
        ref = s_ref / n
        return {'corrected_reference': ref, 'axis_capture_tau': s_med / s_ref,
                'selective_capture_tau': s_sel / (n - s_ref), 'axis_share_tau': share, 'share_ratio_tau': share / ref,
                'bound_tau': 1 / ref}


def flags_for(benchmark, model, extra=None):
    m = model.replace('LB:', '')
    exact = {'MultiViewRandomForest', 'NaiveTissueMeanPredictor', 'fitness_only', lib.FITNESS, 'FitnessOnlyRidge',
             'FitnessOnlyRidgeDm', 'floor_cell_mean'}
    within_split = {'Lasso', 'ElasticNet', 'NaiveMeanEffectsPredictor'}
    ag = 'exact' if m in exact else ('within split (additive)' if m in within_split else '')
    return {'drug_agnostic': ag, 'naive': 'Naive' in m or m == 'floor_cell_mean',
            'reference_pseudo': m in (lib.MEASURED, lib.FITNESS)}


def summarize_setting(bench, setting, arrs, models, V, groups, meta_fn, rel, out_rows, pool_rows, corr_rows,
                      corr_pool_rows, corr_ref_rows):
    raw, rpg, rzg, sel = arrs['raw'], arrs['r_pg'], arrs['r_zg'], arrs['sel']
    key = f'{bench}|{setting}'
    for i, m in enumerate(models):
        if not np.isfinite(raw[0, i]).any():
            continue
        st, ok = capture_reps(raw[:, i], rpg[:, i], rzg[:, i], sel[:, i], V)
        if not ok[0].any():
            continue
        row = {'benchmark': bench, 'setting': setting, 'model': m, **meta_fn(m),
               'drugs': int(ok[0].sum())}
        for k, v in st.items():
            if k == 'drugs_weighted':
                continue
            row.update(lib.summ(v, k))
        row.update(lib.summ(lib.wmedian_rows(raw[:, i], V), 'median_raw'))
        row.update(lib.summ(lib.wmedian_rows(sel[:, i], V), 'median_selective'))
        out_rows.append(row)
        for src, stat, label, prim in rel.get(key, []):
            Rs = rel['draws'].get(f'{src}|{stat}')
            if Rs is None:
                continue
            for kappa in KAPPAS:
                c = corrected_reps(raw[:, i], rpg[:, i], rzg[:, i], sel[:, i], V, Rs, kappa)
                r2 = {'benchmark': bench, 'setting': setting, 'model': m, 'reliability_source': f'{src}|{stat}',
                      'reliability_label': label, 'primary_source': prim, 'kappa': kappa, **meta_fn(m)}
                for k, v in c.items():
                    r2.update(lib.summ(v, k))
                corr_rows.append(r2)
    # model-independent reference rows (from the measured profile's loadings; same for every model on its drugs:
    # use the first model with the most drugs)
    j = int(np.argmax([np.isfinite(raw[0, i]).sum() for i in range(len(models))]))
    for src, stat, label, prim in rel.get(key, []):
        Rs = rel['draws'].get(f'{src}|{stat}')
        if Rs is None:
            continue
        for kappa in KAPPAS:
            c = corrected_reps(raw[:, j], rpg[:, j], rzg[:, j], sel[:, j], V, Rs, kappa)
            st, _ = capture_reps(raw[:, j], rpg[:, j], rzg[:, j], sel[:, j], V)
            corr_ref_rows.append({'benchmark': bench, 'setting': setting, 'reliability_source': f'{src}|{stat}',
                                  'reliability_label': label, 'primary_source': prim, 'kappa': kappa,
                                  **lib.summ(Rs, 'R_sel'), **lib.summ(st['perfect_share'], 'raw_reference'),
                                  **lib.summ(c['corrected_reference'], 'corrected_reference'),
                                  **lib.summ(c['bound_tau'], 'corrected_bound'), 'drugs_of': models[j]})
    for gname, gm in groups.items():
        ix = [models.index(m) for m in gm if m in models and np.isfinite(raw[0, models.index(m)]).any()]
        if not ix:
            continue
        cat = lambda a: np.concatenate([a[:, i] for i in ix], axis=1)  # noqa: E731
        Vc = np.concatenate([V] * len(ix), axis=1)
        st, ok = capture_reps(cat(raw), cat(rpg), cat(rzg), cat(sel), Vc)
        prow = {'benchmark': bench, 'setting': setting, 'group': gname, 'models': len(ix),
                'model_list': ';'.join(models[i] for i in ix)}
        for k, v in st.items():
            if k != 'drugs_weighted':
                prow.update(lib.summ(v, k))
        # median over models of per-model statistics (same draws)
        per = {k: np.vstack([capture_reps(raw[:, i], rpg[:, i], rzg[:, i], sel[:, i], V)[0][k] for i in ix])
               for k in ['axis_capture', 'selective_capture', 'share_ratio', 'cf_share_ratio', 'alignment_ratio']}
        for k, v in per.items():
            prow.update(lib.summ(np.nanmedian(v, 0), f'median_over_models_{k}'))
            prow[f'min_over_models_{k}'] = float(np.nanmin(v[:, 0]))
            prow[f'max_over_models_{k}'] = float(np.nanmax(v[:, 0]))
        pool_rows.append(prow)
        for src, stat, label, prim in rel.get(key, []):
            Rs = rel['draws'].get(f'{src}|{stat}')
            if Rs is None:
                continue
            for kappa in KAPPAS:
                c = corrected_reps(cat(raw), cat(rpg), cat(rzg), cat(sel), Vc, Rs, kappa)
                r2 = {'benchmark': bench, 'setting': setting, 'group': gname, 'models': len(ix),
                      'reliability_source': f'{src}|{stat}', 'reliability_label': label, 'primary_source': prim,
                      'kappa': kappa}
                for k, v in c.items():
                    r2.update(lib.summ(v, k))
                corr_pool_rows.append(r2)


# ----------------------------------------------------------------------------------------------------------------
def leaderboard_within():
    L = lib.leaderboard_setup()
    cd = L['cd']
    Z, Gm = L['dec'][cd.REFERENCE]
    U = L['U']
    models = list(L['P']) + [lib.MEASURED, lib.FITNESS]
    pseudo = {lib.MEASURED: Z.reindex(columns=U), lib.FITNESS: Gm.reindex(columns=U)}
    Zu, Gu = Z.reindex(columns=U), Gm.reindex(columns=U)
    drugs = [d for d in Z.index if d in L['drug_ix']]
    R = L['W'].shape[0]
    arrs = {s: np.full((R, len(models), len(drugs)), np.nan, np.float32) for s in ('raw', 'r_pg', 'r_zg', 'sel')}
    for j, d in enumerate(drugs):
        z, g = Zu.loc[d].values.astype(float), Gu.loc[d].values.astype(float)
        m0 = np.isfinite(z) & np.isfinite(g)
        for m in L['lbm']:
            if d in L['P'][m].index:
                m0 &= np.isfinite(L['P'][m].loc[d].values)
        if m0.sum() < lib.MIN_CELLS:
            continue
        idx = np.where(m0)[0]
        use, pc = [], []
        for m in models:
            src = pseudo[m] if m in pseudo else L['P'][m]
            if d not in src.index:
                continue
            p = src.loc[d].values[idx].astype(float)
            if not np.all(np.isfinite(p)) or np.std(p) < 1e-12:
                continue
            use.append(models.index(m))
            pc.append(p)
        st = lib.within_stats(z[idx], g[idx], np.column_stack(pc), L['W'][:, idx])
        for s in arrs:
            arrs[s][:, use, j] = st[s]
    V = L['V'][:, [L['drug_ix'][d] for d in drugs]]
    keep = np.isfinite(arrs['raw'][0]).any(0)
    arrs = {s: a[:, :, keep] for s, a in arrs.items()}
    V = V[:, keep]
    mclass = L['mclass']
    agn = {'Lasso', 'ElasticNet', 'MultiViewRandomForest'}
    groups = {'published 21 (paper)': L['published21'],
              'published 21 minus drug-agnostic (18)': [m for m in L['published21'] if m not in agn]}
    meta = lambda m: {'class': mclass.get(m, 'reference'), **flags_for('leaderboard', m)}  # noqa: E731
    return arrs, models, V, groups, meta


def retrain_within(setting):
    wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
    base = sorted(wp[(wp.setting == setting) & (wp.panel == 'main')].model.unique())
    rt = lib.load_module('rt59', '59_retrain_score.py')
    oof, _, _ = rt.load_oof(setting)
    extra = [m for m in DREVAL_LEARNED if m in oof and m not in base]
    S = lib.retrain_setup(setting, models=base + extra)
    O = S['obs'][S['obs'].cell.isin(S['cells'])][['drug', 'cell', 'response']]
    Z, G, _, _ = lib.selbench.decompose(O)
    Pm = {m: P.reindex(columns=Z.columns) for m, P in S['P'].items()}
    W, V = lib.draws(len(Z.columns), len(Z.index), 20261031 + sum(map(ord, setting)))
    arrs, models, drugs, ncell = lib.within_engine(Z, G, Pm, W, V, common_mask=True)
    if setting == 'CTRPv2_LNIC50':
        keepm = [i for i, m in enumerate(models) if not m.startswith('LB:')]
        arrs = {s: a[:, keepm] for s, a in arrs.items()}
        models = [models[i] for i in keepm]
    lm = [m for m in DREVAL_LEARNED if m in models]
    groups = {}
    if lm:
        groups['DrEval learned (retrained)'] = lm
    groups['per-drug ridges (this study)'] = [m for m in models if m.startswith('PerDrugRidge')]
    meta = lambda m: {'class': rt.MODEL_INFO.get(m, ('other',))[0], **flags_for('retrain', m)}  # noqa: E731
    return arrs, models, V, groups, meta, {'models': models, 'new_learned': extra, 'cells': len(Z.columns),
                                           'drugs': len(Z.index), 'incomplete': {k: str(v) for k, v in S['incomplete'].items()}}


def improve_within(I, ds, seed):
    Z, G = I['dec'][ds]
    Pm = {m: I['P_within'][(m, ds)] for m in I['models'] if (m, ds) in I['P_within']}
    Pm['floor_cell_mean'] = I['floors'][ds]
    Pm = {m: P.reindex(columns=Z.columns) for m, P in Pm.items()}
    W, V = lib.draws(len(Z.columns), len(Z.index), seed)
    arrs, models, drugs, ncell = lib.within_engine(Z, G, Pm, W, V, common_mask=False)
    groups = {'IMPROVE models (7)': [m for m in I['models'] if m in models]}
    meta = lambda m: {'class': 'IMPROVE model' if m != 'floor_cell_mean' else 'drug-agnostic floor',  # noqa: E731
                      **flags_for('improve', m)}
    return arrs, models, V, groups, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='*', default=['leaderboard', 'retrain', 'improve'])
    args = ap.parse_args()
    t0 = time.time()
    rel = {k: v for k, v in REL_MAP.items()}
    rp = ROUT / 'reliability_draws.npz'
    if rp.exists():
        z = np.load(rp)
        rel['draws'] = {k: z[k] for k in z.files}
        tp = ROUT / 'triad_R_draws.npz'
        if tp.exists():
            zt = np.load(tp)
            rel['draws'].update({k: zt[k] for k in zt.files})
    else:
        rel['draws'] = {}
        print('WARNING: no reliability draws; run 79_reliability.py first for the corrected reference', flush=True)
    rows, prow, crow, cprow, cref = [], [], [], [], []
    info = {}
    if 'leaderboard' in args.parts:
        arrs, models, V, groups, meta = leaderboard_within()
        summarize_setting('leaderboard', 'CTRPv2_lnIC50', arrs, models, V, groups, meta, rel, rows, prow, crow, cprow, cref)
        print('leaderboard done', f'{time.time() - t0:.0f}s', flush=True)
        del arrs
    if 'retrain' in args.parts:
        settings = ['GDSC2_AUC', 'CTRPv2_LNIC50', 'GDSC2_LNIC50']
        wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting'])
        if 'CTRPv2_AUC' in set(wp.setting):                # round 3: appended once 59 has scored it
            settings.append('CTRPv2_AUC')
        for setting in settings:
            arrs, models, V, groups, meta, inf = retrain_within(setting)
            info[setting] = inf
            summarize_setting('retrain', setting, arrs, models, V, groups, meta, rel, rows, prow, crow, cprow, cref)
            print('retrain', setting, inf['models'], f'{time.time() - t0:.0f}s', flush=True)
    if 'improve' in args.parts:
        I = lib.improve_setup(replicate_free=True, need_cross=False)
        for k, ds in enumerate(I['datasets']):
            arrs, models, V, groups, meta = improve_within(I, ds, 20261041 + k)
            summarize_setting('improve', f'{ds}_AUC', arrs, models, V, groups, meta, rel, rows, prow, crow, cprow, cref)
            print('improve', ds, f'{time.time() - t0:.0f}s', flush=True)
    tag = '' if set(args.parts) == {'leaderboard', 'retrain', 'improve'} else '_' + '_'.join(args.parts)
    M = pd.DataFrame(rows)
    # below the drug-agnostic floor (IMPROVE): median raw within 0.02 of the floor or below it
    if (M.benchmark == 'improve').any():
        fl = M[M.model == 'floor_cell_mean'].set_index('setting').median_raw
        M['below_or_near_floor'] = [bool(b == 'improve' and m != 'floor_cell_mean' and r <= fl.get(s, np.nan) + 0.02)
                                    for b, s, m, r in zip(M.benchmark, M.setting, M.model, M.median_raw)]
    M.to_csv(OUT / f'capture_by_model{tag}.csv', index=False, float_format='%.5g')
    P = pd.DataFrame(prow)
    P.to_csv(OUT / f'capture_pooled{tag}.csv', index=False, float_format='%.5g')
    pd.DataFrame(crow).to_csv(ROUT / f'corrected_capture_by_model{tag}.csv', index=False, float_format='%.5g')
    pd.DataFrame(cprow).to_csv(ROUT / f'corrected_capture_pooled{tag}.csv', index=False, float_format='%.5g')
    pd.DataFrame(cref).to_csv(ROUT / f'corrected_reference{tag}.csv', index=False, float_format='%.5g')
    # per-setting distribution over learned models (not drug-agnostic, not naive, not pseudo; median raw > 0.03)
    L_ = M[(M.drug_agnostic == '') & ~M.naive & ~M.reference_pseudo & (M.median_raw > 0.03)]
    S = L_.groupby(['benchmark', 'setting']).agg(
        models=('model', 'size'), axis_capture_median=('axis_capture', 'median'), axis_capture_min=('axis_capture', 'min'),
        axis_capture_max=('axis_capture', 'max'), selective_capture_median=('selective_capture', 'median'),
        selective_capture_min=('selective_capture', 'min'), selective_capture_max=('selective_capture', 'max'),
        alignment_ratio_min=('alignment_ratio', 'min'), alignment_ratio_max=('alignment_ratio', 'max'),
        share_ratio_min=('share_ratio', 'min'), share_ratio_max=('share_ratio', 'max'),
        cf_share_ratio_min=('cf_share_ratio', 'min'), cf_share_ratio_max=('cf_share_ratio', 'max'),
        perfect_share=('perfect_share', 'median'), bound=('bound', 'median')).reset_index()
    S.to_csv(OUT / f'capture_by_setting{tag}.csv', index=False, float_format='%.5g')
    lib.write_json(OUT / f'run{tag}.json', {'seconds': round(time.time() - t0), 'B': lib.B, 'info': info,
                                            'kappas': KAPPAS, 'reliability_map': {k: [list(x) for x in v]
                                                                                  for k, v in REL_MAP.items()},
                                            'learned_filter': 'not drug-agnostic, not naive, not pseudo, median raw > 0.03'})
    pd.set_option('display.width', 250)
    print(S.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
