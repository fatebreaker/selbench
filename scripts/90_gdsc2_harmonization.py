"""Round 2 (WS6b), task 1 (editor item 2, R1-M3c): the harmonization ladder applied to GDSC2-trained predictions.

66_harmonized_range.py's logic with GDSC2 CurveCurator AUC as the source. Predictions: drevalpy out-of-fold
GDSC2_AUC predictions (59; leave-cell-line-out, 10 splits) of DrEval's four learned models (EnsembleMF,
GradientBoosting, SimpleNeuralNetwork, RandomForest) and the per-drug ridges (raw, selective), plus the naive and
drug-agnostic references, scored as in 80_paired_engine.retrain_task on the cells predicted by every panel model.

For each target T (CTRPv2, CTRPv1, GDSC1, CCLE) and the drugs shared with GDSC2 (DrEval CurveCurator fits, per-drug
median tested ranges; common window [max(lo_S, lo_T), min(hi_S, hi_T)], kept if >= 1 log10 unit, as 66):
  a  unharmonized, all drugs        GDSC2 CC AUC (responses.parquet, as 59/80) -> T CC AUC, each dataset's own panel
                                    and axis (identical to 80's retrain pairs)
  a2 unharmonized, harmonized drugs  a restricted to the drugs kept for b/c (axes unchanged)
  b  shared panel                   both datasets' own-range CurveCurator AUC on the kept drugs only (axes over them)
  c  common tested range            both datasets' fitted curves summarized as mean viability over the common window
                                    (66.mean_viability), kept drugs only
Statistics (two-way bootstrap, B = 1,000: one cell-weight draw over the 941 panel cells and one drug-multiplicity draw
over every GDSC2 drug, shared by all steps, targets and models, so differences between steps are paired):
  mean raw r, axis term r_pg r_zg and selective term (raw - axis term) on S (within) and T (cross) and their changes;
  symmetric selective-both r(p, z | g_S, g_T) within and cross, and its change (82); proportional-attenuation
  contrasts (81: within share, share of the loss, excess, axis/selective-term retention and their difference);
  accuracy relative to the source measurement and observed vs expected retention sqrt(R_T / R_S) with R from screen
  triads (84; residual sets: DrEval CC AUC for a/a2, and for b/c the two datasets of the pair replaced by their
  harmonized versions; other datasets CC AUC; independent bootstrap, paired by index);
  step differences c - a2 and b - a2 (paired).
Pools: 'DrEval learned (4)' and 'per-drug ridges (2)'; single models.
Outputs (results/revision_round2/harmonization_gdsc2/): ranges.csv, versions.csv, arrays/*.npz, per_drug.csv.gz,
ladder.csv (one row per target x step x group), step_differences.csv, expected_retention.csv, summary.json.
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

OUT = lib.out_dir('harmonization_gdsc2')
ARR = OUT / 'arrays'
ARR.mkdir(exist_ok=True)
SEED = 20261090
TARGETS = ['CTRPv2', 'CTRPv1', 'GDSC1', 'CCLE']
MIN_WIDTH = 1.0
LEARNED = ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting']
RIDGES = ['PerDrugRidgeRaw', 'PerDrugRidgeSelective']
REF = 'ref:source_measurement'
STEPS = {'a': 'unharmonized, all drugs', 'a2': 'unharmonized, harmonized drugs', 'b': 'shared panel, own ranges',
         'c': 'shared panel, common tested range'}


def build_versions(resp):
    hr = lib.load_module('hr66', '66_harmonized_range.py')
    S = hr.curves('GDSC2')
    rS = S.groupby('pubchem_id')[['lo', 'hi']].median()
    obs = lambda ds: resp[(resp.dataset == ds) & (resp.endpoint == 'auc')][['drug', 'cell', 'response']]  # noqa: E731
    versions = {'GDSC2|CC|AUC': obs('GDSC2')}
    ranges, keep_by_t = [], {}
    for T in TARGETS:
        versions[f'{T}|CC|AUC'] = obs(T)
        D = hr.curves(T)
        rT = D.groupby('pubchem_id')[['lo', 'hi']].median()
        shared = rS.index.intersection(rT.index)
        rg = pd.DataFrame({'lo': np.maximum(rS.loc[shared, 'lo'], rT.loc[shared, 'lo']),
                           'hi': np.minimum(rS.loc[shared, 'hi'], rT.loc[shared, 'hi'])})
        rg['width'] = rg.hi - rg.lo
        keep = rg[rg.width >= MIN_WIDTH]
        keep_by_t[T] = list(keep.index)
        for frame, key in [(S, f'GDSC2|HR-{T}|AUC'), (D, f'{T}|HR-GDSC2|AUC')]:
            x = frame[frame.pubchem_id.isin(keep.index)].copy()
            x['v'] = hr.mean_viability(x, keep.loc[x.pubchem_id, 'lo'].values, keep.loc[x.pubchem_id, 'hi'].values)
            x = x.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).v.mean()
            versions[key] = pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values,
                                          'response': -x.v.values})
        for frame, key in [(S, f'GDSC2|SP-{T}|AUC'), (D, f'{T}|SP-GDSC2|AUC')]:
            x = frame[frame.pubchem_id.isin(keep.index)]
            x = x.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).AUC_curvecurator.mean()
            versions[key] = pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values,
                                          'response': -x.AUC_curvecurator.values})
        for d_, r in rg.iterrows():
            ranges.append({'target': T, 'drug': d_, 'lo_common': r.lo, 'hi_common': r.hi, 'width': r.width,
                           'lo_gdsc2': rS.loc[d_, 'lo'], 'hi_gdsc2': rS.loc[d_, 'hi'], 'lo_target': rT.loc[d_, 'lo'],
                           'hi_target': rT.loc[d_, 'hi'], 'kept': bool(r.width >= MIN_WIDTH)})
        print(T, 'shared drugs', len(shared), 'kept', len(keep), flush=True)
    return versions, pd.DataFrame(ranges), keep_by_t


def score_pair(Zs, Gs, Zt, Gt, P, models, U, W, V, drug_ix, drugs=None):
    """80.retrain_task with fixed draw universes (U cells, drug_ix drugs)."""
    cix = {c: i for i, c in enumerate(U)}
    pdrugs = sorted(set(Zs.index) & set(Zt.index) & set().union(*[set(P[m].index) for m in models if m != REF]))
    if drugs is not None:
        pdrugs = [d for d in pdrugs if d in set(drugs)]
    pdrugs = [d for d in pdrugs if d in drug_ix]
    pcells = sorted(set(Zs.columns) & set(Zt.columns) & set(U))
    Pm = dict(P)
    Pm[REF] = Zs
    allm = list(models) + [REF]
    acc = lib.PairedAccumulator(allm, W.shape[0], ref=REF)
    mats = [M.reindex(index=pdrugs, columns=pcells).values.astype(float) for M in (Zs, Gs, Zt, Gt)]
    Pv = {m: Pm[m].reindex(index=pdrugs, columns=pcells).values.astype(float) for m in allm}
    inP = {m: set(Pm[m].index) for m in allm}
    wcols = np.array([cix[c] for c in pcells])
    n = 0
    for j, d in enumerate(pdrugs):
        zs, gs, zt, gt = (M[j] for M in mats)
        have = [m for m in allm if d in inP[m]]
        ok = np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        for m in have:
            ok &= np.isfinite(Pv[m][j])
        if ok.sum() < lib.MIN_CELLS:
            continue
        idx = np.where(ok)[0]
        use = [m for m in have if np.std(Pv[m][j][idx]) >= 1e-12]
        st = lib.paired_stats(zs[idx], gs[idx], zt[idx], gt[idx], np.column_stack([Pv[m][j][idx] for m in use]),
                              W[:, wcols[idx]])
        acc.add(d, len(idx), st, use, V[:, drug_ix[d]])
        n += 1
    return acc, n, len(pcells)


def group_stats(S, models, gm):
    """Pooled contrasts over the models gm (sums over models and drugs), per draw."""
    g = lambda st, kind: lib.pooled(S, models, gm, st, kind)  # noqa: E731
    rw, rc, n = g('raw', 'sum_w'), g('raw', 'sum_c'), g('raw', 'n')
    mw, mc = g('mediated', 'sum_w'), g('mediated', 'sum_c')
    tw, tc = g('selective_term', 'sum_w'), g('selective_term', 'sum_c')
    sw, sc, sn = g('selBOTH_sym', 'sum_w'), g('selBOTH_sym', 'sum_c'), g('selBOTH_sym', 'n')
    aw, ac, an = g('selBOTH_asym', 'sum_w'), g('selBOTH_asym', 'sum_c'), g('selBOTH_asym', 'n')
    ix_ = [models.index(m) for m in gm if m in models]
    num, den = S['ref']['mod_selB_c'][:, ix_].sum(1), S['ref']['ref_selB_c'][:, ix_].sum(1)
    nref = S['ref']['n_ref'][:, ix_].sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(np.abs(rw - rc) > 1e-9, (mw - mc) / (rw - rc), np.nan)
        return {'raw_within': rw / n, 'raw_cross': rc / n, 'raw_loss': (rw - rc) / n,
                'axis_term_within': mw / n, 'axis_term_cross': mc / n, 'd_axis_term': (mc - mw) / n,
                'selective_term_within': tw / n, 'selective_term_cross': tc / n, 'd_selective_term': (tc - tw) / n,
                'within_share': mw / rw, 'share_of_loss': s, 'excess': s - mw / rw,
                'axis_retained': mc / mw, 'selterm_retained': tc / tw, 'retention_difference': mc / mw - tc / tw,
                'selBOTH_sym_within': sw / sn, 'selBOTH_sym_cross': sc / sn, 'd_selBOTH_sym': (sc - sw) / sn,
                'selBOTH_retained_sym': sc / sw, 'd_selBOTH_asym': (ac - aw) / an,
                'measurement_cross_selBOTH': den / nref, 'ratio_to_measurement': num / den}, int(round(n[0]))


def reading(row):
    loss_clear = row['raw_loss_lo'] > 0
    exc = row['excess_lo'] > 0 or row['excess_hi'] < 0
    if loss_clear and exc:
        return 'axis lost disproportionately' if row['excess'] > 0 else 'selective term lost disproportionately'
    if loss_clear:
        return 'clear loss, not distinguishable from proportional attenuation'
    if row['raw_loss_hi'] < 0:
        return 'clear gain in raw accuracy'
    return 'no clear raw loss'


def main():
    t0 = time.time()
    rt = lib.load_module('rt59', '59_retrain_score.py')
    sr = lib.load_module('sr84', '84_source_reference.py')
    resp = rt.responses()
    versions, ranges, keep_by_t = build_versions(resp)
    ranges.to_csv(OUT / 'ranges.csv', index=False, float_format='%.5g')
    dec = {}
    vrows = []
    for v, long in versions.items():
        Z, G, R, r2 = lib.selbench.decompose(long)
        dec[v] = (Z, G, R)
        vrows.append({'version': v, 'drugs': Z.shape[0], 'cells': Z.shape[1], 'median_axis_r2': float(r2.median())})
        print(v, Z.shape, f'median axis r2 {float(r2.median()):.3f}', flush=True)
    pd.DataFrame(vrows).to_csv(OUT / 'versions.csv', index=False, float_format='%.5g')
    # models: 59's GDSC2_AUC main panel (+ learned models), as 80
    wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
    base = sorted(wp[(wp.setting == 'GDSC2_AUC') & (wp.panel == 'main')].model.unique())
    S = lib.retrain_setup('GDSC2_AUC', models=base + [m for m in LEARNED if m not in base])
    models = S['models']
    U = np.array(sorted(S['cells']))
    all_drugs = sorted(set().union(*[set(S['P'][m].index) for m in models]))
    drug_ix = {d: i for i, d in enumerate(all_drugs)}
    W, V = lib.draws(len(U), len(all_drugs), SEED)
    print('models', models, 'cells', len(U), 'drugs', len(all_drugs), f'{time.time() - t0:.0f}s', flush=True)
    pairs, per_rows, saved = [], [], {}
    for T in TARGETS:
        steps = {'a': ('GDSC2|CC|AUC', f'{T}|CC|AUC', None), 'a2': ('GDSC2|CC|AUC', f'{T}|CC|AUC', keep_by_t[T]),
                 'b': (f'GDSC2|SP-{T}|AUC', f'{T}|SP-GDSC2|AUC', None), 'c': (f'GDSC2|HR-{T}|AUC', f'{T}|HR-GDSC2|AUC', None)}
        for step, (sv, tv, drugs) in steps.items():
            Zs, Gs, _ = dec[sv]
            Zt, Gt, _ = dec[tv]
            acc, n, nc = score_pair(Zs, Gs, Zt, Gt, S['P'], models, U, W, V, drug_ix, drugs)
            key = f'{T}__{step}'
            acc.save(ARR / f'{key}.npz', {'target': T, 'step': step, 'source_version': sv, 'target_version': tv})
            per_rows.append(pd.DataFrame(acc.rows).assign(target=T, step=step))
            saved[(T, step)] = lib.load_paired(ARR / f'{key}.npz')
            pairs.append({'target': T, 'step': step, 'step_label': STEPS[step], 'source_version': sv,
                          'target_version': tv, 'drugs_scored': n, 'cells': nc})
            print(T, step, n, 'drugs', nc, 'cells', f'{time.time() - t0:.0f}s', flush=True)
    per = pd.concat(per_rows, ignore_index=True)
    per.to_csv(OUT / 'per_drug.csv.gz', index=False, float_format='%.6g')
    groups = {'DrEval learned (4)': LEARNED, 'per-drug ridges (2)': RIDGES}
    for m in LEARNED + RIDGES + ['NaiveTissueDrugMeanPredictor', 'FitnessOnlyRidge']:
        if m in models:
            groups[m] = [m]
    rows, draws_keep = [], {}
    for p in pairs:
        mods, meta, Sx = saved[(p['target'], p['step'])]
        for gname, gm in groups.items():
            gm = [m for m in gm if m in mods]
            if not gm:
                continue
            st, nmd = group_stats(Sx, mods, gm)
            draws_keep[(p['target'], p['step'], gname)] = st
            row = {**p, 'group': gname, 'models': len(gm), 'model_drug_pairs': nmd}
            for k, v in st.items():
                row.update(lib.summ(v, k))
            row['loss_clear'] = bool(row['raw_loss_lo'] > 0)
            row['excess_ci_excludes_0'] = bool(row['excess_lo'] > 0 or row['excess_hi'] < 0)
            row['retdiff_ci_excludes_0'] = bool(row['retention_difference_lo'] > 0 or row['retention_difference_hi'] < 0)
            row['selBOTH_loss_clear'] = bool(row['d_selBOTH_sym_hi'] < 0)
            row['reading'] = reading(row)
            rows.append(row)
    L = pd.DataFrame(rows)
    # ---------------- triads: expected retention for a predictor of shared selectivity (84)
    cd = lib.load_module('cd41', '41_crosslab_data.py')
    Vv = cd.versions()
    base_res = {ds: lib.selbench.decompose(Vv[f'{ds}|CC|AUC'])[2] for ds in cd.DATASETS}
    corr_cache = {}
    erows = []
    for k, T in enumerate(TARGETS):
        for step in STEPS:
            if step in ('a', 'a2'):
                fam = 'unharmonized'
                Rs = base_res
            else:
                fam = f'{T}|{step}'
                tag = {'b': 'SP', 'c': 'HR'}[step]
                Rs = dict(base_res)
                Rs['GDSC2'] = dec[f'GDSC2|{tag}-{T}|AUC'][2]
                Rs[T] = dec[f'{T}|{tag}-GDSC2|AUC'][2]
            if fam not in corr_cache:
                corr_cache[fam] = sr.pair_corrs(Rs, SEED + 1 + len(corr_cache))
            corr, drugs, Vt, ce = corr_cache[fam]
            pdrugs = sorted(per[(per.target == T) & (per.step == step)].drug.astype(str).unique())
            RS, nS = sr.triad_R(corr, drugs, Vt, ce, 'GDSC2', pdrugs)
            RT, nT = sr.triad_R(corr, drugs, Vt, ce, T, pdrugs)
            with np.errstate(invalid='ignore', divide='ignore'):
                exp_ret = np.sqrt(np.clip(RT, 0, None) / RS)
            for gname in groups:
                st = draws_keep.get((T, step, gname))
                if st is None:
                    continue
                obs_ret = st['selBOTH_retained_sym']
                erows.append({'target': T, 'step': step, 'step_label': STEPS[step], 'group': gname,
                              'drugs_pair': len(pdrugs), 'drugs_triads_S': nS, 'drugs_triads_T': nT,
                              **lib.summ(RS, 'R_source'), **lib.summ(RT, 'R_target'),
                              **lib.summ(exp_ret, 'expected_retention'), **lib.summ(obs_ret, 'observed_retention'),
                              **lib.summ(obs_ret - exp_ret, 'observed_minus_expected')})
        print('triads', T, f'{time.time() - t0:.0f}s', flush=True)
    E = pd.DataFrame(erows)
    E.to_csv(OUT / 'expected_retention.csv', index=False, float_format='%.5g')
    L = L.merge(E[['target', 'step', 'group'] + [c for c in E.columns if c.startswith(('expected_retention',
                                                                                         'observed_minus_expected',
                                                                                         'R_source', 'R_target'))]],
                on=['target', 'step', 'group'], how='left')
    L.to_csv(OUT / 'ladder.csv', index=False, float_format='%.5g')
    # ---------------- step differences (paired draws)
    drows = []
    for T in TARGETS:
        for gname in groups:
            for s1, s0 in [('c', 'a2'), ('b', 'a2'), ('c', 'b'), ('a2', 'a')]:
                A, B0 = draws_keep.get((T, s1, gname)), draws_keep.get((T, s0, gname))
                if A is None or B0 is None:
                    continue
                row = {'target': T, 'group': gname, 'contrast': f'{s1} - {s0}',
                       'contrast_label': f'{STEPS[s1]} minus {STEPS[s0]}'}
                for k in ['raw_loss', 'd_axis_term', 'd_selective_term', 'd_selBOTH_sym', 'excess',
                          'retention_difference', 'selBOTH_retained_sym', 'raw_within', 'raw_cross',
                          'selBOTH_sym_within', 'selBOTH_sym_cross', 'axis_retained', 'selterm_retained']:
                    row.update(lib.summ(A[k] - B0[k], f'diff_{k}'))
                drows.append(row)
    D = pd.DataFrame(drows)
    D.to_csv(OUT / 'step_differences.csv', index=False, float_format='%.5g')
    # ---------------- check: step a reproduces 80's retrain pairs (point estimates)
    chk = {}
    try:
        p80 = pd.read_csv(lib.OUT2 / 'paired' / 'pairs.csv')
        p80 = p80[(p80.benchmark == 'retrain') & (p80.family == 'GDSC2_AUC')]
        for _, r in p80.iterrows():
            T = r.target.split('|')[0]
            m80, _, S80 = lib.load_paired(lib.OUT2 / 'paired' / 'arrays' / f'retrain__{r.k}.npz')
            st80, _ = group_stats(S80, m80, [m for m in LEARNED if m in m80])
            st, _ = draws_keep[(T, 'a', 'DrEval learned (4)')], None
            chk[T] = {k: float(abs(st80[k][0] - st[k][0])) for k in ['raw_within', 'raw_cross', 'd_selBOTH_sym', 'excess']}
    except Exception as e:  # noqa: BLE001
        chk['error'] = repr(e)
    lib.write_json(OUT / 'summary.json', {'seconds': round(time.time() - t0), 'B': lib.B, 'seed': SEED,
                                          'models': models, 'cells': len(U), 'pairs': pairs,
                                          'check_step_a_vs_80_abs_point_diff': chk,
                                          'kept_drugs': {T: len(v) for T, v in keep_by_t.items()}})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 300)
    cols = ['target', 'step', 'group', 'model_drug_pairs', 'raw_within', 'raw_cross', 'raw_loss', 'raw_loss_lo',
            'raw_loss_hi', 'd_axis_term', 'd_selective_term', 'd_selBOTH_sym', 'd_selBOTH_sym_lo', 'd_selBOTH_sym_hi',
            'excess', 'excess_lo', 'excess_hi', 'observed_retention' if 'observed_retention' in L else 'selBOTH_retained_sym',
            'expected_retention', 'reading']
    print(L[L.group.isin(['DrEval learned (4)', 'per-drug ridges (2)'])][[c for c in cols if c in L]].round(3).to_string(index=False))
    print(D[D.group == 'DrEval learned (4)'][['target', 'contrast', 'diff_d_selBOTH_sym', 'diff_d_selBOTH_sym_lo',
                                               'diff_d_selBOTH_sym_hi', 'diff_excess', 'diff_raw_loss']].round(3).to_string(index=False))
    print('check vs 80:', chk)
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
