"""Round 3, task 2a (editor item 5; R3-M4): expected retention for the tissue-by-drug baseline and the drug-agnostic
floor beside the learned models and the ridges.

The benchmark (84, 90): a predictor of only the selectivity that two datasets share keeps about sqrt(R_T / R_S) of its
selective-both across datasets, R_X being X's reproducible share of selective variance from screen triads. R3 found
that DrEval's tissue-by-drug mean, which uses no molecular data, falls short of it by as much as the learned models.
If so, the learned models' shortfall is no evidence of dataset-specific selectivity beyond tissue level. Here, for
every setting where a tissue-by-drug baseline exists:
  observed retention     pooled cross / within symmetric selective-both (ratio of sums over drugs and models)
  expected retention     sqrt(R_T / R_S) on the pair's drugs; R from 84's / 90's triads, recomputed with their seeds
                         (so the values reproduce 84's and 90's expected_retention.csv; checked)
  observed - expected    per group (independent bootstraps paired by index, as 84/90)
  shortfall difference   (group observed - expected) - (tissue observed - expected) = group observed retention -
                         tissue observed retention: paired within the same two-way draws (same cells, drugs and
                         draws; the expected value cancels). > 0: the group keeps more than the tissue baseline.
The drug-agnostic floor's within selective-both is ~0 (it has no drug-specific signal), so its retention ratio is
undefined; its within and cross selective-both are reported instead.
Settings
  GDSC2 AUC-trained (90)     drevalpy out-of-fold predictions; targets CTRPv2, CTRPv1, GDSC1, CCLE; 90's ladder steps
                             a (unharmonized), a2 (harmonized drugs), b (shared panel), c (shared panel + common range);
                             groups DrEval learned (4), per-drug ridges (2), single models, NaiveTissueDrugMeanPredictor,
                             FitnessOnlyRidge, NaiveMeanEffectsPredictor (90's arrays; nothing rescored).
  CTRPv2-trained leaderboard DrEval's released predictions (80's arrays): 84's pairs (CC ln IC50 -> CC ln IC50 targets,
                             CC AUC -> CC AUC targets); published 21, published 18 (no drug-agnostic), EnsembleMF,
                             NaiveTissueDrugMeanPredictor, fitness_only.
  retraining (80 retrain)    GDSC2 ln IC50, CTRPv2 ln IC50 (and CTRPv2 AUC once run): DrEval learned pool,
                             NaiveTissueDrugMeanPredictor (or LB:), FitnessOnlyRidge.
  IMPROVE (80 improve)       no tissue baseline is released; a training-fold tissue x drug mean (91's construction: per
                             split, the drug's mean response over the other splits' pairs in the test cell's tissue) is
                             scored as a pseudo-model on 80's own-panel pairs with 80's seeds (the IMPROVE models'
                             pooled values reproduce 80's arrays; checked).
Outputs (results/revision_round2/retention_calibration/): baselines_gdsc2.csv, baselines_pairs.csv,
shortfall_differences.csv, checks.json.
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

O2 = lib.OUT2
OUT = lib.out_dir('retention_calibration')
LEARNED = ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting']
RIDGES = ['PerDrugRidgeRaw', 'PerDrugRidgeSelective']
CHECKS = {}


def retention_draws(S, models, gm):
    g = lambda st, kind: lib.pooled(S, models, gm, st, kind)  # noqa: E731
    sw, sc, n = g('selBOTH_sym', 'sum_w'), g('selBOTH_sym', 'sum_c'), g('selBOTH_sym', 'n')
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'within': sw / n, 'cross': sc / n, 'retained': sc / sw}


def group_rows(prefix, S, models, groups, exp_ret, tissue):
    """Rows per group and shortfall differences against the tissue baseline (paired draws)."""
    rows, drows, draws = [], [], {}
    for gname, gm in groups.items():
        gm = [m for m in gm if m in models]
        if not gm:
            continue
        d = retention_draws(S, models, gm)
        draws[gname] = d
        row = {**prefix, 'group': gname, 'models': len(gm),
               **lib.summ(d['within'], 'selBOTH_within'), **lib.summ(d['cross'], 'selBOTH_cross'),
               **lib.summ(d['retained'], 'observed_retention')}
        row['within_ci_includes_0'] = bool(row['selBOTH_within_lo'] <= 0 <= row['selBOTH_within_hi'])
        if exp_ret is not None:
            row.update(lib.summ(exp_ret, 'expected_retention'))
            row.update(lib.summ(d['retained'] - exp_ret, 'observed_minus_expected'))
            if row['within_ci_includes_0']:
                for k in ['observed_retention', 'observed_minus_expected']:
                    row[f'{k}_note'] = 'undefined: within selective-both interval includes 0'
        rows.append(row)
    if tissue in draws:
        t = draws[tissue]['retained']
        for gname, d in draws.items():
            if gname == tissue:
                continue
            diff = d['retained'] - t
            r = {**prefix, 'group': gname, 'versus': tissue, **lib.summ(diff, 'retention_minus_tissue')}
            r['retention_minus_tissue_excludes_0'] = bool(r['retention_minus_tissue_lo'] > 0 or
                                                          r['retention_minus_tissue_hi'] < 0)
            r.update(lib.summ(d['cross'] - draws[tissue]['cross'], 'cross_selBOTH_minus_tissue'))
            drows.append(r)
    return rows, drows


# ----------------------------------------------------------------------------------------------------------------
def gdsc2_setting(t0):
    """90's arrays; expected-retention draws recomputed exactly as 90.main (same residual sets and seeds)."""
    h90 = lib.load_module('h90', '90_gdsc2_harmonization.py')
    sr = lib.load_module('sr84', '84_source_reference.py')
    rt = lib.load_module('rt59', '59_retrain_score.py')
    resp = rt.responses()
    versions, _, _ = h90.build_versions(resp)
    need = [v for v in versions if '|HR-' in v or '|SP-' in v]
    dec = {v: lib.selbench.decompose(versions[v]) for v in need}
    per = pd.read_csv(O2 / 'harmonization_gdsc2' / 'per_drug.csv.gz', usecols=['target', 'step', 'drug'],
                      dtype={'drug': str})
    cd = lib.load_module('cd41', '41_crosslab_data.py')
    Vv = cd.versions()
    base_res = {ds: lib.selbench.decompose(Vv[f'{ds}|CC|AUC'])[2] for ds in cd.DATASETS}
    corr_cache = {}
    old = pd.read_csv(O2 / 'harmonization_gdsc2' / 'expected_retention.csv')
    rows, drows, chk = [], [], []
    for T in h90.TARGETS:
        for step in h90.STEPS:
            if step in ('a', 'a2'):
                fam, Rs = 'unharmonized', base_res
            else:
                fam = f'{T}|{step}'
                tag = {'b': 'SP', 'c': 'HR'}[step]
                Rs = dict(base_res)
                Rs['GDSC2'] = dec[f'GDSC2|{tag}-{T}|AUC'][2]
                Rs[T] = dec[f'{T}|{tag}-GDSC2|AUC'][2]
            if fam not in corr_cache:
                corr_cache[fam] = sr.pair_corrs(Rs, h90.SEED + 1 + len(corr_cache))
            corr, drugs, Vt, ce = corr_cache[fam]
            pdrugs = sorted(per[(per.target == T) & (per.step == step)].drug.unique())
            RS, _ = sr.triad_R(corr, drugs, Vt, ce, 'GDSC2', pdrugs)
            RT, _ = sr.triad_R(corr, drugs, Vt, ce, T, pdrugs)
            with np.errstate(invalid='ignore', divide='ignore'):
                exp_ret = np.sqrt(np.clip(RT, 0, None) / RS)
            o = old[(old.target == T) & (old.step == step)].iloc[0]
            chk.append(abs(o.expected_retention - exp_ret[0]))
            models, meta, S = lib.load_paired(O2 / 'harmonization_gdsc2' / 'arrays' / f'{T}__{step}.npz')
            groups = {'DrEval learned (4)': LEARNED, 'per-drug ridges (2)': RIDGES}
            for m in LEARNED + RIDGES + ['NaiveTissueDrugMeanPredictor', 'FitnessOnlyRidge', 'NaiveMeanEffectsPredictor']:
                groups[m] = [m]
            prefix = {'setting': 'GDSC2 AUC-trained (90)', 'target': T, 'step': step, 'step_label': h90.STEPS[step],
                      'drugs_pair': len(pdrugs), **lib.summ(RS, 'R_source'), **lib.summ(RT, 'R_target')}
            r, d = group_rows(prefix, S, models, groups, exp_ret, 'NaiveTissueDrugMeanPredictor')
            rows += r
            drows += d
        print('GDSC2 setting', T, f'{time.time() - t0:.0f}s', flush=True)
    CHECKS['gdsc2_expected_retention_max_abs_diff_vs_90'] = float(np.max(chk))
    return rows, drows


# ----------------------------------------------------------------------------------------------------------------
def tissue_improve_task(I, t, tissue_P):
    """80.improve_task for one own-panel pair with a training-fold tissue x drug mean added (80's drugs and seed)."""
    src, trg = t['source'], t['target']
    (Zs, Gs), (Zt, Gt) = I['dec'][src], I['dec'][trg]
    Pm = {m: I['P_cross'][(m, src, trg)] for m in I['models'] if (m, src, trg) in I['P_cross']}
    if not Pm:
        return None
    REF = 'source_measurement'
    template = next(iter(Pm.values()))
    Pref = lib.improve_source_measurement(I, src, trg, template)
    cells = Zs.columns.intersection(Zt.columns)
    drugs = Zs.index.intersection(Zt.index).intersection(pd.Index(sorted(set().union(*[set(P.index) for P in Pm.values()]))))
    # the tissue baseline on the pairs the models predicted in the target (any model), identical within and cross
    anyP = None
    for P in Pm.values():
        P = P.reindex(index=drugs, columns=cells)
        anyP = P.notna() if anyP is None else (anyP | P.notna())
    TB = tissue_P.reindex(index=drugs, columns=cells).where(anyP)
    allm = list(Pm) + [REF, 'tissue_drug_mean']
    Pm[REF] = Pref
    Pm['tissue_drug_mean'] = TB
    W, V = lib.draws(len(cells), len(drugs), t['seed'])
    acc = lib.PairedAccumulator(allm, W.shape[0], ref=REF)
    mats = [M.reindex(index=drugs, columns=cells).values.astype(float) for M in (Zs, Gs, Zt, Gt)]
    Pv = {m: Pm[m].reindex(index=drugs, columns=cells).values.astype(float) for m in allm}
    for j, d in enumerate(drugs):
        zs, gs, zt, gt = (M[j] for M in mats)
        base = np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        grp = {}
        for m in allm:
            mk = base & np.isfinite(Pv[m][j])
            if mk.sum() < lib.MIN_CELLS:
                continue
            grp.setdefault(mk.tobytes(), (mk, []))[1].append(m)
        for mk, ms in grp.values():
            idx = np.where(mk)[0]
            cols, use = [], []
            for m in ms:
                p = Pv[m][j][idx]
                if m == REF:
                    p = lib.selbench.rank_normal(p)
                if np.std(p) < 1e-12:
                    continue
                cols.append(p)
                use.append(m)
            if use:
                st = lib.paired_stats(zs[idx], gs[idx], zt[idx], gt[idx], np.column_stack(cols), W[:, idx])
                acc.add(d, len(idx), st, use, V[:, j])
    return acc


def pairs_settings(t0):
    """Leaderboard, retraining and IMPROVE pairs with 84's expected retention (recomputed with 84's seeds)."""
    sr = lib.load_module('sr84', '84_source_reference.py')
    ix = pd.read_csv(O2 / 'paired' / 'pairs.csv')
    old = pd.read_csv(O2 / 'source_reference' / 'expected_retention.csv')
    rsets = sr.residual_sets()
    corr_cache = {}
    for k, (fam, Rs) in enumerate(rsets.items()):
        corr_cache[fam] = sr.pair_corrs(Rs, sr.SEED + k)
    print('triad correlations', f'{time.time() - t0:.0f}s', flush=True)
    per = {bm: pd.read_csv(O2 / 'paired' / f'per_drug_{bm}.csv.gz', usecols=['pair', 'drug'], dtype={'drug': str})
           for bm in ['leaderboard', 'retrain', 'improve']}
    # IMPROVE tissue baseline (91's construction) per source dataset
    I = lib.improve_setup(replicate_free=True)
    lin = lib.load_module('lin91', '91_lineage.py')
    tis, _ = lin.tissue_of_depmap()
    im = I['im']
    tissueP = {}
    for ds in I['datasets']:
        for model in im.MODELS:
            w = im.load_preds(ds, ds, model)
            if w is not None:
                break
        w = im.drop_split_replicates(w).sort_values('split', kind='stable').drop_duplicates([im.DRUG, im.CELL])
        tissueP[ds] = lin.tissue_drug_mean(w, tis)
    print('IMPROVE set-up', f'{time.time() - t0:.0f}s', flush=True)
    rows, drows, chk, chk_im = [], [], [], []
    for _, r in ix.iterrows():
        if r.source == r.target or r.drugs_used == 0:
            continue
        if r.benchmark == 'leaderboard':
            if not (r.family in ('lnIC50-reference', 'AUC-reference') and '|CC|' in r.target
                    and not r.target.startswith('CTRPv2')) or (r.family == 'lnIC50-reference'
                                                                and not r.target.endswith('lnIC50')):
                continue
            fam = 'DrEval CC ' + r.target.split('|')[-1]
            sname, tname = r.source.split('|')[0], r.target.split('|')[0]
        elif r.benchmark == 'retrain':
            fam = 'DrEval CC ' + ('lnIC50' if 'LNIC50' in r.family else 'AUC')
            sname, tname = r.source.split('|')[0], r.target.split('|')[0]
        else:
            if r.family != 'own_panel':
                continue
            fam = 'IMPROVE AUC'
            sname, tname = r.source, r.target
        corr, drugs, V, ce = corr_cache[fam]
        pdrugs = sorted(per[r.benchmark].query('pair == @r.pair').drug.unique())
        RS, _ = sr.triad_R(corr, drugs, V, ce, sname, pdrugs)
        RT, _ = sr.triad_R(corr, drugs, V, ce, tname, pdrugs)
        with np.errstate(invalid='ignore', divide='ignore'):
            exp_ret = np.sqrt(np.clip(RT, 0, None) / RS)
        o = old[(old.benchmark == r.benchmark) & (old.pair == r.pair)]
        if len(o):
            chk.append(abs(o.expected_retention.iloc[0] - exp_ret[0]))
        if r.benchmark == 'improve':
            t = {'source': r.source, 'target': r.target, 'seed': 20261016 + int(r.k)}
            acc = tissue_improve_task(I, t, tissueP[r.source])
            if acc is None:
                continue
            tmp = OUT / '_tmp_improve.npz'
            acc.save(tmp, {})
            models, _, S = lib.load_paired(tmp)
            tmp.unlink()
            # check: the IMPROVE models' pooled sums reproduce 80's arrays
            m80, _, S80 = lib.load_paired(O2 / 'paired' / 'arrays' / f'improve__{r.k}.npz')
            gm = [m for m in r.pool_main.split(';') if m in m80]
            chk_im.append(float(np.max(np.abs(lib.pooled(S, models, gm, 'selBOTH_sym', 'sum_c') -
                                               lib.pooled(S80, m80, gm, 'selBOTH_sym', 'sum_c')))))
            groups = {'IMPROVE 7 (pooled)': gm, 'tissue_drug_mean': ['tissue_drug_mean']}
            tissue = 'tissue_drug_mean'
        else:
            models, _, S = lib.load_paired(O2 / 'paired' / 'arrays' / f'{r.benchmark}__{r.k}.npz')
            if r.benchmark == 'leaderboard':
                groups = {'published 21 (pooled)': r.pool_main.split(';'),
                          'published 18, no drug-agnostic (pooled)': r.pool_alt.split(';'),
                          'EnsembleMF': ['EnsembleMF'], 'RandomForest': ['RandomForest'],
                          'NaiveTissueDrugMeanPredictor': ['NaiveTissueDrugMeanPredictor'],
                          'fitness_only': ['fitness_only']}
                tissue = 'NaiveTissueDrugMeanPredictor'
            else:
                tb = 'NaiveTissueDrugMeanPredictor' if 'NaiveTissueDrugMeanPredictor' in models else \
                    'LB:NaiveTissueDrugMeanPredictor'
                lbm = [f'LB:{m}' for m in LEARNED if f'LB:{m}' in models]
                lm = lbm if lbm else [m for m in LEARNED if m in models]
                groups = {'DrEval learned (pooled)': lm, 'per-drug ridges (2)': RIDGES, tb: [tb],
                          'FitnessOnlyRidge': ['FitnessOnlyRidge']}
                if 'LB:EnsembleMF' in models:
                    groups['published 21 (LB, pooled)'] = r.pool_main.split(';')
                tissue = tb
        prefix = {'benchmark': r.benchmark, 'pair': r.pair, 'family': fam, 'drugs_pair': len(pdrugs),
                  **lib.summ(RS, 'R_source'), **lib.summ(RT, 'R_target')}
        rr, dd = group_rows(prefix, S, models, groups, exp_ret, tissue)
        rows += rr
        drows += dd
    CHECKS['pairs_expected_retention_max_abs_diff_vs_84'] = float(np.max(chk)) if chk else None
    CHECKS['improve_models_cross_sum_max_abs_diff_vs_80'] = float(np.max(chk_im)) if chk_im else None
    print('pairs done', f'{time.time() - t0:.0f}s', flush=True)
    return rows, drows


def main():
    t0 = time.time()
    r1, d1 = gdsc2_setting(t0)
    r2, d2 = pairs_settings(t0)
    G = pd.DataFrame(r1)
    G.to_csv(OUT / 'baselines_gdsc2.csv', index=False, float_format='%.5g')
    P = pd.DataFrame(r2)
    P.to_csv(OUT / 'baselines_pairs.csv', index=False, float_format='%.5g')
    D = pd.DataFrame(d1 + d2)
    D.to_csv(OUT / 'shortfall_differences.csv', index=False, float_format='%.5g')
    lib.write_json(OUT / 'checks.json', {**CHECKS, 'seconds': round(time.time() - t0), 'B': lib.B})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    cols = ['target', 'step', 'group', 'selBOTH_within', 'selBOTH_cross', 'observed_retention', 'observed_retention_lo',
            'observed_retention_hi', 'expected_retention', 'observed_minus_expected', 'observed_minus_expected_lo',
            'observed_minus_expected_hi']
    print(G[cols].round(3).to_string(index=False))
    print(D[['setting', 'benchmark', 'target', 'step', 'pair', 'group', 'retention_minus_tissue',
             'retention_minus_tissue_lo', 'retention_minus_tissue_hi']].round(3).to_string(index=False))
    print(CHECKS)


if __name__ == '__main__':
    main()
