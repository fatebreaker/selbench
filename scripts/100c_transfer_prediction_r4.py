"""Round 4 revision of the transfer-prediction analysis (editor 5a item 1; R1-M1, R3-M1, R1-m10).

Reuses 100a_transfer_prediction.py (imported read-only: its per-draw summaries, estimators, pooling rules, weighted
Spearman, decision rule and bias-corrected intervals) and changes only what the round-4 reports asked for:

1. Units. Primary retraining units = DrEval's models plus DrEval's (drevalpy) naive baselines only: EnsembleMF,
   GradientBoosting, RandomForest, SimpleNeuralNetwork, NaiveMeanEffectsPredictor (drug-agnostic) and
   NaiveTissueDrugMeanPredictor. Excluded from the primary set: this study's own constructs, i.e. the references
   PerDrugRidgeRaw, PerDrugRidgeSelective and FitnessOnlyRidge and the selective-target variants EnsembleMFSelective,
   GradientBoostingSelective and GradientBoostingZ (DrEval architectures trained by this study on its own targets).
   Learned only = the four DrEval learned models (both naive baselines out; R1-m10). Sensitivity unit sets: the
   round-4 units ('all_previous_units', 'learned_previous_units'), the round-4 units without the three references
   only ('no_own_refs'), and the four learned models plus the tissue-by-drug mean ('learned_plus_tissue'). Leaderboard:
   21 published models (primary) and 18 learned; IMPROVE: 7 models (+ the cell-mean floor as a sensitivity). Every
   unit of every setting is listed in units_by_setting.csv. The retraining CTRP v2 ln IC50 setting has one DrEval
   model and is not analysed under the primary units (as before, it is not pooled).
2. Resampling. Drug and cell-line multiplicities are drawn once per bootstrap draw and shared by every model set,
   statistic, setting and target of a benchmark (keyed by drug and cell identifiers): leaderboard as in 100a (42's
   draws, reused from 100a's cache); retraining and IMPROVE re-computed here with one draw over the union of all
   settings' (sources') cells and drugs (100a drew them separately per setting/source). Benchmarks are resampled
   independently of each other (different prediction sets and identifier systems). Model-cluster multiplicities
   are drawn per model set (multinomial over that set's members, the same in every target of the set; 100a drew them
   over the union of a benchmark's names and subset them, so the effective number of models varied across draws).
   Where a set's membership differs between retraining settings (only the round-4 sensitivity sets), each setting's
   set is its own model set. B = 1,000. Point estimates (draw 0) are unchanged by construction and are checked
   against 100a's cache.
3. Intervals. Percentile and bias-corrected percentile (Efron; 100a.bc_interval) for every correlation, contrast,
   concordance and decision difference. Decision intervals condition on the menu of models (drug x cell draws only).
4. Pre-stated primary comparison (named here; everything else secondary): leaderboard, 21 published models, the
   four CurveCurator ln IC50 targets (CTRP v1, GDSC1, GDSC2, CCLE), outcome = mean selective-both in the target,
   contrast = pooled Spearman of selective capture S minus that of the raw median per-drug correlation (Fisher-z
   mean over the four targets with equal weights; equal weights are a choice, stated as such).
   Disclosure: the decision analysis's regret was redefined after the first run of 100a (technical reason in 100a's
   header); the S-minus-raw choice contrast is identical under both definitions.
5. Added outputs: S minus DrEval's normalized metric; the mean selective correlation (sel_mean) and its contrasts;
   the raw-S rank concordance per setting (Spearman across the set's models between the within-dataset raw median
   correlation and S; also raw-normalized and S-sel_mean); the learned-only overall contrast; the GDSC2 ln IC50
   contrast with its interval under the primary units; all three k of the decision analysis, all and learned-only.
Outputs (results/revision_round4/transfer_prediction_r4/): cache/ (retrain_*, improve_* recomputed; leaderboard_*
symlinked to 100a's cache), units_by_setting.csv, units.csv, correlations_by_target.csv, correlations_pooled.csv,
contrast_pooled.csv, contrast_by_target.csv, concordance.csv, decision_pooled.csv, fig_b_pooled_correlations.csv and
fig_d_decision_regret.csv (100a's format, model_set 'all' = primary units, 'learned' = learned only; extra columns
appended), primary_comparison.json, key_numbers_r4.json, verification_point_vs_100a.csv, run.json.
Usage: 100c_transfer_prediction_r4.py [--compute] [--analyse]
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('tp100a', HERE / '100a_transfer_prediction.py')
tp = importlib.util.module_from_spec(_spec)
sys.modules['tp100a'] = tp
_spec.loader.exec_module(tp)
lib = tp.lib

OUT = lib.RES / 'revision_round4' / 'transfer_prediction_r4'
CACHE = OUT / 'cache'
OLD_CACHE = lib.RES / 'revision_round4' / 'transfer_prediction' / 'cache'
SEED_DRAWS = {'retrain': 20261140, 'improve': 20261150}
SEED_MODELS = 20261160
DREVAL_LEARNED = ['EnsembleMF', 'GradientBoosting', 'RandomForest', 'SimpleNeuralNetwork']
DREVAL_NAIVE = ['NaiveMeanEffectsPredictor', 'NaiveTissueDrugMeanPredictor']
OWN_REFS = ['PerDrugRidgeRaw', 'PerDrugRidgeSelective', 'FitnessOnlyRidge']
OWN_SELECTIVE_TARGET = ['EnsembleMFSelective', 'GradientBoostingSelective', 'GradientBoostingZ',
                        'SimpleNeuralNetworkSelective']
IMPROVE7 = ['deepcdr', 'deepttc', 'graphdrp', 'hidra', 'lgbm', 'tcnns', 'uno']
# overall (three-benchmark) schemes: which set of each benchmark enters
SCHEMES = {
    'all': {'leaderboard': 'all', 'retrain': 'all', 'improve': 'all'},
    'learned': {'leaderboard': 'learned', 'retrain': 'learned', 'improve': 'all'},
    'learned_plus_tissue': {'leaderboard': 'learned', 'retrain': 'learned_plus_tissue', 'improve': 'all'},
    'no_own_refs': {'leaderboard': 'all', 'retrain': 'no_own_refs', 'improve': 'all'},
    'all_previous_units': {'leaderboard': 'all', 'retrain': 'all_previous_units', 'improve': 'all'},
    'learned_previous_units': {'leaderboard': 'learned', 'retrain': 'learned_previous_units', 'improve': 'all'},
}
SET_ROLE = {'all': 'primary units', 'learned': 'primary units, learned only', 'learned_plus_tissue': 'sensitivity',
            'no_own_refs': 'sensitivity', 'all_previous_units': 'sensitivity (round-4 units)',
            'learned_previous_units': 'sensitivity (round-4 units)', 'all_plus_floor': 'sensitivity'}
PRIMARY = {'group': 'leaderboard', 'model_set': 'all', 'outcome': 'selB_c', 'method': 'spearman', 'interval': 'full',
           'predictor': 'S', 'vs': 'raw_median'}
CORE = ['raw_median', 'dreval_norm', 'A', 'S', 'sel_mean', 'S_lin']


# ----------------------------------------------------------------------------------------------------------------
# compute: retraining and IMPROVE with one name-keyed drug x cell draw per benchmark
def _shared(states, cells_of, drugs_of, seed):
    cells = sorted(set().union(*[set(cells_of(S)) for S in states]))
    drugs = sorted(set().union(*[set(drugs_of(S)) for S in states]))
    W, V = lib.draws(len(cells), len(drugs), seed)
    ci, di = {c: i for i, c in enumerate(cells)}, {d: i for i, d in enumerate(drugs)}
    for S in states:
        S['W'] = W[:, [ci[c] for c in cells_of(S)]]
        S['V'] = V[:, [di[d] for d in drugs_of(S)]]
    return len(cells), len(drugs)


def compute(parts, t0):
    CACHE.mkdir(parents=True, exist_ok=True)
    tp.CACHE = CACHE
    for f in OLD_CACHE.glob('leaderboard__*.npz'):
        dst = CACHE / f.name
        if not dst.exists():
            dst.symlink_to(f)
    meta = {}
    if 'retrain' in parts:
        tp.G['retrain'] = {s: tp.rt_prepare(s, 20261100 + i) for i, s in enumerate(tp.RT_SETTINGS)}
        nc, nd = _shared(list(tp.G['retrain'].values()), lambda S: S['cells_master'], lambda S: S['drugs_master'],
                         SEED_DRAWS['retrain'])
        meta['retrain'] = {'cells_union': nc, 'drugs_union': nd, 'seed': SEED_DRAWS['retrain']}
        print('retrain set-up (shared draws)', nc, 'cells', nd, 'drugs', f'{time.time() - t0:.0f}s', flush=True)
        tp.run_pool(tp.rt_within_task, tp.RT_SETTINGS, 'retrain', t0)
        tp.run_pool(tp.rt_pair_task, [(s, T) for s in tp.RT_SETTINGS for T in tp.G['retrain'][s]['targets']], 'retrain', t0)
        tp.G.pop('retrain')
    if 'improve' in parts:
        I = lib.improve_setup(replicate_free=True)
        I['src'] = {src: tp.im_prepare(I, src, 20261110 + i) for i, src in enumerate(I['datasets'])}

        def order(d):
            return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]
        nc, nd = _shared(list(I['src'].values()), lambda S: order(S['cix']), lambda S: order(S['dix']), SEED_DRAWS['improve'])
        meta['improve'] = {'cells_union': nc, 'drugs_union': nd, 'seed': SEED_DRAWS['improve']}
        tp.G['improve'] = I
        print('IMPROVE set-up (shared draws)', nc, 'cells', nd, 'drugs', f'{time.time() - t0:.0f}s', flush=True)
        tp.run_pool(tp.im_within_task, I['datasets'], 'improve', t0)
        tasks = [(s, t, v) for s in I['datasets'] for t in I['datasets'] if t != s for v in ['own_panel', 'shared_panel']]
        tp.run_pool(tp.im_pair_task, tasks, 'improve', t0)
        tp.G.pop('improve')
    lib.write_json(OUT / 'compute_meta.json', meta)


# ----------------------------------------------------------------------------------------------------------------
# unit sets and model-cluster draws
def model_sets(rec):
    b = rec['benchmark']
    if b == 'leaderboard':
        s = tp.model_sets(rec)
        return {'all': s['all'], 'learned': s['learned']}
    if b == 'retrain':
        cur = [m for m in rec['models_w'] if m in rec['models_p'] and m not in tp.RT_EXCLUDE and not m.startswith('LB:')
               and not m.startswith('ref')]
        dl = [m for m in DREVAL_LEARNED if m in cur]
        nv = [m for m in DREVAL_NAIVE if m in cur]
        out = {'all': dl + nv, 'learned': dl,
               'learned_plus_tissue': dl + [m for m in ['NaiveTissueDrugMeanPredictor'] if m in cur],
               'no_own_refs': [m for m in cur if m not in OWN_REFS], 'all_previous_units': cur,
               'learned_previous_units': [m for m in cur if m not in tp.RT_AGNOSTIC]}
        return out
    return {'all': list(IMPROVE7), 'all_plus_floor': IMPROVE7 + ['floor_cell_mean']}


def origin(b, m):
    if b == 'leaderboard':
        return 'DrEval released prediction (published model)'
    if b == 'improve':
        return 'IMPROVE drug-agnostic cell-mean floor (this study)' if m == 'floor_cell_mean' else 'IMPROVE released prediction'
    if m in DREVAL_LEARNED:
        return 'DrEval model retrained with drevalpy'
    if m in DREVAL_NAIVE:
        return 'drevalpy naive baseline'
    if m in OWN_REFS:
        return "this study's reference model"
    if m in OWN_SELECTIVE_TARGET:
        return "DrEval architecture trained by this study on a selective/z target (this study's construct)"
    return 'other'


def agnostic(b, m):
    return m in (tp.LB_AGNOSTIC if b == 'leaderboard' else tp.RT_AGNOSTIC if b == 'retrain' else ['floor_cell_mean'])


def cluster_draws(members, key, R):
    """Model multiplicities (R, len(members)): multinomial over this set's members; row 0 = ones."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:7], 16)
    rng = np.random.default_rng(SEED_MODELS + h)
    n = len(members)
    return np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=R - 1)]).astype(float)


# ----------------------------------------------------------------------------------------------------------------
# analysis
def analyse():
    tp.CACHE = CACHE
    recs = tp.build_units()
    R = recs[0]['W']['A'].shape[0]
    preds = tp.PREDICTORS + tp.MATCHED_PREDICTORS
    store, dec_store, rows, unit_rows, conc_rows, setkeys = {}, {}, [], [], [], {}
    clusters = {}
    for rec in recs:
        for sname, ms in model_sets(rec).items():
            ms = [m for m in ms if m in rec['models_w'] and m in rec['models_p']]
            if len(ms) < 4:
                continue
            # one model set per benchmark x set name x setting membership (shared by every target of the set)
            key = f"{rec['benchmark']}|{sname}|{';'.join(sorted(ms))}"
            if key not in clusters:
                clusters[key] = (sorted(ms), cluster_draws(sorted(ms), key, R))
            names, U = clusters[key]
            setkeys[(rec['benchmark'], rec['setting'], sname)] = key
            Wm = U[:, [names.index(m) for m in ms]]
            ones = np.ones_like(Wm)
            X = tp.matrices(rec, ms)
            for i, m in enumerate(ms):
                unit_rows.append({'benchmark': rec['benchmark'], 'setting': rec['setting'], 'pair': rec['pair'],
                                  'model_set': sname, 'model': m, 'primary_target': rec['primary'],
                                  **{k: float(X[k][0, i]) for k in preds + tp.OUTCOMES}})
            for out_k in ['selB_c', 'raw_median_c']:
                Y = X[out_k]
                for p in preds:
                    if not np.isfinite(X[p][0]).sum() >= 4:
                        continue
                    for meth, fn in [('spearman', tp.weighted_spearman), ('pearson', tp.weighted_pearson)]:
                        r_full, r_cond = fn(X[p], Y, Wm), fn(X[p], Y, ones)
                        store[(rec['benchmark'], sname, rec['pair'], p, out_k, meth, 'full')] = r_full
                        store[(rec['benchmark'], sname, rec['pair'], p, out_k, meth, 'cond')] = r_cond
                        bl, bh = tp.bc_interval(r_full)
                        rows.append({'benchmark': rec['benchmark'], 'setting': rec['setting'], 'pair': rec['pair'],
                                     'source': rec['source'], 'target': rec['target'], 'variant': rec['variant'],
                                     'primary_target': rec['primary'], 'model_set': sname, 'models': len(ms),
                                     'drugs_used': rec['drugs_used'], 'predictor': p, 'outcome': out_k, 'method': meth,
                                     'r': float(r_cond[0]), 'lo_full': lib.ci(r_full)[0], 'hi_full': lib.ci(r_full)[1],
                                     'bc_lo_full': bl, 'bc_hi_full': bh, 'lo_cond': lib.ci(r_cond)[0],
                                     'hi_cond': lib.ci(r_cond)[1]})
                    for k in (1, 3, 5):
                        if k < len(ms):
                            dec_store[(rec['benchmark'], sname, rec['pair'], p, out_k, k)] = tp.decision(X[p], Y, k)
    # raw-S rank concordance per setting (within-dataset statistics only; one row per setting x set)
    seen = set()
    for rec in recs:
        for sname, ms in model_sets(rec).items():
            k_ = (rec['benchmark'], rec['setting'], sname)
            if k_ in seen or k_ not in setkeys:
                continue
            seen.add(k_)
            names, U = clusters[setkeys[k_]]
            ms = [m for m in names if m in rec['models_w']]
            if len(ms) < 4:
                continue
            Wm = U[:, [names.index(m) for m in ms]]
            X = {k: np.column_stack([rec['W'][k][:, rec['models_w'].index(m)] for m in ms]) for k in
                 ['raw_median', 'S', 'dreval_norm', 'sel_mean', 'A']}
            for a, b in [('raw_median', 'S'), ('raw_median', 'dreval_norm'), ('S', 'dreval_norm'), ('S', 'sel_mean'),
                         ('raw_median', 'sel_mean'), ('raw_median', 'A')]:
                if np.isfinite(X[a][0]).sum() < 4 or np.isfinite(X[b][0]).sum() < 4:
                    continue
                full = tp.weighted_spearman(X[a], X[b], Wm)
                cond = tp.weighted_spearman(X[a], X[b], np.ones_like(Wm))
                store[('concordance', rec['benchmark'], rec['setting'], sname, f'{a}~{b}')] = full
                bl, bh = tp.bc_interval(full)
                conc_rows.append({'benchmark': rec['benchmark'], 'setting': rec['setting'], 'model_set': sname,
                                  'models': len(ms), 'pair_of_statistics': f'{a} ~ {b}', 'spearman': float(full[0]),
                                  'lo': lib.ci(full)[0], 'hi': lib.ci(full)[1], 'bc_lo': bl, 'bc_hi': bh,
                                  'lo_cond': lib.ci(cond)[0], 'hi_cond': lib.ci(cond)[1]})
    return recs, store, dec_store, pd.DataFrame(rows), pd.DataFrame(unit_rows), pd.DataFrame(conc_rows), clusters


def zmean(arrs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(tp.fz(np.vstack(arrs)), axis=0)


def corr_row(gname, b, sname, out_k, meth, cond, p, n, z):
    r = np.tanh(z)
    lo, hi = lib.ci(r)
    bl, bh = tp.bc_interval(r)
    return {'group': gname, 'benchmark': b, 'model_set': sname, 'outcome': out_k, 'method': meth, 'interval': cond,
            'predictor': p, 'targets': n, 'r_pooled': float(r[0]), 'lo': lo, 'hi': hi, 'bc_lo': bl, 'bc_hi': bh,
            'boot_median': float(np.nanmedian(r[1:])), 'p_gt0': tp.pboot(z)}


def contrast_rows(gname, sname, out_k, meth, cond, Z, ntarg):
    out = []
    for ref in ['raw_median', 'dreval_norm', 'S']:
        for p in Z:
            if p == ref or ref not in Z:
                continue
            d = Z[p] - Z[ref]
            dr = np.tanh(Z[p]) - np.tanh(Z[ref])
            lo, hi = lib.ci(d)
            lo2, hi2 = lib.ci(dr)
            blo, bhi = tp.bc_interval(dr)
            out.append({'group': gname, 'model_set': sname, 'outcome': out_k, 'method': meth, 'interval': cond,
                        'predictor': p, 'vs': ref, 'targets': ntarg, 'dz': float(d[0]), 'dz_lo': lo, 'dz_hi': hi,
                        'dr': float(dr[0]), 'dr_lo': lo2, 'dr_hi': hi2, 'dr_bc_lo': blo, 'dr_bc_hi': bhi,
                        'p': tp.pboot(d), 'share_draws_gt0': float(np.mean(d[1:][np.isfinite(d[1:])] > 0))})
    return out


def pool(store, dec_store, recs):
    groups = tp.groups_of(recs)
    preds = tp.PREDICTORS + tp.MATCHED_PREDICTORS
    rows, crow, drow, bench_z = [], [], [], {}
    snames = sorted({s for r in recs for s in model_sets(r)})
    for gname, (b, pairs) in groups.items():
        for sname in snames:
            for out_k in ['selB_c', 'raw_median_c']:
                for meth in ['spearman', 'pearson']:
                    for cond in ['full', 'cond']:
                        Z = {}
                        for p in preds:
                            arrs = [store[(b, sname, pr, p, out_k, meth, cond)] for pr in pairs
                                    if (b, sname, pr, p, out_k, meth, cond) in store]
                            if not arrs or len(arrs) < len(pairs):
                                continue
                            Z[p] = zmean(arrs)
                            rows.append(corr_row(gname, b, sname, out_k, meth, cond, p, len(arrs), Z[p]))
                            if gname == b:
                                bench_z[(b, sname, out_k, meth, cond, p)] = Z[p]
                        crow += contrast_rows(gname, sname, out_k, meth, cond, Z, len(pairs))
                for p in preds:
                    for k in (1, 3, 5):
                        ds = [dec_store[(b, sname, pr, p, out_k, k)] for pr in pairs if (b, sname, pr, p, out_k, k) in dec_store]
                        if not ds or len(ds) < len(pairs):
                            continue
                        agg = {q: np.nanmean(np.vstack([d[q] for d in ds]), 0) for q in ds[0] if q != 'chosen_set_point'}
                        row = {'group': gname, 'benchmark': b, 'model_set': sname, 'outcome': out_k, 'predictor': p,
                               'k': k, 'targets': len(ds)}
                        for q, v in agg.items():
                            row.update(lib.summ(v, q))
                        for ref in ['raw_median', 'dreval_norm']:
                            rd = [dec_store[(b, sname, pr, ref, out_k, k)] for pr in pairs
                                  if (b, sname, pr, ref, out_k, k) in dec_store]
                            if p != ref and len(rd) == len(ds):
                                dd = np.nanmean(np.vstack([r_['chosen'] for r_ in ds]), 0) - \
                                    np.nanmean(np.vstack([r_['chosen'] for r_ in rd]), 0)
                                row.update(lib.summ(dd, f'chosen_minus_{ref}'))
                                row[f'p_chosen_minus_{ref}'] = tp.pboot(dd)
                                bl, bh = tp.bc_interval(dd)
                                row[f'chosen_minus_{ref}_bc_lo'], row[f'chosen_minus_{ref}_bc_hi'] = bl, bh
                        bl, bh = tp.bc_interval(agg['regret'])
                        row['regret_bc_lo'], row['regret_bc_hi'] = bl, bh
                        row['interval_note'] = 'drug x cell draws; the menu of models is fixed'
                        drow.append(row)
    # overall: equal-weight Fisher-z mean of the three benchmarks under each unit scheme
    for scheme, sets in SCHEMES.items():
        for out_k in ['selB_c', 'raw_median_c']:
            for meth in ['spearman', 'pearson']:
                for cond in ['full', 'cond']:
                    Z = {}
                    for p in preds:
                        zs = [bench_z.get((b, sets[b], out_k, meth, cond, p)) for b in ['leaderboard', 'retrain', 'improve']]
                        if any(z_ is None for z_ in zs):
                            continue
                        Z[p] = np.nanmean(np.vstack(zs), 0)
                        rows.append(corr_row('overall', 'overall', scheme, out_k, meth, cond, p, 3, Z[p]))
                    crow += contrast_rows('overall', scheme, out_k, meth, cond, Z, 3)
                    Z2 = {}
                    for p in ['raw_median', 'dreval_norm', 'S', 'sel_mean', 'S_lin', 'A']:
                        zs = [bench_z.get((b, sets[b], out_k, meth, cond, p)) for b in ['leaderboard', 'retrain']]
                        if any(z_ is None for z_ in zs):
                            continue
                        Z2[p] = np.nanmean(np.vstack(zs), 0)
                        rows.append(corr_row('overall (DrEval benchmarks)', 'overall', scheme, out_k, meth, cond, p, 2, Z2[p]))
                    crow += contrast_rows('overall (DrEval benchmarks)', scheme, out_k, meth, cond, Z2, 2)
    return pd.DataFrame(rows), pd.DataFrame(crow), pd.DataFrame(drow)


def concordance_pooled(store, conc):
    """Raw-S concordance pooled over a benchmark's settings (Fisher-z mean; secondary)."""
    rows = []
    for (b, sname, pr), g in conc.groupby(['benchmark', 'model_set', 'pair_of_statistics']):
        arrs = [store[('concordance', b, s, sname, pr.replace(' ', ''))] for s in g.setting
                if ('concordance', b, s, sname, pr.replace(' ', '')) in store]
        if b == 'retrain':
            arrs = [store[('concordance', b, s, sname, pr.replace(' ', ''))] for s in g.setting
                    if s in tp.RT_POOLED and ('concordance', b, s, sname, pr.replace(' ', '')) in store]
        if len(arrs) < 2:
            continue
        r = np.tanh(zmean(arrs))
        bl, bh = tp.bc_interval(r)
        rows.append({'benchmark': b, 'setting': f'pooled ({len(arrs)} settings)', 'model_set': sname,
                     'pair_of_statistics': pr, 'spearman': float(r[0]), 'lo': lib.ci(r)[0], 'hi': lib.ci(r)[1],
                     'bc_lo': bl, 'bc_hi': bh})
    return pd.DataFrame(rows)


def role(df, kind):
    """'primary' for the one pre-stated primary comparison, else 'secondary'."""
    m = (df.group == PRIMARY['group']) & (df.model_set == PRIMARY['model_set']) & (df.outcome == PRIMARY['outcome'])
    m &= (df.method == PRIMARY['method']) & (df.interval == PRIMARY['interval'])
    if kind == 'contrast':
        m &= (df.predictor == PRIMARY['predictor']) & (df.vs == PRIMARY['vs'])
    else:
        m &= df.predictor.isin([PRIMARY['predictor'], PRIMARY['vs']])
    return np.where(m, 'primary' if kind == 'contrast' else 'primary (components)', 'secondary')


def units_table(recs, clusters):
    rows = []
    for rec in recs:
        for sname, ms in model_sets(rec).items():
            ms_ok = [m for m in ms if m in rec['models_w'] and m in rec['models_p']]
            for m in ms:
                rows.append({'benchmark': rec['benchmark'], 'setting': rec['setting'], 'model_set': sname,
                             'set_role': SET_ROLE.get(sname, ''), 'model': m, 'origin': origin(rec['benchmark'], m),
                             'drug_agnostic': agnostic(rec['benchmark'], m), 'pair': rec['pair'],
                             'in_analysis': m in ms_ok and len(ms_ok) >= 4,
                             **({k: float(rec['W'][k][0, rec['models_w'].index(m)]) for k in CORE}
                                if m in rec['models_w'] else {})})
    U = pd.DataFrame(rows)
    agg = U.groupby(['benchmark', 'setting', 'model_set', 'set_role', 'model', 'origin', 'drug_agnostic'], dropna=False).agg(
        targets=('pair', 'nunique'), targets_in_analysis=('in_analysis', 'sum'),
        **{f'within_{k}': (k, 'first') for k in CORE}).reset_index()
    return agg


def verify_points(recs):
    """Draw 0 of the recomputed retraining/IMPROVE caches against 100a's cache (must be identical)."""
    rows = []
    for f in sorted(CACHE.glob('*.npz')):
        if f.is_symlink():
            continue
        g = OLD_CACHE / f.name
        if not g.exists():
            rows.append({'file': f.name, 'max_absdiff_draw0': np.nan, 'note': 'not in 100a cache'})
            continue
        a, b = np.load(f), np.load(g)
        ma, mb = list(a['models']), list(b['models'])
        mx = 0.0
        for k in a.files:
            if k in ('models', 'meta') or k not in b.files:
                continue
            x, y = a[k], b[k]
            if x.ndim == 2 and x.shape[1] == len(ma):
                for i, m in enumerate(ma):
                    if m in mb:
                        u, v = x[0, i], y[0, mb.index(m)]
                        if np.isfinite(u) or np.isfinite(v):
                            mx = max(mx, float(abs(u - v)) if np.isfinite(u) and np.isfinite(v) else np.inf)
        rows.append({'file': f.name, 'max_absdiff_draw0': mx, 'note': ''})
    return pd.DataFrame(rows)


def key_numbers(Cp, Cc, D, Cn, Cnp):
    kn = {}
    sp = Cp[(Cp.method == 'spearman') & (Cp.interval == 'full')]
    for r in sp.itertuples():
        if r.predictor in CORE and r.group in ('leaderboard', 'retrain', 'improve', 'overall', 'overall (DrEval benchmarks)',
                                               'retrain:GDSC2_AUC', 'retrain:GDSC2_LNIC50'):
            kn[f'corr|{r.group}|{r.model_set}|{r.outcome}|{r.predictor}'] = {
                'r': r.r_pooled, 'lo': r.lo, 'hi': r.hi, 'bc_lo': r.bc_lo, 'bc_hi': r.bc_hi, 'targets': int(r.targets)}
    sc = Cc[(Cc.method == 'spearman') & (Cc.interval == 'full')]
    for r in sc.itertuples():
        if r.predictor in CORE and r.group in ('leaderboard', 'retrain', 'improve', 'overall', 'overall (DrEval benchmarks)',
                                               'retrain:GDSC2_AUC', 'retrain:GDSC2_LNIC50'):
            kn[f'contrast|{r.group}|{r.model_set}|{r.outcome}|{r.predictor}-{r.vs}'] = {
                'dr': r.dr, 'dr_lo': r.dr_lo, 'dr_hi': r.dr_hi, 'dr_bc_lo': r.dr_bc_lo, 'dr_bc_hi': r.dr_bc_hi, 'p': r.p}
    for _, r in D[D.predictor.isin(['raw_median', 'S', 'dreval_norm', 'sel_mean'])].iterrows():
        kn[f"decision|{r.group}|{r.model_set}|{r.outcome}|{r.predictor}|k{r.k}"] = {
            k: (None if pd.isna(r.get(k)) else float(r.get(k))) for k in
            ['chosen', 'chosen_lo', 'chosen_hi', 'regret', 'regret_lo', 'regret_hi', 'regret_bc_lo', 'regret_bc_hi',
             'chosen_minus_raw_median', 'chosen_minus_raw_median_lo', 'chosen_minus_raw_median_hi',
             'chosen_minus_raw_median_bc_lo', 'chosen_minus_raw_median_bc_hi', 'p_chosen_minus_raw_median',
             'chosen_minus_dreval_norm', 'chosen_minus_dreval_norm_lo', 'chosen_minus_dreval_norm_hi',
             'p_chosen_minus_dreval_norm'] if k in r.index}
    for r in Cn.itertuples():
        kn[f'concordance|{r.benchmark}|{r.setting}|{r.model_set}|{r.pair_of_statistics}'] = {
            'spearman': r.spearman, 'lo': r.lo, 'hi': r.hi, 'bc_lo': r.bc_lo, 'bc_hi': r.bc_hi, 'models': int(r.models)}
    for r in Cnp.itertuples():
        kn[f'concordance|{r.benchmark}|{r.setting}|{r.model_set}|{r.pair_of_statistics}'] = {
            'spearman': r.spearman, 'lo': r.lo, 'hi': r.hi, 'bc_lo': r.bc_lo, 'bc_hi': r.bc_hi}
    return kn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compute', action='store_true')
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--parts', nargs='*', default=['retrain', 'improve'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.compute:
        compute(a.parts, t0)
    if not a.analyse:
        print('done', f'{time.time() - t0:.0f}s')
        return
    recs, store, dec_store, Ct, Uu, Cn, clusters = analyse()
    print('analysed', len(recs), 'records', f'{time.time() - t0:.0f}s', flush=True)
    Cp, Cc, D = pool(store, dec_store, recs)
    Cnp = concordance_pooled(store, Cn)
    Cp['role'] = role(Cp, 'corr')
    Cc['role'] = role(Cc, 'contrast')
    Cp['set_role'] = Cp.model_set.map(lambda s: SET_ROLE.get(s, 'scheme: ' + s))
    Cc['set_role'] = Cc.model_set.map(lambda s: SET_ROLE.get(s, 'scheme: ' + s))
    D['set_role'] = D.model_set.map(lambda s: SET_ROLE.get(s, ''))
    Ct.to_csv(OUT / 'correlations_by_target.csv', index=False, float_format='%.5g')
    Uu.to_csv(OUT / 'units.csv', index=False, float_format='%.5g')
    Cp.to_csv(OUT / 'correlations_pooled.csv', index=False, float_format='%.5g')
    Cc.to_csv(OUT / 'contrast_pooled.csv', index=False, float_format='%.5g')
    D.to_csv(OUT / 'decision_pooled.csv', index=False, float_format='%.5g')
    pd.concat([Cn, Cnp], ignore_index=True).to_csv(OUT / 'concordance.csv', index=False, float_format='%.5g')
    units_table(recs, clusters).to_csv(OUT / 'units_by_setting.csv', index=False, float_format='%.5g')
    # per-target S - raw contrasts (as 100a, primary units and the rest)
    rows = []
    for r in recs:
        for sname in model_sets(r):
            for out_k in ['selB_c']:
                for cond in ['full', 'cond']:
                    a_ = store.get((r['benchmark'], sname, r['pair'], 'S', out_k, 'spearman', cond))
                    b_ = store.get((r['benchmark'], sname, r['pair'], 'raw_median', out_k, 'spearman', cond))
                    if a_ is None or b_ is None:
                        continue
                    d = tp.fz(a_) - tp.fz(b_)
                    dr = a_ - b_
                    bl, bh = tp.bc_interval(dr)
                    rows.append({'benchmark': r['benchmark'], 'pair': r['pair'], 'primary_target': r['primary'],
                                 'model_set': sname, 'outcome': out_k, 'method': 'spearman', 'interval': cond,
                                 'r_S': float(a_[0]), 'r_raw': float(b_[0]), 'dr': float(dr[0]), 'dr_lo': lib.ci(dr)[0],
                                 'dr_hi': lib.ci(dr)[1], 'dr_bc_lo': bl, 'dr_bc_hi': bh, 'p': tp.pboot(d)})
    pd.DataFrame(rows).to_csv(OUT / 'contrast_by_target.csv', index=False, float_format='%.5g')
    # figure-ready (100a's format; model_set 'all' = primary units, 'learned' = learned only; extra columns appended)
    keep = ['leaderboard', 'retrain', 'improve', 'overall', 'retrain:GDSC2_AUC', 'retrain:GDSC2_LNIC50']
    fb = Cp[(Cp.method == 'spearman') & (Cp.interval == 'full') & Cp.predictor.isin(tp.PREDICTORS) & Cp.group.isin(keep)]
    fb.to_csv(OUT / 'fig_b_pooled_correlations.csv', index=False, float_format='%.5g')
    fd = D[D.predictor.isin(['raw_median', 'dreval_norm', 'S', 'sel_mean', 'S_lin', 'A']) & D.group.isin(keep)]
    old_cols = list(pd.read_csv(lib.RES / 'revision_round4' / 'transfer_prediction' / 'fig_d_decision_regret.csv', nrows=1).columns)
    fd[[c for c in old_cols if c in fd.columns] + [c for c in fd.columns if c not in old_cols]].to_csv(
        OUT / 'fig_d_decision_regret.csv', index=False, float_format='%.5g')
    # primary comparison
    pc = Cc[(Cc.role == 'primary')].iloc[0]
    comp = Cp[(Cp.role == 'primary (components)')]
    prim = {'definition': 'leaderboard; 21 published models; targets CTRP v1, GDSC1, GDSC2 and CCLE CurveCurator ln IC50 '
                          '(endpoint-matched to the CTRP v2 CurveCurator ln IC50 training data); outcome: mean '
                          'selective-both in the target; statistic: Spearman across models, Fisher-z mean over the four '
                          'targets (equal weights, a choice); contrast: selective capture S minus the raw median per-drug '
                          'correlation; joint bootstrap over models (per set), drugs and cell lines (B = 1,000)',
            'pre_stated_in': '100a_transfer_prediction.py header, rule (a): S against the raw score, all-model sets primary',
            'S_minus_raw_dr': pc.dr, 'percentile_95': [pc.dr_lo, pc.dr_hi], 'bias_corrected_95': [pc.dr_bc_lo, pc.dr_bc_hi],
            'fisher_z_difference': pc.dz, 'fisher_z_percentile_95': [pc.dz_lo, pc.dz_hi], 'bootstrap_p_two_sided': pc.p,
            'components': {r.predictor: {'r': r.r_pooled, 'percentile_95': [r.lo, r.hi], 'bias_corrected_95': [r.bc_lo, r.bc_hi]}
                           for r in comp.itertuples()},
            'secondary': 'every other row (other benchmarks, unit sets, statistics, outcomes, Pearson, models-fixed '
                         'intervals, per-target values, concordance and the decision analysis) is secondary',
            'disclosure_post_hoc': ("The decision analysis's regret was redefined after 100a's first run (the redrawn "
                                    'oracle made regret >= 0 in every draw); the primary regret fixes the benchmark set at '
                                    'the k best by the full-sample outcome. The chosen-set contrasts between statistics '
                                    'are identical under both definitions.')}
    lib.write_json(OUT / 'primary_comparison.json', prim)
    kn = key_numbers(Cp, Cc, D, Cn, Cnp)
    kn = {'primary_comparison': prim, **kn}
    lib.write_json(OUT / 'key_numbers_r4.json', kn)
    V = verify_points(recs)
    V.to_csv(OUT / 'verification_point_vs_100a.csv', index=False, float_format='%.3g')
    lib.write_json(OUT / 'run.json', {
        'seconds_analyse': round(time.time() - t0), 'B': lib.B, 'records': len(recs),
        'model_sets': {k: v[0] for k, v in clusters.items()}, 'schemes_overall': SCHEMES, 'seed_models': SEED_MODELS,
        'seed_draws': SEED_DRAWS, 'leaderboard_draws': "42's two-way draws (100a cache, symlinked)",
        'point_estimates_max_absdiff_vs_100a': float(np.nanmax(V.max_absdiff_draw0)) if len(V) else None})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 400)
    sel = Cp[(Cp.method == 'spearman') & (Cp.interval == 'full') & (Cp.outcome == 'selB_c') & Cp.group.isin(keep) &
             Cp.predictor.isin(CORE) & Cp.model_set.isin(['all', 'learned', 'all_previous_units', 'learned_previous_units'])]
    print(sel[['group', 'model_set', 'predictor', 'targets', 'r_pooled', 'lo', 'hi', 'bc_lo', 'bc_hi']].round(3).to_string(index=False))
    cs = Cc[(Cc.method == 'spearman') & (Cc.interval == 'full') & (Cc.outcome == 'selB_c') & Cc.group.isin(keep) &
            Cc.model_set.isin(['all', 'learned', 'all_previous_units', 'learned_previous_units']) &
            Cc.predictor.isin(['S', 'sel_mean', 'dreval_norm']) & Cc.vs.isin(['raw_median', 'dreval_norm'])]
    print(cs[['group', 'model_set', 'predictor', 'vs', 'dr', 'dr_lo', 'dr_hi', 'dr_bc_lo', 'dr_bc_hi', 'p', 'role']].round(3).to_string(index=False))
    print(Cn[Cn.pair_of_statistics == 'raw_median ~ S'].round(3).to_string(index=False))
    print(Cnp[Cnp.pair_of_statistics == 'raw_median ~ S'].round(3).to_string(index=False))
    dd = D[(D.outcome == 'selB_c') & D.predictor.isin(['S', 'dreval_norm', 'sel_mean']) & D.group.isin(keep) &
           D.model_set.isin(['all', 'learned'])]
    print(dd[['group', 'model_set', 'predictor', 'k', 'chosen', 'chosen_minus_raw_median', 'chosen_minus_raw_median_lo',
              'chosen_minus_raw_median_hi', 'chosen_minus_raw_median_bc_lo', 'chosen_minus_raw_median_bc_hi',
              'p_chosen_minus_raw_median']].round(4).to_string(index=False))
    print('verification', V.max_absdiff_draw0.max() if len(V) else None)
    print('done', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
