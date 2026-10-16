"""Round 2: paired cross-dataset scores with a two-way bootstrap, for all three benchmarks (feeds scripts 81-85).

For each paired target, identical out-of-fold predictions are scored against the source/training dataset S
("within", w) and the target T ("cross", c) on the same drugs and, per drug, the same cell lines. Per model, drug and
bootstrap draw (77_round2_lib.paired_stats): raw r, r_pg, r_zg, mediated and selective terms, selective-T (own
axis), and selective-both given (g_S, g_T) on BOTH sides (symmetric; the within value r(p, z_S | g_S, g_T) is new,
R3-M3d). Sums over drugs (V-weighted) are kept per model and draw (77_round2_lib.PairedAccumulator), together with
the source measurement's (one replicate's) cross selective-both and raw r on each model's drugs.

Benchmarks and pairs
  leaderboard  DrEval's CTRPv2-trained leaderboard predictions (42's cells, models and two-way draws, seed 20261014,
               so pooled sums reproduce results/revision_crosslab/bootstrap_pooled.parquet): ln IC50-reference pairs
               against every version (own matched cells), AUC-reference pairs (CTRPv2 CurveCurator AUC -> each
               dataset's CurveCurator AUC), the 16 harmonized versions of 66 (shared panel, common range, ln IC50
               shared panel / censored to the common range), and drug-matched variants for R2-Major 1 (each ln IC50
               target restricted to the drugs of its range-censored version).
  leaderboard_matched  (round 3; editor item 2, R1-M2, R3-M3) endpoint-matched pairings for the AUC-type targets
               that the ln IC50 reference was the only pairing of: CTRPv2 CurveCurator AUC -> every provider-AUC target
               (CTRPv2 and CTRPv1 CTRP AUC, GDSC1 and GDSC2 Sanger AUC, CCLE ActArea), and CTRPv2's provider AUC
               (CTRP area_under_curve) -> every AUC-type target (the processing-matched pairing for CTRPv1's provider
               AUC). Same cells, models and two-way draws as 'leaderboard' (42's seed), so every pair is paired with
               the existing ones; k = 100 + i (the 46 existing leaderboard arrays are not touched).
  retrain      drevalpy retraining (59): GDSC2 CurveCurator AUC, GDSC2 CurveCurator ln IC50 and CTRPv2 CurveCurator
               ln IC50, main panels, against every other DrEval dataset with the same endpoint (59's cells: per drug,
               cells with every panel model's prediction); draws per target, seed 20261015 + k. Round 3: CTRPv2
               CurveCurator AUC (the fourth cell of the dataset x endpoint design) is appended when its drevalpy run
               is complete, so the earlier settings keep their k and seeds.
  improve      IMPROVE (68), replicate-free (pairs whose replicate rows fell in different test splits removed), every
               source -> target, axes from each dataset's own panel ('own_panel', primary) or from the shared drugs
               ('shared_panel'); each model on its own predicted pairs (68); draws per pair, seed 20261016 + k.
Outputs (results/revision_round2/paired/): arrays/<benchmark>__<k>.npz, per_drug_<benchmark>.csv.gz (point values),
pairs.csv (index), run.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.out_dir('paired')
ARR = OUT / 'arrays'
ARR.mkdir(exist_ok=True)
WORKERS = 3
G = {}
DREVAL_LEARNED = ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting']
MATCHED_K0 = 100
FAM_CC_AUC_PROVIDER = 'AUC-reference (provider-AUC targets)'
FAM_PROVIDER_AUC = 'CTRP provider AUC reference'


def matched_tasks():
    """Round 3: endpoint-matched pairings of the AUC-type targets (k = MATCHED_K0 + i)."""
    tasks = []
    for tgt in ['CTRPv2|CTRP|AUC', 'CTRPv1|CTRP|AUC', 'GDSC1|Sanger|AUC', 'GDSC2|Sanger|AUC', 'CCLE|CCLE|ActArea']:
        tasks.append({'pair': f'CTRPv2|CC|AUC->{tgt}', 'reference': 'CTRPv2|CC|AUC', 'target': tgt,
                      'family': FAM_CC_AUC_PROVIDER})
    for tgt in ['CTRPv2|CC|AUC', 'CTRPv1|CTRP|AUC', 'CTRPv1|CC|AUC', 'GDSC1|Sanger|AUC', 'GDSC1|CC|AUC',
                'GDSC2|Sanger|AUC', 'GDSC2|CC|AUC', 'CCLE|CCLE|ActArea', 'CCLE|CC|AUC']:
        tasks.append({'pair': f'CTRPv2|CTRP|AUC->{tgt}', 'reference': 'CTRPv2|CTRP|AUC', 'target': tgt,
                      'family': FAM_PROVIDER_AUC})
    for i, t in enumerate(tasks):
        t['k'] = MATCHED_K0 + i
    return tasks


# ----------------------------------------------------------------------------------------------------------------
def lb_task(t):
    L = G['L']
    acc, n = lib.leaderboard_pair_subset(L, t['reference'], t['target'], t.get('drugs'))
    meta = {**t, 'drugs_used': n, 'drugs': None}
    acc.save(ARR / f"leaderboard__{t['k']}.npz", meta)
    rows = pd.DataFrame(acc.rows).assign(pair=t['pair'])
    return t['k'], n, rows


def retrain_task(t):
    S = G['retrain'][t['setting']]
    rt = S['rt']
    T = t['target']
    obs_t = S['resp'][(S['resp'].dataset == T) & (S['resp'].endpoint == S['endpoint'])][['drug', 'cell', 'response']]
    Zt, Gt, _, _ = lib.selbench.decompose(obs_t)
    Zs, Gs = S['Zs'], S['Gs']
    models = S['models']
    pdrugs = sorted(set(Zs.index) & set(Zt.index) & set().union(*[set(S['P'][m].index) for m in models]))
    pcells = sorted(set(Zs.columns) & set(Zt.columns) & set(S['cells']))
    if len(pdrugs) < 5 or len(pcells) < lib.MIN_CELLS:
        return t['k'], 0, None
    Pm = {m: S['P'][m].reindex(index=[d for d in pdrugs if d in S['P'][m].index], columns=pcells) for m in models}
    REF = 'ref:source_measurement'
    Pm[REF] = Zs.reindex(index=pdrugs, columns=pcells)
    allm = models + [REF]
    W, V = lib.draws(len(pcells), len(pdrugs), t['seed'])
    acc = lib.PairedAccumulator(allm, W.shape[0], ref=REF)
    mats = [M.reindex(index=pdrugs, columns=pcells).values.astype(float) for M in (Zs, Gs, Zt, Gt)]
    Pv = {m: Pm[m].reindex(index=pdrugs, columns=pcells).values.astype(float) for m in allm}
    n = 0
    for j, d in enumerate(pdrugs):
        zs, gs, zt, gt = (M[j] for M in mats)
        have = [m for m in allm if d in Pm[m].index]
        ok = np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        for m in have:
            ok &= np.isfinite(Pv[m][j])
        if ok.sum() < lib.MIN_CELLS:
            continue
        idx = np.where(ok)[0]
        use = [m for m in have if np.std(Pv[m][j][idx]) >= 1e-12]
        st = lib.paired_stats(zs[idx], gs[idx], zt[idx], gt[idx], np.column_stack([Pv[m][j][idx] for m in use]),
                              W[:, idx])
        acc.add(d, len(idx), st, use, V[:, j])
        n += 1
    meta = {**t, 'drugs_used': n, 'cells': len(pcells), 'models': allm}
    acc.save(ARR / f"retrain__{t['k']}.npz", meta)
    rows = pd.DataFrame(acc.rows).assign(pair=t['pair'])
    return t['k'], n, rows


def improve_task(t):
    I = G['improve']
    src, trg, var = t['source'], t['target'], t['variant']
    if var == 'own_panel':
        (Zs, Gs), (Zt, Gt) = I['dec'][src], I['dec'][trg]
    else:
        im = I['im']
        shared = I['dec'][src][0].index.intersection(I['dec'][trg][0].index)
        Zs, Gs = im.decompose(I['obs'][src], panel=shared)
        Zt, Gt = im.decompose(I['obs'][trg], panel=shared)
    Pm = {m: I['P_cross'][(m, src, trg)] for m in I['models'] if (m, src, trg) in I['P_cross']}
    if not Pm:
        return t['k'], 0, None
    REF = 'source_measurement'
    Pref = lib.improve_source_measurement(I, src, trg, next(iter(Pm.values())))
    allm = list(Pm) + [REF]
    Pm[REF] = Pref
    cells = Zs.columns.intersection(Zt.columns)
    drugs = Zs.index.intersection(Zt.index).intersection(pd.Index(sorted(set().union(*[set(P.index) for m, P in Pm.items()
                                                                                         if m != REF]))))
    W, V = lib.draws(len(cells), len(drugs), t['seed'])
    acc = lib.PairedAccumulator(allm, W.shape[0], ref=REF)
    mats = [M.reindex(index=drugs, columns=cells).values.astype(float) for M in (Zs, Gs, Zt, Gt)]
    Pv = {m: Pm[m].reindex(index=drugs, columns=cells).values.astype(float) for m in allm}
    n = 0
    for j, d in enumerate(drugs):
        zs, gs, zt, gt = (M[j] for M in mats)
        base = np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        groups = {}
        for m in allm:
            mk = base & np.isfinite(Pv[m][j])
            if mk.sum() < lib.MIN_CELLS:
                continue
            groups.setdefault(mk.tobytes(), (mk, []))[1].append(m)
        used = False
        for mk, ms in groups.values():
            idx = np.where(mk)[0]
            cols, use = [], []
            for m in ms:
                p = Pv[m][j][idx]
                if m == REF:
                    p = lib.selbench.rank_normal(p)       # 68: prediction_scale='rank' for the measurement
                if np.std(p) < 1e-12:
                    continue
                cols.append(p)
                use.append(m)
            if not use:
                continue
            st = lib.paired_stats(zs[idx], gs[idx], zt[idx], gt[idx], np.column_stack(cols), W[:, idx])
            acc.add(d, len(idx), st, use, V[:, j])
            used = True
        n += used
    # the reference is matched to each model on that model's own drugs; with per-model masks the reference's cells
    # can differ slightly from the model's (both are restricted to pairs measured in both datasets)
    meta = {**t, 'drugs_used': n, 'cells': len(cells), 'models': allm}
    acc.save(ARR / f"improve__{t['k']}.npz", meta)
    rows = pd.DataFrame(acc.rows).assign(pair=t['pair'])
    return t['k'], n, rows


def leaderboard_pair_subset(L, ref, tgt, drugs=None):
    """leaderboard_pair restricted to a drug subset (axes unchanged)."""
    if drugs is None:
        return lib.leaderboard_pair(L, ref, tgt)
    keep = set(drugs)
    L2 = dict(L)
    L2['drug_ix'] = {d: i for d, i in L['drug_ix'].items() if d in keep}
    return lib.leaderboard_pair(L2, ref, tgt)


lib.leaderboard_pair_subset = leaderboard_pair_subset


def run_pool(fn, tasks, tag, t0):
    rows, idx = [], []
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context('fork')) as ex:
        for k, n, r in ex.map(fn, tasks):
            idx.append({'k': k, 'drugs_used': n})
            if r is not None:
                rows.append(r)
            print(tag, k, n, f'{time.time() - t0:.0f}s', flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='*', default=['leaderboard', 'retrain', 'improve'])
    ap.add_argument('--allow-incomplete-ctrpv2auc', action='store_true',
                    help='score CTRPv2_AUC even if not all four learned models are complete (provisional)')
    args = ap.parse_args()
    t0 = time.time()
    index = []
    # ------------------------------------------------------------------ leaderboard
    if 'leaderboard' in args.parts:
        vt = pd.read_csv(lib.RES / 'revision_crosslab' / 'versions.csv')
        TG = ['CTRPv1', 'GDSC1', 'GDSC2', 'CCLE']
        harm = [f'CTRPv2|{f}-{T}|AUC' for T in TG for f in ('HR', 'SP')] + [f'{T}|{f}|AUC' for T in TG for f in ('HR', 'SP')]
        harm += [f'CTRPv2|{f}-{T}|lnIC50' for T in TG for f in ('SL', 'RL')] + [f'{T}|{f}|lnIC50' for T in TG for f in ('SL', 'RL')]
        L = lib.leaderboard_setup(need_versions=list(vt.version) + harm, harmonized=True)
        G['L'] = L
        print('leaderboard set-up', f'{time.time() - t0:.0f}s', flush=True)
        cd = L['cd']
        tasks = []
        for v in vt.version:
            tasks.append({'pair': f'{cd.REFERENCE}->{v}', 'reference': cd.REFERENCE, 'target': v,
                          'family': 'lnIC50-reference'})
        for ds in cd.DATASETS:
            tasks.append({'pair': f'{cd.AUC_REFERENCE}->{ds}|CC|AUC', 'reference': cd.AUC_REFERENCE,
                          'target': f'{ds}|CC|AUC', 'family': 'AUC-reference'})
        fam = {'HR': 'shared panel, common range AUC', 'SP': 'shared panel, own range AUC',
               'SL': 'shared panel ln IC50', 'RL': 'shared panel + range ln IC50'}
        for T in TG:
            for f in ('HR', 'SP'):
                tasks.append({'pair': f'CTRPv2|{f}-{T}|AUC->{T}|{f}|AUC', 'reference': f'CTRPv2|{f}-{T}|AUC',
                              'target': f'{T}|{f}|AUC', 'family': fam[f]})
            for f in ('SL', 'RL'):
                tasks.append({'pair': f'CTRPv2|{f}-{T}|lnIC50->{T}|{f}|lnIC50', 'reference': f'CTRPv2|{f}-{T}|lnIC50',
                              'target': f'{T}|{f}|lnIC50', 'family': fam[f]})
        for k, t in enumerate(tasks):
            t['k'] = k
        rows, idx = run_pool(lb_task, tasks, 'leaderboard', t0)
        # R2-Major 1 (i): the drugs scored in the range-censored ln IC50 pair, scored without harmonization and with
        # the shared panel only (phase 2)
        t2 = []
        for T in TG:
            rl = sorted(rows[rows.pair == f'CTRPv2|RL-{T}|lnIC50->{T}|RL|lnIC50'].drug.unique())
            for ref, tgt, lab in [(cd.REFERENCE, f'{T}|CC|lnIC50', 'unharmonized'),
                                  (f'CTRPv2|SL-{T}|lnIC50', f'{T}|SL|lnIC50', 'shared panel')]:
                t2.append({'pair': f'{ref}->{tgt}@RLdrugs', 'reference': ref, 'target': tgt, 'drugs': rl,
                           'family': f'ln IC50 {lab}, drugs scored in the range-censored version ({len(rl)})'})
        for k, t in enumerate(t2):
            t['k'] = len(tasks) + k
        rows2, idx2 = run_pool(lb_task, t2, 'leaderboard', t0)
        rows, idx = pd.concat([rows, rows2], ignore_index=True), pd.concat([idx, idx2], ignore_index=True)
        tasks = tasks + t2
        rows.to_csv(OUT / 'per_drug_leaderboard.csv.gz', index=False, float_format='%.6g')
        for t in tasks:
            index.append({'benchmark': 'leaderboard', 'k': t['k'], 'pair': t['pair'], 'source': t['reference'],
                          'target': t['target'], 'family': t['family'],
                          'drugs_used': int(idx.set_index('k').drugs_used.get(t['k'], 0)),
                          'pool_main': ';'.join(L['published21']),
                          'pool_alt': ';'.join([m for m in L['published21']
                                                if m not in ('Lasso', 'ElasticNet', 'MultiViewRandomForest')]),
                          'reference_model': lib.MEASURED})
        G.pop('L')
    # ------------------------------------------------------------------ endpoint-matched leaderboard pairs (round 3)
    if 'leaderboard_matched' in args.parts:
        tasks = matched_tasks()
        L = lib.leaderboard_setup(need_versions=sorted({t['reference'] for t in tasks} | {t['target'] for t in tasks}),
                                  harmonized=False)
        G['L'] = L
        print('leaderboard (matched) set-up', f'{time.time() - t0:.0f}s', flush=True)
        rows, idx = run_pool(lb_task, tasks, 'leaderboard_matched', t0)
        rows.to_csv(OUT / 'per_drug_leaderboard_matched.csv.gz', index=False, float_format='%.6g')
        for t in tasks:
            index.append({'benchmark': 'leaderboard', 'k': t['k'], 'pair': t['pair'], 'source': t['reference'],
                          'target': t['target'], 'family': t['family'],
                          'drugs_used': int(idx.set_index('k').drugs_used.get(t['k'], 0)),
                          'pool_main': ';'.join(L['published21']),
                          'pool_alt': ';'.join([m for m in L['published21']
                                                if m not in ('Lasso', 'ElasticNet', 'MultiViewRandomForest')]),
                          'reference_model': lib.MEASURED})
        G.pop('L')
    # ------------------------------------------------------------------ retraining
    if 'retrain' in args.parts:
        wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
        G['retrain'] = {}
        tasks = []
        k = 0
        rt0 = lib.load_module('rt59', '59_retrain_score.py')
        settings = ['GDSC2_AUC', 'GDSC2_LNIC50', 'CTRPv2_LNIC50']
        if 'CTRPv2_AUC' in set(wp.setting):
            oof4, _, inc4 = rt0.load_oof('CTRPv2_AUC')
            if all(m in oof4 for m in DREVAL_LEARNED) or args.allow_incomplete_ctrpv2auc:
                settings.append('CTRPv2_AUC')          # appended: earlier settings keep their k and seeds
            else:
                print('CTRPv2_AUC: learned models incomplete, not scored', inc4, flush=True)
        for setting in settings:
            base = sorted(wp[(wp.setting == setting) & (wp.panel == 'main')].model.unique())
            rt = lib.load_module('rt59', '59_retrain_score.py')
            oof, _, inc = rt.load_oof(setting)
            extra = [m for m in DREVAL_LEARNED if m in oof and m not in base]
            S = lib.retrain_setup(setting, models=base + extra)
            Zs, Gs, _, _ = lib.selbench.decompose(S['obs'][['drug', 'cell', 'response']])
            S['Zs'], S['Gs'] = Zs, Gs
            G['retrain'][setting] = S
            print('retrain set-up', setting, len(S['models']), 'models', len(S['cells']), 'cells; new learned:', extra,
                  f'{time.time() - t0:.0f}s', flush=True)
            for T in [t for t in rt.DATASETS if t != S['dataset']]:
                tasks.append({'pair': f'{setting}->{T}', 'setting': setting, 'source': S['dataset'], 'target': T,
                              'endpoint': S['endpoint'], 'k': k, 'seed': 20261015 + k})
                lm = [m for m in DREVAL_LEARNED if m in S['models']]
                lbp = rt.published_pool(S['models'])          # the 21 published leaderboard models (59)
                index.append({'benchmark': 'retrain', 'k': k, 'pair': f'{setting}->{T}', 'source': f'{S["dataset"]}|CC|{S["endpoint"]}',
                              'target': f'{T}|CC|{S["endpoint"]}', 'family': setting,
                              'pool_main': ';'.join(lbp) if setting == 'CTRPv2_LNIC50' else ';'.join(lm),
                              'pool_alt': ';'.join([m for m in S['models'] if m.startswith('PerDrugRidge')]),
                              'reference_model': 'ref:source_measurement', 'models': ';'.join(S['models'])})
                k += 1
        rows, idx = run_pool(retrain_task, tasks, 'retrain', t0)
        rows.to_csv(OUT / 'per_drug_retrain.csv.gz', index=False, float_format='%.6g')
        for r in index:
            if r['benchmark'] == 'retrain':
                r['drugs_used'] = int(idx.set_index('k').drugs_used.get(r['k'], 0))
        G.pop('retrain')
    # ------------------------------------------------------------------ IMPROVE (replicate-free)
    if 'improve' in args.parts:
        I = lib.improve_setup(replicate_free=True)
        G['improve'] = I
        print('IMPROVE set-up', f'{time.time() - t0:.0f}s', flush=True)
        tasks, k = [], 0
        for var in ['own_panel', 'shared_panel']:
            for src in I['datasets']:
                for trg in I['datasets']:
                    if trg == src:
                        continue
                    tasks.append({'pair': f'{src}->{trg}|{var}', 'source': src, 'target': trg, 'variant': var, 'k': k,
                                  'seed': 20261016 + k})
                    index.append({'benchmark': 'improve', 'k': k, 'pair': f'{src}->{trg}|{var}', 'source': src,
                                  'target': trg, 'family': var, 'pool_main': ';'.join(I['models']), 'pool_alt': '',
                                  'reference_model': 'source_measurement'})
                    k += 1
        rows, idx = run_pool(improve_task, tasks, 'improve', t0)
        rows.to_csv(OUT / 'per_drug_improve.csv.gz', index=False, float_format='%.6g')
        for r in index:
            if r['benchmark'] == 'improve':
                r['drugs_used'] = int(idx.set_index('k').drugs_used.get(r['k'], 0))
    old = OUT / 'pairs.csv'
    ix = pd.DataFrame(index)
    if old.exists():
        prev = pd.read_csv(old)
        # a rerun of a part replaces that part's rows only: 'leaderboard' owns leaderboard k < MATCHED_K0,
        # 'leaderboard_matched' owns k >= MATCHED_K0, 'retrain' and 'improve' own their benchmark
        drop = np.zeros(len(prev), bool)
        for part in args.parts:
            if part == 'leaderboard':
                drop |= (prev.benchmark == 'leaderboard').values & (prev.k < MATCHED_K0).values
            elif part == 'leaderboard_matched':
                drop |= (prev.benchmark == 'leaderboard').values & (prev.k >= MATCHED_K0).values
            else:
                drop |= (prev.benchmark == part).values
        ix = pd.concat([prev[~drop], ix], ignore_index=True)
    ix.to_csv(old, index=False)
    lib.write_json(OUT / f'run_{"_".join(args.parts)}.json', {'parts': args.parts, 'seconds': round(time.time() - t0),
                                                             'B': lib.B, 'workers': WORKERS})
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
