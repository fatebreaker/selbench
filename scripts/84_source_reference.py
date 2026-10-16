"""Round 2, task 6 (R1-M3d, R3-M3c): the source-measurement reference, its noise-corrected maximum, and the retention
expected for a predictor of shared selectivity.

1. Accuracy relative to one replicate. For each paired target (80_paired_engine.py), a model's cross selective-both
   r(p, z_T | g_T, g_S) relative to that of the source dataset's own measurement z_S scored the same way on the same
   drugs and cells: ratio of sums over drugs (and models, for pooled groups), two-way bootstrap (B = 1,000). Under
   z_S = t + u_S, z_T = t + u_T this is corr(p, t) / corr(z_S, t): accuracy for shared selectivity relative to one
   replicate, not a bound and not a measure of loss.
2. Noise-corrected maximum. A perfect predictor of S's reproducible profile would reach the measurement's value
   divided by sqrt(R_sel,S) (R_sel,S: within-screen reliability of S's selective residual, 79_reliability.py; draws
   paired by index). The model's fraction of that maximum is ratio x sqrt(R_sel,S).
3. Expected retention for a predictor of shared selectivity only (R1-M3d): sqrt(R_T / R_S), with R_X the reproducible
   share of X's selective variance from screen triads, R_X = r_XY r_XZ / r_YZ over the other datasets Y, Z measuring the
   drug (selective residuals, common cells >= 30; per-drug means of bias-corrected products over triads, pooled as a
   ratio of sums over the pair's drugs, as 49's primary estimator). Datasets: the five DrEval CurveCurator datasets
   for the leaderboard and retraining (same endpoint as the pair), the five IMPROVE datasets for IMPROVE. Two-way
   bootstrap (cell weights inside every correlation, drug multiplicities). Observed retention: cross / within
   selective-both (symmetric), pooled. The observed minus expected retention is reported with paired-by-index draws
   (independent bootstraps).
Outputs (results/revision_round2/source_reference/): source_ratio_by_target.csv, triad_reliability.csv,
expected_retention.csv, summary.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

PAIRED = lib.OUT2 / 'paired'
OUT = lib.out_dir('source_reference')
MIN_COMMON = 30
SEED = 20261051
# source dataset -> reliability key in 79's draws (selective residual)
REL_SRC = {'CTRPv2|CC|lnIC50': [('CTRPv1v2_lnIC50|Rsel_median_corr', 'CTRP v1-v2 lower bound', True),
                                ('IMPROVE_CTRPv2_ic50|Rsel_var', 'IMPROVE CTRPv2 IC50 replicates', False)],
           'CTRPv2|CC|AUC': [('CTRPv1v2_AUC|Rsel_median_corr', 'CTRP v1-v2 lower bound (AUC)', True),
                             ('IMPROVE_CTRPv2_auc|Rsel_var', 'IMPROVE CTRPv2 AUC replicates', False)],
           'GDSC2|CC|AUC': [('IMPROVE_GDSCv2_auc|Rsel_var', 'IMPROVE GDSC2 AUC replicates', True)],
           'GDSC2|CC|lnIC50': [('IMPROVE_GDSCv2_ic50|Rsel_var', 'IMPROVE GDSC2 IC50 replicates', True)],
           'CTRPv2': [('IMPROVE_CTRPv2_auc|Rsel_var', 'own replicates', True)],
           'GDSCv1': [('IMPROVE_GDSCv1_auc|Rsel_var', 'own replicates', True)],
           'GDSCv2': [('IMPROVE_GDSCv2_auc|Rsel_var', 'own replicates', True)],
           'CCLE': [('IMPROVE_GDSCv2_auc|Rsel_var', 'no replicates; GDSC2 value borrowed', True)],
           'gCSI': [('IMPROVE_GDSCv2_auc|Rsel_var', 'no replicates; GDSC2 value borrowed', True)]}


# ----------------------------------------------------------------------------------------------------------------
def residual_sets():
    """{family: {dataset: R (drug x cell selective residuals)}} for DrEval CC ln IC50, CC AUC and IMPROVE AUC."""
    cd = lib.load_module('cd41', '41_crosslab_data.py')
    V = cd.versions()
    out = {}
    for ep in ['lnIC50', 'AUC']:
        out[f'DrEval CC {ep}'] = {ds: lib.selbench.decompose(V[f'{ds}|CC|{ep}'])[2] for ds in cd.DATASETS}
    im = lib.load_module('im68', '68_improve_csa.py')
    im.extract()
    obs = im.responses()
    out['IMPROVE AUC'] = {ds: lib.selbench.decompose(obs[ds])[2] for ds in im.DATASETS}
    return out


def pair_corrs(Rs, seed):
    """Per drug and dataset pair: weighted correlations of selective residuals on common cells, (B+1) draws."""
    ce = lib.load_module('ce49', '49_ceiling_estimands.py')
    names = list(Rs)
    cells = sorted(set().union(*[set(R.columns) for R in Rs.values()]))
    drugs = sorted(set().union(*[set(R.index) for R in Rs.values()]))
    cix = {c: i for i, c in enumerate(cells)}
    W, V = lib.draws(len(cells), len(drugs), seed)
    corr = {}
    for d in drugs:
        have = [n for n in names if d in Rs[n].index]
        if len(have) < 3:
            continue
        for a, b in itertools.combinations(have, 2):
            x, y = Rs[a].loc[d], Rs[b].loc[d]
            com = x.index.intersection(y.index)
            xv, yv = x.reindex(com).values.astype(float), y.reindex(com).values.astype(float)
            m = np.isfinite(xv) & np.isfinite(yv)
            if m.sum() < MIN_COMMON:
                continue
            xv, yv = xv[m], yv[m]
            w = W[:, [cix[c] for c in com[m]]]
            n = w.sum(1)
            mx, my = (w @ xv) / n, (w @ yv) / n
            cxx = (w @ (xv * xv)) / n - mx ** 2
            cyy = (w @ (yv * yv)) / n - my ** 2
            cxy = (w @ (xv * yv)) / n - mx * my
            corr[(d, a, b)] = (cxy / np.sqrt(cxx * cyy), int(m.sum()))
    return corr, drugs, V, ce


def triad_R(corr, drugs, V, ce, focal, drug_set):
    """Pooled reproducible share R_focal over drug_set: sum_d mean_k (r_fa r_fb - psi/n) / sum_d mean_k r_ab."""
    dix = {d: i for i, d in enumerate(drugs)}
    R1 = V.shape[0]
    num, den = np.zeros(R1), np.zeros(R1)
    nd = 0
    get = lambda d, a, b: corr.get((d, a, b)) or corr.get((d, b, a))  # noqa: E731
    for d in drug_set:
        if d not in dix:
            continue
        others = sorted({k[1] for k in corr if k[0] == d} | {k[2] for k in corr if k[0] == d})
        others = [o for o in others if o != focal]
        terms_n, terms_d = [], []
        for a, b in itertools.combinations(others, 2):
            fa, fb, ab = get(d, focal, a), get(d, focal, b), get(d, a, b)
            if fa is None or fb is None or ab is None:
                continue
            n = min(fa[1], fb[1], ab[1])
            terms_n.append(ce.bias_corrected_product(fa[0], fb[0], ab[0], n))
            terms_d.append(ab[0])
        if not terms_n:
            continue
        v = V[:, dix[d]]
        num += v * np.mean(terms_n, axis=0)
        den += v * np.mean(terms_d, axis=0)
        nd += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den, nd


def main():
    t0 = time.time()
    ix = pd.read_csv(PAIRED / 'pairs.csv')
    rz = np.load(lib.OUT2 / 'reliability' / 'reliability_draws.npz')
    reld = {k: rz[k] for k in rz.files}
    # ---------------- 1-2. ratios to the source measurement
    rows = []
    per_pair = {}
    for _, r in ix.iterrows():
        f = PAIRED / 'arrays' / f'{r.benchmark}__{r.k}.npz'
        if not f.exists() or r.drugs_used == 0 or r.source == r.target:
            continue
        models, meta, S = lib.load_paired(f)
        per_pair[(r.benchmark, r.k)] = (models, S)
        groups = {}
        if isinstance(r.pool_main, str) and r.pool_main:
            groups['pooled: ' + {'leaderboard': 'published 21', 'retrain': 'DrEval learned / published',
                                 'improve': 'IMPROVE 7'}[r.benchmark]] = r.pool_main.split(';')
        for m in ['EnsembleMF', 'LB:EnsembleMF', 'RandomForest', 'PerDrugRidgeRaw', 'PerDrugRidgeSelective',
                  'ridge_sel_lm', 'ridge_raw_lm', 'GradientBoosting', 'SimpleNeuralNetwork']:
            if m in models:
                groups[m] = [m]
        src_key = str(r.source).replace('|auc', '|AUC').replace('|lnic50', '|lnIC50')
        rel_list = REL_SRC.get(src_key, [])
        for gname, gm in groups.items():
            gm = [m for m in gm if m in models]
            if not gm:
                continue
            g = lambda st, kind: lib.pooled(S, models, gm, st, kind)  # noqa: E731
            ix_ = [models.index(m) for m in gm]
            num = S['ref']['mod_selB_c'][:, ix_].sum(1)
            den = S['ref']['ref_selB_c'][:, ix_].sum(1)
            nref = S['ref']['n_ref'][:, ix_].sum(1)
            rnum, rden = S['ref']['mod_raw_c'][:, ix_].sum(1), S['ref']['ref_raw_c'][:, ix_].sum(1)
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = num / den
                rr = rnum / rden
                sw, sc, n = g('selBOTH_sym', 'sum_w'), g('selBOTH_sym', 'sum_c'), g('selBOTH_sym', 'n')
                retained = sc / sw
            row = {'benchmark': r.benchmark, 'pair': r.pair, 'source': r.source, 'target': r.target,
                   'family': r.family, 'group': gname, 'models': len(gm),
                   **lib.summ(den / nref, 'measurement_cross_selBOTH'), **lib.summ(num / nref, 'model_cross_selBOTH'),
                   **lib.summ(ratio, 'ratio_to_measurement'), **lib.summ(rr, 'raw_ratio_to_measurement'),
                   **lib.summ(retained, 'observed_retention_selBOTH_sym'), **lib.summ(sw / n, 'within_selBOTH_sym')}
            for key, lab, prim in rel_list:
                Rs = reld.get(key)
                if Rs is None:
                    continue
                with np.errstate(invalid='ignore'):
                    row.update(lib.summ(Rs, f'R_sel_source[{lab}]'))
                    row.update(lib.summ(den / nref / np.sqrt(np.clip(Rs, 1e-6, None)),
                                        f'noise_corrected_max[{lab}]'))
                    row.update(lib.summ(ratio * np.sqrt(np.clip(Rs, 0, None)), f'fraction_of_corrected_max[{lab}]'))
            rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'source_ratio_by_target.csv', index=False, float_format='%.5g')
    print('ratios done', f'{time.time() - t0:.0f}s', flush=True)
    # ---------------- 3. triads: expected retention for a predictor of shared selectivity
    rsets = residual_sets()
    fam_of = {'leaderboard': lambda r: 'DrEval CC ' + r.target.split('|')[-1] if '|CC|' in r.target or '|SL|' in r.target
              or '|RL|' in r.target or '|HR|' in r.target or '|SP|' in r.target else None,
              'retrain': lambda r: 'DrEval CC ' + ('lnIC50' if r.family.endswith('LNIC50') else 'AUC'),
              'improve': lambda r: 'IMPROVE AUC'}
    trows, erows = [], []
    corr_cache = {}
    for k, (fam, Rs) in enumerate(rsets.items()):
        corr_cache[fam] = pair_corrs(Rs, SEED + k)
        print('pair correlations', fam, len(corr_cache[fam][0]), f'{time.time() - t0:.0f}s', flush=True)
    for _, r in ix.iterrows():
        key = (r.benchmark, r.k)
        if key not in per_pair:
            continue
        if r.benchmark == 'leaderboard' and not (r.family in ('lnIC50-reference', 'AUC-reference')
                                                 and '|CC|' in r.target and not r.target.startswith('CTRPv2')):
            continue
        if r.benchmark == 'improve' and r.family != 'own_panel':
            continue
        fam = fam_of[r.benchmark](r)
        if fam is None or fam not in corr_cache:
            continue
        corr, drugs, V, ce = corr_cache[fam]
        sname = r.source.split('|')[0] if r.benchmark != 'improve' else r.source
        tname = r.target.split('|')[0]
        if r.benchmark == 'leaderboard' and r.family == 'lnIC50-reference' and not r.target.endswith('lnIC50'):
            continue                                   # mixed-endpoint pairs: no single triad family
        rows_pd = pd.read_csv(PAIRED / f'per_drug_{r.benchmark}.csv.gz', usecols=['pair', 'drug'],
                              dtype={'drug': str}) if r.benchmark not in corr_cache.get('_pd', {}) else None
        if rows_pd is not None:
            corr_cache.setdefault('_pd', {})[r.benchmark] = rows_pd
        pdrugs = sorted(corr_cache['_pd'][r.benchmark].query('pair == @r.pair').drug.unique())
        RS, nS = triad_R(corr, drugs, V, ce, sname, pdrugs)
        RT, nT = triad_R(corr, drugs, V, ce, tname, pdrugs)
        with np.errstate(invalid='ignore'):
            exp_ret = np.sqrt(np.clip(RT, 0, None) / RS)
        models, S = per_pair[key]
        for gname, gm in ([('pooled', r.pool_main.split(';'))] if isinstance(r.pool_main, str) and r.pool_main else []) + \
                [(m, [m]) for m in ['EnsembleMF', 'PerDrugRidgeRaw', 'LB:EnsembleMF', 'EnsembleMF'] if m in models]:
            gm = [m for m in gm if m in models]
            if not gm:
                continue
            sw, sc = lib.pooled(S, models, gm, 'selBOTH_sym', 'sum_w'), lib.pooled(S, models, gm, 'selBOTH_sym', 'sum_c')
            with np.errstate(invalid='ignore', divide='ignore'):
                obs_ret = sc / sw
            erows.append({'benchmark': r.benchmark, 'pair': r.pair, 'family': fam, 'group': gname,
                          'drugs_pair': len(pdrugs), 'drugs_triads_S': nS, 'drugs_triads_T': nT,
                          **lib.summ(RS, 'R_source'), **lib.summ(RT, 'R_target'),
                          **lib.summ(exp_ret, 'expected_retention'), **lib.summ(obs_ret, 'observed_retention'),
                          **lib.summ(obs_ret - exp_ret, 'observed_minus_expected')})
        trows.append({'benchmark': r.benchmark, 'pair': r.pair, 'family': fam, 'source': sname, 'target': tname,
                      **lib.summ(RS, 'R_source'), **lib.summ(RT, 'R_target'), 'drugs_S': nS, 'drugs_T': nT})
    # dataset-level reproducible shares over every drug in >= 3 datasets (a cross-screen lower bound on the within-
    # screen reliability of the selective residual; used by 78 as an additional reliability source)
    tri_draws, tri_rows = {}, []
    for fam in rsets:
        corr, drugs, V, ce = corr_cache[fam]
        for ds in rsets[fam]:
            Rd, nd = triad_R(corr, drugs, V, ce, ds, drugs)
            if nd < 5:
                continue
            tri_draws[f'triad_{fam}_{ds}|R_triad'] = Rd
            tri_rows.append({'family': fam, 'dataset': ds, 'drugs': nd, **lib.summ(Rd, 'R_triad')})
    np.savez_compressed(lib.OUT2 / 'reliability' / 'triad_R_draws.npz', **tri_draws)
    pd.DataFrame(tri_rows).to_csv(OUT / 'triad_reliability_all_drugs.csv', index=False, float_format='%.5g')
    E = pd.DataFrame(erows).drop_duplicates(['benchmark', 'pair', 'group'])
    E.to_csv(OUT / 'expected_retention.csv', index=False, float_format='%.5g')
    pd.DataFrame(trows).to_csv(OUT / 'triad_reliability.csv', index=False, float_format='%.5g')
    # summary
    summ = {}
    for bm, g in T.groupby('benchmark'):
        gg = g[g.group.str.startswith('pooled')]
        summ[bm] = {'pooled_ratio_median': float(gg.ratio_to_measurement.median()) if len(gg) else None,
                    'pooled_ratio_min': float(gg.ratio_to_measurement.min()) if len(gg) else None,
                    'pooled_ratio_max': float(gg.ratio_to_measurement.max()) if len(gg) else None}
    lib.write_json(OUT / 'summary.json', {'ratios': summ, 'seconds': round(time.time() - t0)})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 300)
    print(E[['benchmark', 'pair', 'group', 'R_source', 'R_target', 'expected_retention', 'expected_retention_lo',
             'expected_retention_hi', 'observed_retention', 'observed_retention_lo', 'observed_retention_hi']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
