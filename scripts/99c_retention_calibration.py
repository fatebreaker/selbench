"""Round 3, task 2b (editor item 5; R3-M4): is the expected-retention benchmark sqrt(R_T / R_S) calibrated for
symmetric selective-both?

The benchmark assumes the triad model: each dataset X measures the drug's selective profile as z_X = t + u_X, with t
shared by every dataset and u_X independent between datasets. A predictor of t only (plus noise independent of every
u) then keeps exactly sqrt(R_T / R_S) of its selective-both, R_X = var(t) / var(t + u_X), and screen triads estimate
R_X = r_XY r_XZ / r_YZ. R3 asked whether the benchmark is miscalibrated for selective-both (partialled on both axes)
or whether tissue x drug effects are themselves dataset-specific. Two checks:

1. Empirical, on the data (GDSC2 CurveCurator AUC as source, 90's step a: unharmonized, each dataset's own panel and
   axis; 90's cells, two-way draws and seed): a third screen's measured selectivity as the predictor. For target T
   and every third dataset X (not GDSC2, not T), p = X's selective residual (own-axis residual of X's CurveCurator AUC,
   rank-normal), which under the triad model is t + u_X: a predictor of shared selectivity only. Its observed
   retention (cross / within symmetric selective-both, pooled over its drugs) is compared with the expected retention
   on the same drugs (84/90's triads and seed). If the benchmark is calibrated, observed ~ expected; if not, the gap
   shows the benchmark's own bias. Caveats: X shares an institution with GDSC2 (GDSC1) or with CTRP v2 (CTRP v1) in
   some pairs; such shared error breaks the triad model in a known direction and is flagged.
2. Simulation under the triad model (and two departures), scored exactly like the data (rank-normal profiles,
   leave-drug-out axes, symmetric selective-both via 77.paired_stats, 84.triad_R with the bias-corrected product):
   five datasets with the GDSC2 -> GDSC1 pair's reliabilities (R = 0.445, 0.255; others 0.43, 0.44, 0.50), 127
   drugs, 500 cell lines, axis share ~0.10 (AUC-like) and dataset-specific axis noise (cross-screen axis agreement
   ~0.5). Scenarios:
     A  triad model holds; predictor = t + noise (+ some axis), the GDSC2-trained 'learned' predictor
     B  GDSC1 and GDSC2 share an institutional error component (25% of their error variance), and CTRP v1 and v2
        likewise; predictor of t only
     C  as B, but the predictor also learned GDSC2's share of the Sanger-specific component (dataset-specific
        selectivity, the reading the paper gave)
     D  triad model holds for molecular selectivity, but tissue x drug effects are partly dataset-specific (each
        dataset adds its own tissue x drug deviations, 50% of the tissue variance); predictor = GDSC2's tissue x drug
        means (a tissue baseline fitted on the source)
   50 replicates each; observed and expected retention, their difference, and the true sqrt(R_T / R_S).
Outputs (results/revision_round2/retention_calibration/): third_screen.csv, simulation.csv, simulation_summary.csv.
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

O2 = lib.OUT2
OUT = lib.out_dir('retention_calibration')
LEARNED = ['RandomForest', 'SimpleNeuralNetwork', 'EnsembleMF', 'GradientBoosting']
SAME_INSTITUTION = {frozenset({'GDSC1', 'GDSC2'}), frozenset({'CTRPv1', 'CTRPv2'})}


# ----------------------------------------------------------------------------------------------------------------
def third_screen(t0):
    h90 = lib.load_module('h90', '90_gdsc2_harmonization.py')
    sr = lib.load_module('sr84', '84_source_reference.py')
    rt = lib.load_module('rt59', '59_retrain_score.py')
    cd = lib.load_module('cd41', '41_crosslab_data.py')
    resp = rt.responses()
    obs = lambda ds: resp[(resp.dataset == ds) & (resp.endpoint == 'auc')][['drug', 'cell', 'response']]  # noqa: E731
    Zs, Gs, _, _ = lib.selbench.decompose(obs('GDSC2'))
    wp = pd.read_csv(lib.RES / 'revision_retrain' / 'within_per_drug.csv.gz', usecols=['setting', 'panel', 'model'])
    base = sorted(wp[(wp.setting == 'GDSC2_AUC') & (wp.panel == 'main')].model.unique())
    S = lib.retrain_setup('GDSC2_AUC', models=base + [m for m in LEARNED if m not in base])
    models = S['models']
    U = np.array(sorted(S['cells']))
    all_drugs = sorted(set().union(*[set(S['P'][m].index) for m in models]))
    drug_ix = {d: i for i, d in enumerate(all_drugs)}
    W, V = lib.draws(len(U), len(all_drugs), h90.SEED)                    # 90's draws
    Vv = cd.versions()
    base_res = {ds: lib.selbench.decompose(Vv[f'{ds}|CC|AUC'])[2] for ds in cd.DATASETS}
    corr, drugs_c, Vt, ce = sr.pair_corrs(base_res, h90.SEED + 1)         # 90's 'unharmonized' triads
    old = pd.read_csv(O2 / 'harmonization_gdsc2' / 'ladder.csv')
    rows = []
    for T in h90.TARGETS:
        Zt, Gt, _, _ = lib.selbench.decompose(obs(T))
        # the learned models' and tissue baseline's values on all of the pair's drugs (90's step a) for reference
        mods, _, Sa = lib.load_paired(O2 / 'harmonization_gdsc2' / 'arrays' / f'{T}__a.npz')
        for X in [d for d in cd.DATASETS if d not in ('GDSC2', T)]:
            name = f'third:{X}'
            P = {name: base_res[X]}
            acc, n, nc = h90.score_pair(Zs, Gs, Zt, Gt, P, [name], U, W, V, drug_ix)
            if n < 5:
                continue
            pdrugs = sorted({r['drug'] for r in acc.rows if r['model'] == name})
            tmp = OUT / '_tmp_third.npz'
            acc.save(tmp, {})
            m3, _, S3 = lib.load_paired(tmp)
            tmp.unlink()
            sw, sc = lib.pooled(S3, m3, [name], 'selBOTH_sym', 'sum_w'), lib.pooled(S3, m3, [name], 'selBOTH_sym', 'sum_c')
            n3 = lib.pooled(S3, m3, [name], 'selBOTH_sym', 'n')
            RS, nS = sr.triad_R(corr, drugs_c, Vt, ce, 'GDSC2', pdrugs)
            RT, nT = sr.triad_R(corr, drugs_c, Vt, ce, T, pdrugs)
            with np.errstate(invalid='ignore', divide='ignore'):
                exp_ret = np.sqrt(np.clip(RT, 0, None) / RS)
                obs_ret = sc / sw
            # learned models and the tissue baseline on the same drugs (rescored from 90's arrays is not possible
            # per drug subset, so they are rescored here with 90's draws)
            P2 = {m: S['P'][m] for m in LEARNED + ['NaiveTissueDrugMeanPredictor']}
            acc2, _, _ = h90.score_pair(Zs, Gs, Zt, Gt, P2, list(P2), U, W, V, drug_ix, drugs=pdrugs)
            acc2.save(tmp, {})
            m2, _, S2 = lib.load_paired(tmp)
            tmp.unlink()
            extra = {}
            for gname, gm in [('learned4', LEARNED), ('tissue', ['NaiveTissueDrugMeanPredictor'])]:
                a = lib.pooled(S2, m2, gm, 'selBOTH_sym', 'sum_c') / lib.pooled(S2, m2, gm, 'selBOTH_sym', 'sum_w')
                extra.update(lib.summ(a, f'{gname}_observed_retention_same_drugs'))
                extra.update(lib.summ(a - exp_ret, f'{gname}_observed_minus_expected_same_drugs'))
                extra.update(lib.summ(a - obs_ret, f'{gname}_minus_third_screen_retention'))
            shared = [frozenset({X, 'GDSC2'}) in SAME_INSTITUTION, frozenset({X, T}) in SAME_INSTITUTION]
            rows.append({'target': T, 'third_screen': X, 'drugs': len(pdrugs), 'drugs_triads': int(min(nS, nT)),
                         'shares_institution_with': 'GDSC2 (source)' if shared[0] else ('target' if shared[1] else ''),
                         **lib.summ(sw / n3, 'within_selBOTH'), **lib.summ(sc / n3, 'cross_selBOTH'),
                         **lib.summ(obs_ret, 'observed_retention'), **lib.summ(exp_ret, 'expected_retention'),
                         **lib.summ(obs_ret - exp_ret, 'observed_minus_expected'),
                         **lib.summ(RS, 'R_source'), **lib.summ(RT, 'R_target'), **extra})
            print('third screen', T, X, len(pdrugs), f'{time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
NAMES = ['GDSC2', 'GDSC1', 'CTRPv2', 'CTRPv1', 'CCLE']
RTRUE = {'GDSC2': 0.445, 'GDSC1': 0.255, 'CTRPv2': 0.43, 'CTRPv1': 0.44, 'CCLE': 0.50}


def sim_once(rng, scenario, D=127, n=500, phi=0.25, n_tissues=15, tissue_share=0.3, tissue_specific=0.5):
    """One simulated replicate; returns observed and expected retention for the scenario's predictor."""
    a = rng.normal(size=n)                                         # shared general sensitivity
    tissue = rng.integers(0, n_tissues, n)
    # shared selectivity t (unit variance): molecular part + tissue x drug part (scenario D)
    tis_eff = rng.normal(size=(D, n_tissues))
    if scenario == 'D':
        t = np.sqrt(1 - tissue_share) * rng.normal(size=(D, n)) + np.sqrt(tissue_share) * tis_eff[:, tissue]
    else:
        t = rng.normal(size=(D, n))
    inst = {'Sanger': rng.normal(size=(D, n)), 'Broad': rng.normal(size=(D, n))}
    inst_of = {'GDSC1': 'Sanger', 'GDSC2': 'Sanger', 'CTRPv1': 'Broad', 'CTRPv2': 'Broad'}
    Z, own_tis = {}, {}
    for X in NAMES:
        R = RTRUE[X]
        err_var = (1 - R) / R                                      # var(u_X) so that var(t)/(var(t)+var(u)) = R
        if scenario in ('B', 'C') and X in inst_of:
            u = np.sqrt(phi * err_var) * inst[inst_of[X]] + np.sqrt((1 - phi) * err_var) * rng.normal(size=(D, n))
        else:
            u = np.sqrt(err_var) * rng.normal(size=(D, n))
        if scenario == 'D':
            # dataset-specific tissue x drug deviations: a share of the tissue variance is replaced per dataset;
            # added to the error so that R_X (shared share of selective variance) keeps its value
            dev = rng.normal(size=(D, n_tissues))[:, tissue] * np.sqrt(tissue_share * tissue_specific)
            own_tis[X] = dev
            u = np.sqrt(max(err_var - tissue_share * tissue_specific, 1e-6)) * rng.normal(size=(D, n)) + dev
        ax = np.sqrt(0.5) * a + np.sqrt(0.5) * rng.normal(size=n)  # dataset's own view of the axis (agreement ~0.5)
        lam = rng.uniform(0.2, 0.8, D)[:, None] * np.sqrt(1 / R)   # axis share ~0.1 of each drug's variance
        Z[X] = lam * ax[None] + t + u
    dec = {}
    for X in NAMES:
        long = pd.DataFrame({'drug': np.repeat(np.arange(D), n), 'cell': np.tile(np.arange(n), D),
                             'response': Z[X].ravel()})
        dec[X] = lib.selbench.decompose(long)
    # predictor (GDSC2-trained)
    if scenario == 'D':
        # tissue x drug means of GDSC2's measured profile over the other cell lines of the tissue (a tissue baseline
        # fitted on the source, leave-cell-line-out)
        zs = dec['GDSC2'][0].values
        p = np.zeros_like(zs)
        for k in range(n_tissues):
            m = tissue == k
            if m.sum() > 1:
                tot = zs[:, m].sum(axis=1, keepdims=True)
                p[:, m] = (tot - zs[:, m]) / (m.sum() - 1)
        p = p + 0.05 * rng.normal(size=p.shape)
    else:
        p = t + np.sqrt(10.0) * rng.normal(size=(D, n)) + 0.3 * a[None]
        if scenario == 'C':
            p = p + np.sqrt(phi * (1 - RTRUE['GDSC2']) / RTRUE['GDSC2']) * inst['Sanger']
    W1 = np.ones((1, n))
    sw = sc = 0.0
    for d in range(D):
        st = lib.paired_stats(dec['GDSC2'][0].values[d], dec['GDSC2'][1].values[d], dec['GDSC1'][0].values[d],
                              dec['GDSC1'][1].values[d], p[d][:, None], W1)
        sw += st['selB_w'][0, 0]
        sc += st['selB_c'][0, 0]
    # triads on the selective residuals, as 84 (unweighted: one draw)
    ce = lib.load_module('ce49', '49_ceiling_estimands.py')
    corr = {}
    for d in range(D):
        for x, y in itertools.combinations(NAMES, 2):
            rx, ry = dec[x][2].values[d], dec[y][2].values[d]
            corr[(d, x, y)] = (np.array([np.corrcoef(rx, ry)[0, 1]]), n)
    sr = lib.load_module('sr84', '84_source_reference.py')
    V1 = np.ones((1, D))
    RS, _ = sr.triad_R(corr, list(range(D)), V1, ce, 'GDSC2', list(range(D)))
    RT, _ = sr.triad_R(corr, list(range(D)), V1, ce, 'GDSC1', list(range(D)))
    true = np.sqrt(RTRUE['GDSC1'] / RTRUE['GDSC2'])
    return {'observed_retention': sc / sw, 'expected_retention': float(np.sqrt(max(RT[0], 0) / RS[0])),
            'R_source_hat': float(RS[0]), 'R_target_hat': float(RT[0]), 'true_sqrt_ratio': float(true),
            'within_selBOTH': sw / D, 'cross_selBOTH': sc / D}


def simulation(t0, reps=50):
    rows = []
    for k, sc in enumerate(['A', 'B', 'C', 'D']):
        rng = np.random.default_rng(20261099 + k)
        for r in range(reps):
            rows.append({'scenario': sc, 'rep': r, **sim_once(rng, sc)})
        print('simulation', sc, f'{time.time() - t0:.0f}s', flush=True)
    Sm = pd.DataFrame(rows)
    Sm['observed_minus_expected'] = Sm.observed_retention - Sm.expected_retention
    Sm['observed_minus_true'] = Sm.observed_retention - Sm.true_sqrt_ratio
    desc = {'A': 'triad model holds; predictor = shared selectivity + noise + axis',
            'B': 'GDSC1-GDSC2 and CTRPv1-CTRPv2 share 25% of error variance; predictor of shared selectivity only',
            'C': 'as B; predictor also carries GDSC2\'s Sanger-specific component (dataset-specific selectivity)',
            'D': 'tissue x drug effects 30% of selectivity, half of them dataset-specific; predictor = source tissue means'}
    summ = Sm.groupby('scenario').agg(
        reps=('rep', 'size'), observed_retention_mean=('observed_retention', 'mean'),
        observed_retention_sd=('observed_retention', 'std'), expected_retention_mean=('expected_retention', 'mean'),
        expected_retention_sd=('expected_retention', 'std'), true_sqrt_ratio=('true_sqrt_ratio', 'first'),
        observed_minus_expected_mean=('observed_minus_expected', 'mean'),
        observed_minus_expected_sd=('observed_minus_expected', 'std'),
        R_source_hat_mean=('R_source_hat', 'mean'), R_target_hat_mean=('R_target_hat', 'mean'),
        within_selBOTH_mean=('within_selBOTH', 'mean'), cross_selBOTH_mean=('cross_selBOTH', 'mean')).reset_index()
    summ['description'] = summ.scenario.map(desc)
    summ['observed_minus_expected_se'] = summ.observed_minus_expected_sd / np.sqrt(summ.reps)
    return Sm, summ


def main():
    t0 = time.time()
    T3 = third_screen(t0)
    T3.to_csv(OUT / 'third_screen.csv', index=False, float_format='%.5g')
    Sm, summ = simulation(t0)
    Sm.to_csv(OUT / 'simulation.csv', index=False, float_format='%.5g')
    summ.to_csv(OUT / 'simulation_summary.csv', index=False, float_format='%.5g')
    pd.set_option('display.width', 250)
    print(T3[['target', 'third_screen', 'drugs', 'shares_institution_with', 'within_selBOTH', 'observed_retention',
              'observed_retention_lo', 'observed_retention_hi', 'expected_retention', 'observed_minus_expected',
              'observed_minus_expected_lo', 'observed_minus_expected_hi', 'learned4_observed_minus_expected_same_drugs',
              'tissue_observed_minus_expected_same_drugs']].round(3).to_string(index=False))
    print(summ.round(3).to_string(index=False))
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
