"""Round 2, task 9 (R3-M7, R1-M6): the ceiling statements.

1. Factor-model implied bias of the model/replicate c^2 ratio, with drug-cluster intervals. The pooled confirmatory
   factor models of 50_crosslab_factor.py (with the per-screen selective ridge as indicators) are refitted on the same
   400 drug-cluster bootstrap resamples (same seeds, SEED + 100 + b, so the parameter intervals are 50's), and in each
   refit the discovery-only estimator's implied value is compared with the general-factor truth
   (50.implied_discovery_bias) for the ridge c^2, the replicate c^2 and their ratio. Bias = implied / truth - 1.
   The better-fitting model is chosen by 50's drug-level cross-validation (50_cfa_cv.csv).
2. Typical drug: REML population median of per-drug c^2 (49's hierarchical model; Fisher-z, per-drug SEs) for the
   ridge and for the replicate (measured profile) and the like-for-like replicate (measured selective residual), and
   their ratio, with a paired drug-cluster bootstrap (1,000 draws); also the plain median of per-drug estimates.
3. Like-for-like replicate ratio and the headline c^2 with two-way and drug-cluster intervals (from 49's outputs).
Outputs (results/revision_round2/ceiling/): cfa_implied_bias.csv, typical_drug.csv, headline.csv, summary.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
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

OUT = lib.out_dir('ceiling')
CE = lib.RES / 'revision_ceiling'
WORKERS = 3
N_BOOT = 400
SPECS = {'P0_general_predictors': dict(variant='multiplicative', lab=False, ctg=False, predictors=True),
         'P3_lab_ctg_predictors': dict(variant='multiplicative', lab=True, ctg=True, predictors=True),
         'P3_lab_ctg_predictors_additive': dict(variant='additive', lab=True, ctg=True, predictors=True)}
G = {}


def inputs():
    cf = lib.load_module('cf50', '50_crosslab_factor.py')
    pw = pd.read_csv(CE / '50_pairwise_selective_correlations.csv')
    nscr = pd.concat([pw[['compound', 's1']].rename(columns={'s1': 's'}), pw[['compound', 's2']].rename(columns={'s2': 's'})])
    nscr = nscr.drop_duplicates().groupby('compound').size()
    keep = nscr[nscr >= 3].index
    obs_ss = pw[pw.compound.isin(keep)].assign(kind='ss', train=None)
    pt = pd.read_csv(CE / '50_predictor_screen_correlations.csv')
    pt = pt[pt.model == 'ridge_sel']
    obs_fs = pt[pt.compound.isin(keep)].rename(columns={'screen': 's2', 'pc': 'rho'}).assign(kind='fs', s1=lambda x: x.train)
    obs_fs = obs_fs[['compound', 's1', 's2', 'n', 'rho', 'kind', 'train']]
    disc = pd.read_csv(CE / '49_disattenuation_terms_bc.csv')
    disc = disc[disc.model.isin(['ridge_sel', 'replicate'])]
    return cf, pd.concat([obs_ss, obs_fs]), disc


def boot_one(args):
    """50._boot_cfa with the same resample, plus the implied bias on the resampled drugs."""
    name, b = args
    cf, obs, disc, p0 = G['cf'], G['obs'], G['disc'], G['p0'][name]
    spec = SPECS[name]
    rng = np.random.default_rng(cf.SEED + 100 + b)
    drugs = obs.compound.unique()
    pick = rng.integers(0, len(drugs), len(drugs))
    g = {d: x for d, x in obs.groupby('compound')}
    gd = {d: x for d, x in disc.groupby('compound')}
    parts, dparts = [], []
    for k, i in enumerate(pick):
        x = g[drugs[i]].copy()
        x['compound'] = f'b{k}'
        parts.append(x)
        if drugs[i] in gd:
            y = gd[drugs[i]].copy()
            y['compound'] = f'b{k}'
            dparts.append(y)
    ob = pd.concat(parts, ignore_index=True)
    m = cf.CFA(ob, **spec).fit(p0)
    ib = cf.implied_discovery_bias(m, spec, pd.concat(dparts, ignore_index=True), None)
    return name, b, ib


def typical_drug(n_boot=1000, seed=20261071):
    ce = lib.load_module('ce49', '49_ceiling_estimands.py')
    per = pd.read_csv(CE / '49_c2_hierarchical_perdrug.csv')
    per = per[per.estimable]
    rows = []
    rng = np.random.default_rng(seed)
    for cls in ['all', 'reproducible']:
        sub = per if cls == 'all' else per[per.rep_class == 'reproducible']
        W = {m: sub[sub.model == m].set_index('compound') for m in ['ridge_sel', 'replicate', 'replicate_sel']}
        common = sorted(set(W['ridge_sel'].index) & set(W['replicate'].index) & set(W['replicate_sel'].index))
        z = {m: W[m].loc[common, 'z'].values for m in W}
        s = {m: W[m].loc[common, 'se_z'].values for m in W}
        raw = {m: W[m].loc[common, 'c2_raw'].values for m in W}
        idx = np.vstack([np.arange(len(common)), rng.integers(0, len(common), size=(n_boot, len(common)))])
        out = {}
        for m in W:
            mu = np.array([ce.reml(z[m][i], s[m][i])[0] for i in idx])
            out[f'pop_median_{m}'] = ce.z_to_c2(mu)
            out[f'plain_median_{m}'] = np.median(raw[m][idx], axis=1)
        for den in ['replicate', 'replicate_sel']:
            out[f'pop_median_ratio_ridge_sel/{den}'] = out['pop_median_ridge_sel'] / out[f'pop_median_{den}']
            out[f'plain_median_ratio_ridge_sel/{den}'] = out['plain_median_ridge_sel'] / out[f'plain_median_{den}']
        for k, v in out.items():
            rows.append({'class': cls, 'drugs': len(common), 'quantity': k, **lib.summ(v, 'value')})
    return pd.DataFrame(rows)


def headline():
    tw = pd.read_csv(CE / '49_c2_primary_twoway_bootstrap.csv')
    rs = pd.read_csv(CE / '49_c2_model_replicate_sel_ratio.csv')
    rr = pd.read_csv(CE / '49_c2_model_replicate_ratio.csv')
    rows = []
    for _, r in tw.iterrows():
        rows.append({'quantity': r.model, 'class': r['class'], 'estimate': r.estimate,
                     'drug_cluster_lo': r.get('drug_ci_lo'), 'drug_cluster_hi': r.get('drug_ci_hi'),
                     'two_way_lo': r.twoway_ci_lo, 'two_way_hi': r.twoway_ci_hi, 'drugs': r.n_drugs})
    for lab, t in [('ridge_sel/replicate_sel (drug-cluster, 49_c2_model_replicate_sel_ratio)', rs),
                   ('ridge_sel/replicate (drug-cluster, 49_c2_model_replicate_ratio)', rr)]:
        x = t[(t.model == 'ridge_sel') & (t.kappa == 'none') & (t.aggregation == 'perdrug_romeans_bc')]
        for _, r in x.iterrows():
            rows.append({'quantity': lab, 'class': r['class'], 'estimate': r.ratio, 'drug_cluster_lo': r.ci_lo,
                         'drug_cluster_hi': r.ci_hi, 'two_way_lo': np.nan, 'two_way_hi': np.nan, 'drugs': r.n_drugs})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    cf, obs, disc = inputs()
    G.update({'cf': cf, 'obs': obs, 'disc': disc, 'p0': {}})
    point = {}
    for name, spec in SPECS.items():
        m = cf.CFA(obs, **spec).fit()
        G['p0'][name] = m.p
        point[name] = cf.implied_discovery_bias(m, spec, disc, None)
        print(name, 'ratio implied/truth', round(point[name]['ratio|all|implied_estimate'], 4),
              round(point[name]['ratio|all|general_truth'], 4), f'{time.time() - t0:.0f}s', flush=True)
    tasks = [(name, b) for name in SPECS for b in range(N_BOOT)]
    res = {name: [] for name in SPECS}
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context('fork')) as ex:
        for i, (name, b, ib) in enumerate(ex.map(boot_one, tasks, chunksize=4)):
            res[name].append(ib)
            if i % 200 == 0:
                print('refit', i, f'{time.time() - t0:.0f}s', flush=True)
    rows = []
    cv = pd.read_csv(CE / '50_cfa_cv.csv').set_index('fit').cv_weighted_mse
    for name in SPECS:
        bt = pd.DataFrame(res[name])
        for who in ['ridge_sel', 'replicate', 'ratio']:
            for cfg in ['all', 'E_same_lab', 'T_shares_lab']:
                ki, kt = f'{who}|{cfg}|implied_estimate', f'{who}|{cfg}|general_truth'
                if ki not in point[name]:
                    continue
                rep = np.r_[point[name][ki] / point[name][kt] - 1, (bt[ki] / bt[kt] - 1).values]
                ri = np.r_[point[name][ki], bt[ki].values]
                rt = np.r_[point[name][kt], bt[kt].values]
                rows.append({'fit': name, 'quantity': who, 'configuration': cfg, **lib.summ(ri, 'implied_estimate'),
                             **lib.summ(rt, 'general_truth'), **lib.summ(rep, 'relative_bias'),
                             'refits': int(bt[ki].notna().sum()),
                             'cv_mse_of_matching_measurement_model': float(cv.get(name.replace('P0_general_predictors', 'M0_general')
                                                                                   .replace('P3_lab_ctg_predictors_additive', 'M3_lab_ctg_additive')
                                                                                   .replace('P3_lab_ctg_predictors', 'M3_lab_ctg'), np.nan))})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / 'cfa_implied_bias.csv', index=False, float_format='%.5g')
    T = typical_drug()
    T.to_csv(OUT / 'typical_drug.csv', index=False, float_format='%.5g')
    Hd = headline()
    Hd.to_csv(OUT / 'headline.csv', index=False, float_format='%.5g')
    best = cv.idxmin()
    lib.write_json(OUT / 'summary.json', {'seconds': round(time.time() - t0), 'n_refits': N_BOOT,
                                          'best_fitting_measurement_model_cv': best, 'cv_mse': cv.to_dict()})
    pd.set_option('display.width', 250)
    print(B[B.configuration == 'all'].round(4).to_string(index=False))
    print(T.round(4).to_string(index=False))
    print(Hd.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
