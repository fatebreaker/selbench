"""Power of the registered PDX transfer tests (H8: one-sided Wilcoxon signed-rank over reproducible drugs; H9: one-sided
Mann-Whitney, reproducible versus non-reproducible) for the registered design and for the realised designs (R3 M5).

Registered statement (protocol/amendment2_pdx_confirmation.md, section 6): "About 25 models per drug and up to 38
reproducible drugs give a per-drug standard error of about 0.21. Power is about 75% to detect a median selective
transfer of 0.10 (the BeatAML estimate) and about 95% for 0.15." No code was registered; this script makes the
calculation explicit.

Model. Each drug d has a true selective transfer rho_d whose Fisher z is N(atanh(mu), tau^2). Its estimate is
z_d + e_d with e_d ~ N(0, SE_d^2), SE_d = 1/sqrt(n_d - 3 - k) (k = 0 reproduces the registered 0.21 for n = 25; k = 1
is the partial correlation used by the tests). Optionally, errors are equicorrelated across drugs with correlation
rho_e, because every drug is scored on the same models; rho_e is taken from the permutation null of
54_patient_derived_inference.py (mean off-diagonal correlation of per-drug estimates across label permutations).
Tests are those of the frozen scripts (scipy 1.8): Wilcoxon exact for n <= 25 and otherwise the normal approximation
without continuity correction; Mann-Whitney asymptotic with continuity correction (both groups > 8). Data are
continuous, so there are no ties or zeros. Each scenario uses 20,000 simulated studies (seeded).

Designs:
  registered        38 drugs x 25 models (SE 0.21);
  realised model    31 drugs x their observed number of models with expression (16-23);
  realised tumour   31 drugs x their observed number of distinct tumours (T1, G1: 12-16 tumours; SE 0.28-0.33).
Outputs: results/revision_transfer/power_grid.csv and 55_summary.json.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from common import RESULTS, write_json

OUT = RESULTS / 'revision_transfer'
SEED = 20261016
N_SIM = 20000
ALPHA = 0.05
MUS = [0.05, 0.10, 0.15, 0.20]
TAUS = [0.0, 0.10, 0.15, 0.20]


def wilcoxon_exact_upper(n):
    """P(W+ >= w) under H0 for w = 0..n(n+1)/2 (exact, by dynamic programming)."""
    m = n * (n + 1) // 2
    c = np.zeros(m + 1); c[0] = 1
    for k in range(1, n + 1):
        c[k:] = c[k:] + c[:-k].copy()
    p = c / c.sum()
    return p[::-1].cumsum()[::-1]


_EXACT = {}


def wilcoxon_p(x):
    """One-sided (greater) signed-rank P for each row of x (sims x n), as scipy 1.8 mode='auto' (no zeros/ties)."""
    n = x.shape[1]
    ranks = rankdata(np.abs(x), axis=1)
    w = (ranks * (x > 0)).sum(axis=1)
    if n <= 25:
        if n not in _EXACT:
            _EXACT[n] = wilcoxon_exact_upper(n)
        return _EXACT[n][np.round(w).astype(int)]
    mu = n * (n + 1) / 4; sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    return norm.sf((w - mu) / sd)


def mwu_p(a, b):
    """One-sided (a > b) Mann-Whitney P per row, asymptotic with continuity correction (scipy 1.8 for n1, n2 > 8)."""
    n1, n2 = a.shape[1], b.shape[1]
    r = rankdata(np.concatenate([a, b], axis=1), axis=1)
    u = r[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2; sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return norm.sf((u - mu - 0.5) / sd)


def simulate(rng, mu, tau, se, rho_e, n_sims=N_SIM):
    D = len(se)
    zt = rng.normal(np.arctanh(mu), tau, size=(n_sims, D)) if tau > 0 else np.full((n_sims, D), np.arctanh(mu))
    u = rng.normal(size=(n_sims, 1)); eps = rng.normal(size=(n_sims, D))
    e = (np.sqrt(rho_e) * u + np.sqrt(1 - rho_e) * eps) * se[None, :]
    return zt + e


def analytic_power(mu, tau, se, rho_e=0.0):
    """Normal approximation: Wilcoxon ~ t-test on Fisher z with asymptotic relative efficiency 3/pi; equicorrelated
    errors inflate the variance of the mean by (1 + (D - 1) rho_e) for the error part."""
    D = len(se)
    v_err = np.mean(se ** 2)
    var_mean = (tau ** 2 + v_err * (1 + (D - 1) * rho_e)) / D
    return float(norm.cdf(np.sqrt(3 / np.pi) * np.arctanh(mu) / np.sqrt(var_mean) - norm.ppf(1 - ALPHA)))


def main():
    rng = np.random.default_rng(SEED)
    per = pd.read_csv(OUT / 'pdx_tumour_level_per_drug.csv')
    s54 = json.loads((OUT / '54_summary.json').read_text())
    meta = pd.DataFrame(s54['pdx']['meta'])
    nulls = s54['pdx']['perm_null_design']
    rep = per[per.repro == 'reproducible']; non = per[per.repro == 'non_reproducible']
    tau_model = float(meta[(meta.analysis == 'registered model level') & (meta['class'] == 'reproducible')].tau.iloc[0])
    tau_tum = float(meta[(meta.analysis == 'T1 tumour level') & (meta.grouping == 'G1') & (meta['class'] == 'reproducible')].tau.iloc[0])
    pooled_tum = float(meta[(meta.analysis == 'T1 tumour level') & (meta.grouping == 'G1') & (meta['class'] == 'reproducible')].pooled_r.iloc[0])
    designs = {
        'registered (38 drugs x 25 models, SE=1/sqrt(n-3)=0.213)': (np.full(38, 1 / np.sqrt(25 - 3)), np.full(20, 1 / np.sqrt(25 - 3)), 0.0),
        'registered, partial-correlation SE (38 x 25, SE=1/sqrt(n-4)=0.218)': (np.full(38, 1 / np.sqrt(25 - 4)), np.full(20, 1 / np.sqrt(25 - 4)), 0.0),
        'realised model level (31 drugs, observed models per drug)': (1 / np.sqrt(rep.n_model.values - 4), 1 / np.sqrt(non.n_model.values - 4), 0.0),
        'realised model level, correlated errors': (1 / np.sqrt(rep.n_model.values - 4), 1 / np.sqrt(non.n_model.values - 4),
                                                    nulls['model_mean_offdiag_null_corr']),
        'realised tumour level (31 drugs, observed tumours per drug, T1 G1)': (1 / np.sqrt(rep.n_T1_G1.values - 4), 1 / np.sqrt(non.n_T1_G1.values - 4), 0.0),
        'realised tumour level, correlated errors': (1 / np.sqrt(rep.n_T1_G1.values - 4), 1 / np.sqrt(non.n_T1_G1.values - 4),
                                                     nulls['G1_mean_offdiag_null_corr']),
    }
    rows = []
    for dname, (se_a, se_b, rho_e) in designs.items():
        taus = sorted(set(TAUS + [round(tau_model, 4), round(tau_tum, 4)]))
        for tau in taus:
            for mu in MUS + [round(pooled_tum, 4)]:
                z = simulate(rng, mu, tau, se_a, rho_e)
                p8 = wilcoxon_p(np.tanh(z))
                zb = simulate(rng, 1e-12, tau, se_b, rho_e)  # non-reproducible drugs: no transfer
                p9 = mwu_p(np.tanh(z), np.tanh(zb))
                rows.append({'design': dname, 'drugs_reproducible': len(se_a), 'drugs_non_reproducible': len(se_b),
                             'mean_se': float(np.mean(se_a)), 'tau': tau, 'rho_e': rho_e, 'true_median_transfer': mu,
                             'power_H8_sim': float((p8 < ALPHA).mean()), 'power_H8_analytic': analytic_power(mu, tau, se_a, rho_e),
                             'power_H9_sim_vs_zero_transfer': float((p9 < ALPHA).mean()),
                             'median_of_observed_median': float(np.median(np.median(np.tanh(z), axis=1)))})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / 'power_grid.csv', index=False)

    def pw(design, tau, mu, col='power_H8_sim'):
        g = grid[(grid.design == design) & np.isclose(grid.tau, tau) & np.isclose(grid.true_median_transfer, mu)]
        return float(g[col].iloc[0])

    reg = 'registered (38 drugs x 25 models, SE=1/sqrt(n-3)=0.213)'
    tum = 'realised tumour level (31 drugs, observed tumours per drug, T1 G1)'
    tumc = 'realised tumour level, correlated errors'
    mod = 'realised model level (31 drugs, observed models per drug)'
    summary = {
        'registered_claim': {'0.10': 0.75, '0.15': 0.95},
        'registered_design_power_H8': {f'tau={t}': {f'{m}': pw(reg, t, m) for m in [0.10, 0.15]} for t in TAUS},
        'tau_needed_for_registered_claim': 'about 0.15 on the Fisher-z scale (75% at 0.10, 95% at 0.15); with tau = 0 power is higher',
        'realised_model_level_power_H8_tau0.15': {f'{m}': pw(mod, 0.15, m) for m in MUS},
        'realised_tumour_level_power_H8_tau0.15': {f'{m}': pw(tum, 0.15, m) for m in MUS},
        'realised_tumour_level_power_H8_tau0.15_correlated': {f'{m}': pw(tumc, 0.15, m) for m in MUS},
        'realised_tumour_level_power_H8_tau_hat': {f'{m}': pw(tum, round(tau_tum, 4), m) for m in MUS},
        'realised_tumour_level_power_H9_tau0.15': {f'{m}': pw(tum, 0.15, m, 'power_H9_sim_vs_zero_transfer') for m in MUS},
        'realised_tumour_level_power_H9_tau_hat_at_pooled': pw(tum, round(tau_tum, 4), round(pooled_tum, 4), 'power_H9_sim_vs_zero_transfer'),
        'tau_hat_model_level': tau_model, 'tau_hat_tumour_level': tau_tum, 'pooled_tumour_level_r': pooled_tum,
        'rho_e_model_level': nulls['model_mean_offdiag_null_corr'], 'rho_e_tumour_level': nulls['G1_mean_offdiag_null_corr'],
        'tumours_per_reproducible_drug': rep.n_T1_G1.value_counts().sort_index().to_dict(),
        'mean_se_tumour_level': float(np.mean(1 / np.sqrt(rep.n_T1_G1.values - 4))),
        'mean_se_model_level': float(np.mean(1 / np.sqrt(rep.n_model.values - 4)))}
    write_json(OUT / '55_summary.json', summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
