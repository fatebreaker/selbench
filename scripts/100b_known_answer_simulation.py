"""Round 4 (WS8), analysis 2: known-answer simulation of the decomposition, scored with the paper's own code.

Screens (drug d, cell line c, tissue t_c of K = 20 tissues with fixed effects tau_t, drawn once):
  a_c   = sqrt(lam) tau_{t_c} + sqrt(1 - lam) eps_c                      shared general-sensitivity axis (Var 1)
  y_dc  = l_d a_c + u_{d, t_c} + s_dc + e_dc                             measured response (higher = more sensitive)
  l_d ~ U(0.30, 0.65) (a perfect predictor's axis share r_zg^2 ~ 0.09-0.32, mean ~0.2), u_{d,t} ~ N(0, 0.08)
  (drug-specific tissue effects), s_dc ~ N(0, 0.42) (drug-specific selective signal), e_dc ~ N(0, 0.40) (noise);
  lam = 0.25 (tissue share of the axis). Panels: 150 drugs x 500 cells (full) and 30 x 100 (small).
Predictors with known composition: p_dc = alpha a_c + beta s_dc + gamma m_dc + sigma_p xi_dc, with m_dc =
  l_d sqrt(lam) tau_{t_c} + u_{d,t_c} the tissue-by-drug mean of the noise-free profile (alpha > 0, beta = gamma = 0
  is drug-agnostic; gamma only is tissue-only; beta only is selective-only).
Truth. theta_latent: A, S (and the axis shares) computed with the latent axis a in place of g at N = 20,000 cells,
  pooled over many drug panels (the predictor's true composition; super-population ratio of sums over drugs).
  theta_design: the same with the paper's leave-drug-out axis g built from the panel's D drugs (the estimand of the
  estimator for that panel size; differs from theta_latent by the leakage of an imperfect axis). Both use z =
  rank-normal(y) and the paper's selbench.rank_normal / selbench.fitness_axis / selbench.split_terms / box1.capture.
  Analytic check: without tissue effects, A and S of alpha/beta predictors have closed forms (Gaussian; latent axis),
  compared with the large-N computation.
Estimation (the paper's path): selbench.decompose on the long table; per drug 77_round2_lib.within_stats under a
  two-way bootstrap (box1.two_way_draws: cell weights inside every correlation, drug multiplicities when pooling;
  B = 500); box1.capture for A, S and the axis shares; percentile intervals (box1.interval).
Parts
  truth        theta_latent and theta_design for every predictor at D = 150 and 30; analytic check.
  recovery     (1) bias and RMSE of A, S, the model and perfect-predictor axis shares across replicates (300 per
               panel size); (2) two-way bootstrap coverage of theta_design; (3) three predictors tuned to an equal
               large-N mean raw correlation (0.30) but different composition: how often the 95% interval of the
               difference separates them on median raw, A and S; (4) drug-agnostic predictors: A > 1 when the
               alignment exceeds the loading, S ~ 0.
  reliability  (5) axis reliability (box1.axis_split_half) against leakage of the axis into S
               (theta_design - theta_latent) for D = 6 ... 300 drugs, and the leakage predicted by
               A m/(1 - m) (1 - rho)/rho (m = perfect share, rho = axis reliability).
  censoring    (6) ln IC50-like censoring of the resistant tail (40-60% of each drug's lines undefined): the tie rule
               (box1.censor_resistant, tied at the resistant end) against defined values only, versus the uncensored
               truth (bias and RMSE, 150 replicates).
  attenuation  (7) a second screen on the same cells sharing a and s with its own noise: proportional attenuation
               (per-drug noise set so that both terms shrink by k = 0.75; with and without a processing-like
               monotone distortion and a capped resistant tail), axis-specific loss (target axis a_T = 0.6 a +
               0.8 a'), selective-specific loss (s_T = 0.6 s + 0.8 s'), and a source-panel axis component. 81's
               excess test (box1.attenuation, pooled over four learned-like predictors; reject when the raw loss is
               clear and the excess's 95% interval excludes 0): type I error and power (200 replicates, B = 400);
               symmetric selective-both (77.paired_stats) and its retention for learned-like and drug-agnostic
               predictors.
               Added after the first run showed that extra drug-specific noise in the target rejects proportional
               attenuation in ~90% of replicates (noise lowers the target axis's reliability, which moves part of
               the axis term into the selective term): a reliability-adjusted excess (each screen's axis term divided
               by its axis's split-half reliability), its type I error and power, and the expected noise-only excess
               w k (1 - rho_T/rho_S)/(1 - k).
  agnostic_sweep  (4) drug-agnostic predictors a + sigma xi over sigma: A against the alignment ratio (large N).
  realdata     applies the two reliability results to the paper's benchmarks (read-only inputs): the predicted
               leakage A m/(1 - m) (1 - rho)/rho and corrected S for every model (78's capture), and 81's excess test
               with and without the reliability adjustment for every paired target of 80 (round-2 arrays, 81's pools).
  The S correction S_lat = [S (1 - m) - (1 - rho) A m / rho] / (1 - m / rho) (corrected_S) is exact when the axis's
  error is independent of the predictions and the profile; it is applied post hoc to the reliability and recovery
  outputs (fig_5_..., fig_5b_...).
Outputs: results/revision_round4/simulation/*.csv (figure-ready, one per panel), run.json; key numbers in
results/revision_round4/key_numbers_ws8.json (section 'simulation').
Usage: 100b_known_answer_simulation.py [--parts truth agnostic_sweep recovery reliability censoring attenuation realdata]
       [--reps-scale 1.0]   (default: the five simulation parts; run 'agnostic_sweep' and 'realdata' explicitly)
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)
import box1  # noqa: E402
import selbench  # noqa: E402

OUT = lib.RES / 'revision_round4' / 'simulation'
KEYS = lib.RES / 'revision_round4' / 'key_numbers_ws8.json'
WORKERS = 3
SEED = 20261201
PAR = {'K': 20, 'tissue_exponent': 0.7, 'lam': 0.25, 'l_lo': 0.30, 'l_hi': 0.65, 'var_u': 0.08, 'var_s': 0.42,
       'var_e': 0.40}
N_BIG = 20000
B_REC = 500
B_ATT = 400
TARGET_RAW = 0.30
# name: (alpha, beta, gamma, sigma_p); sigma_p None = tuned to TARGET_RAW (equal-raw set)
PRED = {
    'agnostic': (1.0, 0.0, 0.0, 0.5),
    'agnostic_weak': (1.0, 0.0, 0.0, 3.0),
    'axis_heavy': (1.0, 0.3, 0.3, 0.8),
    'balanced': (0.6, 0.6, 0.0, 0.8),
    'selective': (0.0, 1.0, 0.0, 0.8),
    'tissue_only': (0.0, 0.0, 1.0, 0.3),
    'eq_agnostic': (1.0, 0.0, 0.0, None),
    'eq_mixed': (1.0, 0.8, 0.0, None),
    'eq_selective': (0.25, 1.0, 0.0, None),
}
EQ = ['eq_agnostic', 'eq_mixed', 'eq_selective']
LEARNED_LIKE = {'L1': (1.0, 0.3, 0.2, 0.8), 'L2': (0.8, 0.5, 0.0, 0.8), 'L3': (1.0, 0.15, 0.3, 0.8),
                'L4': (0.6, 0.4, 0.2, 0.8)}
G = {}


# ----------------------------------------------------------------------------------------------------------------
# generator
def tissues():
    rng = np.random.default_rng(SEED)
    K = PAR['K']
    pi = np.arange(1, K + 1, dtype=float) ** -PAR['tissue_exponent']
    pi /= pi.sum()
    tau = rng.standard_normal(K)
    tau = (tau - (pi * tau).sum()) / np.sqrt((pi * (tau - (pi * tau).sum()) ** 2).sum())   # weighted mean 0, var 1
    return pi, tau


def draw_screen(D, N, rng, par=None):
    par = PAR if par is None else par
    pi, tau = G.setdefault('tissues', tissues())
    K = par['K']
    t = rng.choice(K, N, p=pi)
    a = np.sqrt(par['lam']) * tau[t] + np.sqrt(1 - par['lam']) * rng.standard_normal(N)
    l_ = rng.uniform(par['l_lo'], par['l_hi'], D)
    u = rng.normal(0, np.sqrt(par['var_u']), (D, K))
    s = rng.normal(0, np.sqrt(par['var_s']), (D, N))
    e = rng.normal(0, np.sqrt(par['var_e']), (D, N))
    m = l_[:, None] * np.sqrt(par['lam']) * tau[t][None, :] + u[:, t]
    y = l_[:, None] * a[None, :] + u[:, t] + s + e
    return {'t': t, 'a': a, 'l': l_, 'u': u, 's': s, 'e': e, 'm': m, 'y': y}


def predictions(sc, specs, rng):
    """{name: (D, N)} from specs {name: (alpha, beta, gamma, sigma)}; one noise draw per predictor."""
    out = {}
    D, N = sc['y'].shape
    for k, (al, be, ga, sg) in specs.items():
        out[k] = al * sc['a'][None, :] + be * sc['s'] + ga * sc['m'] + sg * rng.standard_normal((D, N))
    return out


def long_table(Y, drugs=None, cells=None):
    D, N = Y.shape
    drugs = [f'd{i}' for i in range(D)] if drugs is None else drugs
    cells = [f'c{j}' for j in range(N)] if cells is None else cells
    return pd.DataFrame({'drug': np.repeat(drugs, N), 'cell': np.tile(cells, D), 'response': Y.ravel()})


def specs_all():
    sp = dict(PRED)
    for k in EQ:
        sp[k] = (*PRED[k][:3], G['eq_sigma'][k])
    return sp


# ----------------------------------------------------------------------------------------------------------------
# large-N truth (paper's functions on matrices)
def rank_matrix(Y):
    return np.vstack([selbench.rank_normal(r) for r in Y])


def terms_against(P, Z, Gm):
    """Per drug split_terms of predictions P (D, N) against Z with axis Gm (D, N) or a latent vector (N,)."""
    rows = []
    for d in range(Z.shape[0]):
        g = Gm if Gm.ndim == 1 else Gm[d]
        ok = np.isfinite(P[d]) & np.isfinite(Z[d]) & np.isfinite(g)
        rows.append(selbench.split_terms(P[d][ok], Z[d][ok], g[ok]))
    return pd.DataFrame(rows)


def capture_point(T):
    c = box1.capture(T.raw.values, T.alignment.values, T.loading.values, sel=T.selective.values)
    c['mean_raw'] = float(np.nanmean(T.raw.values))
    c['median_raw'] = float(np.nanmedian(T.raw.values))
    c['mean_sel'] = float(np.nanmean(T.selective.values))
    return c


def truth_panel(args):
    D, seed, specs = args
    rng = np.random.default_rng(seed)
    sc = draw_screen(D, N_BIG, rng)
    P = predictions(sc, specs, rng)
    Z = rank_matrix(sc['y'])
    Gd = selbench.fitness_axis(pd.DataFrame(Z)).values
    out = {}
    for k, Pk in P.items():
        out[k] = {'design': terms_against(Pk, Z, Gd), 'latent': terms_against(Pk, Z, sc['a'])}
    return out


def truth(D, specs, panels, seed0, pool):
    res = list(pool.map(truth_panel, [(D, seed0 + i, specs) for i in range(panels)]))
    rows = []
    for k in specs:
        for kind in ('design', 'latent'):
            T = pd.concat([r[k][kind] for r in res], ignore_index=True)
            c = capture_point(T)
            rows.append({'D': D, 'predictor': k, 'axis': kind, 'drugs_pooled': len(T), **c})
    return pd.DataFrame(rows)


def eq_sigmas(pool):
    """Noise SD per equal-raw predictor so that the large-N (D = 150, design axis) mean raw = TARGET_RAW: adding
    independent noise of variance s^2 scales every per-drug correlation by sqrt(V0 / (V0 + s^2))."""
    rng = np.random.default_rng(SEED + 7)
    sc = draw_screen(150, N_BIG, rng)
    Z = rank_matrix(sc['y'])
    Gd = selbench.fitness_axis(pd.DataFrame(Z)).values
    out = {}
    for k in EQ:
        al, be, ga, _ = PRED[k]
        P0 = predictions(sc, {k: (al, be, ga, 0.0)}, rng)[k]
        T = terms_against(P0, Z, Gd)
        v0 = P0.var(axis=1)
        r0 = T.raw.values
        lo, hi = 0.0, 50.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if np.mean(r0 * np.sqrt(v0 / (v0 + mid ** 2))) > TARGET_RAW:
                lo = mid
            else:
                hi = mid
        out[k] = (lo + hi) / 2
    return out


def analytic_check(pool):
    """No tissue effects (lam = 0, var_u = 0): closed-form A and S (latent axis, Gaussian) for alpha/beta predictors,
    evaluated at the simulated drugs' loadings (conditional on the drugs) and over the loading distribution, versus
    the large-N latent computation (rank-normal z, N = N_BIG cells)."""
    par = dict(PAR, lam=0.0, var_u=0.0)
    specs = {'agnostic': PRED['agnostic'], 'balanced': PRED['balanced'], 'selective': PRED['selective'],
             'axis_heavy_no_tissue': (1.0, 0.3, 0.0, 0.8)}
    grid = np.linspace(PAR['l_lo'], PAR['l_hi'], 2001)

    def closed(ls, al, be, sg):
        vy = ls ** 2 + par['var_s'] + par['var_e']
        vp = al ** 2 + be ** 2 * par['var_s'] + sg ** 2
        rpz = (al * ls + be * par['var_s']) / np.sqrt(vp * vy)
        rpg = al / np.sqrt(vp) * np.ones_like(ls)
        rzg = ls / np.sqrt(vy)
        return (np.mean(rpg * rzg) / np.mean(rzg ** 2), (np.mean(rpz) - np.mean(rpg * rzg)) / (1 - np.mean(rzg ** 2)),
                np.mean(rzg ** 2), np.mean(rpz))
    rows = []
    rng = np.random.default_rng(SEED + 11)
    Tlat = {k: [] for k in specs}
    ls_all = []
    for i in range(4):
        sc = draw_screen(150, N_BIG, rng, par)
        ls_all.append(sc['l'])
        P = predictions(sc, specs, rng)
        Z = rank_matrix(sc['y'])
        for k in specs:
            Tlat[k].append(terms_against(P[k], Z, sc['a']))
    ls_all = np.concatenate(ls_all)
    for k, (al, be, ga, sg) in specs.items():
        A, S, m, r = closed(grid, al, be, sg)
        Ac, Sc, mc, rc = closed(ls_all, al, be, sg)
        c = capture_point(pd.concat(Tlat[k], ignore_index=True))
        rows.append({'predictor': k, 'A_analytic_population': A, 'S_analytic_population': S,
                     'perfect_share_analytic_population': m, 'A_analytic_same_drugs': Ac, 'S_analytic_same_drugs': Sc,
                     'perfect_share_analytic_same_drugs': mc, 'mean_raw_analytic_same_drugs': rc,
                     'A_largeN': c['axis_capture'], 'S_largeN': c['selective_capture'],
                     'perfect_share_largeN': c['perfect_share'], 'mean_raw_largeN': c['mean_raw']})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# estimation in one simulated screen (paper's path)
def estimate(sc, P, B, seed, Y=None):
    """selbench.decompose on the long table, 77.within_stats per drug under box1.two_way_draws, box1.capture.
    Returns {predictor: {stat: (R,) draws}} and the decomposition."""
    Y = sc['y'] if Y is None else Y
    D, N = Y.shape
    obs = long_table(Y).dropna()
    Z, Gm, _, _ = selbench.decompose(obs)
    cells = [f'c{j}' for j in range(N)]
    Zv = Z.reindex(columns=cells).values
    Gv = Gm.reindex(columns=cells).values
    di = [int(d[1:]) for d in Z.index]
    W, V = box1.two_way_draws(N, len(di), B, seed)
    names = list(P)
    R = W.shape[0]
    arr = {k: np.full((R, len(names), len(di)), np.nan) for k in ('raw', 'r_pg', 'r_zg', 'sel')}
    for j, d in enumerate(di):
        ok = np.isfinite(Zv[j]) & np.isfinite(Gv[j])
        idx = np.where(ok)[0]
        st = lib.within_stats(Zv[j][idx], Gv[j][idx], np.column_stack([P[k][d][idx] for k in names]), W[:, idx])
        for k in arr:
            arr[k][:, :, j] = st[k]
    out = {}
    for i, k in enumerate(names):
        c = box1.capture(arr['raw'][:, i], arr['r_pg'][:, i], arr['r_zg'][:, i], sel=arr['sel'][:, i], weights=V)
        out[k] = {'A': c['axis_capture'], 'S': c['selective_capture'], 'axis_share': c['axis_share'],
                  'perfect_share': c['perfect_share'], 'mean_raw': c['mean_raw'],
                  'median_raw': lib.wmedian_rows(arr['raw'][:, i], V),
                  'mean_sel': (V * np.nan_to_num(arr['sel'][:, i])).sum(1) / (V * np.isfinite(arr['sel'][:, i])).sum(1)}
    return out, Z, arr, V


def recovery_rep(args):
    D, N, seed = args
    rng = np.random.default_rng(seed)
    sc = draw_screen(D, N, rng)
    specs = specs_all()
    P = predictions(sc, specs, rng)
    est, Z, arr, V = estimate(sc, P, B_REC, seed + 1)
    rel = box1.axis_split_half(Z, n_splits=100, seed=seed)[0]
    rows = []
    for k, st in est.items():
        row = {'D': D, 'N': N, 'seed': seed, 'predictor': k, 'axis_split_half': rel}
        for s_, v in st.items():
            lo, hi = box1.interval(v)
            row.update({s_: float(v[0]), f'{s_}_lo': lo, f'{s_}_hi': hi})
        rows.append(row)
    # (3) pairwise differences among the equal-raw predictors (paired draws)
    pairs = []
    for i, a in enumerate(EQ):
        for b in EQ[i + 1:]:
            for s_ in ('median_raw', 'mean_raw', 'A', 'S'):
                d = est[a][s_] - est[b][s_]
                lo, hi = box1.interval(d)
                pairs.append({'D': D, 'N': N, 'seed': seed, 'a': a, 'b': b, 'stat': s_, 'diff': float(d[0]), 'lo': lo,
                              'hi': hi})
    return rows, pairs


# ----------------------------------------------------------------------------------------------------------------
# (5) axis reliability and leakage
def reliability_panel(args):
    D, N, seed = args
    rng = np.random.default_rng(seed)
    sc = draw_screen(D, N, rng)
    specs = {'agnostic': PRED['agnostic'], 'balanced': PRED['balanced'], 'axis_heavy': PRED['axis_heavy'],
             'selective': PRED['selective']}
    P = predictions(sc, specs, rng)
    Z = rank_matrix(sc['y'])
    Zdf = pd.DataFrame(Z)
    Gd = selbench.fitness_axis(Zdf).values
    rel_sh = box1.axis_split_half(Zdf, n_splits=50, seed=seed)[0]
    # reliability of the leave-drug-out axis as a measure of the latent axis: mean over drugs of corr(g_d, a)^2
    rel_lat = float(np.mean([np.corrcoef(Gd[d], sc['a'])[0, 1] ** 2 for d in range(D)]))
    out = {'D': D, 'N': N, 'seed': seed, 'axis_split_half': rel_sh, 'axis_latent_reliability': rel_lat, 'terms': {}}
    for k, Pk in P.items():
        out['terms'][k] = {'design': terms_against(Pk, Z, Gd), 'latent': terms_against(Pk, Z, sc['a'])}
    return out


# ----------------------------------------------------------------------------------------------------------------
# (6) censoring
def censor_rep(args):
    D, N, seed = args
    rng = np.random.default_rng(seed)
    sc = draw_screen(D, N, rng)
    specs = {k: PRED[k] for k in ('agnostic', 'axis_heavy', 'balanced', 'selective', 'tissue_only')}
    P = predictions(sc, specs, rng)
    Y = sc['y']
    frac = rng.uniform(0.4, 0.6, D)
    cens = np.zeros_like(Y, bool)
    for d in range(D):
        k = int(round(frac[d] * N))
        cens[d, np.argsort(Y[d], kind='stable')[:k]] = True          # the most resistant lines: undefined
    rows = []
    variants = {}
    variants['uncensored'] = Y
    Yd = np.where(cens, np.nan, Y)
    variants['defined_only'] = Yd
    obs = long_table(Yd)
    tied = box1.censor_resistant(obs[['drug', 'cell', 'response']])     # NaN = undefined -> tied at resistant end
    variants['tie_rule'] = tied.pivot(index='drug', columns='cell', values='response').reindex(
        index=[f'd{i}' for i in range(D)], columns=[f'c{j}' for j in range(N)]).values
    for v, Yv in variants.items():
        est, Z, arr, V = estimate(sc, P, 0, seed + 3, Y=Yv)
        for k, st in est.items():
            rows.append({'D': D, 'N': N, 'seed': seed, 'variant': v, 'predictor': k,
                         'censored_fraction': float(cens.mean()), **{s_: float(x[0]) for s_, x in st.items()}})
    return rows


# ----------------------------------------------------------------------------------------------------------------
# (7) two screens: attenuation null, axis- and selective-specific losses, selective-both
SCEN = ['identical_noise', 'attenuation', 'attenuation_distorted', 'axis_loss', 'selective_loss', 'source_panel_axis']


def second_screen(sc, scen, rng, k_att=0.75, rho=0.6):
    """Target-screen responses on the same cells (and the source responses, which differ only in
    'source_panel_axis')."""
    D, N = sc['y'].shape
    pi, tau = G['tissues']
    l_, u, s, a, t = sc['l'], sc['u'], sc['s'], sc['a'], sc['t']
    ys = sc['y']
    var_sig = l_ ** 2 * 1.0 + PAR['var_u'] + PAR['var_s']                 # expected signal variance per drug
    e_t = rng.normal(0, np.sqrt(PAR['var_e']), (D, N))
    if scen == 'identical_noise':
        yt = l_[:, None] * a + u[:, t] + s + e_t
    elif scen in ('attenuation', 'attenuation_distorted'):
        # Var_T = Var_S / k^2 with the signal unchanged: noise variance var_e_T,d = (var_sig + var_e) / k^2 - var_sig
        ve = (var_sig + PAR['var_e']) / k_att ** 2 - var_sig
        yt = l_[:, None] * a + u[:, t] + s + rng.standard_normal((D, N)) * np.sqrt(ve)[:, None]
        if scen == 'attenuation_distorted':
            # processing-like monotone distortion (drug-specific logistic squashing) and a capped resistant tail
            c0 = rng.normal(0, 0.5, D)[:, None]
            sl = rng.uniform(0.5, 2.0, D)[:, None]
            yt = 1 / (1 + np.exp(-sl * (yt - c0)))
            cap = np.quantile(yt, 0.2, axis=1, keepdims=True)
            yt = np.maximum(yt, cap)                     # lowest 20% tied (e.g. viability capped at 1)
    elif scen == 'axis_loss':
        a_t = rho * a + np.sqrt(1 - rho ** 2) * rng.standard_normal(N)
        yt = l_[:, None] * a_t + u[:, t] + s + e_t
    elif scen == 'selective_loss':
        s_t = rho * s + np.sqrt(1 - rho ** 2) * rng.normal(0, np.sqrt(PAR['var_s']), (D, N))
        yt = l_[:, None] * a + u[:, t] + s_t + e_t
    elif scen == 'source_panel_axis':
        # a second cell factor b loaded by every source drug (k_d) but by only 20% of target drugs: the source axis
        # carries b; drug d's own response loads on b in both screens only if it is a b-drug
        b = rng.standard_normal(N)
        kd = rng.uniform(0.25, 0.45, D)
        bdrug = rng.random(D) < 0.2
        ys = ys + kd[:, None] * b
        yt = l_[:, None] * a + u[:, t] + s + e_t + (kd * bdrug)[:, None] * b
        sc = dict(sc, y=ys, b=b, bdrug=bdrug)
    else:
        raise ValueError(scen)
    return ys, yt, sc


def attenuation_rep(args):
    scen, D, N, seed = args
    rng = np.random.default_rng(seed)
    sc = draw_screen(D, N, rng)
    ys, yt, sc2 = second_screen(sc, scen, rng)
    specs = dict(LEARNED_LIKE)
    specs['agnostic'] = PRED['agnostic']
    specs['selective'] = PRED['selective']
    P = predictions(sc2, specs, rng)
    if scen == 'source_panel_axis':
        # an exactly drug-agnostic predictor aligned with the source axis (a + b): the case selective-both targets
        P['agnostic_source_axis'] = (sc2['a'] + 0.6 * sc2['b'])[None, :] + 0.5 * rng.standard_normal((D, N))
    Zs, Gs, _, _ = selbench.decompose(long_table(ys))
    Zt, Gt, _, _ = selbench.decompose(long_table(yt))
    cells = [f'c{j}' for j in range(N)]
    W, V = box1.two_way_draws(N, D, B_ATT, seed + 5)
    names = list(P)
    R = W.shape[0]
    keys = ['raw_w', 'r_pg_w', 'r_zg_w', 'selT_w', 'selB_w', 'raw_c', 'r_pg_c', 'r_zg_c', 'selT_c', 'selB_c']
    arr = {k: np.full((R, len(names), D), np.nan) for k in keys}
    drugs = [d for d in Zs.index if d in Zt.index]
    for d in drugs:
        j = int(d[1:])
        zs, gs = Zs.loc[d, cells].values, Gs.loc[d, cells].values
        zt, gt = Zt.loc[d, cells].values, Gt.loc[d, cells].values
        idx = np.where(np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt))[0]
        st = lib.paired_stats(zs[idx], gs[idx], zt[idx], gt[idx], np.column_stack([P[k][j][idx] for k in names]),
                              W[:, idx])
        for k in keys:
            arr[k][:, :, j] = st[k]
    rows = []
    learned = list(LEARNED_LIKE)
    li = [names.index(k) for k in learned]
    cat = lambda k: np.concatenate([arr[k][:, i] for i in li], axis=1)  # noqa: E731
    Vc = np.concatenate([V] * len(li), axis=1)
    att = box1.attenuation(cat('raw_w'), cat('r_pg_w'), cat('r_zg_w'), cat('raw_c'), cat('r_pg_c'), cat('r_zg_c'),
                           weights=Vc)
    row = {'scenario': scen, 'D': D, 'N': N, 'seed': seed, 'group': 'learned-like (4)'}
    for k_, v in att.items():
        lo, hi = box1.interval(v)
        row.update({k_: float(v[0]), f'{k_}_lo': lo, f'{k_}_hi': hi})
    row['loss_clear'] = bool(row['raw_loss_lo'] > 0)
    row['excess_excludes_0'] = bool(row['excess_lo'] > 0 or row['excess_hi'] < 0)
    row['reject_81'] = bool(row['loss_clear'] and row['excess_excludes_0'])
    row['reject_81_positive'] = bool(row['reject_81'] and row['excess'] > 0)
    row['reject_81_negative'] = bool(row['reject_81'] and row['excess'] < 0)
    row['retdiff_excludes_0'] = bool(row['retention_difference_lo'] > 0 or row['retention_difference_hi'] < 0)
    # reliability-adjusted excess: each screen's axis term divided by its axis's split-half reliability rho (the axis
    # term of a perfectly reliable axis; noise that only lowers a screen's axis reliability then leaves it unchanged)
    rho_s = box1.axis_split_half(Zs, n_splits=50, seed=seed)[0]
    rho_t = box1.axis_split_half(Zt, n_splits=50, seed=seed + 1)[0]
    att2 = box1.attenuation(cat('raw_w'), cat('r_pg_w'), cat('r_zg_w') / rho_s, cat('raw_c'), cat('r_pg_c'),
                            cat('r_zg_c') / rho_t, weights=Vc)
    row['rho_source'], row['rho_target'] = rho_s, rho_t
    for k_ in ('excess', 'within_share', 'share_of_loss', 'axis_retained', 'selterm_retained'):
        lo, hi = box1.interval(att2[k_])
        row.update({f'adj_{k_}': float(att2[k_][0]), f'adj_{k_}_lo': lo, f'adj_{k_}_hi': hi})
    row['adj_excess_excludes_0'] = bool(row['adj_excess_lo'] > 0 or row['adj_excess_hi'] < 0)
    row['adj_reject_81'] = bool(row['loss_clear'] and row['adj_excess_excludes_0'])
    row['adj_reject_81_positive'] = bool(row['adj_reject_81'] and row['adj_excess'] > 0)
    row['adj_reject_81_negative'] = bool(row['adj_reject_81'] and row['adj_excess'] < 0)
    row['expected_noise_excess'] = float(row['within_share'] * (1 - row['raw_loss'] / row['raw_mean_within']) *
                                         (1 - rho_t / rho_s) / (row['raw_loss'] / row['raw_mean_within']))
    # selective-both (symmetric) and own-axis selective scores, per predictor and pooled learned-like
    for gname, idxs in [('learned-like (4)', li)] + [(k, [names.index(k)]) for k in names if k not in learned]:
        def pooled_mean(k):
            x = np.concatenate([arr[k][:, i] for i in idxs], axis=1)
            vv = np.concatenate([V] * len(idxs), axis=1)
            ok = np.isfinite(x)
            return (vv * np.where(ok, x, 0)).sum(1) / (vv * ok).sum(1)
        sw, sc_ = pooled_mean('selB_w'), pooled_mean('selB_c')
        tw, tc = pooled_mean('selT_w'), pooled_mean('selT_c')
        rw, rc = pooled_mean('raw_w'), pooled_mean('raw_c')
        for nm, v in [('selB_within', sw), ('selB_cross', sc_), ('selB_retained', sc_ / sw), ('selT_within', tw),
                      ('selT_cross', tc), ('raw_within', rw), ('raw_cross', rc), ('raw_retained', rc / rw)]:
            lo, hi = box1.interval(v)
            row_k = f'{gname}|{nm}'
            row[row_k] = float(v[0])
            row[f'{row_k}_lo'], row[f'{row_k}_hi'] = lo, hi
    if scen == 'source_panel_axis':
        bd = sc2['bdrug']
        i = names.index('agnostic_source_axis')
        for lab, msk in [('b_drugs', bd), ('other_drugs', ~bd)]:
            for k in ('selT_c', 'selB_c'):
                row[f'agnostic_source_axis|{k}|{lab}'] = float(np.nanmean(arr[k][0, i][msk]))
    rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------------------------
def summ_recovery(Rr, Tt):
    """Bias, RMSE and coverage against theta_design (and bias against theta_latent)."""
    rows = []
    for (D, k), g in Rr.groupby(['D', 'predictor']):
        td = Tt[(Tt.D == D) & (Tt.predictor == k) & (Tt.axis == 'design')].iloc[0]
        tl = Tt[(Tt.D == D) & (Tt.predictor == k) & (Tt.axis == 'latent')].iloc[0]
        for s_, tcol in [('A', 'axis_capture'), ('S', 'selective_capture'), ('axis_share', 'axis_share'),
                         ('perfect_share', 'perfect_share'), ('mean_raw', 'mean_raw'), ('median_raw', 'median_raw')]:
            x = g[s_].values
            th, tl_ = float(td[tcol]), float(tl[tcol])
            cov = np.mean((g[f'{s_}_lo'] <= th) & (g[f'{s_}_hi'] >= th))
            covl = np.mean((g[f'{s_}_lo'] <= tl_) & (g[f'{s_}_hi'] >= tl_))
            n = len(x)
            rows.append({'D': D, 'N': int(g.N.iloc[0]), 'predictor': k, 'stat': s_, 'replicates': n,
                         'truth_design': th, 'truth_latent': tl_, 'mean_estimate': float(np.mean(x)),
                         'bias_design': float(np.mean(x) - th), 'bias_design_mcse': float(np.std(x, ddof=1) / np.sqrt(n)),
                         'rmse_design': float(np.sqrt(np.mean((x - th) ** 2))), 'bias_latent': float(np.mean(x) - tl_),
                         'rmse_latent': float(np.sqrt(np.mean((x - tl_) ** 2))), 'sd_estimate': float(np.std(x, ddof=1)),
                         'coverage_design': float(cov), 'coverage_mcse': float(np.sqrt(cov * (1 - cov) / n)),
                         'coverage_latent': float(covl), 'mean_ci_width': float(np.mean(g[f'{s_}_hi'] - g[f'{s_}_lo'])),
                         'mean_axis_split_half': float(g.axis_split_half.mean())})
    return pd.DataFrame(rows)


def run(parts, scale, t0):
    OUT.mkdir(parents=True, exist_ok=True)
    G['tissues'] = tissues()
    eqf = OUT / 'eq_sigma.json'
    if eqf.exists():
        G['eq_sigma'] = json.loads(eqf.read_text())
    else:
        G['eq_sigma'] = eq_sigmas(None)
        lib.write_json(eqf, G['eq_sigma'])
    print('eq sigmas', G['eq_sigma'], f'{time.time() - t0:.0f}s', flush=True)
    pool = ProcessPoolExecutor(WORKERS, mp_context=mp.get_context('fork'))    # forked after G is complete
    try:
        specs = specs_all()
        tf = OUT / 'truth.csv'
        if 'truth' in parts or not tf.exists():
            T = pd.concat([truth(150, specs, 12, SEED + 1000, pool), truth(30, specs, 60, SEED + 2000, pool)],
                          ignore_index=True)
            T.to_csv(tf, index=False, float_format='%.6g')
            Ac = analytic_check(pool)
            Ac.to_csv(OUT / 'analytic_check.csv', index=False, float_format='%.6g')
            print('truth done', f'{time.time() - t0:.0f}s', flush=True)
        T = pd.read_csv(tf)
        if 'agnostic_sweep' in parts:
            # (4) drug-agnostic predictors p = a + sigma xi: A against the alignment ratio (mean r_pg / mean r_zg)
            sig = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
            sp = {f'agnostic_sigma{x:g}': (1.0, 0.0, 0.0, x) for x in sig}
            Tsw = truth(150, sp, 6, SEED + 3000, pool)
            Tsw['sigma'] = Tsw.predictor.str.replace('agnostic_sigma', '').astype(float)
            Tsw.to_csv(OUT / 'fig_4_agnostic_A_sweep.csv', index=False, float_format='%.5g')
            print('agnostic sweep done', f'{time.time() - t0:.0f}s', flush=True)
        if 'recovery' in parts:
            nrep = max(20, int(300 * scale))
            tasks = [(150, 500, SEED + 10000 + i) for i in range(nrep)] + [(30, 100, SEED + 20000 + i) for i in range(nrep)]
            rows, prs = [], []
            for r, p in pool.map(recovery_rep, tasks, chunksize=2):
                rows += r
                prs += p
            Rr = pd.DataFrame(rows)
            Rr.to_csv(OUT / 'recovery_replicates.csv.gz', index=False, float_format='%.6g')
            Pr = pd.DataFrame(prs)
            Pr.to_csv(OUT / 'equal_raw_pairs_replicates.csv.gz', index=False, float_format='%.6g')
            Sm = summ_recovery(Rr, T)
            Sm.to_csv(OUT / 'fig_1_2_recovery_coverage.csv', index=False, float_format='%.5g')
            # (3) separation of equal-raw predictors
            sep = Pr.assign(excl=(Pr.lo > 0) | (Pr.hi < 0)).groupby(['D', 'a', 'b', 'stat']).agg(
                replicates=('diff', 'size'), mean_diff=('diff', 'mean'), sd_diff=('diff', 'std'),
                share_interval_excludes_0=('excl', 'mean'), share_positive=('diff', lambda v: float((v > 0).mean()))).reset_index()
            sep.to_csv(OUT / 'fig_3_equal_raw_separation.csv', index=False, float_format='%.5g')
            print('recovery done', f'{time.time() - t0:.0f}s', flush=True)
        if 'reliability' in parts:
            Ds = [6, 8, 10, 15, 20, 30, 50, 100, 150, 300]
            tasks = []
            for D in Ds:
                npan = max(2, int(np.ceil(1500 / D)))
                npan = min(npan, 60)
                tasks += [(D, 4000, SEED + 30000 + 100 * D + i) for i in range(npan)]
            res = list(pool.map(reliability_panel, tasks, chunksize=1))
            rows = []
            for D in Ds:
                rr = [r for r in res if r['D'] == D]
                rel = float(np.mean([r['axis_split_half'] for r in rr]))
                rell = float(np.mean([r['axis_latent_reliability'] for r in rr]))
                for k in rr[0]['terms']:
                    cd = capture_point(pd.concat([r['terms'][k]['design'] for r in rr], ignore_index=True))
                    cl = capture_point(pd.concat([r['terms'][k]['latent'] for r in rr], ignore_index=True))
                    m = cd['perfect_share']
                    pred_leak = cd['axis_capture'] * m / (1 - m) * (1 - rel) / rel
                    pred_leak_l = cd['axis_capture'] * m / (1 - m) * (1 - rell) / rell
                    rows.append({'D': D, 'panels': len(rr), 'cells': 4000, 'axis_split_half': rel,
                                 'axis_latent_reliability': rell, 'predictor': k,
                                 'S_design': cd['selective_capture'], 'S_latent': cl['selective_capture'],
                                 'leakage': cd['selective_capture'] - cl['selective_capture'],
                                 'predicted_leakage_split_half': pred_leak, 'predicted_leakage_latent_rel': pred_leak_l,
                                 'A_design': cd['axis_capture'], 'A_latent': cl['axis_capture'],
                                 'perfect_share_design': m, 'perfect_share_latent': cl['perfect_share'],
                                 'mean_sel_design': cd['mean_sel'], 'mean_sel_latent': cl['mean_sel']})
            Rl = pd.DataFrame(rows)
            Rl.to_csv(OUT / 'fig_5_axis_reliability_leakage.csv', index=False, float_format='%.5g')
            print('reliability done', f'{time.time() - t0:.0f}s', flush=True)
        if 'censoring' in parts:
            nrep = max(20, int(150 * scale))
            rows = []
            for r in pool.map(censor_rep, [(150, 500, SEED + 40000 + i) for i in range(nrep)], chunksize=2):
                rows += r
            Cr = pd.DataFrame(rows)
            Cr.to_csv(OUT / 'censoring_replicates.csv.gz', index=False, float_format='%.6g')
            srows = []
            for (v, k), g in Cr.groupby(['variant', 'predictor']):
                td = T[(T.D == 150) & (T.predictor == k) & (T.axis == 'design')].iloc[0]
                for s_, tcol in [('A', 'axis_capture'), ('S', 'selective_capture'), ('axis_share', 'axis_share'),
                                 ('perfect_share', 'perfect_share'), ('mean_raw', 'mean_raw'), ('mean_sel', 'mean_sel')]:
                    x = g[s_].values
                    unc = Cr[(Cr.variant == 'uncensored') & (Cr.predictor == k)].set_index('seed')[s_]
                    paired = g.set_index('seed')[s_] - unc.reindex(g.seed).values
                    srows.append({'variant': v, 'predictor': k, 'stat': s_, 'replicates': len(x),
                                  'truth_uncensored_design': float(td[tcol]), 'mean_estimate': float(np.mean(x)),
                                  'bias': float(np.mean(x) - td[tcol]), 'rmse': float(np.sqrt(np.mean((x - td[tcol]) ** 2))),
                                  'mean_change_vs_uncensored_same_screen': float(paired.mean()),
                                  'sd_change_vs_uncensored': float(paired.std())})
            Cs = pd.DataFrame(srows)
            Cs.to_csv(OUT / 'fig_6_censoring.csv', index=False, float_format='%.5g')
            print('censoring done', f'{time.time() - t0:.0f}s', flush=True)
        if 'attenuation' in parts:
            nrep = max(20, int(200 * scale))
            tasks = [(sc_, 150, 500, SEED + 50000 + 1000 * i + j) for i, sc_ in enumerate(SCEN) for j in range(nrep)]
            rows = []
            for r in pool.map(attenuation_rep, tasks, chunksize=2):
                rows += r
            Ar = pd.DataFrame(rows)
            Ar.to_csv(OUT / 'attenuation_replicates.csv.gz', index=False, float_format='%.6g')
            srows = []
            for scn, g in Ar.groupby('scenario'):
                row = {'scenario': scn, 'replicates': len(g), 'loss_clear_rate': g.loss_clear.mean(),
                       'excess_excludes_0_rate': g.excess_excludes_0.mean(), 'reject_81_rate': g.reject_81.mean(),
                       'reject_81_positive_rate': g.reject_81_positive.mean(),
                       'reject_81_negative_rate': g.reject_81_negative.mean(),
                       'retdiff_excludes_0_rate': g.retdiff_excludes_0.mean(), 'mean_excess': g.excess.mean(),
                       'sd_excess': g.excess.std(), 'mean_within_share': g.within_share.mean(),
                       'mean_share_of_loss': g.share_of_loss.median(), 'mean_raw_loss': g.raw_loss.mean(),
                       'mean_axis_retained': g.axis_retained.mean(), 'mean_selterm_retained': g.selterm_retained.mean(),
                       'mean_rho_source': g.rho_source.mean(), 'mean_rho_target': g.rho_target.mean(),
                       'adj_reject_81_rate': g.adj_reject_81.mean(), 'adj_reject_81_positive_rate': g.adj_reject_81_positive.mean(),
                       'adj_reject_81_negative_rate': g.adj_reject_81_negative.mean(), 'mean_adj_excess': g.adj_excess.mean(),
                       'sd_adj_excess': g.adj_excess.std(), 'mean_expected_noise_excess': g.expected_noise_excess.median()}
                for c in g.columns:
                    if '|' in c and not c.endswith(('_lo', '_hi')):
                        row[f'mean:{c}'] = g[c].mean()
                srows.append(row)
            As = pd.DataFrame(srows)
            As.to_csv(OUT / 'fig_7_attenuation_selboth.csv', index=False, float_format='%.5g')
            print('attenuation done', f'{time.time() - t0:.0f}s', flush=True)
    finally:
        pool.shutdown()


# ----------------------------------------------------------------------------------------------------------------
# reliability correction of S (derived from the simulation's leakage; exact when the axis's error is independent of
# the predictions and of the profile, as for a leave-drug-out mean of independent drug errors)
def corrected_S(S, A, m, rho):
    """With an axis of reliability rho, r_zg and r_pg shrink by sqrt(rho): A is unchanged, the perfect-predictor share
    becomes m = rho m_lat and the selective term gains (1 - rho) r_pa r_za. Inverting:
      S_lat = [S (1 - m) - (1 - rho) A m / rho] / (1 - m / rho).
    S, A, m: the paper's estimates; rho: the axis's split-half reliability (box1.axis_split_half)."""
    S, A, m, rho = (np.asarray(x, float) for x in (S, A, m, rho))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (S * (1 - m) - (1 - rho) * A * m / rho) / (1 - m / rho)


def correction_tables():
    out = {}
    f = OUT / 'fig_5_axis_reliability_leakage.csv'
    if f.exists():
        R = pd.read_csv(f)
        R['S_corrected_split_half'] = corrected_S(R.S_design, R.A_design, R.perfect_share_design, R.axis_split_half)
        R['S_corrected_error'] = R.S_corrected_split_half - R.S_latent
        R.to_csv(f, index=False, float_format='%.5g')
        out['reliability'] = R
    f = OUT / 'recovery_replicates.csv.gz'
    if f.exists():
        Rr = pd.read_csv(f)
        T = pd.read_csv(OUT / 'truth.csv')
        Rr['S_corrected'] = corrected_S(Rr.S, Rr.A, Rr.perfect_share, Rr.axis_split_half)
        rows = []
        for (D, k), g in Rr.groupby(['D', 'predictor']):
            tl = float(T[(T.D == D) & (T.predictor == k) & (T.axis == 'latent')].selective_capture.iloc[0])
            td = float(T[(T.D == D) & (T.predictor == k) & (T.axis == 'design')].selective_capture.iloc[0])
            for lab, col in [('uncorrected', 'S'), ('corrected', 'S_corrected')]:
                x = g[col].values
                rows.append({'D': D, 'predictor': k, 'estimate': lab, 'truth_latent': tl, 'truth_design': td,
                             'mean': float(np.mean(x)), 'bias_vs_latent': float(np.mean(x) - tl),
                             'rmse_vs_latent': float(np.sqrt(np.mean((x - tl) ** 2))),
                             'mean_axis_split_half': float(g.axis_split_half.mean())})
        C = pd.DataFrame(rows)
        C.to_csv(OUT / 'fig_5b_S_reliability_correction.csv', index=False, float_format='%.5g')
        out['recovery'] = C
    return out


def realdata_leakage():
    """The correction applied to the paper's benchmarks: split-half reliability of each within-dataset axis, and for
    every model the observed S, the leakage predicted by A m/(1 - m) (1 - rho)/rho and the corrected S (78's capture
    and the axis of the same decomposition)."""
    cap = pd.read_csv(lib.OUT2 / 'capture' / 'capture_by_model.csv')
    rels = {}
    L = lib.leaderboard_setup()
    Z = L['dec'][L['cd'].REFERENCE][0].reindex(columns=L['U'])
    rels[('leaderboard', 'CTRPv2_lnIC50')] = box1.axis_split_half(Z, n_splits=200, seed=1)
    for setting in ['GDSC2_AUC', 'GDSC2_LNIC50']:
        S_ = lib.retrain_setup(setting)
        O = S_['obs'][S_['obs'].cell.isin(S_['cells'])][['drug', 'cell', 'response']]
        Zr, _, _, _ = selbench.decompose(O)
        rels[('retrain', setting)] = box1.axis_split_half(Zr, n_splits=200, seed=1)
    I = lib.improve_setup(replicate_free=True, need_cross=False)
    for ds in I['datasets']:
        rels[('improve', f'{ds}_AUC')] = box1.axis_split_half(I['dec'][ds][0], n_splits=200, seed=1)
    rows = []
    for (b, s_), (rho, lo, hi) in rels.items():
        c = cap[(cap.benchmark == b) & (cap.setting == s_)]
        for r in c.itertuples():
            m = r.perfect_share
            rows.append({'benchmark': b, 'setting': s_, 'model': r.model, 'drug_agnostic': r.drug_agnostic,
                         'axis_split_half': rho, 'axis_split_half_lo': lo, 'axis_split_half_hi': hi,
                         'A': r.axis_capture, 'S': r.selective_capture, 'perfect_share': m,
                         'predicted_leakage': r.axis_capture * m / (1 - m) * (1 - rho) / rho,
                         'S_corrected': float(corrected_S(r.selective_capture, r.axis_capture, m, rho))})
    D = pd.DataFrame(rows)
    D.to_csv(OUT / 'realdata_leakage_correction.csv', index=False, float_format='%.5g')
    return D


def realdata_attenuation_adjusted():
    """81's excess test with each screen's axis term divided by its axis's split-half reliability (the adjustment the
    simulation calls for: noise that only lowers a screen's axis reliability inflates the unadjusted excess by about
    w k (1 - rho_T / rho_S) / (1 - k)). Reads the round-2 paired arrays (read-only; 81's model pools and draws);
    reliabilities are point estimates on each version's own decomposition (cells of the paired design)."""
    ix = pd.read_csv(lib.OUT2 / 'paired' / 'pairs.csv')
    ix = ix[(ix.source != ix.target) & (ix.drugs_used > 0)]
    rel = {}

    def sh(Z, key):
        if key not in rel:
            rel[key] = box1.axis_split_half(Z, n_splits=100, seed=7)[0]
        return rel[key]
    # leaderboard versions
    lbv = sorted(set(ix[ix.benchmark == 'leaderboard'].source) | set(ix[ix.benchmark == 'leaderboard'].target))
    L = lib.leaderboard_setup(need_versions=lbv, harmonized=True)
    for v in lbv:
        Z = L['dec'][v][0]
        sh(Z.reindex(columns=Z.columns.intersection(pd.Index(L['U']))), ('leaderboard', v))
    del L
    # retraining: 80's source decomposition (all cells) and each target with the same endpoint
    rt = lib.load_module('rt59', '59_retrain_score.py')
    resp = rt.responses()
    for setting in ['GDSC2_AUC', 'GDSC2_LNIC50', 'CTRPv2_LNIC50', 'CTRPv2_AUC']:   # CTRPv2_AUC: complete since round 4
        ds, ep = rt.SETTINGS[setting]
        for d_ in rt.DATASETS:
            o = resp[(resp.dataset == d_) & (resp.endpoint == ep)][['drug', 'cell', 'response']]
            if len(o):
                sh(selbench.decompose(o)[0], ('retrain', f'{d_}|{ep}'))
    # IMPROVE: own panel (all drugs) and shared panel (the drugs shared by the pair)
    I = lib.improve_setup(replicate_free=True, need_cross=False)
    for d_ in I['datasets']:
        sh(I['dec'][d_][0], ('improve', d_, 'own'))
    labels = {'leaderboard': {'main': 'published 21', 'alt': 'published 21 minus drug-agnostic (18)'},
              'retrain': {'main': 'DrEval learned (retrained) or published leaderboard', 'alt': 'per-drug ridges'},
              'improve': {'main': 'IMPROVE 7 models'}}
    rows = []
    for _, r in ix.iterrows():
        f = lib.OUT2 / 'paired' / 'arrays' / f'{r.benchmark}__{r.k}.npz'
        if not f.exists():
            continue
        if r.benchmark == 'leaderboard':
            rs, rtg = rel[('leaderboard', r.source)], rel[('leaderboard', r.target)]
        elif r.benchmark == 'retrain':
            setting = r.pair.split('->')[0]
            ds, ep = rt.SETTINGS[setting]
            rs, rtg = rel[('retrain', f'{ds}|{ep}')], rel[('retrain', f'{r.target.split("|")[0]}|{ep}')]
        else:
            src, trg = r.source, r.target
            if r.family == 'shared_panel':
                shared = I['dec'][src][0].index.intersection(I['dec'][trg][0].index)
                rs = sh(I['dec'][src][0].loc[shared], ('improve', src, trg, 'shared_s'))
                rtg = sh(I['dec'][trg][0].loc[shared], ('improve', src, trg, 'shared_t'))
            else:
                rs, rtg = rel[('improve', src, 'own')], rel[('improve', trg, 'own')]
        models, _, S = lib.load_paired(f)
        groups = {}
        if isinstance(r.pool_main, str) and r.pool_main:
            groups['main'] = r.pool_main.split(';')
        if isinstance(r.pool_alt, str) and r.pool_alt:
            groups['alt'] = r.pool_alt.split(';')
        for gk, gm in groups.items():
            gm = [m for m in gm if m in models]
            if not gm:
                continue
            g = lambda st, kind: lib.pooled(S, models, gm, st, kind)  # noqa: E731
            rw, rc, n = g('raw', 'sum_w'), g('raw', 'sum_c'), g('raw', 'n')
            mw, mc = g('mediated', 'sum_w'), g('mediated', 'sum_c')
            out = {}
            with np.errstate(invalid='ignore', divide='ignore'):
                for lab, a, b in [('', mw, mc), ('adj_', mw / rs, mc / rtg)]:
                    s_ = np.where(np.abs(rw - rc) > 1e-9, (a - b) / (rw - rc), np.nan)
                    out[f'{lab}excess'] = s_ - a / rw
                    out[f'{lab}within_share'] = a / rw
                out['raw_loss'] = (rw - rc) / n
            row = {'benchmark': r.benchmark, 'pair': r.pair, 'family': r.family,
                   'group': labels[r.benchmark].get(gk, gk), 'rho_source': rs, 'rho_target': rtg}
            for k_, v in out.items():
                lo, hi = box1.interval(v)
                row.update({k_: float(v[0]), f'{k_}_lo': lo, f'{k_}_hi': hi})
            kk = 1 - row['raw_loss'] / (rw[0] / n[0])
            row['expected_noise_excess'] = float(row['within_share'] * kk * (1 - rtg / rs) / (1 - kk)) if kk < 1 else np.nan
            row['loss_clear'] = bool(row['raw_loss_lo'] > 0)
            for lab in ('', 'adj_'):
                rej = row['loss_clear'] and (row[f'{lab}excess_lo'] > 0 or row[f'{lab}excess_hi'] < 0)
                row[f'{lab}reading'] = ('axis lost disproportionately' if rej and row[f'{lab}excess'] > 0 else
                                        'selective term lost disproportionately' if rej else
                                        'clear loss, proportional' if row['loss_clear'] else 'no clear loss')
            row['reading_changed'] = row['reading'] != row['adj_reading']
            rows.append(row)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / 'realdata_attenuation_reliability_adjusted.csv', index=False, float_format='%.5g')
    pd.DataFrame([{'key': '|'.join(map(str, k)), 'split_half': v} for k, v in rel.items()]).to_csv(
        OUT / 'realdata_axis_reliabilities.csv', index=False, float_format='%.5g')
    return D


def collect_keys():
    kn = {}
    f = OUT / 'truth.csv'
    if f.exists():
        T = pd.read_csv(f)
        kn['truth'] = {f'{r.predictor}|D{r.D}|{r.axis}': {'A': r.axis_capture, 'S': r.selective_capture,
                                                           'axis_share': r.axis_share, 'perfect_share': r.perfect_share,
                                                           'mean_raw': r.mean_raw} for r in T.itertuples()}
    f = OUT / 'analytic_check.csv'
    if f.exists():
        kn['analytic_check'] = pd.read_csv(f).to_dict(orient='records')
    f = OUT / 'fig_1_2_recovery_coverage.csv'
    if f.exists():
        S = pd.read_csv(f)
        kn['recovery'] = {f'{r.predictor}|D{r.D}|{r.stat}': {'truth_design': r.truth_design, 'truth_latent': r.truth_latent,
                                                             'bias_design': r.bias_design, 'rmse_design': r.rmse_design,
                                                             'bias_latent': r.bias_latent, 'coverage': r.coverage_design,
                                                             'coverage_mcse': r.coverage_mcse}
                          for r in S.itertuples() if r.stat in ('A', 'S', 'axis_share', 'perfect_share', 'median_raw')}
        for D in (150, 30):
            for s_ in ('A', 'S', 'axis_share', 'perfect_share'):
                x = S[(S.D == D) & (S.stat == s_)]
                kn[f'coverage_range|D{D}|{s_}'] = [float(x.coverage_design.min()), float(x.coverage_design.max())]
                kn[f'abs_bias_max|D{D}|{s_}'] = float(x.bias_design.abs().max())
                kn[f'rmse_range|D{D}|{s_}'] = [float(x.rmse_design.min()), float(x.rmse_design.max())]
    f = OUT / 'fig_4_agnostic_A_sweep.csv'
    if f.exists():
        kn['agnostic_sweep'] = pd.read_csv(f)[['sigma', 'axis', 'axis_capture', 'selective_capture', 'alignment_ratio',
                                               'mean_raw']].to_dict(orient='records')
    f = OUT / 'fig_3_equal_raw_separation.csv'
    if f.exists():
        kn['equal_raw_separation'] = pd.read_csv(f).to_dict(orient='records')
    f = OUT / 'fig_5_axis_reliability_leakage.csv'
    if f.exists():
        kn['reliability_leakage'] = pd.read_csv(f).to_dict(orient='records')
    f = OUT / 'fig_6_censoring.csv'
    if f.exists():
        kn['censoring'] = pd.read_csv(f).to_dict(orient='records')
    f = OUT / 'fig_7_attenuation_selboth.csv'
    if f.exists():
        kn['attenuation'] = pd.read_csv(f).to_dict(orient='records')
    f = OUT / 'fig_5b_S_reliability_correction.csv'
    if f.exists():
        kn['S_reliability_correction'] = pd.read_csv(f).to_dict(orient='records')
    f = OUT / 'realdata_attenuation_reliability_adjusted.csv'
    if f.exists():
        D = pd.read_csv(f)
        kn['realdata_attenuation_adjusted'] = {
            'pairs_groups': int(len(D)), 'readings_changed': int(D.reading_changed.sum()),
            'changed': D[D.reading_changed][['benchmark', 'pair', 'group', 'excess', 'excess_lo', 'excess_hi', 'adj_excess',
                                             'adj_excess_lo', 'adj_excess_hi', 'reading', 'adj_reading']].to_dict(orient='records'),
            'expected_noise_excess_range': [float(D.expected_noise_excess.min()), float(D.expected_noise_excess.max())]}
    f = OUT / 'realdata_leakage_correction.csv'
    if f.exists():
        kn['realdata_leakage'] = pd.read_csv(f).to_dict(orient='records')
    kn['parameters'] = {**PAR, 'N_big': N_BIG, 'B_recovery': B_REC, 'B_attenuation': B_ATT, 'target_raw': TARGET_RAW,
                        'predictors': {k: list(v) for k, v in PRED.items()},
                        'learned_like': {k: list(v) for k, v in LEARNED_LIKE.items()}}
    f = OUT / 'eq_sigma.json'
    if f.exists():
        kn['parameters']['eq_sigma'] = json.loads(f.read_text())
    return kn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='*', default=['truth', 'recovery', 'reliability', 'censoring', 'attenuation'])
    ap.add_argument('--reps-scale', type=float, default=1.0)
    args = ap.parse_args()
    t0 = time.time()
    warnings.simplefilter('ignore', RuntimeWarning)
    run([p_ for p_ in args.parts if p_ not in ('correction', 'realdata')], args.reps_scale, t0)
    print('simulation parts done', f'{time.time() - t0:.0f}s', flush=True)
    correction_tables()
    if 'realdata' in args.parts:
        realdata_leakage()
        realdata_attenuation_adjusted()
    kn = collect_keys()
    allk = json.loads(KEYS.read_text()) if KEYS.exists() else {}
    allk['simulation'] = kn
    lib.write_json(KEYS, allk)
    runf = OUT / 'run.json'
    prev = json.loads(runf.read_text()) if runf.exists() else {}
    prev[','.join(args.parts)] = {'seconds': round(time.time() - t0), 'reps_scale': args.reps_scale}
    lib.write_json(runf, prev)
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
