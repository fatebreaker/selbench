"""Revision analysis (R3 minor 12, R3-M3): simulation of the c^2 and reliability estimators with like-for-like truths.

Improves 19_simulation_check.py (constant c^2, independent errors, comparison with the median of per-drug truths) by
adding heterogeneous c^2 and reliabilities, correlated same-laboratory errors, raw-trained predictors that contain a
fitness component, a predictor that learns the training laboratory's systematic component, and small samples.

Generative model (per scenario replicate). Four discovery-like screens in two laboratories (A1, A2 | B1, B2) and two
single-screen laboratories (X, Y) used only for cross-laboratory triads. For drug d, screen s and cell c:
    y = beta_sd F_s(c) + sigma_t,d t_d(c) + sigma_e,s [ sqrt(rho_lab) u_lab(s),d(c) + sqrt(1 - rho_lab) eps_sd(c) ]
F_s: screen fitness axis (correlation 0.6 with a shared fitness factor); t_d: selectivity shared by all laboratories
(the estimand's target); u: laboratory-specific component shared by the two screens of one laboratory (error
correlation rho_lab). The study's pipeline is applied unchanged (rank-normal z, leave-drug-out fitness axis g, OLS
residual r). Drugs are measured in all four discovery screens (42%) or in three of them, and in X and Y.
Predictors trained in screen T (all cell-blind: their noise is independent of every screen's error):
    sel       corr(f, t) = c_d (heterogeneous; c_d^2 proportional to the drug's reliability with log-normal noise)
    raw       sel plus a noisy estimate of T's fitness axis (half of the prediction variance)
    lablearn  sel plus the training laboratory's component u in proportion to its variance in T
    replicate z_T (measured, raw profile; the paper's reference)
    replicate_sel  r_T (measured selective residual)
Truths per term k (oracle quantities computed from the simulated t): s_k = corr(r_E1, t) corr(r_E2, t) and
c2_k = corr(f, t)^2 (raw truth) or the mean over E1, E2 of corr(f residualised on g_E, t)^2 (selective truth, used
for every predictor; for replicate_sel and replicate it is the reliability R_T = corr(r_T, t)^2 and the diluted
raw-profile value respectively; R_T is also reported). Oracle sample correlations are corrected for their first-order
small-sample bias (E r^2 = rho^2 + (1 - rho^2)^2 / n; E r_a r_b = rho_a rho_b + psi / n), so truths are population
values; the uncorrected 'sample oracle' truths are kept for reference, and for sel/lablearn the generative c_d^2 gives
an exact population check. Like-for-like truths apply each aggregation to (c2_k s_k, s_k) on the same terms; the
pooled truth is sum_d mean_k(c2_k s_k) / sum_d mean_k(s_k); the median-drug truth is the median over drugs of the
drug's s-weighted c2. Discovery terms are also split into configurations (evaluation pair from one laboratory, or
training screen sharing a laboratory with an evaluation screen), the diagnostic used on the real data in script 50.
Outputs: results/revision_ceiling/51_simulation_runs.csv (every replicate), 51_simulation_bias.csv and
51_simulation_ratio.csv (summaries; `--summarise-only` rebuilds them from the runs file).
"""
import importlib.util
import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, rank_normal, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('ce49', HERE / '49_ceiling_estimands.py')
ce = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ce)

OUT = Path(os.environ.get('REVISION_OUT', RESULTS / 'revision_ceiling'))   # override only for test runs
DISC = ['A1', 'A2', 'B1', 'B2']
XLAB = ['X', 'Y']
LAB = {'A1': 'A', 'A2': 'A', 'B1': 'B', 'B2': 'B', 'X': 'X', 'Y': 'Y'}
NOISE_SD = {'A1': 1.1, 'A2': 0.9, 'B1': 0.85, 'B2': 0.7, 'X': 0.8, 'Y': 0.9}
PREDICTORS = ['sel', 'raw', 'lablearn', 'replicate', 'replicate_sel']
N_REP = int(os.environ.get('REVISION_N_REP', 40))
N_CELL_BOOT = 60
WORKERS = int(os.environ.get('REVISION_WORKERS', 6))
SEED = 5101
AGGS = ce.AGGS

SCENARIOS = {
    'S0_homogeneous': dict(hetero=False, rho_lab=0.0),
    'S1_heterogeneous': dict(hetero=True, rho_lab=0.0),
    'S1b_heterogeneous_c2_independent_of_R': dict(hetero=True, rho_lab=0.0, c2_indep=True),
    'S2_hetero_samelab_err0.10': dict(hetero=True, rho_lab=0.10),
    'S2b_hetero_samelab_err0.25': dict(hetero=True, rho_lab=0.25),
    'S3_hetero_err0.10_n60': dict(hetero=True, rho_lab=0.10, n_cells=60, obs_frac=1.0),
}


def simulate(seed, hetero=True, rho_lab=0.0, c2_indep=False, n_drugs=150, n_cells=400, obs_frac=0.85, rho_f=0.6):
    rng = np.random.default_rng(seed)
    screens = DISC + XLAB
    # which cells each screen measures, which drugs each discovery screen measures
    cell_obs = {s: rng.random(n_cells) < obs_frac for s in screens}
    in4 = rng.random(n_drugs) < 0.42
    drug_screens = []
    for d in range(n_drugs):
        sc = list(DISC) if in4[d] else list(rng.choice(DISC, 3, replace=False))
        drug_screens.append(sorted(sc) + XLAB)
    sig_t = np.full(n_drugs, 0.6) if not hetero else np.exp(rng.normal(np.log(0.55), 0.55, n_drugs))
    t = rng.normal(size=(n_drugs, n_cells))
    t = (t - t.mean(1, keepdims=True)) / t.std(1, keepdims=True)
    u = {L: rng.normal(size=(n_drugs, n_cells)) for L in set(LAB.values())}
    shared_fit = rng.normal(size=n_cells)
    F = {s: rho_f * shared_fit + np.sqrt(1 - rho_f ** 2) * rng.normal(size=n_cells) for s in screens}
    Z, G, R = {}, {}, {}
    for s in screens:
        beta = np.abs(rng.normal(0.8, 0.3, n_drugs))
        e = np.sqrt(rho_lab) * u[LAB[s]] + np.sqrt(1 - rho_lab) * rng.normal(size=(n_drugs, n_cells))
        y = beta[:, None] * F[s][None, :] + sig_t[:, None] * t + NOISE_SD[s] * e
        y[:, ~cell_obs[s]] = np.nan
        z = np.vstack([rank_normal(row) for row in y])
        tot, cnt = np.nansum(z, 0), np.isfinite(z).sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            g = (tot[None, :] - np.where(np.isfinite(z), z, 0)) / (cnt[None, :] - np.isfinite(z))
        r = np.full_like(z, np.nan)
        ok = cell_obs[s]
        for i in range(n_drugs):
            X = np.c_[np.ones(ok.sum()), g[i, ok]]
            r[i, ok] = z[i, ok] - X @ np.linalg.lstsq(X, z[i, ok], rcond=None)[0]
        Z[s], G[s], R[s] = z, g, r
    # approximate population reliability, used only to generate heterogeneous c^2
    rel = np.mean([sig_t ** 2 / (sig_t ** 2 + NOISE_SD[s] ** 2) for s in DISC], axis=0)
    if not hetero:
        c2 = np.full(n_drugs, 0.15)
    elif c2_indep:
        c2 = np.clip(0.35 * np.median(rel) * np.exp(rng.normal(0, 0.8, n_drugs)), 0.002, 0.8)
    else:
        c2 = np.clip(0.35 * rel * np.exp(rng.normal(0, 0.6, n_drugs)), 0.002, 0.8)
    preds = {}
    for T in DISC:
        xi = rng.normal(size=(n_drugs, n_cells))
        c = np.sqrt(c2)[:, None]
        f_sel = c * t + np.sqrt(1 - c ** 2) * xi
        # lab-learning: the training laboratory's component in proportion to its variance in T
        k2 = np.clip(c2 * rho_lab * NOISE_SD[T] ** 2 / sig_t ** 2, 0, 0.9 - c2)
        uu = u[LAB[T]]; uu = (uu - uu.mean(1, keepdims=True)) / uu.std(1, keepdims=True)
        f_lab = c * t + np.sqrt(k2)[:, None] * uu + np.sqrt(np.clip(1 - c ** 2 - k2[:, None], 0.05, None)) * rng.normal(size=(n_drugs, n_cells))
        fit_hat = F[T] + 0.5 * rng.normal(size=n_cells)
        fit_hat = (fit_hat - fit_hat.mean()) / fit_hat.std()
        f_raw = f_sel + fit_hat[None, :]
        preds[('sel', T)], preds[('lablearn', T)], preds[('raw', T)] = f_sel, f_lab, f_raw
        preds[('replicate', T)], preds[('replicate_sel', T)] = Z[T], R[T]
    return dict(Z=Z, G=G, R=R, t=t, preds=preds, drug_screens=drug_screens, c2_gen=c2, n_drugs=n_drugs, n_cells=n_cells)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def resid_on(x, g):
    X = np.c_[np.ones(len(g)), g]
    return x - X @ np.linalg.lstsq(X, x, rcond=None)[0]


def terms_for(sim, crosslab, W):
    """Disattenuation terms with oracle truths, for every predictor."""
    Z, G, R, t = sim['Z'], sim['G'], sim['R'], sim['t']
    rows, boots = [], []
    for d in range(sim['n_drugs']):
        scr = sim['drug_screens'][d]
        disc = [s for s in scr if s in DISC]
        for T in disc:
            if crosslab:
                evals = [(e1, e2) for e1, e2 in itertools.combinations([s for s in scr if s != T], 2)
                         if len({LAB[T], LAB[e1], LAB[e2]}) == 3]
            else:
                evals = list(itertools.combinations([s for s in disc if s != T], 2))
            for E1, E2 in evals:
                for p in PREDICTORS:
                    f = sim['preds'][(p, T)][d]
                    z1, g1, r1, z2, g2, r2 = Z[E1][d], G[E1][d], R[E1][d], Z[E2][d], G[E2][d], R[E2][d]
                    ok = np.isfinite(f) & np.isfinite(z1) & np.isfinite(z2)
                    if ok.sum() < 20:
                        continue
                    X = np.c_[f[ok], z1[ok], g1[ok], z2[ok], g2[ok], r1[ok], r2[ok]]
                    p1, p2, r12, n = ce.dis_term(X)
                    tt = t[d, ok]
                    nk = int(ok.sum())
                    l1, l2 = corr(r1[ok], tt), corr(r2[ok], tt)
                    s_k = l1 * l2
                    c2_raw = corr(f[ok], tt) ** 2
                    c2_e = [corr(resid_on(f[ok], g1[ok]), tt) ** 2, corr(resid_on(f[ok], g2[ok]), tt) ** 2]
                    c2_sel = 0.5 * sum(c2_e)
                    rT = R[T][d]; okT = ok & np.isfinite(rT)
                    R_T = corr(rT[okT], t[d, okT]) ** 2
                    # oracle quantities are sample correlations with t; remove their first-order small-sample bias
                    # (E r^2 = rho^2 + (1 - rho^2)^2 / n; E r_a r_b = rho_a rho_b + psi / n) so that truths are
                    # population values
                    db = lambda r2_, m: r2_ - (1 - r2_) ** 2 / m
                    s_db = s_k - ce.psi_overlap(l1, l2, r12[0]) / nk
                    c2_sel_db = 0.5 * sum(db(x, nk) for x in c2_e)
                    c2_raw_db = db(c2_raw, nk)
                    R_T_db = db(R_T, int(okT.sum()))
                    c2_gen = float(sim['c2_gen'][d]) if p in ('sel', 'lablearn') else np.nan
                    rows.append({'drug': d, 'pred': p, 'train': T, 'e1': E1, 'e2': E2,
                                 'config': 'E_same_lab' if LAB[E1] == LAB[E2] else 'T_shares_lab', 'n': int(ok.sum()), 'pc1': p1[0],
                                 'pc2': p2[0], 'r12': r12[0], 's_sample': s_k, 'c2_raw_sample': c2_raw,
                                 'c2_sel_sample': c2_sel, 'R_T_sample': R_T, 's_true': s_db, 'c2_raw_true': c2_raw_db,
                                 'c2_sel_true': c2_sel_db, 'R_T_true': R_T_db, 'c2_gen': c2_gen})
                    if W is not None:
                        bp1, bp2, br12, bn = ce.dis_term(X, W[:, ok])
                        boots.append(np.stack([bp1, bp2, br12, bn]))
    df = pd.DataFrame(rows)
    df['a'] = df.pc1 * df.pc2
    df['abc'] = ce.bias_corrected_product(df.pc1, df.pc2, df.r12, df.n)
    df['b'] = df.r12
    return df, (np.stack(boots) if W is not None else None)


def truths(df, c2col, scol='s_true'):
    """Like-for-like truths: each aggregation applied to (c2_k s_k, s_k) on the same terms."""
    tt = df[['drug_idx']].copy()
    tt['a'] = df[c2col] * df[scol]
    tt['abc'] = tt['a']
    tt['b'] = df[scol]
    return {k: float(v[0]) for k, v in ce.aggregate(tt).items()}


def drug_level_truth(df, c2col, scol='s_true'):
    g = df.assign(w=df[c2col] * df[scol]).groupby('drug')[['w', scol]].sum()
    return g.w / g[scol]


def hierarchical(df, boots, rng):
    """Per-drug bias-corrected ratio, cell-bootstrap s.e. on the Fisher-z scale, REML, population median c^2."""
    z, se = [], []
    for d, g in df.groupby('drug'):
        idx = g.index.values
        sb = g.b.sum()
        if sb <= 0:
            continue
        est = g.abc.sum() / sb
        p1, p2, r12, n = boots[idx, 0], boots[idx, 1], boots[idx, 2], boots[idx, 3]
        with np.errstate(invalid='ignore', divide='ignore'):
            eb = np.nansum(ce.bias_corrected_product(p1, p2, r12, n), 0) / np.nansum(r12, 0)
        zb = ce.signed_root_z(eb[np.isfinite(eb)])
        s = np.subtract(*np.percentile(zb, [75, 25])) / 1.349
        if np.isfinite(s) and s > 0:
            z.append(ce.signed_root_z(np.array([est]))[0]); se.append(s)
    mu, t2 = ce.reml(np.array(z), np.array(se))
    draws = ce.z_to_c2(rng.normal(mu, np.sqrt(t2), 50000))
    return float(np.median(draws))


def one_replicate(args):
    scen, params, rep = args
    seed = SEED + 1000 * list(SCENARIOS).index(scen) + rep
    rng = np.random.default_rng(seed + 7)
    sim = simulate(seed, **params)
    W = ce.cell_weights(sim['n_cells'], N_CELL_BOOT, rng)
    out = []
    for crosslab in [False, True]:
        if crosslab and params.get('rho_lab', 0.0) == 0.0 and scen != 'S1_heterogeneous':
            continue
        df, boots = terms_for(sim, crosslab, W if not crosslab else None)
        df = df[np.isfinite(df[['a', 'abc', 'b']]).all(axis=1)]
        for p in PREDICTORS:
            dp = df[df.pred == p].reset_index(drop=True)
            bp = boots[df.index[df.pred == p].values] if boots is not None else None
            dp, _ = ce.index_drugs(dp.rename(columns={'drug': 'compound'}))
            dp = dp.rename(columns={'compound': 'drug'})
            truth_col = 'R_T_true' if p == 'replicate_sel' else 'c2_sel_true'
            for kappa in [None, 0.1]:
                dk = dp if kappa is None else dp[dp.b > kappa]
                dk2, _ = ce.index_drugs(dk.rename(columns={'drug': 'compound'}))
                dk2 = dk2.rename(columns={'compound': 'drug'})
                est = {k: float(v[0]) for k, v in ce.aggregate(dk2).items()}
                lfl = truths(dk2, truth_col)
                lfl_raw = truths(dk2, 'c2_raw_true')
                lfl_sample = truths(dk2, truth_col.replace('_true', '_sample'), 's_sample')
                for agg in AGGS:
                    out.append({'scenario': scen, 'rep': rep, 'crosslab': crosslab, 'predictor': p,
                                'kappa': 'none' if kappa is None else '>0.1', 'estimator': agg, 'estimate': est[agg],
                                'truth_like_for_like': lfl[agg], 'truth_like_for_like_rawdef': lfl_raw[agg],
                                'truth_like_for_like_sample_oracle': lfl_sample[agg]})
            # discovery configurations: evaluation pair from one laboratory vs training screen sharing a laboratory
            if not crosslab:
                for cfg in ['E_same_lab', 'T_shares_lab']:
                    dc, _ = ce.index_drugs(dp[dp.config == cfg].rename(columns={'drug': 'compound'}))
                    dc = dc.rename(columns={'compound': 'drug'})
                    out.append({'scenario': scen, 'rep': rep, 'crosslab': crosslab, 'predictor': p, 'kappa': 'none',
                                'estimator': f'{ce.PRIMARY}|{cfg}', 'estimate': float(ce.aggregate(dc)[ce.PRIMARY][0]),
                                'truth_like_for_like': truths(dc, truth_col)[ce.PRIMARY],
                                'truth_like_for_like_rawdef': truths(dc, 'c2_raw_true')[ce.PRIMARY]})
            # population truths (no filter)
            dl = drug_level_truth(dp, truth_col)
            pooled = truths(dp, truth_col)['perdrug_romeans']
            R_pooled = truths(dp, 'R_T_true')['perdrug_romeans']
            pooled_raw = truths(dp, 'c2_raw_true')['perdrug_romeans']
            pooled_gen = truths(dp, 'c2_gen')['perdrug_romeans'] if p in ('sel', 'lablearn') else np.nan
            median_gen = float(np.median(dp.groupby('drug').c2_gen.first())) if p in ('sel', 'lablearn') else np.nan
            hier = hierarchical(dp, bp, rng) if (bp is not None) else np.nan
            out.append({'scenario': scen, 'rep': rep, 'crosslab': crosslab, 'predictor': p, 'kappa': 'none',
                        'estimator': 'hierarchical_median', 'estimate': hier, 'truth_like_for_like': float(np.median(dl)),
                        'truth_like_for_like_rawdef': np.nan})
            out.append({'scenario': scen, 'rep': rep, 'crosslab': crosslab, 'predictor': p, 'kappa': 'none',
                        'estimator': '_truths', 'truth_pooled': pooled, 'truth_median_drug': float(np.median(dl)),
                        'truth_mean_drug': float(np.mean(dl)), 'truth_pooled_rawdef': pooled_raw,
                        'truth_pooled_R_T': R_pooled, 'truth_pooled_generative': pooled_gen,
                        'truth_median_drug_generative': median_gen})
    return out


TRUTH_COLS = ['truth_pooled', 'truth_median_drug', 'truth_mean_drug', 'truth_pooled_rawdef', 'truth_pooled_R_T',
              'truth_pooled_generative', 'truth_median_drug_generative']


def main():
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    if '--summarise-only' in sys.argv:          # rebuild the summaries from an existing runs file
        runs = pd.read_csv(OUT / '51_simulation_runs.csv')
    else:
        tasks = [(s, p, r) for s, p in SCENARIOS.items() for r in range(N_REP)]
        rows = []
        with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context('spawn')) as ex:
            for i, res in enumerate(ex.map(one_replicate, tasks, chunksize=1)):
                rows.extend(res)
                if i % 20 == 0:
                    print('done', i, '/', len(tasks), flush=True)
        runs = pd.DataFrame(rows)
        runs.to_csv(OUT / '51_simulation_runs.csv', index=False)
    tr = runs[runs.estimator == '_truths'].set_index(['scenario', 'rep', 'crosslab', 'predictor'])
    est = runs[runs.estimator != '_truths'].drop(columns=[c for c in TRUTH_COLS if c in runs.columns]).join(
        tr[TRUTH_COLS], on=['scenario', 'rep', 'crosslab', 'predictor'])
    est['bias_lfl'] = est.estimate - est.truth_like_for_like
    est['bias_vs_pooled'] = est.estimate - est.truth_pooled
    est['bias_vs_median_drug'] = est.estimate - est.truth_median_drug
    est['bias_vs_generative'] = est.estimate - est.truth_pooled_generative
    summ = est.groupby(['scenario', 'crosslab', 'predictor', 'kappa', 'estimator']).agg(
        reps=('estimate', 'size'), estimate=('estimate', 'mean'), sd=('estimate', 'std'),
        truth_lfl=('truth_like_for_like', 'mean'), bias_lfl=('bias_lfl', 'mean'),
        truth_lfl_sample_oracle=('truth_like_for_like_sample_oracle', 'mean'),
        truth_pooled_generative=('truth_pooled_generative', 'mean'), bias_vs_generative=('bias_vs_generative', 'mean'),
        truth_median_drug_generative=('truth_median_drug_generative', 'mean'),
        truth_pooled=('truth_pooled', 'mean'), bias_vs_pooled=('bias_vs_pooled', 'mean'),
        truth_median_drug=('truth_median_drug', 'mean'), bias_vs_median_drug=('bias_vs_median_drug', 'mean'),
        truth_mean_drug=('truth_mean_drug', 'mean'), truth_pooled_rawdef=('truth_pooled_rawdef', 'mean'),
        truth_pooled_R_T=('truth_pooled_R_T', 'mean')).reset_index()
    summ['rel_bias_lfl'] = summ.bias_lfl / summ.truth_lfl
    summ['rel_bias_vs_pooled'] = summ.bias_vs_pooled / summ.truth_pooled
    summ['rel_bias_vs_median_drug'] = summ.bias_vs_median_drug / summ.truth_median_drug
    summ['rel_bias_vs_generative'] = summ.bias_vs_generative / summ.truth_pooled_generative
    summ.to_csv(OUT / '51_simulation_bias.csv', index=False)
    # model / replicate ratio (primary estimator) against the pooled truth
    rat = []
    prim = est[est.estimator.str.startswith(ce.PRIMARY) & (est.kappa == 'none')]
    for (scen, cl, rep, estimator), g in prim.groupby(['scenario', 'crosslab', 'rep', 'estimator']):
        v = g.set_index('predictor')
        for m in ['sel', 'raw', 'lablearn']:
            for ref in ['replicate', 'replicate_sel']:
                if m in v.index and ref in v.index:
                    rat.append({'scenario': scen, 'crosslab': cl, 'rep': rep, 'terms': estimator.replace(ce.PRIMARY, 'all'),
                                'model': m, 'reference': ref,
                                'ratio_est': v.loc[m, 'estimate'] / v.loc[ref, 'estimate'],
                                'ratio_true_vs_R_T': v.loc[m, 'truth_pooled'] / v.loc[ref, 'truth_pooled_R_T'],
                                'ratio_true_same_reference': v.loc[m, 'truth_pooled'] / v.loc[ref, 'truth_pooled']})
    rat = pd.DataFrame(rat)
    rsum = rat.groupby(['scenario', 'crosslab', 'terms', 'model', 'reference']).agg(
        ratio_est=('ratio_est', 'mean'), sd=('ratio_est', 'std'), ratio_true_vs_R_T=('ratio_true_vs_R_T', 'mean'),
        ratio_true_same_reference=('ratio_true_same_reference', 'mean')).reset_index()
    rsum['rel_bias_vs_R_T'] = rsum.ratio_est / rsum.ratio_true_vs_R_T - 1
    rsum.to_csv(OUT / '51_simulation_ratio.csv', index=False)
    write_json(OUT / '51_summary.json', {'scenarios': SCENARIOS, 'n_rep': N_REP, 'seed': SEED,
                                         'noise_sd': NOISE_SD, 'labs': LAB})
    pd.set_option('display.width', 250)
    show = summ[(summ.kappa == 'none') | (summ.estimator == 'rom_terms')]
    print(show[['scenario', 'crosslab', 'predictor', 'kappa', 'estimator', 'estimate', 'truth_lfl', 'rel_bias_lfl',
                'truth_pooled', 'rel_bias_vs_pooled', 'truth_median_drug']].round(3).to_string())
    print(rsum.round(3).to_string())


if __name__ == '__main__':
    main()
