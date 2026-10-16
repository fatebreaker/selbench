"""Reliability ceiling for drug-specific selectivity and the fraction of it captured by predictors.

Measurement model: a screen's selective residual for drug d is r_s = l_s * t + e_s, with t the
selectivity shared by independent screens and e_s screen-specific error. For three screens the
reliability of screen A is R_A = rho_AB * rho_AC / rho_BC (triad estimator).

For a predictor f trained only in screen T and two other evaluation screens E1, E2 (cells common to
both, f out-of-fold), the squared correlation of f with the shared selectivity t is
    c^2 = pc(f, z_E1 | g_E1) * pc(f, z_E2 | g_E2) / corr(r_E1, r_E2),
which needs no estimate of individual reliabilities. A measured replicate (z_T itself) gives c^2 = R_T.
Aggregates use ratios of medians across drugs with drug-bootstrap intervals.
"""
import itertools

import numpy as np
import pandas as pd

from common import RESULTS, write_json
from modeling import corr, partial_corr

DEC = RESULTS / 'decomposition'
MOD = RESULTS / 'models'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
MODELS = ['ridge_sel', 'ridge_raw', 'ridge_crispr', 'lineage_raw', 'target_prior', 'fitness_omics', 'replicate']
MIN_COMMON = 20
N_BOOT = 2000


def loo_fitness(Z):
    vals = Z.values; ok = np.isfinite(vals)
    tot = np.nansum(vals, axis=0); cnt = ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = (tot[None, :] - np.where(ok, vals, 0)) / (cnt[None, :] - ok)
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def boot_ratio(num, den, rng):
    num, den = np.asarray(num), np.asarray(den)
    est = np.median(num) / np.median(den)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(num), len(num))
        bs.append(np.median(num[i]) / np.median(den[i]))
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    out = RESULTS / 'ceiling'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in SCREENS}
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in SCREENS}
    G = {s: loo_fitness(Z[s]) for s in SCREENS}
    P = {}
    for m in MODELS:
        for s in SCREENS:
            f = MOD / f'pred_{m}_{s}.parquet'
            if f.exists():
                P[(m, s)] = pd.read_parquet(f)

    # ---- per-drug triad reliabilities of selective residuals ----
    rel_rows = []
    drugs = sorted(set().union(*[set(Z[s].index) for s in SCREENS]))
    for d in drugs:
        present = [s for s in SCREENS if d in Z[s].index]
        if len(present) < 3:
            continue
        for A in present:
            for B, C in itertools.combinations([s for s in present if s != A], 2):
                cells = Z[A].columns.intersection(Z[B].columns).intersection(Z[C].columns)
                if len(cells) < MIN_COMMON:
                    continue
                ab = corr(R[A].loc[d, cells].values, R[B].loc[d, cells].values)
                ac = corr(R[A].loc[d, cells].values, R[C].loc[d, cells].values)
                bc = corr(R[B].loc[d, cells].values, R[C].loc[d, cells].values)
                rel_rows.append({'compound': d, 'screen': A, 'ref1': B, 'ref2': C, 'n': len(cells),
                                 'rho_ab': ab, 'rho_ac': ac, 'rho_bc': bc})
    rel = pd.DataFrame(rel_rows)
    rel.to_csv(out / 'triad_terms.csv', index=False)

    # ---- disattenuated correlation of predictors with shared selectivity ----
    rows = []
    for (m, T), Pm in P.items():
        for d in Pm.index:
            evals = [s for s in SCREENS if s != T and d in Z[s].index]
            for E1, E2 in itertools.combinations(evals, 2):
                cells = Z[E1].columns.intersection(Z[E2].columns)
                f = Pm.loc[d].reindex(cells).values
                z1, z2 = Z[E1].loc[d, cells].values, Z[E2].loc[d, cells].values
                g1, g2 = G[E1].loc[d, cells].values, G[E2].loc[d, cells].values
                ok = np.isfinite(f) & np.isfinite(z1) & np.isfinite(z2) & np.isfinite(g1) & np.isfinite(g2)
                if ok.sum() < MIN_COMMON:
                    continue
                p1, _ = partial_corr(f[ok], z1[ok], g1[ok])
                p2, _ = partial_corr(f[ok], z2[ok], g2[ok])
                r12 = corr(R[E1].loc[d, cells].values[ok], R[E2].loc[d, cells].values[ok])
                rows.append({'model': m, 'train': T, 'compound': d, 'e1': E1, 'e2': E2, 'n': int(ok.sum()),
                             'pc1': p1, 'pc2': p2, 'r12': r12})
    dis = pd.DataFrame(rows)
    dis.to_csv(out / 'disattenuation_terms.csv', index=False)

    # replicate reliability via the same identity: f = z_T (measured)
    summary = {}
    pairs = pd.read_csv(DEC / 'cross_screen_pairs.csv')
    rep_sel = pairs.groupby('compound').rho_sel.mean()
    dis['rep_class'] = pd.cut(dis.compound.map(rep_sel), [-1, 0.15, 0.3, 1.0],
                              labels=['non_reproducible', 'intermediate', 'reproducible']).astype(str)
    for (m, cls), g in dis.groupby(['model', 'rep_class']):
        g = g.dropna(subset=['pc1', 'pc2', 'r12'])
        g = g[g.r12 > 0.1]       # require a measurable shared component in the two evaluation screens
        if len(g) < 10:
            continue
        num = (g.pc1 * g.pc2).values
        est, lo, hi = boot_ratio(num, g.r12.values, rng)
        summary[f'{m}|{cls}'] = {'n_terms': len(g), 'drugs': int(g.compound.nunique()), 'c2': est, 'ci95': [lo, hi]}
    for m, g in dis.groupby('model'):
        g = g.dropna(subset=['pc1', 'pc2', 'r12'])
        g = g[g.r12 > 0.1]
        num = (g.pc1 * g.pc2).values
        est, lo, hi = boot_ratio(num, g.r12.values, rng)
        summary[f'{m}|all'] = {'n_terms': len(g), 'drugs': int(g.compound.nunique()), 'c2': est, 'ci95': [lo, hi]}

    # screen reliabilities (triads), aggregated with ratio of medians
    relsum = {}
    r = rel.dropna()
    r = r[r.rho_bc > 0.1]
    r['rep_class'] = pd.cut(r.compound.map(rep_sel), [-1, 0.15, 0.3, 1.0],
                            labels=['non_reproducible', 'intermediate', 'reproducible']).astype(str)
    for (s, cls), g in r.groupby(['screen', 'rep_class']):
        if len(g) < 10:
            continue
        est, lo, hi = boot_ratio((g.rho_ab * g.rho_ac).values, g.rho_bc.values, rng)
        relsum[f'{s}|{cls}'] = {'n_terms': len(g), 'drugs': int(g.compound.nunique()), 'reliability': est, 'ci95': [lo, hi]}
    for s, g in r.groupby('screen'):
        est, lo, hi = boot_ratio((g.rho_ab * g.rho_ac).values, g.rho_bc.values, rng)
        relsum[f'{s}|all'] = {'n_terms': len(g), 'drugs': int(g.compound.nunique()), 'reliability': est, 'ci95': [lo, hi]}
    write_json(out / 'summary.json', {'captured_fraction_c2': summary, 'screen_reliability': relsum})
    for k, v in sorted(summary.items()):
        print('c2', k, v)
    for k, v in sorted(relsum.items()):
        print('R ', k, v)


if __name__ == '__main__':
    main()
