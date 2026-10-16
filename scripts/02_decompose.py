"""Decompose within-drug cell-line response into a drug-agnostic fitness axis and a drug-specific
(selective) residual, separately in every screen, and measure cross-screen reproducibility.

Within a screen, each drug's sensitivity profile is rank-normal transformed across its measured
cells (z). The fitness axis for drug d is the leave-d-out mean of z over all other drugs in the same
screen. Regressing z_d on that axis gives the fitness share (R^2) and the selective residual r_d.

Outputs (results/decomposition): per-screen z and residual matrices (parquet), per-drug statistics,
pairwise cross-screen correlations, triad reliabilities.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from common import PROCESSED, RESULTS, rank_normal, write_json

MIN_CELLS = 30          # cells per drug within a screen
MIN_COMMON = 20         # common cells for a cross-screen comparison
MIN_DRUGS_PER_CELL = 20


def screen_matrices(long, screen, value='sens'):
    d = long[long.screen == screen]
    M = d.pivot_table(index='compound', columns='depmap_id', values=value, aggfunc='mean')
    M = M.loc[M.notna().sum(axis=1) >= MIN_CELLS]
    M = M.loc[:, M.notna().sum(axis=0) >= MIN_DRUGS_PER_CELL]
    M = M.loc[M.notna().sum(axis=1) >= MIN_CELLS]
    Z = pd.DataFrame(np.vstack([rank_normal(row) for row in M.values]), index=M.index, columns=M.columns)
    return M, Z


def loo_fitness(Z):
    """Leave-one-drug-out mean z per cell, returned as a drug x cell matrix aligned to Z."""
    vals = Z.values
    ok = np.isfinite(vals)
    tot = np.nansum(vals, axis=0)
    cnt = ok.sum(axis=0)
    loo_sum = tot[None, :] - np.where(ok, vals, 0.0)
    loo_cnt = cnt[None, :] - ok.astype(int)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = loo_sum / loo_cnt
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def residualize(Z, G):
    rows, stats = [], []
    for i, d in enumerate(Z.index):
        z, g = Z.values[i], G.values[i]
        ok = np.isfinite(z) & np.isfinite(g)
        r = np.full(z.shape, np.nan)
        X = np.c_[np.ones(ok.sum()), g[ok]]
        beta, *_ = np.linalg.lstsq(X, z[ok], rcond=None)
        r[ok] = z[ok] - X @ beta
        corr = np.corrcoef(z[ok], g[ok])[0, 1]
        rows.append(r)
        stats.append({'compound': d, 'n_cells': int(ok.sum()), 'fitness_slope': beta[1], 'fitness_r': corr,
                      'fitness_r2': corr ** 2})
    return pd.DataFrame(np.vstack(rows), index=Z.index, columns=Z.columns), pd.DataFrame(stats)


def activity_stats(M, screen):
    """Raw-scale activity summaries (sens = -AUC)."""
    auc = -M
    out = pd.DataFrame({'compound': M.index,
                        'auc_median': auc.median(axis=1).values,
                        'auc_iqr': (auc.quantile(0.75, axis=1) - auc.quantile(0.25, axis=1)).values,
                        'frac_auc_below_0p8': (auc < 0.8).sum(axis=1).values / auc.notna().sum(axis=1).values,
                        'auc_sd': auc.std(axis=1).values})
    out['screen'] = screen
    return out


def pair_corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_COMMON or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--value', default='sens', help="column of discovery_long to analyse ('sens' or 'neg_ln_ic50')")
    ap.add_argument('--out', default='decomposition')
    args = ap.parse_args()
    out = RESULTS / args.out
    out.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    long['neg_ln_ic50'] = -long['ln_ic50']
    screens = [s for s in ['PRISM', 'CTRP', 'GDSC1', 'GDSC2'] if long.loc[long.screen == s, args.value].notna().any()]

    Zs, Rs, Gs, stats, acts = {}, {}, {}, [], []
    for s in screens:
        M, Z = screen_matrices(long, s, args.value)
        G = loo_fitness(Z)
        R, st = residualize(Z, G)
        st['screen'] = s
        Zs[s], Rs[s], Gs[s] = Z, R, G
        stats.append(st)
        acts.append(activity_stats(M, s) if args.value == 'sens' else pd.DataFrame())
        Z.to_parquet(out / f'z_{s}.parquet'); R.to_parquet(out / f'resid_{s}.parquet')
        # screen-level fitness axis (all-drug mean, not leave-out) for descriptive use
        Z.mean(axis=0).rename('fitness').to_frame().to_parquet(out / f'fitness_{s}.parquet')
        print(s, Z.shape, 'median fitness R2', round(st.fitness_r2.median(), 3))
    stats = pd.concat(stats, ignore_index=True)
    if args.value == 'sens':
        stats = stats.merge(pd.concat(acts, ignore_index=True), on=['compound', 'screen'], how='left')
    stats.to_csv(out / 'drug_screen_stats.csv', index=False)

    # pairwise cross-screen reproducibility of raw and selective profiles
    pairs = []
    for a, b in itertools.combinations(screens, 2):
        common_drugs = Zs[a].index.intersection(Zs[b].index)
        cells = Zs[a].columns.intersection(Zs[b].columns)
        fa = Zs[a][cells].mean(axis=0).values; fb = Zs[b][cells].mean(axis=0).values
        fit_corr, _ = pair_corr(fa, fb)
        for d in common_drugs:
            za, zb = Zs[a].loc[d, cells].values, Zs[b].loc[d, cells].values
            ra, rb = Rs[a].loc[d, cells].values, Rs[b].loc[d, cells].values
            raw, n = pair_corr(za, zb)
            sel, _ = pair_corr(ra, rb)
            pairs.append({'compound': d, 'screen_a': a, 'screen_b': b, 'n_common': n, 'rho_raw': raw, 'rho_sel': sel,
                          'fitness_axis_corr': fit_corr})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / 'cross_screen_pairs.csv', index=False)

    # triad reliability for drugs in >=3 screens: R_A = rho_AB * rho_AC / rho_BC on pairwise common cells
    key = {(r.compound, r.screen_a, r.screen_b): r for r in pairs.itertuples()}

    def get(d, x, y, col):
        r = key.get((d, x, y)) or key.get((d, y, x))
        return getattr(r, col) if r is not None else np.nan
    triads = []
    for d in pairs.compound.unique():
        present = [s for s in screens if d in Zs[s].index]
        for A, B, C in itertools.permutations(present, 3):
            if B > C:
                continue
            for col in ['rho_raw', 'rho_sel']:
                ab, ac, bc = get(d, A, B, col), get(d, A, C, col), get(d, B, C, col)
                triads.append({'compound': d, 'target': A, 'ref1': B, 'ref2': C, 'metric': col,
                               'rho_ab': ab, 'rho_ac': ac, 'rho_bc': bc})
    triads = pd.DataFrame(triads)
    triads.to_csv(out / 'triads.csv', index=False)

    summary = {'value': args.value, 'screens': {s: list(Zs[s].shape) for s in screens},
               'median_fitness_r2': stats.groupby('screen').fitness_r2.median().to_dict(),
               'pairs': pairs.groupby(['screen_a', 'screen_b']).agg(
                   drugs=('compound', 'size'), median_n=('n_common', 'median'),
                   median_rho_raw=('rho_raw', 'median'), median_rho_sel=('rho_sel', 'median'),
                   fitness_axis_corr=('fitness_axis_corr', 'first')).reset_index().to_dict(orient='records')}
    write_json(out / 'summary.json', summary)
    print(pd.DataFrame(summary['pairs']))


if __name__ == '__main__':
    main()
