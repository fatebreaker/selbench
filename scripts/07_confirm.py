"""Pre-registered confirmatory evaluation of one held-out screen against frozen discovery outputs.

Input: a long table (compound, depmap_id, sens; higher sens = more sensitive) for the held-out screen,
the frozen prediction directory, and the discovery screens used to build it. Implements hypotheses
H1-H5 of protocol/preregistration.md exactly; all secondary model scores are also written.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from common import RESULTS, rank_normal, write_json
from modeling import corr, partial_corr

DEC = RESULTS / 'decomposition'
MIN_CELLS = 30
MIN_COMMON = 20
N_SPLITS = 200
N_BOOT = 1000
REPRODUCIBLE, NONREPRODUCIBLE = 0.30, 0.15
MIN_TEST_N = 8
SEED = 20261014


def build(long):
    M = long.pivot_table(index='compound', columns='depmap_id', values='sens', aggfunc='mean')
    M = M.loc[M.notna().sum(axis=1) >= MIN_CELLS]
    keep = [d for d in M.index if M.loc[d].dropna().nunique() >= 10]
    M = M.loc[keep]
    Z = pd.DataFrame(np.vstack([rank_normal(r) for r in M.values]), index=M.index, columns=M.columns)
    vals = Z.values; ok = np.isfinite(vals)
    tot = np.nansum(vals, axis=0); cnt = ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        Gm = (tot[None, :] - np.where(ok, vals, 0)) / (cnt[None, :] - ok)
    Gl = pd.DataFrame(Gm, index=Z.index, columns=Z.columns)
    R = Z.copy() * np.nan
    for d in Z.index:
        z, g = Z.loc[d].values, Gl.loc[d].values
        m = np.isfinite(z) & np.isfinite(g)
        X = np.c_[np.ones(m.sum()), g[m]]
        b = np.linalg.lstsq(X, z[m], rcond=None)[0]
        r = np.full(len(z), np.nan); r[m] = z[m] - X @ b
        R.loc[d] = r
    return M, Z, Gl, R


def split_half(Z, rng):
    drugs = np.array(Z.index); h = len(drugs) // 2; out = []
    for _ in range(N_SPLITS):
        p = rng.permutation(drugs)
        a, b = Z.loc[p[:h]].mean(), Z.loc[p[h:]].mean()
        m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1]
        out.append(2 * r / (1 + r))
    return float(np.median(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--long', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--pred-dir', default='frozen_predictions')
    ap.add_argument('--discovery', nargs='+', default=['PRISM', 'CTRP', 'GDSC1', 'GDSC2'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = RESULTS / (args.out or f'confirmation_{args.name}')
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = RESULTS / args.pred_dir

    long = pd.read_csv(args.long)
    M, Z, Gl, R = build(long)
    fC = Z.mean(axis=0)
    res = {'screen': args.name, 'drugs': int(Z.shape[0]), 'cells': int(Z.shape[1])}

    # ---------------- H1: screen-specific fitness axis ----------------
    relC = split_half(Z, rng)
    h1 = []
    for s in args.discovery:
        Zs = pd.read_parquet(DEC / f'z_{s}.parquet')
        fs = Zs.mean(axis=0)
        common = fC.index.intersection(fs.index)
        if len(common) < 30:
            h1.append({'discovery': s, 'common_cells': len(common)}); continue
        rel_s = split_half(Zs, rng)
        r = float(np.corrcoef(fC[common], fs[common])[0, 1])
        boots = []
        a, b = fC[common].values, fs[common].values
        for _ in range(N_BOOT):
            i = rng.integers(0, len(a), len(a))
            boots.append(np.corrcoef(a[i], b[i])[0, 1] / np.sqrt(relC * rel_s))
        dis = r / np.sqrt(relC * rel_s)
        h1.append({'discovery': s, 'common_cells': len(common), 'corr': r, 'rel_confirm': relC, 'rel_discovery': rel_s,
                   'disattenuated': dis, 'dis_ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                   'pass': bool(dis < 0.7 and np.percentile(boots, 97.5) < 1.0)})
    tested = [x for x in h1 if 'pass' in x]
    res['H1'] = {'split_half_reliability': relC, 'by_discovery': h1,
                 'pass': bool(tested) and all(x['pass'] for x in tested)}

    # ---------------- H2: cross-screen agreement lives in the selective component -------------
    pairs = []
    for s in args.discovery:
        Zs = pd.read_parquet(DEC / f'z_{s}.parquet'); Rs = pd.read_parquet(DEC / f'resid_{s}.parquet')
        cells = Z.columns.intersection(Zs.columns)
        for d in Z.index.intersection(Zs.index):
            a, b = Z.loc[d, cells].values, Zs.loc[d, cells].values
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < MIN_COMMON:
                continue
            pairs.append({'compound': d, 'discovery': s, 'n': int(m.sum()), 'rho_raw': corr(a, b),
                          'rho_sel': corr(R.loc[d, cells].values, Rs.loc[d, cells].values)})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / 'cross_screen_pairs.csv', index=False)
    if len(pairs):
        mr, ms = pairs.rho_raw.median(), pairs.rho_sel.median()
        res['H2'] = {'pairs': len(pairs), 'drugs': int(pairs.compound.nunique()), 'median_raw': mr, 'median_sel': ms,
                     'ratio': ms / mr if mr > 0 else None, 'pass': bool(mr > 0 and ms / mr >= 0.8)}

    # discovery reproducibility classes (discovery screens only)
    dpairs = pd.read_csv(DEC / 'cross_screen_pairs.csv')
    dpairs = dpairs[dpairs.screen_a.isin(args.discovery) & dpairs.screen_b.isin(args.discovery)]
    disc_rep = dpairs.groupby('compound').rho_sel.mean()
    cls = pd.Series(np.where(disc_rep >= REPRODUCIBLE, 'reproducible',
                             np.where(disc_rep < NONREPRODUCIBLE, 'non_reproducible', 'intermediate')), index=disc_rep.index)

    # frozen predictions
    fams = ['fit_measured', 'fit_omics', 'ridge_raw_cons', 'ridge_sel_cons', 'lineage_sel_cons', 'target_prior',
            'rep_raw_cons', 'rep_sel_cons']
    P = {f: pd.read_parquet(pred_dir / f'{f}.parquet') for f in fams}
    rows = []
    for d in Z.index:
        z, g = Z.loc[d], Gl.loc[d]
        for f in fams:
            if d not in P[f].index:
                continue
            p = P[f].loc[d].reindex(Z.columns).values
            m = np.isfinite(p) & np.isfinite(z.values)
            if m.sum() < MIN_COMMON:
                continue
            sel, n = partial_corr(p, z.values, g.values)
            rows.append({'compound': d, 'family': f, 'n': n, 'raw': corr(p, z.values), 'sel': sel,
                         'class': cls.get(d, 'single_screen')})
        rows.append({'compound': d, 'family': 'own_loo_fitness', 'n': int(np.isfinite(z.values).sum()),
                     'raw': corr(g.values, z.values), 'sel': np.nan, 'class': cls.get(d, 'single_screen')})
    sc = pd.DataFrame(rows)
    sc.to_csv(out / 'scores.csv', index=False)

    # ---------------- H3: discovery class predicts confirmation selective agreement -------------
    rep = sc[sc.family == 'rep_sel_cons'].set_index('compound')
    a = rep.loc[rep['class'] == 'reproducible', 'sel'].dropna()
    b = rep.loc[rep['class'] == 'non_reproducible', 'sel'].dropna()
    h3 = {'n_reproducible': len(a), 'n_non_reproducible': len(b),
          'median_reproducible': float(a.median()) if len(a) else None,
          'median_non_reproducible': float(b.median()) if len(b) else None}
    if len(a) >= 3 and len(b) >= 3:
        h3['mwu_p_one_sided'] = float(mannwhitneyu(a, b, alternative='greater').pvalue)
        h3['pass'] = bool(h3['mwu_p_one_sided'] < 0.05)
    res['H3'] = h3

    # ---------------- H4: drug-agnostic fitness does not transfer -------------
    own = sc[sc.family == 'own_loo_fitness'].set_index('compound').raw
    disc = sc[sc.family == 'fit_measured'].set_index('compound').raw
    common = own.index.intersection(disc.index)
    # the discovery fitness predictor is drug-agnostic: score it on every confirmation drug
    if len(common) < len(own):
        fm = P['fit_measured'].iloc[0].reindex(Z.columns).values
        disc = pd.Series({d: corr(fm, Z.loc[d].values) for d in own.index})
        common = own.index
    diff = (own[common] - disc[common]).dropna()
    res['H4'] = {'drugs': len(diff), 'median_own': float(own[common].median()), 'median_discovery_fitness': float(disc[common].median()),
                 'median_difference': float(diff.median()),
                 'wilcoxon_p_one_sided': float(wilcoxon(diff, alternative='greater').pvalue) if len(diff) >= MIN_TEST_N else None}
    res['H4']['pass'] = bool(res['H4']['median_difference'] >= 0.15 and (res['H4']['wilcoxon_p_one_sided'] or 1) < 0.05)

    # ---------------- H5: models capture part of reproducible selectivity, below measurement ------
    mod = sc[(sc.family == 'ridge_sel_cons') & (sc['class'] == 'reproducible')].set_index('compound').sel
    rp = sc[(sc.family == 'rep_sel_cons') & (sc['class'] == 'reproducible')].set_index('compound').sel
    cm = mod.index.intersection(rp.index)
    h5 = {'drugs': len(cm), 'median_model_sel': float(mod[cm].median()) if len(cm) else None,
          'median_replicate_sel': float(rp[cm].median()) if len(cm) else None}
    if len(cm) >= MIN_TEST_N:
        h5['model_gt0_p'] = float(wilcoxon(mod[cm].dropna(), alternative='greater').pvalue)
        dd = (rp[cm] - mod[cm]).dropna()
        h5['replicate_gt_model_p'] = float(wilcoxon(dd, alternative='greater').pvalue)
        h5['pass_a'] = bool(h5['model_gt0_p'] < 0.05)
        h5['pass_b'] = bool(h5['replicate_gt_model_p'] < 0.05)
    res['H5'] = h5

    summ = sc.groupby(['family', 'class']).agg(n=('raw', 'size'), raw=('raw', 'median'), sel=('sel', 'median')).reset_index()
    summ.to_csv(out / 'score_summary.csv', index=False)
    write_json(out / 'hypotheses.json', res)
    print(pd.Series({k: (v.get('pass') if isinstance(v, dict) else v) for k, v in res.items()}))
    print(summ.to_string())


if __name__ == '__main__':
    main()
