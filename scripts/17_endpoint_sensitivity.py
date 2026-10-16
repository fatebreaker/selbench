"""Endpoint sensitivity: repeat the fitness-axis analyses with GDSC ln IC50 (a potency summary) instead
of AUC, and with gCSI GR_AOC (growth-rate corrected). Reports within-screen reliability, fitness share,
agreement of the axes with every other screen's primary axis, cross-screen raw/selective agreement of
drug profiles, and the fitness-only per-drug correlation within and across screens.
Output: results/endpoint_sensitivity/summary.json and table.csv.
"""
import itertools

import numpy as np
import pandas as pd

from common import PROCESSED, RESULTS, rank_normal, write_json

MIN_CELLS = 30


def build(long, col='sens'):
    M = long.pivot_table(index='compound', columns='depmap_id', values=col, aggfunc='mean')
    M = M.loc[M.notna().sum(axis=1) >= MIN_CELLS]
    M = M.loc[[d for d in M.index if M.loc[d].dropna().nunique() >= 10]]
    return pd.DataFrame(np.vstack([rank_normal(r) for r in M.values]), index=M.index, columns=M.columns)


def loo(Z):
    v = Z.values; ok = np.isfinite(v); tot = np.nansum(v, 0); cnt = ok.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return pd.DataFrame((tot[None] - np.where(ok, v, 0)) / (cnt[None] - ok), index=Z.index, columns=Z.columns)


def resid(Z):
    G = loo(Z); R = Z.copy() * np.nan; r2 = {}
    for d in Z.index:
        z, g = Z.loc[d].values, G.loc[d].values; m = np.isfinite(z) & np.isfinite(g)
        X = np.c_[np.ones(m.sum()), g[m]]; b = np.linalg.lstsq(X, z[m], rcond=None)[0]
        r = np.full(len(z), np.nan); r[m] = z[m] - X @ b; R.loc[d] = r
        r2[d] = np.corrcoef(z[m], g[m])[0, 1] ** 2
    return R, pd.Series(r2), G


def split_half(Z, rng, n=200):
    d = np.array(Z.index); h = len(d) // 2; out = []
    for _ in range(n):
        p = rng.permutation(d); a, b = Z.loc[p[:h]].mean(), Z.loc[p[h:]].mean(); m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1]; out.append(2 * r / (1 + r))
    return float(np.median(out))


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 20 else np.nan


def main():
    out = RESULTS / 'endpoint_sensitivity'; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    long = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    Z = {}
    for s in ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']:
        Z[f'{s}|AUC'] = build(long[long.screen == s])
    long['neg_ln_ic50'] = -long.ln_ic50
    for s in ['GDSC1', 'GDSC2']:
        Z[f'{s}|lnIC50'] = build(long[long.screen == s], 'neg_ln_ic50')
    conf = PROCESSED / 'confirmation'
    Z['gCSI|meanviability'] = build(pd.read_csv(conf / 'gCSI_long.csv.gz'))
    Z['gCSI|GR_AOC'] = build(pd.read_csv(conf / 'gCSI_GRAOC_long.csv.gz'))
    Z['NCI60|GI50'] = build(pd.read_csv(conf / 'NCI60_long.csv.gz'))
    per, axes = {}, {}
    for k, z in Z.items():
        R, r2, G = resid(z)
        per[k] = {'drugs': int(z.shape[0]), 'cells': int(z.shape[1]), 'split_half': split_half(z, rng),
                  'median_fitness_r2': float(r2.median()),
                  'within_fitness_only_per_drug': float(np.nanmedian([corr(G.loc[d].values, z.loc[d].values) for d in z.index]))}
        axes[k] = z.mean(axis=0)
        Z[k] = (z, R)
    rows = []
    for a, b in itertools.combinations(Z, 2):
        if a.split('|')[0] == b.split('|')[0]:
            same = True
        else:
            same = False
        cells = axes[a].index.intersection(axes[b].index)
        if len(cells) < 30:
            continue
        r = corr(axes[a][cells].values, axes[b][cells].values)
        za, ra = Z[a]; zb, rb = Z[b]
        drugs = za.index.intersection(zb.index)
        raws, sels, fit_x = [], [], []
        Gb = loo(zb)
        for d in drugs:
            raws.append(corr(za.loc[d, cells].values, zb.loc[d, cells].values))
            sels.append(corr(ra.loc[d, cells].values, rb.loc[d, cells].values))
        # fitness-only transfer: screen a's all-drug axis scored against screen b's profiles
        for d in zb.index:
            fit_x.append(corr(axes[a].reindex(zb.columns).values, zb.loc[d].values))
        rows.append({'a': a, 'b': b, 'same_screen': same, 'cells': len(cells), 'axis_corr': r,
                     'disattenuated': r / np.sqrt(per[a]['split_half'] * per[b]['split_half']),
                     'drugs': len(drugs), 'median_raw': float(np.nanmedian(raws)) if raws else np.nan,
                     'median_sel': float(np.nanmedian(sels)) if sels else np.nan,
                     'fitness_a_on_b_per_drug': float(np.nanmedian(fit_x))})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / 'table.csv', index=False)
    write_json(out / 'summary.json', {'per_endpoint': per, 'pairs': rows})
    print(pd.DataFrame(per).T.round(3).to_string())
    print(tab.round(3).to_string())


if __name__ == '__main__':
    main()
