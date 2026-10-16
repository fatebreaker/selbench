"""Reviewer-proofing analyses (exploratory; discovery and already-opened confirmation data).

1. Shared fraction of each screen's fitness axis: one-factor triad identities over the six screens,
   lambda_s^2 = rho_st * rho_su / rho_tu on cell lines common to s, t, u (median over triads, cell bootstrap).
2. gCSI doubling time against every fitness axis.
3. Selectivity beyond lineage: cross-screen selective agreement after removing lineage means.
4. Decomposition variants: Pearson on raw sensitivity (no rank transform), active-drug fitness axis,
   >=50 cell lines per drug, and a first-principal-component axis.
5. Drug-bootstrap intervals for the Fig. 2a medians.
Output: results/robustness/summary.json (+ CSV tables).
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import PROCESSED, RESULTS, OWN_RAW, load_models, rank_normal, write_json

DEC = RESULTS / 'decomposition'
DISC = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
N_BOOT = 1000
rng = np.random.default_rng(31)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 20 and np.std(a[m]) > 0 and np.std(b[m]) > 0 else np.nan


def shared_fraction(axes):
    """Triad estimate of the squared loading of each axis on a factor shared across screens."""
    out = {}
    names = list(axes)
    for s in names:
        ests, boots = [], []
        for t, u in itertools.combinations([x for x in names if x != s], 2):
            cells = axes[s].dropna().index.intersection(axes[t].dropna().index).intersection(axes[u].dropna().index)
            if len(cells) < 30:
                continue
            a, b, c = (axes[k][cells].values for k in (s, t, u))
            rst, rsu, rtu = corr(a, b), corr(a, c), corr(b, c)
            if not np.isfinite(rtu) or abs(rtu) < 0.1:
                continue
            ests.append(rst * rsu / rtu)
            bb = []
            for _ in range(200):
                i = rng.integers(0, len(cells), len(cells))
                x1, x2, x3 = corr(a[i], b[i]), corr(a[i], c[i]), corr(b[i], c[i])
                bb.append(x1 * x2 / x3 if abs(x3) > 0.05 else np.nan)
            boots.append(bb)
        if ests:
            B = np.nanmedian(np.array(boots), axis=0)
            out[s] = {'triads': len(ests), 'lambda2_median': float(np.median(ests)),
                      'ci95': [float(np.nanpercentile(B, 2.5)), float(np.nanpercentile(B, 97.5))]}
    return out


def decompose(M, use_rank=True, axis='mean', active=None):
    Z = M.copy()
    if use_rank:
        Z = pd.DataFrame(np.vstack([rank_normal(r) for r in M.values]), index=M.index, columns=M.columns)
    else:
        Z = M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1).replace(0, np.nan), axis=0)
    src = Z if active is None else Z.loc[Z.index.intersection(active)]
    if axis == 'mean':
        v = src.values; ok = np.isfinite(v); tot = np.nansum(v, 0); cnt = ok.sum(0)
        G = pd.DataFrame(np.tile(tot / np.maximum(cnt, 1), (len(Z), 1)), index=Z.index, columns=Z.columns)
        # leave-one-out where the drug contributes to the axis
        for i, d in enumerate(Z.index):
            if d in src.index:
                row = src.loc[d].values; okr = np.isfinite(row)
                G.iloc[i] = np.where(okr, (tot - np.where(okr, row, 0)) / np.maximum(cnt - okr, 1), tot / np.maximum(cnt, 1))
    else:   # first principal component of the mean-imputed matrix (all drugs; descriptive)
        X = src.T.fillna(0).values
        u, sv, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        pc = u[:, 0] * np.sign(np.corrcoef(u[:, 0], np.nanmean(src.values, 0))[0, 1])
        G = pd.DataFrame(np.tile(pc, (len(Z), 1)), index=Z.index, columns=Z.columns)
    R = Z.copy() * np.nan; r2 = []
    for d in Z.index:
        z, g = Z.loc[d].values, G.loc[d].values; m = np.isfinite(z) & np.isfinite(g)
        X = np.c_[np.ones(m.sum()), g[m]]; b = np.linalg.lstsq(X, z[m], rcond=None)[0]
        r = np.full(len(z), np.nan); r[m] = z[m] - X @ b; R.loc[d] = r
        r2.append(np.corrcoef(z[m], g[m])[0, 1] ** 2)
    return Z, R, G.iloc[0], float(np.median(r2))


def main():
    out_dir = RESULTS / 'robustness'; out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    # ---------- 1. shared fraction of fitness axes ----------
    fa = pd.read_csv(RESULTS / 'fitness_axis' / 'fitness_axes.csv', index_col=0)
    axes = {s: fa[s] for s in fa.columns}
    res['shared_fraction_auc'] = shared_fraction(axes)
    ic = {s: pd.read_parquet(RESULTS / 'decomposition_lnic50' / f'z_{s}.parquet').mean() for s in ['GDSC1', 'GDSC2']}
    axes_ic = {k: v for k, v in axes.items() if k not in ('GDSC1', 'GDSC2')}
    axes_ic.update({f'{k}_lnIC50': v for k, v in ic.items()})
    res['shared_fraction_with_ic50'] = shared_fraction(axes_ic)
    import json
    rel = {s: v['split_half'][0] for s, v in json.loads((RESULTS / 'fitness_axis' / 'summary.json').read_text())['per_screen'].items()}
    for k, v in res['shared_fraction_auc'].items():
        v['share_of_reliable_variance'] = v['lambda2_median'] / rel[k]

    # ---------- 2. gCSI doubling time ----------
    g = pd.read_csv(OWN_RAW / 'confirmation' / 'OUTPUT' / 'gCSI_GRmetrics_v1.3.tsv', sep='\t', usecols=['CellLineName', 'doublingtime'])
    cmap = pd.read_csv(PROCESSED / 'confirmation' / 'gCSI_cell_mapping.csv').dropna().set_index('source_name').depmap_id
    dt = g.drop_duplicates('CellLineName').assign(depmap_id=lambda d: d.CellLineName.map(cmap)).dropna().groupby('depmap_id').doublingtime.mean()
    gr_axis = pd.read_csv(PROCESSED / 'confirmation' / 'gCSI_GRAOC_long.csv.gz')
    M = gr_axis.pivot_table(index='compound', columns='depmap_id', values='sens')
    M = M.loc[M.notna().sum(1) >= 30]
    zgr = pd.DataFrame(np.vstack([rank_normal(r) for r in M.values]), index=M.index, columns=M.columns).mean()
    dts = {}
    for name, ax in list(axes.items()) + [('gCSI_GR_AOC', zgr)] + [(f'{k}_lnIC50', v) for k, v in ic.items()]:
        c = ax.dropna().index.intersection(dt.index)
        dts[name] = {'n': len(c), 'spearman_vs_doubling_time': float(spearmanr(ax[c], dt[c]).correlation)}
    res['doubling_time'] = dts

    # ---------- 3. selectivity beyond lineage ----------
    models = load_models().set_index('ModelID')
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in DISC}
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in DISC}
    rows = []
    for a, b in itertools.combinations(DISC, 2):
        cells = Z[a].columns.intersection(Z[b].columns)
        lin = models.OncotreeLineage.reindex(cells).fillna('Unknown')
        for d in Z[a].index.intersection(Z[b].index):
            ra, rb = R[a].loc[d, cells], R[b].loc[d, cells]
            m = ra.notna() & rb.notna()
            if m.sum() < 20:
                continue
            la, lb = ra[m] - ra[m].groupby(lin[m]).transform('mean'), rb[m] - rb[m].groupby(lin[m]).transform('mean')
            rows.append({'compound': d, 'a': a, 'b': b, 'sel': corr(ra[m].values, rb[m].values),
                         'sel_beyond_lineage': corr(la.values, lb.values)})
    bl = pd.DataFrame(rows)
    bl.to_csv(out_dir / 'beyond_lineage_pairs.csv', index=False)
    per_drug = bl.groupby('compound')[['sel', 'sel_beyond_lineage']].mean()
    res['beyond_lineage'] = {'pairs': len(bl), 'drugs': len(per_drug),
                             'median_sel': float(bl.sel.median()), 'median_sel_beyond_lineage': float(bl.sel_beyond_lineage.median()),
                             'frac_reproducible_sel': float((per_drug.sel >= 0.3).mean()),
                             'frac_reproducible_beyond_lineage': float((per_drug.sel_beyond_lineage >= 0.3).mean())}

    # ---------- 4. decomposition variants ----------
    long = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    stats = pd.read_csv(DEC / 'drug_screen_stats.csv')
    variants = {'primary (rank-normal, all-drug mean)': dict(use_rank=True, axis='mean'),
                'no rank transform (standardized AUC)': dict(use_rank=False, axis='mean'),
                'active-drug axis': dict(use_rank=True, axis='mean', active='active'),
                'first principal component axis': dict(use_rank=True, axis='pc1'),
                '>=50 cell lines per drug': dict(use_rank=True, axis='mean', min_cells=50)}
    vrows = []
    for vname, kw in variants.items():
        Zs, Rs, Gs, r2s = {}, {}, {}, {}
        for s in DISC:
            M = long[long.screen == s].pivot_table(index='compound', columns='depmap_id', values='sens', aggfunc='mean')
            M = M.loc[M.notna().sum(1) >= kw.get('min_cells', 30)]
            M = M.loc[:, M.notna().sum(0) >= 20]
            act = None
            if kw.get('active') == 'active':
                act = set(stats[(stats.screen == s) & (stats.frac_auc_below_0p8 >= 0.1)].compound)
            Zs[s], Rs[s], Gs[s], r2s[s] = decompose(M, kw['use_rank'], kw['axis'], act)
        pr = []
        for a, b in itertools.combinations(DISC, 2):
            cells = Zs[a].columns.intersection(Zs[b].columns)
            fx = corr(Gs[a][cells].values, Gs[b][cells].values)
            raws, sels = [], []
            for d in Zs[a].index.intersection(Zs[b].index):
                raws.append(corr(Zs[a].loc[d, cells].values, Zs[b].loc[d, cells].values))
                sels.append(corr(Rs[a].loc[d, cells].values, Rs[b].loc[d, cells].values))
            pr.append((fx, np.nanmedian(raws), np.nanmedian(sels)))
        pr = np.array(pr)
        vrows.append({'variant': vname, **{f'fitness_r2_{s}': r2s[s] for s in DISC},
                      'fitness_axis_corr_min': float(pr[:, 0].min()), 'fitness_axis_corr_max': float(pr[:, 0].max()),
                      'median_raw': float(np.median(pr[:, 1])), 'median_sel': float(np.median(pr[:, 2]))})
        print(vrows[-1], flush=True)
    pd.DataFrame(vrows).to_csv(out_dir / 'decomposition_variants.csv', index=False)
    res['variants'] = vrows

    # ---------- 5. drug-bootstrap CIs for Fig. 2a ----------
    sc = pd.read_csv(RESULTS / 'models' / 'scores.csv.gz')
    ci = {}
    for (m, within), g2 in sc.assign(within=sc['train'] == sc['eval']).groupby(['model', 'within']):
        for metric in ['raw', 'sel']:
            v = g2[metric].dropna().values
            if len(v) < 10:
                continue
            boots = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(N_BOOT)]
            ci[f'{m}|{"within" if within else "cross"}|{metric}'] = [float(np.median(v)), float(np.percentile(boots, 2.5)),
                                                                     float(np.percentile(boots, 97.5)), len(v)]
    res['fig2_bootstrap'] = ci
    write_json(out_dir / 'summary.json', res)
    for k in ['shared_fraction_auc', 'shared_fraction_with_ic50', 'doubling_time', 'beyond_lineage']:
        print(k, res[k])


if __name__ == '__main__':
    main()
