"""Revision (round 1), fitness-axis workstream, part 4: proliferation (R2-5) and the leave-class-out axis (R1-M8, R3-M4a).

A. Proliferation.
   Proxies: gCSI doubling time (gCSI GR v1.3 metadata; Genentech's nominal per-line value), DepMap CRISPR-inferred growth
   rate (24Q4; mean over libraries; higher = faster) and the NCI-60 doubling times measured by the DTP in the screen's
   own medium (CellMiner NCI60_CELL_LINE_METADATA.xls, column 'doubling time'). We report (1) agreement of the proxies,
   (2) Spearman correlation of every axis with each proxy, (3) cross-screen axis correlations before and after
   partialling out log doubling time and/or the CRISPR growth rate (same cells), (4) the NCI-60 axis against the NCI-60
   doubling time (its own measured growth), (5) gCSI axes from GR_AOC versus mean viability (fitness share, cross-screen
   agreement, doubling-time correlation), with a check that the gCSI GR values are a deterministic function of relative
   cell count and the nominal doubling time (GR = 2^(1 + log2(x) Td / T) - 1), and (6) triad shared fractions of the
   axes before and after residualizing log doubling time.
B. Leave-class-out axis. For each discovery screen (AUC; z of 02_decompose) and drug d, the axis is the mean z over
   the other drugs EXCLUDING drugs of d's mechanism class (12_drug_classes.classify on compounds.csv; 'Other /
   non-oncology' is a catch-all and is treated as d's own singleton class in the primary variant), or drugs sharing any
   annotated target with d (compounds.csv targets), or either. Selective residuals and fitness shares are recomputed;
   the 353 multi-screen drugs are re-scored exactly as in 02_decompose/12_drug_classes (per pair of screens, Pearson of
   residuals on common cells, >=20 cells; per drug, mean over pairs; reproducible >= 0.3, non-reproducible < 0.15).
Outputs (results/revision_axis/): growth_proxies.csv, axis_vs_growth.csv, pairs_partial_growth.csv, gcsi_gr_vs_viability.csv,
triad_growth_residualized.csv, leave_class_out_screen.csv, leave_class_out_drugs.csv, leave_class_out_summary.csv,
leave_class_out_classes.csv, summary_48.json.
"""
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import importlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import OWN_RAW, PROCESSED, RESULTS, norm_cell, write_json  # noqa: E402

H = importlib.import_module('45_axis_activity_lineage')
DC = importlib.import_module('12_drug_classes')
OUT = RESULTS / 'revision_axis'
DEC = RESULTS / 'decomposition'
DISC = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
CATCH_ALL = 'Other / non-oncology'
SEED = 4801
MIN_AXIS_DRUGS = 10
MIN_COMMON = 20


# ================================================================================================= A. proliferation
def gcsi_doubling_time():
    g = pd.read_csv(OWN_RAW / 'confirmation' / 'OUTPUT' / 'gCSI_GRmetrics_v1.3.tsv', sep='\t', usecols=['CellLineName', 'doublingtime'])
    cmap = pd.read_csv(PROCESSED / 'confirmation' / 'gCSI_cell_mapping.csv').dropna().set_index('source_name').depmap_id
    per_line_sd = g.groupby('CellLineName').doublingtime.std().fillna(0)
    dt = g.drop_duplicates('CellLineName').assign(depmap_id=lambda d: d.CellLineName.map(cmap)).dropna()
    return dt.groupby('depmap_id').doublingtime.mean(), float(per_line_sd.max())


def nci60_doubling_time():
    m = pd.read_excel(OWN_RAW / 'confirmation' / 'nci60_raw' / 'output' / 'documentation' / 'NCI60_CELL_LINE_METADATA.xls', header=None)
    hdr = m.iloc[7].tolist()
    body = m.iloc[8:].copy()
    body.columns = hdr
    body = body[['Cell Line Name', 'doubling time g']].dropna()
    body['dt'] = pd.to_numeric(body['doubling time g'], errors='coerce')
    body = body.dropna(subset=['dt'])
    body['key'] = body['Cell Line Name'].astype(str).map(lambda s: norm_cell(s.split(':', 1)[-1]))
    mp = pd.read_csv(PROCESSED / 'confirmation' / 'NCI60_cell_mapping.csv').dropna(subset=['depmap_id'])
    keymap = mp.drop_duplicates('key').set_index('key').depmap_id
    body['depmap_id'] = body.key.map(keymap)
    return body.dropna(subset=['depmap_id']).groupby('depmap_id').dt.mean(), int(len(body))


def gr_formula_check():
    v = pd.read_csv(OWN_RAW / 'confirmation' / 'OUTPUT' / 'gCSI_GRvalues_v1.3.tsv', sep='\t',
                    usecols=['TrtDuration', 'doublingtime', 'GR', 'relative_cell_count'])
    v = v.dropna()
    v = v[v.relative_cell_count > 0]
    pred = 2 ** (1 + np.log2(v.relative_cell_count) * v.doublingtime / v.TrtDuration) - 1
    dev = (pred - v.GR).abs()
    return {'rows': int(len(v)), 'median_abs_dev': float(dev.median()), 'q99_abs_dev': float(dev.quantile(0.99)),
            'frac_within_0.005': float((dev < 0.005).mean()), 'corr': float(np.corrcoef(pred, v.GR)[0, 1])}


def proliferation(rng):
    nat = pd.read_csv(OUT / 'axes_native.csv', index_col=0)
    dre = pd.read_csv(OUT / 'dreval_axes_depmap.csv', index_col=0)
    gr = pd.read_csv(PROCESSED / 'features' / 'growth_rate.csv', index_col=0).growth_rate.dropna()
    dt, dt_sd = gcsi_doubling_time()
    ndt, n_nci_meta = nci60_doubling_time()
    ldt, lndt = np.log(dt), np.log(ndt)
    # (1) proxy agreement
    prox = []
    for a, b, an, bn in [(dt, gr, 'gCSI doubling time', 'DepMap CRISPR growth rate'), (ndt, gr, 'NCI-60 doubling time', 'DepMap CRISPR growth rate'),
                         (ndt, dt, 'NCI-60 doubling time', 'gCSI doubling time')]:
        c = a.index.intersection(b.index)
        prox.append({'proxy_a': an, 'proxy_b': bn, 'n': len(c), 'spearman': float(spearmanr(a[c], b[c]).correlation),
                     'pearson_log': H.corr(np.log(a[c]) if 'doubling' in an else a[c], np.log(b[c]) if 'doubling' in bn else b[c], min_n=10)})
    prox = pd.DataFrame(prox)
    # (2) axes against proxies
    axes = {k: nat[k] for k in nat.columns}
    axes.update({f'DrEval {k}': dre[k] for k in dre.columns if k.endswith('|all')})
    rows = []
    for k, ax in axes.items():
        ax = ax.dropna()
        row = {'axis': k, 'cells': len(ax)}
        for pn, p in [('gCSI_log_doubling_time', ldt), ('DepMap_growth_rate', gr), ('NCI60_log_doubling_time', lndt)]:
            c = ax.index.intersection(p.index)
            row[f'n_{pn}'] = len(c)
            row[f'spearman_{pn}'] = float(spearmanr(ax[c], p[c]).correlation) if len(c) >= 10 else np.nan
        rows.append(row)
    avg = pd.DataFrame(rows)
    # (3) cross-screen pairs, partialling growth proxies (identical cells for all four versions)
    main_axes = ['PRISM|all', 'CTRP|all', 'GDSC1|all', 'GDSC2|all', 'gCSI|all', 'gCSI_GRAOC|all', 'CCLE|all',
                 'GDSC1_lnIC50|all', 'GDSC2_lnIC50|all', 'PRISM|active', 'CTRP|active', 'GDSC1|active', 'GDSC2|active']
    prow = []
    for a, b in itertools.combinations(main_axes, 2):
        if a.split('|')[0].split('_')[0] == b.split('|')[0].split('_')[0]:
            continue
        xa, xb = nat[a].dropna(), nat[b].dropna()
        c = xa.index.intersection(xb.index).intersection(ldt.index).intersection(gr.index)
        if len(c) < 30:
            continue
        x, y = xa[c].values, xb[c].values
        prow.append({'a': a, 'b': b, 'cells_with_both_proxies': len(c), 'r': H.corr(x, y),
                     'partial_log_dt': H.partial_corr(x, y, ldt[c].values[:, None]),
                     'partial_growth_rate': H.partial_corr(x, y, gr[c].values[:, None]),
                     'partial_both': H.partial_corr(x, y, np.c_[ldt[c].values, gr[c].values]),
                     'r_axis_a_logdt': H.corr(x, ldt[c].values), 'r_axis_b_logdt': H.corr(y, ldt[c].values)})
        cc = xa.index.intersection(xb.index).intersection(ldt.index)   # all cells with doubling time
        prow[-1].update({'cells_with_dt': len(cc), 'r_dt_cells': H.corr(xa[cc], xb[cc]),
                         'partial_log_dt_dt_cells': H.partial_corr(xa[cc].values, xb[cc].values, ldt[cc].values[:, None])})
    pp = pd.DataFrame(prow)
    # (4) NCI-60 axis against its own measured doubling time; other axes on the NCI-60 lines
    nci = []
    for k in ['NCI60|all', 'PRISM|all', 'CTRP|all', 'GDSC1|all', 'GDSC2|all', 'gCSI|all', 'GDSC1_lnIC50|all', 'GDSC2_lnIC50|all', 'CCLE|all']:
        ax = nat[k].dropna()
        c = ax.index.intersection(lndt.index)
        if len(c) < 10:
            continue
        rs = spearmanr(ax[c], lndt[c])
        r = H.corr(ax[c], lndt[c], min_n=10)
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(c), len(c))
            bs.append(H.corr(ax[c].values[i], lndt[c].values[i], min_n=10))
        nci.append({'axis': k, 'n': len(c), 'spearman_vs_nci60_log_dt': float(rs.correlation), 'p_spearman': float(rs.pvalue),
                    'pearson_vs_nci60_log_dt': r, 'r2': r ** 2, 'pearson_ci_lo': float(np.nanpercentile(bs, 2.5)),
                    'pearson_ci_hi': float(np.nanpercentile(bs, 97.5))})
    nci = pd.DataFrame(nci)
    # (5) gCSI GR_AOC versus mean viability
    s45 = json.loads((OUT / 'summary_45.json').read_text())
    var = pd.DataFrame(s45['axis_variants'])
    lp = pd.read_csv(OUT / 'axis_pairs_lineage.csv')
    g5 = []
    for k in ['gCSI|all', 'gCSI_GRAOC|all']:
        vv = var[(var.screen == k.split('|')[0]) & (var.variant == 'all')].iloc[0]
        ax = nat[k].dropna()
        c = ax.index.intersection(ldt.index)
        row = {'axis': k, 'drugs': int(vv.drugs), 'cells': int(vv.cells), 'split_half': float(vv.split_half),
               'median_fitness_r2': float(vv.median_r2_all_drugs_on_axis),
               'spearman_log_dt': float(spearmanr(ax[c], ldt[c]).correlation), 'n_dt': len(c)}
        for other in ['PRISM|all', 'CTRP|all', 'GDSC1|all', 'GDSC2|all', 'CCLE|all', 'GDSC1_lnIC50|all', 'GDSC2_lnIC50|all', 'NCI60|all']:
            q = lp[((lp.a == k) & (lp.b == other)) | ((lp.b == k) & (lp.a == other))]
            if len(q):
                row[f'r_{other}'] = float(q.r.iloc[0])
                row[f'disatt_{other}'] = float(q.disattenuated.iloc[0])
        g5.append(row)
    g5 = pd.DataFrame(g5)
    # (6) triad shared fraction before/after residualizing log doubling time (cells with doubling time)
    tri = []
    for setname, keys in [('all-drug axes', ['PRISM|all', 'CTRP|all', 'GDSC1|all', 'GDSC2|all', 'gCSI|all', 'CCLE|all']),
                          ('active-drug axes', ['PRISM|active', 'CTRP|active', 'GDSC1|active', 'GDSC2|active', 'gCSI|active', 'CCLE|active']),
                          ('all-drug axes, gCSI as GR_AOC', ['PRISM|all', 'CTRP|all', 'GDSC1|all', 'GDSC2|all', 'gCSI_GRAOC|all', 'CCLE|all'])]:
        for resid in [False, True]:
            ax = {}
            for k in keys:
                a = nat[k].dropna()
                a = a[a.index.isin(ldt.index)]
                if resid:
                    a = pd.Series(H.residualize_on(a.values, ldt[a.index].values[:, None]), index=a.index)
                ax[k] = a
            res, _ = H.shared_fraction(ax, np.random.default_rng(SEED + 7))
            for k, v in res.items():
                tri.append({'axis_set': setname, 'log_dt_residualized': resid, 'axis': k, 'triads': v['triads'],
                            'lambda2': v['lambda2_median'], 'ci_lo': v['ci95'][0], 'ci_hi': v['ci95'][1]})
    tri = pd.DataFrame(tri)
    info = {'gcsi_dt_lines': int(len(dt)), 'gcsi_dt_max_within_line_sd': dt_sd, 'nci60_dt_lines_metadata': n_nci_meta,
            'nci60_dt_lines_mapped': int(len(ndt)), 'gr_formula_check': gr_formula_check(),
            'gray_time_zero': 'not in local files (PharmacoDB GRAY AAC/IC50 only; DMSO-relative)',
            'gdsc2_day1_plates': 'not in local files (GDSC2 fitted release only)'}
    return prox, avg, pp, nci, g5, tri, info


# ================================================================================================ B. leave-class-out
def exclusion_matrices(drugs, cls, targets):
    """Boolean |D| x |D| matrices: True where drug j is excluded from drug i's axis (always including i itself)."""
    n = len(drugs)
    c = np.array([cls.get(d, CATCH_ALL) for d in drugs])
    eye = np.eye(n, dtype=bool)
    same_class = (c[:, None] == c[None, :])
    lco = same_class & (c[:, None] != CATCH_ALL) | eye
    lco_strict = same_class | eye
    tsets = [set(targets.get(d, ())) for d in drugs]
    allt = sorted(set().union(*tsets))
    T = np.zeros((n, len(allt)), dtype=np.int32)
    idx = {t: i for i, t in enumerate(allt)}
    for i, ts in enumerate(tsets):
        for t in ts:
            T[i, idx[t]] = 1
    lto = (T @ T.T > 0) | eye
    return {'standard': eye, 'leave_class_out': lco, 'leave_target_out': lto, 'leave_class_or_target_out': lco | lto,
            'leave_class_out_strict': lco_strict}


def excluded_axis(Z, E):
    v = Z.values
    ok = np.isfinite(v)
    vf = np.where(ok, v, 0.0)
    tot, cnt = vf.sum(axis=0), ok.sum(axis=0)
    S = E.astype(float) @ vf
    C = E.astype(float) @ ok.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = (tot[None, :] - S) / (cnt[None, :] - C)
    G[(cnt[None, :] - C) < MIN_AXIS_DRUGS] = np.nan
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def leave_class_out():
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    text = comp.targets.fillna('') + ' ' + comp.moa.fillna('') + ' ' + comp.target_or_mechanism.fillna('')
    cls = text.map(DC.classify).to_dict()
    targets = comp.targets.fillna('').map(lambda s: tuple(sorted({t.strip().upper() for t in s.split(';') if t.strip()}))).to_dict()
    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0)
    assert all(cls[d] == c for d, c in dt['class'].items())
    stats = pd.read_csv(DEC / 'drug_screen_stats.csv')
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in DISC}
    variants = ['standard', 'leave_class_out', 'leave_target_out', 'leave_class_or_target_out', 'leave_class_out_strict']
    R, r2, srows = {}, {}, []
    for s in DISC:
        E = exclusion_matrices(list(Z[s].index), cls, targets)
        act = set(stats[(stats.screen == s) & (stats.frac_auc_below_0p8 >= 0.1)].compound)
        Gstd = None
        for v in variants:
            G = excluded_axis(Z[s], E[v])
            if v == 'standard':
                Gstd = G
                # identity check against 02_decompose's leave-one-out axis
                ref = H.loo_axis(Z[s])
                assert np.nanmax(np.abs(ref.values - G.values)) < 1e-9
            R[(s, v)] = H.residuals(Z[s], G)
            r2[(s, v)] = H.fitness_r2(Z[s], G)
            same = [H.corr(G.values[i], Gstd.values[i]) for i in range(len(G))]
            srows.append({'screen': s, 'variant': v, 'drugs': len(G), 'mean_excluded_drugs': float(E[v].sum(axis=1).mean() - 1),
                          'median_excluded_drugs': float(np.median(E[v].sum(axis=1) - 1)),
                          'drugs_with_exclusions': int((E[v].sum(axis=1) > 1).sum()),
                          'median_fitness_r2_all': float(r2[(s, v)].median()),
                          'median_fitness_r2_active': float(r2[(s, v)].reindex(sorted(act)).median()),
                          'median_corr_with_standard_axis_row': float(np.nanmedian(same)),
                          'min_corr_with_standard_axis_row': float(np.nanmin(same))})
        print(s, 'leave-class-out axes done', flush=True)
    st = pd.DataFrame(srows)
    # cross-screen re-scoring of multi-screen drugs (02_decompose pairs)
    prow = []
    for a, b in itertools.combinations(DISC, 2):
        common = Z[a].index.intersection(Z[b].index)
        cells = Z[a].columns.intersection(Z[b].columns)
        for d in common:
            za, zb = Z[a].loc[d, cells].values, Z[b].loc[d, cells].values
            m = np.isfinite(za) & np.isfinite(zb)
            row = {'compound': d, 'screen_a': a, 'screen_b': b, 'n_common': int(m.sum()),
                   'rho_raw': H.corr(za, zb, MIN_COMMON)}
            for v in variants:
                row[f'rho_sel_{v}'] = H.corr(R[(a, v)].loc[d, cells].values, R[(b, v)].loc[d, cells].values, MIN_COMMON)
            prow.append(row)
    P = pd.DataFrame(prow)
    # identity check: the standard variant reproduces the paper's cross_screen_pairs.csv
    ref = pd.read_csv(DEC / 'cross_screen_pairs.csv').set_index(['compound', 'screen_a', 'screen_b'])
    chk = P.set_index(['compound', 'screen_a', 'screen_b'])
    common_idx = ref.index.intersection(chk.index)
    check = {'pairs_paper': len(ref), 'pairs_here': len(chk), 'common': len(common_idx),
             'max_abs_diff_raw': float((ref.loc[common_idx, 'rho_raw'] - chk.loc[common_idx, 'rho_raw']).abs().max()),
             'max_abs_diff_sel': float((ref.loc[common_idx, 'rho_sel'] - chk.loc[common_idx, 'rho_sel_standard']).abs().max())}
    per = P.groupby('compound').agg(n_pairs=('rho_raw', 'size'), rho_raw=('rho_raw', 'mean'),
                                    **{f'rho_sel_{v}': (f'rho_sel_{v}', 'mean') for v in variants})
    per['class'] = per.index.map(lambda d: cls.get(d, CATCH_ALL))
    per['name'] = comp.name.reindex(per.index)
    per = per.loc[per.index.intersection(dt.index)] if len(per) != len(dt) else per
    srows = []
    for v in variants:
        x = per[f'rho_sel_{v}']
        srows.append({'variant': v, 'drugs': int(x.notna().sum()), 'median_raw': float(per.rho_raw.median()),
                      'median_selective': float(x.median()), 'frac_reproducible_ge_0.3': float((x >= 0.3).mean()),
                      'n_reproducible': int((x >= 0.3).sum()), 'frac_nonreproducible_lt_0.15': float((x < 0.15).mean()),
                      'n_nonreproducible': int((x < 0.15).sum()),
                      'median_change_vs_standard': float((x - per.rho_sel_standard).median()),
                      'drugs_up_0.05': int(((x - per.rho_sel_standard) > 0.05).sum()),
                      'drugs_down_0.05': int(((x - per.rho_sel_standard) < -0.05).sum()),
                      'reclassified_to_reproducible': int(((x >= 0.3) & (per.rho_sel_standard < 0.3)).sum()),
                      'reclassified_from_reproducible': int(((x < 0.3) & (per.rho_sel_standard >= 0.3)).sum()),
                      'raw_sel_corr': float(per.rho_raw.corr(x))})
    summ = pd.DataFrame(srows)
    crow = []
    for c_, g in per.groupby('class'):
        for v in variants:
            x = g[f'rho_sel_{v}']
            crow.append({'class': c_, 'variant': v, 'drugs': len(g), 'median_selective': float(x.median()),
                         'frac_reproducible': float((x >= 0.3).mean())})
    classes = pd.DataFrame(crow).pivot_table(index=['class', 'drugs'], columns='variant', values=['median_selective', 'frac_reproducible'])
    classes.columns = [f'{a}|{b}' for a, b in classes.columns]
    classes = classes.reset_index().sort_values('median_selective|standard', ascending=False)
    return st, per, summ, classes, check


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option('display.width', 250)
    prox, avg, pp, nci, g5, tri, info = proliferation(rng)
    prox.to_csv(OUT / 'growth_proxies.csv', index=False)
    avg.to_csv(OUT / 'axis_vs_growth.csv', index=False)
    pp.to_csv(OUT / 'pairs_partial_growth.csv', index=False)
    nci.to_csv(OUT / 'nci60_own_doubling_time.csv', index=False)
    g5.to_csv(OUT / 'gcsi_gr_vs_viability.csv', index=False)
    tri.to_csv(OUT / 'triad_growth_residualized.csv', index=False)
    print(info, flush=True)
    for t in [prox, avg, pp, nci, g5, tri]:
        print(t.round(3).to_string(index=False), flush=True)

    st, per, summ, classes, check = leave_class_out()
    st.to_csv(OUT / 'leave_class_out_screen.csv', index=False)
    per.to_csv(OUT / 'leave_class_out_drugs.csv')
    summ.to_csv(OUT / 'leave_class_out_summary.csv', index=False)
    classes.to_csv(OUT / 'leave_class_out_classes.csv', index=False)
    print(check, flush=True)
    for t in [st, summ, classes]:
        print(t.round(3).to_string(index=False), flush=True)
    write_json(OUT / 'summary_48.json', {
        'proliferation': {'info': info, 'proxies': prox.to_dict(orient='records'), 'axis_vs_growth': avg.to_dict(orient='records'),
                          'pairs_partial': pp.to_dict(orient='records'), 'nci60_own_dt': nci.to_dict(orient='records'),
                          'gcsi_gr_vs_viability': g5.to_dict(orient='records'), 'triads_dt_residualized': tri.to_dict(orient='records')},
        'leave_class_out': {'identity_check_vs_paper': check, 'screens': st.to_dict(orient='records'),
                            'summary': summ.to_dict(orient='records'), 'classes': classes.to_dict(orient='records')},
        'settings': {'seed': SEED, 'min_axis_drugs': MIN_AXIS_DRUGS, 'catch_all_class': CATCH_ALL}})


if __name__ == '__main__':
    main()
