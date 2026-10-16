"""Round 2, task 2 (R3-M2, R1-M1d): within-screen reliability of the raw profile and of the selective residual, from
every within-screen replicate source in the project. Run before 78_capture.py, which uses the draws saved here for the
reliability-corrected perfect-predictor reference.

Error model. For drug d, the measured rank-normal profile is z = tau + e, with e drug-specific measurement noise,
independent of the predictions p and of the leave-drug-out axis g (the axis averages hundreds of drugs and is treated
as noise-free: split-half reliability 0.98-0.99). Then r_zg^2 = rho_d r_tau,g^2 and all noise sits in the selective
residual: rho_d = r_zg^2 + (1 - r_zg^2) R_sel, where rho_d is the reliability of the raw profile and R_sel that of the
selective residual. Cell-level noise shared across drugs (a run- or plate-specific shift of a cell line) violates this:
it enters the axis itself; see 'plate ICC' below and 78's kappa sensitivity.

Sources (each with its own endpoint and processing; stated in every output row):
  1. IMPROVE replicated drug-cell pairs (CTRPv2, GDSCv1, GDSCv2; data/improve response.tsv; IMPROVE's harmonized AUC,
     one Hill fit over 1e-10..1e-4 M; also IMPROVE's IC50 field, finite values in [0, 12], as a censored IC50-type
     endpoint). For each pair with >= 2 rows (exact duplicates dropped), the two rows with the lowest study IDs are
     mapped onto the drug's rank-normal scale (empirical CDF of the averaged profile used for scoring). Per drug, over
     its replicated cells: raw replicate correlation r(z_A, z_B); selective replicate correlation of the residuals on
     the full-screen axis (slope from all cells); error variance s2 = var(z_A - z_B)/2 (in profile units, var z = 1).
     Dataset-level estimators: error-variance form (primary; does not depend on which cells were replicated):
       rho = 1 - mean_d s2_d,   R_sel = 1 - sum_d s2_d / sum_d (1 - r_zg,d^2);
     pooled within-drug correlation over replicated cells (secondary), and medians of per-drug correlations over
     drugs with >= 20 replicated cells. Plate ICC (GDSC): replicate differences grouped by (cell line, study pair);
     the intraclass correlation of differences across drugs measured in the same two plates estimates the share of
     replicate noise that is shared across drugs within a run.
  2. GDSC1's two screening sites (results/revision_axis/gdsc1_sites_perdrug.csv; 30 compounds screened at both;
     Sanger fits; AUC and ln IC50): per-drug raw (pearson_z) and selective agreement; drug bootstrap only (no cell-
     level data are stored).
  3. PRISM's two runs (results/revision_axis/prism_runs_perdrug.csv; 136 compounds, HTS002 vs MTS010, AUC; and AUC
     capped at 1 in both): per-drug raw and selective agreement; drug bootstrap only.
  4. CTRP v1 vs v2, DrEval CurveCurator ln IC50 and AUC (41's versions; different screens of one institution): per
     drug raw r(z_1, z_2) and selective agreement r(res_1, res_2) on matched cells, two-way bootstrap. With independent
     errors r_12 <= sqrt(rho_1 rho_2): a LOWER bound on the geometric-mean reliability (an upper bound on any
     correction).
Outputs (results/revision_round2/reliability/): per_drug_reliability.csv, reliability_summary.csv, plate_icc.csv,
reliability_draws.npz (draw 0 = point estimate; B = 1,000 two-way or drug draws), run.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.out_dir('reliability')
SEED = 20261021
MIN_REP_CELLS = 3          # per drug, for the error-variance estimator
MIN_CORR_CELLS = 20        # per drug, for per-drug correlations
IC50_RANGE = (0.0, 12.0)


def wvar_center(x, w):
    s = w.sum(1)
    m = (w * x[None]).sum(1) / s
    return s, m


def improve_source(ds, field, rng_seed):
    """Replicate-based reliability for one IMPROVE dataset and response field ('auc' or 'ic50')."""
    im = lib.load_module('im68', '68_improve_csa.py')
    im.extract()
    f = im.find(im.DATA / 'csa_data', 'response.tsv')[0]
    y = pd.read_csv(f, sep='\t', low_memory=False, usecols=['source', im.DRUG, im.CELL, 'study', 'auc', 'ic50'])
    d = y[y.source == ds].rename(columns={im.DRUG: 'drug', im.CELL: 'cell'})
    d['drug'], d['cell'], d['study'] = d.drug.astype(str), d.cell.astype(str), d.study.astype(str)
    if field == 'auc':
        d = d[np.isfinite(d.auc)].assign(v=-d.auc)
    else:
        d = d[np.isfinite(d.ic50) & (d.ic50 >= IC50_RANGE[0]) & (d.ic50 <= IC50_RANGE[1])].assign(v=d.ic50)
    prof = d.groupby(['drug', 'cell'], as_index=False).v.mean().rename(columns={'v': 'response'})
    Z, G, _, r2 = lib.selbench.decompose(prof)
    # replicate pairs: >= 2 rows, two lowest study IDs, exact duplicates dropped
    d = d[d.drug.isin(Z.index)]
    d['nrow'] = d.groupby(['drug', 'cell']).v.transform('size')
    rep = d[d.nrow >= 2].copy()
    rep['sk'] = pd.to_numeric(rep.study, errors='coerce')
    rep = rep.sort_values(['drug', 'cell', 'sk', 'study'], kind='mergesort')
    rep['k'] = rep.groupby(['drug', 'cell']).cumcount()
    a = rep[rep.k == 0].set_index(['drug', 'cell'])
    b = rep[rep.k == 1].set_index(['drug', 'cell'])
    pr = a[['v', 'study']].join(b[['v', 'study']], lsuffix='_a', rsuffix='_b', how='inner')
    n_pairs_all = len(pr)
    pr = pr[pr.v_a != pr.v_b]
    n_identical = n_pairs_all - len(pr)
    pr = pr.reset_index()
    # map replicate values onto the drug's rank-normal scale (empirical CDF of the averaged profile)
    rows = []
    cells = list(Z.columns)
    cix = {c: i for i, c in enumerate(cells)}
    for drug, g in pr.groupby('drug'):
        zrow = Z.loc[drug]
        yv = prof[prof.drug == drug].set_index('cell').response.reindex(zrow.index)
        ok = yv.notna() & zrow.notna()
        ys, zs = yv[ok].values, zrow[ok].values
        o = np.argsort(ys, kind='mergesort')
        ys, zs = ys[o], zs[o]
        za = np.interp(g.v_a.values, ys, zs)
        zb = np.interp(g.v_b.values, ys, zs)
        gg = G.loc[drug].reindex(g.cell).values
        zz, gz = zrow.values.astype(float), G.loc[drug].values.astype(float)
        m = np.isfinite(zz) & np.isfinite(gz)
        bslope = np.cov(zz[m], gz[m])[0, 1] / np.var(gz[m], ddof=1)
        rzg = np.corrcoef(zz[m], gz[m])[0, 1]
        for c, x1, x2, gv, sa, sb in zip(g.cell, za, zb, gg, g.study_a, g.study_b):
            if np.isfinite(gv):
                rows.append((drug, cix[c], x1, x2, gv, bslope, rzg, sa, sb))
    T = pd.DataFrame(rows, columns=['drug', 'ci', 'za', 'zb', 'g', 'b', 'r_zg', 'study_a', 'study_b'])
    T['ra'], T['rb'] = T.za - T.b * T.g, T.zb - T.b * T.g
    drugs = sorted(T.drug.unique())
    W, V = lib.draws(len(cells), len(drugs), rng_seed)
    R = W.shape[0]
    D = len(drugs)
    per = {k: np.full((R, D), np.nan) for k in ['r_raw', 'r_sel', 's2', 'n']}
    pool = {k: np.zeros(R) for k in ['saa', 'sbb', 'sab', 'qaa', 'qbb', 'qab']}
    rzg = np.zeros(D)
    for j, (drug, g) in enumerate(T.groupby('drug', sort=True)):
        w = W[:, g.ci.values]                                     # (R, n)
        n = w.sum(1)
        rzg[j] = g.r_zg.iloc[0]
        stats = {}
        for nm, (x1, x2) in {'raw': (g.za.values, g.zb.values), 'sel': (g.ra.values, g.rb.values)}.items():
            m1, m2 = (w * x1).sum(1) / n, (w * x2).sum(1) / n
            d1, d2 = x1[None] - m1[:, None], x2[None] - m2[:, None]
            c11, c22, c12 = (w * d1 * d1).sum(1), (w * d2 * d2).sum(1), (w * d1 * d2).sum(1)
            stats[nm] = (c11, c22, c12)
        dd = g.za.values - g.zb.values
        md = (w * dd).sum(1) / n
        s2 = (w * (dd[None] - md[:, None]) ** 2).sum(1) / np.maximum(n - 1, 1) / 2.0
        ok_n = len(g)
        per['n'][:, j] = ok_n
        per['s2'][:, j] = np.where(ok_n >= MIN_REP_CELLS, s2, np.nan)
        if ok_n >= MIN_CORR_CELLS:
            for nm in ('raw', 'sel'):
                c11, c22, c12 = stats[nm]
                with np.errstate(invalid='ignore', divide='ignore'):
                    per[f'r_{nm}'][:, j] = c12 / np.sqrt(c11 * c22)
        if ok_n >= MIN_REP_CELLS:
            v = V[:, j]
            for nm, pre in [('raw', 's'), ('sel', 'q')]:
                c11, c22, c12 = stats[nm]
                pool[f'{pre}aa'] += v * c11
                pool[f'{pre}bb'] += v * c22
                pool[f'{pre}ab'] += v * c12
    ok = np.isfinite(per['s2'])
    with np.errstate(invalid='ignore', divide='ignore'):
        s2m = np.nansum(np.where(ok, per['s2'], 0) * V, 1) / (ok * V).sum(1)
        rho_var = 1 - s2m
        rsel_var = 1 - np.nansum(np.where(ok, per['s2'], 0) * V, 1) / (ok * V * (1 - rzg[None] ** 2)).sum(1)
        rho_pool = pool['sab'] / np.sqrt(pool['saa'] * pool['sbb'])
        rsel_pool = pool['qab'] / np.sqrt(pool['qaa'] * pool['qbb'])
    med_raw = lib.wmedian_rows(per['r_raw'], V)
    med_sel = lib.wmedian_rows(per['r_sel'], V)
    # plate ICC: replicate differences grouped by (cell, study_a, study_b) with >= 2 drugs (point estimate + drug
    # resampling of groups is not meaningful; a cell-line bootstrap of groups is used)
    T['dd'] = T.za - T.zb
    T['grp'] = T.ci.astype(str) + '|' + T.study_a + '|' + T.study_b
    icc = plate_icc(T, W)
    pdr = pd.DataFrame({'source': f'IMPROVE {ds} replicated pairs', 'dataset': ds,
                        'endpoint': 'AUC (IMPROVE Hill fit, 1e-10..1e-4 M)' if field == 'auc' else
                        'IC50 field (IMPROVE fit; finite values in [0, 12]; censored)',
                        'drug': drugs, 'n_rep_cells': per['n'][0].astype(int), 'r_raw': per['r_raw'][0],
                        'r_sel': per['r_sel'][0], 's2': per['s2'][0], 'r_zg_full': rzg})
    with np.errstate(invalid='ignore'):
        pdr['rho_var'] = 1 - pdr.s2
        pdr['Rsel_var'] = 1 - pdr.s2 / (1 - pdr.r_zg_full ** 2)
    info = {'replicated_pairs': int(n_pairs_all), 'identical_pairs_dropped': int(n_identical),
            'pairs_used': int(len(T)), 'drugs': D, 'median_rep_cells_per_drug': float(np.median(per['n'][0])),
            'drugs_ge20_rep_cells': int((per['n'][0] >= MIN_CORR_CELLS).sum()),
            'profile_drugs': int(len(Z)), 'profile_cells': int(Z.shape[1]),
            'mean_rzg2_replicated_drugs': float(np.mean(rzg ** 2))}
    draws = {'rho_var': rho_var, 'Rsel_var': rsel_var, 'rho_pooled_corr': rho_pool, 'Rsel_pooled_corr': rsel_pool,
             'rho_median_corr': med_raw, 'Rsel_median_corr': med_sel, 'plate_icc': icc}
    return draws, pdr, info


def plate_icc(T, W):
    """One-way ANOVA intraclass correlation of replicate differences within (cell line, plate pair) groups with
    >= 2 drugs: the share of replicate-difference variance shared across the drugs of one run of a cell line.
    Cell-line bootstrap (weights W on the group's cell line)."""
    g = T.groupby('grp')
    T = T[g.dd.transform('size') >= 2].copy()
    if T.grp.nunique() < 10:
        return np.full(W.shape[0], np.nan)
    codes, uniq = pd.factorize(T.grp)
    T['code'] = codes
    gc = T.groupby('code').ci.first().values
    k = T.groupby('code').size().values.astype(float)
    mu_g = T.groupby('code').dd.mean().values
    ss_w_g = T.groupby('code').dd.apply(lambda x: ((x - x.mean()) ** 2).sum()).values
    out = np.full(W.shape[0], np.nan)
    for b in range(W.shape[0]):
        wg = W[b, gc]
        if wg.sum() < 10:
            continue
        N = (wg * k).sum()
        grand = (wg * k * mu_g).sum() / N
        msb = (wg * k * (mu_g - grand) ** 2).sum() / max(wg.sum() - 1, 1)
        msw = (wg * ss_w_g).sum() / max(N - wg.sum(), 1)
        n0 = (N - (wg * k ** 2).sum() / N) / max(wg.sum() - 1, 1)
        out[b] = (msb - msw) / (msb + (n0 - 1) * msw)
    return out


def drug_boot_source(df, raw_col, sel_col, seed, label):
    rng = np.random.default_rng(seed)
    x = df[[raw_col, sel_col]].dropna()
    n = len(x)
    idx = np.vstack([np.arange(n), rng.integers(0, n, size=(lib.B, n))])
    r, s = x[raw_col].values[idx], x[sel_col].values[idx]
    return {'rho_median_corr': np.median(r, 1), 'Rsel_median_corr': np.median(s, 1),
            'rho_mean_corr': r.mean(1), 'Rsel_mean_corr': s.mean(1)}, n


def ctrp_v1_v2(endpoint, seed):
    """Per-drug raw and selective agreement of CTRP v1 and v2 (DrEval CurveCurator), two-way bootstrap."""
    cd = lib.load_module('cd41', '41_crosslab_data.py')
    V = cd.versions()
    Z1, G1, _, _ = lib.selbench.decompose(V[f'CTRPv2|CC|{endpoint}'])
    Z2, G2, _, _ = lib.selbench.decompose(V[f'CTRPv1|CC|{endpoint}'])
    cells = Z1.columns.intersection(Z2.columns)
    drugs = Z1.index.intersection(Z2.index)
    W, Vd = lib.draws(len(cells), len(drugs), seed)
    R, D = W.shape[0], len(drugs)
    raw, sel = np.full((R, D), np.nan), np.full((R, D), np.nan)
    rows = []
    for j, d in enumerate(drugs):
        z1, g1 = Z1.loc[d, cells].values.astype(float), G1.loc[d, cells].values.astype(float)
        z2, g2 = Z2.loc[d, cells].values.astype(float), G2.loc[d, cells].values.astype(float)
        m = np.isfinite(z1) & np.isfinite(g1) & np.isfinite(z2) & np.isfinite(g2)
        if m.sum() < lib.MIN_CELLS:
            continue
        X = np.column_stack([z1[m], g1[m], z2[m], g2[m]])
        X = (X - X.mean(0)) / X.std(0)
        w = W[:, m]
        n = w.sum(1)[:, None]
        mu = (w @ X) / n
        Cm = {}
        for a in range(4):
            for b in range(a, 4):
                Cm[(a, b)] = (w @ (X[:, a] * X[:, b])) / n[:, 0] - mu[:, a] * mu[:, b]
        c = lambda a, b: Cm[(min(a, b), max(a, b))]  # noqa: E731
        raw[:, j] = c(0, 2) / np.sqrt(c(0, 0) * c(2, 2))
        b1, b2 = c(0, 1) / c(1, 1), c(2, 3) / c(3, 3)
        cov = c(0, 2) - b2 * c(0, 3) - b1 * c(1, 2) + b1 * b2 * c(1, 3)
        v1, v2 = c(0, 0) - c(0, 1) ** 2 / c(1, 1), c(2, 2) - c(2, 3) ** 2 / c(3, 3)
        sel[:, j] = cov / np.sqrt(v1 * v2)
        rows.append({'source': 'CTRP v1 vs v2 (DrEval CurveCurator; different screens, one institution)',
                     'dataset': 'CTRPv2', 'endpoint': f'{endpoint} (CurveCurator)', 'drug': d, 'n_rep_cells': int(m.sum()),
                     'r_raw': raw[0, j], 'r_sel': sel[0, j]})
    return {'rho_median_corr': lib.wmedian_rows(raw, Vd), 'Rsel_median_corr': lib.wmedian_rows(sel, Vd),
            'rho_mean_corr': np.nansum(raw * Vd, 1) / (np.isfinite(raw) * Vd).sum(1),
            'Rsel_mean_corr': np.nansum(sel * Vd, 1) / (np.isfinite(sel) * Vd).sum(1)}, pd.DataFrame(rows)


def main():
    t0 = time.time()
    allr, perdrug, info, draws_out = [], [], {}, {}

    def record(key, source, dataset, endpoint, kind, dr, n, note):
        for stat, v in dr.items():
            draws_out[f'{key}|{stat}'] = np.asarray(v, float)
            lo, hi = lib.ci(v)
            allr.append({'key': key, 'source': source, 'dataset': dataset, 'endpoint': endpoint, 'kind': kind,
                         'stat': stat, 'estimate': float(np.asarray(v)[0]), 'lo': lo, 'hi': hi, 'drugs': n, 'note': note})
    k = 0
    for ds in ['CTRPv2', 'GDSCv1', 'GDSCv2']:
        for field in ['auc', 'ic50']:
            dr, pdr, inf = improve_source(ds, field, SEED + k)
            k += 1
            key = f'IMPROVE_{ds}_{field}'
            info[key] = inf
            perdrug.append(pdr)
            record(key, f'IMPROVE {ds} replicated pairs', ds, pdr.endpoint.iloc[0], 'within-screen replicate rows',
                   dr, inf['drugs'], f"{inf['pairs_used']} replicated pairs; median {inf['median_rep_cells_per_drug']:.0f} "
                                     f"replicated cells per drug; two-way bootstrap")
            print(key, {s: round(float(v[0]), 3) for s, v in dr.items()}, inf, f'{time.time() - t0:.0f}s', flush=True)
    s = pd.read_csv(lib.RES / 'revision_axis' / 'gdsc1_sites_perdrug.csv')
    for ep, g in s.groupby('endpoint'):
        dr, n = drug_boot_source(g, 'pearson_z', 'selective', SEED + 100 + k, 'GDSC1 sites')
        k += 1
        record(f'GDSC1_sites_{ep}', "GDSC1's two sites (Sanger vs MGH), dual-site compounds", 'GDSC1',
               f'{ep} (Sanger fit)', 'between sites within one screen', dr, n, 'drug bootstrap only')
        perdrug.append(g.assign(source="GDSC1's two sites", dataset='GDSC1', endpoint=ep, drug=g.compound,
                                n_rep_cells=g.cells, r_raw=g.pearson_z, r_sel=g.selective)[
            ['source', 'dataset', 'endpoint', 'drug', 'n_rep_cells', 'r_raw', 'r_sel']])
    p = pd.read_csv(lib.RES / 'revision_axis' / 'prism_runs_perdrug.csv')
    for ep, g in p.groupby('endpoint'):
        dr, n = drug_boot_source(g, 'raw', 'selective', SEED + 200 + k, 'PRISM runs')
        k += 1
        record(f'PRISM_runs_{ep}', "PRISM's two runs (HTS002 vs MTS010), same compounds", 'PRISM', ep,
               'between runs within one screen (format and pools differ)', dr, n, 'drug bootstrap only')
        perdrug.append(g.assign(source="PRISM's two runs", dataset='PRISM', drug=g.compound_short,
                                r_raw=g.raw, r_sel=g.selective)[['source', 'dataset', 'endpoint', 'drug', 'r_raw', 'r_sel']])
    for ep in ['lnIC50', 'AUC']:
        dr, pdr = ctrp_v1_v2(ep, SEED + 300 + k)
        k += 1
        record(f'CTRPv1v2_{ep}', 'CTRP v1 vs v2 (lower bound on geometric-mean reliability)', 'CTRPv2',
               f'{ep} (CurveCurator)', 'between screens (lower bound)', dr, len(pdr), 'two-way bootstrap')
        perdrug.append(pdr)
        print('CTRPv1v2', ep, {s: round(float(v[0]), 3) for s, v in dr.items()}, flush=True)
    summ = pd.DataFrame(allr)
    summ.to_csv(OUT / 'reliability_summary.csv', index=False, float_format='%.5g')
    pd.concat(perdrug, ignore_index=True).to_csv(OUT / 'per_drug_reliability.csv', index=False, float_format='%.5g')
    np.savez_compressed(OUT / 'reliability_draws.npz', **draws_out)
    lib.write_json(OUT / 'run.json', {'seconds': round(time.time() - t0), 'B': lib.B, 'seed': SEED, 'info': info,
                                      'ic50_range': IC50_RANGE, 'min_rep_cells': MIN_REP_CELLS,
                                      'min_corr_cells': MIN_CORR_CELLS})
    pd.set_option('display.width', 250)
    print(summ[['key', 'stat', 'estimate', 'lo', 'hi', 'drugs']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
