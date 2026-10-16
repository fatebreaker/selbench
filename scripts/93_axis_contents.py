"""Round 2 (WS6b), task 4 (editor item 9; R2-Major 2): what the general-sensitivity axis contains.

Run with the analysis interpreter (../../.venv/bin/python; BeatAML clinical files are .xlsx). Uses the frozen
07_confirm.build and 72_pdm_confirm (Amendment 3) read-only, as 54 and 75 do; nothing frozen is run or written.

A. Pre-specified pan-sensitivity determinants (R2-Major 2; fixed before looking): BCL2L1, NQO1, ABCB1 and SLFN11
   expression (DepMap 24Q4 log2(TPM+1), data/processed/features/expression_all.parquet), TP53 mutation
   (features/mutation.parquet, 0/1) and the CRISPR-inferred growth rate (features/growth_rate.csv). Lineage: DepMap
   OncotreeLineage (features/lineage.parquet). Axes: results/revision_axis/axes_native.csv, the four discovery screens'
   all-drug and active-drug axes and GDSC1/GDSC2 ln IC50 axes. Per axis, on the cell lines with the axis, all six
   markers and a lineage: Spearman correlation with each marker across all lines and within lineage (correlation of
   ranks after removing lineage means); variance explained (OLS R^2) by the six markers jointly, by lineage, by both,
   and by the markers beyond lineage (adjusted R^2; percentile bootstrap intervals of in-sample R^2 are biased upwards,
   most for the lineage indicators, so the point estimates are primary). Per pair of discovery screens (same axis type, common lines): Pearson r, partial
   r given the six markers, given lineage and given both, and the share of the covariance accounted for by the markers
   (1 - cov of residuals / cov). Cell-line bootstrap (B = 1,000; the axes are held fixed).
B. BeatAML axis without FLT3-active and broad multi-kinase drugs. Rule (stated before computing): an inhibitor is
   excluded if BeatAML's own drug-gene table (beataml_drug_families.xlsx, sheet drug_gene) lists FLT3 among its targets
   or lists >= 10 distinct target genes (broad multi-kinase). The axis is the mean rank-normal response over the
   remaining drugs (54's statistic: Z.mean over drugs, registered processing, >= 30 samples per drug). FLT3-ITD
   association: difference in mean axis between FLT3-ITD-positive and -negative samples in axis s.d. units (54), with
   Mann-Whitney P and a sample bootstrap (B = 2,000) interval, for the all-drug axis (paper +0.72), the restricted axis,
   and the excluded drugs alone; paired difference all - restricted. Also the selective transfer of the excluded drugs'
   frozen predictions given the registered leave-drug-out axis versus the restricted axis.
C. Axis transfer (does the cell-line-predicted axis predict the patient-derived axis?). The drug-agnostic prediction
   of 72 (per sample, the mean over the cohort's predicted compounds of the frozen pred_raw standardised within compound)
   (1) correlated across samples with the cohort's measured axis (mean rank-normal response over drugs), and (2) scored
   per drug as a raw correlation with each drug's response (72's 'agnostic' family; organoids 0.05, -0.08 to 0.20),
   mean over drugs. BeatAML (samples; patient-cluster bootstrap), PDX_Bruna (models; tumour-cluster bootstrap, G1),
   organoids (liver, sarcoma; corrected orientation; patient bootstrap within dataset). B = 2,000.
D. Organoid liver transfer with histotype partialled out (R2-Major 3c): reproducible liver units, selective transfer
   r(p, z | g) and r(p, z | g, histotype indicators) (histotype = CoderData cancer_type of the patient's samples);
   pooled as registered (precision-weighted Fisher z, weights n - 4) with a patient bootstrap (2,000 draws).
Outputs (results/revision_round2/axis_contents/): marker_associations.csv, marker_variance.csv, pair_agreement.csv,
beataml_flt3.csv, beataml_flt3_drugs.csv, axis_transfer.csv, organoid_histotype.csv, summary.json.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, rankdata  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OWN_RAW, PROCESSED, RESULTS  # noqa: E402

OUT = RESULTS / 'revision_round2' / 'axis_contents'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261093
B = 1000
B2 = 2000
MARKERS = ['BCL2L1', 'NQO1', 'ABCB1', 'SLFN11', 'TP53_mut', 'growth_rate']
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _module('pdm_confirm_ro93', '72_pdm_confirm.py')      # frozen; read-only (as 75)
CONFIRM = C.confirm                                         # frozen 07_confirm, loaded by 72


def build(long, min_cells):
    old = CONFIRM.MIN_CELLS
    CONFIRM.MIN_CELLS = min_cells
    try:
        return CONFIRM.build(long)
    finally:
        CONFIRM.MIN_CELLS = old


def ci(v, level=95):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    a = (100 - level) / 2
    return float(np.percentile(v, a)), float(np.percentile(v, 100 - a))


def summ(point, reps, name):
    lo, hi = ci(reps)
    return {name: float(point), f'{name}_lo': lo, f'{name}_hi': hi}


# ----------------------------------------------------------------------------------------------------------------
def resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    return y - X1 @ b


def r2(y, X, adjusted=True):
    """OLS R^2 (adjusted for the number of predictors by default)."""
    e = resid(y, X)
    yc = y - y.mean()
    r = 1 - (e @ e) / (yc @ yc)
    if adjusted:
        n, k = len(y), np.linalg.matrix_rank(np.column_stack([np.ones(len(y)), X])) - 1
        r = 1 - (1 - r) * (n - 1) / (n - k - 1)
    return r


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return (a @ b) / den if den > 1e-12 else np.nan


def dummies(lab):
    u, inv = np.unique(lab, return_inverse=True)
    D = np.zeros((len(lab), len(u)))
    D[np.arange(len(lab)), inv] = 1
    return D[:, 1:] if len(u) > 1 else np.zeros((len(lab), 0))


def marker_frame():
    F = PROCESSED / 'features'
    e = pd.read_parquet(F / 'expression_all.parquet', columns=['BCL2L1', 'NQO1', 'ABCB1', 'SLFN11'])
    mut = pd.read_parquet(F / 'mutation.parquet', columns=['TP53']).TP53.rename('TP53_mut')
    gr = pd.read_csv(F / 'growth_rate.csv', index_col=0).growth_rate
    lin = pd.read_parquet(F / 'lineage.parquet')
    lab = lin.idxmax(axis=1).where(lin.sum(axis=1) > 0)
    M = e.join(mut, how='outer').join(gr, how='outer')
    return M[MARKERS], lab.rename('lineage')


def part_a():
    M, lab = marker_frame()
    A = pd.read_csv(RESULTS / 'revision_axis' / 'axes_native.csv').set_index('depmap_id')
    axes = [f'{s}|{k}' for s in SCREENS for k in ('all', 'active')] + ['GDSC1_lnIC50|all', 'GDSC2_lnIC50|all']
    rng = np.random.default_rng(SEED)
    assoc, var = [], []
    for ax in axes:
        d = A[[ax]].join(M).join(lab).dropna()
        n = len(d)
        y = d[ax].values
        X = d[MARKERS].values
        L = dummies(d.lineage.values)
        idx = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(B)]
        res = {k: [] for k in ['r2_markers', 'r2_lineage', 'r2_both', 'r2_markers_beyond_lineage']}
        sp_all = {m: [] for m in MARKERS}
        sp_within = {m: [] for m in MARKERS}
        for ii in idx:
            yy, XX, LL = y[ii], X[ii], L[ii]
            keep = LL.std(0) > 0
            LL = LL[:, keep]
            ry = rankdata(yy)
            ryl = resid(ry, LL)
            for j, m in enumerate(MARKERS):
                rx = rankdata(XX[:, j])
                sp_all[m].append(corr(ry, rx))
                sp_within[m].append(corr(ryl, resid(rx, LL)))
            a, b_ = r2(yy, XX), r2(yy, LL)
            c = r2(yy, np.column_stack([XX, LL]))
            res['r2_markers'].append(a)
            res['r2_lineage'].append(b_)
            res['r2_both'].append(c)
            res['r2_markers_beyond_lineage'].append((c - b_) / (1 - b_))
        for m in MARKERS:
            assoc.append({'axis': ax, 'marker': m, 'cells': n, **summ(sp_all[m][0], sp_all[m][1:], 'spearman_all'),
                          **summ(sp_within[m][0], sp_within[m][1:], 'spearman_within_lineage')})
        vrow = {'axis': ax, 'cells': n, 'lineages': int(d.lineage.nunique()),
                **{k2: v2 for k, v in res.items() for k2, v2 in summ(v[0], v[1:], k).items()}}
        for k in res:     # basic (reverse-percentile) interval: corrects the upward bias of bootstrapped in-sample R^2
            vrow[f'{k}_lo_basic'] = 2 * vrow[k] - vrow[f'{k}_hi']
            vrow[f'{k}_hi_basic'] = 2 * vrow[k] - vrow[f'{k}_lo']
        var.append(vrow)
        print('A', ax, n, round(res['r2_markers'][0], 3), flush=True)
    pairs = []
    for kind in ['active', 'all']:
        for i, s1 in enumerate(SCREENS):
            for s2 in SCREENS[i + 1:]:
                pairs.append((f'{s1}|{kind}', f'{s2}|{kind}'))
    pairs.append(('GDSC1_lnIC50|all', 'GDSC2_lnIC50|all'))
    prow = []
    for a1, a2 in pairs:
        d = A[[a1, a2]].join(M).join(lab).dropna()
        n = len(d)
        y1, y2, X, L = d[a1].values, d[a2].values, d[MARKERS].values, dummies(d.lineage.values)
        idx = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(B)]
        out = {k: [] for k in ['r', 'partial_markers', 'partial_lineage', 'partial_both', 'cov_share_markers',
                               'cov_share_lineage', 'cov_share_both']}
        for ii in idx:
            a, b_, XX, LL = y1[ii], y2[ii], X[ii], L[ii]
            LL = LL[:, LL.std(0) > 0]
            out['r'].append(corr(a, b_))
            c0 = np.cov(a, b_)[0, 1]
            for lab_, Cv in [('markers', XX), ('lineage', LL), ('both', np.column_stack([XX, LL]))]:
                ea, eb = resid(a, Cv), resid(b_, Cv)
                out[f'partial_{lab_}'].append(corr(ea, eb))
                out[f'cov_share_{lab_}'].append(1 - np.mean(ea * eb) * n / (n - 1) / c0)
        prow.append({'axis_a': a1, 'axis_b': a2, 'cells': n,
                     **{k2: v2 for k, v in out.items() for k2, v2 in summ(v[0], v[1:], k).items()}})
        print('A pair', a1, a2, n, round(out['r'][0], 3), round(out['partial_markers'][0], 3), flush=True)
    return pd.DataFrame(assoc), pd.DataFrame(var), pd.DataFrame(prow)


# ----------------------------------------------------------------------------------------------------------------
def beataml_inputs():
    long = pd.read_csv(PROCESSED / 'beataml' / 'beataml_long.csv.gz')
    smp = pd.Index(long['sample'].unique())
    mp = pd.read_excel(OWN_RAW / 'beataml_inputs' / 'beataml_waves1to4_sample_mapping.xlsx', sheet_name='sample_map')
    mp = mp.dropna(subset=['dbgap_rnaseq_sample']).drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
    clin = pd.read_excel(OWN_RAW / 'beataml_inputs' / 'beataml_wv1to4_clinical.xlsx', sheet_name='summary')
    clin = clin.dropna(subset=['dbgap_rnaseq_sample']).drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
    patient = mp.dbgap_subject_id.reindex(smp)
    return long, patient, clin.reindex(smp)


def itd_effect(axis, pos):
    """(mean pos - mean neg) / sd(axis); axis, pos aligned arrays (pos 0/1, NaN excluded beforehand)."""
    return (axis[pos == 1].mean() - axis[pos == 0].mean()) / axis.std(ddof=1)


def partial_rpz(p, z, g):
    m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
    if m.sum() < 20:
        return np.nan, int(m.sum())
    return corr(resid(p[m], g[m][:, None]), resid(z[m], g[m][:, None])), int(m.sum())


def part_b():
    long, patient, clin = beataml_inputs()
    M, Z, G, R = build(long.rename(columns={'sample': 'depmap_id'}), 30)
    dm = pd.read_csv(PROCESSED / 'beataml' / 'drug_mapping.csv')
    names = {}
    for r in dm.itertuples():
        names.setdefault(r.compound, set()).add(r.source_name)
    for c in Z.index:
        if c.startswith('BA:'):
            names.setdefault(c, set()).add(c[3:])
    dg = pd.read_excel(OWN_RAW / 'beataml_inputs' / 'beataml_drug_families.xlsx', sheet_name='drug_gene')
    tg = dg.groupby('inhibitor').Symbol.apply(lambda s: set(s.dropna()))
    rows = []
    for c in Z.index:
        t = set().union(*[tg.get(nm, set()) for nm in names.get(c, set())])
        has = any(nm in tg.index for nm in names.get(c, set()))
        rows.append({'compound': c, 'names': '|'.join(sorted(names.get(c, set()))), 'has_target_annotation': has,
                     'n_targets': len(t), 'flt3_target': 'FLT3' in t, 'multikinase_ge10': len(t) >= 10})
    D = pd.DataFrame(rows)
    D['excluded'] = D.flt3_target | D.multikinase_ge10
    excl = set(D[D.excluded].compound)
    samples = list(Z.columns)
    cl = clin.reindex(samples)
    itd = cl['FLT3-ITD'].map({'positive': 1, 'negative': 0}).values
    axes = {'all drugs (paper)': Z.mean(axis=0).values,
            'without FLT3-active and multi-kinase drugs': Z.loc[[d for d in Z.index if d not in excl]].mean(axis=0).values,
            'FLT3-active and multi-kinase drugs only': Z.loc[[d for d in Z.index if d in excl]].mean(axis=0).values}
    ok = np.isfinite(itd) & np.all([np.isfinite(v) for v in axes.values()], axis=0)
    rng = np.random.default_rng(SEED + 1)
    pos = np.where(ok)[0]
    boots = [pos] + [rng.choice(pos, len(pos), replace=True) for _ in range(B2)]
    eff = {k: np.array([itd_effect(v[b], itd[b]) for b in boots]) for k, v in axes.items()}
    out = []
    # the paper's statistic on every sample with the all-drug axis (54: 519 samples)
    ok_all = np.isfinite(itd) & np.isfinite(axes['all drugs (paper)'])
    pa = np.where(ok_all)[0]
    ea = np.array([itd_effect(axes['all drugs (paper)'][b], itd[b]) for b in
                   [pa] + [rng.choice(pa, len(pa), replace=True) for _ in range(B2)]])
    a_ = axes['all drugs (paper)'][ok_all]
    out.append({'axis': 'all drugs (paper; every sample with the axis)', 'drugs': len(Z.index), 'samples': int(ok_all.sum()),
                'itd_positive': int((itd[ok_all] == 1).sum()), **summ(ea[0], ea[1:], 'flt3_itd_effect_sd'),
                'mannwhitney_p': float(mannwhitneyu(a_[itd[ok_all] == 1], a_[itd[ok_all] == 0],
                                                    alternative='two-sided').pvalue)}) 
    for k, v in axes.items():
        a = v[ok]
        p = mannwhitneyu(a[itd[ok] == 1], a[itd[ok] == 0], alternative='two-sided').pvalue
        out.append({'axis': k, 'drugs': int({'all drugs (paper)': len(Z.index),
                                              'without FLT3-active and multi-kinase drugs': len(Z.index) - len(excl),
                                              'FLT3-active and multi-kinase drugs only': len(excl)}[k]),
                    'samples': int(ok.sum()), 'itd_positive': int((itd[ok] == 1).sum()),
                    **summ(eff[k][0], eff[k][1:], 'flt3_itd_effect_sd'), 'mannwhitney_p': float(p),
                    'r_with_all_drug_axis': float(corr(v[ok], axes['all drugs (paper)'][ok])),
                    'note': 'samples with all three axes and FLT3-ITD status (paired comparison)'})
    d = eff['all drugs (paper)'] - eff['without FLT3-active and multi-kinase drugs']
    out.append({'axis': 'difference: all drugs minus restricted', **summ(d[0], d[1:], 'flt3_itd_effect_sd')})
    # selective transfer of the excluded drugs' frozen predictions: registered leave-drug-out axis vs restricted axis
    P = pd.read_parquet(RESULTS / 'beataml_transfer' / 'pred_sel.parquet').reindex(columns=samples)
    g_lto = axes['without FLT3-active and multi-kinase drugs']
    drows = []
    for c in sorted(excl):
        if c not in P.index:
            continue
        p, z, g = P.loc[c].values.astype(float), Z.loc[c].values.astype(float), G.loc[c].values.astype(float)
        r_reg, n = partial_rpz(p, z, g)
        r_lto, _ = partial_rpz(p, z, g_lto)
        raw = corr(*[x[np.isfinite(p) & np.isfinite(z)] for x in (p, z)]) if (np.isfinite(p) & np.isfinite(z)).sum() >= 20 else np.nan
        drows.append({'compound': c, 'names': D.set_index('compound').names.get(c), 'n': n, 'raw': raw,
                      'selective_registered_axis': r_reg, 'selective_restricted_axis': r_lto,
                      'flt3_target': bool(D.set_index('compound').flt3_target.get(c))})
    DR = pd.DataFrame(drows)
    info = {'drugs': len(Z.index), 'samples': len(samples), 'excluded': len(excl),
            'excluded_flt3': int(D.flt3_target.sum()), 'excluded_multikinase_only': int((D.multikinase_ge10 & ~D.flt3_target).sum()),
            'drugs_with_target_annotation': int(D.has_target_annotation.sum()),
            'excluded_with_frozen_prediction': int(len(DR)),
            'median_selective_registered_axis': float(DR.selective_registered_axis.median()) if len(DR) else None,
            'median_selective_restricted_axis': float(DR.selective_restricted_axis.median()) if len(DR) else None,
            'median_selective_flt3_registered': float(DR[DR.flt3_target].selective_registered_axis.median()) if len(DR) else None,
            'median_selective_flt3_restricted': float(DR[DR.flt3_target].selective_restricted_axis.median()) if len(DR) else None}
    print('B', json.dumps(info), flush=True)
    return pd.DataFrame(out), D.merge(DR, on=['compound', 'names'], how='left'), info, (long, patient, Z)


# ----------------------------------------------------------------------------------------------------------------
def agnostic(P):
    return C.agnostic(P)


def transfer_stats(Z, agn, W, drugs=None):
    """W: (R, n) sample weights (row 0 = ones). Returns per-draw r(agnostic, axis) and mean over drugs of the
    weighted raw r(agnostic, z_d) (drugs with >= 10 samples having both)."""
    axis = Z.mean(axis=0).values
    a = agn.reindex(Z.columns).values.astype(float)
    m = np.isfinite(a) & np.isfinite(axis)

    def wcorr(x, y, w):
        s = w.sum(1)
        mx, my = (w @ x) / s, (w @ y) / s
        cxy = (w @ (x * y)) / s - mx * my
        cxx = (w @ (x * x)) / s - mx ** 2
        cyy = (w @ (y * y)) / s - my ** 2
        return cxy / np.sqrt(cxx * cyy)
    r_axis = wcorr(a[m], axis[m], W[:, m])
    per = []
    for d in (Z.index if drugs is None else [x for x in Z.index if x in set(drugs)]):
        z = Z.loc[d].values.astype(float)
        mm = np.isfinite(z) & np.isfinite(a)
        if mm.sum() < 10:
            continue
        per.append(wcorr(a[mm], z[mm], W[:, mm]))
    per = np.vstack(per)
    return r_axis, np.nanmean(per, axis=0), per.shape[0], int(m.sum()), per


def cluster_weights(clusters, rng, R):
    u, inv = np.unique(clusters, return_inverse=True)
    W = [np.ones(len(clusters))]
    for _ in range(R):
        cnt = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        W.append(cnt[inv].astype(float))
    return np.vstack(W)


def part_c(bea):
    rows = []
    rng = np.random.default_rng(SEED + 2)
    long, patient, Z = bea
    P = pd.read_parquet(RESULTS / 'beataml_transfer' / 'pred_raw.parquet')
    agn = agnostic(P.reindex(index=[d for d in Z.index if d in P.index]))
    cols = [c for c in Z.columns if c in agn.dropna().index]
    Zs = Z[cols]
    W = cluster_weights(patient.reindex(cols).values, rng, B2)
    r_axis, mraw, nd, ns, _ = transfer_stats(Zs, agn, W, drugs=P.index)
    rows.append({'cohort': 'BeatAML', 'unit': 'sample (patient-cluster bootstrap)', 'samples': ns, 'drugs': nd,
                 'compounds_in_agnostic': int(sum(d in P.index for d in Z.index)),
                 **summ(r_axis[0], r_axis[1:], 'r_agnostic_vs_measured_axis'),
                 **summ(mraw[0], mraw[1:], 'mean_raw_r_agnostic_per_drug')})
    # PDX_Bruna (model level, tumour-cluster bootstrap G1)
    pl = pd.read_csv(PROCESSED / 'pdx_bruna' / 'pdx_long.csv.gz')
    Mm, Zm, Gm, Rm = build(pl.rename(columns={'sample': 'depmap_id'}), 20)
    Pp = pd.read_parquet(RESULTS / 'pdx_transfer' / 'pred_raw.parquet')
    agn = agnostic(Pp.reindex(index=[d for d in Zm.index if d in Pp.index]))
    cols = [c for c in Zm.columns if c in agn.dropna().index]
    tum = np.array([c.split('-')[0] for c in cols])
    W = cluster_weights(tum, rng, B2)
    r_axis, mraw, nd, ns, _ = transfer_stats(Zm[cols], agn, W, drugs=Pp.index)
    rows.append({'cohort': 'PDX_Bruna', 'unit': 'model (tumour-cluster bootstrap, G1)', 'samples': ns, 'drugs': nd,
                 'compounds_in_agnostic': int(sum(d in Pp.index for d in Zm.index)), 'tumours': int(len(set(tum))),
                 **summ(r_axis[0], r_axis[1:], 'r_agnostic_vs_measured_axis'),
                 **summ(mraw[0], mraw[1:], 'mean_raw_r_agnostic_per_drug')})
    # organoids (corrected orientation), per dataset and pooled over units
    pooled_units = []
    for ds in ['liver', 'sarcoma']:
        path = PROCESSED / 'pdm_corrected' / ds / 'patient_long.csv.gz'
        built = C.build(path)
        if built is None:
            continue
        Zo, Glo = built
        pr = C.patient_predictions(ds, 'raw')
        agn = agnostic(pr)
        cols = [c for c in Zo.columns if c in agn.dropna().index]
        W = np.vstack([np.ones(len(cols))] + [np.bincount(rng.integers(0, len(cols), len(cols)), minlength=len(cols))
                                              .astype(float) for _ in range(B2)])
        r_axis, mraw, nd, ns, per = transfer_stats(Zo[cols], agn, W, drugs=pr.index)
        pooled_units.append(per)
        rows.append({'cohort': f'organoids ({ds})', 'unit': 'patient (patient bootstrap)', 'samples': ns, 'drugs': nd,
                     'compounds_in_agnostic': int(pr.shape[0]),
                     **summ(r_axis[0], r_axis[1:], 'r_agnostic_vs_measured_axis'),
                     **summ(mraw[0], mraw[1:], 'mean_raw_r_agnostic_per_drug')})
    # pooled over both collections' units (patients resampled within dataset, independent draws), as 72's estimate
    allu = np.nanmean(np.vstack(pooled_units), axis=0)
    rows.append({'cohort': 'organoids (liver + sarcoma units)', 'unit': 'patient (patient bootstrap within dataset)',
                 'drugs': int(sum(p_.shape[0] for p_ in pooled_units)),
                 **summ(allu[0], allu[1:], 'mean_raw_r_agnostic_per_drug')})
    # check against 72's registered secondary estimate (organoid agnostic family, all 65 units)
    U = pd.read_csv(RESULTS / 'confirmation_PDM_corrected' / 'units.csv')
    chk = float(U[U.family == 'agnostic'].raw.mean())
    return pd.DataFrame(rows), {'organoid_agnostic_mean_raw_units_csv': chk}


# ----------------------------------------------------------------------------------------------------------------
def part_d():
    built = C.build(PROCESSED / 'pdm_corrected' / 'liver' / 'patient_long.csv.gz')
    Z, Gl = built
    preds = {f: C.patient_predictions('liver', f) for f in ['sel', 'raw']}
    D = C.Dataset('liver', Z, Gl, preds, C.classes())
    s = pd.read_csv(OWN_RAW / 'pdm_inputs' / 'liver_samples.csv', dtype=str)
    s['patient'] = s.common_name.map(C.loader.patient_key)
    hist = s.groupby('patient').cancer_type.agg(lambda x: x.value_counts().index[0])
    hv = hist.reindex(D.cols).fillna('unknown').values
    units = [u for u in D.units if u['class'] == 'reproducible']
    rng = np.random.default_rng(SEED + 3)
    n_pat = len(D.cols)
    counts = [np.ones(n_pat)] + [np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat).astype(float)
                                 for _ in range(B2)]

    def wpartial(p, z, Cm, w):
        X = np.column_stack([np.ones(len(p)), Cm])
        sw = np.sqrt(w)
        bp = np.linalg.lstsq(X * sw[:, None], p * sw, rcond=None)[0]
        bz = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)[0]
        ep, ez = p - X @ bp, z - X @ bz
        den = np.sqrt(np.sum(w * ep * ep) * np.sum(w * ez * ez))
        return np.sum(w * ep * ez) / den if den > 1e-12 else np.nan
    rows = []
    res = {'registered': [], 'histotype_adjusted': []}
    for u in units:
        p = D.P['sel'].loc[u['compound']].values.astype(float)[u['idx']]
        z, g = u['z'], u['g']
        h = dummies(hv[u['idx']])
        r0 = [wpartial(p, z, g[:, None], c[u['idx']]) for c in counts]
        r1 = [wpartial(p, z, np.column_stack([g, h]), c[u['idx']]) for c in counts]
        res['registered'].append(r0)
        res['histotype_adjusted'].append(r1)
        nn = [c[u['idx']].sum() for c in counts]
        rows.append({'compound': u['compound'], 'n': len(u['idx']), 'selective': r0[0], 'selective_histotype_adjusted': r1[0],
                     'histotypes_present': int(len(np.unique(hv[u['idx']])))})
        res.setdefault('n', []).append(nn)
    Nw = np.array(res['n'])                                   # (units, draws)

    def pooled(Rm):
        Rm = np.array(Rm)
        zf = np.arctanh(np.clip(Rm, -0.999999, 0.999999))
        w = Nw - 4
        ok = np.isfinite(zf) & (w > 0)
        return np.tanh((np.where(ok, zf * w, 0)).sum(0) / np.where(ok, w, 0).sum(0))
    p0, p1 = pooled(res['registered']), pooled(res['histotype_adjusted'])
    U = pd.read_csv(RESULTS / 'confirmation_PDM_corrected' / 'units.csv')
    uu = U[(U.dataset == 'liver') & (U.family == 'sel') & (U['class'] == 'reproducible')].set_index('compound').selective
    T = pd.DataFrame(rows)
    chk = float(np.nanmax(np.abs(T.set_index('compound').selective - uu.reindex(T.compound).values)))
    out = {'units': len(units), 'patients': n_pat, 'histotype_counts': pd.Series(hv).value_counts().to_dict(),
           **summ(p0[0], p0[1:], 'pooled_selective'), **summ(p1[0], p1[1:], 'pooled_selective_histotype_adjusted'),
           **summ(p1[0] - p0[0], (p1 - p0)[1:], 'difference_adjusted_minus_registered'),
           'check_max_abs_diff_vs_units_csv': chk}
    print('D', json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}), flush=True)
    return T, out


def main():
    assoc, var, pairs = part_a()
    assoc.to_csv(OUT / 'marker_associations.csv', index=False, float_format='%.5g')
    var.to_csv(OUT / 'marker_variance.csv', index=False, float_format='%.5g')
    pairs.to_csv(OUT / 'pair_agreement.csv', index=False, float_format='%.5g')
    bt, bd, binfo, bea = part_b()
    bt.to_csv(OUT / 'beataml_flt3.csv', index=False, float_format='%.5g')
    bd.to_csv(OUT / 'beataml_flt3_drugs.csv', index=False, float_format='%.5g')
    tr, tinfo = part_c(bea)
    tr.to_csv(OUT / 'axis_transfer.csv', index=False, float_format='%.5g')
    ht, hinfo = part_d()
    ht.to_csv(OUT / 'organoid_histotype.csv', index=False, float_format='%.5g')

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    (OUT / 'summary.json').write_text(json.dumps(conv({
        'B_cells': B, 'B_patients': B2, 'markers': MARKERS,
        'flt3_rule': 'excluded if BeatAML drug_gene lists FLT3 among the targets or >= 10 distinct target genes',
        'beataml': binfo, 'transfer_checks': tinfo, 'organoid_histotype': hinfo}), indent=1))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 200)
    print(var.round(3).to_string(index=False))
    print(pairs[['axis_a', 'axis_b', 'cells', 'r', 'r_lo', 'r_hi', 'partial_markers', 'partial_markers_lo',
                 'partial_markers_hi', 'cov_share_markers', 'partial_lineage', 'partial_both']].round(3).to_string(index=False))
    print(bt.round(3).to_string(index=False))
    print(tr.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
