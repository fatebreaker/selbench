"""Patient-derived transfer tests at the level of independent biological units (R2 Majors 6 and 10; R3 M5).

PDX_Bruna (H8/H9). The 37 PDX-derived cultures are derivatives of fewer PDX tumours. Tumour rule: the tumour
identifier is the model name before the first hyphen (HCI005-X2C -> HCI005; STG282-X1CR1 -> STG282); the suffix
encodes the xenograft passage (X<n>) and culture derivation (C, CR, CR1). Grouping G1 keeps STG139M (by its name a
metastasis-derived line, plausibly from the same patient as STG139) as its own tumour; G2 merges it with STG139.
Analyses (all with the registered drug classes, the frozen predictions pred_sel, and the frozen 07_confirm.build
processing imported read-only):
  * registered model-level H8/H9 (reproduced) and naive model-label permutation;
  * T1 (primary tumour-level): raw sensitivities averaged over each tumour's derivatives, then the same processing as
    the registered analysis on the registered 80-drug panel (rank-normal within drug over tumours; leave-drug-out
    tumour fitness axis); predictions averaged over each tumour's expressed derivatives; per-drug selective partial
    correlation over tumours with both (>= 10; every H8/H9 drug has 11-15). Tests: one-sided Wilcoxon (H8) and
    Mann-Whitney (H9) as registered (scipy defaults), 5,000 permutations of tumour labels (expressed tumours
    exchangeable; statistics median and difference of medians), drug-bootstrap and drug x tumour bootstrap CIs, and a
    random-effects meta-analysis on Fisher z with tau^2;
  * T2: registered model-level z, axis and prediction averaged within tumour over the derivatives that have both;
  * T3: one derivative per tumour, drawn at random (1,000 draws);
  * conservative tumour-block permutation at the registered model level: tumour-mean predictions broadcast to
    derivatives, tumours permuted (observed statistic uses the same broadcast predictions), plus a variant that keeps
    sample-specific predictions and permutes tumours only within block-size strata;
  * tumour-cluster bootstrap CI for the registered H8 median; drug-bootstrap CI for the H8 median;
  * class-level test (drugs aggregated by mechanism class first; old and curated classes);
  * the ln IC50 secondary endpoint at tumour level; within- versus between-tumour similarity and ICC.
BeatAML (H6/H7). 520 samples from 479 patients (dbgap_subject_id of the sample-mapping file). Patient-block
permutations (sample-specific predictions within block-size strata; patient-mean predictions broadcast), a
patient-averaged reanalysis, class-level tests, random-effects estimates, composition and a sensitivity analysis
excluding remission, relapse and non-AML samples, correlates of the BeatAML fitness axis, and per-drug transfer with
patient-bootstrap CIs and empirical-Bayes shrinkage.
Outputs: results/revision_transfer/pdx_*.csv, beataml_*.csv, 54_summary.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, spearmanr, t as tdist, wilcoxon

from common import OWN_RAW, PROCESSED, RESULTS, write_json

HERE = Path(__file__).resolve().parent
OUT = RESULTS / 'revision_transfer'
SEED = 20261015
N_PERM = 5000
N_BOOT = 10000
N_BOOT_TWOWAY = 2000
N_DRAWS = 1000
MIN_TUMOURS = 10
REPRODUCIBLE, NONREPRODUCIBLE = 0.30, 0.15


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PDXC = _load('pdx_confirm_frozen', '31_pdx_confirm.py')   # frozen; loads 07_confirm as PDXC.confirm
CONFIRM = PDXC.confirm


def build(long, min_cells):
    """Frozen 07_confirm.build with the given drug-inclusion threshold (module global restored afterwards)."""
    old = CONFIRM.MIN_CELLS
    CONFIRM.MIN_CELLS = min_cells
    try:
        return CONFIRM.build(long)
    finally:
        CONFIRM.MIN_CELLS = old


def rng_for(*keys):
    return np.random.default_rng([SEED, *keys])


# ------------------------------------------------------------------ partial-correlation engine
class DrugDesign:
    """Fixed per-drug quantities for the selective partial correlation r(p, z | g) over units with z, g and p."""

    def __init__(self, z, g, has_pred, min_n):
        m = np.isfinite(z) & np.isfinite(g) & has_pred
        self.ok = m.sum() >= min_n
        self.idx = np.where(m)[0]
        self.n = int(m.sum())
        if not self.ok:
            return
        X = np.c_[np.ones(self.n), g[self.idx]]
        y = z[self.idx]
        self.ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        self.nry = np.sqrt(self.ry @ self.ry)
        self.X = X
        self.XtXi = np.linalg.inv(X.T @ X)
        self.ok = self.ry.std() >= 1e-12

    def partial(self, Pp):
        """Pp: (B x n) predictions on self.idx (one row per permutation). Returns (B,) partial correlations."""
        Pc = Pp - Pp.mean(axis=1, keepdims=True)
        num = Pc @ self.ry
        XtP = Pc @ self.X
        quad = np.einsum('bi,bi->b', Pc, Pc) - np.einsum('bi,ij,bj->b', XtP, self.XtXi, XtP)
        with np.errstate(invalid='ignore', divide='ignore'):
            r = num / (np.sqrt(np.clip(quad, 0, None)) * self.nry)
        r[quad <= 1e-24] = np.nan
        return r


def designs_for(Z, G, has_pred, drugs, min_n):
    return {d: DrugDesign(Z.loc[d].values.astype(float), G.loc[d].values.astype(float), has_pred, min_n) for d in drugs}


def perm_matrix(designs, values, index, drugs):
    """values: dict drug -> 1-D array of prediction values; index: (B x n_units) gather indices into values.
    Returns DataFrame-like array (B x D) of partial correlations."""
    out = np.full((index.shape[0], len(drugs)), np.nan)
    for j, d in enumerate(drugs):
        des = designs[d]
        if not des.ok:
            continue
        out[:, j] = des.partial(values[d][index[:, des.idx]])
    return out


def unit_permutations(n_units, movable, B, rng):
    """(B x n_units) permutations that shuffle only the positions in `movable` among themselves."""
    idx = np.tile(np.arange(n_units), (B, 1))
    movable = np.asarray(movable)
    for b in range(B):
        idx[b, movable] = movable[rng.permutation(len(movable))]
    return idx


def block_broadcast_index(unit_cluster, clusters, B, rng, identity=False):
    """Cluster-level permutation with cluster-mean values broadcast to member units. unit_cluster: array of cluster
    positions (index into `clusters`, -1 for units without a prediction). Returns (B x n_units) indices into the
    cluster-mean value vector (units without a prediction point to a NaN slot at the end)."""
    K = len(clusters)
    out = np.empty((B, len(unit_cluster)), dtype=int)
    for b in range(B):
        sigma = np.arange(K) if identity else rng.permutation(K)
        out[b] = np.where(unit_cluster >= 0, sigma[np.clip(unit_cluster, 0, None)], K)
    return out


def stratified_block_permutations(unit_cluster, B, rng):
    """Permute clusters only among clusters with the same number of member units; a cluster's units receive the
    source cluster's units in random order. Units with unit_cluster == -1 stay in place."""
    n = len(unit_cluster)
    members = {}
    for i, c in enumerate(unit_cluster):
        if c >= 0:
            members.setdefault(c, []).append(i)
    by_size = {}
    for c, m in members.items():
        by_size.setdefault(len(m), []).append(m)
    out = np.tile(np.arange(n), (B, 1))
    for size, mem in sorted(by_size.items()):
        mem = np.array(mem)                                   # (clusters x size) unit positions
        k = len(mem)
        for b in range(B):
            src = mem[rng.permutation(k)]                     # source cluster for every target cluster
            order = np.argsort(rng.random((k, size)), axis=1)  # random within-block order
            out[b, mem.ravel()] = np.take_along_axis(src, order, axis=1).ravel()
    return out


def ref_partial(x, y, g, min_n):
    """The frozen 31_pdx_confirm.partial logic with an explicit minimum number of units."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(g)
    if m.sum() < min_n:
        return np.nan, int(m.sum())
    X = np.c_[np.ones(m.sum()), g[m]]
    rx = x[m] - X @ np.linalg.lstsq(X, x[m], rcond=None)[0]
    ry = y[m] - X @ np.linalg.lstsq(X, y[m], rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return np.nan, int(m.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


def weighted_partial(W, p, z, g):
    """W: (B x n) non-negative weights; returns (B,) weighted partial correlations r(p, z | g)."""
    S0 = W.sum(axis=1)

    def m(a):
        return (W @ a) / S0

    mp, mz, mg = m(p), m(z), m(g)
    cpz = m(p * z) - mp * mz; cpg = m(p * g) - mp * mg; czg = m(z * g) - mz * mg
    vp = m(p * p) - mp ** 2; vz = m(z * z) - mz ** 2; vg = m(g * g) - mg ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        rpz = cpz / np.sqrt(vp * vz); rpg = cpg / np.sqrt(vp * vg); rzg = czg / np.sqrt(vz * vg)
        r = (rpz - rpg * rzg) / np.sqrt((1 - rpg ** 2) * (1 - rzg ** 2))
    distinct = (W > 0).sum(axis=1)
    r[distinct < 6] = np.nan
    return r


# ------------------------------------------------------------------ statistics
def one_sided_tests(a, b=None):
    a = pd.Series(a).dropna()
    out = {'n_a': int(len(a)), 'median_a': float(a.median()) if len(a) else np.nan,
           'mean_a': float(a.mean()) if len(a) else np.nan,
           'n_negative_a': int((a < 0).sum())}
    if len(a) >= 8:
        out['wilcoxon_p'] = float(wilcoxon(a, alternative='greater').pvalue)
    if b is not None:
        b = pd.Series(b).dropna()
        out.update({'n_b': int(len(b)), 'median_b': float(b.median()) if len(b) else np.nan})
        if len(a) >= 3 and len(b) >= 3:
            out['mwu_p'] = float(mannwhitneyu(a, b, alternative='greater').pvalue)
    return out


def perm_p(null, obs):
    null = np.asarray(null)
    null = null[np.isfinite(null)]
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


def _fz(x):
    return np.arctanh(np.clip(x, -0.999999, 0.999999))


def perm_pvals(obs, null, a, b, A, Bn, suffix=''):
    """Permutation P values for the transfer (A: median and mean Fisher z over class-a drugs) and contrast
    (Bn: difference of medians and of mean Fisher z, class a minus class b) statistics."""
    o_med = np.nanmedian(obs[a]); o_dmed = o_med - np.nanmedian(obs[b])
    o_mz = np.nanmean(_fz(obs[a])); o_dmz = o_mz - np.nanmean(_fz(obs[b]))
    n_med = np.nanmedian(null[:, a], axis=1); n_mz = np.nanmean(_fz(null[:, a]), axis=1)
    return {f'{A}_perm_p{suffix}': perm_p(n_med, o_med),
            f'{Bn}_perm_p{suffix}': perm_p(n_med - np.nanmedian(null[:, b], axis=1), o_dmed),
            f'{A}_perm_p_meanz{suffix}': perm_p(n_mz, o_mz),
            f'{Bn}_perm_p_meanz{suffix}': perm_p(n_mz - np.nanmean(_fz(null[:, b]), axis=1), o_dmz)}


def boot_median_ci(v, rng, B=N_BOOT):
    v = np.asarray(pd.Series(v).dropna())
    meds = np.median(rng.choice(v, size=(B, len(v)), replace=True), axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def re_meta(r, n, k=1):
    """DerSimonian-Laird random-effects meta-analysis of (partial) correlations on Fisher z; var = 1/(n-3-k).
    Returns tau^2, pooled estimate with normal and Hartung-Knapp-Sidik-Jonkman CIs, prediction interval, I^2."""
    r, n = np.asarray(r, float), np.asarray(n, float)
    ok = np.isfinite(r) & np.isfinite(n) & (n - 3 - k > 0)
    r, n = r[ok], n[ok]
    K = len(r)
    if K < 3:
        return {'k': K}
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)); v = 1.0 / (n - 3 - k)
    w = 1 / v
    zf = (w * z).sum() / w.sum()
    Q = (w * (z - zf) ** 2).sum()
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (K - 1)) / C)
    ws = 1 / (v + tau2)
    mu = (ws * z).sum() / ws.sum()
    se = np.sqrt(1 / ws.sum())
    q_hk = (ws * (z - mu) ** 2).sum() / (K - 1)
    se_hk = np.sqrt(max(q_hk, 1e-300) / ws.sum())
    tq = tdist.ppf(0.975, K - 1)
    pi_half = tdist.ppf(0.975, max(K - 2, 1)) * np.sqrt(tau2 + se ** 2)
    return {'k': K, 'tau2': float(tau2), 'tau': float(np.sqrt(tau2)), 'Q': float(Q), 'I2': float(max(0.0, (Q - (K - 1)) / Q)) if Q > 0 else 0.0,
            'mu_z': float(mu), 'se_z': float(se), 'pooled_r': float(np.tanh(mu)),
            'ci95_r': [float(np.tanh(mu - 1.96 * se)), float(np.tanh(mu + 1.96 * se))],
            'ci95_r_hksj': [float(np.tanh(mu - tq * se_hk)), float(np.tanh(mu + tq * se_hk))],
            'p_one_sided': float(norm.sf(mu / se)), 'p_one_sided_hksj': float(tdist.sf(mu / se_hk, K - 1)),
            'pi95_r': [float(np.tanh(mu - pi_half)), float(np.tanh(mu + pi_half))], 'fixed_r': float(np.tanh(zf)),
            'mean_n': float(n.mean())}


def re_difference(m1, m2):
    if 'mu_z' not in m1 or 'mu_z' not in m2:
        return {}
    d = m1['mu_z'] - m2['mu_z']; se = np.sqrt(m1['se_z'] ** 2 + m2['se_z'] ** 2)
    return {'diff_z': float(d), 'se': float(se), 'ci95_z': [float(d - 1.96 * se), float(d + 1.96 * se)],
            'p_one_sided': float(norm.sf(d / se))}


def eb_shrink(r, n, meta, k=1):
    z = np.arctanh(np.clip(np.asarray(r, float), -0.999999, 0.999999)); v = 1.0 / (np.asarray(n, float) - 3 - k)
    tau2, mu = meta['tau2'], meta['mu_z']
    B = tau2 / (tau2 + v)
    post = mu + B * (z - mu); psd = np.sqrt(B * v)
    return np.tanh(post), np.tanh(post - 1.96 * psd), np.tanh(post + 1.96 * psd)


def fisher_ci(r, n, k=1):
    z = np.arctanh(np.clip(np.asarray(r, float), -0.999999, 0.999999)); se = 1 / np.sqrt(np.asarray(n, float) - 3 - k)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def class_level(per_drug, cls_col, value_col='r', rep_col='repro'):
    """Aggregate per-drug transfer by mechanism class (mean within class), then test across classes."""
    out = {}
    rep = per_drug[per_drug[rep_col] == 'reproducible'].dropna(subset=[value_col])
    non = per_drug[per_drug[rep_col] == 'non_reproducible'].dropna(subset=[value_col])
    a = rep.groupby(cls_col)[value_col].mean()
    b = non.groupby(cls_col)[value_col].mean()
    out['classes_reproducible'] = int(len(a)); out['classes_non_reproducible'] = int(len(b))
    out['class_means_reproducible'] = {k: round(float(v), 4) for k, v in a.sort_values(ascending=False).items()}
    out['median_class_mean_reproducible'] = float(a.median())
    out['n_classes_negative'] = int((a < 0).sum())
    if len(a) >= 3:
        out['wilcoxon_p_classes'] = float(wilcoxon(a, alternative='greater').pvalue)
    if len(a) >= 3 and len(b) >= 3:
        out['median_class_mean_non_reproducible'] = float(b.median())
        out['mwu_p_classes'] = float(mannwhitneyu(a, b, alternative='greater').pvalue)
    return out


def names_and_classes():
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    cur = pd.read_csv(OUT / 'curated_classes.csv').set_index('compound')
    dpairs = pd.read_csv(RESULTS / 'decomposition' / 'cross_screen_pairs.csv')
    rep = dpairs.groupby('compound').rho_sel.mean()
    repro = pd.Series(np.where(rep >= REPRODUCIBLE, 'reproducible', np.where(rep < NONREPRODUCIBLE, 'non_reproducible',
                                                                           'intermediate')), index=rep.index)
    return comp.name, cur.old_class, cur.new_class, repro, rep


# ================================================================== PDX
def tumour_of(model, grouping):
    t = model.split('-')[0]
    if grouping == 'G2' and t == 'STG139M':
        t = 'STG139'
    return t


def pdx_structure(models, expr_models):
    rows = []
    for m in models:
        suffix = m.split('-', 1)[1] if '-' in m else ''
        rows.append({'model': m, 'tumour_G1': tumour_of(m, 'G1'), 'tumour_G2': tumour_of(m, 'G2'), 'derivative': suffix,
                     'has_expression': m in expr_models})
    return pd.DataFrame(rows)


def pdx_similarity(struct, Zm):
    """Within- versus between-tumour similarity of expression, frozen predictions and responses; per-drug ICC."""
    ge = pd.read_csv(OWN_RAW / 'pdx_bruna_inputs' / 'gene_expression.csv').set_index('cell_line_name')
    ge = ge.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    pred = pd.read_parquet(RESULTS / 'pdx_transfer' / 'pred_sel.parquet')
    tum = struct.set_index('model').tumour_G1
    out = {}

    def within_between(C):
        w, b = [], []
        idx = list(C.index)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                v = C.iloc[i, j]
                if not np.isfinite(v):
                    continue
                (w if tum[idx[i]] == tum[idx[j]] else b).append(v)
        return {'within_mean': float(np.mean(w)), 'between_mean': float(np.mean(b)), 'n_within_pairs': len(w),
                'n_between_pairs': len(b)}

    gc = ge.sub(ge.mean(axis=0), axis=1)
    out['expression_gene_centred'] = within_between(gc.T.corr())
    out['expression_raw'] = within_between(ge.T.corr())
    pc = pred.sub(pred.mean(axis=1), axis=0)
    out['frozen_predictions_drug_centred'] = within_between(pc.corr())
    out['response_z_profiles'] = within_between(Zm.corr(min_periods=10))
    iccs = []
    for d in Zm.index:
        s = Zm.loc[d].dropna()
        g = s.groupby(tum.reindex(s.index).values)
        sizes = g.size()
        if (sizes >= 2).sum() < 2:
            continue
        k = len(sizes); N = len(s)
        msb = (sizes * (g.mean() - s.mean()) ** 2).sum() / (k - 1)
        msw = ((s - g.transform('mean')) ** 2).sum() / (N - k)
        k0 = (N - (sizes ** 2).sum() / N) / (k - 1)
        iccs.append((msb - msw) / (msb + (k0 - 1) * msw))
    out['response_icc_within_tumour'] = {'median': float(np.median(iccs)), 'iqr': [float(np.percentile(iccs, 25)),
                                                                                    float(np.percentile(iccs, 75))],
                                         'drugs': len(iccs)}
    return out


def tumour_level_build(long, struct, grouping, drugs):
    tm = struct.set_index('model')[f'tumour_{grouping}']
    lt = long[long.compound.isin(drugs)].copy()
    lt['depmap_id'] = lt['sample'].map(tm)
    lt = lt.groupby(['compound', 'depmap_id'], as_index=False).sens.mean()
    M, Z, G, R = build(lt, MIN_TUMOURS)
    return Z, G


def tumour_mean_predictions(P, struct, grouping, columns):
    tm = struct.set_index('model')[f'tumour_{grouping}']
    Pt = P.T.groupby(tm.reindex(P.columns).values).mean().T
    return Pt.reindex(columns=columns)


def pdx_analysis(summary):
    rng_key = 100
    name, old_cls, new_cls, repro, repmean = names_and_classes()
    long = pd.read_csv(PROCESSED / 'pdx_bruna' / 'pdx_long.csv.gz')
    Pall = pd.read_parquet(RESULTS / 'pdx_transfer' / 'pred_sel.parquet')
    Mm, Zm, Gm, Rm = build(long.rename(columns={'sample': 'depmap_id'}), PDXC.MIN_MODELS_PER_DRUG)
    models = list(Zm.columns)
    struct = pdx_structure(models, set(Pall.columns))
    struct['n_drugs_measured'] = struct.model.map(long.groupby('sample').compound.nunique())
    struct.to_csv(OUT / 'pdx_tumour_map.csv', index=False)
    S = {'models': len(models), 'models_with_expression': int(struct.has_expression.sum())}
    for gname in ['G1', 'G2']:
        S[f'tumours_{gname}'] = int(struct[f'tumour_{gname}'].nunique())
        S[f'tumours_with_expression_{gname}'] = int(struct.loc[struct.has_expression, f'tumour_{gname}'].nunique())
        S[f'derivatives_per_tumour_{gname}'] = struct.groupby(f'tumour_{gname}').size().value_counts().sort_index().to_dict()
        S[f'expressed_derivatives_per_tumour_{gname}'] = struct[struct.has_expression].groupby(f'tumour_{gname}').size().value_counts().sort_index().to_dict()
    S['multi_derivative_tumours'] = struct.groupby('tumour_G1').model.apply(list)[struct.groupby('tumour_G1').size() > 1].to_dict()
    S['ambiguous'] = ('STG139M (two derivatives, X4C and X5C) versus STG139 (X9C, X13C): treated as separate tumours in G1 '
                      'and merged in G2; HCI002-X2C/X2CR, HCI010-X2C/X2CR and STG282-X1C/X1CR/X1CR1 share the same xenograft '
                      'passage and differ only in the culture-derivation suffix, so they are the closest pseudo-replicates')
    S['similarity'] = pdx_similarity(struct, Zm)
    summary['pdx_structure'] = S

    # ---------------- registered model level
    drugs = [d for d in Zm.index if d in Pall.index and repro.get(d) in ('reproducible', 'non_reproducible')]
    P = Pall.reindex(columns=models)
    has_pred = np.isfinite(P.values).any(axis=0)
    des_m = designs_for(Zm, Gm, has_pred, drugs, PDXC.MIN_PAIRS)
    ident = np.arange(len(models))[None, :]
    vals = {d: P.loc[d].values.astype(float) for d in drugs}
    r_reg = perm_matrix(des_m, vals, ident, drugs)[0]
    reg = pd.read_csv(RESULTS / 'confirmation_PDX_Bruna' / 'scores.csv')
    reg = reg[reg.family == 'sel'].set_index('compound')
    chk = float(np.nanmax(np.abs(r_reg - reg.sel.reindex(drugs).values)))
    assert chk < 1e-10, chk
    per = pd.DataFrame({'compound': drugs, 'name': [name.get(d) for d in drugs], 'repro': [repro[d] for d in drugs],
                        'old_class': [old_cls.get(d) for d in drugs], 'curated_class': [new_cls.get(d) for d in drugs],
                        'discovery_rho_sel': [repmean[d] for d in drugs], 'r_model': r_reg,
                        'n_model': [des_m[d].n for d in drugs]})
    tm1 = struct.set_index('model').tumour_G1
    per['n_tumours'] = [len(set(tm1[np.array(models)[des_m[d].idx]])) for d in drugs]
    isrep = (per.repro == 'reproducible').values; isnon = (per.repro == 'non_reproducible').values
    tests = []

    def add(analysis, grouping, r, n_units, extra=None):
        a, b = r[isrep], r[isnon]
        t = one_sided_tests(a, b)
        row = {'analysis': analysis, 'grouping': grouping, 'drugs_reproducible': t['n_a'], 'drugs_non_reproducible': t.get('n_b'),
               'H8_median': t['median_a'], 'H8_wilcoxon_p': t.get('wilcoxon_p'), 'H8_n_negative': t['n_negative_a'],
               'H9_median_non_reproducible': t.get('median_b'), 'H9_difference_of_medians': t['median_a'] - t.get('median_b', np.nan),
               'H9_mwu_p': t.get('mwu_p'), 'median_units_per_drug': float(np.median(n_units[isrep | isnon]))}
        row.update(extra or {})
        tests.append(row)
        return row

    rng = rng_for(rng_key, 1)
    expr_pos = np.where(has_pred)[0]
    perm = unit_permutations(len(models), expr_pos, N_PERM, rng)
    null = perm_matrix(des_m, vals, perm, drugs)
    cc0 = pd.DataFrame(null[:, isrep]).corr().values
    model_null_corr = float(np.nanmean(cc0[~np.eye(len(cc0), dtype=bool)]))
    add('registered model level (37 models; 25 with expression)', 'model', r_reg, per.n_model.values,
        {**perm_pvals(r_reg, null, isrep, isnon, 'H8', 'H9'),
         'perm_scheme': 'model labels permuted among the 25 expressed models, jointly across drugs',
         'H8_drug_boot_ci95': boot_median_ci(r_reg[isrep], rng_for(rng_key, 2))})

    # tumour-cluster bootstrap of the registered estimator (models weighted by their tumour's resampling count)
    for gname in ['G1', 'G2']:
        tm = struct.set_index('model')[f'tumour_{gname}']
        tumours_e = sorted(set(tm[np.array(models)[expr_pos]]))
        tpos = {t: i for i, t in enumerate(tumours_e)}
        unit_cluster = np.array([tpos.get(tm[m], -1) if has_pred[i] else -1 for i, m in enumerate(models)])
        rngb = rng_for(rng_key, 3, int(gname[1]))
        C = rngb.multinomial(len(tumours_e), np.ones(len(tumours_e)) / len(tumours_e), size=N_BOOT_TWOWAY).astype(float)
        meds, meds2 = [], []
        rb = np.full((N_BOOT_TWOWAY, len(drugs)), np.nan)
        for j, d in enumerate(drugs):
            des = des_m[d]
            if not des.ok:
                continue
            uc = unit_cluster[des.idx]
            W = C[:, uc]
            rb[:, j] = weighted_partial(W, vals[d][des.idx], Zm.loc[d].values[des.idx], Gm.loc[d].values[des.idx])
        rr = rb[:, isrep]
        meds = np.nanmedian(rr, axis=1)
        pick = rngb.integers(0, rr.shape[1], size=(N_BOOT_TWOWAY, rr.shape[1]))
        meds2 = np.nanmedian(np.take_along_axis(rr, pick, axis=1), axis=1)
        tests[0][f'H8_tumour_cluster_boot_ci95_{gname}'] = [float(np.nanpercentile(meds, 2.5)), float(np.nanpercentile(meds, 97.5))]
        tests[0][f'H8_drug_x_tumour_boot_ci95_{gname}'] = [float(np.nanpercentile(meds2, 2.5)), float(np.nanpercentile(meds2, 97.5))]

        # conservative tumour-block permutation at model level: tumour-mean predictions broadcast to derivatives
        Pt = tumour_mean_predictions(Pall, struct, gname, tumours_e)
        vals_b = {d: np.r_[Pt.loc[d].values.astype(float), np.nan] for d in drugs}
        idx_id = block_broadcast_index(unit_cluster, tumours_e, 1, rngb, identity=True)
        r_b = perm_matrix(des_m, vals_b, idx_id, drugs)[0]
        idx_b = block_broadcast_index(unit_cluster, tumours_e, N_PERM, rng_for(rng_key, 4, int(gname[1])))
        nb = perm_matrix(des_m, vals_b, idx_b, drugs)
        add('tumour-block permutation, tumour-mean predictions broadcast to derivatives (model level)', gname, r_b,
            per.n_model.values,
            {**perm_pvals(r_b, nb, isrep, isnon, 'H8', 'H9'),
             'perm_scheme': f'{len(tumours_e)} expressed tumours exchangeable; derivatives move with their tumour'})
        # variant: sample-specific predictions, tumours permuted within block-size strata
        idx_s = stratified_block_permutations(unit_cluster, N_PERM, rng_for(rng_key, 5, int(gname[1])))
        ns = perm_matrix(des_m, vals, idx_s, drugs)
        add('tumour-block permutation within block-size strata, sample-specific predictions (model level)', gname, r_reg,
            per.n_model.values,
            {**perm_pvals(r_reg, ns, isrep, isnon, 'H8', 'H9'),
             'perm_scheme': 'tumours permuted only among tumours with the same number of expressed derivatives '
                            '(unique-size tumours shuffle internally)'})

    # ---------------- T1: tumour-level averaging (primary tumour-level analysis)
    meta_rows = []
    m_rep = re_meta(per.r_model[isrep], per.n_model[isrep]); m_non = re_meta(per.r_model[isnon], per.n_model[isnon])
    meta_rows.append({'analysis': 'registered model level', 'grouping': 'model', 'class': 'reproducible', **m_rep})
    meta_rows.append({'analysis': 'registered model level', 'grouping': 'model', 'class': 'non_reproducible', **m_non})
    meta_rows.append({'analysis': 'registered model level', 'grouping': 'model', 'class': 'difference', **re_difference(m_rep, m_non)})
    nulls_saved = {'model_mean_offdiag_null_corr': model_null_corr}
    for gname in ['G1', 'G2']:
        Zt, Gt = tumour_level_build(long, struct, gname, list(Zm.index))
        assert set(Zt.index) == set(Zm.index), 'tumour-level build dropped drugs'
        tumours = list(Zt.columns)
        tumours_e = sorted(set(struct.loc[struct.has_expression, f'tumour_{gname}']))
        Pt = tumour_mean_predictions(Pall, struct, gname, tumours)
        has_pt = np.isfinite(Pt.values).any(axis=0)
        des_t = designs_for(Zt, Gt, has_pt, drugs, MIN_TUMOURS)
        vals_t = {d: Pt.loc[d].values.astype(float) for d in drugs}
        r_t = perm_matrix(des_t, vals_t, np.arange(len(tumours))[None, :], drugs)[0]
        # cross-check the engine against the frozen partial() logic on the tumour-level data
        for j, d in enumerate(drugs):
            rr, nn = ref_partial(vals_t[d], Zt.loc[d].values, Gt.loc[d].values, MIN_TUMOURS)
            assert (np.isnan(rr) and np.isnan(r_t[j])) or abs(rr - r_t[j]) < 1e-10, (d, rr, r_t[j])
        per[f'r_T1_{gname}'] = r_t
        per[f'n_T1_{gname}'] = [des_t[d].n for d in drugs]
        rngt = rng_for(rng_key, 6, int(gname[1]))
        permt = unit_permutations(len(tumours), np.where(has_pt)[0], N_PERM, rngt)
        nt = perm_matrix(des_t, vals_t, permt, drugs)
        nulls_saved[gname] = nt
        # two-way (drug x tumour) bootstrap of the tumour-level H8 median
        Cw = rngt.multinomial(len(tumours), np.ones(len(tumours)) / len(tumours), size=N_BOOT_TWOWAY).astype(float)
        rb = np.full((N_BOOT_TWOWAY, len(drugs)), np.nan)
        for j, d in enumerate(drugs):
            des = des_t[d]
            if des.ok:
                rb[:, j] = weighted_partial(Cw[:, des.idx], vals_t[d][des.idx], Zt.loc[d].values[des.idx], Gt.loc[d].values[des.idx])
        rr = rb[:, isrep]
        pick = rngt.integers(0, rr.shape[1], size=(N_BOOT_TWOWAY, rr.shape[1]))
        m2 = np.nanmedian(np.take_along_axis(rr, pick, axis=1), axis=1)
        mt = re_meta(r_t[isrep], per[f'n_T1_{gname}'][isrep]); mn = re_meta(r_t[isnon], per[f'n_T1_{gname}'][isnon])
        # tumour-bootstrap CI of the RE pooled estimate (per-drug variances held at their observed values)
        mus = []
        for b in range(N_BOOT_TWOWAY):
            x = rb[b, isrep]
            mm = re_meta(x, per[f'n_T1_{gname}'][isrep])
            if 'mu_z' in mm:
                mus.append(mm['pooled_r'])
        mt['tumour_boot_ci95_pooled_r'] = [float(np.percentile(mus, 2.5)), float(np.percentile(mus, 97.5))]
        row = add(f'T1 tumour level: sensitivities and predictions averaged within tumour ({len(tumours)} tumours; '
                  f'{len(tumours_e)} with expression)', gname, r_t, per[f'n_T1_{gname}'].values,
                  {**perm_pvals(r_t, nt, isrep, isnon, 'H8', 'H9'),
                   'perm_scheme': f'{int(has_pt.sum())} expressed tumours exchangeable, {N_PERM} permutations',
                   'H8_drug_boot_ci95': boot_median_ci(r_t[isrep], rng_for(rng_key, 7, int(gname[1]))),
                   'H8_drug_x_tumour_boot_ci95': [float(np.nanpercentile(m2, 2.5)), float(np.nanpercentile(m2, 97.5))],
                   'H8_RE_pooled_r': mt.get('pooled_r'), 'H8_RE_ci95': mt.get('ci95_r'), 'H8_RE_tau2': mt.get('tau2')})
        for c, m in [('reproducible', mt), ('non_reproducible', mn)]:
            meta_rows.append({'analysis': 'T1 tumour level', 'grouping': gname, 'class': c, **m})
        meta_rows.append({'analysis': 'T1 tumour level', 'grouping': gname, 'class': 'difference', **re_difference(mt, mn)})
        # inter-drug correlation of per-drug r under the permutation null (design effect for the power script)
        cc = pd.DataFrame(nt[:, isrep]).corr().values
        nulls_saved[f'{gname}_mean_offdiag_null_corr'] = float(np.nanmean(cc[~np.eye(len(cc), dtype=bool)]))
        nulls_saved[f'{gname}_null_sd_median'] = float(np.nanstd(np.nanmedian(nt[:, isrep], axis=1)))

        # ---------------- T2: registered model-level z, g, p averaged within tumour over matched derivatives
        tm = struct.set_index('model')[f'tumour_{gname}']
        r2, n2 = [], []
        for d in drugs:
            des = des_m[d]
            sub = pd.DataFrame({'t': tm[np.array(models)[des.idx]].values, 'z': Zm.loc[d].values[des.idx],
                                'g': Gm.loc[d].values[des.idx], 'p': vals[d][des.idx]}).groupby('t').mean()
            rr, nn = ref_partial(sub.p.values, sub.z.values, sub.g.values, MIN_TUMOURS)
            r2.append(rr); n2.append(len(sub))
        per[f'r_T2_{gname}'] = r2; per[f'n_T2_{gname}'] = n2
        add('T2 tumour level: registered z, axis and prediction averaged within tumour over matched derivatives', gname,
            np.array(r2), np.array(n2), {'H8_drug_boot_ci95': boot_median_ci(np.array(r2)[isrep], rng_for(rng_key, 8, int(gname[1])))})
        m2r = re_meta(np.array(r2)[isrep], np.array(n2)[isrep]); m2n = re_meta(np.array(r2)[isnon], np.array(n2)[isnon])
        meta_rows.append({'analysis': 'T2 tumour level', 'grouping': gname, 'class': 'reproducible', **m2r})
        meta_rows.append({'analysis': 'T2 tumour level', 'grouping': gname, 'class': 'non_reproducible', **m2n})
        meta_rows.append({'analysis': 'T2 tumour level', 'grouping': gname, 'class': 'difference', **re_difference(m2r, m2n)})

    # ---------------- T3: one derivative per tumour, random draws (G1)
    rng3 = rng_for(rng_key, 9)
    draws = np.full((N_DRAWS, len(drugs)), np.nan)
    for j, d in enumerate(drugs):
        des = des_m[d]
        if not des.ok:
            continue
        tt = tm1[np.array(models)[des.idx]].values
        groups = [np.where(tt == t)[0] for t in pd.unique(tt)]
        pick = np.stack([g[rng3.integers(0, len(g), N_DRAWS)] for g in groups], axis=1)  # (draws x tumours)
        p = vals[d][des.idx][pick]; z = Zm.loc[d].values[des.idx][pick]; g = Gm.loc[d].values[des.idx][pick]
        draws[:, j] = partial_rows(p, z, g)
    st = []
    for b in range(N_DRAWS):
        t = one_sided_tests(draws[b, isrep], draws[b, isnon])
        st.append({'draw': b, 'H8_median': t['median_a'], 'H8_wilcoxon_p': t.get('wilcoxon_p'),
                   'H9_median_non_reproducible': t.get('median_b'), 'H9_mwu_p': t.get('mwu_p')})
    st = pd.DataFrame(st)
    st.to_csv(OUT / 'pdx_one_per_tumour_draws.csv', index=False)
    per['r_T3_median_over_draws'] = np.nanmedian(draws, axis=0)
    tests.append({'analysis': f'T3 one expressed derivative per tumour, {N_DRAWS} random draws (summary over draws)',
                  'grouping': 'G1', 'drugs_reproducible': int(isrep.sum()), 'drugs_non_reproducible': int(isnon.sum()),
                  'H8_median': float(st.H8_median.median()), 'H8_wilcoxon_p': float(st.H8_wilcoxon_p.median()),
                  'H9_median_non_reproducible': float(st.H9_median_non_reproducible.median()),
                  'H9_difference_of_medians': float((st.H8_median - st.H9_median_non_reproducible).median()),
                  'H9_mwu_p': float(st.H9_mwu_p.median()),
                  'H8_frac_draws_p_lt_0.05': float((st.H8_wilcoxon_p < 0.05).mean()),
                  'H9_frac_draws_p_lt_0.05': float((st.H9_mwu_p < 0.05).mean()),
                  'H8_median_range_5_95': [float(st.H8_median.quantile(0.05)), float(st.H8_median.quantile(0.95))]})

    # ---------------- ln IC50 secondary endpoint (registered and T1 G1)
    ll = pd.read_csv(PROCESSED / 'pdx_bruna' / 'pdx_lnic50_long.csv.gz')
    Ml, Zl, Gl_, Rl = build(ll.rename(columns={'sample': 'depmap_id'}), PDXC.MIN_MODELS_PER_DRUG)
    dl = [d for d in Zl.index if d in Pall.index and repro.get(d) in ('reproducible', 'non_reproducible')]
    has_l = np.isfinite(Pall.reindex(columns=Zl.columns).values).any(axis=0)
    des_l = designs_for(Zl, Gl_, has_l, dl, PDXC.MIN_PAIRS)
    vl = {d: Pall.reindex(columns=Zl.columns).loc[d].values.astype(float) for d in dl}
    r_l = perm_matrix(des_l, vl, np.arange(Zl.shape[1])[None, :], dl)[0]
    rep_l = np.array([repro[d] == 'reproducible' for d in dl]); non_l = ~rep_l
    tl = one_sided_tests(r_l[rep_l], r_l[non_l])
    Ztl, Gtl = tumour_level_build(ll, struct, 'G1', list(Zl.index))
    Ptl = tumour_mean_predictions(Pall, struct, 'G1', list(Ztl.columns))
    des_tl = designs_for(Ztl, Gtl, np.isfinite(Ptl.values).any(axis=0), dl, MIN_TUMOURS)
    vtl = {d: Ptl.loc[d].values.astype(float) for d in dl}
    r_tl = perm_matrix(des_tl, vtl, np.arange(Ztl.shape[1])[None, :], dl)[0]
    ntl = perm_matrix(des_tl, vtl, unit_permutations(Ztl.shape[1], np.where(np.isfinite(Ptl.values).any(axis=0))[0], N_PERM,
                                                     rng_for(rng_key, 10)), dl)
    ttl = one_sided_tests(r_tl[rep_l], r_tl[non_l])
    for lab, t, extra in [('ln IC50 secondary endpoint, registered model level', tl, {}),
                          ('ln IC50 secondary endpoint, T1 tumour level', ttl,
                           perm_pvals(r_tl, ntl, rep_l, non_l, 'H8', 'H9'))]:
        tests.append({'analysis': lab, 'grouping': 'model' if 'registered' in lab else 'G1', 'drugs_reproducible': t['n_a'],
                      'drugs_non_reproducible': t.get('n_b'), 'H8_median': t['median_a'], 'H8_wilcoxon_p': t.get('wilcoxon_p'),
                      'H8_n_negative': t['n_negative_a'], 'H9_median_non_reproducible': t.get('median_b'),
                      'H9_difference_of_medians': t['median_a'] - t.get('median_b', np.nan), 'H9_mwu_p': t.get('mwu_p'), **extra})

    # ---------------- per-drug table with CIs and shrinkage
    lo, hi = fisher_ci(per.r_model, per.n_model)
    per['r_model_ci95_lo'], per['r_model_ci95_hi'] = lo, hi
    lo, hi = fisher_ci(per.r_T1_G1, per.n_T1_G1)
    per['r_T1_G1_ci95_lo'], per['r_T1_G1_ci95_hi'] = lo, hi
    for c, mask in [('reproducible', isrep), ('non_reproducible', isnon)]:
        mm = re_meta(per.r_T1_G1[mask], per.n_T1_G1[mask])
        s, l, h = eb_shrink(per.r_T1_G1[mask], per.n_T1_G1[mask], mm)
        per.loc[mask, 'r_T1_G1_shrunk'] = s; per.loc[mask, 'r_T1_G1_shrunk_lo'] = l; per.loc[mask, 'r_T1_G1_shrunk_hi'] = h
        mm = re_meta(per.r_model[mask], per.n_model[mask])
        s, l, h = eb_shrink(per.r_model[mask], per.n_model[mask], mm)
        per.loc[mask, 'r_model_shrunk'] = s; per.loc[mask, 'r_model_shrunk_lo'] = l; per.loc[mask, 'r_model_shrunk_hi'] = h
    per = per.sort_values(['repro', 'r_model'], ascending=[False, False])
    per.to_csv(OUT / 'pdx_tumour_level_per_drug.csv', index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / 'pdx_tumour_level_tests.csv', index=False)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(OUT / 'pdx_meta_analysis.csv', index=False)

    # ---------------- class-level tests
    cl = {}
    for col in ['old_class', 'curated_class']:
        for val in ['r_model', 'r_T1_G1']:
            cl[f'{col}|{val}'] = class_level(per, col, val)
    pd.DataFrame([{'key': k, **{kk: (json.dumps(vv) if isinstance(vv, dict) else vv) for kk, vv in v.items()}} for k, v in cl.items()]
                 ).to_csv(OUT / 'pdx_class_level.csv', index=False)
    named = ['GX15-070', 'MK 2206', 'MK-0683', 'AZD7762', 'AFATINIB']
    summary['pdx'] = {'tests': tests.to_dict(orient='records'), 'meta': meta.to_dict(orient='records'), 'class_level': cl,
                      'perm_null_design': nulls_saved_summary(nulls_saved),
                      'named_top_drugs': per[per.name.isin(named)][['name', 'n_model', 'r_model', 'r_model_ci95_lo', 'r_model_ci95_hi',
                                                                      'r_model_shrunk', 'n_T1_G1', 'r_T1_G1', 'r_T1_G1_ci95_lo',
                                                                      'r_T1_G1_ci95_hi', 'r_T1_G1_shrunk']].round(3).to_dict(orient='records'),
                      'per_drug_units': {'median_models': float(per.loc[per.repro == 'reproducible', 'n_model'].median()),
                                         'median_tumours': float(per.loc[per.repro == 'reproducible', 'n_tumours'].median()),
                                         'tumours_per_drug_counts': per.n_tumours.value_counts().sort_index().to_dict()}}
    return per


def nulls_saved_summary(ns):
    return {k: v for k, v in ns.items() if not isinstance(v, np.ndarray)}


def partial_rows(p, z, g):
    """Partial correlation r(p, z | g) for each row of (draws x n) arrays (NaN if n < MIN_TUMOURS)."""
    def c(a, b):
        return ((a - a.mean(1, keepdims=True)) * (b - b.mean(1, keepdims=True))).mean(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rpz = c(p, z) / np.sqrt(c(p, p) * c(z, z)); rpg = c(p, g) / np.sqrt(c(p, p) * c(g, g))
        rzg = c(z, g) / np.sqrt(c(z, z) * c(g, g))
        r = (rpz - rpg * rzg) / np.sqrt((1 - rpg ** 2) * (1 - rzg ** 2))
    if p.shape[1] < MIN_TUMOURS:
        r[:] = np.nan
    return r


# ================================================================== BeatAML
def parse_pct(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace('%', '')
    if s.startswith('>') or s.startswith('<'):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return np.nan


def beataml_inputs():
    long = pd.read_csv(PROCESSED / 'beataml' / 'beataml_long.csv.gz')
    smp = pd.Index(long['sample'].unique())
    mp = pd.read_excel(OWN_RAW / 'beataml_inputs' / 'beataml_waves1to4_sample_mapping.xlsx', sheet_name='sample_map')
    mp = mp.dropna(subset=['dbgap_rnaseq_sample']).drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
    clin = pd.read_excel(OWN_RAW / 'beataml_inputs' / 'beataml_wv1to4_clinical.xlsx', sheet_name='summary')
    clin = clin.dropna(subset=['dbgap_rnaseq_sample']).drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
    patient = mp.dbgap_subject_id.reindex(smp)
    assert patient.notna().all()
    assert (clin.dbgap_subject_id.reindex(smp).values == patient.values).all()
    return long, patient, clin.reindex(smp)


def beataml_scores(long, min_cells=30, patient=None):
    """Registered processing (frozen build, MIN_CELLS 30) and frozen pred_sel; returns Z, G, P (aligned)."""
    M, Z, G, R = build(long.rename(columns={'sample': 'depmap_id'}), min_cells)
    P = pd.read_parquet(RESULTS / 'beataml_transfer' / 'pred_sel.parquet').reindex(columns=Z.columns)
    return Z, G, P


def beataml_analysis(summary):
    key = 200
    name, old_cls, new_cls, repro, repmean = names_and_classes()
    long, patient, clin = beataml_inputs()
    Z, G, P = beataml_scores(long)
    samples = list(Z.columns)
    pat = patient.reindex(samples).values
    drugs = [d for d in Z.index if d in P.index and repro.get(d) in ('reproducible', 'non_reproducible')]
    has_pred = np.isfinite(P.values).any(axis=0)
    des = designs_for(Z, G, has_pred, drugs, 20)
    vals = {d: P.loc[d].values.astype(float) for d in drugs}
    r_reg = perm_matrix(des, vals, np.arange(len(samples))[None, :], drugs)[0]
    reg = pd.read_csv(RESULTS / 'confirmation_BeatAML' / 'scores.csv')
    reg = reg[reg.family == 'sel'].set_index('compound')
    chk = float(np.nanmax(np.abs(r_reg - reg.sel.reindex(drugs).values)))
    assert chk < 1e-10, chk
    per = pd.DataFrame({'compound': drugs, 'name': [name.get(d) for d in drugs], 'repro': [repro[d] for d in drugs],
                        'old_class': [old_cls.get(d) for d in drugs], 'curated_class': [new_cls.get(d) for d in drugs],
                        'discovery_rho_sel': [repmean[d] for d in drugs], 'r': r_reg, 'n': [des[d].n for d in drugs]})
    per['n_patients'] = [len(set(pat[des[d].idx])) for d in drugs]
    isrep = (per.repro == 'reproducible').values; isnon = (per.repro == 'non_reproducible').values
    S = {'samples': len(samples), 'patients': int(pd.Series(pat).nunique()),
         'samples_per_patient': pd.Series(pat).value_counts().value_counts().sort_index().to_dict(),
         'patients_with_multiple_samples': int((pd.Series(pat).value_counts() > 1).sum()),
         'samples_from_multi_sample_patients': int(pd.Series(pat).value_counts()[lambda s: s > 1].sum()),
         'registered_check_max_abs_diff': chk}
    rows = []

    def add(analysis, r, extra=None, units=None):
        t = one_sided_tests(r[isrep], r[isnon])
        row = {'analysis': analysis, 'drugs_reproducible': t['n_a'], 'drugs_non_reproducible': t.get('n_b'),
               'H6_median': t['median_a'], 'H6_wilcoxon_p': t.get('wilcoxon_p'), 'H6_n_negative': t['n_negative_a'],
               'H7_median_non_reproducible': t.get('median_b'), 'H7_difference_of_medians': t['median_a'] - t.get('median_b', np.nan),
               'H7_mwu_p': t.get('mwu_p')}
        if units is not None:
            row['median_units_per_drug'] = float(np.median(units))
        row.update(extra or {})
        rows.append(row)
        return row

    # (a) naive sample-label permutation
    na = perm_matrix(des, vals, unit_permutations(len(samples), np.where(has_pred)[0], N_PERM, rng_for(key, 1)), drugs)
    # (b) patient blocks within block-size strata, sample-specific predictions (registered statistic)
    pidx = pd.factorize(pat)[0]
    nb = perm_matrix(des, vals, stratified_block_permutations(pidx, N_PERM, rng_for(key, 2)), drugs)
    # (c) patient-mean predictions broadcast, all patients exchangeable
    pats = pd.unique(pat)
    Pp = P.T.groupby(pat).mean().T.reindex(columns=pats)
    vals_c = {d: np.r_[Pp.loc[d].values.astype(float), np.nan] for d in drugs}
    uc = np.array([list(pats).index(x) for x in pat])
    r_c = perm_matrix(des, vals_c, block_broadcast_index(uc, pats, 1, None, identity=True), drugs)[0]
    nc = perm_matrix(des, vals_c, block_broadcast_index(uc, pats, N_PERM, rng_for(key, 3)), drugs)
    # patient-cluster bootstrap of the registered estimator, with and without drug resampling
    rngb = rng_for(key, 4)
    Cp = rngb.multinomial(len(pats), np.ones(len(pats)) / len(pats), size=N_BOOT_TWOWAY).astype(float)
    rb = np.full((N_BOOT_TWOWAY, len(drugs)), np.nan)
    for j, d in enumerate(drugs):
        ds = des[d]
        if ds.ok:
            rb[:, j] = weighted_partial(Cp[:, uc[ds.idx]], vals[d][ds.idx], Z.loc[d].values[ds.idx], G.loc[d].values[ds.idx])
    rr = rb[:, isrep]
    pick = rngb.integers(0, rr.shape[1], size=(N_BOOT_TWOWAY, rr.shape[1]))
    m2 = np.nanmedian(np.take_along_axis(rr, pick, axis=1), axis=1)
    add('registered sample level (520 samples, 479 patients)', r_reg,
        {**perm_pvals(r_reg, na, isrep, isnon, 'H6', 'H7', '_sample_labels'),
         **perm_pvals(r_reg, nb, isrep, isnon, 'H6', 'H7', '_patient_blocks'),
         'H6_drug_boot_ci95': boot_median_ci(r_reg[isrep], rng_for(key, 5)),
         'H6_patient_cluster_boot_ci95': [float(np.nanpercentile(np.nanmedian(rr, axis=1), 2.5)), float(np.nanpercentile(np.nanmedian(rr, axis=1), 97.5))],
         'H6_drug_x_patient_boot_ci95': [float(np.nanpercentile(m2, 2.5)), float(np.nanpercentile(m2, 97.5))]}, per.n.values)
    add('patient-mean predictions broadcast to samples, patients permuted (sample level)', r_c,
        perm_pvals(r_c, nc, isrep, isnon, 'H6', 'H7', '_patient_blocks'), per.n.values)

    # (d) patient-level reanalysis: sensitivities and predictions averaged within patient
    lp = long.copy(); lp['sample'] = lp['sample'].map(patient)
    lp = lp.groupby(['compound', 'sample'], as_index=False).sens.mean()
    Zp, Gp, _ = beataml_scores(lp)
    Ppat = P.T.groupby(pat).mean().T.reindex(columns=Zp.columns)
    dp = [d for d in drugs if d in Zp.index]
    has_pp = np.isfinite(Ppat.values).any(axis=0)
    desp = designs_for(Zp, Gp, has_pp, dp, 20)
    vp = {d: Ppat.loc[d].values.astype(float) for d in dp}
    r_p_list = perm_matrix(desp, vp, np.arange(Zp.shape[1])[None, :], dp)[0]
    r_p = pd.Series(r_p_list, index=dp).reindex(drugs).values
    per['r_patient_level'] = r_p; per['n_patient_level'] = pd.Series([desp[d].n for d in dp], index=dp).reindex(drugs).values
    npl = perm_matrix(desp, vp, unit_permutations(Zp.shape[1], np.where(has_pp)[0], N_PERM, rng_for(key, 6)), dp)
    rep_p = np.array([repro[d] == 'reproducible' for d in dp]); non_p = ~rep_p
    add(f'patient level: sensitivities and predictions averaged within patient ({Zp.shape[1]} patients)', r_p,
        {**perm_pvals(np.array(r_p_list), npl, rep_p, non_p, 'H6', 'H7', '_patient_labels'),
         'H6_drug_boot_ci95': boot_median_ci(r_p[isrep], rng_for(key, 7))}, per.n_patient_level.values)

    # random-effects estimates
    meta_rows = []
    for lab, rv, nv in [('registered sample level', per.r, per.n), ('patient level', per.r_patient_level, per.n_patient_level)]:
        a = re_meta(rv[isrep], nv[isrep]); b = re_meta(rv[isnon], nv[isnon])
        meta_rows += [{'analysis': lab, 'class': 'reproducible', **a}, {'analysis': lab, 'class': 'non_reproducible', **b},
                      {'analysis': lab, 'class': 'difference', **re_difference(a, b)}]

    # composition and sensitivity analyses
    stage = clin.diseaseStageAtSpecimenCollection.fillna('missing')
    dx = clin.dxAtSpecimenAcquisition.fillna('missing')
    non_aml = ~dx.str.startswith('ACUTE MYELOID')
    comp = {'disease_stage': stage.value_counts().to_dict(), 'diagnosis_at_acquisition': dx.value_counts().to_dict(),
            'specimen_type': clin.specimenType.fillna('missing').value_counts().to_dict(),
            'remission': int((stage == 'Remission').sum()), 'relapse': int((stage == 'Relapse').sum()),
            'residual': int((stage == 'Residual').sum()), 'non_aml_diagnosis': int(non_aml.sum()),
            'excluded_remission_relapse_nonaml': int(((stage == 'Remission') | (stage == 'Relapse') | non_aml).sum())}
    S['composition'] = comp
    comp_rows = [{'variable': 'diseaseStageAtSpecimenCollection', 'level': k, 'samples': v} for k, v in comp['disease_stage'].items()]
    comp_rows += [{'variable': 'dxAtSpecimenAcquisition', 'level': k, 'samples': v} for k, v in comp['diagnosis_at_acquisition'].items()]
    comp_rows += [{'variable': 'specimenType', 'level': k, 'samples': v} for k, v in comp['specimen_type'].items()]
    comp_rows += [{'variable': 'samples_per_patient', 'level': str(k), 'samples': int(k) * int(v)} for k, v in S['samples_per_patient'].items()]
    pd.DataFrame(comp_rows).to_csv(OUT / 'beataml_composition.csv', index=False)
    subsets = {
        'exclude remission, relapse and non-AML diagnoses': ~((stage == 'Remission') | (stage == 'Relapse') | non_aml),
        'initial-diagnosis AML only': (stage == 'Initial Diagnosis') & ~non_aml,
        'one sample per patient (initial diagnosis, then bone marrow, then sample ID)': None,
    }
    order = pd.DataFrame({'sample': samples, 'patient': pat, 'stage0': (stage.reindex(samples) != 'Initial Diagnosis').values,
                          'bm0': (clin.specimenType.reindex(samples) != 'Bone Marrow Aspirate').values}).sort_values(['patient', 'stage0', 'bm0', 'sample'])
    first = set(order.drop_duplicates('patient')['sample'])
    subsets['one sample per patient (initial diagnosis, then bone marrow, then sample ID)'] = pd.Series([s in first for s in samples], index=samples)
    sens_rows = []
    for lab, keep in subsets.items():
        keep_s = set(keep[keep].index)
        lsub = long[long['sample'].isin(keep_s)]
        Zs, Gs, Ps = beataml_scores(lsub)
        ds = [d for d in drugs if d in Zs.index]
        dss = designs_for(Zs, Gs, np.isfinite(Ps.values).any(axis=0), ds, 20)
        rs = pd.Series(perm_matrix(dss, {d: Ps.loc[d].values.astype(float) for d in ds}, np.arange(Zs.shape[1])[None, :], ds)[0], index=ds)
        ns_ = pd.Series([dss[d].n for d in ds], index=ds)
        rr_ = rs.reindex(drugs).values
        t = one_sided_tests(rr_[isrep], rr_[isnon])
        a = re_meta(rs.reindex(drugs).values[isrep], ns_.reindex(drugs).values[isrep])
        sens_rows.append({'subset': lab, 'samples': Zs.shape[1], 'patients': int(pd.Series(patient.reindex(Zs.columns)).nunique()),
                          'inhibitors_retained': Zs.shape[0], 'drugs_reproducible': t['n_a'], 'drugs_non_reproducible': t.get('n_b'),
                          'H6_median': t['median_a'], 'H6_wilcoxon_p': t.get('wilcoxon_p'), 'H7_median_non_reproducible': t.get('median_b'),
                          'H7_mwu_p': t.get('mwu_p'), 'RE_pooled_r_reproducible': a.get('pooled_r'), 'RE_ci95': a.get('ci95_r'),
                          'RE_tau2': a.get('tau2')})
    pd.DataFrame(sens_rows).to_csv(OUT / 'beataml_sensitivity_subsets.csv', index=False)

    # fitness-axis correlates
    axis = Z.mean(axis=0)
    sd = axis.std()
    cl = clin.reindex(samples)
    cov = pd.DataFrame({'axis': axis.values, 'blasts_bm': cl['%.Blasts.in.BM'].map(parse_pct).values,
                        'blasts_pb': cl['%.Blasts.in.PB'].map(parse_pct).values,
                        'wbc': pd.to_numeric(cl.wbcCount, errors='coerce').values,
                        'flt3_itd': cl['FLT3-ITD'].map({'positive': 1, 'negative': 0}).values,
                        'npm1': cl.NPM1.map({'positive': 1, 'negative': 0}).values,
                        'bone_marrow': (cl.specimenType == 'Bone Marrow Aspirate').astype(float).values,
                        'initial_diagnosis': (cl.diseaseStageAtSpecimenCollection == 'Initial Diagnosis').astype(float).values},
                       index=samples)
    corr_rows = []
    for c in ['blasts_bm', 'blasts_pb', 'wbc']:
        m = cov[[c, 'axis']].dropna()
        r = spearmanr(m[c], m.axis)
        corr_rows.append({'covariate': c, 'type': 'spearman', 'estimate': float(r.correlation), 'p': float(r.pvalue), 'n': len(m)})
    for c in ['flt3_itd', 'npm1', 'bone_marrow', 'initial_diagnosis']:
        m = cov[[c, 'axis']].dropna()
        a, b = m.axis[m[c] == 1], m.axis[m[c] == 0]
        corr_rows.append({'covariate': c, 'type': 'difference_in_axis_sd_units (1 minus 0)', 'estimate': float((a.mean() - b.mean()) / sd),
                          'p': float(mannwhitneyu(a, b, alternative='two-sided').pvalue), 'n': len(m), 'n_positive': int(len(a))})
    import statsmodels.formula.api as smf
    mv = cov.assign(log_wbc=np.log(cov.wbc), axis_sd=cov.axis / sd).dropna(subset=['axis_sd', 'flt3_itd', 'npm1', 'log_wbc', 'blasts_bm'])
    fit = smf.ols('axis_sd ~ flt3_itd + npm1 + log_wbc + blasts_bm', data=mv).fit(cov_type='HC3')
    for k in ['flt3_itd', 'npm1', 'log_wbc', 'blasts_bm']:
        corr_rows.append({'covariate': k, 'type': 'multivariable OLS coefficient (axis in s.d. units; HC3)', 'estimate': float(fit.params[k]),
                          'p': float(fit.pvalues[k]), 'n': int(fit.nobs)})
    corr_rows.append({'covariate': 'all four', 'type': 'multivariable R2', 'estimate': float(fit.rsquared), 'p': float(fit.f_pvalue), 'n': int(fit.nobs)})
    pd.DataFrame(corr_rows).to_csv(OUT / 'beataml_axis_correlates.csv', index=False)

    # per-drug table, patient-bootstrap CIs and empirical-Bayes shrinkage
    lo, hi = fisher_ci(per.r, per.n)
    per['r_ci95_lo'], per['r_ci95_hi'] = lo, hi
    per['r_patient_boot_lo'] = np.nanpercentile(rb, 2.5, axis=0)
    per['r_patient_boot_hi'] = np.nanpercentile(rb, 97.5, axis=0)
    for c, mask in [('reproducible', isrep), ('non_reproducible', isnon)]:
        mm = re_meta(per.r[mask], per.n[mask])
        s, l, h = eb_shrink(per.r[mask], per.n[mask], mm)
        per.loc[mask, 'r_shrunk'] = s; per.loc[mask, 'r_shrunk_lo'] = l; per.loc[mask, 'r_shrunk_hi'] = h
    comp_t = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    tgt = comp_t.targets.reindex(per.compound).fillna('').values
    per['mek_inhibitor'] = [bool(any(k in t for k in ['MAP2K1', 'MAP2K2', 'MEK1', 'MEK2'])) and c == 'MAPK (RAF/MEK/ERK)'
                            for t, c in zip(tgt, per.curated_class)]
    per = per.sort_values(['repro', 'r'], ascending=[False, False])
    per.to_csv(OUT / 'beataml_per_drug.csv', index=False)
    rep_tab = per[per.repro == 'reproducible'].copy()
    rep_tab['rank_raw'] = rep_tab.r.rank(ascending=False, method='first')
    rep_tab['rank_shrunk'] = rep_tab.r_shrunk.rank(ascending=False, method='first')
    top = {}
    for kk in [5, 10]:
        t1 = rep_tab[rep_tab.rank_raw <= kk]; t2 = rep_tab[rep_tab.rank_shrunk <= kk]
        top[f'top{kk}_raw'] = list(t1.sort_values('r', ascending=False).name)
        top[f'top{kk}_raw_mek'] = int(t1.mek_inhibitor.sum())
        top[f'top{kk}_shrunk'] = list(t2.sort_values('r_shrunk', ascending=False).name)
        top[f'top{kk}_shrunk_mek'] = int(t2.mek_inhibitor.sum())
        top[f'top{kk}_curated_classes'] = t1.curated_class.value_counts().to_dict()
    top['mek_inhibitors_in_reproducible_set'] = list(rep_tab[rep_tab.mek_inhibitor].name)
    top['median_r_excluding_mek'] = float(rep_tab[~rep_tab.mek_inhibitor].r.median())
    top['wilcoxon_p_excluding_mek'] = float(wilcoxon(rep_tab[~rep_tab.mek_inhibitor].r, alternative='greater').pvalue)
    top['median_r_excluding_mapk_class'] = float(rep_tab[rep_tab.curated_class != 'MAPK (RAF/MEK/ERK)'].r.median())
    top['wilcoxon_p_excluding_mapk_class'] = float(wilcoxon(rep_tab[rep_tab.curated_class != 'MAPK (RAF/MEK/ERK)'].r, alternative='greater').pvalue)
    named = ['ARRY-886', 'TRAMETINIB', '212631-79-3', 'BMS-754807', '17-AAG']
    top['named_in_manuscript'] = per[per.name.isin(named)][['name', 'n', 'r', 'r_ci95_lo', 'r_ci95_hi', 'r_patient_boot_lo',
                                                            'r_patient_boot_hi', 'r_shrunk', 'r_shrunk_lo', 'r_shrunk_hi']].round(3).to_dict(orient='records')

    # class-level tests (old and curated classes); class statistic under patient-block permutation
    cl_res = {}
    for col in ['old_class', 'curated_class']:
        cl_res[col] = class_level(per.rename(columns={'r': 'rr'}), col, 'rr')
        cls_of = per.set_index('compound')[col].reindex(drugs).values
        rep_classes = sorted(set(cls_of[isrep]))
        M = np.stack([np.nanmean(nb[:, isrep & (cls_of == c)], axis=1) for c in rep_classes], axis=1)
        obs = np.nanmedian([np.nanmean(r_reg[isrep & (cls_of == c)]) for c in rep_classes])
        cl_res[col]['perm_p_median_class_mean_patient_blocks'] = perm_p(np.nanmedian(M, axis=1), obs)
    pd.DataFrame([{'scheme': k, **{kk: (json.dumps(vv) if isinstance(vv, dict) else vv) for kk, vv in v.items()}} for k, v in cl_res.items()]
                 ).to_csv(OUT / 'beataml_class_level.csv', index=False)

    tests = pd.DataFrame(rows)
    tests.to_csv(OUT / 'beataml_tests.csv', index=False)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(OUT / 'beataml_meta_analysis.csv', index=False)
    summary['beataml'] = {'structure': S, 'tests': tests.to_dict(orient='records'), 'meta': meta.to_dict(orient='records'),
                          'sensitivity_subsets': sens_rows, 'axis_correlates': corr_rows, 'top_drugs': top,
                          'class_level': cl_res}
    return per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert (OUT / 'curated_classes.csv').exists(), 'run 53_reproducibility_classes.py first'
    summary = {'settings': {'seed': SEED, 'n_perm': N_PERM, 'n_boot': N_BOOT, 'n_boot_twoway': N_BOOT_TWOWAY,
                            'n_draws': N_DRAWS, 'min_tumours': MIN_TUMOURS,
                            'tests': 'scipy 1.8 defaults as in the frozen scripts: wilcoxon(zero_method=wilcox, '
                                     'correction=False, exact if n<=25 else normal approximation); mannwhitneyu('
                                     'use_continuity=True, exact if min(n1,n2)<=8 without ties, else asymptotic); '
                                     'all one-sided (greater)',
                            'perm_p': '(1 + #null >= observed) / (1 + permutations)'}}
    parts = sys.argv[1:] or ['pdx', 'beataml']
    if 'pdx' in parts:
        pdx_analysis(summary)
    if 'beataml' in parts:
        beataml_analysis(summary)
    path = OUT / '54_summary.json'
    if path.exists() and len(parts) < 2:
        old = json.loads(path.read_text())
        old.update(summary)
        summary = old
    write_json(path, summary)
    for k in ['pdx', 'beataml']:
        if k in summary:
            print(k)
            print(pd.DataFrame(summary[k]['tests']).drop(columns=[c for c in ['perm_scheme'] if c in pd.DataFrame(summary[k]['tests']).columns]).round(4).to_string())


if __name__ == '__main__':
    main()
