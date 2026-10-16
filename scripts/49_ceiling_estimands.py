"""Revision analysis (R3-M3, R3 minors 1 and 12-related, R1 minor 10): estimand, sensitivity and uncertainty of the
c^2 ceiling and of the triad reliabilities. Exploratory; nothing frozen is read for confirmation.

Inputs (unchanged): results/decomposition/{z,resid}_*.parquet, results/models/pred_*_*.parquet,
results/decomposition/cross_screen_pairs.csv, results/ceiling/{disattenuation,triad}_terms.csv (for verification).

What is computed
1. The per-term quantities of 06_ceiling.py are recomputed with a weighted-moment engine (identical to 06 at unit
   weights; verified to machine precision) so that the same terms can be re-evaluated under cell resampling.
   Term k (drug d, predictor trained in T, evaluation pair E1,E2): a_k = pc(f,z_E1|g_E1) * pc(f,z_E2|g_E2) and
   b_k = corr(r_E1, r_E2). Small-sample correction: E[pc1*pc2] = rho1*rho2 + cov(pc1,pc2), with the Pearson-Filon
   covariance of two correlations sharing f, psi_k / n_k; a*_k = a_k - psi_k / n_k.
2. Aggregations of c^2 = "a / b" for every predictor, on all drugs (>=3 screens) and by reproducibility class, under
   denominator filters kappa in {none, >0, >0.05, >0.1, >0.2, >0.3}:
     rom_terms           median_k a_k / median_k b_k                         (paper; 06_ceiling)
     romeans_terms       mean_k a_k / mean_k b_k
     median_term_ratio   median_k (a_k / b_k)
     perdrug_rom         median_d mean_k(a) / median_d mean_k(b)             (per-drug averaging, then RoM)
     perdrug_romeans     sum_d mean_k(a) / sum_d mean_k(b)                   (per-drug averaging, then ratio of means)
     perdrug_romeans_bc  as perdrug_romeans with a*_k                         (PROPOSED PRIMARY, at kappa = none)
   with 95% drug-cluster bootstrap intervals (drugs resampled with all their terms; 2,000 draws) and, for the paper's
   aggregation, the original term-level bootstrap for contrast. Model/replicate ratios use the same drug draws.
   The same table is produced for the triad reliabilities R_A = rho_AB rho_AC / rho_BC (filter on rho_BC).
3. Two-way (drug x cell) bootstrap of the primary estimand: cell lines are resampled once per draw (shared by every
   drug and term, because cells are crossed with drugs) and drugs are resampled as clusters.
4. Hierarchical estimate: per-drug c^2_d = sum_k a*_k / sum_k b_k; its sampling s.e. on the Fisher-z scale of the
   signed root (atanh(sign*sqrt|c^2|)) from the cell bootstrap; a normal random-effects model (drug random effects,
   REML) gives empirical-Bayes per-drug estimates and the population distribution of c^2 (median, IQR, fraction above
   thresholds), with drug-cluster bootstrap intervals for the population median.
5. Reconciliation of c^2 with correlation ratios (R1 minor 10).
Predictors are those of 06_ceiling plus 'replicate_sel', the measured selective residual r_T of the training screen
(same cells as the paper's 'replicate', the raw measured profile z_T). Partialling the evaluation screen's axis does
not remove T's own screen-specific fitness from z_T, so r_T is the like-for-like measured reference for a selective
predictor; its c^2 is the triad reliability R_T computed on the c^2 terms.

Outputs: results/revision_ceiling/49_*.csv and 49_summary.json.
"""
import itertools
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from common import RESULTS, write_json  # noqa: E402

DEC = RESULTS / 'decomposition'
MOD = RESULTS / 'models'
OUT = RESULTS / 'revision_ceiling'
SCREENS = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
MODELS = ['ridge_sel', 'ridge_raw', 'ridge_crispr', 'lineage_raw', 'target_prior', 'fitness_omics', 'replicate',
          'replicate_sel']   # replicate: measured raw profile z_T (paper); replicate_sel: its selective residual r_T
MIN_COMMON = 20
KAPPAS = [None, 0.0, 0.05, 0.1, 0.2, 0.3]
AGGS = ['rom_terms', 'romeans_terms', 'median_term_ratio', 'perdrug_rom', 'perdrug_romeans', 'perdrug_romeans_bc']
PRIMARY = 'perdrug_romeans_bc'
N_BOOT = 2000          # drug-cluster bootstrap draws
N_CELL_BOOT = 400      # cell-bootstrap draws (two-way bootstrap and per-drug s.e.)
THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]
SEED = 4901
CLASS_BINS, CLASS_LABELS = [-1, 0.15, 0.3, 1.0], ['non_reproducible', 'intermediate', 'reproducible']


# ----------------------------------------------------------------------------------------------------------------
# weighted-moment engine (shared with scripts 50 and 52)
# ----------------------------------------------------------------------------------------------------------------
def wcov(X, pairs, W=None):
    """Weighted covariances of the columns of X (n x p) for the listed index pairs; W is (B x n) or None."""
    if W is None:
        W = np.ones((1, X.shape[0]))
    sw = W.sum(1)
    m1 = (W @ X) / sw[:, None]
    P = np.stack([X[:, i] * X[:, j] for i, j in pairs], 1)
    m2 = (W @ P) / sw[:, None]
    return {p: m2[:, q] - m1[:, p[0]] * m1[:, p[1]] for q, p in enumerate(pairs)}, sw


def _safe_div(num, den):
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan)


def pcorr_from(C, x, y, g):
    """Partial correlation of x and y given g (with intercept) from a covariance dict."""
    cxx = C[(x, x)] - C[(x, g)] ** 2 / C[(g, g)]
    cyy = C[(y, y)] - C[(y, g)] ** 2 / C[(g, g)]
    cxy = C[(x, y)] - C[(x, g)] * C[(y, g)] / C[(g, g)]
    bad = (cxx < 1e-12) | (cyy < 1e-12)
    return np.where(bad, np.nan, _safe_div(cxy, np.sqrt(np.clip(cxx, 1e-300, None) * np.clip(cyy, 1e-300, None))))


def corr_from(C, x, y):
    bad = (C[(x, x)] < 1e-12) | (C[(y, y)] < 1e-12)
    return np.where(bad, np.nan, _safe_div(C[(x, y)], np.sqrt(np.clip(C[(x, x)] * C[(y, y)], 1e-300, None))))


# columns: 0 f, 1 z1, 2 g1, 3 z2, 4 g2, 5 r1, 6 r2
DIS_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2), (3, 3), (4, 4), (0, 3), (0, 4), (3, 4),
             (5, 5), (6, 6), (5, 6)]


def dis_term(X, W=None):
    """pc(f,z1|g1), pc(f,z2|g2), corr(r1,r2) and n for term data X (n x 7)."""
    C, sw = wcov(X, DIS_PAIRS, W)
    return pcorr_from(C, 0, 1, 2), pcorr_from(C, 0, 3, 4), corr_from(C, 5, 6), sw


def pair_corr(x, y, W=None):
    """Weighted Pearson correlation of two complete vectors (B draws if W is given) and the weight total."""
    C, sw = wcov(np.c_[x, y], [(0, 0), (1, 1), (0, 1)], W)
    return corr_from(C, 0, 1), sw


def tri_term(ra, rb, rc, base, W=None):
    """rho_AB, rho_AC, rho_BC, each over its own complete cells within `base` (as 06_ceiling), and the A-pair n."""
    out, ns = [], []
    for x, y in [(ra, rb), (ra, rc), (rb, rc)]:
        ok = base & np.isfinite(x) & np.isfinite(y)
        if ok.sum() < MIN_COMMON:
            nb = 1 if W is None else W.shape[0]
            out.append(np.full(nb, np.nan)); ns.append(np.full(nb, np.nan)); continue
        r, sw = pair_corr(x[ok], y[ok], None if W is None else W[:, ok])
        out.append(r); ns.append(sw)
    return out[0], out[1], out[2], 0.5 * (ns[0] + ns[1])


def psi_overlap(r_jk, r_jh, r_kh):
    """n * Cov(r_jk, r_jh) for two correlations sharing variable j (Pearson-Filon / Olkin-Siotani)."""
    return r_kh * (1 - r_jk ** 2 - r_jh ** 2) - 0.5 * r_jk * r_jh * (1 - r_jk ** 2 - r_jh ** 2 - r_kh ** 2)


def bias_corrected_product(p1, p2, r12, n):
    return p1 * p2 - psi_overlap(p1, p2, r12) / n


def loo_fitness(Z):
    vals = Z.values; ok = np.isfinite(vals)
    tot = np.nansum(vals, axis=0); cnt = ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = (tot[None, :] - np.where(ok, vals, 0)) / (cnt[None, :] - ok)
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def cell_weights(n_cells, n_boot, rng):
    """Multinomial cell-resampling weights (n_boot x n_cells)."""
    W = np.zeros((n_boot, n_cells))
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, n_cells, n_cells), minlength=n_cells)
    return W


# ----------------------------------------------------------------------------------------------------------------
# aggregation with drug-cluster bootstrap
# ----------------------------------------------------------------------------------------------------------------
def weighted_median(values, weights):
    """Exact median of the sample in which values[i] is repeated weights[b, i] times (np.median convention).

    values: (n,), weights: (B, n) non-negative integers. Returns (B,)."""
    o = np.argsort(values, kind='mergesort')
    v = values[o]
    cw = np.cumsum(weights[:, o], axis=1)
    tot = cw[:, -1]
    r1 = (tot + 1) // 2
    r2 = tot // 2 + 1
    i1 = np.minimum((cw < r1[:, None]).sum(1), len(v) - 1)
    i2 = np.minimum((cw < r2[:, None]).sum(1), len(v) - 1)
    out = 0.5 * (v[i1] + v[i2])
    return np.where(tot > 0, out, np.nan)


def aggregate(terms, counts=None, num='a', num_bc='abc', den='b', n_drugs=None):
    """All aggregations for a term table (columns drug_idx, a, abc, b). counts: (B x n_drugs) or None (= ones)."""
    if counts is not None:
        n_drugs = counts.shape[1]
    elif n_drugs is None:
        n_drugs = int(terms.drug_idx.max()) + 1 if len(terms) else 0
    if counts is None:
        counts = np.ones((1, n_drugs))
    di = terms.drug_idx.values
    a, abc, b = terms[num].values, terms[num_bc].values, terms[den].values
    tw = counts[:, di]                                     # term weights (B x n_terms)
    out = {}
    out['rom_terms'] = weighted_median(a, tw) / weighted_median(b, tw)
    out['romeans_terms'] = (tw @ a) / (tw @ b)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = a / b
    ok = np.isfinite(ratio)
    out['median_term_ratio'] = weighted_median(ratio[ok], tw[:, ok])
    nk = np.bincount(di, minlength=n_drugs).astype(float)
    ma = np.bincount(di, weights=a, minlength=n_drugs) / nk
    mabc = np.bincount(di, weights=abc, minlength=n_drugs) / nk
    mb = np.bincount(di, weights=b, minlength=n_drugs) / nk
    present = nk > 0
    cw = counts[:, present]
    out['perdrug_rom'] = weighted_median(ma[present], cw) / weighted_median(mb[present], cw)
    out['perdrug_romeans'] = (cw @ ma[present]) / (cw @ mb[present])
    out['perdrug_romeans_bc'] = (cw @ mabc[present]) / (cw @ mb[present])
    return out


def drug_counts(n_drugs, n_boot, rng):
    C = np.zeros((n_boot, n_drugs))
    for b in range(n_boot):
        C[b] = np.bincount(rng.integers(0, n_drugs, n_drugs), minlength=n_drugs)
    return C


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return [np.nan, np.nan]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def filt(df, kappa, col='b'):
    return df if kappa is None else df[df[col] > kappa]


def index_drugs(df):
    df = df.copy()
    codes, uniq = pd.factorize(df.compound, sort=True)
    df['drug_idx'] = codes
    return df, uniq


# ----------------------------------------------------------------------------------------------------------------
# random-effects (REML) on the Fisher-z scale
# ----------------------------------------------------------------------------------------------------------------
def reml(y, s):
    """Normal-normal random-effects model y_i ~ N(mu, tau^2 + s_i^2); returns mu, tau^2 (REML)."""
    y, s = np.asarray(y, float), np.asarray(s, float)

    def negll(t2):
        v = t2 + s ** 2
        w = 1 / v
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(v)) + np.sum((y - mu) ** 2 / v) + np.log(np.sum(w)))
    hi = max(4 * np.var(y), 1e-4)
    r = minimize_scalar(negll, bounds=(0.0, hi), method='bounded', options={'xatol': 1e-8})
    t2 = float(r.x) if negll(r.x) <= negll(0.0) else 0.0
    w = 1 / (t2 + s ** 2)
    return float(np.sum(w * y) / np.sum(w)), t2


def signed_root_z(c2, clip=0.995):
    c = np.sign(c2) * np.sqrt(np.abs(c2))
    return np.arctanh(np.clip(c, -clip, clip))


def z_to_c2(z):
    return np.tanh(z) ** 2 * np.sign(z)


def population_summary(mu, t2, rng, n=200000):
    z = rng.normal(mu, np.sqrt(max(t2, 0.0)), n)
    c2 = z_to_c2(z)
    out = {'median': float(np.median(c2)), 'q25': float(np.percentile(c2, 25)), 'q75': float(np.percentile(c2, 75)),
           'mean': float(np.mean(c2))}
    for t in THRESHOLDS:
        out[f'frac_gt_{t}'] = float(np.mean(c2 > t))
    return out


# ----------------------------------------------------------------------------------------------------------------
def load_inputs():
    Z = {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in SCREENS}
    R = {s: pd.read_parquet(DEC / f'resid_{s}.parquet') for s in SCREENS}
    G = {s: loo_fitness(Z[s]) for s in SCREENS}
    P = {}
    for m in MODELS:
        for s in SCREENS:
            f = MOD / f'pred_{m}_{s}.parquet'
            if f.exists():
                P[(m, s)] = pd.read_parquet(f)
    for s in SCREENS:   # selective residual of the measured profile, on the same cells as the 'replicate' reference
        P[('replicate_sel', s)] = R[s].reindex(columns=P[('replicate', s)].columns)
    return Z, R, G, P


class Aligned:
    """A drug x cell matrix re-indexed to a global cell order, with fast row lookup."""

    def __init__(self, df, cells):
        self.v = df.reindex(columns=cells).values.astype(float)
        self.row = {d: i for i, d in enumerate(df.index)}

    def __contains__(self, d):
        return d in self.row

    def __getitem__(self, d):
        return self.v[self.row[d]]


def compute_dis_terms(Z, R, G, P, cells, W, screens=SCREENS, member=None):
    """Recompute 06_ceiling's disattenuation terms on the global cell index; returns table and bootstrap arrays."""
    if member is None:
        member = {s: np.isin(cells, np.asarray(Z[s].columns)) for s in screens}
    A = {s: (Aligned(Z[s], cells), Aligned(G[s], cells), Aligned(R[s], cells)) for s in screens}
    rows, boots = [], []
    for (m, T), Pm in P.items():
        Pv = Aligned(Pm, cells)
        for d in Pm.index:
            evals = [s for s in screens if s != T and d in A[s][0]]
            f_all = Pv[d]
            for E1, E2 in itertools.combinations(evals, 2):
                # cells measured in both evaluation screens (06: Z[E1].columns & Z[E2].columns)
                in_both = member[E1] & member[E2]
                z1, g1, r1 = (A[E1][k][d] for k in range(3))
                z2, g2, r2 = (A[E2][k][d] for k in range(3))
                ok = in_both & np.isfinite(f_all) & np.isfinite(z1) & np.isfinite(z2) & np.isfinite(g1) & np.isfinite(g2)
                if ok.sum() < MIN_COMMON:
                    continue
                X = np.c_[f_all[ok], z1[ok], g1[ok], z2[ok], g2[ok], r1[ok], r2[ok]]
                p1, p2, r12, n = dis_term(X)
                rows.append({'model': m, 'train': T, 'compound': d, 'e1': E1, 'e2': E2, 'n': int(ok.sum()),
                             'pc1': float(p1[0]), 'pc2': float(p2[0]), 'r12': float(r12[0])})
                if W is not None:
                    bp1, bp2, br12, bn = dis_term(X, W[:, ok])
                    boots.append(np.stack([bp1, bp2, br12, bn]))
    return pd.DataFrame(rows), (np.stack(boots) if boots else None)


def compute_triads(Z, R, cells, W, screens=SCREENS, member=None):
    if member is None:
        member = {s: np.isin(cells, np.asarray(Z[s].columns)) for s in screens}
    rows, boots = [], []
    Rg = {s: Aligned(R[s], cells) for s in screens}
    drugs = sorted(set().union(*[set(Z[s].index) for s in screens]))
    for d in drugs:
        present = [s for s in screens if d in Rg[s]]
        if len(present) < 3:
            continue
        for A_ in present:
            for B_, C_ in itertools.combinations([s for s in present if s != A_], 2):
                base = member[A_] & member[B_] & member[C_]
                if base.sum() < MIN_COMMON:
                    continue
                ra, rb, rc = Rg[A_][d], Rg[B_][d], Rg[C_][d]
                ab, ac, bc, n = tri_term(ra, rb, rc, base)
                rows.append({'compound': d, 'screen': A_, 'ref1': B_, 'ref2': C_, 'n': int(base.sum()), 'n_ok': float(n[0]),
                             'rho_ab': float(ab[0]), 'rho_ac': float(ac[0]), 'rho_bc': float(bc[0])})
                if W is not None:
                    bab, bac, bbc, bn = tri_term(ra, rb, rc, base, W)
                    boots.append(np.stack([bab, bac, bbc, bn]))
    return pd.DataFrame(rows), (np.stack(boots) if boots else None)


def add_classes(df, rep_sel):
    df = df.copy()
    df['rep_class'] = pd.cut(df.compound.map(rep_sel), CLASS_BINS, labels=CLASS_LABELS).astype(str)
    return df


def class_subsets(df):
    yield 'all', df
    for c in CLASS_LABELS:
        yield c, df[df.rep_class == c]


def estimand_table(terms, unit_col, unit_values, rng, label_col, den_col='b'):
    """Aggregations x filters x classes with drug-cluster bootstrap (and term bootstrap for the paper's RoM)."""
    rows, ratio_store = [], {}
    for u in unit_values:
        tu = terms[terms[unit_col] == u]
        for cls, tc in class_subsets(tu):
            for kappa in KAPPAS:
                tf = filt(tc, kappa, den_col)
                if len(tf) < 10:
                    continue
                tf, uniq = index_drugs(tf)
                est = {k: float(v[0]) for k, v in aggregate(tf).items()}
                cnt = drug_counts(len(uniq), N_BOOT, rng)
                bs = aggregate(tf, cnt)
                # the paper's term-level bootstrap of the ratio of medians (06_ceiling.boot_ratio)
                tb = []
                a, b = tf.a.values, tf.b.values
                for _ in range(N_BOOT):
                    i = rng.integers(0, len(a), len(a))
                    tb.append(np.median(a[i]) / np.median(b[i]))
                for agg in AGGS:
                    lo, hi = ci(bs[agg])
                    row = {label_col: u, 'class': cls, 'kappa': 'none' if kappa is None else f'>{kappa}', 'aggregation': agg,
                           'estimate': est[agg], 'ci_lo': lo, 'ci_hi': hi, 'n_terms': len(tf), 'n_drugs': len(uniq)}
                    if agg == 'rom_terms':
                        row['term_boot_ci_lo'], row['term_boot_ci_hi'] = ci(tb)
                    rows.append(row)
                ratio_store[(u, cls, kappa)] = (tf, uniq)
    return pd.DataFrame(rows), ratio_store


def ratio_table(store, models, ref, rng):
    """Model / replicate ratio under each aggregation on the drugs present for both (same drug draws)."""
    rows = []
    for (m, cls, kappa), (tm, um) in store.items():
        if m not in models or m == ref or (ref, cls, kappa) not in store:
            continue
        tr, ur = store[(ref, cls, kappa)]
        common = np.intersect1d(um, ur)
        if len(common) < 10:
            continue
        tm2, _ = index_drugs(tm[tm.compound.isin(common)].drop(columns='drug_idx'))
        tr2, _ = index_drugs(tr[tr.compound.isin(common)].drop(columns='drug_idx'))
        cnt = drug_counts(len(common), N_BOOT, rng)
        em, er = aggregate(tm2), aggregate(tr2)
        bm, br = aggregate(tm2, cnt), aggregate(tr2, cnt)
        for agg in AGGS:
            rows.append({'model': m, 'class': cls, 'kappa': 'none' if kappa is None else f'>{kappa}', 'aggregation': agg,
                         'c2_model': float(em[agg][0]), 'c2_replicate': float(er[agg][0]),
                         'ratio': float(em[agg][0] / er[agg][0]), 'ci_lo': ci(bm[agg] / br[agg])[0],
                         'ci_hi': ci(bm[agg] / br[agg])[1], 'ratio_corr_units': float(np.sqrt(max(em[agg][0] / er[agg][0], 0))),
                         'n_drugs': len(common)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    Z, R, G, P = load_inputs()
    cells = np.array(sorted(set().union(*[set(Z[s].columns) for s in SCREENS])))
    Wc = cell_weights(len(cells), N_CELL_BOOT, np.random.default_rng(SEED + 1))
    pairs = pd.read_csv(DEC / 'cross_screen_pairs.csv')
    rep_sel = pairs.groupby('compound').rho_sel.mean()
    summary = {'n_boot_drug': N_BOOT, 'n_boot_cell': N_CELL_BOOT, 'seed': SEED, 'primary_aggregation': PRIMARY}

    # ---------------- 1. recompute and verify the terms ----------------
    dis, dis_boot = compute_dis_terms(Z, R, G, P, cells, Wc)
    ref = pd.read_csv(RESULTS / 'ceiling' / 'disattenuation_terms.csv')
    key = ['model', 'train', 'compound', 'e1', 'e2']
    chk = ref.merge(dis, on=key, suffixes=('_ref', ''))
    summary['verify_dis_terms'] = {'rows_ref': len(ref), 'rows_new': len(dis), 'rows_matched': len(chk),
                                   'max_abs_diff': float(np.nanmax(np.abs(chk[['pc1', 'pc2', 'r12']].values -
                                                                          chk[['pc1_ref', 'pc2_ref', 'r12_ref']].values)))}
    tri, tri_boot = compute_triads(Z, R, cells, Wc)
    tref = pd.read_csv(RESULTS / 'ceiling' / 'triad_terms.csv')
    tchk = tref.merge(tri, on=['compound', 'screen', 'ref1', 'ref2'], suffixes=('_ref', ''))
    summary['verify_triad_terms'] = {'rows_ref': len(tref), 'rows_new': len(tri), 'rows_matched': len(tchk),
                                     'max_abs_diff': float(np.nanmax(np.abs(tchk[['rho_ab', 'rho_ac', 'rho_bc']].values -
                                                                            tchk[['rho_ab_ref', 'rho_ac_ref', 'rho_bc_ref']].values)))}
    print('verification', summary['verify_dis_terms'], summary['verify_triad_terms'], flush=True)

    keep = np.isfinite(dis[['pc1', 'pc2', 'r12']]).all(axis=1).values
    dis, dis_boot = dis[keep].reset_index(drop=True), dis_boot[keep]
    dis = add_classes(dis, rep_sel)
    dis['a'] = dis.pc1 * dis.pc2
    dis['abc'] = bias_corrected_product(dis.pc1, dis.pc2, dis.r12, dis.n)
    dis['b'] = dis.r12
    dis.to_csv(OUT / '49_disattenuation_terms_bc.csv', index=False)
    keep = np.isfinite(tri[['rho_ab', 'rho_ac', 'rho_bc']]).all(axis=1).values
    tri, tri_boot = tri[keep].reset_index(drop=True), tri_boot[keep]
    tri = add_classes(tri, rep_sel)
    tri['a'] = tri.rho_ab * tri.rho_ac
    tri['abc'] = bias_corrected_product(tri.rho_ab, tri.rho_ac, tri.rho_bc, tri.n_ok)
    tri['b'] = tri.rho_bc

    # ---------------- 2. estimand tables with drug-cluster bootstrap ----------------
    c2tab, store = estimand_table(dis, 'model', MODELS, rng, 'model')
    c2tab.to_csv(OUT / '49_c2_estimands.csv', index=False)
    rtab = ratio_table(store, MODELS, 'replicate', rng)
    rtab.to_csv(OUT / '49_c2_model_replicate_ratio.csv', index=False)
    reltab, _ = estimand_table(tri, 'screen', SCREENS, rng, 'screen')
    reltab.to_csv(OUT / '49_reliability_estimands.csv', index=False)
    print('estimand tables done', flush=True)

    # ---------------- 3. two-way (drug x cell) bootstrap of the primary estimand ----------------
    tw_rows = []
    dis_b = dis.copy()
    for m in MODELS:
        for cls, tc in [('all', dis_b[dis_b.model == m]), ('reproducible', dis_b[(dis_b.model == m) & (dis_b.rep_class == 'reproducible')])]:
            idx = tc.index.values
            tc, uniq = index_drugs(tc)
            est = aggregate(tc)[PRIMARY][0]
            cnt = drug_counts(len(uniq), N_CELL_BOOT, rng)
            drug_only = aggregate(tc, drug_counts(len(uniq), N_BOOT, rng))[PRIMARY]
            two, cell_only = [], []
            for b in range(N_CELL_BOOT):
                p1, p2, r12, n = dis_boot[idx, 0, b], dis_boot[idx, 1, b], dis_boot[idx, 2, b], dis_boot[idx, 3, b]
                tb = tc[['drug_idx']].copy()
                tb['a'] = p1 * p2; tb['abc'] = bias_corrected_product(p1, p2, r12, n); tb['b'] = r12
                tb = tb[np.isfinite(tb[['a', 'abc', 'b']]).all(axis=1)]
                cell_only.append(aggregate(tb, n_drugs=len(uniq))[PRIMARY][0])
                two.append(aggregate(tb, cnt[b:b + 1])[PRIMARY][0])
            tw_rows.append({'model': m, 'class': cls, 'estimate': est, 'drug_ci_lo': ci(drug_only)[0], 'drug_ci_hi': ci(drug_only)[1],
                            'cell_ci_lo': ci(cell_only)[0], 'cell_ci_hi': ci(cell_only)[1],
                            'twoway_ci_lo': ci(two)[0], 'twoway_ci_hi': ci(two)[1], 'n_drugs': len(uniq), 'n_terms': len(tc)})
    # two-way bootstrap of the model/replicate ratio (primary) on common drugs
    for ref_name, cls in itertools.product(['replicate', 'replicate_sel'], ['all', 'reproducible']):
        sub = dis_b if cls == 'all' else dis_b[dis_b.rep_class == 'reproducible']
        common = np.intersect1d(sub[sub.model == 'ridge_sel'].compound.unique(), sub[sub.model == ref_name].compound.unique())
        cnt = drug_counts(len(common), N_CELL_BOOT, rng)
        vals = {}
        for m in ['ridge_sel', ref_name]:
            tc = sub[(sub.model == m) & sub.compound.isin(common)]
            idx = tc.index.values
            tc = tc.copy(); tc['drug_idx'] = pd.Categorical(tc.compound, categories=common).codes
            vals[m] = [aggregate(tc)[PRIMARY][0]]
            for b in range(N_CELL_BOOT):
                p1, p2, r12, n = dis_boot[idx, 0, b], dis_boot[idx, 1, b], dis_boot[idx, 2, b], dis_boot[idx, 3, b]
                tb = tc[['drug_idx']].copy()
                tb['a'] = p1 * p2; tb['abc'] = bias_corrected_product(p1, p2, r12, n); tb['b'] = r12
                tb = tb[np.isfinite(tb[['a', 'abc', 'b']]).all(axis=1)]
                vals[m].append(aggregate(tb, cnt[b:b + 1])[PRIMARY][0])
        v1, v2 = np.array(vals['ridge_sel']), np.array(vals[ref_name])
        tw_rows.append({'model': f'ridge_sel/{ref_name}', 'class': cls, 'estimate': float(v1[0] / v2[0]),
                        'twoway_ci_lo': ci(v1[1:] / v2[1:])[0], 'twoway_ci_hi': ci(v1[1:] / v2[1:])[1], 'n_drugs': len(common)})
    twdf = pd.DataFrame(tw_rows)
    twdf.to_csv(OUT / '49_c2_primary_twoway_bootstrap.csv', index=False)
    print(twdf.round(4).to_string(), flush=True)

    # ---------------- 4. hierarchical (random-effects, Fisher-z) per-drug c^2 ----------------
    per_rows, hsum = [], []
    prng = np.random.default_rng(SEED + 2)
    for m in MODELS:
        dm = dis_b[dis_b.model == m]
        for d, g in dm.groupby('compound'):
            idx = g.index.values
            sb = g.b.sum()
            est = g.abc.sum() / sb
            p1, p2, r12, n = dis_boot[idx, 0], dis_boot[idx, 1], dis_boot[idx, 2], dis_boot[idx, 3]
            num_b = np.nansum(bias_corrected_product(p1, p2, r12, n), axis=0)
            den_b = np.nansum(r12, axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                est_b = num_b / den_b
            zb = signed_root_z(est_b[np.isfinite(est_b)])
            se_z = float(np.subtract(*np.percentile(zb, [75, 25])) / 1.349) if len(zb) > 20 else np.nan
            per_rows.append({'model': m, 'compound': d, 'rep_class': g.rep_class.iloc[0], 'n_terms': len(g),
                             'mean_b': float(g.b.mean()), 'c2_raw': float(est), 'z': float(signed_root_z(np.array([est]))[0]),
                             'se_z': se_z, 'estimable': bool(sb > 0 and np.isfinite(se_z) and se_z > 0)})
    per = pd.DataFrame(per_rows)
    for m in MODELS:
        for cls in ['all', 'reproducible']:
            sel = (per.model == m) & per.estimable & ((per.rep_class == 'reproducible') if cls == 'reproducible' else True)
            sub = per[sel]
            if len(sub) < 10:
                continue
            mu, t2 = reml(sub.z.values, sub.se_z.values)
            B = t2 / (t2 + sub.se_z.values ** 2)
            post_z = mu + B * (sub.z.values - mu)
            post_sd = np.sqrt(B * sub.se_z.values ** 2)
            per.loc[sub.index, f'eb_z_{cls}'] = post_z
            per.loc[sub.index, f'eb_c2_{cls}'] = z_to_c2(post_z)
            pop = population_summary(mu, t2, prng)
            draws = prng.normal(post_z[:, None], post_sd[:, None], (len(sub), 2000))
            fr = {f'eb_frac_gt_{t}': float(np.mean(z_to_c2(draws) > t)) for t in THRESHOLDS}
            # drug-cluster bootstrap of the REML population median and mean
            bmed = []
            for _ in range(1000):
                i = prng.integers(0, len(sub), len(sub))
                mb, tb2 = reml(sub.z.values[i], sub.se_z.values[i])
                bmed.append(float(np.median(z_to_c2(prng.normal(mb, np.sqrt(tb2), 20000)))))
            eb_c2 = z_to_c2(post_z)
            hsum.append({'model': m, 'class': cls, 'n_drugs': len(sub), 'n_not_estimable': int(((per.model == m) & ~per.estimable & ((per.rep_class == 'reproducible') if cls == 'reproducible' else True)).sum()),
                         'mu_z': mu, 'tau_z': float(np.sqrt(t2)), 'pop_median_c2': pop['median'],
                         'pop_median_ci_lo': ci(bmed)[0], 'pop_median_ci_hi': ci(bmed)[1],
                         'pop_q25': pop['q25'], 'pop_q75': pop['q75'], 'pop_mean_c2': pop['mean'],
                         **{f'pop_frac_gt_{t}': pop[f'frac_gt_{t}'] for t in THRESHOLDS},
                         'eb_median_c2': float(np.median(eb_c2)), 'eb_q25': float(np.percentile(eb_c2, 25)),
                         'eb_q75': float(np.percentile(eb_c2, 75)), **fr,
                         'raw_median_c2': float(np.median(sub.c2_raw)), 'median_se_z': float(np.median(sub.se_z))})
    per.to_csv(OUT / '49_c2_hierarchical_perdrug.csv', index=False)
    hdf = pd.DataFrame(hsum)
    hdf.to_csv(OUT / '49_c2_hierarchical_summary.csv', index=False)
    print(hdf[['model', 'class', 'n_drugs', 'mu_z', 'tau_z', 'pop_median_c2', 'pop_median_ci_lo', 'pop_median_ci_hi',
               'pop_q25', 'pop_q75', 'eb_median_c2']].round(4).to_string(), flush=True)

    # ---------------- 5. reconciliation with correlation ratios (R1 minor 10) ----------------
    rec = []
    rtab_sel = ratio_table(store, MODELS, 'replicate_sel', rng)
    rtab_sel.to_csv(OUT / '49_c2_model_replicate_sel_ratio.csv', index=False)
    for ref_name, tab in [('replicate', rtab), ('replicate_sel', rtab_sel)]:
        for cls in ['all', 'reproducible']:
            for agg, kap in [('rom_terms', '>0.1'), (PRIMARY, 'none')]:
                r = tab[(tab.model == 'ridge_sel') & (tab['class'] == cls) & (tab.aggregation == agg) & (tab.kappa == kap)]
                if len(r):
                    r = r.iloc[0]
                    rec.append({'source': f'ceiling {agg} kappa {kap}; reference {ref_name}', 'class': cls, 'c2_model': r.c2_model,
                                'c2_replicate': r.c2_replicate, 'variance_ratio': r.ratio, 'correlation_ratio': r.ratio_corr_units})
    lb = pd.read_csv(RESULTS / 'dreval_crosslab_leaderboard' / 'summary.csv')
    cs = pd.read_csv(RESULTS / 'dreval_crossstudy' / 'summary.csv')
    for tgt in ['GDSC1', 'GDSC2', 'CTRPv1']:
        mdl = lb[(lb.model == 'ridge_sel_lm') & (lb.target == tgt)].sel_cross
        meas = cs[(cs.algorithm == 'ref:CTRPv2_measured') & (cs['mode'] == 'LCO') & (cs.target == tgt)].sel_cross
        if len(mdl) and len(meas):
            cr = float(mdl.iloc[0] / meas.iloc[0])
            rec.append({'source': f'DrEval ridge_sel_lm vs CTRPv2 measured, scored in {tgt}', 'class': 'leaderboard drugs',
                        'sel_model': float(mdl.iloc[0]), 'sel_replicate': float(meas.iloc[0]), 'correlation_ratio': cr,
                        'variance_ratio': cr ** 2})
    rel_ctrp = reltab[(reltab.screen == 'CTRP') & (reltab['class'] == 'all')]
    for agg, kap in [('rom_terms', '>0.1'), (PRIMARY, 'none')]:
        v = rel_ctrp[(rel_ctrp.aggregation == agg) & (rel_ctrp.kappa == kap)]
        if len(v):
            for r in rec:
                if r['source'].startswith('DrEval'):
                    r[f'implied_c2_with_R_CTRP_{agg}'] = r['variance_ratio'] * float(v.estimate.iloc[0])
                    r[f'R_CTRP_{agg}'] = float(v.estimate.iloc[0])
    recdf = pd.DataFrame(rec)
    recdf.to_csv(OUT / '49_reconciliation.csv', index=False)
    print(recdf.round(4).to_string(), flush=True)

    # ---------------- key numbers ----------------
    def pick(tab, **kw):
        q = tab
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.iloc[0].to_dict() if len(q) else None
    key = {}
    for m in MODELS:
        for cls in ['all', 'reproducible']:
            key[f'{m}|{cls}|paper_rom_kappa0.1'] = pick(c2tab, model=m, **{'class': cls}, kappa='>0.1', aggregation='rom_terms')
            key[f'{m}|{cls}|primary'] = pick(c2tab, model=m, **{'class': cls}, kappa='none', aggregation=PRIMARY)
    summary['key_c2'] = key
    summary['twoway'] = twdf.to_dict(orient='records')
    summary['hierarchical'] = hdf.to_dict(orient='records')
    summary['reconciliation'] = recdf.to_dict(orient='records')
    write_json(OUT / '49_summary.json', summary)


if __name__ == '__main__':
    main()
