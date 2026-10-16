"""Revision analysis (R3-M3 request 4, R3 minor 7): cross-laboratory triads, tetrad tests and a pooled confirmatory
factor model with laboratory and readout factors. Exploratory; confirmation screens are re-used descriptively.

Screens, laboratories and readouts (data/processed/screen_protocols.csv):
  PRISM Broad barcode (5 d) | CTRP Broad CellTiter-Glo | GDSC1 Sanger/MGH Syto60 or resazurin | GDSC2 Sanger CTG |
  gCSI Genentech CTG | NCI60 NCI SRB (48 h) | CCLE Novartis CTG | FIMM FIMM CTG | GRAY Gray lab (LBNL/OHSU) CTG.
Confirmation screens are rebuilt from data/processed/confirmation/*_long.csv.gz with selbench.decompose (identical to
the frozen 07_confirm.build; checked against results/confirmation_*/cross_screen_pairs.csv).

1. Cross-laboratory triads. Reliability of screen A from triads (A,B,C) whose three laboratories differ (variant
   'crosslab'), and additionally whose three readouts differ ('crosslab_readout'). c^2 of the per-screen cell-blind
   predictors (results/models, trained in one discovery screen T) from evaluation pairs E1,E2 such that T, E1 and E2
   come from three different laboratories (and readouts), and of the frozen consensus predictors
   (results/frozen_predictions, trained on Broad + Sanger screens) from pairs of confirmation screens of two other
   laboratories. Aggregated with the 49_ceiling_estimands estimators (primary: per-drug ratio of means, bias-
   corrected, no filter) and drug-cluster bootstrap; compared with the discovery-only estimate on the same drugs.
2. Discovery configurations. Discovery terms split by whether the evaluation pair shares a laboratory (T elsewhere)
   or the training screen shares a laboratory with one evaluation screen; c^2 of model and replicate by configuration.
3. Tetrad tests on drug-level correlation matrices (four-way common cell lines): discovery quadruple
   (PRISM, CTRP, GDSC1, GDSC2) and quadruples with gCSI or NCI-60 that isolate laboratory and CellTiter-Glo pairings.
4. Pooled confirmatory factor model (meta-analytic, drug-specific scale a_d) on all pairwise selective correlations of
   drugs measured in >= 3 of the nine screens, fitted by weighted least squares (weights n - 3):
       corr(r_s, r_s') = a_d [lambda_s lambda_s' + theta_lab 1(same lab) + psi_CTG 1(both CellTiter-Glo)]
   (multiplicative; an additive variant puts theta and psi outside a_d), optionally with the per-screen selective
   ridge f_T as indicators: corr(f_T, r_s) = a_d [gamma_T lambda_s + delta_lab 1(lab T = lab s) + delta_CTG 1(both CTG)].
   Drug-cluster bootstrap intervals. The fitted model is used to compute what the discovery-only estimator would
   report for model and replicate c^2, i.e. the direction of the bias from correlated errors.
5. R3 minor 7: shared fraction lambda^2 of each screen's fitness axis re-estimated by a bounded one-factor model with
   a cell-line bootstrap, and by pooled triads; values whose interval spans most of [0, 1] are flagged not estimable.
Outputs: results/revision_ceiling/50_*.csv and 50_summary.json.
"""
import importlib.util
import itertools
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common import PROCESSED, PROJECT, RESULTS, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('ce49', HERE / '49_ceiling_estimands.py')
ce = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ce)
sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402

DEC = RESULTS / 'decomposition'
OUT = Path(os.environ.get('REVISION_OUT', RESULTS / 'revision_ceiling'))   # override only for test runs
DISC = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
CONF = ['gCSI', 'NCI60', 'CCLE', 'FIMM', 'GRAY']
ALL = DISC + CONF
LAB = {'PRISM': 'Broad', 'CTRP': 'Broad', 'GDSC1': 'Sanger', 'GDSC2': 'Sanger', 'gCSI': 'Genentech', 'NCI60': 'NCI',
       'CCLE': 'Novartis', 'FIMM': 'FIMM', 'GRAY': 'Gray'}
READOUT = {'PRISM': 'barcode', 'CTRP': 'CTG', 'GDSC1': 'Syto60', 'GDSC2': 'CTG', 'gCSI': 'CTG', 'NCI60': 'SRB',
           'CCLE': 'CTG', 'FIMM': 'CTG', 'GRAY': 'CTG'}
PER_SCREEN_MODELS = ['ridge_sel', 'ridge_raw', 'ridge_crispr', 'lineage_raw', 'target_prior', 'fitness_omics', 'replicate']
FROZEN = ['ridge_sel_cons', 'ridge_raw_cons', 'lineage_sel_cons', 'target_prior', 'fit_omics', 'rep_sel_cons', 'rep_raw_cons']
MIN_COMMON = 20
N_BOOT = int(os.environ.get('REVISION_N_BOOT', 2000))
N_BOOT_CFA = int(os.environ.get('REVISION_N_BOOT_CFA', 400))
N_BOOT_AXIS = int(os.environ.get('REVISION_N_BOOT_AXIS', 1000))
WORKERS = int(os.environ.get('REVISION_WORKERS', 6))
SEED = 5001
PRIMARY = ce.PRIMARY


# ----------------------------------------------------------------------------------------------------------------
def build_confirmation(name, discovery_compounds):
    """selbench.decompose on the full long table (the fitness axis uses every compound); rows kept for drugs that
    also occur in a discovery screen."""
    long = pd.read_csv(PROCESSED / 'confirmation' / f'{name}_long.csv.gz')
    obs = long.rename(columns={'compound': 'drug', 'depmap_id': 'cell', 'sens': 'response'})
    Z, G, R, _ = selbench.decompose(obs)
    keep = [d for d in Z.index if d in discovery_compounds]
    return Z.loc[keep], G.loc[keep], R.loc[keep]


def load_all():
    Z, R, G = {}, {}, {}
    for s in DISC:
        Z[s] = pd.read_parquet(DEC / f'z_{s}.parquet')
        R[s] = pd.read_parquet(DEC / f'resid_{s}.parquet')
        G[s] = ce.loo_fitness(Z[s])
    disc_compounds = set().union(*[set(Z[s].index) for s in DISC])
    for s in CONF:
        Z[s], G[s], R[s] = build_confirmation(s, disc_compounds)
        print('built', s, Z[s].shape, flush=True)
    return Z, R, G


def verify_confirmation(Z, R):
    """Selective cross-screen correlations versus the frozen confirmation runs (07_confirm)."""
    out = {}
    for s in CONF:
        ref = pd.read_csv(RESULTS / f'confirmation_{s}' / 'cross_screen_pairs.csv')
        diffs = []
        for r in ref.itertuples():
            cells = Z[s].columns.intersection(Z[r.discovery].columns)
            a, b = R[s].loc[r.compound, cells].values, R[r.discovery].loc[r.compound, cells].values
            ok = np.isfinite(a) & np.isfinite(b)
            diffs.append(abs(np.corrcoef(a[ok], b[ok])[0, 1] - r.rho_sel) if ok.sum() >= MIN_COMMON else np.nan)
        out[s] = {'pairs': len(ref), 'max_abs_diff': float(np.nanmax(diffs))}
    return out


def pairwise_table(Rg, member, drugs_by_screen):
    """Selective-residual correlation for every drug and screen pair (pairwise-complete common cells)."""
    rows = []
    for s1, s2 in itertools.combinations(ALL, 2):
        base = member[s1] & member[s2]
        if base.sum() < MIN_COMMON:
            continue
        for d in drugs_by_screen[s1] & drugs_by_screen[s2]:
            x, y = Rg[s1][d], Rg[s2][d]
            ok = base & np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MIN_COMMON:
                continue
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            if np.isfinite(r):
                rows.append({'compound': d, 's1': s1, 's2': s2, 'n': int(ok.sum()), 'rho': float(r)})
    return pd.DataFrame(rows)


def predictor_table(P, Ag, member, drugs_by_screen):
    """Partial correlation of each per-screen predictor f_T with z_s given g_s for every other screen s."""
    rows = []
    for (m, T), Pv in P.items():
        for d in Pv.row:
            f = Pv[d]
            for s in ALL:
                if s == T or d not in drugs_by_screen[s]:
                    continue
                z, g = Ag[s][0][d], Ag[s][1][d]
                ok = member[s] & np.isfinite(f) & np.isfinite(z) & np.isfinite(g)
                if ok.sum() < MIN_COMMON:
                    continue
                C, _ = ce.wcov(np.c_[f[ok], z[ok], g[ok]], [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)])
                pc = float(ce.pcorr_from(C, 0, 1, 2)[0])
                if np.isfinite(pc):
                    rows.append({'model': m, 'train': T, 'compound': d, 'screen': s, 'n': int(ok.sum()), 'pc': pc})
    return pd.DataFrame(rows)


def labs_distinct(*s):
    return len({LAB[x] for x in s}) == len(s)


def readouts_distinct(*s):
    return len({READOUT[x] for x in s}) == len(s)


def crosslab_triads(Rg, member, drugs_by_screen):
    rows = []
    drugs = sorted(set().union(*drugs_by_screen.values()))
    for d in drugs:
        present = [s for s in ALL if d in drugs_by_screen[s]]
        if len(present) < 3:
            continue
        for A in present:
            for B, C in itertools.combinations([s for s in present if s != A], 2):
                if not labs_distinct(A, B, C):
                    continue
                base = member[A] & member[B] & member[C]
                if base.sum() < MIN_COMMON:
                    continue
                ab, ac, bc, n = ce.tri_term(Rg[A][d], Rg[B][d], Rg[C][d], base)
                if not np.all(np.isfinite([ab[0], ac[0], bc[0]])):
                    continue
                rows.append({'compound': d, 'screen': A, 'ref1': B, 'ref2': C, 'n_ok': float(n[0]),
                             'rho_ab': float(ab[0]), 'rho_ac': float(ac[0]), 'rho_bc': float(bc[0]),
                             'readouts_distinct': readouts_distinct(A, B, C)})
    t = pd.DataFrame(rows)
    t['a'] = t.rho_ab * t.rho_ac
    t['abc'] = ce.bias_corrected_product(t.rho_ab, t.rho_ac, t.rho_bc, t.n_ok)
    t['b'] = t.rho_bc
    return t


def crosslab_terms(P, Ag, member, drugs_by_screen, train_lab, allowed_evals):
    """Disattenuation terms whose training laboratory and two evaluation laboratories all differ."""
    rows = []
    for (m, T), Pv in P.items():
        tlabs = train_lab(T)
        for d in Pv.row:
            f = Pv[d]
            evals = [s for s in allowed_evals if s != T and d in drugs_by_screen[s] and LAB[s] not in tlabs]
            for E1, E2 in itertools.combinations(evals, 2):
                if LAB[E1] == LAB[E2]:
                    continue
                z1, g1, r1 = (Ag[E1][k][d] for k in range(3))
                z2, g2, r2 = (Ag[E2][k][d] for k in range(3))
                ok = member[E1] & member[E2] & np.isfinite(f) & np.isfinite(z1) & np.isfinite(z2) & np.isfinite(g1) & np.isfinite(g2)
                if ok.sum() < MIN_COMMON:
                    continue
                X = np.c_[f[ok], z1[ok], g1[ok], z2[ok], g2[ok], r1[ok], r2[ok]]
                p1, p2, r12, n = ce.dis_term(X)
                if not np.all(np.isfinite([p1[0], p2[0], r12[0]])):
                    continue
                ro = (T in READOUT and readouts_distinct(T, E1, E2))
                rows.append({'model': m, 'train': T, 'compound': d, 'e1': E1, 'e2': E2, 'n': int(ok.sum()),
                             'pc1': float(p1[0]), 'pc2': float(p2[0]), 'r12': float(r12[0]), 'readouts_distinct': ro})
    t = pd.DataFrame(rows)
    if len(t):
        t['a'] = t.pc1 * t.pc2
        t['abc'] = ce.bias_corrected_product(t.pc1, t.pc2, t.r12, t.n)
        t['b'] = t.r12
    return t


def est_with_ci(terms, rng, n_boot=N_BOOT, aggs=(PRIMARY, 'rom_terms', 'romeans_terms')):
    if len(terms) < 3 or terms.compound.nunique() < 3:
        return {}
    t, uniq = ce.index_drugs(terms)
    est = ce.aggregate(t)
    bs = ce.aggregate(t, ce.drug_counts(len(uniq), n_boot, rng))
    out = {agg: (float(est[agg][0]), *ce.ci(bs[agg])) for agg in aggs}
    out.update({'n_drugs': len(uniq), 'n_terms': len(t)})
    return out


def paired_ratio_diff(sets, rng, n_boot=N_BOOT):
    """sets: dict name -> (model terms, replicate terms). Ratios of the primary estimator on the drugs common to
    every set, with shared drug draws; returns estimates, the difference of the first two sets and its interval."""
    common = None
    for tm, tr in sets.values():
        c = set(tm.compound) & set(tr.compound)
        common = c if common is None else common & c
    common = np.array(sorted(common))
    if len(common) < 5:
        return None
    cnt = ce.drug_counts(len(common), n_boot, rng)
    cat = lambda t: t[t.compound.isin(common)].assign(drug_idx=lambda x: pd.Categorical(x.compound, categories=common).codes)
    est, boot = {}, {}
    for k, (tm, tr) in sets.items():
        a, b = cat(tm), cat(tr)
        est[k] = ce.aggregate(a, n_drugs=len(common))[PRIMARY][0] / ce.aggregate(b, n_drugs=len(common))[PRIMARY][0]
        boot[k] = ce.aggregate(a, cnt)[PRIMARY] / ce.aggregate(b, cnt)[PRIMARY]
    k1, k2 = list(sets)[:2]
    lo, hi = ce.ci(boot[k1] - boot[k2])
    return {'n_drugs': len(common), **{f'ratio_{k}': float(v) for k, v in est.items()},
            'diff': float(est[k1] - est[k2]), 'diff_ci_lo': lo, 'diff_ci_hi': hi}


def paired_ratio(tm, tr, rng, n_boot=N_BOOT):
    """Model/replicate ratio (primary aggregation) on common drugs with shared drug draws."""
    common = np.intersect1d(tm.compound.unique(), tr.compound.unique())
    if len(common) < 3:
        return (np.nan, np.nan, np.nan, len(common))
    cat = lambda t: t[t.compound.isin(common)].assign(drug_idx=lambda x: pd.Categorical(x.compound, categories=common).codes)
    a, b = cat(tm), cat(tr)
    cnt = ce.drug_counts(len(common), n_boot, rng)
    ea, eb = ce.aggregate(a, n_drugs=len(common))[PRIMARY][0], ce.aggregate(b, n_drugs=len(common))[PRIMARY][0]
    ba, bb = ce.aggregate(a, cnt)[PRIMARY], ce.aggregate(b, cnt)[PRIMARY]
    lo, hi = ce.ci(ba / bb)
    return (float(ea / eb), lo, hi, len(common))


# ----------------------------------------------------------------------------------------------------------------
# tetrads
# ----------------------------------------------------------------------------------------------------------------
QUADS = {
    'discovery': ('PRISM', 'CTRP', 'GDSC1', 'GDSC2'),
    'broadlab_vs_ctg_gCSI': ('PRISM', 'CTRP', 'GDSC1', 'gCSI'),
    'sangerlab_vs_ctg_gCSI': ('PRISM', 'GDSC1', 'GDSC2', 'gCSI'),
    'broadlab_NCI60': ('PRISM', 'CTRP', 'GDSC1', 'NCI60'),
    'sangerlab_NCI60': ('PRISM', 'GDSC1', 'GDSC2', 'NCI60'),
    'ctg_vs_broadlab_NCI60': ('CTRP', 'GDSC2', 'PRISM', 'NCI60'),
}


def pairing_label(p, q):
    lab = [LAB[p[0]] == LAB[p[1]], LAB[q[0]] == LAB[q[1]]]
    ctg = [READOUT[p[0]] == READOUT[p[1]] == 'CTG', READOUT[q[0]] == READOUT[q[1]] == 'CTG']
    parts = []
    if any(lab):
        parts.append('same-lab pair')
    if any(ctg):
        parts.append('CTG pair')
    return ' + '.join(parts) if parts else 'no shared lab/readout'


def tetrads(Rg, member, drugs_by_screen, rng):
    rows, per = [], []
    for qname, (w, x, y, z) in QUADS.items():
        base = member[w] & member[x] & member[y] & member[z]
        drugs = sorted(drugs_by_screen[w] & drugs_by_screen[x] & drugs_by_screen[y] & drugs_by_screen[z])
        pairings = [((w, x), (y, z)), ((w, y), (x, z)), ((w, z), (x, y))]
        vals = []
        for d in drugs:
            rr = {}
            ok_all = True
            for s1, s2 in itertools.combinations((w, x, y, z), 2):
                a, b = Rg[s1][d], Rg[s2][d]
                ok = base & np.isfinite(a) & np.isfinite(b)
                if ok.sum() < MIN_COMMON:
                    ok_all = False; break
                rr[(s1, s2)] = rr[(s2, s1)] = np.corrcoef(a[ok], b[ok])[0, 1]
            if not ok_all:
                continue
            prods = [rr[p] * rr[q] for p, q in pairings]
            vals.append(prods)
            per.append({'quad': qname, 'compound': d, **{f'{p[0]}-{p[1]}x{q[0]}-{q[1]}': v for (p, q), v in zip(pairings, prods)},
                        **{f'rho_{a}_{b}': rr[(a, b)] for a, b in itertools.combinations((w, x, y, z), 2)}})
        if len(vals) < 8:
            continue
        V = np.array(vals)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            diff = V[:, i] - V[:, j]
            bs = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(N_BOOT)]
            bm = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(N_BOOT)]
            p_i, p_j = pairings[i], pairings[j]
            rows.append({'quad': qname, 'n_drugs': len(V),
                         'product_1': f'rho({p_i[0][0]},{p_i[0][1]})*rho({p_i[1][0]},{p_i[1][1]})',
                         'type_1': pairing_label(*p_i),
                         'product_2': f'rho({p_j[0][0]},{p_j[0][1]})*rho({p_j[1][0]},{p_j[1][1]})',
                         'type_2': pairing_label(*p_j),
                         'mean_1': float(V[:, i].mean()), 'mean_2': float(V[:, j].mean()),
                         'mean_diff': float(diff.mean()), 'mean_diff_ci_lo': ce.ci(bs)[0], 'mean_diff_ci_hi': ce.ci(bs)[1],
                         'median_diff': float(np.median(diff)), 'median_diff_ci_lo': ce.ci(bm)[0],
                         'median_diff_ci_hi': ce.ci(bm)[1], 'frac_positive': float(np.mean(diff > 0)),
                         'wilcoxon_p_two_sided': float(wilcoxon(diff).pvalue)})
    return pd.DataFrame(rows), pd.DataFrame(per)


# ----------------------------------------------------------------------------------------------------------------
# pooled confirmatory factor model
# ----------------------------------------------------------------------------------------------------------------
LAB_PARAMS = ['Broad', 'Sanger']


class CFA:
    """Pooled factor model on pairwise correlations with a drug-specific scale a_d (alternating WLS)."""

    def __init__(self, obs, variant='multiplicative', lab=True, ctg=True, predictors=False, lab_by_screen=False):
        self.obs = obs.reset_index(drop=True)
        self.variant, self.lab, self.ctg, self.pred = variant, lab, ctg, predictors
        self.si = {s: i for i, s in enumerate(ALL)}
        o = self.obs
        self.kind_ss = (o.kind == 'ss').values
        self.i = o.s1.map(self.si).values
        self.j = o.s2.map(self.si).values
        self.lab_same = np.array([LAB[a] == LAB[b] for a, b in zip(o.s1, o.s2)])
        self.lab_idx = np.array([LAB_PARAMS.index(LAB[a]) if (LAB[a] == LAB[b] and LAB[a] in LAB_PARAMS) else -1
                                 for a, b in zip(o.s1, o.s2)])
        self.ctg_same = np.array([READOUT[a] == 'CTG' and READOUT[b] == 'CTG' for a, b in zip(o.s1, o.s2)])
        self.ti = np.array([DISC.index(t) if isinstance(t, str) and t in DISC else -1 for t in o.train])
        self.rho = o.rho.values.astype(float)
        self.w = np.clip(o.n.values - 3.0, 1.0, None)
        codes, self.drugs = pd.factorize(o.compound, sort=True)
        self.codes = codes
        self.nd = len(self.drugs)
        self.names = ([f'lambda_{s}' for s in ALL] + ([f'theta_{l}' for l in LAB_PARAMS] if lab else []) +
                      (['psi_CTG'] if ctg else []) +
                      ([f'gamma_{t}' for t in DISC] + (['delta_lab'] if lab else []) + (['delta_CTG'] if ctg else [])
                       if predictors else []))

    def unpack(self, p):
        k = len(ALL)
        lam = p[:k]
        th = p[k:k + 2] if self.lab else np.zeros(2); k += 2 if self.lab else 0
        psi = p[k] if self.ctg else 0.0; k += 1 if self.ctg else 0
        if self.pred:
            gam = p[k:k + 4]; k += 4
            dl = p[k] if self.lab else 0.0; k += 1 if self.lab else 0
            dc = p[k] if self.ctg else 0.0
        else:
            gam, dl, dc = np.zeros(4), 0.0, 0.0
        return lam, th, psi, gam, dl, dc

    def parts(self, p):
        """Returns (multiplied-by-a part, additive part) of the structural value for each observation."""
        lam, th, psi, gam, dl, dc = self.unpack(p)
        lab_term = np.where(self.lab_idx >= 0, th[np.clip(self.lab_idx, 0, 1)], 0.0)
        ctg_term = np.where(self.ctg_same, psi, 0.0)
        gen_ss = lam[self.i] * lam[self.j]
        gen_fs = gam[np.clip(self.ti, 0, 3)] * lam[self.j]
        corr_fs = np.where(self.lab_same, dl, 0.0) + np.where(self.ctg_same, dc, 0.0)
        gen = np.where(self.kind_ss, gen_ss, gen_fs)
        cor = np.where(self.kind_ss, lab_term + ctg_term, corr_fs)
        if self.variant == 'multiplicative':
            return gen + cor, np.zeros_like(gen)
        return gen, cor

    def update_a(self, p):
        m, add = self.parts(p)
        y = self.rho - add
        num = np.bincount(self.codes, self.w * y * m, self.nd)
        den = np.bincount(self.codes, self.w * m * m, self.nd)
        with np.errstate(invalid='ignore', divide='ignore'):
            a = np.where(den > 0, num / den, 0.0)
        return np.clip(a, 0.0, None)

    def resid(self, p, a):
        m, add = self.parts(p)
        return np.sqrt(self.w) * (self.rho - a[self.codes] * m - add)

    def normalise(self, p, a):
        k = a.mean()
        if not np.isfinite(k) or k <= 0:
            return p, a
        lam, th, psi, gam, dl, dc = self.unpack(p)
        lam = lam * np.sqrt(k); gam = gam * np.sqrt(k)
        if self.variant == 'multiplicative':
            th, psi, dl, dc = th * k, psi * k, dl * k, dc * k
        q = list(lam) + (list(th) if self.lab else []) + ([psi] if self.ctg else [])
        if self.pred:
            q += list(gam) + ([dl] if self.lab else []) + ([dc] if self.ctg else [])
        return np.array(q), a / k

    def fit(self, p0=None, max_iter=60, tol=1e-9):
        if p0 is None:
            p0 = np.r_[np.full(len(ALL), 0.6), np.zeros(len(self.names) - len(ALL))]
            if self.pred:
                p0[self.names.index('gamma_PRISM'):self.names.index('gamma_PRISM') + 4] = 0.3
        lb = np.array([0.0 if n.startswith('lambda') else -1.5 for n in self.names])
        ub = np.full(len(self.names), 1.5)
        p = np.clip(p0, lb + 1e-9, ub - 1e-9)
        a = self.update_a(p)
        prev = np.inf
        for _ in range(max_iter):
            r = least_squares(self.resid, p, args=(a,), bounds=(lb, ub), method='trf', x_scale='jac')
            p = r.x
            a = self.update_a(p)
            p, a = self.normalise(p, a)
            obj = float(np.sum(self.resid(p, a) ** 2))
            if prev - obj < tol * max(1.0, prev):
                break
            prev = obj
        self.p, self.a, self.obj = p, a, float(np.sum(self.resid(p, a) ** 2))
        return self

    def params(self):
        return dict(zip(self.names, self.p))


def _boot_cfa(args):
    obs, spec, p0, seed = args
    rng = np.random.default_rng(seed)
    drugs = obs.compound.unique()
    pick = rng.integers(0, len(drugs), len(drugs))
    g = {d: x for d, x in obs.groupby('compound')}
    parts = []
    for k, i in enumerate(pick):
        x = g[drugs[i]].copy(); x['compound'] = f'b{k}'
        parts.append(x)
    ob = pd.concat(parts, ignore_index=True)
    m = CFA(ob, **spec).fit(p0)
    return m.params(), m.obj


def derived(par, spec):
    """Shares of same-lab / same-readout correlation that are not general, and predictor ratios."""
    lam = {s: par[f'lambda_{s}'] for s in ALL}
    out = {f'R_general_{s}': lam[s] ** 2 for s in ALL}
    if spec.get('lab'):
        out['share_lab_PRISM_CTRP'] = par['theta_Broad'] / (lam['PRISM'] * lam['CTRP'] + par['theta_Broad'])
        out['share_lab_GDSC1_GDSC2'] = par['theta_Sanger'] / (lam['GDSC1'] * lam['GDSC2'] + par['theta_Sanger'])
    if spec.get('ctg'):
        out['share_ctg_CTRP_GDSC2'] = par['psi_CTG'] / (lam['CTRP'] * lam['GDSC2'] + par['psi_CTG'])
    if spec.get('predictors'):
        for t in DISC:
            out[f'c2_general_ridge_sel_{t}'] = par[f'gamma_{t}'] ** 2
            out[f'ratio_model_replicate_{t}'] = par[f'gamma_{t}'] ** 2 / lam[t] ** 2
        out['mean_ratio_model_replicate'] = float(np.mean([out[f'ratio_model_replicate_{t}'] for t in DISC]))
    return out


def implied_discovery_bias(model, spec, dis_terms, tri_terms=None):
    """What the discovery-only per-drug ratio-of-means estimator returns under the fitted factor model,
    versus the general-factor truth for the same terms and drugs (model: ridge_sel; replicate: z_T)."""
    par = model.params()
    lam = {s: par[f'lambda_{s}'] for s in ALL}
    th = {'Broad': par.get('theta_Broad', 0.0), 'Sanger': par.get('theta_Sanger', 0.0)}
    psi = par.get('psi_CTG', 0.0)
    gam = {t: par.get(f'gamma_{t}', np.nan) for t in DISC}
    dl, dc = par.get('delta_lab', 0.0), par.get('delta_CTG', 0.0)
    a_of = dict(zip(model.drugs, model.a))
    mult = spec.get('variant', 'multiplicative') == 'multiplicative'

    def ss(s1, s2, a):
        corr = (th[LAB[s1]] if LAB[s1] == LAB[s2] and LAB[s1] in th else 0.0) + \
               (psi if READOUT[s1] == READOUT[s2] == 'CTG' else 0.0)
        return a * (lam[s1] * lam[s2] + (corr if mult else 0.0)) + (0.0 if mult else corr)

    def fs(t, s, a):
        corr = (dl if LAB[t] == LAB[s] else 0.0) + (dc if READOUT[t] == READOUT[s] == 'CTG' else 0.0)
        return a * (gam[t] * lam[s] + (corr if mult else 0.0)) + (0.0 if mult else corr)
    out = {}
    for who in ['ridge_sel', 'replicate']:
        rows = []
        for r in dis_terms[dis_terms.model == who].itertuples():
            a = a_of.get(r.compound)
            if a is None:
                continue
            if who == 'replicate':
                p1, p2 = ss(r.train, r.e1, a), ss(r.train, r.e2, a)
                truth_num = a * lam[r.train] ** 2 * a * lam[r.e1] * lam[r.e2]
            else:
                p1, p2 = fs(r.train, r.e1, a), fs(r.train, r.e2, a)
                truth_num = a * gam[r.train] ** 2 * a * lam[r.e1] * lam[r.e2]
            rows.append({'compound': r.compound, 'num': p1 * p2, 'den': ss(r.e1, r.e2, a), 'tnum': truth_num,
                         'tden': a * lam[r.e1] * lam[r.e2], 'config': 'E_same_lab' if LAB[r.e1] == LAB[r.e2] else 'T_shares_lab'})
        t = pd.DataFrame(rows)
        if not len(t):
            continue
        for cfg, tt in [('all', t)] + list(t.groupby('config')):
            g = tt.groupby('compound')[['num', 'den', 'tnum', 'tden']].mean()
            out[f'{who}|{cfg}|implied_estimate'] = float(g.num.sum() / g.den.sum())
            out[f'{who}|{cfg}|general_truth'] = float(g.tnum.sum() / g.tden.sum())
    for cfg in ['all', 'E_same_lab', 'T_shares_lab']:
        k1, k2 = f'ridge_sel|{cfg}|implied_estimate', f'replicate|{cfg}|implied_estimate'
        if k1 in out and k2 in out:
            out[f'ratio|{cfg}|implied_estimate'] = out[k1] / out[k2]
            out[f'ratio|{cfg}|general_truth'] = out[f'ridge_sel|{cfg}|general_truth'] / out[f'replicate|{cfg}|general_truth']
    if tri_terms is not None:
        # discovery triad reliabilities R_A = rho_AB rho_AC / rho_BC (pooled, per-drug ratio of means)
        rows = []
        for r in tri_terms.itertuples():
            a = a_of.get(r.compound)
            if a is None:
                continue
            rows.append({'screen': r.screen, 'compound': r.compound, 'num': ss(r.screen, r.ref1, a) * ss(r.screen, r.ref2, a),
                         'den': ss(r.ref1, r.ref2, a), 'tnum': a * lam[r.screen] ** 2 * a * lam[r.ref1] * lam[r.ref2],
                         'tden': a * lam[r.ref1] * lam[r.ref2]})
        t = pd.DataFrame(rows)
        for scr, tt in t.groupby('screen'):
            g = tt.groupby('compound')[['num', 'den', 'tnum', 'tden']].mean()
            out[f'triad_R_{scr}|all|implied_estimate'] = float(g.num.sum() / g.den.sum())
            out[f'triad_R_{scr}|all|general_truth'] = float(g.tnum.sum() / g.tden.sum())
    return out


# ----------------------------------------------------------------------------------------------------------------
# fitness-axis shared fraction (R3 minor 7)
# ----------------------------------------------------------------------------------------------------------------
def axis_pair_corrs(F, W=None):
    names = list(F.columns)
    X = F.values
    out = {}
    for a, b in itertools.combinations(range(len(names)), 2):
        ok = np.isfinite(X[:, a]) & np.isfinite(X[:, b])
        if ok.sum() < 30:
            continue
        C, sw = ce.wcov(np.c_[X[ok, a], X[ok, b]], [(0, 0), (1, 1), (0, 1)], None if W is None else W[:, ok])
        out[(names[a], names[b])] = (ce.corr_from(C, 0, 1), sw)
    return out


def one_factor_fit(pairs, names):
    keys = list(pairs)
    rho = np.array([pairs[k][0] for k in keys]); n = np.array([pairs[k][1] for k in keys])
    ii = np.array([names.index(k[0]) for k in keys]); jj = np.array([names.index(k[1]) for k in keys])
    w = np.sqrt(np.clip(n - 3, 1, None))
    r = least_squares(lambda l: w * (rho - l[ii] * l[jj]), np.full(len(names), 0.5), bounds=(0.0, 1.0))
    return r.x ** 2


def pooled_triads(pairs, names):
    out = {}
    for s in names:
        num = den = 0.0; k = 0
        for t, u in itertools.combinations([x for x in names if x != s], 2):
            g = lambda a, b: pairs.get((a, b), pairs.get((b, a), (np.nan,)))[0]
            rst, rsu, rtu = g(s, t), g(s, u), g(t, u)
            if np.all(np.isfinite([rst, rsu, rtu])):
                num += rst * rsu; den += rtu; k += 1
        out[s] = (num / den if den > 0 else np.nan, k)
    return out


def fitness_axis_shared(rng):
    F = pd.read_csv(RESULTS / 'fitness_axis' / 'fitness_axes.csv', index_col=0)
    names = list(F.columns)
    base = axis_pair_corrs(F)
    pt = pooled_triads({k: (float(v[0][0]), float(v[1][0])) for k, v in base.items()}, names)
    lam2 = one_factor_fit({k: (float(v[0][0]), float(v[1][0])) for k, v in base.items()}, names)
    W = ce.cell_weights(len(F), N_BOOT_AXIS, rng)
    bp = axis_pair_corrs(F, W)
    boots_cfa, boots_pt = [], []
    for b in range(N_BOOT_AXIS):
        pb = {k: (float(v[0][b]), float(v[1][b])) for k, v in bp.items()}
        boots_cfa.append(one_factor_fit(pb, names))
        boots_pt.append([pooled_triads(pb, names)[s][0] for s in names])
    boots_cfa, boots_pt = np.array(boots_cfa), np.array(boots_pt)
    prev = __import__('json').loads((RESULTS / 'robustness' / 'summary.json').read_text())['shared_fraction_auc']
    rows = []
    for k, s in enumerate(names):
        lo, hi = ce.ci(boots_cfa[:, k])
        plo, phi = ce.ci(boots_pt[:, k])
        rows.append({'screen': s, 'n_cells': int(F[s].notna().sum()), 'paper_lambda2_median_triads': prev.get(s, {}).get('lambda2_median'),
                     'paper_ci': str(prev.get(s, {}).get('ci95')), 'cfa_lambda2': float(lam2[k]), 'cfa_ci_lo': lo, 'cfa_ci_hi': hi,
                     'pooled_triads_lambda2': float(np.clip(pt[s][0], 0, 1)), 'pooled_triads_raw': float(pt[s][0]),
                     'pooled_triads_ci_lo': plo, 'pooled_triads_ci_hi': phi, 'triads': pt[s][1],
                     'pooled_triads_ci_includes_zero': bool(np.isfinite(plo) and plo <= 0.0),
                     # not estimable: the bounded interval reaches the boundary at 0 or is wider than 0.4
                     'estimable': bool(np.isfinite(lo) and lo > 0.01 and (hi - lo) < 0.4)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary = {'seed': SEED, 'labs': LAB, 'readouts': READOUT}
    Z, R, G = load_all()
    summary['verify_confirmation_rebuild'] = verify_confirmation(Z, R)
    print('verify', summary['verify_confirmation_rebuild'], flush=True)
    cells = np.array(sorted(set().union(*[set(Z[s].columns) for s in ALL])))
    member = {s: np.isin(cells, np.asarray(Z[s].columns)) for s in ALL}
    drugs_by_screen = {s: set(Z[s].index) for s in ALL}
    Rg = {s: ce.Aligned(R[s], cells) for s in ALL}
    Ag = {s: (ce.Aligned(Z[s], cells), ce.Aligned(G[s], cells), Rg[s]) for s in ALL}
    summary['screens'] = {s: list(Z[s].shape) for s in ALL}

    # ---------------- per-screen and frozen predictors ----------------
    P = {}
    for m in PER_SCREEN_MODELS:
        for T in DISC:
            f = RESULTS / 'models' / f'pred_{m}_{T}.parquet'
            if f.exists():
                P[(m, T)] = ce.Aligned(pd.read_parquet(f), cells)
    PF = {(m, 'consensus'): ce.Aligned(pd.read_parquet(RESULTS / 'frozen_predictions' / f'{m}.parquet'), cells) for m in FROZEN}

    # ---------------- 1a. cross-lab triad reliabilities ----------------
    tri = crosslab_triads(Rg, member, drugs_by_screen)
    tri.to_csv(OUT / '50_crosslab_triad_terms.csv', index=False)
    disc_tri = pd.read_csv(RESULTS / 'ceiling' / 'triad_terms.csv').dropna()
    disc_tri['n_ok'] = disc_tri.n
    disc_tri['a'] = disc_tri.rho_ab * disc_tri.rho_ac
    disc_tri['abc'] = ce.bias_corrected_product(disc_tri.rho_ab, disc_tri.rho_ac, disc_tri.rho_bc, disc_tri.n)
    disc_tri['b'] = disc_tri.rho_bc
    rel_rows = []
    for s in ALL:
        ts = tri[tri.screen == s]
        no_nci = ~(ts.screen.eq('NCI60') | ts.ref1.eq('NCI60') | ts.ref2.eq('NCI60'))
        for variant, sub in [('crosslab', ts), ('crosslab_readout', ts[ts.readouts_distinct]), ('crosslab_no_NCI60', ts[no_nci]),
                             ('crosslab_n_ge_100', ts[ts.n_ok >= 100])]:
            e = est_with_ci(sub, rng)
            if not e:
                continue
            row = {'screen': s, 'variant': variant, 'n_drugs': e['n_drugs'], 'n_terms': e['n_terms'],
                   'median_n_cells': float(sub.n_ok.median()),
                   'reliability_primary': e[PRIMARY][0], 'ci_lo': e[PRIMARY][1], 'ci_hi': e[PRIMARY][2],
                   'reliability_rom': e['rom_terms'][0], 'rom_ci_lo': e['rom_terms'][1], 'rom_ci_hi': e['rom_terms'][2]}
            if s in DISC:
                dsub = disc_tri[(disc_tri.screen == s) & disc_tri.compound.isin(sub.compound.unique())]
                d = est_with_ci(dsub, rng)
                if d:
                    row.update({'discovery_triads_same_drugs_primary': d[PRIMARY][0], 'disc_ci_lo': d[PRIMARY][1],
                                'disc_ci_hi': d[PRIMARY][2], 'discovery_triads_same_drugs_rom': d['rom_terms'][0],
                                'discovery_n_drugs': d['n_drugs']})
            rel_rows.append(row)
    reltab = pd.DataFrame(rel_rows)
    reltab.to_csv(OUT / '50_crosslab_reliability.csv', index=False)
    print(reltab.round(3).to_string(), flush=True)

    # ---------------- 1b. cross-lab c^2 (per-screen predictors; frozen consensus predictors) ----------------
    terms = crosslab_terms(P, Ag, member, drugs_by_screen, lambda T: {LAB[T]}, ALL)
    terms.to_csv(OUT / '50_crosslab_c2_terms.csv', index=False)
    fterms = crosslab_terms(PF, Ag, member, drugs_by_screen, lambda T: {'Broad', 'Sanger'}, CONF)
    fterms.to_csv(OUT / '50_crosslab_c2_terms_frozen.csv', index=False)
    disc = pd.read_csv(OUT / '49_disattenuation_terms_bc.csv')
    c2_rows = []
    def c2_variants(t):
        no_nci = ~(t.e1.eq('NCI60') | t.e2.eq('NCI60'))
        return [('crosslab', t), ('crosslab_readout', t[t.readouts_distinct]), ('crosslab_no_NCI60', t[no_nci]),
                ('crosslab_n_ge_100', t[t.n >= 100])]
    for m in PER_SCREEN_MODELS:
        for variant, sub in c2_variants(terms[terms.model == m]):
            e = est_with_ci(sub, rng)
            if not e:
                continue
            dsub = disc[(disc.model == m) & disc.compound.isin(sub.compound.unique())]
            d = est_with_ci(dsub, rng)
            c2_rows.append({'model': m, 'variant': variant, 'n_drugs': e['n_drugs'], 'n_terms': e['n_terms'],
                            'median_n_cells': float(sub.n.median()), 'c2_primary': e[PRIMARY][0], 'ci_lo': e[PRIMARY][1],
                            'ci_hi': e[PRIMARY][2], 'c2_rom': e['rom_terms'][0],
                            'discovery_same_drugs_primary': d.get(PRIMARY, (np.nan,) * 3)[0],
                            'disc_ci_lo': d.get(PRIMARY, (np.nan,) * 3)[1], 'disc_ci_hi': d.get(PRIMARY, (np.nan,) * 3)[2],
                            'discovery_same_drugs_rom': d.get('rom_terms', (np.nan,))[0]})
    for m in FROZEN:
        sub = fterms[fterms.model == m]
        e = est_with_ci(sub, rng)
        if e:
            c2_rows.append({'model': m, 'variant': 'frozen_consensus_confirmation_pairs', 'n_drugs': e['n_drugs'],
                            'n_terms': e['n_terms'], 'median_n_cells': float(sub.n.median()), 'c2_primary': e[PRIMARY][0],
                            'ci_lo': e[PRIMARY][1], 'ci_hi': e[PRIMARY][2], 'c2_rom': e['rom_terms'][0]})
    c2tab = pd.DataFrame(c2_rows)
    # model / replicate ratios on common drugs
    ratio_rows = []
    for variant, sel in c2_variants(terms):
        for m in ['ridge_sel', 'ridge_raw', 'ridge_crispr']:
            r, lo, hi, nd = paired_ratio(sel[sel.model == m], sel[sel.model == 'replicate'], rng)
            common = np.intersect1d(sel[sel.model == m].compound.unique(), sel[sel.model == 'replicate'].compound.unique())
            dm = disc[(disc.model == m) & disc.compound.isin(common)]
            dr = disc[(disc.model == 'replicate') & disc.compound.isin(common)]
            r2, lo2, hi2, nd2 = paired_ratio(dm, dr, rng)
            ratio_rows.append({'model': m, 'variant': variant, 'ratio_crosslab': r, 'ci_lo': lo, 'ci_hi': hi, 'n_drugs': nd,
                               'ratio_discovery_same_drugs': r2, 'disc_ci_lo': lo2, 'disc_ci_hi': hi2})
    # paired: cross-laboratory minus discovery ratio on the same drugs and drug draws
    pr_rows = []
    for variant, sel in c2_variants(terms):
        res = paired_ratio_diff({'crosslab': (sel[sel.model == 'ridge_sel'], sel[sel.model == 'replicate']),
                                 'discovery': (disc[disc.model == 'ridge_sel'], disc[disc.model == 'replicate'])}, rng)
        if res:
            pr_rows.append({'variant': variant, **res})
    pd.DataFrame(pr_rows).to_csv(OUT / '50_crosslab_minus_discovery_ratio.csv', index=False)
    fr = paired_ratio(fterms[fterms.model == 'ridge_sel_cons'], fterms[fterms.model == 'rep_sel_cons'], rng)
    ratio_rows.append({'model': 'ridge_sel_cons / rep_sel_cons (consensus measurement)', 'variant': 'frozen_consensus_confirmation_pairs',
                       'ratio_crosslab': fr[0], 'ci_lo': fr[1], 'ci_hi': fr[2], 'n_drugs': fr[3]})
    rtab = pd.DataFrame(ratio_rows)
    c2tab.to_csv(OUT / '50_crosslab_c2.csv', index=False)
    rtab.to_csv(OUT / '50_crosslab_ratio.csv', index=False)
    print(c2tab.round(3).to_string(), '\n', rtab.round(3).to_string(), flush=True)

    # per-screen replicate in confirmation-only pairs (reference for the frozen consensus predictor)
    rep_conf = crosslab_terms({k: v for k, v in P.items() if k[0] in ('replicate', 'ridge_sel')}, Ag, member,
                              drugs_by_screen, lambda T: {LAB[T]}, CONF)
    fr_rows = []
    for m in ['ridge_sel', 'replicate']:
        e = est_with_ci(rep_conf[rep_conf.model == m], rng)
        if e:
            fr_rows.append({'model': f'{m} (single discovery screen)', 'variant': 'confirmation_pairs_only', 'n_drugs': e['n_drugs'],
                            'c2_primary': e[PRIMARY][0], 'ci_lo': e[PRIMARY][1], 'ci_hi': e[PRIMARY][2]})
    pd.DataFrame(fr_rows).to_csv(OUT / '50_confirmation_pairs_single_screen.csv', index=False)

    # ---------------- 2. discovery configurations ----------------
    disc['config'] = np.where(disc.e1.map(LAB) == disc.e2.map(LAB), 'E_same_lab', 'T_shares_lab')
    four = disc.groupby('compound').apply(lambda g: len(set(g.train) | set(g.e1) | set(g.e2)) == 4)
    four_drugs = four[four].index
    cfg_rows = []
    for cls_name, dd in [('all', disc), ('four_screen_drugs', disc[disc.compound.isin(four_drugs)])]:
        for cfg in ['E_same_lab', 'T_shares_lab']:
            for m in ['ridge_sel', 'ridge_raw', 'ridge_crispr', 'lineage_raw', 'target_prior', 'replicate']:
                sub = dd[(dd.model == m) & (dd.config == cfg)]
                e = est_with_ci(sub, rng)
                if e:
                    cfg_rows.append({'drugs': cls_name, 'config': cfg, 'model': m, 'n_drugs': e['n_drugs'], 'n_terms': e['n_terms'],
                                     'c2_primary': e[PRIMARY][0], 'ci_lo': e[PRIMARY][1], 'ci_hi': e[PRIMARY][2],
                                     'c2_rom': e['rom_terms'][0]})
            sm, sr = dd[(dd.model == 'ridge_sel') & (dd.config == cfg)], dd[(dd.model == 'replicate') & (dd.config == cfg)]
            r, lo, hi, nd = paired_ratio(sm, sr, rng)
            cfg_rows.append({'drugs': cls_name, 'config': cfg, 'model': 'ridge_sel/replicate', 'n_drugs': nd,
                             'c2_primary': r, 'ci_lo': lo, 'ci_hi': hi})
    cfgtab = pd.DataFrame(cfg_rows)
    cfgtab.to_csv(OUT / '50_discovery_configurations.csv', index=False)
    # paired diagnostic: does the training screen sharing a laboratory with an evaluation screen inflate the model
    # as much as the replicate? (A model that does not learn laboratory-specific selectivity would show a lower
    # ratio in T_shares_lab terms; see 51_ceiling_simulation.)
    diag = []
    for cls_name, dd in [('all', disc), ('four_screen_drugs', disc[disc.compound.isin(four_drugs)])]:
        res = paired_ratio_diff({'T_shares_lab': (dd[(dd.model == 'ridge_sel') & (dd.config == 'T_shares_lab')],
                                                  dd[(dd.model == 'replicate') & (dd.config == 'T_shares_lab')]),
                                 'E_same_lab': (dd[(dd.model == 'ridge_sel') & (dd.config == 'E_same_lab')],
                                                dd[(dd.model == 'replicate') & (dd.config == 'E_same_lab')])}, rng)
        common = None
        for m in ['ridge_sel', 'replicate']:
            for cfg in ['E_same_lab', 'T_shares_lab']:
                c = set(dd[(dd.model == m) & (dd.config == cfg)].compound)
                common = c if common is None else common & c
        common = np.array(sorted(common))
        cnt = ce.drug_counts(len(common), N_BOOT, rng)
        for m in ['ridge_sel', 'replicate']:
            e, bts = {}, {}
            for cfg in ['E_same_lab', 'T_shares_lab']:
                t = dd[(dd.model == m) & (dd.config == cfg) & dd.compound.isin(common)]
                t = t.assign(drug_idx=pd.Categorical(t.compound, categories=common).codes)
                e[cfg] = ce.aggregate(t, n_drugs=len(common))[PRIMARY][0]
                bts[cfg] = ce.aggregate(t, cnt)[PRIMARY]
            lo, hi = ce.ci(bts['T_shares_lab'] / bts['E_same_lab'])
            res[f'inflation_T_over_E_{m}'] = float(e['T_shares_lab'] / e['E_same_lab'])
            res[f'inflation_T_over_E_{m}_ci'] = f'{lo:.3f}-{hi:.3f}'
        diag.append({'drugs': cls_name, **res})
    pd.DataFrame(diag).to_csv(OUT / '50_configuration_diagnostic.csv', index=False)
    print(cfgtab.round(3).to_string(), flush=True)
    # implied GDSC1 reliability by discovery triad (four-screen drugs)
    g1 = disc_tri[(disc_tri.screen == 'GDSC1') & disc_tri.compound.isin(four_drugs)]
    g1_rows = []
    for (b1, b2), sub in g1.groupby(['ref1', 'ref2']):
        e = est_with_ci(sub, rng)
        g1_rows.append({'screen': 'GDSC1', 'refs': f'{b1}+{b2}', 'n_drugs': e['n_drugs'], 'R_primary': e[PRIMARY][0],
                        'ci_lo': e[PRIMARY][1], 'ci_hi': e[PRIMARY][2], 'R_rom': e['rom_terms'][0]})
    for s in DISC:
        sub_all = disc_tri[(disc_tri.screen == s) & disc_tri.compound.isin(four_drugs)]
        for (b1, b2), sub in sub_all.groupby(['ref1', 'ref2']):
            if s == 'GDSC1':
                continue
            e = est_with_ci(sub, rng)
            g1_rows.append({'screen': s, 'refs': f'{b1}+{b2}', 'n_drugs': e['n_drugs'], 'R_primary': e[PRIMARY][0],
                            'ci_lo': e[PRIMARY][1], 'ci_hi': e[PRIMARY][2], 'R_rom': e['rom_terms'][0]})
    pd.DataFrame(g1_rows).to_csv(OUT / '50_discovery_triad_types.csv', index=False)

    # ---------------- 3. tetrads ----------------
    tet, tet_per = tetrads(Rg, member, drugs_by_screen, rng)
    tet.to_csv(OUT / '50_tetrads.csv', index=False)
    tet_per.to_csv(OUT / '50_tetrads_perdrug.csv', index=False)
    print(tet.round(4).to_string(), flush=True)

    # ---------------- 4. pooled confirmatory factor model ----------------
    pw = pairwise_table(Rg, member, drugs_by_screen)
    pw.to_csv(OUT / '50_pairwise_selective_correlations.csv', index=False)
    nscr = pd.concat([pw[['compound', 's1']].rename(columns={'s1': 's'}), pw[['compound', 's2']].rename(columns={'s2': 's'})])
    nscr = nscr.drop_duplicates().groupby('compound').size()
    keep = nscr[nscr >= 3].index
    obs_ss = pw[pw.compound.isin(keep)].assign(kind='ss', train=None)
    pt = predictor_table({k: v for k, v in P.items() if k[0] == 'ridge_sel'}, Ag, member, drugs_by_screen)
    pt.to_csv(OUT / '50_predictor_screen_correlations.csv', index=False)
    obs_fs = pt[pt.compound.isin(keep)].rename(columns={'screen': 's2', 'pc': 'rho'}).assign(kind='fs', s1=lambda x: x.train)
    obs_fs = obs_fs[['compound', 's1', 's2', 'n', 'rho', 'kind', 'train']]
    specs = {
        'M0_general': dict(variant='multiplicative', lab=False, ctg=False),
        'M1_lab': dict(variant='multiplicative', lab=True, ctg=False),
        'M2_ctg': dict(variant='multiplicative', lab=False, ctg=True),
        'M3_lab_ctg': dict(variant='multiplicative', lab=True, ctg=True),
        'M3_lab_ctg_additive': dict(variant='additive', lab=True, ctg=True),
        'P0_general_predictors': dict(variant='multiplicative', lab=False, ctg=False, predictors=True),
        'P3_lab_ctg_predictors': dict(variant='multiplicative', lab=True, ctg=True, predictors=True),
        'P3_lab_ctg_predictors_additive': dict(variant='additive', lab=True, ctg=True, predictors=True),
    }
    cfa_rows, fits, der_rows = [], {}, []
    disc_terms = pd.read_csv(OUT / '49_disattenuation_terms_bc.csv')
    for name, spec in specs.items():
        obs = pd.concat([obs_ss, obs_fs]) if spec.get('predictors') else obs_ss
        m = CFA(obs, **spec).fit()
        fits[name] = m
        n_obs, n_par = len(m.obs), len(m.names) + m.nd
        args = [(obs, spec, m.p, SEED + 100 + b) for b in range(N_BOOT_CFA)]
        # 'spawn': forking after pyarrow has started its thread pool can deadlock the workers
        with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context('spawn')) as ex:
            boots = list(ex.map(_boot_cfa, args, chunksize=4))
        B = pd.DataFrame([b[0] for b in boots])
        for k, v in m.params().items():
            cfa_rows.append({'fit': name, 'parameter': k, 'estimate': float(v), 'ci_lo': ce.ci(B[k])[0], 'ci_hi': ce.ci(B[k])[1]})
        dv = derived(m.params(), spec)
        dvb = pd.DataFrame([derived(dict(zip(m.names, row)), spec) for row in B[m.names].values])
        for k, v in dv.items():
            cfa_rows.append({'fit': name, 'parameter': k, 'estimate': float(v), 'ci_lo': ce.ci(dvb[k])[0], 'ci_hi': ce.ci(dvb[k])[1]})
        ib = implied_discovery_bias(m, spec, disc_terms if spec.get('predictors') else
                                    disc_terms[disc_terms.model == 'replicate'], disc_tri)
        for k, v in ib.items():
            der_rows.append({'fit': name, 'quantity': k, 'value': v})
        cfa_rows.append({'fit': name, 'parameter': 'WRSS', 'estimate': m.obj, 'ci_lo': np.nan, 'ci_hi': np.nan})
        cfa_rows.append({'fit': name, 'parameter': 'n_obs', 'estimate': n_obs, 'ci_lo': np.nan, 'ci_hi': np.nan})
        cfa_rows.append({'fit': name, 'parameter': 'n_drugs', 'estimate': m.nd, 'ci_lo': np.nan, 'ci_hi': np.nan})
        cfa_rows.append({'fit': name, 'parameter': 'n_free_parameters', 'estimate': n_par, 'ci_lo': np.nan, 'ci_hi': np.nan})
        print(name, 'WRSS', round(m.obj, 2), {k: round(v, 3) for k, v in m.params().items()}, flush=True)
    cfatab = pd.DataFrame(cfa_rows)
    cfatab.to_csv(OUT / '50_cfa_parameters.csv', index=False)
    biastab = pd.DataFrame(der_rows)
    biastab.to_csv(OUT / '50_cfa_implied_discovery_bias.csv', index=False)
    print(biastab.round(4).to_string(), flush=True)
    # nested-model comparison by drug-level cross-validation (5 folds of drugs; held-out drugs' a_d refitted)
    cv_rows = []
    drugs = obs_ss.compound.unique()
    fold = np.random.default_rng(SEED + 7).permutation(np.arange(len(drugs)) % 5)
    fmap = dict(zip(drugs, fold))
    for name in ['M0_general', 'M1_lab', 'M2_ctg', 'M3_lab_ctg', 'M3_lab_ctg_additive']:
        spec = specs[name]; err = 0.0; wsum = 0.0
        for k in range(5):
            tr = obs_ss[obs_ss.compound.map(fmap) != k]; te = obs_ss[obs_ss.compound.map(fmap) == k]
            mt = CFA(tr, **spec).fit(fits[name].p)
            # held-out drugs: structure fixed, a_d estimated from half of each drug's pairs, error on the other half
            te = te.reset_index(drop=True)
            half = (te.groupby('compound').cumcount() % 2).values
            me_fit = CFA(te[half == 0], **spec); a_fit = me_fit.update_a(mt.p)
            me_eval = CFA(te[half == 1], **spec)
            a_map = dict(zip(me_fit.drugs, a_fit))
            a_eval = np.array([a_map.get(d, 0.0) for d in me_eval.drugs])
            r = me_eval.resid(mt.p, a_eval)
            err += float(np.sum(r ** 2)); wsum += float(np.sum(me_eval.w))
        cv_rows.append({'fit': name, 'cv_weighted_mse': err / wsum})
    cvtab = pd.DataFrame(cv_rows)
    cvtab.to_csv(OUT / '50_cfa_cv.csv', index=False)
    print(cvtab.to_string(), flush=True)

    # ---------------- 5. fitness-axis shared fraction (R3 minor 7) ----------------
    ax = fitness_axis_shared(np.random.default_rng(SEED + 9))
    ax.to_csv(OUT / '50_fitness_axis_shared_fraction.csv', index=False)
    print(ax.round(3).to_string(), flush=True)

    summary.update({'reliability': reltab.to_dict(orient='records'), 'c2': c2tab.to_dict(orient='records'),
                    'ratio': rtab.to_dict(orient='records'), 'configurations': cfgtab.to_dict(orient='records'),
                    'tetrads': tet.to_dict(orient='records'),
                    'cfa': cfatab.to_dict(orient='records'), 'cfa_implied_bias': biastab.to_dict(orient='records'),
                    'cfa_cv': cvtab.to_dict(orient='records'), 'fitness_axis': ax.to_dict(orient='records')})
    write_json(OUT / '50_summary.json', summary)


if __name__ == '__main__':
    main()
