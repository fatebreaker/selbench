"""Amendment 3 confirmatory evaluation in patient-derived in vitro tumour models of five cancer types (frozen before
retrieval).

Inputs: data/processed/pdm/<dataset>/{patient_long,sample_long}.csv.gz (compound, unit, sens; higher = more
sensitive) written by 71_load_pdm.py; the frozen transfer predictions results/pdm_transfer/<dataset>/pred_{sel,raw}.parquet
(compounds x CoderData samples with expression) and samples.csv written by 70_pdm_transfer_predictions.py; the discovery
reproducibility classes (results/decomposition/cross_screen_pairs.csv: mean selective cross-screen correlation >= 0.30
reproducible, < 0.15 non-reproducible), as for H6-H9.

Processing per dataset, at the patient level (that of the frozen 07_confirm.build with the thresholds registered here):
a drug needs >= 12 measured patients and >= 10 distinct values, and a dataset needs >= 8 retained drugs. Responses are
rank-normal within drug; the general-sensitivity axis is the leave-drug-out mean over all retained drugs of the dataset
(drugs outside discovery included). A patient's prediction is the mean of the frozen predictions over that patient's
samples with expression. A unit is a (dataset, compound) pair with >= 10 patients having both a prediction and a
response. Per unit: raw r(p, z), alignment r(p, g), loading r(z, g), the axis term r(p, g) r(z, g) and the selective
transfer r(p, z | g) (the exact split of the paper).

Pooled statistic: the precision-weighted mean of Fisher-z selective transfer (weights n - 4), back-transformed.
H10 (transfer of drug-specific signal to patient-derived models): the pooled selective transfer of the selective-trained
    predictions (pred_sel) over reproducible units is positive. One-sided permutation P < 0.05 from 10,000 permutations,
    each shuffling, within every dataset, the patient labels of the predictions jointly across drugs (so the dependence
    between drugs measured on the same patients is preserved). Needs >= 8 reproducible units. The 95% interval resamples
    patients within datasets (2,000 draws).
H11 (discovery reproducibility predicts transfer): pooled(reproducible) - pooled(non-reproducible) > 0 for pred_sel.
    One-sided permutation P < 0.05 from 10,000 shufflings of the class labels among the compounds of the two classes (a
    compound keeps one label in every dataset). Needs >= 3 units and >= 3 compounds per class.
Registered secondary estimates (no confirmatory claim): the H8/H9-style Wilcoxon and Mann-Whitney tests; per-dataset
estimates and a DerSimonian-Laird random-effects summary with a Hartung-Knapp interval; pred_raw; mean raw accuracy, the
axis term and the axis share of raw accuracy (ratio of means, as for the leaderboard) with patient-bootstrap intervals;
the drug-agnostic prediction (for each patient, the mean over compounds of pred_raw standardised within compound across
the dataset's patients) and its raw accuracy against pred_raw's; each dataset's axis split-half reliability and the
perfect-predictor axis share (mean loading squared); the sample-level analysis (response samples with their own
expression; descriptive).
Outputs: results/confirmation_PDM/{units.csv, sample_level_units.csv, per_dataset.csv, hypotheses.json}.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, t as tdist, wilcoxon

from common import PROCESSED, RESULTS, write_json

HERE = Path(__file__).resolve().parent


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


confirm = _module('confirm', '07_confirm.py')
loader = _module('pdm_loader', '71_load_pdm.py')
DATASETS = loader.DATASETS
BASE = PROCESSED / 'pdm'
TRANSFER = RESULTS / 'pdm_transfer'
OUT = RESULTS / 'confirmation_PDM'
REPRODUCIBLE, NONREPRODUCIBLE = 0.30, 0.15
MIN_PATIENTS_PER_DRUG = 12
MIN_DRUGS_PER_DATASET = 8
MIN_PAIRS = 10
MIN_UNITS_H10 = 8
MIN_PER_CLASS_H11 = 3
N_PERM = 10000
N_BOOT = 2000
SEED = confirm.SEED


def classes():
    dpairs = pd.read_csv(RESULTS / 'decomposition' / 'cross_screen_pairs.csv')
    rep = dpairs.groupby('compound').rho_sel.mean()
    return pd.Series(np.where(rep >= REPRODUCIBLE, 'reproducible',
                              np.where(rep < NONREPRODUCIBLE, 'non_reproducible', 'intermediate')), index=rep.index)


def terms(p, z, g):
    if min(np.std(p), np.std(z), np.std(g)) < 1e-12:
        return None
    c = np.corrcoef(np.vstack([p, z, g]))
    r_pz, r_pg, r_zg = c[0, 1], c[0, 2], c[1, 2]
    den = np.sqrt(max((1 - r_pg ** 2) * (1 - r_zg ** 2), 0))
    return {'raw': r_pz, 'selective': (r_pz - r_pg * r_zg) / den if den > 1e-9 else np.nan, 'alignment': r_pg,
            'loading': r_zg, 'mediated': r_pg * r_zg}


def fz(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def pooled_z(r, n):
    r, n = np.asarray(r, float), np.asarray(n, float)
    ok = np.isfinite(r) & (n > 4)
    if not ok.any():
        return np.nan
    w = n[ok] - 4
    return float((w * fz(r[ok])).sum() / w.sum())


def patient_predictions(ds, fam, key=loader.patient_key):
    P = pd.read_parquet(TRANSFER / ds / f'pred_{fam}.parquet')
    P.columns = P.columns.astype(str)
    s = pd.read_csv(TRANSFER / ds / 'samples.csv', dtype=str)
    s = s[s.improve_sample_id.isin(P.columns)]
    s['patient'] = s.common_name.map(key)
    return pd.DataFrame({p: P[ids].mean(axis=1) for p, ids in s.groupby('patient').improve_sample_id.apply(list).items()})


def agnostic(Pp):
    S = Pp.sub(Pp.mean(axis=1), axis=0).div(Pp.std(axis=1).replace(0, np.nan), axis=0)
    return S.mean(axis=0)


def build(long_path):
    """(Z, Gl) of a dataset, or None if fewer than MIN_DRUGS_PER_DATASET drugs pass the drug filters."""
    confirm.MIN_CELLS = MIN_PATIENTS_PER_DRUG
    long = pd.read_csv(long_path, dtype={'unit': str}).rename(columns={'unit': 'depmap_id'})
    g = long.dropna(subset=['sens']).groupby('compound')
    ok = (g.depmap_id.nunique() >= MIN_PATIENTS_PER_DRUG) & (g.sens.nunique() >= 10)
    if int(ok.sum()) < MIN_DRUGS_PER_DATASET:
        return None
    M, Z, Gl, R = confirm.build(long)
    return Z, Gl


class Dataset:
    """One dataset's patient-level matrices aligned to a common patient order."""

    def __init__(self, ds, Z, Gl, preds, cls):
        self.ds, self.Z, self.Gl, self.cols = ds, Z, Gl, list(Z.columns)
        self.P = {f: preds[f].reindex(columns=self.cols) for f in preds}
        self.P['agnostic'] = pd.DataFrame([agnostic(preds['raw']).reindex(self.cols)] * len(Z.index), index=Z.index)
        self.has_pred = np.isfinite(self.P['sel'].reindex(Z.index.intersection(preds['sel'].index)).values).any(axis=0) \
            if len(Z.index.intersection(preds['sel'].index)) else np.zeros(len(self.cols), bool)
        self.units = []
        for d in Z.index.intersection(preds['sel'].index):
            z, g = Z.loc[d].values, Gl.loc[d].values
            p = self.P['sel'].loc[d].values.astype(float)
            m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
            if m.sum() >= MIN_PAIRS:
                self.units.append({'compound': d, 'idx': np.where(m)[0], 'z': z[m], 'g': g[m],
                                   'class': cls.get(d, 'single_screen')})

    def unit_rows(self):
        rows = []
        for u in self.units:
            for fam in ['sel', 'raw', 'agnostic']:
                p = self.P[fam].loc[u['compound']].values.astype(float)[u['idx']]
                t = terms(p, u['z'], u['g'])
                if t is not None:
                    rows.append({'dataset': self.ds, 'compound': u['compound'], 'class': u['class'], 'family': fam,
                                 'n': len(u['idx']), **t})
        return rows

    def perm_selective(self, rng, n_perm, fam='sel'):
        """Selective transfer of every unit under n_perm joint permutations of the patients' predictions."""
        pos = np.where(self.has_pred)[0]
        perm = np.tile(np.arange(len(self.cols)), (n_perm, 1))
        perm[:, pos] = pos[np.argsort(rng.random((n_perm, len(pos))), axis=1)]
        out = np.full((n_perm, len(self.units)), np.nan)
        for j, u in enumerate(self.units):
            pfull = self.P[fam].loc[u['compound']].values.astype(float)
            Pb = pfull[perm[:, u['idx']]]
            X = np.c_[np.ones(len(u['g'])), u['g']]
            H = X @ np.linalg.pinv(X)
            Pr = Pb - Pb @ H
            zr = u['z'] - H @ u['z']
            den = np.linalg.norm(Pr, axis=1) * np.linalg.norm(zr)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[:, j] = np.where(den > 1e-12, Pr @ zr / den, np.nan)
        return out

    def boot_terms(self, counts, fam):
        """Weighted (bootstrap-count) raw, mediated and selective terms of every unit: arrays (B, units)."""
        B = counts.shape[0]
        res = {k: np.full((B, len(self.units)), np.nan) for k in ['raw', 'mediated', 'selective', 'n']}
        for j, u in enumerate(self.units):
            p = self.P[fam].loc[u['compound']].values.astype(float)[u['idx']]
            W = counts[:, u['idx']].astype(float)
            sw = W.sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                V = np.vstack([p, u['z'], u['g']])
                mu = (W @ V.T) / sw[:, None]
                C = np.einsum('bn,in,jn->bij', W, V, V) / sw[:, None, None] - mu[:, :, None] * mu[:, None, :]
                sd = np.sqrt(np.clip(np.einsum('bii->bi', C), 0, None))
                r = C / (sd[:, :, None] * sd[:, None, :])
                r_pz, r_pg, r_zg = r[:, 0, 1], r[:, 0, 2], r[:, 1, 2]
                den = np.sqrt(np.clip((1 - r_pg ** 2) * (1 - r_zg ** 2), 0, None))
                res['raw'][:, j], res['mediated'][:, j] = r_pz, r_pg * r_zg
                res['selective'][:, j] = np.where(den > 1e-9, (r_pz - r_pg * r_zg) / den, np.nan)
            res['n'][:, j] = sw
        return res


def wpooled_rows(R, N, mask):
    """Row-wise precision-weighted Fisher-z mean over the units in `mask` (arrays (B, units))."""
    W = np.where(np.isfinite(R[:, mask]) & (N[:, mask] > 4), N[:, mask] - 4, 0.0)
    Zf = np.where(W > 0, fz(np.nan_to_num(R[:, mask])), 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (W * Zf).sum(axis=1) / W.sum(axis=1)


def random_effects(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    k = len(est)
    if k < 2:
        return {'k': int(k)}
    w = 1 / se ** 2
    mu_fe = (w * est).sum() / w.sum()
    q = (w * (est - mu_fe) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    mu = (ws * est).sum() / ws.sum()
    hk = np.sqrt(max((ws * (est - mu) ** 2).sum() / ((k - 1) * ws.sum()), 1e-300))
    q975 = tdist.ppf(0.975, k - 1)
    return {'k': int(k), 'pooled_r': float(np.tanh(mu)), 'ci95_hksj': [float(np.tanh(mu - q975 * hk)),
                                                                        float(np.tanh(mu + q975 * hk))],
            'tau2': float(tau2), 'Q': float(q)}


def main(base=BASE, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cls = classes()
    sets, per_ds = [], []
    for ds in DATASETS:
        path = base / ds / 'patient_long.csv.gz'
        if not path.exists():
            continue
        built = build(path)
        if built is None or built[0].shape[0] < MIN_DRUGS_PER_DATASET:
            per_ds.append({'dataset': ds, 'excluded': 'fewer than 8 drugs with >= 12 patients and >= 10 distinct values'})
            continue
        Z, Gl = built
        row = {'dataset': ds, 'patients': int(Z.shape[1]), 'drugs_retained': int(Z.shape[0])}
        preds = {f: patient_predictions(ds, f) for f in ['sel', 'raw']}
        D = Dataset(ds, Z, Gl, preds, cls)
        r2 = []
        for d in Z.index:
            z, g = Z.loc[d].values, Gl.loc[d].values
            m = np.isfinite(z) & np.isfinite(g)
            if m.sum() >= 3 and np.std(g[m]) > 1e-12:
                r2.append(np.corrcoef(z[m], g[m])[0, 1] ** 2)
        row.update({'split_half_reliability': confirm.split_half(Z, rng),
                    'perfect_predictor_axis_share_mean_rzg2': float(np.mean(r2)) if r2 else np.nan,
                    'patients_with_prediction': int(D.has_pred.sum()), 'units': len(D.units)})
        per_ds.append(row)
        if D.units:
            sets.append(D)
    units = pd.DataFrame([r for D in sets for r in D.unit_rows()])
    units.to_csv(out / 'units.csv', index=False)
    res = {'datasets': per_ds, 'thresholds': {'min_patients_per_drug': MIN_PATIENTS_PER_DRUG,
                                               'min_drugs_per_dataset': MIN_DRUGS_PER_DATASET, 'min_pairs': MIN_PAIRS}}
    if units.empty:
        res['H10'] = res['H11'] = {'estimable': False}
        write_json(out / 'hypotheses.json', res)
        return res

    # unit bookkeeping across datasets (same order as units in each Dataset)
    uclass = np.array([u['class'] for D in sets for u in D.units])
    ucomp = np.array([u['compound'] for D in sets for u in D.units])
    obs = units[units.family == 'sel'].set_index(['dataset', 'compound'])
    keys = [(D.ds, u['compound']) for D in sets for u in D.units]
    r_obs = obs.selective.reindex(keys).values
    n_obs = obs.n.reindex(keys).values.astype(float)
    rep, non = uclass == 'reproducible', uclass == 'non_reproducible'

    # H10: patient-label permutation within datasets, jointly across drugs
    perm = np.hstack([D.perm_selective(rng, N_PERM) for D in sets])
    t_obs = pooled_z(r_obs[rep], n_obs[rep])
    h10 = {'units': int(rep.sum()), 'compounds': int(len(set(ucomp[rep]))),
           'datasets': sorted({k[0] for k, f in zip(keys, rep) if f})}
    counts = [np.stack([np.bincount(rng.integers(0, len(D.cols), len(D.cols)), minlength=len(D.cols))
                        for _ in range(N_BOOT)]) for D in sets]
    boot = {fam: [D.boot_terms(c, fam) for D, c in zip(sets, counts)] for fam in ['sel', 'raw', 'agnostic']}
    cat = {fam: {k: np.hstack([b[k] for b in boot[fam]]) for k in ['raw', 'mediated', 'selective', 'n']} for fam in boot}
    if rep.sum() >= MIN_UNITS_H10:
        t_perm = wpooled_rows(perm, np.tile(n_obs, (N_PERM, 1)), rep)
        tb = wpooled_rows(cat['sel']['selective'], cat['sel']['n'], rep)
        h10.update({'pooled_selective_transfer': float(np.tanh(t_obs)),
                    'ci95_patient_bootstrap': [float(np.tanh(np.nanpercentile(tb, 2.5))),
                                               float(np.tanh(np.nanpercentile(tb, 97.5)))],
                    'permutation_p_one_sided': float((1 + (t_perm >= t_obs).sum()) / (1 + N_PERM)),
                    'n_permutations': N_PERM})
        h10['pass'] = bool(h10['permutation_p_one_sided'] < 0.05)
        a = r_obs[rep][np.isfinite(r_obs[rep])]
        if len(a) >= MIN_UNITS_H10:
            h10['secondary_wilcoxon_p_one_sided'] = float(wilcoxon(a, alternative='greater').pvalue)
            h10['secondary_median_unit'] = float(np.median(a))
    else:
        h10['estimable'] = False

    # H11: class-label permutation among the compounds of the two classes
    h11 = {'units_reproducible': int(rep.sum()), 'units_non_reproducible': int(non.sum()),
           'compounds_reproducible': int(len(set(ucomp[rep]))), 'compounds_non_reproducible': int(len(set(ucomp[non])))}
    if min(h11.values()) >= MIN_PER_CLASS_H11:
        d_obs = pooled_z(r_obs[rep], n_obs[rep]) - pooled_z(r_obs[non], n_obs[non])
        pool = sorted(set(ucomp[rep]) | set(ucomp[non]))
        lab = np.array([cls[c] for c in pool])
        idx = np.array([pool.index(c) if c in pool else -1 for c in ucomp])
        inpool = idx >= 0
        d_perm = np.empty(N_PERM)
        for b in range(N_PERM):
            pl = rng.permutation(lab)
            ul = np.where(inpool, pl[np.clip(idx, 0, None)], 'other')
            d_perm[b] = pooled_z(r_obs[ul == 'reproducible'], n_obs[ul == 'reproducible']) - \
                pooled_z(r_obs[ul == 'non_reproducible'], n_obs[ul == 'non_reproducible'])
        h11.update({'pooled_reproducible': float(np.tanh(pooled_z(r_obs[rep], n_obs[rep]))),
                    'pooled_non_reproducible': float(np.tanh(pooled_z(r_obs[non], n_obs[non]))),
                    'difference_fisher_z': float(d_obs),
                    'permutation_p_one_sided': float((1 + (d_perm >= d_obs).sum()) / (1 + N_PERM))})
        h11['pass'] = bool(h11['permutation_p_one_sided'] < 0.05)
        a, b = r_obs[rep][np.isfinite(r_obs[rep])], r_obs[non][np.isfinite(r_obs[non])]
        if len(a) and len(b):
            h11['secondary_mwu_p_one_sided'] = float(mannwhitneyu(a, b, alternative='greater').pvalue)
    else:
        h11['estimable'] = False
    res['H10'], res['H11'] = h10, h11

    # registered secondary estimates
    sec = {}
    allu = np.ones(len(keys), bool)
    for fam in ['sel', 'raw', 'agnostic']:
        c = cat[fam]
        f = units[units.family == fam].set_index(['dataset', 'compound']).reindex(keys)
        e = {'units': int(f.raw.notna().sum()), 'mean_raw': float(f.raw.mean()),
             'mean_raw_ci95': [float(np.nanpercentile(np.nanmean(c['raw'], axis=1), q)) for q in (2.5, 97.5)],
             'pooled_selective_all_units': float(np.tanh(pooled_z(f.selective.values, f.n.values)))}
        with np.errstate(invalid='ignore', divide='ignore'):
            share_b = np.nanmean(c['mediated'], axis=1) / np.nanmean(c['raw'], axis=1)
        e['axis_share_of_raw_ratio_of_means'] = float(f.mediated.mean() / f.raw.mean()) if f.raw.mean() > 0 else None
        e['axis_share_ci95'] = [float(np.nanpercentile(share_b, q)) for q in (2.5, 97.5)]
        sec[fam] = e
    diff_b = np.nanmean(cat['raw']['raw'], axis=1) - np.nanmean(cat['agnostic']['raw'], axis=1)
    sec['raw_minus_agnostic_mean_raw'] = {'estimate': sec['raw']['mean_raw'] - sec['agnostic']['mean_raw'],
                                          'ci95': [float(np.nanpercentile(diff_b, q)) for q in (2.5, 97.5)]}
    # per dataset (reproducible units, pred_sel) and random effects across datasets
    est, se, pdrows = [], [], []
    for D, b in zip(sets, boot['sel']):
        m = np.array([u['class'] == 'reproducible' for u in D.units])
        k = [(D.ds, u['compound']) for u in D.units]
        r = obs.selective.reindex(k).values; n = obs.n.reindex(k).values.astype(float)
        if m.sum() == 0:
            continue
        tb = wpooled_rows(b['selective'], b['n'], m)
        e, s = pooled_z(r[m], n[m]), float(np.nanstd(tb, ddof=1))
        est.append(e); se.append(s)
        pdrows.append({'dataset': D.ds, 'reproducible_units': int(m.sum()), 'pooled_selective': float(np.tanh(e)),
                       'se_fisher_z_bootstrap': s})
    sec['per_dataset_reproducible'] = pdrows
    sec['random_effects_reproducible'] = random_effects(est, se)
    sec['by_class'] = units.groupby(['family', 'class']).agg(units=('selective', 'size'), median_raw=('raw', 'median'),
                                                             median_selective=('selective', 'median')).reset_index(
    ).to_dict(orient='records')
    # sample level (descriptive): response samples with their own expression
    srows = []
    for ds in DATASETS:
        path = base / ds / 'sample_long.csv.gz'
        if not path.exists():
            continue
        built = build(path)
        if built is None or built[0].shape[0] < MIN_DRUGS_PER_DATASET:
            continue
        Zs, Gs = built
        P = pd.read_parquet(TRANSFER / ds / 'pred_sel.parquet')
        P.columns = P.columns.astype(str)
        for d in Zs.index.intersection(P.index):
            p = P.loc[d].reindex(Zs.columns).values.astype(float)
            z, g = Zs.loc[d].values, Gs.loc[d].values
            m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
            if m.sum() >= MIN_PAIRS:
                t = terms(p[m], z[m], g[m])
                if t is not None:
                    srows.append({'dataset': ds, 'compound': d, 'class': cls.get(d, 'single_screen'), 'n': int(m.sum()),
                                  **t})
    sl = pd.DataFrame(srows)
    sl.to_csv(out / 'sample_level_units.csv', index=False)
    if len(sl):
        r = sl[sl['class'] == 'reproducible']
        sec['sample_level_reproducible'] = {'units': int(len(r)),
                                            'pooled_selective': float(np.tanh(pooled_z(r.selective, r.n)))}
    res['secondary'] = sec
    pd.DataFrame(per_ds).to_csv(out / 'per_dataset.csv', index=False)
    write_json(out / 'hypotheses.json', res)
    print({k: res[k] for k in ['H10', 'H11']}, flush=True)
    return res


if __name__ == '__main__':
    sys.exit(main() and None)
