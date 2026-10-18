"""Round 8 (a colleague's comment on the canonical-biomarker comparison): what does the cell-blind model add beyond a
known biomarker? EXPLORATORY, not confirmatory.

For the drugs of 52_biomarker_baselines with a pre-specified primary biomarker, measured in >= 3 discovery screens,
four predictors are scored with the same matched terms (training screen T, evaluation pair E1, E2) and the same c^2
estimator as in 52, all on the cells where the biomarker, the ridge and both evaluation screens are observed:
  biomarker     the training-free canonical biomarker (as in 52)
  ridge         kernel ridge on T's selective residual (out-of-fold predictions; 04)
  ridge_beyond  the ridge after least-squares regression on the biomarker and on both evaluation screens' axes (no
                response of the drug is used): reproducible selectivity the model captures that the biomarker misses
  combined      biomarker plus ridge, weighted by least squares on T's selective residual of the drug (E1 and E2 are not
                used to set the weights)
Per-drug cell-bootstrap intervals (52's 400 draws) and pooled ratios over drugs with drug-cluster bootstrap intervals
(2,000 draws), as in 52.
Outputs: results/revision_round8/biomarker_increment/{perdrug.csv, pooled.csv, summary.json}.
"""
import importlib.util
import os
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import RESULTS, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('b52', HERE / '52_biomarker_baselines.py')
b52 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(b52)
ce = b52.ce

OUT = RESULTS / 'revision_round8' / 'biomarker_increment'
SCREENS = b52.SCREENS
PREDICTORS = ['biomarker', 'ridge', 'ridge_beyond', 'combined']
N_BOOT = b52.N_BOOT
SEED = 10901


def ols_resid(y, X):
    X1 = np.c_[np.ones(len(y)), X]
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    return y - X1 @ beta


def combined_weights(fb, fr, rT, okT):
    """Least-squares weights of biomarker and ridge for T's selective residual (intercept dropped: c^2 is shift-free)."""
    if okT.sum() < ce.MIN_COMMON or np.std(fb[okT]) < 1e-12 or np.std(fr[okT]) < 1e-12:
        return None
    X1 = np.c_[np.ones(okT.sum()), fb[okT], fr[okT]]
    return np.linalg.lstsq(X1, rT[okT], rcond=None)[0][1:]


def drug_terms(d, fb, A, member, W, ridge):
    present = [s for s in SCREENS if d in A[s][0]]
    terms = {k: [] for k in PREDICTORS}
    for T, E1, E2 in b52.c2_terms(fb, d, A, member, T_list=SCREENS):
        if d not in ridge[T]:
            continue
        fr = np.asarray(ridge[T][d], dtype=float)
        z1, g1 = A[E1][0][d], A[E1][1][d]
        z2, g2 = A[E2][0][d], A[E2][1][d]
        ok = (member[E1] & member[E2] & np.isfinite(fb) & np.isfinite(fr) & np.isfinite(z1) & np.isfinite(z2)
              & np.isfinite(g1) & np.isfinite(g2))
        if ok.sum() < ce.MIN_COMMON or np.std(fb[ok]) < 1e-12:
            continue
        rT = A[T][2][d]
        okT = member[T] & np.isfinite(fb) & np.isfinite(fr) & np.isfinite(rT)
        w = combined_weights(fb, fr, rT, okT)
        if w is None:
            continue
        vec = {k: np.full(len(fb), np.nan) for k in PREDICTORS}
        vec['biomarker'][ok] = fb[ok]
        vec['ridge'][ok] = fr[ok]
        vec['ridge_beyond'][ok] = ols_resid(fr[ok], np.c_[fb[ok], g1[ok], g2[ok]])
        vec['combined'][ok] = w[0] * fb[ok] + w[1] * fr[ok]
        out = {k: b52.eval_term(vec[k], d, E1, E2, A, member, W) for k in PREDICTORS}
        if any(v is None for v in out.values()):
            continue
        for k in PREDICTORS:
            terms[k].append(out[k])
    return terms, len(present)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cat = b52.catalog_frame()
    Z = {s: pd.read_parquet(b52.DEC / f'z_{s}.parquet') for s in SCREENS}
    R = {s: pd.read_parquet(b52.DEC / f'resid_{s}.parquet') for s in SCREENS}
    G = {s: ce.loo_fitness(Z[s]) for s in SCREENS}
    cells = np.array(sorted(set().union(*[set(Z[s].columns) for s in SCREENS])))
    member = {s: np.isin(cells, np.asarray(Z[s].columns)) for s in SCREENS}
    A = {s: (ce.Aligned(Z[s], cells), ce.Aligned(G[s], cells), ce.Aligned(R[s], cells)) for s in SCREENS}
    W = ce.cell_weights(len(cells), b52.N_CELL_BOOT, np.random.default_rng(b52.SEED + 1))   # 52's cell draws
    cat['n_screens'] = [sum(c in Z[s].index for s in SCREENS) for c in cat.compound]
    feats = b52.cell_features(cat)
    ridge = {T: ce.Aligned(pd.read_parquet(RESULTS / 'models' / f'pred_ridge_sel_{T}.parquet'), cells) for T in SCREENS}

    rows = []
    for r in cat[(cat.n_screens >= 3) & cat.primary_for_drug].itertuples():
        fb = feats[r.biomarker].reindex(cells).values.astype(float)
        terms, _ = drug_terms(r.compound, fb, A, member, W, ridge)
        res = {k: b52.drug_c2(terms[k]) for k in PREDICTORS}
        if any(v is None for v in res.values()):
            continue
        row = {'compound': r.compound, 'drug_name': r.drug_name, 'biomarker': r.biomarker, 'tier': r.tier,
               'n_screens': r.n_screens, 'n_terms': res['biomarker'][3], 'den_sum': res['biomarker'][5]}
        for k in PREDICTORS:
            row[f'c2_{k}'] = res[k][0]
            row[f'c2_{k}_ci_lo'], row[f'c2_{k}_ci_hi'] = ce.ci(res[k][2])
            row[f'num_{k}'] = res[k][4]
        for a, b in [('combined', 'biomarker'), ('combined', 'ridge')]:
            dd = res[a][2] - res[b][2]
            row[f'diff_{a}_minus_{b}'] = res[a][0] - res[b][0]
            row[f'diff_{a}_minus_{b}_ci_lo'], row[f'diff_{a}_minus_{b}_ci_hi'] = ce.ci(dd)
        rows.append(row)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / 'perdrug.csv', index=False)

    pool_rows = []
    for label, sub in [('core, primary biomarker per drug', per[per.tier == 'core']),
                       ('all tiers, primary biomarker per drug', per)]:
        n = sub.n_terms.values
        dd = sub.den_sum.values / n
        C = ce.drug_counts(len(sub), N_BOOT, rng)
        est = {k: (sub[f'num_{k}'].values / n).sum() / dd.sum() for k in PREDICTORS}
        bst = {k: (C @ (sub[f'num_{k}'].values / n)) / (C @ dd) for k in PREDICTORS}
        row = {'set': label, 'n_drugs': len(sub)}
        for k in PREDICTORS:
            row[f'c2_{k}'] = est[k]
            row[f'c2_{k}_ci_lo'], row[f'c2_{k}_ci_hi'] = ce.ci(bst[k])
        for a, b in [('combined', 'biomarker'), ('combined', 'ridge')]:
            row[f'diff_{a}_minus_{b}'] = est[a] - est[b]
            row[f'diff_{a}_minus_{b}_ci_lo'], row[f'diff_{a}_minus_{b}_ci_hi'] = ce.ci(bst[a] - bst[b])
            row[f'n_{a}_above_{b}_ci_excludes_0'] = int((sub[f'diff_{a}_minus_{b}_ci_lo'] > 0).sum())
            row[f'n_{b}_above_{a}_ci_excludes_0'] = int((sub[f'diff_{a}_minus_{b}_ci_hi'] < 0).sum())
        row['n_ridge_beyond_ci_excludes_0'] = int((sub.c2_ridge_beyond_ci_lo > 0).sum())
        row['median_c2_ridge_beyond'] = float(sub.c2_ridge_beyond.median())
        pool_rows.append(row)
    pool = pd.DataFrame(pool_rows)
    pool.to_csv(OUT / 'pooled.csv', index=False)
    print(per[['drug_name', 'biomarker', 'tier', 'c2_biomarker', 'c2_ridge', 'c2_ridge_beyond', 'c2_combined',
               'diff_combined_minus_biomarker_ci_lo', 'diff_combined_minus_biomarker_ci_hi']].round(3).to_string(index=False))
    print(pool.T.to_string())
    core = per[per.tier == 'core']
    fam = (core.assign(gain=core.diff_combined_minus_biomarker_ci_lo > 0).groupby('biomarker')
           .agg(drugs=('compound', 'size'), combined_above_biomarker=('gain', 'sum')).reset_index())
    fam.to_csv(OUT / 'by_biomarker_core.csv', index=False)
    print(fam.to_string(index=False))
    write_json(OUT / 'summary.json', {'seed': SEED, 'cell_boot_seed': b52.SEED + 1, 'n_drugs': len(per),
                                      'pooled': pool.to_dict(orient='records'),
                                      'core_by_biomarker': fam.to_dict(orient='records'),
                                      'status': 'exploratory; biomarker list fixed a priori (52); not confirmatory'})


if __name__ == '__main__':
    main()
