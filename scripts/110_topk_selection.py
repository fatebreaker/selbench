"""Round 9 (a colleague's suggestion): choosing cell lines for follow-up. EXPLORATORY; implements the timestamped plan
protocol/analysis_plan_topk_selection.md (OpenTimestamps proof next to it) without changes.

DrEval's CTRPv2 setting (CurveCurator ln IC50, leave-cell-line-out out-of-fold predictions of 59's main panel: the
leaderboard models and this study's drevalpy runs). For each drug measured in CTRPv2 and GDSC2, the candidates are the
cell lines predicted by every model, measured in CTRPv2 (ln IC50) and in GDSC2 (CurveCurator AUC). Each rule picks
k = 10 candidates per drug:
  R   the model's k highest predicted sensitivities;
  S   selectivity-aware: the k highest residuals of the prediction on the model's own leave-drug-out axis (59's
      zscore_axis), least squares with intercept over the candidates (not for drug-agnostic predictors).
References: CTRPv2's measured leave-drug-out axis (general sensitivity measured on other drugs) and CTRPv2's own
measurement of the drug (a second screen; its S version is the residual on that axis). Ties are scored exactly as the
expectation over uniform random tie-breaks.
Outcomes in GDSC2 (decomposed on the full AUC dataset): O1, share of picks in the top 10% of candidates by sensitivity
z_T; O2 (primary), share in the top 10% by the drug-specific residual r_T of z_T on GDSC2's axis over the candidates;
mean z_T and mean r_T / SD(r_T) of the picks; O3, distinct cell lines picked / (k x drugs) and mean pairwise Jaccard
overlap of the drugs' picks (averaged over 200 random tie-breaks). Comparisons: S - R per learned model; each rule
against the cancer-type baseline (tissue-by-drug mean); general-sensitivity rules against random; Spearman across
predictors of the within-CTRPv2 score (raw or selective median correlation, 59) with GDSC2 O2 of raw-rule picks.
Drug bootstrap (2,000 draws). Post hoc (not in the plan): O2 of picks by GDSC2's own axis; the S - R and model-choice
summaries without the three published models that are drug-agnostic within each split (41). Sensitivity: k = 5, 20; drugs with >= 30 candidates; GDSC2 ln IC50 outcome; S rule on
CTRPv2's measured axis.
Outputs: results/revision_round9/topk_selection/{per_drug.csv.gz, rules.csv, sensitivity.csv, summary.json}.
"""
import importlib.util
import os
import sys
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import RESULTS, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'selbench'))
import selbench  # noqa: E402

_spec = importlib.util.spec_from_file_location('s59', HERE / '59_retrain_score.py')
s59 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s59)

OUT = RESULTS / 'revision_round9' / 'topk_selection'
K = 10
MIN_CAND = 100
TOP = 0.10
N_BOOT = 2000
N_TIE = 200
SEED = 11001
MIN_CELLS = 30
CANCER_TYPE = 'LB:NaiveTissueDrugMeanPredictor'
GENERAL = ['FitnessOnlyRidge', 'LB:NaiveMeanEffectsPredictor', 'LB:NaiveTissueMeanPredictor']
OURS = ['EnsembleMF', 'EnsembleMFSelective', 'PerDrugRidgeRaw', 'PerDrugRidgeSelective']
NOT_POOLED = ['LB:SVR']
AGNOSTIC_IN_SPLIT = ['LB:ElasticNet', 'LB:Lasso', 'LB:MultiViewRandomForest']   # 41: drug-agnostic within each split
REF_AXIS = 'ref:CTRPv2 measured axis'
REF_SCREEN = 'ref:CTRPv2 measurement'


def ols_resid(y, x):
    X = np.c_[np.ones(len(y)), x]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def pick_weights(score, k):
    """Expected inclusion of each candidate in the top k under uniform random tie-breaking."""
    order = np.sort(score)[::-1]
    v = order[k - 1]
    tie = np.isclose(score, v, rtol=1e-12, atol=1e-12)
    above = (score > v) & ~tie
    w = above.astype(float)
    w[tie] = (k - above.sum()) / tie.sum()
    return w


def pick_draw(score, k, rng):
    """One random tie-break of the top k (indices)."""
    jitter = rng.random(len(score))
    return np.lexsort((jitter, -score))[:k]


def top_mask(y, frac):
    n = int(np.ceil(frac * len(y)))
    m = np.zeros(len(y), bool)
    m[np.argsort(-y, kind='stable')[:n]] = True
    return m


def group_of(m, learned):
    if m in learned:
        return 'learned'
    if m == CANCER_TYPE:
        return 'cancer type'
    if m in GENERAL or m == REF_AXIS:
        return 'general sensitivity'
    if m == REF_SCREEN:
        return 'second screen'
    return 'not pooled'


def load():
    oof, _, incomplete = s59.load_oof('CTRPv2_LNIC50')
    models = [m for m in sorted(oof) if not m.endswith('Dm')]          # 59's main panel
    cells = sorted(set.intersection(*[set(oof[m].cell) for m in models]))
    P = {m: oof[m][oof[m].cell.isin(cells)].pivot_table(index='drug', columns='cell', values='prediction')
         for m in models}
    H = {m: s59.zscore_axis(p.loc[p.notna().sum(axis=1) >= MIN_CELLS]) for m, p in P.items()}
    resp = s59.responses()
    obs = {key: resp[(resp.dataset == ds) & (resp.endpoint == ep)][['drug', 'cell', 'response']]
           for key, (ds, ep) in {'S': ('CTRPv2', 'lnic50'), 'T': ('GDSC2', 'auc'), 'TL': ('GDSC2', 'lnic50')}.items()}
    dec = {key: selbench.decompose(o)[:2] for key, o in obs.items()}
    learned = s59.published_pool(models) + [m for m in OURS if m in models]
    return models, learned, cells, P, H, dec, incomplete


def candidates(d, models, cells, P, Zs, Zt, Gt):
    if d not in Zs.index or d not in Zt.index or any(d not in P[m].index for m in models):
        return None
    ok = np.ones(len(cells), bool)
    for m in models:
        ok &= np.isfinite(P[m].loc[d].reindex(cells).values)
    ok &= np.isfinite(Zs.loc[d].reindex(cells).values)
    ok &= np.isfinite(Zt.loc[d].reindex(cells).values) & np.isfinite(Gt.loc[d].reindex(cells).values)
    return np.asarray(cells)[ok]


def score_rules(d, cand, models, learned, P, H, Zs, Gs, s_axis='own'):
    """Selection scores of every rule over the candidates of drug d."""
    sc = {}
    gs = Gs.loc[d].reindex(cand).values
    for m in models:
        p = P[m].loc[d].reindex(cand).values
        sc[(m, 'R')] = p
        if m in GENERAL:
            continue
        h = gs if s_axis == 'measured' else (H[m].loc[d].reindex(cand).values if d in H[m].index else None)
        if h is not None and np.isfinite(h).all() and np.std(h) > 1e-12:
            sc[(m, 'S')] = ols_resid(p, h)
    zs = Zs.loc[d].reindex(cand).values
    sc[(REF_AXIS, 'R')] = gs
    sc[(REF_SCREEN, 'R')] = zs
    sc[(REF_SCREEN, 'S')] = ols_resid(zs, gs)
    return sc


def evaluate(dec_t, k, min_cand, s_axis, models, learned, cells, P, H, dec, with_o3=False, rng=None):
    (Zs, Gs), (Zt, Gt) = dec['S'], dec_t
    drugs = sorted(set(Zs.index) & set(Zt.index))
    rows, picks = [], {}
    base, diag = [], []
    for d in drugs:
        cand = candidates(d, models, cells, P, Zs, Zt, Gt)
        if cand is None or len(cand) < max(min_cand, k):
            continue
        zt, gt = Zt.loc[d].reindex(cand).values, Gt.loc[d].reindex(cand).values
        rt = ols_resid(zt, gt)
        h1, h2 = top_mask(zt, TOP), top_mask(rt, TOP)
        sdr = np.std(rt)
        base.append(h1.mean())
        diag.append(float(pick_weights(gt, k) @ h2) / k)      # post hoc: GDSC2's own axis as the pick rule
        for (m, u), s in score_rules(d, cand, models, learned, P, H, Zs, Gs, s_axis).items():
            w = pick_weights(s, k)
            rows.append({'drug': d, 'model': m, 'rule': u, 'n_cand': len(cand), 'o1': float(w @ h1) / k,
                         'o2': float(w @ h2) / k, 'mean_z': float(w @ zt) / k, 'mean_r_sd': float(w @ rt) / k / sdr})
            if with_o3:
                picks.setdefault((m, u), {})[d] = (cand, s)
    per = pd.DataFrame(rows)
    o3 = {}
    if with_o3:
        cidx = {c: i for i, c in enumerate(cells)}
        for key, by_drug in picks.items():
            ties = any(np.any(pick_weights(s, k) % 1 > 0) for _, s in by_drug.values())
            dist, jac = [], []
            for _ in range(N_TIE if ties else 1):
                B = np.zeros((len(by_drug), len(cells)))
                for i, (c, s) in enumerate(by_drug.values()):
                    B[i, [cidx[x] for x in c[pick_draw(s, k, rng)]]] = 1
                dist.append((B.sum(axis=0) > 0).sum() / (k * len(by_drug)))
                inter = B @ B.T
                iu = np.triu_indices(len(by_drug), 1)
                jac.append((inter[iu] / (2 * k - inter[iu])).mean())
            o3[key] = (float(np.mean(dist)), float(np.mean(jac)))
    return per, float(np.mean(base)), o3, float(np.mean(diag))


def summarize(per, base, learned, within, rng, o3=None):
    """Rule table, comparisons and model-choice analysis with a drug bootstrap."""
    drugs = sorted(per.drug.unique())
    D = len(drugs)
    idx = rng.integers(0, D, size=(N_BOOT, D))
    M = {}
    for (m, u), g in per.groupby(['model', 'rule']):
        g = g.set_index('drug').reindex(drugs)
        M[(m, u)] = {c: g[c].values for c in ['o1', 'o2', 'mean_z', 'mean_r_sd']}
    boot = lambda v: v[idx].mean(axis=1)                                   # noqa: E731
    ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]   # noqa: E731
    base_b = np.full(N_BOOT, base)
    rows = []
    ct = M[(CANCER_TYPE, 'R')]
    for (m, u), v in sorted(M.items()):
        row = {'model': m, 'rule': u, 'group': group_of(m, learned), 'drugs': int(np.isfinite(v['o2']).sum())}
        for c in ['o1', 'o2', 'mean_z', 'mean_r_sd']:
            row[c] = float(np.mean(v[c]))
            row[f'{c}_ci_lo'], row[f'{c}_ci_hi'] = ci(boot(v[c]))
        for c in ['o1', 'o2']:
            dd = v[c] - ct[c]
            row[f'{c}_minus_cancer_type'] = float(dd.mean())
            row[f'{c}_minus_cancer_type_ci_lo'], row[f'{c}_minus_cancer_type_ci_hi'] = ci(boot(dd))
            row[f'{c}_minus_random'] = float(v[c].mean() - base)
            row[f'{c}_minus_random_ci_lo'], row[f'{c}_minus_random_ci_hi'] = ci(boot(v[c]) - base_b)
        if u == 'S' and (m, 'R') in M:
            for c in ['o1', 'o2']:
                dd = v[c] - M[(m, 'R')][c]
                row[f'{c}_S_minus_R'] = float(dd.mean())
                row[f'{c}_S_minus_R_ci_lo'], row[f'{c}_S_minus_R_ci_hi'] = ci(boot(dd))
        if o3 is not None and (m, u) in o3:
            row['distinct_share'], row['mean_jaccard'] = o3[(m, u)]
        rows.append(row)
    T = pd.DataFrame(rows)
    L = T[T.group == 'learned']
    LS = L[L.rule == 'S']
    out = {'drugs': D, 'k': K, 'random_expectation': base, 'n_learned': int(L.model.nunique())}
    # 1. S versus R over learned models
    diffs = np.array([M[(m, 'S')]['o2'] - M[(m, 'R')]['o2'] for m in LS.model])
    out['S_minus_R_o2_median_over_models'] = float(np.median(LS.o2_S_minus_R))
    out['S_minus_R_o2_median_over_models_ci'] = ci(np.median(diffs[:, idx].mean(axis=2), axis=0))
    out['S_minus_R_o2_n_above'] = int((LS.o2_S_minus_R_ci_lo > 0).sum())
    out['S_minus_R_o2_n_below'] = int((LS.o2_S_minus_R_ci_hi < 0).sum())
    LS2 = LS[~LS.model.isin(AGNOSTIC_IN_SPLIT)]                          # post hoc: as the paper's 18 learned
    out['post_hoc_not_agnostic_S_minus_R_o2_median'] = float(np.median(LS2.o2_S_minus_R))
    out['post_hoc_not_agnostic_S_minus_R_o2_n_below'] = int((LS2.o2_S_minus_R_ci_hi < 0).sum())
    out['post_hoc_not_agnostic_S_minus_R_o2_n_above'] = int((LS2.o2_S_minus_R_ci_lo > 0).sum())
    out['post_hoc_not_agnostic_n'] = len(LS2)
    out['S_minus_R_o1_median_over_models'] = float(np.median(LS.o1_S_minus_R))
    out['S_minus_R_o1_n_above'] = int((LS.o1_S_minus_R_ci_lo > 0).sum())
    out['S_minus_R_o1_n_below'] = int((LS.o1_S_minus_R_ci_hi < 0).sum())
    for u in ['R', 'S']:
        Lu = L[L.rule == u]
        out[f'learned_{u}_median_o1'] = float(Lu.o1.median())
        out[f'learned_{u}_median_o2'] = float(Lu.o2.median())
        for c in ['o1', 'o2']:
            out[f'learned_{u}_{c}_n_above_cancer_type'] = int((Lu[f'{c}_minus_cancer_type_ci_lo'] > 0).sum())
            out[f'learned_{u}_{c}_n_below_cancer_type'] = int((Lu[f'{c}_minus_cancer_type_ci_hi'] < 0).sum())
    # 3. model choice: Spearman across predictors of the within-CTRPv2 score with GDSC2 O2 of raw-rule picks
    for label, pool in [('all', [m for m in within.index if m not in NOT_POOLED and (m, 'R') in M]),
                        ('learned', [m for m in within.index if m in learned and (m, 'R') in M]),
                        ('post_hoc_learned_not_agnostic', [m for m in within.index if m in learned
                                                           and m not in AGNOSTIC_IN_SPLIT and (m, 'R') in M])]:
        O = np.array([M[(m, 'R')]['o2'] for m in pool])
        est = O.mean(axis=1)
        Ob = O[:, idx].mean(axis=2)
        for score in ['median_raw', 'median_selective']:
            x = within.loc[pool, score].values
            r = spearmanr(x, est)[0]
            rb = np.array([spearmanr(x, Ob[:, b])[0] for b in range(N_BOOT)])
            out[f'choice_{label}_{score}_spearman'] = float(r)
            out[f'choice_{label}_{score}_spearman_ci'] = ci(rb)
            out[f'_rb_{label}_{score}'] = rb
            top = pool[int(np.argmax(x))]
            out[f'choice_{label}_{score}_top_model'] = top
            out[f'choice_{label}_{score}_top_model_o2'] = float(M[(top, 'R')]['o2'].mean())
        d = out.pop(f'_rb_{label}_median_selective') - out.pop(f'_rb_{label}_median_raw')
        out[f'choice_{label}_spearman_difference'] = (out[f'choice_{label}_median_selective_spearman']
                                                     - out[f'choice_{label}_median_raw_spearman'])
        out[f'choice_{label}_spearman_difference_ci'] = ci(d)
        out[f'choice_{label}_n_models'] = len(pool)
    return T, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models, learned, cells, P, H, dec, incomplete = load()
    w = pd.read_csv(RESULTS / 'revision_retrain' / 'within_summary.csv')
    within = w[(w.setting == 'CTRPv2_LNIC50') & (w.panel == 'main')].set_index('model')[['median_raw', 'median_selective']]
    args = dict(models=models, learned=learned, cells=cells, P=P, H=H, dec=dec)

    per, base, o3, diag = evaluate(dec['T'], K, MIN_CAND, 'own', with_o3=True, rng=np.random.default_rng(SEED + 1),
                                   **args)
    per['group'] = [group_of(m, learned) for m in per.model]
    per.to_csv(OUT / 'per_drug.csv.gz', index=False)
    T, out = summarize(per, base, learned, within, np.random.default_rng(SEED), o3)
    T.to_csv(OUT / 'rules.csv', index=False)

    sens = []
    for label, dec_t, k, mc, sa in [('k = 5', dec['T'], 5, MIN_CAND, 'own'), ('k = 20', dec['T'], 20, MIN_CAND, 'own'),
                                    ('drugs with >= 30 candidates', dec['T'], K, 30, 'own'),
                                    ('GDSC2 ln IC50 outcome', dec['TL'], K, MIN_CAND, 'own'),
                                    ('S rule on CTRPv2 measured axis', dec['T'], K, MIN_CAND, 'measured')]:
        p2, b2, _, _ = evaluate(dec_t, k, mc, sa, **args)
        _, o = summarize(p2, b2, learned, within, np.random.default_rng(SEED))
        sens.append({'analysis': label, **{k_: (v if not isinstance(v, list) else f'{v[0]:.3f} to {v[1]:.3f}')
                                         for k_, v in o.items()}})
    pd.DataFrame(sens).to_csv(OUT / 'sensitivity.csv', index=False)

    out['post_hoc_o2_of_gdsc2_own_axis_picks'] = diag
    out['post_hoc_note'] = ('not in the plan: O2 of picks by GDSC2\'s own leave-drug-out axis (one screen\'s axis does '
                            'not remove all general sensitivity, so drug-agnostic picks exceed random on O2); S - R and '
                            'model-choice summaries without the three published models drug-agnostic within each split')
    out.update({'seed': SEED, 'n_boot': N_BOOT, 'min_candidates': MIN_CAND, 'top_fraction': TOP,
                'n_models_panel': len(models), 'incomplete_models_excluded': incomplete,
                'median_candidates': float(per.groupby('drug').n_cand.first().median()),
                'plan': 'protocol/analysis_plan_topk_selection.md (OpenTimestamps), implemented without changes',
                'status': 'exploratory'})
    write_json(OUT / 'summary.json', out)
    show = ['model', 'rule', 'group', 'o1', 'o2', 'mean_r_sd', 'o2_minus_cancer_type', 'o2_S_minus_R', 'distinct_share']
    print(T[show].sort_values(['group', 'o2']).round(3).to_string(index=False))
    print({k_: v for k_, v in out.items()})
    print(pd.DataFrame(sens)[['analysis', 'drugs', 'S_minus_R_o2_median_over_models', 'S_minus_R_o2_n_above',
                              'S_minus_R_o2_n_below', 'choice_all_median_raw_spearman',
                              'choice_all_median_selective_spearman']].to_string(index=False))


if __name__ == '__main__':
    main()
