"""Revision workstream: summaries of the paired cross-laboratory scores (outputs of 42_crosslab_scores.py).

All intervals are two-way (drug x cell) bootstrap percentile intervals (1,000 draws; 42_crosslab_scores.py). Pooled
quantities sum over the 21 published leaderboard models (classical + neural/factorization, SVR excluded) and the
drugs scored in both screens; they are intervals over drugs and cell lines, not over a population of methods.

Outputs (results/revision_crosslab/):
  summary_by_target.csv        per pair x drug subset (all; active in CTRPv2): median raw within -> cross (median over
                               models of per-model drug medians), pooled ratios (mediated retained, alignment r_pg,
                               target loading r_zg, selective-T and selective-BOTH retained, selective term retained),
                               absolute differences cross - within (mediated term, selective term, selective-T,
                               selective-BOTH), raw loss and the fitness share of the loss (reported only where the
                               raw-loss interval excludes 0)
  selective_metrics_by_target.csv  every per-drug metric (raw, selective-T/BOTH, adjusted, lineage, leave-class-out,
                               Spearman): pooled means within/cross, difference, ratio, median of model medians
  leakage_floors.csv           scores of the drug-agnostic predictors under every selective metric
  tost_selectivity.csv         equivalence of the selective-score change (cross - within): 90% CI and bootstrap
                               TOST P against margins of +-0.03 and +-0.05
  oracle_reference.csv         perfect-predictor (p = z) fitness share = mean r_zg^2; published models' through-
                               fitness share; one-replicate reference (CTRPv2's own measurement scored in the target)
  paired_comparisons.csv       ridge_sel_lm vs every published model (selective-BOTH, model-axis-adjusted, and
                               selective-T for reference) and vs the raw-target ridge with its own axis removed:
                               median paired difference with two-way bootstrap CI, Wilcoxon P, Holm-adjusted P
  censoring_effects.csv        ln IC50 censoring of the target dataset (per-drug fraction of pairs without a CurveCurator
                               ln IC50) against the target loading r_zg (Spearman over drugs) and the mediated-term
                               retention of low- vs high-censoring drugs (median split; drug-only bootstrap, 1,000 draws)
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from common import RESULTS, write_json  # noqa: E402

OUT = RESULTS / 'revision_crosslab'
SEED = 20261014
MARGINS = [0.03, 0.05]
AGNOSTIC = {'fitness_only': 'exact (omics-predicted fitness, this study)',
            'NaiveTissueMeanPredictor': 'exact (naive tissue mean)',
            'MultiViewRandomForest': 'exact (published model)',
            'NaiveMeanEffectsPredictor': 'within CV split (drug offsets differ between splits)',
            'Lasso': 'within CV split (additive model)', 'ElasticNet': 'within CV split (additive model)',
            'ref_CTRPv2_fitness_ldo': 'by construction (CTRPv2 measured leave-drug-out axis; not cell-blind)'}
METRIC_STATS = {'raw': ('raw_w', 'raw_c'), 'selT': ('selT_w', 'selT_c'), 'selBOTH': ('selT_w', 'selB_c'),
                'adjusted': ('adj_w', 'adj_c'), 'lineage': ('lin_w', 'lin_c'), 'lcoT': ('lcoT_w', 'lcoT_c'),
                'lcoBOTH': ('lcoT_w', 'lcoB_c'), 'lcxT': ('lcxT_w', 'lcxT_c'), 'lcxBOTH': ('lcxT_w', 'lcxB_c'),
                'spearman_raw': ('spr_w', 'spr_c'), 'spearman_selT': ('spsel_w', 'spsel_c')}
PRIMARY_FAMILY = ['selBOTH', 'adjusted', 'own_axis_removed_raw_ridge', 'own_axis_removed_raw_ridge_vs_selT']
EXACT_PUBLISHED = ['MultiViewRandomForest']   # exactly drug-agnostic published model (41_crosslab_data.agnostic_check)


def point_ci(rep, level=95):
    rep = np.asarray(rep, float)
    b = rep[1:][np.isfinite(rep[1:])]
    a = (100 - level) / 2
    lo, hi = (np.percentile(b, [a, 100 - a]) if len(b) >= 10 else (np.nan, np.nan))
    return float(rep[0]), float(lo), float(hi)


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = len(idx)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


def load_reps():
    pooled = pd.read_parquet(OUT / 'bootstrap_pooled.parquet')
    R = {k: g.sort_values('draw').value.values for k, g in pooled.groupby(['pair', 'subset', 'stat', 'kind'])}
    mm = pd.read_parquet(OUT / 'bootstrap_medmed.parquet')
    MM = {k: g.sort_values('draw').value.values for k, g in mm.groupby(['pair', 'subset', 'stat'])}
    return R, MM


def pair_meta():
    info = pd.read_csv(OUT / 'pair_info.csv')
    vt = pd.read_csv(OUT / 'versions.csv').set_index('version')
    info['dataset'] = info.target.map(vt.dataset)
    info['laboratory'] = info.target.map(vt.laboratory)
    info['processing'] = info.target.map(vt.processing)
    info['endpoint'] = info.target.map(vt.endpoint)
    info['identity'] = info.reference == info.target
    return info


def summary_by_target(info, R, MM):
    rows = []
    for _, pi in info.iterrows():
        for sub in ['all', 'active']:
            if (pi.pair, sub, 'raw', 'n') not in R:
                continue

            def g(stat, kind):
                return R[(pi.pair, sub, stat, kind)]
            n = g('raw', 'n')
            row = {k: pi[k] for k in ['pair', 'reference', 'target', 'dataset', 'laboratory', 'processing', 'endpoint',
                                      'cellset', 'family', 'identity']}
            row.update({'subset': sub, 'drugs': int(round(g('oracle_rzg2', 'n')[0])),
                        'model_drug_pairs': int(round(n[0]))})
            for side in ['w', 'c']:
                p, lo, hi = point_ci(MM[(pi.pair, sub, f'raw_{side}')])
                tag = 'within' if side == 'w' else 'cross'
                row.update({f'raw_median_{tag}': p, f'raw_median_{tag}_lo': lo, f'raw_median_{tag}_hi': hi})
            mw, mc = g('raw', 'sum_w') / n, g('raw', 'sum_c') / n
            for nm, rep in [('raw_mean_within', mw), ('raw_mean_cross', mc), ('raw_loss', mw - mc)]:
                p, lo, hi = point_ci(rep)
                row.update({nm: p, f'{nm}_lo': lo, f'{nm}_hi': hi})
            for nm, stat in [('mediated_retained', 'mediated'), ('alignment_ratio_r_pg', 'r_pg'),
                             ('target_loading_ratio_r_zg', 'r_zg'), ('selT_retained', 'selT'),
                             ('selBOTH_retained', 'selBOTH'), ('selective_term_retained', 'selective_term')]:
                with np.errstate(invalid='ignore', divide='ignore'):
                    rep = g(stat, 'sum_c') / g(stat, 'sum_w')
                p, lo, hi = point_ci(rep)
                den = point_ci(g(stat, 'sum_w') / g(stat, 'n'))
                row.update({nm: p, f'{nm}_lo': lo, f'{nm}_hi': hi,
                            f'{nm}_denominator_ci_includes_0': bool(den[1] <= 0 <= den[2])})
            for nm, stat in [('mediated', 'mediated'), ('selective_term', 'selective_term'), ('selT', 'selT'),
                             ('selBOTH', 'selBOTH'), ('r_pg', 'r_pg'), ('r_zg', 'r_zg')]:
                nn = g(stat, 'n')
                for side, kind in [('within', 'sum_w'), ('cross', 'sum_c')]:
                    row[f'{nm}_mean_{side}'] = float(g(stat, kind)[0] / nn[0])
                p, lo, hi = point_ci((g(stat, 'sum_c') - g(stat, 'sum_w')) / nn)
                row.update({f'd_{nm}': p, f'd_{nm}_lo': lo, f'd_{nm}_hi': hi})
            with np.errstate(invalid='ignore', divide='ignore'):
                share = (g('mediated', 'sum_w') - g('mediated', 'sum_c')) / (g('raw', 'sum_w') - g('raw', 'sum_c'))
            p, lo, hi = point_ci(share)
            clear = (row['raw_loss_lo'] > 0) or (row['raw_loss_hi'] < 0)
            row['raw_loss_clearly_nonzero'] = bool(clear and not pi.identity)
            row.update({'share_of_loss_through_fitness': p if row['raw_loss_clearly_nonzero'] else np.nan,
                        'share_of_loss_lo': lo if row['raw_loss_clearly_nonzero'] else np.nan,
                        'share_of_loss_hi': hi if row['raw_loss_clearly_nonzero'] else np.nan,
                        'share_of_loss_unrestricted': p})
            rows.append(row)
    return pd.DataFrame(rows)


def selective_metrics(info, R, MM):
    rows = []
    stat_of = {'raw': 'raw', 'selT': 'selT', 'selBOTH': 'selBOTH', 'adjusted': 'adjusted', 'lineage': 'lineage',
               'lcoT': 'lcoT', 'lcoBOTH': 'lcoBOTH', 'lcxT': 'lcxT', 'lcxBOTH': 'lcxBOTH',
               'spearman_raw': 'spearman_raw', 'spearman_selT': 'spearman_selT'}
    for _, pi in info.iterrows():
        for sub in ['all', 'active']:
            if (pi.pair, sub, 'raw', 'n') not in R:
                continue
            for metric, stat in stat_of.items():
                sw, sc, n = (R[(pi.pair, sub, stat, k)] for k in ['sum_w', 'sum_c', 'n'])
                row = {'pair': pi.pair, 'target': pi.target, 'cellset': pi.cellset, 'family': pi.family, 'subset': sub,
                       'metric': metric, 'model_drug_pairs': int(round(n[0]))}
                for nm, rep in [('mean_within', sw / n), ('mean_cross', sc / n), ('diff_cross_minus_within', (sc - sw) / n)]:
                    p, lo, hi = point_ci(rep)
                    row.update({nm: p, f'{nm}_lo': lo, f'{nm}_hi': hi})
                with np.errstate(invalid='ignore', divide='ignore'):
                    p, lo, hi = point_ci(sc / sw)
                row.update({'ratio_cross_over_within': p, 'ratio_lo': lo, 'ratio_hi': hi})
                ws, cs_ = METRIC_STATS[metric]
                for side, s in [('within', ws), ('cross', cs_)]:
                    p, lo, hi = point_ci(MM[(pi.pair, sub, s)])
                    row.update({f'median_of_model_medians_{side}': p, f'median_of_model_medians_{side}_lo': lo,
                                f'median_of_model_medians_{side}_hi': hi})
                rows.append(row)
    return pd.DataFrame(rows)


def leakage_floors(info, mmed):
    t = mmed[mmed.model.isin(list(AGNOSTIC) + ['ridge_sel_lm', 'ridge_raw_lm', 'EnsembleMF'])].copy()
    rows = []
    for metric, (ws, cs_) in METRIC_STATS.items():
        for side, s in [('within', ws), ('cross', cs_)]:
            x = t[t.stat == s].assign(metric=metric, side=side)
            rows.append(x)
    t = pd.concat(rows, ignore_index=True)
    t['agnostic_type'] = t.model.map(AGNOSTIC).fillna('not drug-agnostic (comparator)')
    t = t.merge(info[['pair', 'target', 'cellset', 'family', 'identity']], on='pair')
    return t[['pair', 'target', 'cellset', 'family', 'identity', 'subset', 'model', 'agnostic_type', 'metric', 'side',
              'drugs', 'median', 'median_lo', 'median_hi', 'mean', 'mean_lo', 'mean_hi']]


def tost(info, R):
    rows = []
    for _, pi in info[~info.identity].iterrows():
        for sub in ['all', 'active']:
            if (pi.pair, sub, 'raw', 'n') not in R:
                continue
            for metric in ['selBOTH', 'selT', 'lcoBOTH', 'lcxBOTH', 'selective_term']:
                sw, sc, n = (R[(pi.pair, sub, metric, k)] for k in ['sum_w', 'sum_c', 'n'])
                d = (sc - sw) / n
                p, lo90, hi90 = point_ci(d, 90)
                _, lo95, hi95 = point_ci(d, 95)
                b = d[1:][np.isfinite(d[1:])]
                row = {'pair': pi.pair, 'target': pi.target, 'cellset': pi.cellset, 'family': pi.family, 'subset': sub,
                       'metric': metric, 'mean_within': float(sw[0] / n[0]), 'mean_cross': float(sc[0] / n[0]),
                       'diff_cross_minus_within': p, 'ci90_lo': lo90, 'ci90_hi': hi90, 'ci95_lo': lo95, 'ci95_hi': hi95}
                for mg in MARGINS:
                    p_lo, p_hi = float((b <= -mg).mean()), float((b >= mg).mean())
                    row[f'tost_p_margin_{mg:.2f}'] = max(p_lo, p_hi)
                    row[f'equivalent_margin_{mg:.2f}'] = bool(lo90 > -mg and hi90 < mg)
                    row[f'p_noninferior_margin_{mg:.2f}'] = p_lo
                rows.append(row)
    return pd.DataFrame(rows)


def oracle(info, R, mmed, per):
    rows = []
    pub = per[per['class'].isin(['classical', 'neural/factorization']) & (per.model != 'SVR')]
    for _, pi in info.iterrows():
        sub = 'all'
        if (pi.pair, sub, 'raw', 'n') not in R:
            continue
        sw, sc, n = (R[(pi.pair, sub, 'oracle_rzg2', k)] for k in ['sum_w', 'sum_c', 'n'])
        row = {'pair': pi.pair, 'target': pi.target, 'cellset': pi.cellset, 'family': pi.family, 'drugs': int(round(n[0]))}
        for nm, rep in [('oracle_share_within_mean_rzg2', sw / n), ('oracle_share_cross_mean_rzg2', sc / n)]:
            p, lo, hi = point_ci(rep)
            row.update({nm: p, f'{nm}_lo': lo, f'{nm}_hi': hi})
        with np.errstate(invalid='ignore', divide='ignore'):
            for side, k in [('within', 'sum_w'), ('cross', 'sum_c')]:
                p, lo, hi = point_ci(R[(pi.pair, sub, 'mediated', k)] / R[(pi.pair, sub, 'raw', k)])
                row.update({f'published_through_fitness_share_{side}': p, f'published_through_fitness_share_{side}_lo': lo,
                            f'published_through_fitness_share_{side}_hi': hi})
        # per-model through-fitness shares (point estimates) within the reference
        t = pub[pub.pair == pi.pair]
        sh = t.groupby('model').apply(lambda x: x.med_w.mean() / x.raw_w.mean() if x.raw_w.mean() > 0 else np.nan,
                                      include_groups=False)
        row['published_through_fitness_share_within_per_model_min'] = float(sh.min())
        row['published_through_fitness_share_within_per_model_max'] = float(sh.max())
        row['published_through_fitness_share_within_per_model_median'] = float(sh.median())
        ref = mmed[(mmed.pair == pi.pair) & (mmed.subset == sub) & (mmed.model == 'ref_CTRPv2_measured')].set_index('stat')
        for s in ['raw_c', 'selT_c', 'selB_c', 'adj_c', 'lin_c', 'raw_w', 'med_c']:
            if s in ref.index:
                row.update({f'one_replicate_{s}_median': ref.loc[s, 'median'], f'one_replicate_{s}_median_lo': ref.loc[s, 'median_lo'],
                            f'one_replicate_{s}_median_hi': ref.loc[s, 'median_hi'], f'one_replicate_{s}_mean': ref.loc[s, 'mean']})
        fit = mmed[(mmed.pair == pi.pair) & (mmed.subset == sub) & (mmed.model == 'ref_CTRPv2_fitness_ldo')].set_index('stat')
        for s in ['raw_w', 'raw_c', 'selT_c']:
            if s in fit.index:
                row.update({f'measured_fitness_ldo_{s}_median': fit.loc[s, 'median'],
                            f'measured_fitness_ldo_{s}_median_lo': fit.loc[s, 'median_lo'],
                            f'measured_fitness_ldo_{s}_median_hi': fit.loc[s, 'median_hi']})
        rows.append(row)
    return pd.DataFrame(rows)


def paired(info, per, boot, mmed):
    stat_cols = {'selB_c', 'adj_c', 'selT_c'}
    W = {k: g.set_index(['model', 'drug']) for k, g in per.groupby('pair')}
    # exactly drug-agnostic published model: no drug-specific component, so its model-axis-adjusted score is 0 by
    # construction; the paired difference is the ridge's own adjusted score (two-way CI of its drug median)
    extra = []
    for (pair, sub), g in mmed[(mmed.model == 'ridge_sel_lm') & (mmed.stat == 'adj_c')].groupby(['pair', 'subset']):
        r = g.iloc[0]
        for m in EXACT_PUBLISHED:
            if not ((boot.pair == pair) & (boot.subset == sub)).any():
                continue
            extra.append({'pair': pair, 'subset': sub, 'metric': 'adjusted', 'ours': 'ridge_sel_lm', 'ours_stat': 'adj_c',
                          'model': m, 'model_stat': 'adj_c (structurally 0)', 'drugs': r.drugs, 'median_diff': r['median'],
                          'median_lo': r.median_lo, 'median_hi': r.median_hi, 'p_boot_median': np.nan,
                          'mean_diff': r['mean'], 'mean_lo': r.mean_lo, 'mean_hi': r.mean_hi, 'p_boot_mean': np.nan})
    boot = pd.concat([boot, pd.DataFrame(extra)], ignore_index=True)
    ps, wil, fb = [], [], []
    for _, r in boot.iterrows():
        t = W[r.pair]
        a = t.loc[r.ours][r.ours_stat]
        if r.model_stat == 'adj_c (structurally 0)':
            b = t.loc[r.model]['raw_c'] * 0.0
        else:
            b = t.loc[r.model][r.model_stat]
        x = (a - b).dropna()
        if r.subset == 'active':
            x = x[x.index.isin(per.loc[per.active_ctrpv2, 'drug'].unique())]
        assert r.ours_stat in stat_cols and (r.model_stat in stat_cols or r.model_stat == 'adj_c (structurally 0)')
        wil.append(len(x))
        fb.append(float((x > 0).mean()) if len(x) else np.nan)
        ps.append(float(wilcoxon(x).pvalue) if len(x) >= 10 and np.any(x != 0) else np.nan)
    boot = boot.copy()
    boot['wilcoxon_n'] = wil
    boot['wilcoxon_p'] = ps
    boot['frac_better'] = boot.frac_better.fillna(pd.Series(fb, index=boot.index))
    boot = boot.merge(info[['pair', 'target', 'dataset', 'processing', 'endpoint', 'cellset', 'family', 'identity']], on='pair')
    prim = (boot.family == 'lnIC50-reference') & (boot.cellset == 'own') & (boot.subset == 'all') & boot.metric.isin(PRIMARY_FAMILY)
    boot['holm_family'] = np.where(prim, 'primary: all Task-4 tests (selective-BOTH, model-axis-adjusted, own-axis-removed raw ridge; lnIC50 reference, own cells, all drugs)', 'secondary (adjusted within its group)')
    boot['wilcoxon_p_holm'] = np.nan
    boot.loc[prim, 'wilcoxon_p_holm'] = holm(boot.loc[prim, 'wilcoxon_p'].values)
    for key, idx in boot[~prim].groupby(['family', 'cellset', 'subset', 'metric']).groups.items():
        boot.loc[idx, 'wilcoxon_p_holm'] = holm(boot.loc[idx, 'wilcoxon_p'].values)
    boot['holm_within_metric_primary'] = np.nan
    for m, idx in boot[prim].groupby('metric').groups.items():
        boot.loc[idx, 'holm_within_metric_primary'] = holm(boot.loc[idx, 'wilcoxon_p'].values)
    boot['ci_excludes_0'] = (boot.median_lo > 0) | (boot.median_hi < 0)
    return boot


def censoring_effects(info, per, R):
    cen = pd.read_csv(OUT / 'censoring_by_drug.csv.gz', dtype={'drug': str})
    rng = np.random.default_rng(SEED)
    rows = []
    for _, pi in info[(info.family == 'lnIC50-reference') & (info.cellset == 'own') & ~info.identity].iterrows():
        t = per[per.pair == pi.pair]
        ds = pi.dataset
        miss = cen[cen.dataset == ds].set_index('drug').frac_missing_lnIC50_cc
        ref = t[t.model == 'ref_CTRPv2_measured'].set_index('drug')
        x = pd.concat([miss.rename('miss'), ref.rzg_c.rename('rzg_c'), ref.rzg_w.rename('rzg_w')], axis=1, join='inner').dropna()
        rho, p = spearmanr(x.miss, x.rzg_c) if len(x) >= 10 else (np.nan, np.nan)
        pub = t[t['class'].isin(['classical', 'neural/factorization']) & (t.model != 'SVR')]
        pub = pub[pub.med_w.notna() & pub.med_c.notna()]
        sums = pub.groupby('drug')[['med_w', 'med_c']].sum()
        sums = sums[sums.index.isin(x.index)]
        hi_miss = (x.miss.reindex(sums.index) > x.miss.median()).values
        med = x.miss.median()
        Wb = np.vstack([np.ones(len(sums)), rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=1000)])
        bs = {}
        for h in (False, True):
            m_ = (hi_miss == h).astype(float)
            with np.errstate(invalid='ignore', divide='ignore'):
                bs[h] = (Wb * m_) @ sums.med_c.values / ((Wb * m_) @ sums.med_w.values)
        pt = {h: bs[h][0] for h in bs}
        bs = {h: v[1:] for h, v in bs.items()}
        row = {'pair': pi.pair, 'target': pi.target, 'dataset': ds, 'processing': pi.processing, 'endpoint': pi.endpoint,
               'drugs': len(x), 'median_frac_missing_lnIC50_cc_target': float(med),
               'spearman_missing_vs_r_zg_cross': float(rho), 'spearman_p': float(p),
               'spearman_missing_vs_r_zg_within': float(spearmanr(x.miss, x.rzg_w)[0]) if len(x) >= 10 else np.nan}
        for h, nm in [(False, 'low_missing'), (True, 'high_missing')]:
            b = np.array(bs[h], float)
            b = b[np.isfinite(b)]
            row.update({f'mediated_retained_{nm}': pt[h], f'mediated_retained_{nm}_lo': float(np.percentile(b, 2.5)),
                        f'mediated_retained_{nm}_hi': float(np.percentile(b, 97.5))})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(OUT), help='input/output directory (testing)')
    OUT = Path(ap.parse_args().dir)
    info = pair_meta()
    R, MM = load_reps()
    per = pd.read_csv(OUT / 'per_drug_scores.csv.gz', dtype={'drug': str})
    mmed = pd.read_csv(OUT / 'model_medians.csv')
    boot = pd.read_csv(OUT / 'paired_boot.csv')
    S = summary_by_target(info, R, MM)
    S.to_csv(OUT / 'summary_by_target.csv', index=False, float_format='%.6g')
    SM = selective_metrics(info, R, MM)
    SM.to_csv(OUT / 'selective_metrics_by_target.csv', index=False, float_format='%.6g')
    F = leakage_floors(info, mmed)
    F.to_csv(OUT / 'leakage_floors.csv', index=False, float_format='%.6g')
    T = tost(info, R)
    T.to_csv(OUT / 'tost_selectivity.csv', index=False, float_format='%.6g')
    O = oracle(info, R, mmed, per)
    O.to_csv(OUT / 'oracle_reference.csv', index=False, float_format='%.6g')
    C = paired(info, per, boot, mmed)
    C.to_csv(OUT / 'paired_comparisons.csv', index=False, float_format='%.6g')
    CE = censoring_effects(info, per, R)
    CE.to_csv(OUT / 'censoring_effects.csv', index=False, float_format='%.6g')
    prim = C[C.holm_family.str.startswith('primary')]
    write_json(OUT / 'summaries_run.json', {
        'pairs': int(len(info)), 'holm_primary_tests': int(len(prim)),
        'selBOTH_ridge_better_holm_0.05': int(((prim.metric == 'selBOTH') & (prim.wilcoxon_p_holm < 0.05) & (prim.median_diff > 0)).sum()),
        'selBOTH_tests': int((prim.metric == 'selBOTH').sum())})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    cols = ['pair', 'subset', 'drugs', 'raw_median_within', 'raw_median_cross', 'mediated_retained', 'mediated_retained_lo',
            'mediated_retained_hi', 'alignment_ratio_r_pg', 'target_loading_ratio_r_zg', 'd_selBOTH', 'd_selBOTH_lo',
            'd_selBOTH_hi', 'share_of_loss_through_fitness']
    print(S[S.subset == 'all'][cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
