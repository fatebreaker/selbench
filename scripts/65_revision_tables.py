"""Supplementary tables for the round-1 revision analyses (results/revision_*).

`tables(helpers)` returns a list of (key, LaTeX) pairs; 15_supplement_tables.py merges them with its own tables and
numbers all of them in order of first citation. `helpers` supplies table(), fmt(), pfmt() and esc() from script 15,
so the formatting is identical.
"""
import json

import numpy as np
import pandas as pd

from common import RESULTS

AX = RESULTS / 'revision_axis'
XL = RESULTS / 'revision_crosslab'
TR = RESULTS / 'revision_transfer'
PAIR_WITHIN = 'CTRPv2|CC|lnIC50->CTRPv2|CC|lnIC50'
PUBLISHED = ['AdaBoostDecisionTree', 'DIPK', 'DrugGNN', 'ElasticNet', 'EnsembleMF', 'GradientBoosting', 'KNNRegressor',
             'Lasso', 'LightGBM', 'MultiViewLightGBM', 'MultiViewNeuralNetwork', 'MultiViewRandomForest',
             'MultiViewXGBoost', 'PaccMann', 'PharmaFormer', 'Precily', 'RandomForest', 'SRMF', 'SVR',
             'SimpleNeuralNetwork', 'SparseGO', 'XGBoost']
NAIVE = ['NaiveMeanEffectsPredictor', 'NaiveTissueDrugMeanPredictor', 'NaiveTissueMeanPredictor']
OURS = {'fitness_only': 'axis only (omics-predicted)', 'ridge_raw_lm': 'kernel ridge, raw target (landmark genes)',
        'ridge_sel_lm': 'kernel ridge, selective target (landmark genes)',
        'ridge_raw_expr': 'kernel ridge, raw target (5,000 genes)',
        'ridge_sel_expr': 'kernel ridge, selective target (5,000 genes)',
        'ridge_raw': 'kernel ridge, raw target (multi-omics)', 'ridge_sel': 'kernel ridge, selective target (multi-omics)'}
AGNOSTIC = {'fitness_only', 'MultiViewRandomForest', 'NaiveTissueMeanPredictor', 'NaiveMeanEffectsPredictor'}
TARGET_NAME = {'CC': 'CurveCurator', 'Sanger': 'provider', 'CTRP': 'provider', 'CCLE': 'provider',
               'SP': 'CurveCurator, shared panel,', 'HR': 'CurveCurator, shared panel + range,',
               'SL': 'CurveCurator, shared panel,', 'RL': 'CurveCurator, shared panel + range,'}


def ci(lo, hi, d=2):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return '--'
    return f'{lo:.{d}f} to {hi:.{d}f}'.replace('-', '$-$').replace(' to $-$', ' to $-$')


def est(v, lo, hi, d=2):
    if not np.isfinite(v):
        return '--'
    s = f'{v:.{d}f}'.replace('-', '$-$')
    return s + f' ({ci(lo, hi, d)})' if np.isfinite(lo) else s


def target_label(t):
    ds, proc, ep = t.split('|')
    ep = {'lnIC50': 'ln IC$_{50}$', 'AUC': 'AUC', 'ActArea': 'ActArea'}[ep]
    return f'{ds} {TARGET_NAME[proc]} {ep}'


def parse_ci(s):
    a, b = [float(x) for x in str(s).strip('[]() ').split(',')]
    return a, b


def tables(H):
    table, fmt, pfmt, esc = H['table'], H['fmt'], H['pfmt'], H['esc']
    T = []

    # ------------------------------------------------------------ axis agreement by relationship
    pc = pd.read_csv(AX / 'pair_class_summary.csv')
    rows = []
    for view, vlab in [('harmonized endpoint, all-drug axes', 'all drugs'),
                       ('harmonized endpoint, active-drug axes', 'active drugs'),
                       ('harmonized endpoint, shared-drug axes', 'shared drugs')]:
        for r in pc[pc.view == view].itertuples():
            rows.append(['CurveCurator ' + ('AUC' if r.endpoint_family == 'AUC-type' else 'ln IC$_{50}$'), vlab,
                         esc(r.pair_class.split(': ')[1]), int(r.pairs), f'{r.median_r:.2f}',
                         f'{r.min_r:.2f}--{r.max_r:.2f}' if np.isfinite(r.min_r) else '--',
                         f'{r.median_disattenuated:.2f}'])
    wl = pd.read_csv(AX / 'within_lab_controls.csv')
    for ctl, ep, var, lab in [('GDSC1 Sanger site vs MGH site', 'AUC', 'all', 'GDSC1 sites, Sanger AUC'),
                              ('GDSC1 Sanger site vs MGH site', 'AUC', 'active', 'GDSC1 sites, Sanger AUC'),
                              ('GDSC1 Sanger site vs MGH site', 'AUC', 'dual_site_same_molecule', 'GDSC1 sites, Sanger AUC'),
                              ('GDSC1 Sanger site vs MGH site', 'lnIC50', 'all', 'GDSC1 sites, Sanger ln IC$_{50}$'),
                              ('PRISM HTS002 vs MTS010', 'drc_AUC', 'shared_all', 'PRISM runs, AUC'),
                              ('PRISM HTS002 vs MTS010', 'drc_AUC_capped', 'shared_all', 'PRISM runs, AUC capped at 1'),
                              ('PRISM HTS002 vs MTS010', 'mean_viability', 'shared_all', 'PRISM runs, mean viability')]:
        r = wl[(wl.control == ctl) & (wl.endpoint == ep) & (wl.variant == var)].iloc[0]
        vlab = {'all': 'all drugs', 'active': 'active drugs', 'dual_site_same_molecule': '14 drugs at both sites',
                'shared_all': '136 shared compounds'}[var]
        rows.append([lab, vlab, 'same screen, other run or site', 1, f'{r.r:.2f}', '--', f'{r.disattenuated:.2f}'])
    T.append(('AxisPairs', table(
        r'\textbf{Agreement of general-sensitivity axes by relationship between the two measurements.} Upper block: '
        r'DrEval CurveCurator refits of CTRP~v2, CTRP~v1, GDSC1, GDSC2 and CCLE (one fitter); median, range and '
        r'median disattenuated Pearson correlation over the screen pairs of each class, for axes built from all drugs, '
        r'active drugs ($\geq$10\% of lines with CurveCurator AUC $<$0.8) or the drugs shared by the pair. Lower block: '
        r'within-screen controls with identical processing (GDSC1 Sanger and MGH sites, Sanger fits; PRISM HTS002 and '
        r'MTS010 runs of the same compounds and cell lines). Disattenuation uses split-half reliability with respect to '
        r'drug sampling, which does not include run-to-run error.',
        'tab:axispairs', ['Data', 'Axis drugs', 'Relationship', 'Pairs', 'Median $r$', 'Range', 'Disatt.'], rows,
        'lllrrlr')))

    # ------------------------------------------------------------ variance components
    V = pd.read_csv(AX / 'variance_components.csv')
    rows = []
    comp_lab = {'cell': 'shared by all screens', 'lab': 'institution', 'screen': 'screen within institution',
                'residual': 'screen, run or site (residual)'}
    for m in V.model.unique():
        if not m.startswith('REML'):
            continue
        d = V[V.model == m]
        lab = m.replace('REML, ', '').split(':')[0]
        for r in d.itertuples():
            rows.append([esc(lab), comp_lab.get(r.component, esc(r.component)),
                         est(r.fraction, r.ci_lo, r.ci_hi), int(r.cells), int(r.observations)])
            lab = ''
    T.append(('Variance', table(
        r'\textbf{Nested variance decomposition of the general-sensitivity axis.} Restricted maximum likelihood fits of '
        r'the standardized axis (statsmodels MixedLM) with random effects for cell line, cell line $\times$ '
        r'institution and, where screens of one institution can be separated, cell line $\times$ screen. The residual is '
        r"reliable screen-, run- or site-specific variance plus a small sampling error. ``Shared cells'' uses the cell "
        r"lines measured in every screen; ``unbalanced'' uses all lines measured in $\geq$2 screens. Fractions of "
        r'cell-level variance with 95\% cell-bootstrap intervals.',
        'tab:variance', ['Model', 'Component', 'Fraction (95\\% CI)', 'Cells', 'Obs.'], rows, 'p{6.2cm}lllr')))

    # ------------------------------------------------------------ shared fraction: all, active, inactive drugs
    S = pd.read_csv(AX / 'triad_shared_fraction.csv')
    rows = []
    for key, lab in [('all-drug axes (paper)', 'all drugs'),
                     ('active-drug axes (discovery + gCSI active; NCI-60 all)', 'active drugs'),
                     ('inactive-drug discovery axes (gCSI, NCI-60 all)', 'inactive drugs'),
                     ('active-drug axes, NCI-60 excluded', 'active drugs, NCI-60 excluded')]:
        for r in S[S.axis_set == key].itertuples():
            lam = f'{r.lambda2:.2f} ({ci(r.ci_lo, r.ci_hi)})' if r.lambda2 > 0 else 'n.e. ($\\leq$0)'
            rows.append([lab, r.screen.replace('NCI60', 'NCI-60'), int(r.triads), lam, f'{r.split_half:.2f}'])
            lab = ''
    T.append(('SharedActive', table(
        r'\textbf{Share of each axis common to all screens, by the drugs that define it.} Triad identity '
        r'$\lambda_s^2=r_{st}r_{su}/r_{tu}$ over triads with $\geq$30 common lines and $|r_{tu}|\geq0.1$; median over '
        r'triads with 95\% cell-bootstrap intervals; non-positive estimates are reported as not estimable. Active: '
        r'$\geq$10\% of lines with AUC $<$0.8. Inactive compounds make up 43--60\% of each discovery library.',
        'tab:sharedactive', ['Axis drugs', 'Screen', 'Triads', '$\\lambda^2$ (95\\% CI)', 'Split-half'], rows,
        'llrlr')))

    # ------------------------------------------------------------ lineage and growth
    L = pd.read_csv(AX / 'lineage_by_screen.csv')
    rows = []
    for r in L.itertuples():
        rows.append([esc(r.axis.replace('|', ', ').replace('_lnIC50', ' ln IC50').replace('_GRAOC', ' GR')), int(r.cells),
                     int(r.lineages), fmt(r.lineage_r2), fmt(r.lineage_r2_adj), int(r.n_suspension),
                     fmt(r.susp_minus_rest_sd_units), fmt(r.cohens_d_susp_vs_adherent)])
    T.append(('Lineage', table(
        r'\textbf{Lineage structure of the axes.} Lineage $R^2$: share of axis variance explained by OncoTree lineage '
        r'(and its adjusted value). Suspension effect: mean axis value of suspension lines minus all other lines, in '
        r'units of the axis standard deviation; Cohen\'s $d$ against adherent lines only. Higher axis values mean more '
        r'sensitive.',
        'tab:lineage', ['Axis', 'Cells', 'Lineages', 'Lineage $R^2$', 'Adj.', 'Susp.', 'Susp.\\ (s.d.)', "Cohen's $d$"],
        rows, 'lrrrrrrr')))

    g = json.loads((AX / 'summary_48.json').read_text())['proliferation']
    rows = [[esc(p['proxy_a']), esc(p['proxy_b']), int(p['n']), fmt(p['spearman'])] for p in g['proxies']]
    pp = pd.read_csv(AX / 'pairs_partial_growth.csv')
    dlt = (pp.partial_both - pp.r)
    rows.append(['cross-screen axis $r$', 'after partialling both proxies', len(pp),
                 f'median change {dlt.median():.3f} (IQR {dlt.quantile(0.25):.3f} to {dlt.quantile(0.75):.3f})'.replace('-', '$-$')])
    for r in g['nci60_own_dt']:
        if r['axis'] == 'NCI60|all':
            rows.append(['NCI-60 axis', 'NCI-60 doubling time (own medium)', int(r['n']),
                         f"Spearman {r['spearman_vs_nci60_log_dt']:.2f}; $R^2$ {r['r2']:.2f}".replace('-', '$-$')])
    chk = g['info']['gr_formula_check']
    rows.append(['gCSI GR values', 'from relative count and nominal doubling time', int(chk['rows']),
                 f"{100 * chk['frac_within_0.005']:.1f}\\% within 0.005"])
    T.append(('Growth', table(
        r'\textbf{Proliferation and the axis.} Agreement of growth proxies (gCSI nominal doubling time, DepMap '
        r'CRISPR-inferred growth rate, NCI-60 doubling time measured in the screen\'s own medium); change in '
        r'cross-screen axis correlations after partialling out log doubling time and growth rate on the same cells; '
        r'association of the NCI-60 axis with its own doubling time; and a check that gCSI growth-rate (GR) values are '
        r'a deterministic function of relative cell count and the nominal doubling time. Measured time-zero plates were '
        r'not available for GDSC2 or GRAY in the released data.',
        'tab:growth', ['Quantity', 'Against', '$n$', 'Value'], rows, 'p{3.4cm}p{5.2cm}rp{5.2cm}')))

    # ------------------------------------------------------------ leaderboard metrics within CTRPv2
    M = pd.read_csv(XL / 'model_medians.csv')
    M = M[M.pair == PAIR_WITHIN]
    stats = ['raw_w', 'selT_w', 'lin_w', 'adj_w', 'spr_w', 'spsel_w']
    rows = []
    order = PUBLISHED + NAIVE + list(OURS)
    piv = M[M.subset == 'all'].set_index(['model', 'stat'])
    pivA = M[M.subset == 'active'].set_index(['model', 'stat'])
    for m in order:
        if (m, 'raw_w') not in piv.index:
            continue
        g_ = lambda s, P=piv: P.loc[(m, s)] if (m, s) in P.index else None
        r0 = g_('raw_w'); r1 = g_('selT_w')
        vals = [OURS.get(m, m) + (' $\\dagger$' if m in AGNOSTIC else ''), int(r0['drugs']),
                est(r0['median'], r0['median_lo'], r0['median_hi']), est(r1['median'], r1['median_lo'], r1['median_hi'])]
        for s in ['lin_w', 'adj_w', 'spr_w', 'spsel_w']:
            q = g_(s)
            vals.append(fmt(q['median']) if q is not None else '--')
        qa = g_('selT_w', pivA)
        vals.append(fmt(qa['median']) if qa is not None else '--')
        rows.append(vals)
    T.append(('LeaderboardMetrics', table(
        r'\textbf{DrEval leaderboard within CTRP~v2 under every per-drug metric.} Leave-cell-line-out predictions '
        r'scored against CTRP~v2 CurveCurator ln~IC$_{50}$ on the cell lines predicted by every model. Medians over '
        r'drugs (two-way 95\% bootstrap intervals for raw and selective). Selective, partial correlation given the '
        r'leave-drug-out axis; lineage, additionally given tissue indicators; model axis, additionally given the '
        r'model\'s own leave-drug-out axis (diagnostic; undefined for exactly drug-agnostic predictors); Spearman, '
        r'rank versions of raw and selective; active, selective over the 292 drugs active in CTRP~v2. $\dagger$, '
        r'drug-agnostic predictor (identical prediction for every drug, or identical up to a constant within a split).',
        'tab:leaderboardmetrics', ['Model', 'Drugs', 'Raw', 'Selective', 'Lineage', 'Model axis', 'Spearman raw',
                                   'Spearman sel.', 'Sel., active'], rows, 'p{3.9cm}rllrrrrr')))

    # ------------------------------------------------------------ rankings and critical differences
    R = pd.read_csv(XL / 'rankings.csv')
    keys = [('raw_within', 'Raw'), ('selT_within', 'Selective'), ('dreval_normalized_common_drugs', 'DrEval norm.'),
            ('adjusted_within', 'Model axis'), ('selBOTH_crosslab', 'Sel.-both, other datasets')]
    P = R.pivot_table(index='model', columns='ranking', values='rank')
    P = P.sort_values('raw_within')
    RI = pd.read_csv(RESULTS / 'revision_round2' / 'rankings' / 'rank_intervals.csv')
    RI = RI.set_index(['metric', 'model'])
    ri_key = {'raw_within': 'raw', 'selT_within': 'selective', 'dreval_normalized_common_drugs': 'dreval_normalized'}

    def rk(m, k):
        if not np.isfinite(P.loc[m, k]):
            return '--'
        if k in ri_key and (ri_key[k], m) in RI.index:
            q = RI.loc[(ri_key[k], m)]
            return f'{int(q["rank"])} ({int(q.rank_lo)}--{int(q.rank_hi)})'
        return int(P.loc[m, k])
    rows = [[m] + [rk(m, k) for k, _ in keys] for m in P.index]
    cd = pd.read_csv(XL / 'critical_difference.csv').set_index('ranking')
    cc = pd.read_csv(XL / 'ranking_correlations.csv')
    rho = lambda a, b: cc[((cc.ranking_a == a) & (cc.ranking_b == b)) | ((cc.ranking_a == b) & (cc.ranking_b == a))].spearman_rho.iloc[0]
    cap = (r'\textbf{Rankings of the 22 leaderboard models.} Ranks (1 = best) on the drugs predicted by every model. '
           r'Raw and selective: median per-drug correlation and selective correlation within CTRP~v2; DrEval norm.: '
           r'DrEval\'s normalized Pearson on the common drugs; model axis: the diagnostic score; selective-both, other '
           r'datasets: mean rank over GDSC1, GDSC2 and CTRP~v1 (CurveCurator ln~IC$_{50}$). In parentheses: 95\% '
           r'two-way (cell line $\times$ drug) bootstrap rank intervals (364 common drugs, $B=1{,}000$); 79\%, 59\% and '
           r'90\% of model pairs are ordered with confidence on the raw, selective and DrEval metrics. Nemenyi critical '
           f"differences (mean ranks): {cd.loc['raw_within', 'nemenyi_cd_mean_rank']:.1f} for per-drug metrics within "
           f"CTRP~v2 (364 drugs), {cd.loc['selBOTH_GDSC1', 'nemenyi_cd_mean_rank']:.1f}--"
           f"{cd.loc['selBOTH_GDSC2', 'nemenyi_cd_mean_rank']:.1f} in the other datasets and "
           f"{cd.loc['dreval_normalized_common_drugs', 'nemenyi_cd_mean_rank']:.1f} for DrEval\'s metric (ten splits). "
           f"Spearman correlation of rankings: raw vs selective {rho('raw_within', 'selT_within'):.2f}; DrEval vs "
           f"selective {rho('dreval_normalized_reported', 'selT_within'):.2f}; DrEval vs raw "
           f"{rho('raw_within', 'dreval_normalized_reported'):.2f}.")
    T.append(('Rankings', table(cap, 'tab:rankings', ['Model'] + [k[1] for k in keys], rows, 'llllrr')))

    # ------------------------------------------------------------ processing-controlled cross-dataset targets
    Sm = pd.read_csv(XL / 'summary_by_target.csv')
    Sm = Sm[(Sm.subset == 'all') & (Sm.cellset == 'own') & ~Sm.identity]
    Tt = pd.read_csv(XL / 'tost_selectivity.csv')
    Tt = Tt[(Tt.subset == 'all') & (Tt.cellset == 'own') & (Tt.metric == 'selBOTH')]
    Lf = pd.read_csv(XL / 'leakage_floors.csv')
    floor_models = ['fitness_only', 'MultiViewRandomForest', 'NaiveTissueMeanPredictor', 'NaiveMeanEffectsPredictor']
    Lf = Lf[(Lf.subset == 'all') & (Lf.cellset == 'own') & (Lf.metric == 'selBOTH') & (Lf.side == 'cross')
            & Lf.model.isin(floor_models)].groupby('pair')['median'].max()
    HRD = XL / 'harmonized_range'
    if (HRD / 'summary_by_target.csv').exists():
        h = pd.read_csv(HRD / 'summary_by_target.csv')
        Sm = pd.concat([Sm, h[h.subset == 'all']], ignore_index=True)
        ht = pd.read_csv(HRD / 'tost_selectivity.csv')
        Tt = pd.concat([Tt, ht[(ht.subset == 'all') & (ht.metric == 'selBOTH')]], ignore_index=True)
        hf = pd.read_csv(HRD / 'leakage_floors.csv')
        Lf = pd.concat([Lf, hf[hf.model.isin(floor_models)].groupby('pair')['median'].max()])
    Sm = Sm.assign(_order=Sm.target.str.split('|').str[0].map({'CTRPv2': 0, 'CTRPv1': 1, 'GDSC2': 2, 'GDSC1': 3, 'CCLE': 4}))
    Sm = Sm.sort_values(['_order', 'target', 'family'])
    Tt = Tt.set_index('pair')
    rows = []
    for r in Sm.itertuples():
        t = Tt.loc[r.pair]
        ref = 'ln IC$_{50}$' if r.family == 'lnIC50-reference' else 'AUC'
        share = est(r.share_of_loss_through_fitness, r.share_of_loss_lo, r.share_of_loss_hi) \
            if (r.raw_loss_clearly_nonzero and r.raw_loss > 0) else '--'
        rows.append([target_label(r.target), ref, int(r.drugs),
                     f'{r.raw_median_within:.2f}$\\to${r.raw_median_cross:.2f}',
                     est(r.mediated_retained, r.mediated_retained_lo, r.mediated_retained_hi),
                     est(r.alignment_ratio_r_pg, r.alignment_ratio_r_pg_lo, r.alignment_ratio_r_pg_hi),
                     est(r.target_loading_ratio_r_zg, r.target_loading_ratio_r_zg_lo, r.target_loading_ratio_r_zg_hi),
                     share,
                     est(t.diff_cross_minus_within, t.ci90_lo, t.ci90_hi, 3) + (' *' if t['equivalent_margin_0.05'] else ''),
                     fmt(Lf.get(r.pair, np.nan))])
    T.append(('CrossTargets', table(
        r'\textbf{Identical predictions scored against each processing version of each dataset.} CTRP~v2-trained '
        r'leave-cell-line-out predictions of the 21 published leaderboard models with per-drug signal, scored against '
        r'the CTRP~v2 version processed as in the Reference column and against the target version, on identical drugs '
        r'and, per drug, identical cell lines. Raw: median over models of median per-drug correlation. Axis term, '
        r'alignment $r_{pg}$ and loading $r_{zg}$: pooled ratio target/CTRP~v2 (95\% two-way bootstrap intervals over '
        r'drugs and cell lines). Share: axis-term share of the loss in raw correlation, shown only where the loss '
        r'interval excludes 0. $\Delta$ selective-both: target minus CTRP~v2 (90\% interval; *, equivalent within '
        r'$\pm$0.05 by two one-sided tests). Floor: best median selective-both score of a drug-agnostic predictor in '
        r'the target. CTRP~v2 rows re-score the same measurements after reprocessing. Shared panel: both datasets\' axes '
        r'built from the drugs they share; + range: every CurveCurator curve integrated over the concentration range '
        r'tested in both datasets (ln~IC$_{50}$: values outside that range censored; CTRP~v2 reference processed '
        r'identically).',
        'tab:crosstargets', ['Target version', 'Reference', 'Drugs', 'Raw', 'Axis term', 'Alignment', 'Loading',
                             'Share', '$\\Delta$ sel.-both', 'Floor'], rows,
        'p{3.0cm}lrlp{1.6cm}p{1.6cm}p{1.6cm}p{1.6cm}p{2.0cm}r').replace('\\small', '\\scriptsize', 1)))

    # ------------------------------------------------------------ censoring of CurveCurator ln IC50
    C = pd.read_csv(XL / 'censoring.csv')
    rows = [[r.dataset, int(r.drug_cell_pairs), f'{100 * r.frac_missing_lnIC50_cc:.0f}\\%',
             f'{r.median_AUC_cc_lnIC50_missing:.2f}', f'{r.median_AUC_cc_lnIC50_present:.2f}'] for r in C.itertuples()]
    E = pd.read_csv(XL / 'censoring_effects.csv')
    rows2 = [[target_label(r.target), int(r.drugs), f'{100 * r.median_frac_missing_lnIC50_cc_target:.0f}\\%',
              fmt(r.spearman_missing_vs_r_zg_cross), est(r.mediated_retained_low_missing, r.mediated_retained_low_missing_lo,
                                                        r.mediated_retained_low_missing_hi),
              est(r.mediated_retained_high_missing, r.mediated_retained_high_missing_lo,
                  r.mediated_retained_high_missing_hi)] for r in E.itertuples()]
    T.append(('Censoring', table(
        r'\textbf{Censoring of CurveCurator ln~IC$_{50}$.} Upper rows: share of drug--cell pairs without an ln~IC$_{50}$ '
        r'(no 50\% crossing) and the median CurveCurator AUC of pairs without and with one. The lower block lists, per '
        r'target version, the median share of censored pairs per drug, the Spearman correlation across drugs between '
        r'censoring and the target loading $r_{zg}$, and the retained axis term for drugs with below- and above-median '
        r'censoring (95\% two-way intervals).',
        'tab:censoring', ['Dataset', 'Pairs', 'Censored', 'AUC (censored)', 'AUC (with IC$_{50}$)'], rows, 'lrrrr')
        .replace(r'\bottomrule', r'\midrule' + '\n' + r'Target & Drugs & Censored & $\rho$(cens., $r_{zg}$) & Retained, low / high \\' + '\n'
                 + '\n'.join(f'{a} & {b} & {c} & {d} & {e} / {f}' + r' \\' for a, b, c, d, e, f in rows2) + '\n' + r'\bottomrule')))

    # ------------------------------------------------------------ confirmation estimates
    Es = pd.read_csv(TR / 'confirmation_estimates.csv')
    hq = [('H1', 'largest disattenuated fitness-axis correlation'),
          ('H2', 'median selective / median raw cross-screen correlation'),
          ('H3', 'difference (reproducible - non-reproducible)'),
          ('H4', 'median own-minus-discovery fitness correlation (gain)'),
          ('H5a', 'median model selective correlation (ridge_sel_cons)'),
          ('H5a-floor', 'FLOOR: median selective correlation of drug-agnostic fit_measured'),
          ('H5b', 'median paired difference measurement - model')]
    rows = []
    for s in ['gCSI', 'NCI60', 'CCLE', 'FIMM', 'GRAY']:
        lab = s.replace('NCI60', 'NCI-60')
        for h, q in hq:
            r = Es[(Es.screen == s) & (Es.hypothesis == h) & (Es.quantity == q)]
            if not len(r):
                continue
            r = r.iloc[0]
            lo, hi = parse_ci(r.ci95_two_way_boot)
            disc = str(r.discovery_value)
            disc = disc if disc in ('nan', 'None') else (fmt(float(disc)) if not disc.startswith('[') else
                                                         '--'.join(f'{x:.2f}' for x in parse_ci(disc)))
            p = r.registered_p if np.isfinite(r.registered_p) else np.nan
            rows.append([lab, h.replace('-floor', ' floor'), est(r.estimate, lo, hi), disc.replace('nan', '--'),
                         pfmt(p) if s != 'NCI60' else ('n.i.' if np.isfinite(p) else '--')])
            lab = ''
    T.append(('Estimates', table(
        r'\textbf{Registered confirmations as estimates.} Each registered quantity with a 95\% interval from a two-way '
        r'bootstrap over drugs and cell lines (H1: the uncertainty of both split-half reliabilities is propagated), the '
        r'corresponding discovery value (range over discovery screens for H1 and H4) and the registered $P$ value. '
        r'H5a floor: the frozen drug-agnostic discovery axis scored exactly as H5a. NCI-60 $P$ values are not '
        r'interpretable (n.i.), because its 39,682 single-tested compounds on 58 lines are not independent units; only '
        r'its effect sizes are reported. GDSC2 rehearsal values: H1 0.13--0.32, H2 1.05, H3 0.31, H4 0.28, H5a 0.39, '
        r'floor 0.08, H5b 0.13.',
        'tab:estimates', ['Screen', 'Quantity', 'Estimate (95\\% CI)', 'Discovery', 'Registered $P$'], rows,
        'lllll')))

    # ------------------------------------------------------------ Holm over the 25 registered tests (round 2, script 86)
    Hm = pd.read_csv(RESULTS / 'revision_round2' / 'holm' / 'holm_25.csv')
    k_reg, k_cor, k_unit = ('(a/b) registered P, registered PDM run', '(a/b) registered P, corrected PDM run',
                            '(c) listed unit-level P, corrected PDM run')
    rows = []
    for r in Hm.itertuples(index=False):
        d = dict(zip(Hm.columns, r))
        unit = d[f'p_used | {k_unit}']
        rows.append([d['screen'].replace('NCI60', 'NCI-60').replace('_', '\\_').replace('PDM (organoids)', 'organoids'),
                     d['hypothesis'], int(d['n']) if np.isfinite(d['n']) else '--',
                     pfmt(d[f'p_used | {k_reg}']), pfmt(d[f'holm | {k_reg}']), pfmt(d[f'p_used | {k_cor}']),
                     pfmt(d[f'holm | {k_cor}']), pfmt(unit), pfmt(d[f'holm | {k_unit}'])])
    T.append(('Holm', table(
        r'\textbf{Multiplicity across the 25 registered tests with a $P$ value.} The 23 tests of the original freeze and '
        r'Amendments~1--2, and the organoid tests H10 and H11 that Amendment~3 registered for the same Holm family. '
        r'Registered run: the organoid tests as registered (orientation rule as frozen); corrected run: after deviation~4 '
        r'(H10 $P\leq1\times10^{-4}$ is the permutation floor). Unit level: $P$ values computed over independent units '
        r'(patients, tumours) or, for H3 and H5b, the tail of a two-way (drug $\times$ cell) bootstrap with $B=1{,}000$ '
        r'(a tail of 0 is entered as $1/B$). Surviving Holm at 0.05: 18 of 25 (registered run), 19 of 25 (corrected run), '
        r'14 of 25 at the unit level (12 with the registered run). NCI-60\'s four tests (39,682 compounds on 58 lines, not '
        r'independent units) survive in every variant; without them, 15 of 21 survive with the corrected run and 14 with '
        r'the registered run (10 and 8 at the unit level), and no other decision changes. In the most '
        r'conservative family (registered organoid run, unit-level $P$) none of the transfer tests H6, H8 and H10 '
        r'survives. The five bootstrap $P$ values near their thresholds (gCSI H3 and H5b, CCLE H5b, FIMM H3, GRAY H5b) were '
        r'recomputed with $B$ = 10,000 (0.014, 0.0074, 0.0077, 0.034 and 0.0074 against 0.010, 0.010, 0.015, 0.024 and '
        r'0.006 with $B$ = 1,000); no Holm decision changed in any family.',
        'tab:holm', ['Screen', 'Hyp.', '$n$', 'Reg.\\ $P$', 'Holm', 'Corr.\\ $P$', 'Holm', 'Unit $P$', 'Holm (unit)'],
        rows, 'llrllllll')))

    # ------------------------------------------------------------ patient-derived transfer at the level of units
    Pt = pd.read_csv(TR / 'pdx_tumour_level_tests.csv')
    rows = []
    for r in Pt.itertuples():
        rows.append(['PDX', esc(r.analysis.split(' (')[0]), r.grouping, f'{r.H8_median:.2f}', pfmt(r.H8_wilcoxon_p),
                     f'{r.H9_median_non_reproducible:.2f}'.replace('-', '$-$'), pfmt(r.H9_mwu_p)])
    Bt = pd.read_csv(TR / 'beataml_tests.csv')
    for r in Bt.itertuples():
        rows.append(['BeatAML', esc(r.analysis.split(' (')[0].split(':')[0]), '', f'{r.H6_median:.2f}',
                     pfmt(r.H6_wilcoxon_p), f'{r.H7_median_non_reproducible:.2f}'.replace('-', '$-$'), pfmt(r.H7_mwu_p)])
    Bs = pd.read_csv(TR / 'beataml_sensitivity_subsets.csv')
    for r in Bs.itertuples():
        rows.append(['BeatAML', esc(r.subset), f'{int(r.samples)} samples', f'{r.H6_median:.2f}', pfmt(r.H6_wilcoxon_p),
                     f'{r.H7_median_non_reproducible:.2f}'.replace('-', '$-$'), pfmt(r.H7_mwu_p)])
    T.append(('UnitLevel', table(
        r'\textbf{Patient-derived transfer at the level of independent units.} PDX\_Bruna: 37 models derive from 20 '
        r'tumours (16 with expression); G1 keeps STG139M separate from STG139, G2 merges them. T1, sensitivities and '
        r'predictions averaged within tumour, then processed as registered; T2, registered profiles averaged within '
        r'tumour; T3, one derivative per tumour (median over 1,000 draws). BeatAML: 520 samples from 479 patients. '
        r'Columns: median selective transfer of reproducible drugs and one-sided Wilcoxon $P$ (H6/H8); median of '
        r'non-reproducible drugs and one-sided Mann--Whitney $P$ for the difference (H7/H9).',
        'tab:unitlevel', ['Set', 'Analysis', 'Units', 'Repr.\\ median', '$P$ (H6/H8)', 'Non-repr.', '$P$ (H7/H9)'],
        rows, 'lp{6.6cm}llllll')))

    # ------------------------------------------------------------ random-effects estimates and power
    Pm = pd.read_csv(TR / 'pdx_meta_analysis.csv')
    Bm = pd.read_csv(TR / 'beataml_meta_analysis.csv')
    rows = []
    for r in Bm[Bm['class'].isin(['reproducible', 'non_reproducible'])].to_dict('records'):
        lo, hi = parse_ci(r['ci95_r'])
        rows.append(['BeatAML', r['analysis'], r['class'].replace('_', '-'), int(r['k']), est(r['pooled_r'], lo, hi),
                     fmt(r['tau']), f"{100 * r['I2']:.0f}\\%"])
    for r in Pm[Pm['class'].isin(['reproducible', 'non_reproducible']) & Pm.analysis.isin(
            ['registered model level', 'T1 tumour level'])].to_dict('records'):
        lo, hi = parse_ci(r['ci95_r_hksj'])
        rows.append(['PDX', f"{r['analysis']} ({r['grouping']})", r['class'].replace('_', '-'), int(r['k']),
                     est(r['pooled_r'], lo, hi), fmt(r['tau']), f"{100 * r['I2']:.0f}\\%"])
    pw = json.loads((TR / '55_summary.json').read_text())
    cap = (r'\textbf{Random-effects estimates of patient-derived transfer.} DerSimonian--Laird random-effects '
           r'meta-analysis over drugs on Fisher $z$ (intervals: BeatAML, Wald; PDX, Hartung--Knapp), with between-drug '
           r'$\tau$ and $I^2$. Power (simulation of the registered tests, $\tau$ = 0.15): tumour-level H8 '
           f"{pw['realised_tumour_level_power_H8_tau0.15']['0.1']:.2f} and "
           f"{pw['realised_tumour_level_power_H8_tau0.15']['0.15']:.2f} for true median transfer 0.10 and 0.15 (registered "
           f"design: 0.75 and 0.95); tumour-level H9 {pw['realised_tumour_level_power_H9_tau0.15']['0.1']:.2f} and "
           f"{pw['realised_tumour_level_power_H9_tau0.15']['0.15']:.2f}.")
    T.append(('RandomEffects', table(cap, 'tab:randomeffects', ['Set', 'Analysis', 'Class', 'Drugs', 'Pooled $r$ (95\\% CI)',
                                                                 '$\\tau$', '$I^2$'], rows, 'llllllr')))

    # ------------------------------------------------------------ curated classes within activity strata
    K = pd.read_csv(TR / 'class_summary_curated.csv')
    K = K[K.scheme == 'curated'].sort_values('rank')
    A = pd.read_csv(TR / 'activity_regression.csv')
    A = A[(A.scheme == 'curated') & (A.spec == 'logit_quadratic')].set_index('class')
    rows = []
    for r in K.to_dict('records'):
        a = A.loc[r['class']] if r['class'] in A.index else None
        rows.append([esc(r['class']), int(r['drugs']), est(r['median_rho_sel'], r['median_ci95_lo'], r['median_ci95_hi']),
                     f"{100 * r['frac_reproducible']:.0f}\\%",
                     est(a.adjusted_mean, a.adj_ci95_lo, a.adj_ci95_hi) if a is not None else '--',
                     int(a.rank_adjusted) if a is not None and np.isfinite(a.rank_adjusted) else '--'])
    bins = pd.read_csv(TR / 'activity_bins.csv')
    cap = (r'\textbf{Curated mechanism classes and drug activity.} Each of the 353 multi-screen drugs was assigned by '
           r'manual review (102 reassignments; Supplementary Data). Median selective cross-screen agreement (95\% '
           r'drug-bootstrap interval), share reproducible, and class means adjusted for activity (logistic-quadratic '
           r'terms in the smallest active fraction over screens and the smallest AUC s.d.) with the adjusted rank. '
           r'Activity strata (smallest share of lines with AUC $<$0.8): '
           + '; '.join(f"{b.activity_bin.replace('%', chr(92) + '%')}: {int(b.drugs)} drugs, median {b.median_rho_sel:.2f}, "
                       f'{100 * b.frac_reproducible:.0f}\\% reproducible' for b in bins.itertuples()).replace('<', '$<$')
           .replace('>=', '$\\geq$') + '.')
    T.append(('ClassesCurated', table(cap, 'tab:classescurated', ['Class', 'Drugs', 'Median (95\\% CI)', 'Repr.',
                                                                   'Activity-adjusted mean', 'Adj.\\ rank'], rows,
                                      'lrllll')))
    T.append(('Specification', specification(H)))
    T += ceiling_tables(H)
    T += new_tables(H)
    T += retrain_tables(H)
    T += round2_tables(H)
    T += round4_tables(H)
    T += round5_tables(H)
    return T


CE = RESULTS / 'revision_ceiling'
PRED = {'ridge_sel': 'kernel ridge, selective target', 'ridge_raw': 'kernel ridge, raw target',
        'ridge_crispr': 'kernel ridge + CRISPR', 'lineage_raw': 'lineage only', 'target_prior': 'target dependency',
        'fitness_omics': 'omics-predicted axis', 'replicate': 'one more screen (profile)',
        'replicate_sel': 'one more screen (selective residual)'}


def ceiling_tables(H):
    table, fmt, esc = H['table'], H['fmt'], H['esc']
    out = []
    E = pd.read_csv(CE / '49_c2_estimands.csv')
    R = pd.read_csv(CE / '49_c2_model_replicate_ratio.csv')
    Hh = pd.read_csv(CE / '49_c2_hierarchical_summary.csv').set_index(['model', 'class'])
    rows = []
    for m, lab in PRED.items():
        for cls in ['all', 'reproducible']:
            g = lambda agg, kap: E[(E.model == m) & (E['class'] == cls) & (E.aggregation == agg) & (E.kappa == kap)]
            pr, rom, rmeans = g('perdrug_romeans_bc', 'none'), g('rom_terms', '>0.1'), g('romeans_terms', '>0.1')
            pdr = g('perdrug_rom', 'none')
            rr = R[(R.model == m) & (R['class'] == cls) & (R.aggregation == 'perdrug_romeans_bc') & (R.kappa == 'none')]
            hh = Hh.loc[(m, cls)] if (m, cls) in Hh.index else None
            rows.append([lab if cls == 'all' else '', cls, int(pr.n_drugs.iloc[0]),
                         est(pr.estimate.iloc[0], pr.ci_lo.iloc[0], pr.ci_hi.iloc[0]),
                         est(rr.ratio.iloc[0], rr.ci_lo.iloc[0], rr.ci_hi.iloc[0]) if len(rr) and not m.startswith('replicate') else '--',
                         fmt(rom.estimate.iloc[0]), fmt(rmeans.estimate.iloc[0]), fmt(pdr.estimate.iloc[0]),
                         fmt(hh.pop_median_c2) if hh is not None else '--'])
    sens = E[(E.model == 'ridge_sel') & (E['class'] == 'all')]
    kap = ['none', '>0.0', '>0.05', '>0.1', '>0.2', '>0.3']
    srow = lambda agg: ', '.join(f'{sens[(sens.aggregation == agg) & (sens.kappa == k)].estimate.iloc[0]:.3f}' for k in kap)
    cap = (r'\textbf{The ceiling under every estimand.} $c^2$ is the squared correlation between a predictor and the '
           r'selectivity shared across screens, from triad identities over drugs in $\geq$3 discovery screens. Primary: '
           r'share of the panel\'s reproducible selective variance captured, pooled over drugs (per-drug means, ratio of '
           r'means, Pearson--Filon bias correction, no filter), with 95\% drug-cluster bootstrap intervals; '
           r'model/replicate: its ratio to one more screen\'s measured profile. Other aggregations: ratio of medians over '
           r'terms with $\rho_{E_1E_2}>0.1$ (original analysis), ratio of means over terms, per-drug ratio of medians and '
           r'the median drug of a hierarchical model (typical drug). Selective ridge, all drugs, as the filter on '
           r'$\rho_{E_1E_2}$ tightens (none, $>$0, $>$0.05, $>$0.1, $>$0.2, $>$0.3): ratio of medians '
           + srow('rom_terms') + '; primary ' + srow('perdrug_romeans_bc') + '.')
    out.append(('Ceiling', table(cap, 'tab:ceiling', ['Predictor', 'Drugs', '$n$', 'Primary $c^2$ (95\\% CI)',
                                                      'Model/replicate', 'Ratio of medians', 'Ratio of means',
                                                      'Per-drug RoM', 'Typical drug'], rows,
                                 'p{3.0cm}lrllrrrr')))

    X = pd.read_csv(CE / '50_crosslab_c2.csv')
    Xr = pd.read_csv(CE / '50_crosslab_ratio.csv')
    rows = []
    vlab = {'crosslab': 'three laboratories', 'crosslab_readout': 'three laboratories and readouts',
            'crosslab_no_NCI60': 'three laboratories, NCI-60 excluded', 'crosslab_n_ge_100': '$\\geq$100 common lines'}
    for v, lab in vlab.items():
        x = X[(X.model == 'ridge_sel') & (X.variant == v)].iloc[0]
        r = Xr[(Xr.model == 'ridge_sel') & (Xr.variant == v)]
        rows.append([lab, int(x.n_drugs), est(x.c2_primary, x.ci_lo, x.ci_hi),
                     est(r.ratio_crosslab.iloc[0], r.ci_lo.iloc[0], r.ci_hi.iloc[0]) if len(r) else '--',
                     est(x.discovery_same_drugs_primary, x.disc_ci_lo, x.disc_ci_hi),
                     est(r.ratio_discovery_same_drugs.iloc[0], r.disc_ci_lo.iloc[0], r.disc_ci_hi.iloc[0]) if len(r) else '--'])
    P = pd.read_csv(CE / '50_cfa_parameters.csv').set_index(['fit', 'parameter'])
    B = pd.read_csv(CE / '50_cfa_implied_discovery_bias.csv').set_index(['fit', 'quantity']).value
    cv = pd.read_csv(CE / '50_cfa_cv.csv').set_index('fit').cv_weighted_mse

    def par(f, q):
        if (f, q) not in P.index:
            return '--'
        r = P.loc[(f, q)]
        return est(r.estimate, r.ci_lo, r.ci_hi, 3)

    def bias(f):
        a_, b_ = B.get((f, 'ratio|all|implied_estimate')), B.get((f, 'ratio|all|general_truth'))
        return f'{100 * (a_ / b_ - 1):+.1f}\\%' if a_ is not None and b_ else '--'
    cap = (r'\textbf{Cross-laboratory ceiling and a laboratory- and readout-aware factor model.} Upper rows: '
           r'selective-ridge $c^2$ and model/replicate ratio (primary estimator; 95\% drug-cluster intervals) from triads '
           r'whose three screens come from three different laboratories (and readouts), including the confirmation screens, '
           r'next to the discovery estimate on the same drugs. Factor model (weighted least squares on 1,813 pairwise '
           r'selective correlations of 259 drugs in nine screens; 400 drug-cluster refits): same-laboratory components '
           r'for Broad ($\theta_\mathrm{Broad}$) and Sanger ($\theta_\mathrm{Sanger}$), a shared-readout component for '
           r'CellTiter-Glo ($\psi_\mathrm{CTG}$), with multiplicative or additive structure. Implied bias: discovery-only '
           r'model/replicate ratio relative to the general-factor truth. Five-fold drug cross-validation MSE: general '
           + f"{cv['M0_general']:.4f}, multiplicative {cv['M3_lab_ctg']:.4f}, additive {cv['M3_lab_ctg_additive']:.4f}.")
    t = table(cap, 'tab:ceilingfactor', ['Triads', 'Drugs', '$c^2$ (95\\% CI)', 'Model/replicate',
                                         'Discovery $c^2$, same drugs', 'Discovery ratio'], rows, 'p{4.0cm}rllll')
    extra = [r'\midrule', r'\multicolumn{6}{l}{\textit{Factor model}} \\',
             r'Component & & Multiplicative & Additive & & \\']
    for q, lab in [('theta_Broad', '$\\theta_\\mathrm{Broad}$'), ('theta_Sanger', '$\\theta_\\mathrm{Sanger}$'),
                   ('psi_CTG', '$\\psi_\\mathrm{CTG}$')]:
        extra.append(f"{lab} & & {par('M3_lab_ctg', q)} & {par('M3_lab_ctg_additive', q)} & & " + r'\\')
    extra.append(f"Implied bias of the ratio & & {bias('P3_lab_ctg_predictors')} & "
                 f"{bias('P3_lab_ctg_predictors_additive')} & & " + r'\\')
    t = t.replace(r'\bottomrule', '\n'.join(extra) + '\n' + r'\bottomrule')
    out.append(('CeilingFactor', t))

    Bp = pd.read_csv(CE / '52_biomarker_c2_perdrug.csv')
    Bp = Bp[Bp.primary_for_drug & Bp.estimable.astype(bool) & Bp.c2_biomarker.notna()].sort_values('c2_biomarker', ascending=False)
    rows = [[esc(r['drug_name']), esc(r['biomarker']), r['tier'], est(r['c2_biomarker'], r['c2_biomarker_ci_lo'], r['c2_biomarker_ci_hi']),
             est(r['c2_ridge_sel'], r['c2_ridge_ci_lo'], r['c2_ridge_ci_hi']) if np.isfinite(r['c2_ridge_sel']) else '--',
             fmt(r['c2_replicate'])] for r in Bp.to_dict('records')]
    pooled = pd.read_csv(CE / '52_biomarker_c2_pooled.csv').set_index('set')
    tr = pd.read_csv(CE / '52_biomarker_transfer_summary.csv')
    pc = pooled.loc['core, primary biomarker per drug']
    cap = (r'\textbf{Single canonical biomarkers against the cell-blind models.} 22 biomarkers for 86 compounds fixed '
           r'before scoring; each scored as a training-free, cell-blind predictor with the primary $c^2$ estimator '
           r'(95\% cell-bootstrap intervals). Shown: the pre-specified primary biomarker of each estimable drug. Pooled '
           f"over the {int(pc.n_drugs)} core drugs: biomarker {pc.c2_biomarker_pooled:.2f} ({pc.ci_lo:.2f}--{pc.ci_hi:.2f}), "
           f"ridge {pc.c2_ridge_pooled:.2f} ({pc.ridge_ci_lo:.2f}--{pc.ridge_ci_hi:.2f}), one more screen "
           f"{pc.c2_replicate_pooled_same_drugs:.2f}. Transfer (median selective correlation): BeatAML biomarker "
           f"{tr[(tr.cohort == 'BeatAML') & (tr.set == 'all listed pairs')].median_biomarker_sel.iloc[0]:.2f} versus frozen "
           f"model {tr[(tr.cohort == 'BeatAML') & (tr.set == 'all listed pairs')].median_frozen_model_sel_same_drugs.iloc[0]:.2f}; "
           f"PDX biomarker {tr[(tr.cohort == 'PDX_Bruna') & (tr.set == 'all listed pairs')].median_biomarker_sel.iloc[0]:.2f} "
           f"versus {tr[(tr.cohort == 'PDX_Bruna') & (tr.set == 'all listed pairs')].median_frozen_model_sel_same_drugs.iloc[0]:.2f} "
           r'(no paired difference excludes 0). Exploratory: intervals are not adjusted for multiplicity.')
    out.append(('Canonical', table(cap, 'tab:canonical', ['Drug', 'Biomarker', 'Tier', 'Biomarker $c^2$', 'Ridge $c^2$',
                                                          'One more screen'], rows, 'llllll')))

    S = pd.read_csv(CE / '51_simulation_bias.csv')
    S = S[(~S.crosslab) & (S.kappa.isin(['none', '>0.1']))]
    keep = [('perdrug_romeans_bc', 'none', 'primary'), ('rom_terms', '>0.1', 'ratio of medians')]
    rows = []
    for sc_ in S.scenario.unique():
        for pr in ['lablearn', 'nonlearn', 'raw', 'replicate']:
            for agg, kp, lab in keep:
                x = S[(S.scenario == sc_) & (S.predictor == pr) & (S.estimator == agg) & (S.kappa == kp)]
                if not len(x):
                    continue
                x = x.iloc[0]
                rows.append([esc(sc_), pr, lab, f'{x.estimate:.3f}', f'{x.truth_pooled:.3f}',
                             f'{100 * x.bias_vs_pooled / x.truth_pooled:+.1f}\\%' if x.truth_pooled else '--',
                             f'{x.truth_lfl:.3f}', f'{100 * x.bias_lfl / x.truth_lfl:+.1f}\\%' if x.truth_lfl else '--'])
    cap = (r'\textbf{Simulation of the ceiling estimators.} 150 drugs $\times$ 400 cell lines in four discovery screens '
           r'from two laboratories (plus single-screen laboratories for cross-laboratory triads), processed with the study '
           r'pipeline; 40 replicates per scenario. Scenarios: homogeneous $c^2$ (S0); heterogeneous $c^2$ proportional to '
           r'(S1) or independent of (S1b) reliability; same-laboratory error correlation 0.10 (S2) or 0.25 (S2b); 60 cell '
           r'lines (S3). Predictors: a model that learns the laboratory component (lablearn) or not (nonlearn), a '
           r'raw-trained model carrying half a noisy copy of the training axis, and the replicate. Truths: the pooled '
           r"share and the like-for-like truth of each estimator (``same estimator'' applied to the oracle). The primary "
           r'estimator tracks the pooled share; the ratio of medians tracks the median drug.')
    out.append(('SimulationCeiling', table(cap, 'tab:simceiling', ['Scenario', 'Predictor', 'Estimator', 'Estimate',
                                                                    'Pooled truth', 'Bias', 'Like-for-like truth', 'Bias'],
                                           rows, 'lllrrrrr')))
    return out


def specification(H):
    """How each headline exploratory number moves across the variants that were tried (R3 M7; editor item 6)."""
    table, fmt = H['table'], H['fmt']
    M = pd.read_csv(XL / 'model_medians.csv')
    pG1 = 'CTRPv2|CC|lnIC50->GDSC1|CC|lnIC50'

    def med(pair, model, stat, subset='all'):
        x = M[(M.pair == pair) & (M.model == model) & (M.stat == stat) & (M.subset == subset)]
        return est(x['median'].iloc[0], x['median_lo'].iloc[0], x['median_hi'].iloc[0]) if len(x) else '--'
    P = pd.read_csv(XL / 'per_drug_scores.csv.gz', usecols=['pair', 'model', 'raw_w', 'med_w', 'class', 'active_ctrpv2'],
                    low_memory=False)
    P = P[(P.pair == PAIR_WITHIN) & P['class'].isin(['classical', 'neural/factorization']) & (P.model != 'SVR')]
    rows = []

    def share(q):
        sh = q.groupby('model').apply(lambda d: d.med_w.sum() / d.raw_w.sum())
        return f'{sh.median():.2f} ({sh.min():.2f}--{sh.max():.2f})'
    o = pd.read_csv(XL / 'oracle_reference.csv').set_index('pair').loc[PAIR_WITHIN]
    rows += [['Axis share of published models (median, range)', 'all drugs (primary)', share(P)],
             ['', 'drugs active in CTRP~v2', share(P[P.active_ctrpv2])],
             ['', 'perfect predictor', est(o.oracle_share_within_mean_rzg2, o.oracle_share_within_mean_rzg2_lo,
                                           o.oracle_share_within_mean_rzg2_hi)]]
    for m, lab in [('EnsembleMF', 'Best published model (EnsembleMF), selective within CTRP~v2'),
                   ('ridge_sel_lm', 'Selective-target ridge, selective within CTRP~v2')]:
        rows += [[lab, 'all drugs (primary)', med(PAIR_WITHIN, m, 'selT_w')],
                 ['', 'active drugs', med(PAIR_WITHIN, m, 'selT_w', 'active')],
                 ['', 'lineage-adjusted', med(PAIR_WITHIN, m, 'lin_w')],
                 ['', 'Spearman', med(PAIR_WITHIN, m, 'spsel_w')]]
    rows += [['Selectivity in GDSC1 (CurveCurator ln IC$_{50}$)', 'EnsembleMF, selective-both (primary)', med(pG1, 'EnsembleMF', 'selB_c')],
             ['', 'EnsembleMF, model-axis diagnostic', med(pG1, 'EnsembleMF', 'adj_c')],
             ['', 'ridge, selective target, landmark genes', med(pG1, 'ridge_sel_lm', 'selB_c')],
             ['', 'ridge, selective target, 5,000 genes', med(pG1, 'ridge_sel_expr', 'selB_c')],
             ['', 'ridge, selective target, multi-omics', med(pG1, 'ridge_sel', 'selB_c')],
             ['', 'ridge, raw target, landmark genes, model axis', med(pG1, 'ridge_raw_lm', 'adj_c')]]
    R = pd.read_csv(RESULTS / 'dreval' / 'normalized_metric.csv').set_index('model').normalized_pearson
    rows += [['Axis-only predictor on DrEval\'s normalized Pearson', 'full ln IC$_{50}$ scale', fmt(R['fitness_only'])],
             ['', 'increment over the naive prediction', fmt(R['fitness_increment'])],
             ['', 'increment centred within tissue', fmt(R['fitness_increment_tissue_centred'])]]
    S = pd.read_csv(XL / 'summary_by_target.csv')
    S = S[(S.subset == 'all') & (S.cellset == 'own')].set_index('pair')
    for pair, lab in [('CTRPv2|CC|lnIC50->GDSC1|CC|lnIC50', 'CurveCurator ln IC$_{50}$ (primary)'),
                      ('CTRPv2|CC|lnIC50->GDSC1|CC|AUC', 'CurveCurator AUC, ln IC$_{50}$ reference'),
                      ('CTRPv2|CC|AUC->GDSC1|CC|AUC', 'CurveCurator AUC, AUC reference'),
                      ('CTRPv2|CC|lnIC50->GDSC1|Sanger|lnIC50', 'provider ln IC$_{50}$'),
                      ('CTRPv2|CC|lnIC50->GDSC1|Sanger|AUC', 'provider AUC')]:
        r = S.loc[pair]
        rows.append(['GDSC1 axis term retained (pooled)' if 'primary' in lab else '', lab,
                     est(r.mediated_retained, r.mediated_retained_lo, r.mediated_retained_hi)])
    lco = pd.read_csv(AX / 'leave_class_out_summary.csv').set_index('variant')
    rb = json.loads((RESULTS / 'robustness' / 'summary.json').read_text())['beyond_lineage']
    xl = json.loads((TR / '53_summary.json').read_text())['crosslab']
    rows += [['Share of multi-screen drugs reproducible', 'standard axis (primary)', f"{100 * lco.loc['standard', 'frac_reproducible_ge_0.3']:.0f}\\%"],
             ['', 'leave-class-out axis', f"{100 * lco.loc['leave_class_out', 'frac_reproducible_ge_0.3']:.0f}\\%"],
             ['', 'leave-target-out axis', f"{100 * lco.loc['leave_target_out', 'frac_reproducible_ge_0.3']:.0f}\\%"],
             ['', 'lineage means removed', f"{100 * rb['frac_reproducible_beyond_lineage']:.0f}\\%"],
             ['', 'cross-institution screen pairs only', f"{100 * xl['frac_reproducible_cross_lab_only']:.0f}\\%"]]
    Bt = pd.read_csv(TR / 'beataml_tests.csv')
    Bs = pd.read_csv(TR / 'beataml_sensitivity_subsets.csv')
    rows += [['BeatAML transfer, median over reproducible drugs', 'registered (samples)', fmt(Bt.H6_median.iloc[0])],
             ['', 'patients', fmt(Bt.H6_median.iloc[2])]]
    rows += [['', s_, fmt(v)] for s_, v in zip(Bs.subset, Bs.H6_median)]
    Pt = pd.read_csv(TR / 'pdx_tumour_level_tests.csv')
    for r in Pt.itertuples():
        lab = r.analysis.split(' (')[0].split(':')[0]
        rows.append(['PDX transfer, median over reproducible drugs' if r.Index == 0 else '',
                     esc_(lab) + ('' if r.grouping == 'model' else f' ({r.grouping})'), fmt(r.H8_median)])
    return table(
        r'\textbf{Specification table: headline exploratory numbers under the variants that were tried.} Medians with '
        r'95\% intervals (two-way bootstrap for the leaderboard metrics; drug bootstrap elsewhere). The primary variant '
        r'of each quantity is the one reported in the main text.',
        'tab:specification', ['Quantity', 'Variant', 'Value'], rows, 'p{5.6cm}p{6.4cm}l').replace('\\small', '\\scriptsize', 1)


def esc_(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


IM = RESULTS / 'revision_improve' / 'no_split_replicates'   # primary since round 2: replicate pairs split across folds removed
IM_NAME = {'CCLE': 'CCLE', 'CTRPv2': 'CTRP v2', 'GDSCv1': 'GDSC1', 'GDSCv2': 'GDSC2', 'gCSI': 'gCSI'}
IM_MODEL = {'deepcdr': 'DeepCDR', 'deepttc': 'DeepTTC', 'graphdrp': 'GraphDRP', 'hidra': 'HiDRA', 'lgbm': 'LightGBM',
            'tcnns': 'tCNNS', 'uno': 'UNO', 'floor_cell_mean': 'drug-agnostic floor (cell mean)'}


def new_tables(H):
    """Amendment 3 (patient-derived organoids) and the IMPROVE second benchmark."""
    table, fmt, esc = H['table'], H['fmt'], H['esc']
    out = []
    M = pd.read_csv(TR / 'pdm_meta_analysis.csv')
    T = pd.read_csv(TR / 'pdm_tests.csv')
    hj = json.loads((RESULTS / 'confirmation_PDM_corrected' / 'hypotheses.json').read_text())
    rows = []
    for r in M.to_dict('records'):
        lo, hi = parse_ci(r['re_ci95_hksj']) if isinstance(r['re_ci95_hksj'], str) else (np.nan, np.nan)
        rows.append(['registered rule' if r['analysis'].startswith('registered') else 'corrected (deviation 4)',
                     r['scope'], r['class'].replace('_', '-'), r['units'], r['compounds'],
                     est(r['pooled_r'], r['ci95_lo'], r['ci95_hi']), est(r['re_pooled_r'], lo, hi)])
    p = {(a.startswith('corrected'), h): v for a, h, v in zip(T.analysis, T.hypothesis, T.permutation_p)}
    ds = '; '.join(f"{d['dataset']} {d['patients']} patients, {d['drugs_retained']} drugs, axis reliability "
                   f"{d['split_half_reliability']:.2f}, perfect-predictor axis share {d['perfect_predictor_axis_share_mean_rzg2']:.2f}"
                   for d in hj['datasets'] if 'excluded' not in d)
    sec = hj['secondary']
    cap = (r'\textbf{Amendment 3: transfer to patient-derived organoids.} Frozen selective-trained predictions scored '
           r'against CoderData 2.4 organoid and micro-tissue screens at the patient level. Datasets meeting the registered '
           f'thresholds: {ds}. Bladder, pancreatic and MPNST were excluded (too few patients per drug or too few drugs). '
           r'Registered estimand: precision-weighted pooled Fisher $z$ (weights $n-4$), patient-bootstrap 95\% interval; '
           r'random effects over units (DerSimonian--Laird, Hartung--Knapp interval) for comparison with BeatAML and PDX. '
           f"Permutation $P$ (one-sided): registered rule H10 {p[(False, 'H10')]:.2f}, H11 {p[(False, 'H11')]:.2f}; "
           f"corrected H10 {p[(True, 'H10')]:.4f}, H11 {p[(True, 'H11')]:.3f}. Under the registered rule the "
           r'curve-fitted (liver) estimates are the exact negatives of the corrected ones (Supplementary Note 2, item 4). '
           f"Drug-agnostic prediction (corrected): mean raw $r$ {sec['agnostic']['mean_raw']:.2f} against "
           f"{sec['raw']['mean_raw']:.2f} for drug-specific raw-trained predictions (difference "
           f"{sec['raw_minus_agnostic_mean_raw']['estimate']:.2f}, 95\\% CI "
           f"{ci(*sec['raw_minus_agnostic_mean_raw']['ci95'])}).").replace('$-$0.00', '0.00')
    out.append(('PDM', table(cap, 'tab:pdm', ['Analysis', 'Datasets', 'Class', 'Units', 'Compounds',
                                              'Pooled $r$ (95\\% CI)', 'Random effects (95\\% CI)'], rows, 'lllrrll')))
    if (IM / 'within_summary.csv').exists():
        W = pd.read_csv(IM / 'within_summary.csv')
        B = pd.read_csv(IM / 'within_pooled_bootstrap.csv')
        rows = []
        for d in ['CTRPv2', 'GDSCv1', 'GDSCv2', 'CCLE', 'gCSI']:
            x = W[W.dataset == d]
            for m in list(IM_MODEL):
                r = x[x.model == m]
                if not len(r):
                    continue
                r = r.iloc[0]
                rows.append([IM_NAME[d], IM_MODEL[m], int(r.drugs), fmt(r.median_raw), fmt(r.median_selective),
                             fmt(r.axis_share), fmt(r.oracle_share)])
            b = B[B.dataset == d].set_index('stat')
            if 'fitness_share_source' in b.index:
                q = b.loc['fitness_share_source']
                rows.append([IM_NAME[d], 'seven models pooled', '', est(b.loc['median_raw_source', 'estimate'],
                             b.loc['median_raw_source', 'lo'], b.loc['median_raw_source', 'hi']),
                             est(b.loc['median_selective_source', 'estimate'], b.loc['median_selective_source', 'lo'],
                                 b.loc['median_selective_source', 'hi']), est(q.estimate, q.lo, q.hi), ''])
        cap = (r'\textbf{Second benchmark: the IMPROVE cross-study analysis, within each dataset.} Out-of-fold predictions '
               r'of seven models from ten random 80/10/10 splits of drug--cell pairs (each pair predicted by the split in '
               r'which it was a test pair), scored per drug against the leave-drug-out axis of that dataset. AUC over one '
               r'fixed range (10$^{-10}$--10$^{-4}$ M), sensitivity $=-$AUC. Axis share: $\sum$ mediated / $\sum$ raw over '
               r'drugs; perfect-predictor share: mean $r_{zg}^2$. Pooled rows: two-way (drug $\times$ cell) bootstrap 95\% '
               r'intervals, the same resamples applied to all seven models. Pairs whose replicate rows were test pairs of '
               r'different splits are removed (1.6\% of CTRP~v2, 18.6\% of GDSC1 and 26.8\% of GDSC2 pairs).')
        out.append(('ImproveWithin', table(cap, 'tab:improvewithin', ['Dataset', 'Model', 'Drugs', 'Median raw',
                                                                      'Median selective', 'Axis share',
                                                                      'Perfect-predictor share'], rows, 'llrllll')))
    if (IM / 'paired_pooled.csv').exists():
        Pp = pd.read_csv(IM / 'paired_pooled.csv')
        S = pd.read_csv(IM / 'selective_both_change.csv')
        S0 = pd.read_csv(IM.parent / 'selective_both_change.csv')   # before removing split-crossing replicate pairs
        rows = []
        for (src, trg, var), g in Pp.groupby(['source', 'target', 'axis_panel'], sort=False):
            g = g.set_index('stat')
            s = S[(S.source == src) & (S.target == trg) & (S.axis_panel == var)]
            sb = est(s.delta_selective_both_mean.iloc[0], s.ci90_lo.iloc[0], s.ci90_hi.iloc[0]) if len(s) else '--'
            s0 = S0[(S0.source == src) & (S0.target == trg) & (S0.axis_panel == var)]
            sb0 = fmt(s0.delta_selective_both_mean.iloc[0]).replace('-', '$-$') if len(s0) else '--'

            def e(k):
                return est(g.loc[k, 'estimate'], g.loc[k, 'lo'], g.loc[k, 'hi']) if k in g.index else '--'
            rows.append([IM_NAME[src], IM_NAME[trg], 'own' if var == 'own_panel' else 'shared', int(g.drugs.iloc[0]),
                         fmt(g.loc['median_raw_source', 'estimate']), fmt(g.loc['median_raw_target', 'estimate']),
                         e('delta_raw'), e('mediated_retained'), sb, sb0])
        cap = (r'\textbf{Second benchmark: identical predictions scored in the source and another dataset (IMPROVE).} '
               r'Each pair measured in both datasets is predicted once, by the split-$k$ model in which it was a source test '
               r'pair; the cross-dataset file holds the identical prediction (verified). Matched cell lines per drug '
               r'($\geq$30); axes built from the drugs of each dataset or from the drugs both share. Pooled over seven models; '
               r'two-way bootstrap 95\% intervals; change in selective-both: 90\% drug-bootstrap interval. Raw: median '
               r'per-drug correlation in the source and the target. Pairs whose replicate rows were test pairs of '
               r'different splits (so that a replicate was in training) are removed; the last column gives the change in '
               r'selective-both before their removal.')
        out.append(('ImproveCross', table(cap, 'tab:improvecross', ['Source', 'Target', 'Axis panel', 'Drugs',
                                                                    'Raw (source)', 'Raw (target)', '$\\Delta$ raw',
                                                                    'Axis term retained', '$\\Delta$ selective-both',
                                                                    'With replicates'],
                                          rows, 'lllrlllllr')))
    return out


# ================================================================== retraining in drevalpy (workstream 5)
RT = RESULTS / 'revision_retrain'
RT_SETTING = {'GDSC2_AUC': 'GDSC2 CurveCurator AUC (retrained)', 'CTRPv2_LNIC50': 'CTRP v2 CurveCurator ln IC$_{50}$',
              'GDSC2_LNIC50': 'GDSC2 CurveCurator ln IC$_{50}$ (retrained)', 'CTRPv2_AUC': 'CTRP v2 CurveCurator AUC (retrained)'}
RT_MODEL = {'EnsembleMF': 'EnsembleMF', 'EnsembleMFSelective': 'EnsembleMF, selective target',
            'GradientBoosting': 'GradientBoosting', 'GradientBoostingZ': 'GradientBoosting, rank-normal target',
            'GradientBoostingSelective': 'GradientBoosting, selective target', 'RandomForest': 'RandomForest',
            'SimpleNeuralNetwork': 'SimpleNeuralNetwork',
            'SimpleNeuralNetworkSelective': 'SimpleNeuralNetwork, selective target (did not learn)',
            'NaiveMeanEffectsPredictor': 'NaiveMeanEffectsPredictor', 'NaiveTissueDrugMeanPredictor':
            'NaiveTissueDrugMeanPredictor', 'FitnessOnlyRidge': 'axis only (omics-predicted)',
            'PerDrugRidgeRaw': 'per-drug kernel ridge, raw target', 'PerDrugRidgeSelective':
            'per-drug kernel ridge, selective target', 'ref:source_measurement': 'training dataset\'s own measurement'}
RT_LB_KEEP = ['EnsembleMF', 'RandomForest', 'GradientBoosting', 'SimpleNeuralNetwork', 'NaiveTissueDrugMeanPredictor',
              'NaiveMeanEffectsPredictor']
RT_TARGETS = {'GDSC2_AUC': ['CTRPv2', 'GDSC1', 'CTRPv1', 'CCLE'], 'CTRPv2_LNIC50': ['GDSC1', 'GDSC2', 'CTRPv1', 'CCLE'],
              'GDSC2_LNIC50': ['CTRPv2', 'GDSC1', 'CTRPv1', 'CCLE'], 'CTRPv2_AUC': ['GDSC1', 'GDSC2', 'CTRPv1', 'CCLE']}


def rt_name(m):
    if m.startswith('LB:'):
        return RT_MODEL.get(m[3:], m[3:]) + ' (released)'
    return RT_MODEL.get(m, m)


def zero(s):
    return s.replace('$-$0.00', '0.00')


def retrain_tables(H):
    """Within-dataset scores, paired cross-dataset scores and like-for-like target contrasts of the retraining."""
    table, fmt, esc = H['table'], H['fmt'], H['esc']
    out = []
    if not (RT / 'within_summary.csv').exists():
        return out
    W = pd.read_csv(RT / 'within_summary.csv')
    W = W[W.panel == 'main']
    N = pd.read_csv(RT / 'normalized_pearson.csv').set_index(['setting', 'model'])
    F = pd.read_csv(RT / 'dataset_fitness.csv')
    rows = []
    for st in ['GDSC2_AUC', 'CTRPv2_LNIC50']:
        x = W[W.setting == st]
        if st == 'CTRPv2_LNIC50':
            x = x[~x.model.str.startswith('LB:') | x.model.str[3:].isin(RT_LB_KEEP)]
        x = x.sort_values('median_raw', ascending=False)
        lead = RT_SETTING[st]
        for r in x.itertuples():
            npz = N.loc[(st, r.model)] if (st, r.model) in N.index else None
            np_s = (f'{npz.normalized_pearson:.2f} $\\pm$ {npz.split_sd:.2f}'
                    if npz is not None and np.isfinite(npz.normalized_pearson) else '--')
            share = (est(r.mediated_share, r.mediated_share_lo, r.mediated_share_hi)
                     if r.median_raw > 0.03 else '--')
            rows.append([lead, esc(rt_name(r.model)),
                         esc(r.training_target), int(r.drugs),
                         est(r.median_raw, r.median_raw_boot_lo, r.median_raw_boot_hi),
                         est(r.median_selective, r.median_selective_boot_lo, r.median_selective_boot_hi),
                         est(r.median_adjusted, r.median_adjusted_boot_lo, r.median_adjusted_boot_hi), share,
                         f'{r.oracle_mean_rzg2:.2f}', np_s])
            lead = ''
    lb = W[(W.setting == 'CTRPv2_LNIC50') & W.model.str.startswith('LB:') & ~W.model.str.contains('Naive')
           & (W.median_raw > 0.03)]
    fit = '; '.join(f"{r.dataset} {'ln IC$_{50}$' if r.endpoint == 'lnic50' else 'AUC'} {r.mean_fitness_r2:.2f}"
                    for r in F.sort_values('mean_fitness_r2', ascending=False).itertuples())
    cap = (r'\textbf{Retraining in drevalpy: scores within the training dataset.} Leave-cell-line-out, ten splits, '
           r'DrEval\'s features and hyperparameter grids; out-of-fold predictions on the cell lines and drugs common to '
           r'every model of the setting. Medians over drugs with two-way (cell line $\times$ drug) bootstrap 95\% '
           r'intervals; axis share $=\sum$ axis term$/\sum$ raw over drugs (not shown for models without per-drug '
           r'signal); perfect-predictor share $=$ mean $r_{zg}^2$; DrEval\'s normalized Pearson (mean $\pm$ s.d.\ over '
           r'splits). Released rows are DrEval\'s leaderboard predictions on the same splits; the '
           f'{len(lb)} released learned models with per-drug signal had axis shares of {lb.mediated_share.min():.2f}--'
           f'{lb.mediated_share.max():.2f} (median {lb.mediated_share.median():.2f}). '
           r'Perfect-predictor share of each DrEval dataset and endpoint (all drugs and cell lines): ' + fit + '.')
    out.append(('Retrain', table(cap, 'tab:retrain', ['Setting', 'Model', 'Target', 'Drugs', 'Median raw',
                                                      'Median selective', 'Model-axis-adjusted', 'Axis share',
                                                      'Perfect pred.', 'Normalized Pearson'], rows,
                                 'lllrllllll')))

    S = pd.read_csv(RT / 'paired_summary.csv')
    S = S[S.panel == 'main']
    Pp = pd.read_csv(RT / 'paired_pooled.csv')
    keep = ['ref:source_measurement', 'EnsembleMF', 'LB:EnsembleMF', 'RandomForest', 'LB:RandomForest',
            'PerDrugRidgeSelective', 'FitnessOnlyRidge']
    rows = []
    for st in ['GDSC2_AUC', 'GDSC2_LNIC50', 'CTRPv2_AUC', 'CTRPv2_LNIC50']:
        for t in RT_TARGETS[st]:
            p = Pp[(Pp.setting == st) & (Pp.target == t)]
            x = S[(S.setting == st) & (S.target == t)]
            ref = x[x.model == 'ref:source_measurement']
            m_ref = ref.mean_sel_BOTH_t.iloc[0] if len(ref) else np.nan
            lead = f'{RT_SETTING[st].split(" (")[0]} $\\rightarrow$ {t}'
            if len(p):
                p = p.iloc[0]
                rows.append([lead, f'{int(p.models)} learned models pooled', int(x.drugs.max()),
                             f'{p.mean_raw_s:.2f} $\\rightarrow$ {p.mean_raw_t:.2f}',
                             est(p.delta_mediated, p.delta_mediated_lo, p.delta_mediated_hi),
                             est(p.delta_selective_term, p.delta_selective_term_lo, p.delta_selective_term_hi),
                             zero(f'{p.mean_sel_BOTH_s:.2f} $\\rightarrow$ {p.mean_sel_BOTH_t:.2f}'),
                             est(p.delta_sel_BOTH, p.delta_sel_BOTH_lo, p.delta_sel_BOTH_hi),
                             f'{p.mean_sel_BOTH_t / m_ref:.2f}' if np.isfinite(m_ref) else '--'])
                lead = ''
            for m in keep:
                r = x[x.model == m]
                if not len(r):
                    continue
                r = r.iloc[0]
                is_ref = m.startswith('ref:')
                rows.append([lead, esc(rt_name(m)), int(r.drugs),
                             f'{r.mean_raw_s:.2f} $\\rightarrow$ {r.mean_raw_t:.2f}'.replace('-', '$-$'),
                             zero(f'{r.delta_mediated:.2f}'.replace('-', '$-$')),
                             zero(f'{r.delta_selective_term:.2f}'.replace('-', '$-$')),
                             zero(f'{r.mean_sel_BOTH_s:.2f} $\\rightarrow$ {r.mean_sel_BOTH_t:.2f}'.replace('-', '$-$')),
                             '--' if is_ref else est(r.delta_sel_BOTH, r.delta_sel_BOTH_lo, r.delta_sel_BOTH_hi),
                             '1' if is_ref else zero(f'{r.sel_BOTH_t_frac_of_measured_mean:.2f}'.replace('-', '$-$'))])
                lead = ''
    cap = (r'\textbf{Retraining in drevalpy: identical predictions scored in the training dataset and in another '
           r'dataset.} Out-of-fold predictions of each cell line scored against the training dataset and against the '
           r'other dataset\'s measurement of the same drugs, on matched cell lines per drug. Means over drugs; changes '
           r'are cross minus within, with two-way bootstrap 95\% intervals. Axis term $r_{pg}r_{zg}$; selective term '
           r'$=$ raw minus axis term. Selective-both: partial correlation given both datasets\' axes. The training '
           r'dataset\'s own measurement, rank-normalized and scored as a prediction on the same cells, gives the '
           r'agreement of the two datasets; the last column is each prediction\'s cross-dataset selective-both as a '
           r'fraction of it. Pooled rows: DrEval\'s four learned models retrained on GDSC2 AUC, GDSC2 ln~IC$_{50}$ or CTRP~v2 '
           r'AUC (each scored in other datasets on the same endpoint), or the 21 '
           r'released published models (CTRP v2 ln IC$_{50}$).')
    out.append(('RetrainCross', table(cap, 'tab:retraincross', ['Training $\\rightarrow$ target', 'Model', 'Drugs',
                                                                'Raw', '$\\Delta$ axis term', '$\\Delta$ selective term',
                                                                'Selective-both', '$\\Delta$ selective-both',
                                                                'Fraction of measurement'], rows, 'llrllllll')))

    C = pd.read_csv(RT / 'contrasts.csv')
    C = C[(C.panel == 'main') & (C.family == 'target effect within model class')
          & C.metric.isin(['selective (= selective-BOTH within)', 'selective-BOTH', 'model-axis-adjusted'])]
    C = C[~C.contrast.str.contains('global NN|raw z - raw response|DepMap')]
    rows = []
    for st in ['GDSC2_AUC', 'CTRPv2_LNIC50']:
        for con, g in C[C.setting == st].groupby('contrast', sort=False):
            for where, h in g.groupby('where', sort=False):
                cell = {}
                for r in h.itertuples():
                    k = 'adj' if r.metric == 'model-axis-adjusted' else 'sel'
                    star = '*' if r.holm_p_family_setting < 0.05 else ''
                    cell[k] = est(r.mean_diff, r.mean_diff_lo, r.mean_diff_hi, 3) + star
                rows.append([RT_SETTING[st].split(' (')[0], esc(con.replace('target: ', '')), esc(where),
                             int(h.drugs.iloc[0]), cell.get('sel', '--'), cell.get('adj', '--')])
    cap = (r'\textbf{Retraining in drevalpy: the training target within model class.} Mean per-drug difference, '
           r'selective-target model minus raw-target model of the same class, features, splits and tuning, with two-way '
           r'bootstrap 95\% intervals; * Holm-adjusted $P<0.05$ within family (setting, panel and metric). Selective: '
           r'partial correlation given the evaluation dataset\'s axis within the training dataset and given both '
           r'datasets\' axes across datasets. Model-axis-adjusted: also given the model\'s own leave-drug-out axis, which '
           r'removes axis leakage but also any selectivity a model shares across drugs. The selective-target '
           r'SimpleNeuralNetwork collapsed to the drug mean and is not shown.')
    out.append(('RetrainTarget', table(cap, 'tab:retraintarget', ['Setting', 'Contrast', 'Where', 'Drugs',
                                                                  'Selective', 'Model-axis-adjusted'], rows,
                                       'lllrll')))
    return out


# ================================================================== round-2 analyses (scripts 77-89, results/revision_round2)
R2 = RESULTS / 'revision_round2'
R2_SETTING = {'CTRPv2_lnIC50': 'CTRP v2 ln IC$_{50}$ (released)', 'CTRPv2_LNIC50': 'CTRP v2 ln IC$_{50}$ (retrained)',
              'GDSC2_AUC': 'GDSC2 AUC (retrained)', 'GDSC2_LNIC50': 'GDSC2 ln IC$_{50}$ (retrained)'}


R2_MODEL = {'fitness_only': 'omics-predicted axis', 'FitnessOnlyRidge': 'omics-predicted axis',
            'ridge_raw': 'per-drug ridge (raw)', 'ridge_sel': 'per-drug ridge (selective)',
            'ridge_raw_lm': 'per-drug ridge (raw, landmark genes)', 'ridge_sel_lm': 'per-drug ridge (selective, landmark genes)',
            'ridge_raw_expr': 'per-drug ridge (raw, expression)', 'ridge_sel_expr': 'per-drug ridge (selective, expression)',
            'PerDrugRidgeRaw': 'per-drug ridge (raw)', 'PerDrugRidgeSelective': 'per-drug ridge (selective)',
            'ref_CTRPv2_measured': 'CTRP v2 measurement', 'floor_cell_mean': 'cell-mean floor'}


def r2_pair(p):
    """'CTRPv2|CC|lnIC50->GDSC1|HR|AUC' -> readable source -> target."""
    def one(x):
        ds, proc, ep = (x.split('|') + ['', '', ''])[:3]
        proc = {'CC': 'CurveCurator', 'HR': 'common range', 'SP': 'shared panel', 'SL': 'shared panel',
                'RL': 'panel + range'}.get(proc.split('-')[0], proc)
        ep = {'lnIC50': 'ln IC$_{50}$', 'AUC': 'AUC'}.get(ep.split('@')[0], ep)
        return f'{ds.replace("v", " v") if ds.startswith("CTRP") else ds} {proc} {ep}'.replace('  ', ' ')
    a, b = str(p).replace('_', ' ').split('->')
    tail = ' (same 58 drugs)' if '@RLdrugs' in b else ''
    return f'{one(a)} $\\rightarrow$ {one(b)}{tail}'


def round2_tables(H):
    table, fmt = H['table'], H['fmt']
    out = []
    e = lambda r, k, d=2: est(r[k], r[k + '_lo'], r[k + '_hi'], d)

    # ------------------------------------------------------------ capture (78)
    C = pd.read_csv(R2 / 'capture' / 'capture_by_model.csv')
    Pc = pd.read_csv(R2 / 'capture' / 'capture_pooled.csv')
    rows = []
    for (bm, st), g in C[C.benchmark != 'improve'].groupby(['benchmark', 'setting'], sort=False):
        for r in g.sort_values('axis_capture', ascending=False).itertuples():
            r = r._asdict()
            tag = r'$^\dagger$' if isinstance(r['drug_agnostic'], str) else ''   # 'exact' or 'within split (additive)'
            rows.append([R2_SETTING.get(st, st), esc_(R2_MODEL.get(r['model'], r['model'])) + tag, e(r, 'axis_capture'),
                         e(r, 'selective_capture', 3),
                         fmt(r['alignment_ratio']), fmt(r['axis_share']), fmt(r['perfect_share']), fmt(r['share_ratio']),
                         fmt(r['cf_share_ratio'])])
    for r in Pc.itertuples():
        r = r._asdict()
        lab = R2_SETTING.get(r['setting'], r['setting'].replace('_AUC', ' AUC (IMPROVE)').replace('GDSCv', 'GDSC'))
        rows.append([lab, 'pooled: ' + esc_(r['group']), e(r, 'axis_capture'), e(r, 'selective_capture', 3),
                     e(r, 'alignment_ratio'), e(r, 'axis_share'), e(r, 'perfect_share'), e(r, 'share_ratio'),
                     fmt(r['cf_share_ratio'])])
    out.append(('Capture', table(
        r'\textbf{How much of a perfect predictor\'s axis and selective terms each model recovers.} Axis capture: '
        r'$\sum_d r_{pg}r_{zg}/\sum_d r_{zg}^2$; selective capture: $\sum_d$ selective term $/\sum_d(1-r_{zg}^2)$, both 1 '
        r'for a perfect predictor (the measured profile). Alignment: mean $r_{pg}$ / mean $r_{zg}$. Axis share: $\sum$ '
        r'axis term / $\sum$ raw; perfect: mean $r_{zg}^2$; ratio: axis share / perfect share, bounded by 1/(mean '
        r'$r_{zg}^2$) and equal to $A/(Am+S(1-m))$; counterfactual: the ratio with $r_{pg}$ set to $r_{zg}$ per drug '
        r'and the model\'s own partial correlation kept. Pooled rows: two-way (cell line $\times$ drug) bootstrap 95\% '
        r'intervals ($B=1{,}000$, draws shared across models); model rows likewise. IMPROVE: random drug--cell-pair '
        r'splits (replicate-free), pooled per dataset. $\dagger$, drug-agnostic predictor.',
        'tab:capture', ['Setting', 'Model', 'Axis capture', 'Selective capture', 'Align.', 'Axis share', 'Perfect',
                        'Ratio', 'Counterf.'], rows, 'llllllllr')))

    # ------------------------------------------------------------ reliability-corrected reference (79, 78)
    Rf = pd.read_csv(R2 / 'reliability' / 'corrected_reference.csv')
    Cp = pd.read_csv(R2 / 'reliability' / 'corrected_capture_pooled.csv')
    rows = []
    for r in Rf[(Rf.kappa == 1.0)].itertuples():
        r = r._asdict()
        cp = Cp[(Cp.setting == r['setting']) & (Cp.reliability_source == r['reliability_source']) & (Cp.kappa == 1.0)]
        cp = cp.iloc[0].to_dict() if len(cp) else None
        lab = R2_SETTING.get(r['setting'], r['setting'].replace('_AUC', ' AUC (IMPROVE)').replace('GDSCv', 'GDSC'))
        rlab = (r['reliability_label'].replace('lower bound on R_sel; ', '')
                .replace(': a lower bound on the within-screen R_sel', ''))   # cross-screen agreement is not a bound
        rows.append([lab, esc_(rlab) + (' (primary)' if r['primary_source'] else ''),
                     e(r, 'R_sel'), fmt(r['raw_reference']), e(r, 'corrected_reference'),
                     e(cp, 'axis_capture_tau') if cp else '--', e(cp, 'selective_capture_tau', 3) if cp else '--',
                     e(cp, 'share_ratio_tau') if cp else '--'])
    out.append(('Reliability', table(
        r'\textbf{Perfect-predictor reference corrected for measurement error.} Error model: $z=\tau+e$ with $e$ '
        r'drug-specific noise independent of the prediction and the axis; the axis is treated as noise-free (split-half '
        r'0.98--0.99), so the profile\'s reliability is $\rho_d=r_{zg}^2+(1-r_{zg}^2)R_{\mathrm{sel}}$ and a predictor of '
        r'the reproducible profile $\tau$ has axis share $r_{zg}^2/\rho_d$. $R_{\mathrm{sel}}$: reliability of the '
        r'selective residual from the source named (several differ from the corrected endpoint in fit, range or screen). '
        r'Cross-screen agreement would bound $R_{\mathrm{sel}}$ from below if errors were independent, but within-screen '
        r'replicates agreed less (GDSC1 0.24 against 0.26--0.35 from screen triads, GDSC2 0.38 against 0.42--0.51) and 12--34\% of replicate noise is '
        r'shared across the drugs of a plate, so no source is a bound and the reference is reported as a range. Capture '
        r'relative to $\tau$ uses disattenuated terms; pooled over the models of Supplementary Table~\stCapture. Treating '
        r'a fraction of the axis as non-reproducible ($\kappa$ = 0.8, 0.55) shrinks every correction (Methods). 95\% '
        r'intervals propagate the uncertainty of $R_{\mathrm{sel}}$.',
        'tab:reliability', ['Setting', 'Reliability source', '$R_{\\mathrm{sel}}$', 'Perfect (raw)', 'Perfect (corr.)',
                            'Axis capture', 'Sel. capture', 'Ratio'], rows, 'p{2.6cm}p{5.2cm}llllll')))

    # ------------------------------------------------------------ proportional attenuation (81)
    A = pd.read_csv(R2 / 'attenuation' / 'attenuation_by_target.csv')
    A = A[A.group.isin(['published 21', 'IMPROVE 7 models']) | A.group.astype(str).str.startswith('DrEval learned')]
    ADJ = RESULTS / 'revision_round4' / 'simulation' / 'realdata_attenuation_reliability_adjusted.csv'
    if ADJ.exists():                   # reliability-adjusted excess (100b): each axis term divided by its reliability
        J = pd.read_csv(ADJ)[['benchmark', 'pair', 'group', 'adj_excess', 'adj_excess_lo', 'adj_excess_hi']]
        A = A.merge(J, on=['benchmark', 'pair', 'group'], how='left')
    rows = []
    for r in A.itertuples():
        r = r._asdict()
        bm = {'leaderboard': 'DrEval', 'retrain': 'retrained', 'improve': 'IMPROVE'}.get(r['benchmark'], r['benchmark'])
        lo, hi = (r['adj_excess_lo'], r['adj_excess_hi']) if pd.notna(r.get('adj_excess', np.nan)) else (r['excess_lo'], r['excess_hi'])
        flag = 'axis' if (lo > 0 and r['loss_clear']) else ('selective' if (hi < 0 and r['loss_clear']) else '')
        adj = e(r, 'adj_excess') if pd.notna(r.get('adj_excess', np.nan)) else '--'
        rows.append([bm, r2_pair(r['pair']), e(r, 'raw_loss'), fmt(r['within_share']), e(r, 'share_of_loss'),
                     e(r, 'excess'), adj, e(r, 'retention_difference'), e(r, 'd_selBOTH_sym', 3), flag])
    out.append(('Attenuation', table(
        r'\textbf{Is a cross-dataset loss specific to the axis?} Identical out-of-fold predictions scored in the training '
        r'dataset and another dataset on matched cells, pooled over models (DrEval: the 21 published leaderboard models; '
        r'retrained: DrEval\'s four learned models retrained on GDSC2 AUC, GDSC2 ln~IC$_{50}$ or CTRP~v2 AUC, or as released '
        r'for CTRP~v2 ln~IC$_{50}$, each scored on the same endpoint; IMPROVE: seven models, replicate-free, own-panel '
        r'axes). Raw loss: training minus target mean per-drug correlation. Within: the axis term\'s share of the '
        r'training-side correlation. If the target only attenuated both terms by the same factor, the axis term\'s share '
        r'of the loss would equal it; excess: share of loss minus within share; retention difference: axis-term minus '
        r'selective-term retention. $\Delta$ selective-both: symmetric definition (Methods). Adjusted excess: the same '
        r'with each dataset\'s axis term divided by its split-half axis reliability (Methods; its validity in IMPROVE\'s '
        r'pair splits is doubtful). Rejects: the adjusted excess interval excludes 0 for a clear loss (axis: '
        r'disproportionate loss through the axis; selective: through selectivity). Two-way bootstrap 95\% intervals.',
        'tab:attenuation', ['Set', 'Training $\\rightarrow$ target', 'Raw loss', 'Within', 'Share of loss', 'Excess',
                            'Adjusted excess', 'Retention diff.', '$\\Delta$ sel.-both', 'Rejects'], rows,
        'lp{5.0cm}llllllll')))

    # ------------------------------------------------------------ equivalence on the retained fraction (83)
    Q = pd.read_csv(R2 / 'equivalence' / 'equivalence_34.csv')
    Q = Q[Q.selective_both == 'symmetric']
    rows = [[r2_pair(r.pair), f'{r.mean_within:.3f}', f'{r.mean_cross:.3f}',
             est(r.retained_fraction, r.retained_fraction_lo, r.retained_fraction_hi),
             'yes' if r.noninferior_f_08 else '', 'yes' if r.tost_equivalent_003 else '',
             'yes' if r.tost_equivalent_005 else '']
            for r in Q.rename(columns={'noninferior_f_0.8': 'noninferior_f_08', 'tost_equivalent_0.03': 'tost_equivalent_003',
                                       'tost_equivalent_0.05': 'tost_equivalent_005'}).itertuples()]
    out.append(('Equivalence', table(
        r'\textbf{Selectivity kept by identical CTRP~v2-trained predictions in 34 target versions.} Mean selective-both '
        r'(symmetric) over the 21 published models in CTRP~v2 (within) and in the target (cross); retained fraction: '
        r'cross/within with two-way bootstrap 95\% interval. Kept $\geq$80\%: one-sided 95\% bound above 0.8. TOST: '
        r'equivalence of cross and within within $\pm$0.03 or $\pm$0.05. Within-dataset selective-both is only '
        r'0.04--0.15, so a $\pm$0.05 margin can admit the loss of most of it.',
        'tab:equivalence', ['Training $\\rightarrow$ target', 'Within', 'Cross', 'Retained', 'Kept $\\geq$80\\%',
                            'TOST $\\pm$0.03', 'TOST $\\pm$0.05'], rows, 'p{6.4cm}llllll')))

    # ------------------------------------------------------------ the range step and CTRP v2's high-dose tail (85)
    S = pd.read_csv(R2 / 'harmonization' / 'training_side_steps.csv')
    rows = [['harmonization step', esc_(r.target) + ': ' + esc_(r.step), est(r.training_side_median_raw, r.training_side_lo,
             r.training_side_hi), est(r.target_side_median_raw, r.target_side_lo, r.target_side_hi),
             est(r.mediated_retained, r.mediated_retained_lo, r.mediated_retained_hi)] for r in S.itertuples()]
    Ax = pd.read_csv(R2 / 'harmonization' / 'range_axes_correlations.csv')
    rows += [['axis agreement', f'{esc_(r.target)}: {esc_(r.axis_a)} vs {esc_(r.axis_b)}', est(r.r, r.r_lo, r.r_hi), '', '']
             for r in Ax.itertuples()]
    Tr = pd.read_csv(R2 / 'harmonization' / 'range_tail_reproducibility.csv')
    rows += [['reproducibility', f'{esc_(r.axis_a)} vs {esc_(r.axis_b)} ({r.drugs} drugs)', est(r.r, r.r_lo, r.r_hi), '', '']
             for r in Tr.itertuples()]
    Ms = pd.read_csv(R2 / 'harmonization' / 'range_model_scores.csv')
    rows += [['model scores', f'{esc_(r.target_range_from)}: {esc_(r.version)}', est(r.median_raw, r.median_raw_lo, r.median_raw_hi),
              est(r.median_selective, r.median_selective_lo, r.median_selective_hi),
              est(r.perfect_share, r.perfect_share_lo, r.perfect_share_hi)] for r in Ms.itertuples()]
    out.append(('RangeTail', table(
        r'\textbf{The range step and CTRP~v2\'s high-dose tail.} Harmonization steps: median per-drug correlation of the '
        r'published models on the training (CTRP~v2) and target side, and the axis term retained. Axis agreement and '
        r'reproducibility: correlations over cell lines of axes built from AUC over the common concentration window, '
        r'over CTRP~v2\'s concentrations above the target\'s range (high tail) or over the full range, from DrEval\'s '
        r'fitted CurveCurator curves. Model scores: median raw and selective per-drug correlation of the published '
        r'models against each version, and the perfect-predictor axis share (last column). Columns 3--5 change meaning '
        r'by block as described. Two-way bootstrap 95\% intervals.',
        'tab:rangetail', ['Block', 'Quantity', 'Value / training side', 'Target side / selective', 'Axis retained / perfect'],
        rows, 'lp{6.6cm}lll')))

    # ------------------------------------------------------------ expected vs observed retention (84)
    X = pd.read_csv(R2 / 'source_reference' / 'expected_retention.csv')
    rows = [[{'leaderboard': 'DrEval', 'retrain': 'retrained', 'improve': 'IMPROVE'}.get(r.benchmark, r.benchmark),
             r2_pair(r.pair), esc_(r.group), est(r.R_source, r.R_source_lo, r.R_source_hi),
             est(r.R_target, r.R_target_lo, r.R_target_hi), est(r.expected_retention, r.expected_retention_lo, r.expected_retention_hi),
             est(r.observed_retention, r.observed_retention_lo, r.observed_retention_hi),
             est(r.observed_minus_expected, r.observed_minus_expected_lo, r.observed_minus_expected_hi)]
            for r in X.itertuples()]
    out.append(('Retention', table(
        r'\textbf{Selectivity kept across datasets against that expected for a predictor of shared selectivity.} $R$: the '
        r'reproducible share of each dataset\'s selective variance, from screen triads as in the ceiling analysis. A '
        r'predictor of only the selectivity that both datasets share would keep about $\sqrt{R_T/R_S}$ of its '
        r'selective-both; observed: cross/within selective-both (symmetric). Observed below expected means the models '
        r'carried selectivity specific to the training dataset. Two-way bootstrap 95\% intervals.',
        'tab:retention', ['Set', 'Training $\\rightarrow$ target', 'Models', '$R_S$', '$R_T$', 'Expected', 'Observed',
                          'Difference'], rows, 'lp{5.2cm}llllll')))
    # ------------------------------------------------------------ lineage-adjusted selectivity and the tissue baseline (WS6b)
    Lm = R2 / 'lineage' / 'lineage_by_model.csv'
    if Lm.exists():
        L = pd.read_csv(Lm)
        L = L[L.kind.isin(['learned', 'this study (ridge)', 'tissue-by-drug baseline'])]
        Lp = pd.read_csv(R2 / 'lineage' / 'lineage_pooled.csv')
        rows = []
        for r in L.itertuples():
            r = r._asdict()
            lab = R2_SETTING.get(r['setting'], str(r['setting']).replace('_AUC', ' AUC (IMPROVE)').replace('GDSCv', 'GDSC'))
            base = r['kind'] == 'tissue-by-drug baseline'
            rows.append([lab, esc_(R2_MODEL.get(r['model'], r['model'])).replace('LB:', '') + (' (baseline)' if base else ''),
                         e(r, 'mean_selective', 3), e(r, 'mean_selective_lin', 3),
                         '--' if base else e(r, 'selective_mean_minus_baseline', 3)])
        for r in Lp.itertuples():
            r = r._asdict()
            lab = R2_SETTING.get(r['setting'], str(r['setting']).replace('_AUC', ' AUC (IMPROVE)').replace('GDSCv', 'GDSC'))
            rows.append([lab, 'pooled: ' + esc_(r['group']), e(r, 'mean_selective', 3), e(r, 'mean_selective_lin', 3),
                         e(r, 'mean_selective_minus_baseline', 3)])
        out.append(('LineageScores', table(
            r'\textbf{Selectivity given tissue of origin, and the tissue-by-drug baseline.} Mean per-drug selective '
            r'correlation (given the evaluation dataset\'s axis) and the lineage-adjusted version (additionally given tissue '
            r'indicators of the evaluation cells; tissues with $\geq$5 cells), with the difference from the tissue-by-drug '
            r'mean baseline (DrEval\'s NaiveTissueDrugMeanPredictor, or a training-fold tissue $\times$ drug mean where it was '
            r'not released) scored identically. Two-way (cell line $\times$ drug) bootstrap 95\% intervals.',
            'tab:lineagescores', ['Setting', 'Model', 'Selective', 'Lineage-adjusted', 'Minus tissue baseline'], rows,
            'lllll')))
    # ------------------------------------------------------------ dataset x endpoint (97, WS6b)
    TB = R2 / 'two_by_two'
    if (TB / 'cells.csv').exists():
        Cc = pd.read_csv(TB / 'cells.csv')
        Ct = pd.read_csv(TB / 'contrasts.csv')
        Cm = pd.read_csv(TB / 'models.csv')
        rows = []
        for r in Cc.itertuples():
            r = r._asdict()
            rows.append([esc_(r['cell']).replace('ln IC50', 'ln IC$_{50}$'), 'four learned (pooled)', e(r, 'axis_share'),
                         f"{fmt(r['perfect_share'])} / {fmt(r['corrected_reference'])}", e(r, 'axis_capture'),
                         e(r, 'selective_capture', 3), f"{r['floor_median_raw']:.2f} / {r['best_learned_median_raw']:.2f}",
                         f"{r['tissue_baseline_median_selective']:.3f} / {r['learned_median_selective_max']:.3f}"])
        cell_models = {c: set(m.split(';')) for c, m in zip(Cc['cell'], Cc['models'])}
        for r in Cm[[m in cell_models.get(c, ()) for c, m in zip(Cm['cell'], Cm['model'])]].itertuples():
            r = r._asdict()
            mname = r['model'][3:] + ' (released)' if r['model'].startswith('LB:') else r['model']
            rows.append([esc_(r['cell']).replace('ln IC50', 'ln IC$_{50}$'), esc_(mname), e(r, 'axis_share'),
                         fmt(r['perfect_share']), e(r, 'axis_capture'), e(r, 'selective_capture', 3),
                         f"{r['median_raw']:.2f}", f"{r['median_selective']:.3f}"])
        for ds, ECf in [('GDSC2', 'summary.csv'), ('CTRP v2', 'summary_CTRPv2.csv')]:
            EC = R2 / 'endpoint_cells' / ECf
            if not EC.exists():
                continue
            for r in pd.read_csv(EC).itertuples():
                r = r._asdict()
                if not np.isfinite(r.get('axis_share', np.nan)):
                    rows.append([f'{ds} (cell selection)', esc_(r['version']), '--', fmt(r['perfect_share']), '--', '--', '', ''])
                    continue
                rows.append([f'{ds} (cell selection)', esc_(r['version']).replace('ln IC50', 'ln IC$_{50}$'), e(r, 'axis_share'),
                             e(r, 'perfect_share'), e(r, 'axis_capture'), e(r, 'selective_capture', 3), '', ''])
        for r in Ct.itertuples():
            r = r._asdict()
            rows.append(['contrast', esc_(r['contrast']).replace('ln IC50', 'ln IC$_{50}$'), e(r, 'axis_share'),
                         e(r, 'perfect_share'), e(r, 'axis_capture'), e(r, 'selective_capture', 3), '', ''])
        out.append(('TwoByTwo', table(
            r'\textbf{Dataset and endpoint.} DrEval\'s four learned models retrained in drevalpy on GDSC2 AUC and GDSC2 '
            r'ln~IC$_{50}$ (same cells, drugs, splits and tuning) and on CTRP~v2 AUC, and as released for CTRP~v2 ln~IC$_{50}$ '
            r'(the leaderboard). Axis share, '
            r'perfect-predictor share (raw / reliability-corrected), axis and selective capture; median per-drug raw $r$ of the '
            r'drug-agnostic floor / best learned model; median selective $r$ of the tissue-by-drug baseline / best learned model '
            r'(model rows: the model\'s own median raw and selective $r$). Contrasts: differences between cells with two-way '
            r'bootstrap 95\% intervals (independent bootstraps paired by draw). Cell selection: AUC-trained predictions '
            r'rescored on the pairs with a defined ln~IC$_{50}$, beside the ln~IC$_{50}$-trained models on the same pairs.',
            'tab:twobytwo', ['Cell', 'Models', 'Axis share', 'Perfect', 'Axis capture', 'Selective capture', 'Floor / best raw',
                             'Tissue / best sel.'], rows, 'llllllll')))

    # ------------------------------------------------------------ what the axis contains (93, WS6b)
    AC = R2 / 'axis_contents'
    if (AC / 'marker_associations.csv').exists():
        Ma = pd.read_csv(AC / 'marker_associations.csv')
        Mv = pd.read_csv(AC / 'marker_variance.csv').set_index('axis')
        markers = ['BCL2L1', 'NQO1', 'ABCB1', 'SLFN11', 'TP53_mut', 'growth_rate']
        rows = []
        for ax_, g in Ma.groupby('axis', sort=False):
            g = g.set_index('marker')
            v = Mv.loc[ax_] if ax_ in Mv.index else None
            rows.append([esc_(ax_.replace('|', ', ').replace('_lnIC50', ' ln IC50'))] +
                        [f"{g.loc[m, 'spearman_within_lineage']:+.2f}".replace('-', '$-$') if m in g.index else '--' for m in markers] +
                        [fmt(v['r2_markers']) if v is not None else '--', fmt(v['r2_markers_beyond_lineage']) if v is not None else '--'])
        Pa = pd.read_csv(AC / 'pair_agreement.csv')
        for r in Pa.itertuples():
            rows.append([f"{esc_(r.axis_a.replace('|', ', '))} vs {esc_(r.axis_b.replace('|', ', '))}", f'{r.r:.2f}',
                         f'{r.partial_markers:.2f}', f'{r.cov_share_markers:.2f}', '', '', '', '', ''])
        out.append(('AxisContents', table(
            r'\textbf{What the general-sensitivity axis contains.} Upper block: Spearman correlation of each discovery axis (all '
            r'drugs or active drugs) with six pan-sensitivity determinants fixed before the analysis, within lineage (rank '
            r'residuals on OncoTree lineage); adjusted $R^2$ of the six markers jointly and beyond lineage. Lower block: '
            r'agreement of two screens\' axes ($r$), the partial correlation given the six markers, and the share of their '
            r'covariance the markers explain (columns 2--4). Cell-line bootstrap intervals are in the source files.',
            'tab:axiscontents', ['Axis', 'BCL2L1', 'NQO1', 'ABCB1', 'SLFN11', 'TP53 mut.', 'Growth', '$R^2$', 'Beyond lineage'],
            rows, 'lllllllll')))
        rows = []
        for r in pd.read_csv(AC / 'axis_transfer.csv').itertuples():
            rows.append([esc_(r.cohort), esc_(r.unit), f'{r.samples:.0f}',
                         est(r.r_agnostic_vs_measured_axis, r.r_agnostic_vs_measured_axis_lo, r.r_agnostic_vs_measured_axis_hi),
                         est(r.mean_raw_r_agnostic_per_drug, r.mean_raw_r_agnostic_per_drug_lo, r.mean_raw_r_agnostic_per_drug_hi)
                         if np.isfinite(r.mean_raw_r_agnostic_per_drug) else '--'])
        for r in pd.read_csv(AC / 'beataml_flt3.csv').itertuples():
            rows.append(['BeatAML FLT3-ITD', esc_(r.axis), f'{r.samples:.0f}',
                         est(r.flt3_itd_effect_sd, r.flt3_itd_effect_sd_lo, r.flt3_itd_effect_sd_hi) + ' s.d.', ''])
        bt = json.loads((AC / 'summary.json').read_text())['beataml']
        out.append(('AxisTransfer', table(
            r'\textbf{Does the general-sensitivity axis transfer to patient-derived samples, and what drives it in AML?} '
            r'Correlation between the cell-line-predicted (drug-agnostic) axis and the axis measured in each cohort, and the '
            r'drug-agnostic prediction\'s mean per-drug raw correlation; cluster bootstrap 95\% intervals. Lower rows: '
            r'difference in the BeatAML axis between FLT3-ITD-positive and -negative samples (s.d. of the axis) for the axis from '
            r'all drugs, without FLT3-active and multi-kinase inhibitors (24 of 153), or from those drugs only. For the '
            f"{bt['excluded_with_frozen_prediction']} excluded drugs with frozen predictions, selective transfer given the "
            r'restricted axis instead of the registered one: median '
            f"{bt['median_selective_registered_axis']:.3f} to {bt['median_selective_restricted_axis']:.3f}; FLT3-targeting "
            f"drugs among them {bt['median_selective_flt3_registered']:.3f} to {bt['median_selective_flt3_restricted']:.3f}. "
            r'Removing FLT3 activity from the axis does not reduce their transfer.',
            'tab:axistransfer', ['Cohort', 'Unit or axis', 'Samples', 'Axis transfer / FLT3-ITD effect', 'Per-drug raw $r$'],
            rows, 'lp{5.4cm}lll')))

    # ------------------------------------------------------------ harmonization ladder, GDSC2-trained (90, WS6b)
    HG = R2 / 'harmonization_gdsc2' / 'ladder.csv'
    if HG.exists():
        L = pd.read_csv(HG)
        L = L[L.group.isin(['DrEval learned (4)', 'per-drug ridges (2)'])]
        rows = [[esc_(r.target), esc_(r.step_label), esc_(r.group), int(r.drugs_scored), est(r.raw_loss, r.raw_loss_lo, r.raw_loss_hi),
                 est(r.excess, r.excess_lo, r.excess_hi), est(r.d_selBOTH_sym, r.d_selBOTH_sym_lo, r.d_selBOTH_sym_hi, 3),
                 est(r.observed_minus_expected, r.observed_minus_expected_lo, r.observed_minus_expected_hi)]
                for r in L.itertuples()]
        out.append(('HarmonizationGDSC', table(
            r'\textbf{Harmonization of GDSC2-trained predictions.} Out-of-fold predictions of models retrained on GDSC2 '
            r'CurveCurator AUC, scored in GDSC2 and another dataset: unharmonized (all drugs, or the drugs kept by harmonization), '
            r'with axes from the shared drugs, and with every curve summarized over the concentration range tested in both. Raw '
            r'loss; excess of the axis term\'s share of the loss over its within-dataset share (proportional-attenuation test); '
            r'change in selective-both (symmetric); observed minus expected retention of selective-both for a predictor of '
            r'shared selectivity. Two-way bootstrap 95\% intervals.',
            'tab:harmonizationgdsc', ['Target', 'Step', 'Models', 'Drugs', 'Raw loss', 'Excess', '$\\Delta$ sel.-both',
                                      'Obs.\\ $-$ exp.'], rows, 'lp{3.6cm}lrllll')))
    # ------------------------------------------------------------ endpoint-matched vs switched pairings (99a, WS7)
    EM = R2 / 'endpoint_matched' / 'pair_classification.csv'
    if EM.exists():
        D = pd.read_csv(EM).sort_values(['endpoint', 'comparison', 'target'])
        rows = []
        for r in D.itertuples():
            r = r._asdict()
            flag = r['rejection_direction'] if isinstance(r['rejection_direction'], str) and r['rejects_proportional'] else ''
            rows.append([esc_(r['endpoint']), r2_pair(r['pair']), int(r['drugs']), e(r, 'raw_loss'), fmt(r['within_share']),
                         e(r, 'excess'), esc_(flag), e(r, 'retained_fraction'),
                         'yes' if r['keeps_80pct'] is True or str(r['keeps_80pct']) == 'True' else '',
                         'yes' if str(r['leakage_overlapping']) == 'True' else ''])
        out.append(('EndpointMatched', table(
            r'\textbf{Cross-dataset attribution with matched and switched endpoints.} Identical CTRP~v2-trained predictions of '
            r'the 21 published models, scored against a CTRP~v2 reference and a target version. Endpoint-matched: an AUC-type '
            r'target scored against a CTRP~v2 AUC-type reference, or ln~IC$_{50}$ against ln~IC$_{50}$; switched: an AUC-type '
            r'target against the ln~IC$_{50}$ reference (the leaderboard endpoint). Raw loss; the axis term\'s within-dataset '
            r'share; excess of its share of the loss over that share (proportional-attenuation test) and the direction when '
            r'the interval excludes 0 for a clear loss; retained fraction of symmetric selective-both, whether it keeps '
            r'$\geq$80\% (one-sided 95\% bound), and whether drug-agnostic predictors reach a similar level (leakage). Two-way '
            r'bootstrap 95\% intervals.',
            'tab:endpointmatched', ['Endpoints', 'Reference $\\rightarrow$ target', 'Drugs', 'Raw loss', 'Within', 'Excess',
                                    'Rejects', 'Retained', '$\\geq$80\\%', 'Leakage'], rows, 'lp{5.2cm}rllllllll')))
    # ------------------------------------------------------------ calibrating expected retention (99b/99c, WS7)
    RC = R2 / 'retention_calibration' / 'baselines_gdsc2.csv'
    if RC.exists():
        D = pd.read_csv(RC)
        D = D[D.step.isin(['a', 'c']) & D.group.isin(['DrEval learned (4)', 'per-drug ridges (2)', 'NaiveTissueDrugMeanPredictor'])]
        rows = [[esc_(r.target), esc_(r.step_label), esc_(R2_MODEL.get(r.group, r.group)).replace('NaiveTissueDrugMeanPredictor', 'tissue-by-drug mean'),
                 est(r.observed_retention, r.observed_retention_lo, r.observed_retention_hi),
                 est(r.expected_retention, r.expected_retention_lo, r.expected_retention_hi),
                 est(r.observed_minus_expected, r.observed_minus_expected_lo, r.observed_minus_expected_hi)] for r in D.itertuples()]
        S = pd.read_csv(R2 / 'retention_calibration' / 'simulation_summary.csv')
        rows += [['simulation ' + r.scenario, esc_(r.description), '', f'{r.observed_retention_mean:.2f}', f'{r.expected_retention_mean:.2f}',
                  f'{r.observed_minus_expected_mean:+.3f} (s.d. {r.observed_minus_expected_sd:.3f})'.replace('-', '$-$')]
                 for r in S.itertuples()]
        out.append(('RetentionCalibration', table(
            r'\textbf{Calibrating the expected retention of selectivity.} GDSC2 AUC-trained predictions scored in GDSC2 and '
            r'another dataset, unharmonized and with panel and range harmonized: observed retention of selective-both, the '
            r'retention $\sqrt{R_T/R_S}$ expected for a predictor of shared selectivity, and their difference, for DrEval\'s '
            r'four learned models, the per-drug ridges and the tissue-by-drug mean (no molecular data). Lower rows: simulations '
            r'(50 replicates each) of the benchmark under the triad model (A), with error shared within an institution (B), '
            r'with a predictor that carries one dataset\'s specific component (C), and with tissue-by-drug effects half of '
            r'which are dataset-specific (D). Two-way bootstrap 95\% intervals.',
            'tab:retentioncalibration', ['Target', 'Step', 'Models', 'Observed', 'Expected', 'Difference'], rows,
            'lp{4.6cm}llll')))
    return out


R4 = RESULTS / 'revision_round4'
R4_GROUP = {'leaderboard': 'leaderboard (21 released models)', 'retrain:GDSC2_AUC': 'retrained, GDSC2 AUC',
            'retrain:GDSC2_LNIC50': 'retrained, GDSC2 ln IC$_{50}$', 'retrain': 'retraining (both)',
            'improve': 'IMPROVE (7 models, pair splits)', 'overall': 'three benchmarks'}
R4_PRED = {'raw_median': 'raw score (median $r$)', 'dreval_norm': 'DrEval normalized', 'A': 'axis capture',
           'S': 'selective capture', 'sel_mean': 'mean selective', 'S_lin': 'selective capture given tissue'}
R4_SIM = {'agnostic': 'drug-agnostic', 'agnostic_weak': 'drug-agnostic, weak', 'axis_heavy': 'axis-heavy',
          'balanced': 'balanced', 'eq_agnostic': 'equal raw: drug-agnostic', 'eq_mixed': 'equal raw: mixed',
          'eq_selective': 'equal raw: selective only', 'selective': 'selective only', 'tissue_only': 'tissue only'}


def round4_tables(H):
    """Transfer prediction (100a/100c) and the known-answer simulation (100b/102c)."""
    table = H['table']
    out = []
    TP4 = R4 / 'transfer_prediction_r4'
    TP = TP4 if (TP4 / 'fig_b_pooled_correlations.csv').exists() else R4 / 'transfer_prediction'
    if (TP / 'fig_b_pooled_correlations.csv').exists():
        B = pd.read_csv(TP / 'fig_b_pooled_correlations.csv')
        B = B[(B.method == 'spearman') & (B.interval == 'full')]
        sets = [('all', 'primary units'), ('learned', 'learned only'), ('all_previous_units', 'round-4 units (sensitivity)')]
        rows = []
        for g in R4_GROUP:
            for ms, mlab in sets:
                x = B[(B.group == g) & (B.model_set == ms)]
                if not len(x):
                    continue
                for pr, plab in R4_PRED.items():
                    a_ = x[(x.predictor == pr) & (x.outcome == 'selB_c')]
                    b_ = x[(x.predictor == pr) & (x.outcome == 'raw_median_c')]
                    if not len(a_):
                        continue
                    a_, b_ = a_.iloc[0], (b_.iloc[0] if len(b_) else None)
                    bc = (f" [BC {a_.bc_lo:.2f} to {a_.bc_hi:.2f}]".replace('-', '$-$')
                          if 'bc_lo' in a_ and pd.notna(a_.bc_lo) else '')
                    rows.append([R4_GROUP[g], mlab, plab, int(a_.targets), est(a_.r_pooled, a_.lo, a_.hi) + bc,
                                 est(b_.r_pooled, b_.lo, b_.hi) if b_ is not None else '--'])
        units = ''
        if (TP / 'units_by_setting.csv').exists():
            U = pd.read_csv(TP / 'units_by_setting.csv')
            U = U[U.model_set == 'all']
            units = ' Primary units: ' + '; '.join(
                f"{st.replace('_', ' ')}: " + ', '.join(sorted(g.model.astype(str).unique()))
                for (bm, st), g in U[U.benchmark != 'improve'].groupby(['benchmark', 'setting'])) + \
                '; IMPROVE: the seven released models in each source dataset.'
        out.append(('TransferPrediction', table(
            r"\textbf{Which within-dataset statistic predicts the accuracy that identical predictions keep in another "
            r"dataset?} Spearman correlation across models between a statistic of each model's within-dataset evaluation "
            r"and the same predictions' accuracy in each other dataset (same endpoint, matched drugs and cells), pooled as "
            r"equal-weight Fisher-$z$ means over targets (and over benchmarks for the last block). Outcomes: mean "
            r"selective-both and median raw per-drug correlation in the target. 95\% percentile intervals from a bootstrap "
            r"that resamples models (as clusters, per model set), drugs and cell lines; bias-corrected (BC) intervals in "
            r"brackets. Primary comparison, stated before the analysis: selective capture against the raw correlation on "
            r"the leaderboard. Retraining has four to six units, too few for informative intervals. Learned only: "
            r"drug-agnostic models and naive baselines excluded; round-4 units: including this study's own models." + esc_(units)
            + r' Contrasts, per-target values and the decision analysis (top $k$ = 1, 3, 5) are in Supplementary Data.',
            'tab:transferprediction', ['Benchmark', 'Models', 'Within-dataset statistic', 'Targets',
                                       'Predicts selective-both', 'Predicts raw accuracy'], rows, 'llp{3.2cm}rp{4.2cm}l')))
    SD = R4 / 'simulation'
    KT = SD / 'known_answer_table.csv'
    if KT.exists():
        K = pd.read_csv(KT)
        notes = json.loads((SD / 'known_answer_notes.json').read_text()) if (SD / 'known_answer_notes.json').exists() else {}
        rows = []
        for (pan, pr), g in K.groupby(['panel', 'predictor_label'], sort=False):
            g = g.set_index('stat')
            if not {'A', 'S'} <= set(g.index):
                continue
            def cell(st):
                r = g.loc[st]
                return (f"{r.truth_panel:.3f} / {r.truth_latent:.3f} / {r.mean_estimate:.3f} / "
                        f"{100 * r.coverage_vs_panel:.0f}\\% / {100 * r.coverage_vs_latent:.0f}\\%")
            rows.append([esc_(pan.replace(' x ', r' $\times$ ')), esc_(pr), cell('A'), cell('S')])
        F = pd.read_csv(SD / 'fig_5_axis_reliability_leakage.csv')
        A7 = pd.read_csv(SD / 'fig_7_attenuation_selboth.csv')
        att = '; '.join(f"{r.scenario.replace('_', ' ')}: unadjusted {100 * r.reject_81_rate:.0f}\\%, adjusted "
                        f"{100 * r.adj_reject_81_rate:.0f}\\%" for r in A7.itertuples())
        ag = F[F.predictor == 'agnostic'].sort_values('D')
        leak = ', '.join(f'{int(r.D)} drugs ($\\rho$ = {r.axis_split_half:.2f}) {r.leakage:.3f}' for r in ag.itertuples()
                         if r.D in (6, 10, 30, 150))
        omit = '; '.join(str(o) for o in notes.get('omissions', []))
        out.append(('KnownAnswer', table(
            r"\textbf{Known-answer simulation of the estimators.} Screens simulated with a shared axis, tissue effects, "
            r"drug-specific signal and noise; predictors of known composition, scored with the study's code (300 "
            r"replicates; two-way bootstrap, $B$ = 500). Each cell: panel-level truth (the panel's own leave-drug-out axis) / "
            r"latent truth (the generating axis) / mean estimate / coverage of the 95\% interval against the panel-level "
            r"and the latent truth. Against the latent truth, selective capture of axis-aligned predictors is inflated by "
            r"the axis's measurement noise, so its intervals rarely cover it. Leakage of an unreliable axis into the "
            r"drug-agnostic predictor's selective capture: " + leak +
            r" (predicted $A\,m(1-\rho)/[(1-m)\rho]$). Rejection rate of the proportional-attenuation test (200 "
            r"replicates), without and with each dataset's axis term divided by its reliability: " + att + '.' +
            (r' What the generating model omits: ' + esc_(omit) + '.' if omit else ''),
            'tab:knownanswer', ['Panel', 'Predictor', 'Axis capture', 'Selective capture'], rows, 'p{2.4cm}p{3.2cm}ll')))
    return out


DEP5 = RESULTS / 'revision_round5' / 'crispr_dependency'
DEP5_NAME = {'ridge': 'ridge (per gene)', 'random_forest': 'random forest', 'mlp': 'MLP (global)', 'knn': 'k-NN',
             'pca_regression': 'PCA regression', 'omics_axis': 'omics-predicted axis (gene-agnostic)',
             'tissue_axis': 'tissue mean of the axis (gene-agnostic)', 'tissue_gene_mean': 'tissue-by-gene mean',
             'learned': 'five learned models (pooled)', 'gene_agnostic': 'two gene-agnostic predictors (pooled)'}
DEP5_CONFIG = {'primary': 'active genes, active-gene axis', 'sd02': 'genes with s.d. $\\geq$ 0.2',
               'allgene_axis': 'active genes, all-gene axis'}


def round5_tables(H):
    """Gene dependency as a second domain (105b/105c)."""
    table = H['table']
    out = []
    if not (DEP5 / 'models.csv').exists():
        return out
    Mo = pd.read_csv(DEP5 / 'models.csv')
    Po = pd.read_csv(DEP5 / 'pooled.csv')
    rows = []
    for cfg in ['primary', 'sd02', 'allgene_axis']:
        for r in Po[Po.config == cfg].itertuples():
            r = r._asdict()
            rows.append([DEP5_CONFIG.get(cfg, cfg), DEP5_NAME.get(r['group'], r['group']), est(r['axis_capture'], r['axis_capture_lo'], r['axis_capture_hi']),
                         est(r['selective_capture'], r['selective_capture_lo'], r['selective_capture_hi'], 3),
                         est(r['axis_share'], r['axis_share_lo'], r['axis_share_hi']),
                         est(r['perfect_share'], r['perfect_share_lo'], r['perfect_share_hi'], 3),
                         est(r['share_ratio'], r['share_ratio_lo'], r['share_ratio_hi']),
                         f"{r['median_raw']:.2f} / {r['median_selective']:.2f}".replace('-', '$-$')])
        if cfg != 'primary':
            continue
        for r in Mo[Mo.config == cfg].itertuples():
            r = r._asdict()
            rows.append(['', DEP5_NAME.get(r['model'], r['model']), est(r['axis_capture'], r['axis_capture_lo'], r['axis_capture_hi']),
                         est(r['selective_capture'], r['selective_capture_lo'], r['selective_capture_hi'], 3),
                         est(r['axis_share'], r['axis_share_lo'], r['axis_share_hi']),
                         '', est(r['share_ratio'], r['share_ratio_lo'], r['share_ratio_hi']) if r['kind'] == 'learned' else '--',
                         f"{r['median_raw']:.2f} / {r['median_selective']:.2f}".replace('-', '$-$')])
    out.append(('Dependency', table(
        r'\textbf{The decomposition in CRISPR gene-dependency prediction.} DepMap 24Q4 Chronos gene effects (sensitivity '
        r'$=-$effect) of 1,103 cell lines with expression; 1,077 selectively active genes (not common essential or '
        r'nonessential controls; $\geq$10\% of lines with effect $<-0.5$). Five model families trained on expression with '
        r'ten-fold leave-cell-line-out splits (seed 42), two gene-agnostic predictors (the omics-predicted axis scaled per '
        r'gene by its training loading; the tissue mean of the axis) and the tissue-by-gene mean. Axis capture, selective '
        r'capture, models\' axis share, perfect-predictor axis share, share ratio (models\' over the perfect predictor\'s) '
        r'and median per-gene raw / selective $r$; two-way (cell line $\times$ gene) bootstrap 95\% intervals ($B$ = 1,000). '
        r'Sensitivity: genes with s.d. $\geq$ 0.2, retrained; the axis built from all genes.',
        'tab:dependency', ['Gene set and axis', 'Model', 'Axis capture', 'Selective capture', 'Axis share', 'Perfect share',
                           'Share ratio', 'Median raw / sel.'], rows, 'p{3.0cm}p{3.6cm}llllll')))
    Pr = pd.read_csv(DEP5 / 'predictability.csv')
    Pr = Pr[Pr.config == 'primary']
    rows = []
    for r in Pr.itertuples():
        r = r._asdict()
        rows.append(['predictability', DEP5_NAME.get(r['model'], r['model']), f"$r \\geq$ {r['threshold']:.1f}",
                     est(r['n_pass_raw'], r['n_pass_raw_lo'], r['n_pass_raw_hi'], 0),
                     est(r['frac_axis_term_ge_half_r'], r['frac_axis_term_ge_half_r_lo'], r['frac_axis_term_ge_half_r_hi']),
                     f"{r['n_below_on_selective_term']:.0f}" if pd.notna(r['n_below_on_selective_term']) else '--'])
    Ax = pd.read_csv(DEP5 / 'axis.csv')
    for _, r in Ax[(Ax.axis == 'active_axis') & Ax.statistic.isin(['pearson_r', 'adjusted_r2', 'split_half_reliability'])].iterrows():
        lab = {'pearson_r': 'Pearson $r$', 'adjusted_r2': 'adjusted $R^2$', 'split_half_reliability': 'split-half reliability'}[r['statistic']]
        rows.append(['axis', H['esc'](str(r['with'])), lab,
                     est(r['value'], r['lo'], r['hi']) if pd.notna(r['lo']) else f"{r['value']:.2f}", '', ''])
    out.append(('DependencyAxis', table(
        r"\textbf{Gene-dependency prediction: per-gene ``predictability'' and what the general-dependency axis tracks.} "
        r'Upper block: genes whose out-of-fold per-gene correlation reaches the threshold, the fraction of them whose axis '
        r'term $r_{pg}r_{zg}$ is at least half of $r$, and how many would fall below the threshold on the selective term '
        r'alone (two-way bootstrap 95\% intervals). Lower block: the active-gene axis against screen quality (Achilles '
        r'screen QC), growth, Cas9 activity, lineage and screen library, and its split-half reliability over random halves '
        r'of the genes (cell-line bootstrap intervals).',
        'tab:dependencyaxis', ['Block', 'Model or covariate', 'Statistic', 'Value', 'Axis term $\\geq r/2$', 'Below on selective'],
        rows, 'lp{5.6cm}lllr')))
    return out

