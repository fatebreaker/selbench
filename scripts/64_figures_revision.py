"""Main figures of the revised manuscript (round 1 of review).

Figure 1  the general-sensitivity axis: reliable within a screen, partly shared between screens
Figure 2  DrEval re-scoring and the processing-controlled paired cross-screen decomposition
Figure 3  a second benchmark (IMPROVE) and drevalpy retraining on GDSC2 AUC (results/revision_improve, _retrain)
Figure 4  reproducible selectivity, the ceiling and the models           (results/revision_ceiling, revision_transfer)
Figure 5  pre-registered confirmation as estimation, with tumour- and patient-level transfer

Style, palette and helpers come from 13_figures.py; only saved results are read.
"""
import importlib
import json
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

F = importlib.import_module('13_figures')
plt = F.plt
BLUE, ORANGE, AQUA, GRAY, LIGHT_GRAY, INK, INK2, GRID = (F.BLUE, F.ORANGE, F.AQUA, F.GRAY, F.LIGHT_GRAY, F.INK,
                                                         F.INK2, F.GRID)
RESULTS, PROCESSED = F.RESULTS, F.PROCESSED
hgrid, save, panel_label, fit_width = F.hgrid, F.save, F.panel_label, F.fit_width
W = F.FIG_W                      # 180 mm; text 5-7 pt, panel letters 8 pt (13_figures.py)
AX = RESULTS / 'revision_axis'
XL = RESULTS / 'revision_crosslab'
TR = RESULTS / 'revision_transfer'
CE = RESULTS / 'revision_ceiling'
RT = RESULTS / 'revision_retrain'
LIGHT_BLUE = '#86b6ef'
BASE_COL = '#8250c4'   # baselines without learned drug-specific signal: drug-agnostic predictors, tissue-by-drug mean
PAIR_WITHIN = 'CTRPv2|CC|lnIC50->CTRPv2|CC|lnIC50'
RNG = np.random.default_rng(20261015)


def ci_str(s):
    a, b = [float(x) for x in str(s).strip('[]() ').split(',')]
    return a, b


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ================================================================== Figure 1
def panel_schematic(ax):
    ax.axis('off')
    rng = np.random.default_rng(1)
    g = np.sort(rng.normal(size=12))
    load = np.r_[np.linspace(1.0, 0.3, 6)]
    sel = rng.normal(scale=0.6, size=(6, 12)); sel[1, :3] += 2.2; sel[4, 8:11] += 2.0
    part = load[:, None] * g[None, :]
    z = part + sel
    mats = [(z, 'profiles $z$'), (part, 'loading x $g$'), (sel, 'selective\nresidual')]
    for k, (m, t) in enumerate(mats):
        a = ax.inset_axes([0.0 + k * 0.36, 0.36, 0.28, 0.44])
        a.imshow(m, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(LIGHT_GRAY); sp.set_linewidth(0.5)
        a.set_title(t, fontsize=6.2, color=INK, pad=3)
        if k == 0:
            a.set_ylabel('drugs', fontsize=6.2, color=INK2); a.set_xlabel('cell lines', fontsize=6.2, color=INK2)
    ax.text(0.315, 0.58, '=', transform=ax.transAxes, ha='center', va='center', fontsize=7, color=INK2)
    ax.text(0.675, 0.58, '+', transform=ax.transAxes, ha='center', va='center', fontsize=7, color=INK2)
    ax.text(0.0, 0.0, '$g$ = leave-drug-out mean of $z$ over the other drugs of\nthe same screen (a leave-one-out GLDS); '
            'selective\nresidual = $z$ after regression on $g$', transform=ax.transAxes, fontsize=5.6, color=INK2,
            va='bottom')


def panel_axis_share(ax):
    stats = pd.read_csv(RESULTS / 'decomposition' / 'drug_screen_stats.csv')
    data = [stats[(stats.screen == s) & (stats.frac_auc_below_0p8 >= 0.1)].fitness_r2.values for s in F.DISC]
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.0),
                    whiskerprops=dict(color=GRAY, lw=0.7), capprops=dict(color=GRAY, lw=0.7), boxprops=dict(lw=0))
    for p in bp['boxes']:
        p.set_facecolor('#b7d3f6')
    ax.set_xticks(range(1, 5)); ax.set_xticklabels([F.LABEL[s] for s in F.DISC], fontsize=5.6, rotation=35, ha='right')
    ax.set_ylabel("share of an active drug's profile\nexplained by $g$ ($R^2$)")
    ax.set_ylim(0, 0.8); hgrid(ax)
    for k, d in enumerate(data):
        ax.text(k + 1.27, np.median(d), f'{np.median(d):.2f}', ha='left', va='center', fontsize=5.6, color=INK)
    ax.set_title('within each screen', loc='left', color=INK, fontsize=6.6)


def ladder_rows():
    """Agreement of axes over shared cell lines, at fixed processing, by relationship between the two measurements."""
    ds = pd.read_csv(AX / 'dreval_datasets.csv')
    ap = pd.read_csv(AX / 'all_axis_pairs.csv')
    wl = pd.read_csv(AX / 'within_lab_controls.csv')
    rows = []
    for ep, fam in [('CC_AUC', 'AUC'), ('CC_lnIC50', 'lnIC50')]:
        for r in ds[(ds.endpoint == ep) & (ds.dataset != 'CCLE')].itertuples():
            rows.append(('split', fam, 'all', r.split_half, f'{r.dataset}'))
            rows.append(('split', fam, 'active', r.split_half_active, f'{r.dataset}'))
    pr = wl[(wl.control == 'PRISM HTS002 vs MTS010') & (wl.endpoint == 'drc_AUC')].iloc[0]
    rows.append(('run', 'AUC', 'all', pr.r, 'PRISM HTS002 vs MTS010'))
    for ep, fam in [('AUC', 'AUC'), ('lnIC50', 'lnIC50')]:
        for v in ['all', 'active']:
            q = wl[(wl.control == 'GDSC1 Sanger site vs MGH site') & (wl.endpoint == ep) & (wl.variant == v)]
            if len(q):
                rows.append(('site', fam, v, q.r.iloc[0], 'GDSC1 Sanger vs MGH'))
    dre = ap[ap.source == '46 DrEval']
    for ep, fam in [('CC_AUC', 'AUC'), ('CC_lnIC50', 'lnIC50')]:
        for v in ['all', 'active']:
            for r in dre[(dre.endpoint == ep) & (dre.variant == v)].itertuples():
                cls = 'inst' if r.pair_class.startswith('L') else 'other'
                rows.append((cls, fam, v, r.r, f'{r.a}-{r.b}'))
    return pd.DataFrame(rows, columns=['rel', 'family', 'variant', 'r', 'label'])


def panel_ladder(ax):
    R = ladder_rows()
    order = [('split', 'same screen,\nrandom drug halves'), ('run', 'same screen and\ncompounds, other run'),
             ('site', 'same screen,\nother site'), ('inst', 'same institution,\nother screen'),
             ('other', 'other institution')]
    for k, (rel, lab) in enumerate(order):
        for fam, col, dy in [('AUC', ORANGE, -0.16), ('lnIC50', AQUA, 0.16)]:
            for v, filled in [('all', False), ('active', True)]:
                d = R[(R.rel == rel) & (R.family == fam) & (R.variant == v)]
                if not len(d):
                    continue
                y = np.full(len(d), k + dy) + RNG.uniform(-0.05, 0.05, len(d))
                ax.scatter(d.r, y, s=13, facecolor=col if filled else 'white', edgecolor=col, lw=0.8, zorder=3)
            d = R[(R.rel == rel) & (R.family == fam) & (R.variant == 'all')]
            if len(d) > 1:
                ax.plot([d.r.median()] * 2, [k + dy - 0.13, k + dy + 0.13], color=INK, lw=1.1, zorder=4)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o[1] for o in order], fontsize=5.8)
    ax.invert_yaxis(); ax.set_xlim(0, 1.02); xgrid(ax)
    ax.set_xlabel('agreement of general-sensitivity axes ($r$)')
    hs = [plt.Line2D([], [], marker='o', ls='', color=ORANGE, ms=3.6, label='AUC'),
          plt.Line2D([], [], marker='o', ls='', color=AQUA, ms=3.6, label='ln IC$_{50}$'),
          plt.Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, ms=3.6, label='all drugs'),
          plt.Line2D([], [], marker='o', ls='', color=INK2, ms=3.6, label='active drugs')]
    ax.legend(handles=hs, loc='upper left', fontsize=5.4, ncol=2, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0.1)
    ax.set_title('processing held fixed in each comparison', loc='left', color=INK, fontsize=6.6)


def panel_shared_fraction(ax):
    T = pd.read_csv(AX / 'triad_shared_fraction.csv')
    sets = [('all-drug axes (paper)', 'all drugs', LIGHT_BLUE),
            ('active-drug axes (discovery + gCSI active; NCI-60 all)', 'active drugs', BLUE),
            ('inactive-drug discovery axes (gCSI, NCI-60 all)', 'inactive drugs', GRAY)]
    for j, (key, lab, col) in enumerate(sets):
        for k, s in enumerate(F.DISC):
            r = T[(T.axis_set == key) & (T.screen == s)]
            if not len(r):
                continue
            r = r.iloc[0]
            x = k + (j - 1) * 0.22
            ax.errorbar(x, r.lambda2, yerr=[[r.lambda2 - r.ci_lo], [r.ci_hi - r.lambda2]], fmt='o', ms=3.6, color=col,
                        ecolor=col, elinewidth=0.8, label=lab if k == 0 else None)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels([F.LABEL[s] for s in F.DISC], fontsize=5.8)
    ax.set_ylabel('share of the axis common\nto all screens ($\\lambda^2$, triads)')
    ax.set_ylim(-0.1, 0.62); hgrid(ax)
    ax.legend(loc='upper left', fontsize=5.4, handletextpad=0.1, borderaxespad=0.1)
    ax.set_title('split-half reliability 0.95-0.99 for every set', loc='left', color=INK2, fontsize=6.0)


def panel_variance(ax):
    V = pd.read_csv(AX / 'variance_components.csv')
    models = [('REML, native AUC-type with runs/sites, unbalanced: cell + cell x lab + cell x screen(lab) + residual '
               '(run/site)', 'provider AUC,\n6 screens + runs/sites'),
              ('REML, DrEval CC_AUC, 5 screens, unbalanced (cells in >=2 screens): cell + cell x institution + residual',
               'CurveCurator AUC,\n5 screens'),
              ('REML, DrEval CC_lnIC50, 5 screens, unbalanced (cells in >=2 screens): cell + cell x institution + residual',
               'CurveCurator ln IC$_{50}$,\n5 screens')]
    comps = [('cell', 'shared by all screens', BLUE), ('lab', 'institution', ORANGE), ('screen', 'screen', LIGHT_BLUE),
             ('residual', 'screen, run or site', LIGHT_GRAY)]
    for k, (m, lab) in enumerate(models):
        d = V[V.model == m].set_index('component')
        left = 0.0
        for c, clab, col in comps:
            if c not in d.index:
                continue
            f = d.loc[c, 'fraction']
            ax.barh(k, f, left=left, color=col, height=0.55, edgecolor='white', lw=0.5)
            if f > 0.06:
                ax.text(left + f / 2, k, f'{f:.2f}', ha='center', va='center', fontsize=5.4,
                        color='white' if col in (BLUE, ORANGE) else INK)
            left += f
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m[1] for m in models], fontsize=5.8)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel('share of cell-level variance of the axis (REML)')
    ax.legend(handles=[Patch(facecolor=c[2], label=c[1]) for c in comps], loc='lower left', ncol=4, fontsize=5.4,
              handlelength=0.9, columnspacing=0.6, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)


def ed_figure2():
    """Extended Data Fig. 2 (formerly Extended Data Fig. 1): the axis is reliable within a screen and partly shared between screens."""
    fig = plt.figure(figsize=(W, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.2], height_ratios=[1.0, 0.85], wspace=0.6, hspace=0.6)
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, 'a', x=-0.42); panel_axis_share(ax)
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, 'b', x=-0.45); panel_ladder(ax)
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, 'c', x=-0.3); panel_shared_fraction(ax)
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, 'd', x=-0.22); panel_variance(ax)
    save(fig, 'extended_data_fig2_axis')


EX = RESULTS / 'revision_round4' / 'fig1_example'
EX_SHORT = {'perfect predictor (measured profile)': 'perfect predictor\n(measured profile)',
            'EnsembleMF (best released model)': 'EnsembleMF\n(best released model)',
            'omics-predicted axis (drug-agnostic)': 'omics-predicted axis\n(drug-agnostic)',
            'tissue-by-drug mean': 'tissue-by-drug\nmean'}


def panel_example_profile(ax):
    """One drug's profile against the leave-drug-out axis (101_fig1_example.py)."""
    C = pd.read_csv(EX / 'cells.csv')
    T = pd.read_csv(EX / 'terms.csv')
    ax.scatter(C.g, C.z, s=4, color=LIGHT_GRAY, lw=0, zorder=2)
    b1, b0 = np.polyfit(C.g, C.z, 1)
    xs = np.linspace(C.g.min(), C.g.max(), 10)
    ax.plot(xs, b0 + b1 * xs, color=INK, lw=0.9, zorder=3)
    r = T.r_zg.iloc[0]
    ax.text(0.03, 0.97, f"{T.name.iloc[0]} ({int(T.cells.iloc[0])} cell lines)\n$r_{{zg}}$ = {r:.2f}; "
            f"axis share {r ** 2:.2f}", transform=ax.transAxes, fontsize=5.4, color=INK2, va='top')
    ax.set_xlabel('general-sensitivity axis $g$')
    ax.set_ylabel('drug profile $z$ (rank-normal)')
    hgrid(ax)


def panel_example_terms(ax):
    """Each predictor's per-drug correlation for the same drug, split into axis and selective terms."""
    T = pd.read_csv(EX / 'terms.csv')
    for k, r in enumerate(T.itertuples()):
        ax.barh(k, r.axis_term, color=GRAY, height=0.62, zorder=3)
        ax.barh(k, r.selective_term, left=r.axis_term, color=ORANGE, height=0.62, zorder=3)
        lab = 'r = 1' if k == 0 else f'r = {r.raw:.2f}'
        ax.text(r.raw + 0.02, k, lab, va='center', fontsize=5.2, color=INK2)
    ax.set_yticks(range(len(T)))
    ax.set_yticklabels([EX_SHORT.get(x, x) for x in T.predictor], fontsize=5.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18); xgrid(ax)
    ax.set_xlabel('per-drug correlation $r_{pz}$\n= axis term + selective term')
    ax.legend(handles=[Patch(color=GRAY, label='axis term $r_{pg}r_{zg}$'),
                       Patch(color=ORANGE, label='selective term')], loc='center right', bbox_to_anchor=(1.02, 0.5),
              fontsize=5.0, handlelength=1.0, borderaxespad=0.2, frameon=False)


def panel_rank_shift(ax):
    """Rank of the 22 released models on the raw per-drug score and on the selective score (88_rank_intervals.py)."""
    R = pd.read_csv(RESULTS / 'revision_round2' / 'rankings' / 'rank_intervals.csv')
    C = pd.read_csv(RESULTS / 'revision_round2' / 'capture' / 'capture_by_model.csv')
    agn = set(C[(C.benchmark == 'leaderboard') & C.drug_agnostic.notna()].model)
    raw = R[R.metric == 'raw'].set_index('model')['rank']
    sel = R[R.metric == 'selective'].set_index('model')['rank']
    n = len(raw)
    for m in raw.index:
        col, lw, z_ = (BASE_COL, 1.0, 3) if m in agn else ((BLUE, 1.0, 3) if m == 'EnsembleMF' else (LIGHT_GRAY, 0.6, 2))
        ax.plot([0, 1], [raw[m], sel[m]], color=col, lw=lw, zorder=z_, marker='o', ms=2.2)
        if m in agn or m == 'EnsembleMF' or raw[m] <= 3:
            ax.text(-0.04, raw[m], NAMES.get(m, m), ha='right', va='center', fontsize=5.0, color=col if col != LIGHT_GRAY else INK2)
            ax.text(1.04, sel[m], f"{int(sel[m])}", ha='left', va='center', fontsize=5.0, color=col if col != LIGHT_GRAY else INK2)
    ax.set_xlim(-0.9, 1.25); ax.set_ylim(n + 0.8, 0.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['raw per-drug\ncorrelation', 'selective\nscore'], fontsize=5.6)
    ax.set_ylabel('rank among 22 released models')
    ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x', length=0)
    ax.legend(handles=[plt.Line2D([], [], color=BASE_COL, marker='o', ms=2.2, lw=1.0, label='drug-agnostic'),
                       plt.Line2D([], [], color=BLUE, marker='o', ms=2.2, lw=1.0, label='EnsembleMF')],
              loc='lower left', bbox_to_anchor=(0.0, -0.02), fontsize=5.0, frameon=False, handlelength=1.4)


def figure1():
    """Fig. 1 (headline): the decomposition, a worked example, capture against a perfect predictor in every
    leave-cell-line-out setting, and how rankings change."""
    fig = plt.figure(figsize=(W, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.15], hspace=0.55)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.15, 0.85, 1.0], wspace=0.75)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.45, 0.85], wspace=0.55)
    ax = fig.add_subplot(top[0]); panel_label(ax, 'a', x=-0.02, y=1.12); panel_schematic(ax)
    ax = fig.add_subplot(top[1]); panel_label(ax, 'b', x=-0.4, y=1.12); panel_example_profile(ax)
    ax = fig.add_subplot(top[2]); panel_label(ax, 'c', x=-0.75, y=1.12); panel_example_terms(ax)
    ax = fig.add_subplot(bot[0]); panel_label(ax, 'd', x=-0.2, y=1.2); panel_capture(ax)
    ax = fig.add_subplot(bot[1]); panel_label(ax, 'e', x=-0.1, y=1.2); panel_rank_shift(ax)
    save(fig, 'figure1_headline')


# ================================================================== Figure 2
NAMES = dict(F.DREVAL_NAMES)
NAMES.update({'fitness_only': 'axis only (one value per cell line)', 'NaiveTissueDrugMeanPredictor': 'tissue x drug mean',
              'ridge_sel_lm': 'kernel ridge, selective target', 'ridge_raw_lm': 'kernel ridge, raw target'})
AGNOSTIC = ['fitness_only', 'NaiveTissueMeanPredictor', 'NaiveMeanEffectsPredictor', 'MultiViewRandomForest']


def per_drug(pair, cols):
    P = pd.read_csv(XL / 'per_drug_scores.csv.gz', usecols=['pair', 'model', 'drug', 'class'] + cols)
    return P[P.pair == pair]


def boot_means(P, cols, n=1000):
    """Drug-bootstrap of per-model means (drugs resampled independently for each model)."""
    out = []
    for m, d in P.groupby('model'):
        X = d[cols].to_numpy(float)
        X = X[np.isfinite(X).all(1)]
        idx = RNG.integers(0, len(X), (n, len(X)))
        B = X[idx].mean(1)
        row = {'model': m, 'drugs': len(X)}
        for j, c in enumerate(cols):
            row[c] = X[:, j].mean(); row[c + '_lo'], row[c + '_hi'] = np.percentile(B[:, j], [2.5, 97.5])
        if 'med_w' in cols and 'raw_w' in cols:
            sh = B[:, cols.index('med_w')] / B[:, cols.index('raw_w')]
            row['share'] = row['med_w'] / row['raw_w']; row['share_lo'], row['share_hi'] = np.percentile(sh, [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out)


def model_class():
    S = pd.read_csv(RESULTS / 'dreval' / 'model_summary.csv')
    return S.set_index('model')['class'].to_dict()


def panel_leaderboard(ax):
    """Per-drug versus selective correlation within CTRPv2 (unseen cell lines), with the drug-agnostic floor."""
    S, fm, _ = F.dreval_summary()
    floor = S[S.model.isin(AGNOSTIC)].selective.max()
    ax.axhspan(-0.1, floor, color='#f1f0ea', zorder=0)
    ax.text(0.005, floor + 0.004, f'drug-agnostic floor ({floor:.2f})', fontsize=5.2, color=INK2, va='bottom')
    for r in S.itertuples():
        col = F.DREVAL_COL[r.klass]
        mk = '^' if r.model in AGNOSTIC else 'o'
        ax.errorbar(r.raw, r.selective, xerr=[[r.raw - r.raw_lo], [r.raw_hi - r.raw]],
                    yerr=[[r.selective - r.selective_lo], [r.selective_hi - r.selective]], fmt=mk, ms=4.0, color=col,
                    mec='white', mew=0.5, elinewidth=0.5, ecolor=col, zorder=3)
    ax.axvline(fm, color=GRAY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(fm + 0.008, 0.15, 'measured axis $g$ (not cell-blind)', rotation=90, ha='left', va='center', fontsize=5.2,
            color=INK2)
    label = {'ridge_sel_lm': (0.014, 0.0, 'left'), 'ridge_raw_lm': (0.0, 0.018, 'center'),
             'EnsembleMF': (-0.014, 0.006, 'right'), 'RandomForest': (0.014, -0.004, 'left'),
             'NaiveTissueDrugMeanPredictor': (-0.014, 0.006, 'right')}
    for m, (dx, dy, ha) in label.items():
        r = S[S.model == m]
        if len(r):
            ax.text(r.raw.iloc[0] + dx, r.selective.iloc[0] + dy, NAMES.get(m, m), fontsize=5.2, color=INK2, ha=ha,
                    va='center', zorder=4)
    pts = S[S.model.isin(['fitness_only', 'MultiViewRandomForest'])]
    ax.annotate('axis only,\nMultiView-\nRandomForest', xy=(pts.raw.max() + 0.006, pts.selective.mean()),
                xytext=(0.445, 0.075), fontsize=5.2, color=INK2, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.5))
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlim(0.0, 0.6); ax.set_ylim(-0.03, 0.35); hgrid(ax)
    ax.set_xlabel('per-drug correlation (median over drugs)'); ax.set_ylabel('selective correlation (median over drugs)')
    for cls, lab in [('naive', 'naive baseline'), ('classical', 'classical ML'), ('neural/factorization', 'neural / factorization'),
                     ('this study', 'kernel ridge (this study)')]:
        ax.scatter([], [], s=14, color=F.DREVAL_COL[cls], label=lab)
    ax.scatter([], [], s=14, marker='^', color=GRAY, label='drug-agnostic')
    ax.legend(loc='upper left', fontsize=5.3, ncol=2, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.1)
    ax.set_title('DrEval leaderboard (CTRPv2 ln IC$_{50}$, unseen cell lines)', loc='left', color=INK, fontsize=6.6)


def panel_decomposition(ax):
    """Mean per-drug correlation split into axis-mediated and selective terms, with drug-bootstrap intervals."""
    P = per_drug(PAIR_WITHIN, ['raw_w', 'med_w', 'selterm_w'])
    cls = model_class()
    keep = [m for m in P.model.unique() if cls.get(m) in ('classical', 'neural/factorization', 'naive')
            or m in ('fitness_only', 'ridge_raw_lm', 'ridge_sel_lm')]
    B = boot_means(P[P.model.isin(keep)], ['raw_w', 'med_w', 'selterm_w'])
    B = B[B.raw_w > 0.03].sort_values('raw_w')
    y = np.arange(len(B))
    ax.barh(y, B.med_w, color=LIGHT_GRAY, height=0.66, label='through the axis ($r_{pg}r_{zg}$)')
    ax.barh(y, B.selterm_w.clip(lower=0), left=B.med_w, color=ORANGE, height=0.66, label='selective term')
    ax.errorbar(B.raw_w, y, xerr=[B.raw_w - B.raw_w_lo, B.raw_w_hi - B.raw_w], fmt='none', ecolor=INK, elinewidth=0.6)
    for yi, r in zip(y, B.itertuples()):
        ax.text(0.505, yi, f'{100 * r.share:.0f} ({100 * r.share_lo:.0f}-{100 * r.share_hi:.0f})', va='center',
                fontsize=5.0, color=INK2)
    ax.text(0.505, len(B) - 0.1, '% through\nthe axis', va='bottom', fontsize=5.0, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES.get(m, m) for m in B.model], fontsize=5.0)
    for t, m in zip(ax.get_yticklabels(), B.model):
        c = cls.get(m)
        t.set_color(AQUA if m.startswith('ridge') else (GRAY if c == 'naive' or m == 'fitness_only' else INK))
    ax.set_xlim(0, 0.5); xgrid(ax)
    ax.set_xlabel('mean per-drug correlation (95% drug-bootstrap interval)')
    ax.legend(loc='lower right', fontsize=5.3, handlelength=1.0, borderaxespad=0.1)


TARGETS = [  # (pair, group, row label)
    ('CTRPv2|CC|lnIC50->CTRPv2|CC|AUC', 'CTRPv2', 'CTRPv2 re-processed: CC AUC'),
    ('CTRPv2|CC|lnIC50->CTRPv2|CTRP|AUC', 'CTRPv2', 'CTRPv2 re-processed: provider AUC'),
    ('CTRPv2|CC|lnIC50->CTRPv1|CC|lnIC50', 'CTRPv1', 'CTRPv1: CC ln IC$_{50}$'),
    ('CTRPv2|CC|AUC->CTRPv1|CC|AUC', 'CTRPv1', 'CTRPv1: CC AUC'),
    ('CTRPv2|CC|lnIC50->CTRPv1|CTRP|AUC', 'CTRPv1', 'CTRPv1: provider AUC'),
    ('CTRPv2|SP-CTRPv1|AUC->CTRPv1|SP|AUC', 'CTRPv1', 'CTRPv1: CC AUC, shared panel'),
    ('CTRPv2|HR-CTRPv1|AUC->CTRPv1|HR|AUC', 'CTRPv1', 'CTRPv1: CC AUC, shared panel + range'),
    ('CTRPv2|RL-CTRPv1|lnIC50->CTRPv1|RL|lnIC50', 'CTRPv1', 'CTRPv1: CC ln IC$_{50}$, shared panel + range'),
    ('CTRPv2|CC|lnIC50->GDSC2|CC|lnIC50', 'GDSC2', 'GDSC2: CC ln IC$_{50}$'),
    ('CTRPv2|CC|AUC->GDSC2|CC|AUC', 'GDSC2', 'GDSC2: CC AUC'),
    ('CTRPv2|CC|lnIC50->GDSC2|Sanger|lnIC50', 'GDSC2', 'GDSC2: provider ln IC$_{50}$'),
    ('CTRPv2|CC|lnIC50->GDSC2|Sanger|AUC', 'GDSC2', 'GDSC2: provider AUC'),
    ('CTRPv2|SP-GDSC2|AUC->GDSC2|SP|AUC', 'GDSC2', 'GDSC2: CC AUC, shared panel'),
    ('CTRPv2|HR-GDSC2|AUC->GDSC2|HR|AUC', 'GDSC2', 'GDSC2: CC AUC, shared panel + range'),
    ('CTRPv2|RL-GDSC2|lnIC50->GDSC2|RL|lnIC50', 'GDSC2', 'GDSC2: CC ln IC$_{50}$, shared panel + range'),
    ('CTRPv2|CC|lnIC50->GDSC1|CC|lnIC50', 'GDSC1', 'GDSC1: CC ln IC$_{50}$'),
    ('CTRPv2|CC|AUC->GDSC1|CC|AUC', 'GDSC1', 'GDSC1: CC AUC'),
    ('CTRPv2|CC|lnIC50->GDSC1|Sanger|lnIC50', 'GDSC1', 'GDSC1: provider ln IC$_{50}$'),
    ('CTRPv2|CC|lnIC50->GDSC1|Sanger|AUC', 'GDSC1', 'GDSC1: provider AUC'),
    ('CTRPv2|SP-GDSC1|AUC->GDSC1|SP|AUC', 'GDSC1', 'GDSC1: CC AUC, shared panel'),
    ('CTRPv2|HR-GDSC1|AUC->GDSC1|HR|AUC', 'GDSC1', 'GDSC1: CC AUC, shared panel + range'),
    ('CTRPv2|RL-GDSC1|lnIC50->GDSC1|RL|lnIC50', 'GDSC1', 'GDSC1: CC ln IC$_{50}$, shared panel + range'),
]
HRDIR = XL / 'harmonized_range'
FLOOR_MODELS = ['fitness_only', 'MultiViewRandomForest', 'NaiveTissueMeanPredictor', 'NaiveMeanEffectsPredictor']


def target_rows():
    S = pd.read_csv(XL / 'summary_by_target.csv')
    S = S[(S.subset == 'all') & (S.cellset == 'own')]
    T = pd.read_csv(XL / 'tost_selectivity.csv')
    T = T[(T.subset == 'all') & (T.cellset == 'own') & (T.metric == 'selBOTH')]
    L = pd.read_csv(XL / 'leakage_floors.csv')
    L = L[(L.subset == 'all') & (L.cellset == 'own') & (L.metric == 'selBOTH') & (L.side == 'cross')
          & L.model.isin(FLOOR_MODELS)]
    floor = L.groupby('pair')['median'].max()
    if (HRDIR / 'summary_by_target.csv').exists():
        hs = pd.read_csv(HRDIR / 'summary_by_target.csv')
        ht = pd.read_csv(HRDIR / 'tost_selectivity.csv')
        S = pd.concat([S, hs[hs.subset == 'all']], ignore_index=True)
        T = pd.concat([T, ht[(ht.subset == 'all') & (ht.metric == 'selBOTH')]], ignore_index=True)
        hf = pd.read_csv(HRDIR / 'leakage_floors.csv')
        hf = hf[hf.model.isin(FLOOR_MODELS)].groupby('pair')['median'].max()
        floor = pd.concat([floor, hf])
    S, T = S.set_index('pair'), T.set_index('pair')
    # round 2: symmetric selective-both (82) and the retained-fraction test (83) replace the asymmetric change and TOST
    Y = pd.read_csv(RESULTS / 'revision_round2' / 'symmetric_selboth' / 'selboth_by_target.csv')
    Y = Y[(Y.benchmark == 'leaderboard') & (Y.group == 'main')].set_index('pair')
    E = pd.read_csv(RESULTS / 'revision_round2' / 'equivalence' / 'equivalence_34.csv')
    E = E[E.selective_both == 'symmetric'].set_index('pair')['noninferior_f_0.8']
    rows = []
    for pair, grp, lab in TARGETS:
        if pair not in S.index:
            continue
        s, t = S.loc[pair], T.loc[pair]
        rows.append(dict(pair=pair, group=grp, label=lab, drugs=int(s.drugs),
                         ret=s.mediated_retained, ret_lo=s.mediated_retained_lo, ret_hi=s.mediated_retained_hi,
                         rpg=s.alignment_ratio_r_pg, rpg_lo=s.alignment_ratio_r_pg_lo, rpg_hi=s.alignment_ratio_r_pg_hi,
                         rzg=s.target_loading_ratio_r_zg, rzg_lo=s.target_loading_ratio_r_zg_lo,
                         rzg_hi=s.target_loading_ratio_r_zg_hi, draw=-s.raw_loss, draw_lo=-s.raw_loss_hi,
                         draw_hi=-s.raw_loss_lo,
                         dsel=Y.loc[pair, 'change_sym'] if pair in Y.index else t.diff_cross_minus_within,
                         dsel_lo=Y.loc[pair, 'change_sym_lo90'] if pair in Y.index else t.ci90_lo,
                         dsel_hi=Y.loc[pair, 'change_sym_hi90'] if pair in Y.index else t.ci90_hi,
                         kept=bool(E[pair]) if pair in E.index else None, floor=floor.get(pair, np.nan)))
    return pd.DataFrame(rows)


def rows_frame(ax, R, labels):
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels([f'{r.label} ({r.drugs})' for r in R.itertuples()] if labels else [], fontsize=6.0)
    ax.set_ylim(len(R) - 0.5, -2.2)
    for k in range(1, len(R)):
        if R.group.iloc[k] != R.group.iloc[k - 1]:
            ax.axhline(k - 0.5, color=LIGHT_GRAY, lw=0.6)
    ax.axhline(-0.5, color=LIGHT_GRAY, lw=0.6)
    ax.tick_params(axis='y', length=0)


def panel_retention(ax, R):
    for k, r in enumerate(R.itertuples()):
        for off, v, lo, hi, col, mk in [(-0.24, r.rpg, r.rpg_lo, r.rpg_hi, BLUE, 's'),
                                        (0.0, r.ret, r.ret_lo, r.ret_hi, INK, 'o'),
                                        (0.24, r.rzg, r.rzg_lo, r.rzg_hi, ORANGE, 'D')]:
            ax.errorbar(v, k + off, xerr=[[v - lo], [hi - v]], fmt=mk, ms=3.0 if mk != 'o' else 3.8, color=col,
                        ecolor=col, elinewidth=0.7, zorder=3)
    ax.axvline(1, color=GRAY, lw=0.6, ls=(0, (3, 2)))
    rows_frame(ax, R, True)
    ax.set_xlim(0, 1.9); xgrid(ax)
    ax.set_xlabel('ratio, other dataset / CTRPv2\n(95% two-way bootstrap interval)')
    hs = [plt.Line2D([], [], marker='o', ls='', color=INK, ms=3.6, label='axis term $r_{pg}r_{zg}$'),
          plt.Line2D([], [], marker='s', ls='', color=BLUE, ms=3.2, label='alignment $r_{pg}$'),
          plt.Line2D([], [], marker='D', ls='', color=ORANGE, ms=3.0, label='loading $r_{zg}$')]
    ax.legend(handles=hs, loc='upper left', fontsize=5.0, handletextpad=0.1, borderaxespad=0.15, ncol=2,
              columnspacing=0.5)


def panel_change(ax, R):
    for k, r in enumerate(R.itertuples()):
        ax.errorbar(r.draw, k - 0.17, xerr=[[r.draw - r.draw_lo], [r.draw_hi - r.draw]], fmt='d', ms=3.4, color=GRAY,
                    ecolor=GRAY, elinewidth=0.7, zorder=3)
        ax.errorbar(r.dsel, k + 0.17, xerr=[[r.dsel - r.dsel_lo], [r.dsel_hi - r.dsel]], fmt='o', ms=3.4,
                    color=ORANGE, mfc='white' if r.kept is False else ORANGE, mec=ORANGE, ecolor=ORANGE, elinewidth=0.7,
                    zorder=3)
        hi = r.floor >= 0.05
        ax.text(1.04, k, f'{max(r.floor, 0.0):.2f}', transform=ax.get_yaxis_transform(), va='center', fontsize=5.2,
                color=ORANGE if hi else INK2, fontweight='bold' if hi else 'normal')
    ax.text(1.04, -0.6, 'drug-agnostic\nfloor', transform=ax.get_yaxis_transform(), va='bottom', fontsize=5.0,
            color=INK2)
    ax.axvline(0, color=GRAY, lw=0.6)
    rows_frame(ax, R, False)
    ax.set_xlim(-0.25, 0.12); xgrid(ax)
    ax.set_xlabel('change, other dataset - CTRPv2\n(mean over models and drugs)')
    hs = [plt.Line2D([], [], marker='d', ls='', color=GRAY, ms=3.4, label='per-drug (raw), 95% CI'),
          plt.Line2D([], [], marker='o', ls='', color=ORANGE, ms=3.4, label='selective-both, 90% CI'),
          plt.Line2D([], [], marker='o', ls='', mfc='white', mec=ORANGE, ms=3.4, label='not shown to keep $\\geq$80%')]
    ax.legend(handles=hs, loc='upper left', fontsize=5.0, handletextpad=0.1, borderaxespad=0.15)


def panel_paired(ax, target='GDSC1|CC|lnIC50'):
    """Same out-of-fold predictions scored in CTRPv2 and in another screen: per-model mean terms with intervals."""
    pair = f'CTRPv2|CC|lnIC50->{target}'
    P = per_drug(pair, ['med_w', 'med_c', 'selterm_w', 'selterm_c'])
    cls = model_class()
    P = P[P.model.map(lambda m: cls.get(m) in ('classical', 'neural/factorization')) & (P.model != 'SVR')]
    B = boot_means(P, ['med_w', 'med_c', 'selterm_w', 'selterm_c'])
    lim = 0.4
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.6, zorder=1, label='fully retained')
    ax.plot([0, lim], [0, lim / 2], color=GRAY, lw=0.5, ls=(0, (2, 2)), zorder=1, label='half retained')
    for a, b, col, ec, lab in [('med_w', 'med_c', LIGHT_GRAY, INK2, 'axis term'),
                               ('selterm_w', 'selterm_c', ORANGE, 'white', 'selective term')]:
        ax.errorbar(B[a], B[b], xerr=[B[a] - B[a + '_lo'], B[a + '_hi'] - B[a]], yerr=[B[b] - B[b + '_lo'], B[b + '_hi'] - B[b]],
                    fmt='none', ecolor=GRAY if col == LIGHT_GRAY else col, elinewidth=0.4, alpha=0.7, zorder=2)
        ax.scatter(B[a], B[b], s=14, color=col, edgecolor=ec, lw=0.4, zorder=3, label=lab)
    ax.set_xlim(-0.03, lim); ax.set_ylim(-0.03, lim)
    ax.set_xlabel('mean term, scored in CTRPv2')
    ax.set_ylabel('same predictions, scored in GDSC1')
    ax.legend(loc='upper left', fontsize=5.4, handletextpad=0.1, borderaxespad=0.1)
    hgrid(ax)
    ax.set_title(f'{len(B)} published models,\nGDSC1 CC ln IC$_{{50}}$ (95% CIs)', loc='left', color=INK, fontsize=6.4)


# (figure2 of the round-5 layout: replaced by the regrouped figures below)


# ================================================================== Figure 4
def panel_classes(ax):
    K = pd.read_csv(TR / 'class_summary_curated.csv')
    K = K[K.scheme == 'curated'].sort_values('median_rho_sel')
    K = K[K.drugs >= 5]                      # classes with fewer than five drugs are not ranked (Supplementary Table)
    A = pd.read_csv(TR / 'activity_regression.csv')
    A = A[(A.scheme == 'curated') & (A.spec == 'logit_quadratic')].set_index('class')
    for k, r in enumerate(K.to_dict('records')):
        ax.errorbar(r['median_rho_sel'], k + 0.15, xerr=[[r['median_rho_sel'] - r['median_ci95_lo']],
                                                          [r['median_ci95_hi'] - r['median_rho_sel']]],
                    fmt='o', ms=3.4, color=BLUE, ecolor=BLUE, elinewidth=0.7, zorder=3)
        if r['class'] in A.index:
            a_ = A.loc[r['class']]
            ax.errorbar(a_.adjusted_mean, k - 0.15, xerr=[[a_.adjusted_mean - a_.adj_ci95_lo], [a_.adj_ci95_hi - a_.adjusted_mean]],
                        fmt='D', ms=2.8, mfc='white', mec=GRAY, color=GRAY, ecolor=GRAY, elinewidth=0.6, zorder=2)
        ax.text(0.665, k, f"{100 * r['frac_reproducible']:.0f}%", va='center', fontsize=5.0, color=INK2)
    ax.text(0.665, len(K) - 0.45, '% repr.', va='bottom', fontsize=5.0, color=INK2)
    ax.set_yticks(range(len(K)))
    ax.set_yticklabels([f"{r['class']} ({int(r['drugs'])})" for r in K.to_dict('records')], fontsize=5.2)
    ax.axvline(0.3, color=GRAY, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(-0.05, 0.74); ax.set_ylim(-0.6, len(K) + 0.1); xgrid(ax)
    ax.set_xlabel('cross-screen selective agreement')
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.4, label='median (95% CI)'),
                       plt.Line2D([], [], marker='D', ls='', mfc='white', mec=GRAY, ms=3.0,
                                  label='activity-adjusted mean')], loc='lower left',
              bbox_to_anchor=(-0.02, 1.0), fontsize=5.0, handletextpad=0.1, borderaxespad=0.1, ncol=2,
              columnspacing=0.8)
    ax.set_title('353 multi-screen drugs by mechanism class', loc='left', color=INK, fontsize=6.4, pad=13)


def panel_c2(ax):
    E = pd.read_csv(CE / '49_c2_estimands.csv')
    E = E[(E.kappa == 'none') & (E.aggregation == 'perdrug_romeans_bc')].set_index(['model', 'class'])
    B = pd.read_csv(CE / '52_biomarker_c2_pooled.csv').set_index('set').loc['core, primary biomarker per drug']
    rows = [('fitness_omics', 'omics-predicted axis'), ('lineage_raw', 'lineage only'),
            ('target_prior', 'target dependency (CRISPR)'), ('ridge_raw', 'kernel ridge, raw target'),
            ('ridge_sel', 'kernel ridge, selective target'), ('ridge_crispr', '+ CRISPR gene effects'),
            ('replicate_sel', 'one more screen (selective residual)')]
    for k, (m, lab) in enumerate(rows):
        for cls, col, dy in [('all', BLUE, -0.15), ('reproducible', ORANGE, 0.15)]:
            r = E.loc[(m, cls)]
            ax.errorbar(r.estimate, k + dy, xerr=[[r.estimate - r.ci_lo], [r.ci_hi - r.estimate]], fmt='o', ms=3.2,
                        color=col, ecolor=col, elinewidth=0.7, zorder=3)
    k0 = len(rows) + 0.9
    sub = [('c2_biomarker_pooled', 'ci_lo', 'ci_hi', 'single canonical biomarker'),
           ('c2_ridge_pooled', 'ridge_ci_lo', 'ridge_ci_hi', 'kernel ridge, selective target'),
           ('c2_replicate_pooled_same_drugs', None, None, 'one more screen (measured)')]
    for j, (c, lo, hi, lab) in enumerate(sub):
        v = B[c]
        xerr = [[v - B[lo]], [B[hi] - v]] if lo else None
        ax.errorbar(v, k0 + j, xerr=xerr, fmt='s', ms=3.2, color=AQUA if j == 0 else INK2, ecolor=AQUA if j == 0 else INK2,
                    elinewidth=0.7, zorder=3)
    ax.axhline(len(rows) + 0.3, color=LIGHT_GRAY, lw=0.6)
    ax.text(0.005, len(rows) + 0.45, f"{int(B.n_drugs)} drugs with a canonical biomarker", fontsize=5.2, color=INK2,
            va='top')
    ax.set_yticks(list(range(len(rows))) + [k0 + j for j in range(3)])
    ax.set_yticklabels([r[1] for r in rows] + [s_[3] for s_ in sub], fontsize=5.4)
    ax.set_ylim(k0 + 2.6, -0.6)
    ax.set_xlim(0, 0.46); xgrid(ax)
    ax.set_xlabel('reproducible selective variance\ncaptured ($c^2$; 95% drug-cluster CI)')
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.2, label='drugs in $\\geq$3 screens (141)'),
                       plt.Line2D([], [], marker='o', ls='', color=ORANGE, ms=3.2, label='reproducible drugs (57)')],
              loc='center right', bbox_to_anchor=(1.0, 0.82), fontsize=5.0, handletextpad=0.1, borderaxespad=0.1)


def panel_biomarker(ax):
    D = pd.read_csv(CE / '52_biomarker_c2_perdrug.csv')
    D = D[D.primary_for_drug & D.estimable.astype(bool) & D.c2_ridge_sel.notna() & D.c2_biomarker.notna()]
    lim = 0.7
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.6)
    col = np.where(D.tier == 'core', AQUA, LIGHT_BLUE)
    ax.scatter(D.c2_ridge_sel, D.c2_biomarker, s=12, color=col, edgecolor='white', lw=0.4, zorder=3)
    show = {'topotecan': (0.01, 0.55, 'left'), 'venetoclax': (0.01, 0.45, 'left'), '17-AAG': (0.20, 0.405, 'left'),
            'trametinib': (0.47, 0.445, 'left'), 'dabrafenib': (0.42, 0.61, 'right'),
            'afatinib': (0.43, 0.255, 'left')}           # label positions in data coordinates, in free space
    for r in D.to_dict('records'):
        if r['drug_name'] in show:
            tx, ty, ha = show[r['drug_name']]
            ax.annotate(f"{r['drug_name']} ({r['biomarker'].split(' ')[0]})", xy=(r['c2_ridge_sel'], r['c2_biomarker']),
                        xytext=(tx, ty), textcoords='data', fontsize=5.0, color=INK2, ha=ha, va='center',
                        arrowprops=dict(arrowstyle='-', color=LIGHT_GRAY, lw=0.4, shrinkA=1, shrinkB=2))
    # pooled c^2 over the 29 core drugs with drug-cluster 95% intervals (52_biomarker_baselines.py)
    P = json.loads((CE / '52_summary.json').read_text())['pooled'][0]
    ax.errorbar(P['c2_ridge_pooled'], P['c2_biomarker_pooled'],
                xerr=[[P['c2_ridge_pooled'] - P['ridge_ci_lo']], [P['ridge_ci_hi'] - P['c2_ridge_pooled']]],
                yerr=[[P['c2_biomarker_pooled'] - P['ci_lo']], [P['ci_hi'] - P['c2_biomarker_pooled']]],
                fmt='s', ms=4, color=INK, ecolor=INK, elinewidth=0.8, zorder=4)
    ax.text(0.98, 0.03, f"square: pooled over 29 drugs\nridge {P['c2_ridge_pooled']:.2f}, biomarker "
            f"{P['c2_biomarker_pooled']:.2f}\ndifference {P['diff']:+.2f} ({P['diff_ci_lo']:.2f} to {P['diff_ci_hi']:.2f})",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.0, color=INK)
    ax.set_xlim(-0.02, lim); ax.set_ylim(-0.02, lim); hgrid(ax)
    ax.set_xlabel('$c^2$, kernel ridge (selective target)'); ax.set_ylabel('$c^2$, single canonical biomarker')
    ax.scatter([], [], s=12, color=AQUA, label='core biomarkers'); ax.scatter([], [], s=12, color=LIGHT_BLUE, label='extended')
    ax.legend(loc='upper left', fontsize=5.2, handletextpad=0.1, borderaxespad=0.1)


def panel_crossdataset(ax):
    M = pd.read_csv(XL / 'model_medians.csv')
    cls = model_class()
    targets = ['GDSC1', 'GDSC2', 'CTRPv1']
    L = pd.read_csv(XL / 'leakage_floors.csv')
    for k, t in enumerate(targets):
        pair = f'CTRPv2|CC|lnIC50->{t}|CC|lnIC50'
        d = M[(M.pair == pair) & (M.subset == 'all')]
        sel = d[d.stat == 'selB_c'].set_index('model')
        adj = d[d.stat == 'adj_c'].set_index('model')
        fl = L[(L.pair == pair) & (L.subset == 'all') & (L.cellset == 'own') & (L.metric == 'selBOTH') & (L.side == 'cross')
               & L.model.isin(['fitness_only', 'MultiViewRandomForest', 'NaiveTissueMeanPredictor',
                               'NaiveMeanEffectsPredictor'])]['median'].max()
        ax.add_patch(plt.Rectangle((k - 0.42, -0.05), 0.84, fl + 0.05, color='#f1f0ea', zorder=0, lw=0))
        pub = [m for m in sel.index if cls.get(m) in ('classical', 'neural/factorization') and m != 'SVR']
        x = k - 0.22 + RNG.uniform(-0.08, 0.08, len(pub))
        ax.scatter(x, sel.loc[pub, 'median'], s=7, color=LIGHT_GRAY, lw=0, zorder=2)
        for m, dx, col, mk, src in [('EnsembleMF', -0.02, BLUE, 'o', sel), ('EnsembleMF', 0.06, BLUE, 'D', adj),
                                    ('ridge_sel_lm', 0.2, AQUA, 'o', sel), ('ridge_raw_lm', 0.28, AQUA, 'D', adj)]:
            r = src.loc[m]
            ax.errorbar(k + dx, r['median'], yerr=[[r['median'] - r.median_lo], [r.median_hi - r['median']]], fmt=mk,
                        ms=3.2 if mk == 'o' else 2.8, color=col, mfc=col if mk == 'o' else 'white', mec=col, ecolor=col,
                        elinewidth=0.7, zorder=3)
        ref = sel.loc['ref_CTRPv2_measured', 'median']
        ax.plot([k - 0.38, k + 0.38], [ref, ref], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(targets))); ax.set_xticklabels(targets, fontsize=6)
    ax.set_ylim(-0.05, 0.8); hgrid(ax)
    ax.set_ylabel('selective-both correlation in another dataset\n(median over drugs; 95% two-way CI)')
    hs = [plt.Line2D([], [], marker='o', ls='', color=LIGHT_GRAY, ms=3, label='published models'),
          plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.2, label='EnsembleMF'),
          plt.Line2D([], [], marker='D', ls='', mfc='white', mec=BLUE, ms=2.8, label='EnsembleMF, own axis removed'),
          plt.Line2D([], [], marker='o', ls='', color=AQUA, ms=3.2, label='kernel ridge, selective target'),
          plt.Line2D([], [], marker='D', ls='', mfc='white', mec=AQUA, ms=2.8, label='ridge, raw target, own axis removed'),
          plt.Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.8, label="CTRP v2 measurement (replicate)"),
          Patch(facecolor='#f1f0ea', label='drug-agnostic floor')]
    ax.legend(handles=hs, loc='upper left', fontsize=5.0, handletextpad=0.2, borderaxespad=0.1, ncol=2,
              columnspacing=0.6)
    ax.set_title('CTRPv2-trained, unseen cell lines; CurveCurator ln IC$_{50}$', loc='left', color=INK, fontsize=6.2)


# (figure4 of the round-5 layout: replaced by the regrouped figures below)


# ================================================================== Figure 5
HYP = [('H1', 'largest disattenuated fitness-axis correlation', 'H1  largest axis $r$\nwith a discovery screen', (0.70, 'lt')),
       ('H2', 'median selective / median raw cross-screen correlation', 'H2  selective / raw\nagreement', (0.80, 'ge')),
       ('H3', 'difference (reproducible - non-reproducible)', 'H3  reproducible minus\nnon-reproducible', (0.0, 'p')),
       ('H4', 'median own-minus-discovery fitness correlation (gain)', 'H4  gain of own axis\nover discovery axis',
        (0.15, 'ge')),
       ('H5a', 'median model selective correlation (ridge_sel_cons)', 'H5a  frozen model,\nselective $r$', (0.0, 'p')),
       ('H5b', 'median paired difference measurement - model', 'H5b  measurement\nminus model', (0.0, 'p'))]
SCREENS = [('gCSI', 'gCSI'), ('NCI60', 'NCI-60'), ('CCLE', 'CCLE'), ('FIMM', 'FIMM'), ('GRAY', 'GRAY')]


def first_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        s = str(v).strip('[]() ')
        if not s or s == 'nan':
            return np.nan
        return [float(x) for x in s.split(',')]


def registered_outcomes():
    """Registered pass/fail per screen and hypothesis (corrected H4 flag, as in Supplementary Note 2)."""
    out = {}
    for k, _ in SCREENS:
        h = F.jload(f'confirmation_{k}/hypotheses.json')
        h4 = h['H4']['median_difference'] >= 0.15 and (h['H4']['wilcoxon_p_one_sided'] is not None
                                                     and h['H4']['wilcoxon_p_one_sided'] < 0.05)
        out[k] = {'H1': h['H1']['pass'], 'H2': h.get('H2', {}).get('pass'), 'H3': h['H3'].get('pass'), 'H4': h4,
                  'H5a': h['H5'].get('pass_a'), 'H5b': h['H5'].get('pass_b')}
    return out


def panel_estimates(fig, spec):
    E = pd.read_csv(TR / 'confirmation_estimates.csv')
    st = registered_outcomes()
    sub = spec.subgridspec(1, len(HYP), wspace=0.22)
    axes = []
    for j, (h, q, title, (thr, kind)) in enumerate(HYP):
        ax = fig.add_subplot(sub[j]); axes.append(ax)
        for i, (s, lab) in enumerate(SCREENS):
            r = E[(E.screen == s) & (E.hypothesis == h) & (E.quantity == q)]
            if not len(r):
                continue
            r = r.iloc[0]
            lo, hi = ci_str(r.ci95_two_way_boot)
            ok = st[s][h]
            col = BLUE if ok is not None else GRAY
            ax.errorbar(r.estimate, i, xerr=[[r.estimate - lo], [hi - r.estimate]], fmt='o', ms=3.8, color=col,
                        mfc=col if ok else 'white', mec=col, ecolor=col, elinewidth=0.8, zorder=3)
            if ok is None:
                ax.text(hi, i - 0.12, ' n.e.', fontsize=5.0, color=GRAY, va='bottom')
            disc = first_float(r.discovery_value)
            if isinstance(disc, list):
                ax.plot(disc, [i - 0.3] * 2, color=INK2, lw=1.0, zorder=2)
            elif np.isfinite(disc):
                ax.scatter(disc, i - 0.3, marker='D', s=9, facecolor='white', edgecolor=INK2, lw=0.6, zorder=2)
            reh = first_float(r.rehearsal_GDSC2_value)
            if isinstance(reh, list):
                reh = max(reh)
            if np.isfinite(reh):
                ax.scatter(reh, i + 0.3, marker='x', s=9, color=GRAY, lw=0.7, zorder=2)
            if h == 'H5a':
                fl = E[(E.screen == s) & (E.hypothesis == 'H5a-floor') & E.quantity.str.startswith('FLOOR')]
                if len(fl):
                    f = fl.iloc[0]; flo, fhi = ci_str(f.ci95_two_way_boot)
                    ax.errorbar(f.estimate, i + 0.3, xerr=[[f.estimate - flo], [fhi - f.estimate]], fmt='^', ms=2.8,
                                color=GRAY, ecolor=GRAY, elinewidth=0.6, zorder=2)
        if kind in ('lt', 'ge'):
            ax.axvline(thr, color=ORANGE, lw=0.8, ls=(0, (3, 1.5)))
        ax.axvline(0, color=GRAY, lw=0.5)
        ax.set_yticks(range(len(SCREENS)))
        ax.set_yticklabels([s[1] for s in SCREENS] if j == 0 else [], fontsize=6)
        ax.set_ylim(len(SCREENS) - 0.45, -0.6)
        ax.set_title(title, fontsize=5.6, color=INK, loc='left')
        ax.tick_params(axis='x', labelsize=5.4)
        xgrid(ax)
    return axes


PATIENT_P = {  # registered unit-level P (H6/H8; H7/H9) and clustered re-analysis P (revision_transfer)
    'BeatAML, samples (registered)': ('H6', 'H7'), 'BeatAML, patients': ('H6', 'H7'),
    'PDX, models (registered)': ('H8', 'H9'), 'PDX, tumours': ('H8', 'H9'),
    'Organoids, patients': ('H10', 'H11')}


def patient_pvalues():
    H = pd.read_csv(TR / 'holm_registered_tests.csv').set_index('hypothesis')
    reg = {h: H.loc[h, 'registered_p'] for h in ['H6', 'H7', 'H8', 'H9']}
    unit = {h: H.loc[h, 'unit_level_p'] for h in ['H6', 'H7', 'H8', 'H9']}
    fmt = lambda p: f'{p:.3f}' if p >= 0.001 else f'{p:.4f}'
    return {'BeatAML, samples (registered)': f"H6 P = {fmt(reg['H6'])}\nH7 P = {fmt(reg['H7'])}",
            'BeatAML, patients': f"H6 P = {fmt(unit['H6'])}\nH7 P = {fmt(unit['H7'])}",
            'PDX, models (registered)': f"H8 P = {fmt(reg['H8'])}\nH9 P = {fmt(reg['H9'])}",
            'PDX, tumours': f"H8 P = {fmt(unit['H8'])}\nH9 P = {fmt(unit['H9'])}",
            'Organoids, patients\n(corrected orientation)': pdm_pvalues(fmt)}


def pdm_pvalues(fmt):
    T = pd.read_csv(TR / 'pdm_tests.csv')
    T = T[T.analysis.str.startswith('corrected')].set_index('hypothesis').permutation_p
    p10 = f"P \u2264 {fmt(T['H10'])}" if T['H10'] <= 1.0001e-4 else f"P = {fmt(T['H10'])}"   # permutation floor
    return f"H10 {p10}\nH11 P = {fmt(T['H11'])}"


def panel_patient(ax):
    """Pooled selective transfer (random effects over drugs), registered unit and tumour or patient level."""
    P = pd.read_csv(TR / 'pdx_meta_analysis.csv')
    B = pd.read_csv(TR / 'beataml_meta_analysis.csv')
    rows = []
    for an, lab in [('registered sample level', 'BeatAML, samples (registered)'), ('patient level', 'BeatAML, patients')]:
        for cls in ['reproducible', 'non_reproducible']:
            r = B[(B.analysis == an) & (B['class'] == cls)].iloc[0]
            lo, hi = ci_str(r.ci95_r_hksj)
            rows.append((lab, cls, r.pooled_r, lo, hi, *ci_str(r.pi95_r)))
    for an, grp, lab in [('registered model level', 'model', 'PDX, models (registered)'), ('T1 tumour level', 'G1', 'PDX, tumours')]:
        for cls in ['reproducible', 'non_reproducible']:
            r = P[(P.analysis == an) & (P.grouping == grp) & (P['class'] == cls)].iloc[0]
            lo, hi = ci_str(r.ci95_r_hksj)
            rows.append((lab, cls, r.pooled_r, lo, hi, *ci_str(r.pi95_r)))
    M = pd.read_csv(TR / 'pdm_meta_analysis.csv')
    for cls in ['reproducible', 'non_reproducible']:
        r = M[M.analysis.str.startswith('corrected') & (M.scope == 'all') & (M['class'] == cls)].iloc[0]
        lo, hi = ci_str(r.re_ci95_hksj)
        rows.append(('Organoids, patients\n(corrected orientation)', cls, r.re_pooled_r, lo, hi, np.nan, np.nan))
    labs = list(dict.fromkeys(r[0] for r in rows))
    for lab, cls, m, lo, hi, plo, phi in rows:
        y = labs.index(lab) + (-0.15 if cls == 'reproducible' else 0.15)
        col, face = (BLUE, BLUE) if cls == 'reproducible' else (GRAY, LIGHT_GRAY)
        if np.isfinite(plo):   # 95% prediction interval for a new drug
            ax.plot([plo, phi], [y, y], color=col, lw=0.5, alpha=0.45, zorder=2)
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt='o', ms=3.6, color=col, mfc=face, mec=col, ecolor=col,
                    elinewidth=0.8, zorder=3)
    pv = patient_pvalues()
    for k, lab in enumerate(labs):
        ax.text(0.50, k, pv[lab], va='center', fontsize=5.0, color=INK2)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5.8)
    ax.set_ylim(len(labs) - 0.5, -0.5)
    ax.set_xlabel('pooled selective transfer $r$\n(random effects over drugs, 95% CI)')
    xgrid(ax); ax.set_xlim(-0.32, 0.78); ax.set_xticks([-0.2, 0, 0.2, 0.4])
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.6, label='reproducible drugs'),
                       plt.Line2D([], [], marker='o', ls='', color=GRAY, mfc=LIGHT_GRAY, ms=3.6,
                                  label='non-reproducible drugs'),
                       plt.Line2D([], [], color=BLUE, lw=0.5, alpha=0.45, label='95% prediction interval')],
              loc='lower left', bbox_to_anchor=(0.0, 1.0),
              ncol=3, fontsize=5.4, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.2)


def beataml_labels(top):
    """Names of the strongest BeatAML transfers, marking MEK inhibitors (curated class) and small sample counts."""
    sc = pd.read_csv(RESULTS / 'confirmation_BeatAML' / 'scores.csv')
    best = sc[(sc.family == 'sel') & (sc['class'] == 'reproducible')].sort_values('sel', ascending=False).head(top)
    cls = pd.read_csv(TR / 'curated_classes.csv').set_index('compound').new_class
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound').name
    gen = F.generic_names()
    out = {}
    for r in best.itertuples():
        nm = gen.get(r.compound, str(comp.get(r.compound, '')))
        if cls.get(r.compound) == 'MAPK (RAF/MEK/ERK)':
            nm += '$^\\dagger$'      # MEK inhibitor (caption)
        if r.n < 100:
            nm += f' ({r.n})'
        out[str(comp.get(r.compound, ''))] = nm
    return out


R4 = RESULTS / 'revision_round4'
TPD, SIMD = R4 / 'transfer_prediction', R4 / 'simulation'
TPB = R4 / 'transfer_prediction_r4' if (R4 / 'transfer_prediction_r4' / 'fig_b_pooled_correlations.csv').exists() else TPD
# pooled correlations: round-4 primary units (DrEval's models and naive baselines; 100c)


def _lb_transfer():
    A = pd.read_csv(TPD / 'fig_a_model_scatter.csv')
    A = A[A.benchmark == 'leaderboard']
    g = A.groupby('model')
    return pd.DataFrame({'raw': g.raw_median.first(), 'S': g.S.first(), 'selB': g.selB_c.mean(),
                         'agn': g.drug_agnostic.first().astype(str).str.lower().isin(['true', '1'])})


def panel_transfer_scatter(ax, xcol, xlab, rho_key):
    """Released leaderboard models: a within-CTRP v2 statistic against the selectivity they keep in the four other
    datasets (mean selective-both over the CurveCurator ln IC50 targets; 100a_transfer_prediction.py)."""
    D = _lb_transfer()
    B = pd.read_csv(TPB / 'fig_b_pooled_correlations.csv')
    r = B[(B.group == 'leaderboard') & (B.model_set == 'all') & (B.outcome == 'selB_c') & (B.method == 'spearman')
          & (B.interval == 'full') & (B.predictor == rho_key)].iloc[0]
    for agn, col in [(False, BLUE), (True, BASE_COL)]:
        d = D[D.agn == agn]
        ax.scatter(d[xcol], d.selB, s=12, color=col, lw=0, zorder=3, label='drug-agnostic' if agn else 'learned')
    em = D.loc['EnsembleMF']
    ax.annotate('EnsembleMF', (em[xcol], em.selB), xytext=(4, -2), textcoords='offset points', fontsize=5.0, color=INK2)
    ax.text(0.03, 0.97, f"Spearman {r.r_pooled:.2f}\n({r.lo:.2f} to {r.hi:.2f})", transform=ax.transAxes,
            fontsize=5.2, color=INK2, va='top')
    ax.set_xlabel(xlab)
    ax.set_ylabel('selective-both in other datasets\n(mean over 4 targets)')
    hgrid(ax)


PRED_LAB = {'raw_median': 'raw correlation', 'dreval_norm': 'normalized', 'S': 'selective capture'}
GROUPS = [('leaderboard', 'leaderboard (21)'), ('retrain:GDSC2_AUC', 'retrained, GDSC2 AUC'),
          ('retrain:GDSC2_LNIC50', 'retrained, GDSC2 ln IC$_{50}$'), ('improve', 'IMPROVE (pair splits)'),
          ('overall', 'three benchmarks')]


def panel_transfer_forest(ax):
    """Pooled Spearman correlation across models between a within-dataset statistic and the same predictions'
    accuracy in other datasets: target selective-both (filled) and target raw correlation (open)."""
    B = pd.read_csv(TPB / 'fig_b_pooled_correlations.csv')
    B = B[(B.model_set == 'all') & (B.method == 'spearman') & (B.interval == 'full')]
    cols = {'raw_median': GRAY, 'dreval_norm': AQUA, 'S': BLUE}
    y, ticks, labs = 0, [], []
    for g, glab in GROUPS:
        ys = []
        for pr in ['raw_median', 'dreval_norm', 'S']:
            for out, dy, filled in [('selB_c', -0.12, True), ('raw_median_c', 0.12, False)]:
                r = B[(B.group == g) & (B.predictor == pr) & (B.outcome == out)]
                if not len(r):
                    continue
                r = r.iloc[0]
                c = cols[pr]
                ax.errorbar(r.r_pooled, y + dy, xerr=[[r.r_pooled - r.lo], [r.hi - r.r_pooled]], fmt='o', ms=2.6,
                            color=c, ecolor=c, elinewidth=0.6, mfc=c if filled else 'white', mec=c, zorder=3)
            if len(B[(B.group == g) & (B.predictor == pr)]):
                ys.append(y); y += 1
        ticks.append(np.mean(ys)); labs.append(glab)
        ax.axhline(y - 0.5, color=LIGHT_GRAY, lw=0.4)
        y += 0.3
    ax.axvline(0, color=GRAY, lw=0.5)
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=5.0)
    ax.set_ylim(y - 0.6, -0.7); ax.set_xlim(-1.0, 1.05); xgrid(ax)
    ax.set_xlabel('Spearman correlation across models\n(pooled over targets; 95% CI)')
    h = [plt.Line2D([], [], marker='o', ls='', color=cols[k], ms=2.8, label=v) for k, v in PRED_LAB.items()]
    h += [plt.Line2D([], [], marker='o', ls='', color=INK2, ms=2.8, label='outcome: selective-both'),
          plt.Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, ms=2.8, label='outcome: raw accuracy')]
    ax.legend(handles=h, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=5.0, frameon=False,
              handletextpad=0.1, columnspacing=0.6, labelspacing=0.2)


def panel_sim_equal(ax):
    """Three simulated predictors with the same raw per-drug correlation (150 drugs x 500 cells; 300 replicates)."""
    R = pd.read_csv(SIMD / 'fig_1_2_recovery_coverage.csv')
    R = R[(R.D == 150)].set_index(['predictor', 'stat'])
    preds = [('eq_agnostic', 'drug-agnostic'), ('eq_mixed', 'mixed'), ('eq_selective', 'mostly selective')]
    stats = [('median_raw', 'raw correlation', GRAY), ('A', 'axis capture', LIGHT_BLUE), ('S', 'selective capture', BLUE)]
    w = 0.25
    for i, (p_, lab) in enumerate(preds):
        for j, (st, slab, c) in enumerate(stats):
            r = R.loc[(p_, st)]
            ax.bar(i + (j - 1) * w, r.mean_estimate, width=w, color=c, zorder=3, label=slab if i == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([p[1] for p in preds], fontsize=5.4)
    ax.set_ylabel('estimate (mean of 300 replicates)')
    hgrid(ax)
    ax.legend(loc='upper right', fontsize=5.0, frameon=False, handlelength=1.0)


def panel_sim_recovery(ax):
    """Estimated against true axis and selective capture for every simulated predictor."""
    R = pd.read_csv(SIMD / 'fig_1_2_recovery_coverage.csv')
    for st, c in [('A', LIGHT_BLUE), ('S', BLUE)]:
        for D_, mk in [(150, 'o'), (30, '^')]:
            d = R[(R.stat == st) & (R.D == D_)]
            ax.scatter(d.truth_design, d.mean_estimate, s=10, marker=mk, color=c, lw=0, zorder=3)
    lim = (-0.1, 2.1)
    ax.plot(lim, lim, color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    cov = R[R.stat.isin(['A', 'S'])].groupby('stat').coverage_design.agg(['min', 'max'])
    ax.text(0.03, 0.97, f"panel-level 95% coverage\nA {100 * cov.loc['A', 'min']:.0f}-{100 * cov.loc['A', 'max']:.0f}%, "
            f"S {100 * cov.loc['S', 'min']:.0f}-{100 * cov.loc['S', 'max']:.0f}%", transform=ax.transAxes, fontsize=5.0,
            color=INK2, va='top')
    ax.set_xlabel('true capture'); ax.set_ylabel('estimated capture')
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=LIGHT_BLUE, ms=3, label='axis capture'),
                       plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3, label='selective capture'),
                       plt.Line2D([], [], marker='o', ls='', color=INK2, ms=3, label='150 drugs x 500 cells'),
                       plt.Line2D([], [], marker='^', ls='', color=INK2, ms=3, label='30 drugs x 100 cells')],
              loc='lower right', fontsize=5.0, frameon=False, handletextpad=0.1, labelspacing=0.2)
    hgrid(ax)


def panel_sim_leakage(ax):
    """Axis leaking into selective capture as the axis gets less reliable (fewer drugs), against the prediction
    A m/(1-m) (1-rho)/rho; vertical line, the leaderboard axis's split-half reliability."""
    F = pd.read_csv(SIMD / 'fig_5_axis_reliability_leakage.csv')
    for pr, c, lab in [('agnostic', BASE_COL, 'drug-agnostic'), ('axis_heavy', GRAY, 'axis-heavy'),
                       ('balanced', BLUE, 'balanced')]:
        d = F[F.predictor == pr].sort_values('axis_split_half')
        ax.plot(d.axis_split_half, d.predicted_leakage_latent_rel, color=c, lw=0.7, zorder=2)
        ax.scatter(d.axis_split_half, d.leakage, s=8, color=c, lw=0, zorder=3, label=lab)
    X = pd.read_csv(SIMD / 'realdata_leakage_correction.csv')
    rho = X[X.benchmark == 'leaderboard'].axis_split_half.median()
    ax.axvline(rho, color=INK2, lw=0.5, ls=(0, (2, 2)))
    ax.text(rho - 0.01, 0.215, 'leaderboard\naxis', fontsize=5.0, color=INK2, ha='right', va='top')
    ax.set_xlabel('axis split-half reliability'); ax.set_ylabel('leakage into selective capture')
    ax.set_xlim(0.55, 1.0); ax.set_ylim(-0.01, 0.22)
    ax.legend(loc='center right', bbox_to_anchor=(0.97, 0.55), fontsize=5.0, frameon=False, handletextpad=0.1)
    hgrid(ax)


# (figure5 of the round-5 layout: replaced by the regrouped figures below)


def ed_figure6():
    """Extended Data Fig. 6 (formerly Extended Data Fig. 2): the frozen confirmatory analyses, reported as estimates."""
    fig = plt.figure(figsize=(W, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.62)
    axes = panel_estimates(fig, gs[0])
    panel_label(axes[0], 'a', x=-0.55, y=1.12)
    hs = [plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.6, label='estimate, 95% drug x cell bootstrap'),
          plt.Line2D([], [], marker='o', ls='', mfc='white', mec=BLUE, ms=3.6, label='registered test not met'),
          plt.Line2D([], [], marker='D', ls='', mfc='white', mec=INK2, ms=3.0, label='discovery (line: range)'),
          plt.Line2D([], [], marker='x', ls='', color=GRAY, ms=3.4, label='GDSC2 rehearsal'),
          plt.Line2D([], [], marker='^', ls='', color=GRAY, ms=3.0, label='drug-agnostic floor'),
          plt.Line2D([], [], color=ORANGE, ls=(0, (3, 1.5)), lw=0.8, label='registered threshold')]
    fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.515), ncol=6, fontsize=5.2, handletextpad=0.2,
               columnspacing=0.8)
    bot = gs[1].subgridspec(1, 4, width_ratios=[1.3, 0.68, 0.68, 0.68], wspace=1.35)
    ax = fig.add_subplot(bot[0]); panel_label(ax, 'b', x=-0.62, y=1.1); panel_patient(ax)
    for k, (folder, lab, top) in enumerate([('confirmation_BeatAML', 'BeatAML samples', 5),
                                             ('confirmation_PDX_Bruna', 'PDX-derived cells', 3),
                                             ('confirmation_PDM_corrected', 'organoids', 0)]):
        ax = fig.add_subplot(bot[k + 1]); panel_label(ax, 'cde'[k], x=-0.45, y=1.1)
        show = beataml_labels(top) if folder == 'confirmation_BeatAML' else (None if top else {})
        F.panel_patient_transfer(ax, folder, show, f'selective transfer per drug\n({lab})', top=top,
                                 gap=0.06 if folder != 'confirmation_BeatAML' else 0.045)
        ax.set_xticklabels(['non-\nrepr.', 'inter-\nmed.', 'repr.'], fontsize=5.8)
    save(fig, 'extended_data_fig6_confirmation')


# ================================================================== Figure 3: second benchmark (IMPROVE) and retraining
IM = RESULTS / 'revision_improve' / 'no_split_replicates'   # primary since round 2: replicate pairs split across folds removed
IM_ORDER = ['CTRPv2', 'GDSCv1', 'GDSCv2', 'CCLE', 'gCSI']
IM_LABEL = {'CTRPv2': 'CTRP\nv2', 'GDSCv1': 'GDSC\n1', 'GDSCv2': 'GDSC\n2', 'CCLE': 'CCLE', 'gCSI': 'gCSI'}
IM_MODELS = ['deepcdr', 'deepttc', 'graphdrp', 'hidra', 'lgbm', 'tcnns', 'uno']


def panel_improve_share(ax):
    """Axis share of each model's per-drug accuracy within each dataset, against a perfect predictor."""
    W = pd.read_csv(IM / 'within_summary.csv')
    B = pd.read_csv(IM / 'within_pooled_bootstrap.csv')
    for k, d in enumerate(IM_ORDER):
        x = W[(W.dataset == d) & W.model.isin(IM_MODELS)]
        ax.scatter(k + RNG.uniform(-0.17, 0.17, len(x)), x.axis_share, s=10, color=LIGHT_BLUE, lw=0, zorder=2)
        b = B[(B.dataset == d) & (B.stat == 'fitness_share_source')].iloc[0]
        ax.errorbar(k + 0.28, b.estimate, yerr=[[b.estimate - b.lo], [b.hi - b.estimate]], fmt='o', ms=3.6, color=BLUE,
                    elinewidth=0.9, zorder=3)
        o = W[W.dataset == d].oracle_share.iloc[0]
        ax.plot([k - 0.3, k + 0.38], [o, o], color=ORANGE, lw=1.1, zorder=2)
    ax.set_xticks(range(len(IM_ORDER))); ax.set_xticklabels([IM_LABEL[d] for d in IM_ORDER], fontsize=5.6)
    ax.set_ylim(0, 1); ax.set_ylabel('axis share of per-drug accuracy\n($\\Sigma$ axis term / $\\Sigma$ raw)')
    hgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=LIGHT_BLUE, ms=3.4, label='each of 7 models'),
                       plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.6, label='pooled, 95% CI'),
                       plt.Line2D([], [], color=ORANGE, lw=1.1, label='perfect predictor')],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.3, handletextpad=0.2, columnspacing=0.6,
              borderaxespad=0.2)


def panel_improve_floor(ax):
    """Median per-drug raw correlation of each model against the drug-agnostic floor (the cell line's mean)."""
    W = pd.read_csv(IM / 'within_summary.csv')
    for k, d in enumerate(IM_ORDER):
        x = W[(W.dataset == d) & W.model.isin(IM_MODELS)]
        ax.scatter(k + RNG.uniform(-0.17, 0.17, len(x)), x.median_raw, s=10, color=BLUE, lw=0, zorder=2)
        f = W[(W.dataset == d) & (W.model == 'floor_cell_mean')].median_raw.iloc[0]
        ax.plot([k - 0.3, k + 0.3], [f, f], color=GRAY, lw=1.6, zorder=1)
        ax.scatter([k + 0.3], [f], marker='^', s=14, color=GRAY, zorder=3)
    ax.set_xticks(range(len(IM_ORDER))); ax.set_xticklabels([IM_LABEL[d] for d in IM_ORDER], fontsize=5.6)
    ax.set_ylim(0, 0.75); ax.set_ylabel('median per-drug raw $r$')
    hgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.4, label='each of 7 models'),
                       plt.Line2D([], [], marker='^', color=GRAY, ms=3.6, lw=1.6,
                                  label='drug-agnostic floor (cell mean)')],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=5.3, handletextpad=0.5, columnspacing=0.6,
              borderaxespad=0.2, handlelength=2.2)


def panel_improve_transfer(ax):
    """Selective-both of identical predictions in the target against the agreement of the two datasets' measurements."""
    S = pd.read_csv(IM / 'selective_both_change.csv')
    S = S[S.axis_panel == 'own_panel']
    lim = 0.6
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    sz = 6 + 1.2 * np.sqrt(S.drugs)
    ax.scatter(S.measurement_cross_mean, S.models_cross_mean_same_drugs, s=sz, color=BLUE, alpha=0.85, lw=0, zorder=3)
    ax.scatter(S.measurement_cross_mean, S.within_mean, s=sz, facecolor='white', edgecolor=LIGHT_BLUE, lw=0.7, zorder=2)
    for r in S.itertuples():
        ax.plot([r.measurement_cross_mean] * 2, [r.models_cross_mean_same_drugs, r.within_mean], color=LIGHT_GRAY,
                lw=0.5, zorder=1)
    # two-way bootstrap 95% intervals of the models' cross selective-both (84_source_reference.py)
    Q = pd.read_csv(RESULTS / 'revision_round2' / 'source_reference' / 'source_ratio_by_target.csv')
    Q = Q[(Q.benchmark == 'improve') & Q.group.str.startswith('pooled') & (Q.family == 'own_panel')]
    Q = Q.set_index(['source', 'target'])
    for r in S.itertuples():
        if (r.source, r.target) in Q.index:
            q = Q.loc[(r.source, r.target)]
            ax.plot([r.measurement_cross_mean] * 2, [q.model_cross_selBOTH_lo, q.model_cross_selBOTH_hi], color=BLUE,
                    lw=0.6, alpha=0.6, zorder=2)
    ratio = Q.ratio_to_measurement
    ax.text(0.98, 0.03, f'models / measurement\nmedian {ratio.median():.2f}\n(pairs {ratio.min():.2f}\u2013{ratio.max():.2f})',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=5.2, color=INK2)
    ax.set_xlim(0.2, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('selective-both of the source measurement\nscored in the target (agreement)')
    ax.set_ylabel('selective-both of identical predictions')
    hgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', mfc='white', mec=LIGHT_BLUE, ms=3.6, label='in the source'),
                       plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.6, label='in the target')],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.3, handletextpad=0.2, columnspacing=0.6,
              borderaxespad=0.2)


RT_ORDER = ['CTRPv1', 'CTRPv2', 'CCLE', 'GDSC1', 'GDSC2', 'gCSI']
RT_IM = {'CTRPv2': 'CTRPv2', 'CCLE': 'CCLE', 'GDSC1': 'GDSCv1', 'GDSC2': 'GDSCv2', 'gCSI': 'gCSI'}
RT_LEARNED = ['EnsembleMF', 'GradientBoosting', 'SimpleNeuralNetwork', 'RandomForest']
RT_TARGETS = {'CTRPv2_LNIC50': ['CTRPv1', 'GDSC1', 'GDSC2', 'CCLE'], 'GDSC2_AUC': ['CTRPv2', 'CTRPv1', 'GDSC1', 'CCLE']}
RT_COLOR = {'CTRPv2_LNIC50': BLUE, 'GDSC2_AUC': AQUA}
RT_LABEL = {'CTRPv2_LNIC50': 'trained on CTRP v2 ln IC$_{50}$', 'GDSC2_AUC': 'trained on GDSC2 AUC'}


def panel_axis_weight(ax):
    """Perfect-predictor axis share (mean r_zg^2) of each dataset and endpoint."""
    F = pd.read_csv(RT / 'dataset_fitness.csv').set_index(['dataset', 'endpoint'])
    W = pd.read_csv(IM / 'within_summary.csv').groupby('dataset').oracle_share.first()
    for k, d in enumerate(RT_ORDER):
        ax.plot([0, 0.35], [k, k], color=GRID, lw=0.5, zorder=0)
        for ep, dy, fc in [('lnic50', -0.2, BLUE), ('auc', 0.0, 'white')]:
            if (d, ep) in F.index:
                ax.scatter([F.loc[(d, ep), 'mean_fitness_r2']], [k + dy], marker='o', s=16, facecolor=fc,
                           edgecolor=BLUE, lw=0.8, zorder=3)
        if RT_IM.get(d) in W.index:
            ax.scatter([W[RT_IM[d]]], [k + 0.2], marker='^', s=16, color=AQUA, lw=0, zorder=3)
    ax.set_yticks(range(len(RT_ORDER)))
    ax.set_yticklabels(['CTRP v1', 'CTRP v2', 'CCLE', 'GDSC1', 'GDSC2', 'gCSI'], fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.35); ax.set_ylim(len(RT_ORDER) - 0.5, -0.5)
    ax.set_xlabel('perfect predictor\'s axis share\n(mean $r_{zg}^2$)')
    xgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.6, label='ln IC$_{50}$'),
                       plt.Line2D([], [], marker='o', ls='', mfc='white', mec=BLUE, ms=3.6, label='AUC'),
                       plt.Line2D([], [], marker='^', ls='', color=AQUA, ms=3.6, label='IMPROVE')],
              loc='lower left', bbox_to_anchor=(-0.05, 1.0), ncol=3, fontsize=5.0, handletextpad=0.1,
              columnspacing=0.5, borderaxespad=0.2)


def panel_overweight(ax):
    """Each learned model's axis share against the perfect predictor's, in every benchmark and training setting."""
    R = pd.read_csv(RT / 'within_summary.csv')
    R = R[R.panel == 'main']
    lb = R[(R.setting == 'CTRPv2_LNIC50') & R.model.str.startswith('LB:') & ~R.model.str.contains('Naive')
           & (R.median_raw > 0.03)]
    gd = R[(R.setting == 'GDSC2_AUC') & R.model.isin(RT_LEARNED)]
    W = pd.read_csv(IM / 'within_summary.csv')
    W = W[W.model.isin(IM_MODELS)]
    xs = np.linspace(0, 0.35, 10)
    for k, ls in [(1, '-'), (2, (0, (3, 2))), (4, (0, (1, 1.5)))]:
        ax.plot(xs, k * xs, color=ORANGE if k == 1 else GRAY, lw=0.8 if k == 1 else 0.6, ls=ls, zorder=1)
        xt = min(0.33, 0.98 / k) if k > 1 else 0.33
        ax.text(xt, k * xt + 0.012, '1$\\times$' if k == 1 else f'{k}$\\times$', fontsize=5.0,
                color=ORANGE if k == 1 else INK2, ha='right', va='bottom')
    ax.scatter(lb.oracle_mean_rzg2 + RNG.uniform(-0.006, 0.006, len(lb)), lb.mediated_share, s=9, color=BLUE, lw=0,
               alpha=0.85, zorder=3)
    ax.scatter(W.oracle_share, W.axis_share, s=9, marker='^', color=AQUA, lw=0, alpha=0.85, zorder=3)
    ax.scatter(gd.oracle_mean_rzg2, gd.mediated_share, s=16, marker='s', color=INK, lw=0, zorder=4)
    ax.set_xlim(0, 0.35); ax.set_ylim(0, 1.05)
    ax.set_xlabel('perfect predictor\'s axis share')
    ax.set_ylabel('learned model\'s axis share')
    hgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=BLUE, ms=3.2, label=f'DrEval ({len(lb)})'),
                       plt.Line2D([], [], marker='^', ls='', color=AQUA, ms=3.4, label=f'IMPROVE ({len(W)})'),
                       plt.Line2D([], [], marker='s', ls='', color=INK, ms=3.4,
                                  label=f'GDSC2 AUC,\nretrained ({len(gd)})'),
                       plt.Line2D([], [], color=ORANGE, lw=0.8, label='perfect\npredictor')],
              loc='upper left', ncol=1, fontsize=5.0, handletextpad=0.3, borderaxespad=0.1, frameon=False,
              handlelength=1.2)


def panel_capture(ax):
    """Ratios to a perfect predictor's axis and selective terms that each leave-cell-line-out model recovers
    (78_capture.py), with each learned model's selective capture when tissue is also partialled out (91_lineage.py;
    tick) and the tissue-by-drug baseline (x). IMPROVE's pair-split models are kept out (Supplementary Table)."""
    C = pd.read_csv(RESULTS / 'revision_round2' / 'capture' / 'capture_by_model.csv')
    L = pd.read_csv(RESULTS / 'revision_round2' / 'lineage' / 'lineage_by_model.csv')
    agn = C.drug_agnostic.notna()   # 'exact' or 'within split (additive)'
    lb = C[(C.benchmark == 'leaderboard') & ~C.naive & ~C.reference_pseudo & ~C.model.str.startswith('ridge')
           & (C.model != 'SVR')]          # the 21 published models with per-drug signal, as in the text
    om = lb[lb.model == 'fitness_only']   # this study's omics-predicted axis, shown separately
    lb = lb[lb.model != 'fitness_only']
    sets = [(lb[~agn.loc[lb.index]], 'o', BLUE, 9, f'published, learned ({(~agn.loc[lb.index]).sum()})', 'leaderboard',
             None),
            (lb[agn.loc[lb.index]], 'o', 'white', 9, f'published, drug-agnostic ({agn.loc[lb.index].sum()})', None, None),
            (om, 'P', 'omics', 14, 'omics-predicted axis', None, None)]
    for st, mk, lab in [('GDSC2_AUC', 's', 'retrained, GDSC2 AUC'), ('GDSC2_LNIC50', 'D', 'GDSC2 ln IC$_{50}$'),
                        ('CTRPv2_AUC', 'v', 'CTRP v2 AUC')]:
        x = C[(C.benchmark == 'retrain') & (C.setting == st) & C.model.isin(RT_LEARNED)]
        if len(x):
            sets.append((x, mk, INK, 13, lab, 'retrain', st))
    ax.plot([0, 2.0], [0, 2.0], color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(0.2, 0.285, 'equal\ncapture', fontsize=5.0, color=INK2, ha='right', va='top')
    h_main, h_rt = [], []
    for d, mk, col, sz, lab, bench, st in sets:
        if bench is not None:            # the same models' selective capture given the axis and tissue
            q = L[(L.benchmark == bench) & (L.kind == 'learned')]
            q = (q[q.setting == st] if st else q).set_index('model').reindex(d.model)
            ok = q.selective_capture_lin.notna().values
            xs, y0, y1 = d.axis_capture.values[ok], d.selective_capture.values[ok], q.selective_capture_lin.values[ok]
            ax.vlines(xs, y1, y0, color=LIGHT_GRAY, lw=0.5, zorder=2)
            ax.scatter(xs, y1, s=9, marker='_', color=BLUE if bench == 'leaderboard' else INK, lw=0.7, zorder=3)
        if col == 'white':
            ax.scatter(d.axis_capture, d.selective_capture, s=sz, marker=mk, facecolor='white', edgecolor=BASE_COL,
                       lw=0.7, zorder=4)
            h_main.append(plt.Line2D([], [], marker=mk, ls='', mfc='white', mec=BASE_COL, ms=3.0, label=lab))
        elif col == 'omics':
            ax.scatter(d.axis_capture, d.selective_capture, s=sz, marker=mk, color=BASE_COL, lw=0, zorder=4)
            h_main.append(plt.Line2D([], [], marker=mk, ls='', color=BASE_COL, ms=3.4, label=lab))
        else:
            ax.scatter(d.axis_capture, d.selective_capture, s=sz, marker=mk, color=col, lw=0, alpha=0.9, zorder=4)
            (h_main if bench == 'leaderboard' else h_rt).append(
                plt.Line2D([], [], marker=mk, ls='', color=col, ms=2.8, label=lab))
    T = L[(L.kind == 'tissue-by-drug baseline') & (L.benchmark != 'improve')
          & ~((L.benchmark == 'retrain') & (L.setting == 'CTRPv2_LNIC50'))]    # the latter duplicates the released one
    ax.scatter(T.axis_capture, T.selective_capture, s=14, marker='x', color=BASE_COL, lw=0.8, zorder=5)
    h_main.append(plt.Line2D([], [], marker='_', ls='', color=INK, ms=4, mew=0.8, label='given tissue'))
    h_rt.append(plt.Line2D([], [], marker='x', ls='', color=BASE_COL, ms=3.0, mew=0.8, label='tissue-by-drug mean'))
    ax.vlines(1.0, -0.04, 0.245, color=GRAY, lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    ax.text(1.02, 0.236, 'beyond 1: more aligned with\nthe axis than the profiles', fontsize=5.0, color=INK2, ha='left',
            va='top')
    ax.annotate('perfect predictor (1, 1)', xy=(1.0, 0.30), xytext=(1.0, 0.262), ha='center', va='center',
                fontsize=5.0, color=INK, arrowprops=dict(arrowstyle='->', color=INK, lw=0.6))
    P = pd.read_csv(RESULTS / 'revision_round2' / 'capture' / 'capture_pooled.csv')
    pool = P[(P.benchmark == 'leaderboard') & (P.group == 'published 21 (paper)')].iloc[0]
    ax.plot([pool.axis_capture], [pool.selective_capture], marker='*', ms=7, color=BLUE, mec='white', mew=0.5, ls='',
            zorder=6)
    ax.annotate(f'pooled over the {int(pool.models)}\npublished models:\n{100 * pool.axis_capture:.0f}% of the axis term,\n'
                f'{100 * pool.selective_capture:.0f}% of the selective term', xy=(pool.axis_capture, pool.selective_capture),
                xytext=(1.03, 0.168), fontsize=5.0, color=INK, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5, shrinkB=3))
    ax.set_xlim(-0.08, 1.45); ax.set_ylim(-0.04, 0.30)
    ax.axhline(0, color=GRAY, lw=0.4, zorder=1)
    ax.set_xlabel('axis capture\n(ratio to perfect predictor\'s axis term)')
    ax.set_ylabel('selective capture\n(ratio to perfect predictor\'s selective term)')
    hgrid(ax)
    ax.legend(handles=h_main + h_rt, loc='lower left', bbox_to_anchor=(-0.04, 1.0), fontsize=5.0, handletextpad=0.1,
              borderaxespad=0.1, frameon=False, labelspacing=0.15, ncol=2, columnspacing=0.5)


def panel_retrain_terms(ax):
    """Change of the axis and selective terms when identical predictions are scored in another dataset."""
    P = pd.read_csv(RT / 'paired_pooled.csv')
    y, labels, k = [], [], 0
    for st in ['CTRPv2_LNIC50', 'GDSC2_AUC']:
        for t in RT_TARGETS[st]:
            r = P[(P.setting == st) & (P.target == t)].iloc[0]
            for col, dy, c, mk in [('delta_mediated', -0.16, GRAY, 'd'), ('delta_selective_term', 0.16, ORANGE, 'o')]:
                v, lo, hi = r[col], r[col + '_lo'], r[col + '_hi']
                ax.errorbar(v, k + dy, xerr=[[v - lo], [hi - v]], fmt=mk, ms=3.2, color=c, ecolor=c, elinewidth=0.7,
                            zorder=3)
            labels.append(t); y.append(k); k += 1
        k += 0.6
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.0)
    ax.invert_yaxis()
    for st, (a, b) in zip(['CTRPv2_LNIC50', 'GDSC2_AUC'], [(y[0], y[3]), (y[4], y[7])]):
        ax.text(1.02, (a + b) / 2, RT_LABEL[st].replace('trained on ', 'trained on\n'), transform=ax.get_yaxis_transform(),
                rotation=270, va='center', ha='left', fontsize=5.0, color=RT_COLOR[st])
    ax.set_xlim(-0.22, 0.12); xgrid(ax)
    ax.set_xlabel('change in mean per-drug term\n(other $-$ training dataset)')
    ax.legend(handles=[plt.Line2D([], [], marker='d', ls='', color=GRAY, ms=3.2, label='axis term'),
                       plt.Line2D([], [], marker='o', ls='', color=ORANGE, ms=3.2, label='selective term')],
              loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=5.3, handletextpad=0.2, columnspacing=0.6,
              borderaxespad=0.2)


def panel_retrain_transfer(ax):
    """Pooled selective-both in the training and in the other dataset, against the agreement of the measurements."""
    P = pd.read_csv(RT / 'paired_pooled.csv')
    S = pd.read_csv(RT / 'paired_summary.csv')
    S = S[(S.panel == 'main') & (S.model == 'ref:source_measurement')]
    Q2 = pd.read_csv(RESULTS / 'revision_round2' / 'source_reference' / 'source_ratio_by_target.csv')
    Q2 = Q2[(Q2.benchmark == 'retrain') & Q2.group.str.startswith('pooled')]
    TB = pd.read_csv(RESULTS / 'revision_round2' / 'retention_calibration' / 'baselines_pairs.csv')
    TB = TB[(TB.benchmark == 'retrain') & TB.group.isin(['NaiveTissueDrugMeanPredictor', 'LB:NaiveTissueDrugMeanPredictor'])]
    lim = 0.5
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for st in ['CTRPv2_LNIC50', 'GDSC2_AUC']:
        c = RT_COLOR[st]
        for t in RT_TARGETS[st]:
            r = P[(P.setting == st) & (P.target == t)].iloc[0]
            m = S[(S.setting == st) & (S.target == t)].mean_sel_BOTH_t.iloc[0]
            sz = 6 + 1.2 * np.sqrt(S[(S.setting == st) & (S.target == t)].drugs.iloc[0])
            ax.plot([m, m], [r.mean_sel_BOTH_s, r.mean_sel_BOTH_t], color=LIGHT_GRAY, lw=0.5, zorder=1)
            ax.scatter([m], [r.mean_sel_BOTH_s], s=sz, facecolor='white', edgecolor=c, lw=0.7, zorder=2)
            ax.scatter([m], [r.mean_sel_BOTH_t], s=sz, color=c, lw=0, zorder=3)
            q = Q2[Q2.pair == f'{st}->{t}']
            if len(q):
                q = q.iloc[0]
                ax.plot([m, m], [q.model_cross_selBOTH_lo, q.model_cross_selBOTH_hi], color=c, lw=0.6, alpha=0.6, zorder=2)
            if st == 'GDSC2_AUC':      # CTRPv1 and CTRPv2 sit among other markers: labelled in free space
                pos = {'CTRPv1': (0.365, 0.25), 'CTRPv2': (0.365, 0.105),
                       'GDSC1': (0.212, 0.045)}.get(t, (m + 0.01, r.mean_sel_BOTH_t))
                ax.annotate(t, (m, r.mean_sel_BOTH_t), xytext=pos, fontsize=5.0, color=INK2, va='center',
                            arrowprops=dict(arrowstyle='-', color=LIGHT_GRAY, lw=0.4, shrinkA=0, shrinkB=2))
            b = TB[(TB.pair == f'{st}->{t}')]     # DrEval's tissue-by-drug mean scored identically (99b)
            if len(b):
                b = b.iloc[0]
                ax.plot([m - 0.008] * 2, [b.selBOTH_within, b.selBOTH_cross], color=BASE_COL, lw=0.4, alpha=0.6, zorder=1)
                ax.scatter([m - 0.008], [b.selBOTH_within], s=10, marker='+', color=BASE_COL, lw=0.7, zorder=4)
                ax.scatter([m - 0.008], [b.selBOTH_cross], s=10, marker='x', color=BASE_COL, lw=0.7, zorder=4)
    ax.set_xlim(0.2, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('selective-both of the training\nmeasurement in the target')
    ax.set_ylabel('selective-both of identical predictions')
    hgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=RT_COLOR[st], ms=3.4,
                                  label=RT_LABEL[st].replace('trained on ', '')) for st in RT_COLOR]
              + [plt.Line2D([], [], marker='o', ls='', mfc='white', mec=GRAY, ms=3.4, label='in the training dataset'),
                 plt.Line2D([], [], marker='o', ls='', color=GRAY, ms=3.4, label='in the target'),
                 plt.Line2D([], [], marker='x', ls='', color=BASE_COL, ms=3.0, mew=0.7, label='tissue-by-drug mean')],
              loc='upper left', ncol=1, fontsize=5.0, handletextpad=0.2, borderaxespad=0.2, frameon=False)


DEP = RESULTS / 'revision_round5' / 'crispr_dependency'
AMP_SETTINGS = [   # (benchmark, setting, label, marker, colour) of capture_pooled.csv's learned-model pools
    ('leaderboard', 'CTRPv2_lnIC50', 'DrEval leaderboard', 'o', BLUE),
    ('retrain', 'GDSC2_AUC', 'GDSC2 AUC', 's', INK),
    ('retrain', 'GDSC2_LNIC50', 'GDSC2 ln IC$_{50}$', 'D', INK),
    ('retrain', 'CTRPv2_AUC', 'CTRP v2 AUC', 'v', INK),
    ('improve', 'CCLE_AUC', 'IMPROVE (5)', '^', AQUA), ('improve', 'CTRPv2_AUC', None, '^', AQUA),
    ('improve', 'GDSCv1_AUC', None, '^', AQUA), ('improve', 'GDSCv2_AUC', None, '^', AQUA),
    ('improve', 'gCSI_AUC', None, '^', AQUA)]


def amplification_points():
    """One row per setting: the pooled learned models' axis share against the perfect predictor's (78; 105c)."""
    C = pd.read_csv(RESULTS / 'revision_round2' / 'capture' / 'capture_pooled.csv')
    C = C[~C.group.str.contains('ridges|minus|this study')]
    rows = []
    for bm, st, lab, mk, col in AMP_SETTINGS:
        x = C[(C.benchmark == bm) & (C.setting == st)]
        if len(x):
            rows.append(dict(x.iloc[0], label=lab, marker=mk, colour=col, domain='drug response'))
    if (DEP / 'pooled.csv').exists():
        D = pd.read_csv(DEP / 'pooled.csv')
        x = D[(D.config == 'primary') & (D.group == 'learned')]
        if len(x):
            rows.append(dict(x.iloc[0], benchmark='crispr', setting='DepMap_Chronos',
                             label='gene dependency', marker='*', colour=ORANGE, domain='gene dependency'))
    return pd.DataFrame(rows)


def panel_amplification(ax):
    """Models amplify whatever general-sensitivity axis the data contain: pooled learned models' axis share against a
    perfect predictor's, in every benchmark setting and in gene-dependency prediction."""
    A = amplification_points()
    for k, ls in [(1, '-'), (2, (0, (3, 2))), (4, (0, (1, 1.5)))]:
        xs = np.linspace(0, min(0.33, 1.02 / k), 10)      # stop below the legend band
        ax.plot(xs, k * xs, color=ORANGE if k == 1 else GRAY, lw=0.8 if k == 1 else 0.6, ls=ls, zorder=1)
        xt = 0.33 if k == 1 else min(0.32, 0.98 / k)
        ax.text(xt, k * xt + 0.015, f'{k}$\\times$', fontsize=5.0, color=ORANGE if k == 1 else INK2, ha='right',
                va='bottom')
    hs = []
    for _, row in A.iterrows():
        ax.errorbar(row.perfect_share, row.axis_share,
                    xerr=[[row.perfect_share - row.perfect_share_lo], [row.perfect_share_hi - row.perfect_share]],
                    yerr=[[row.axis_share - row.axis_share_lo], [row.axis_share_hi - row.axis_share]],
                    fmt=row.marker, ms=6 if row.marker == '*' else 3.4, color=row.colour, ecolor=row.colour,
                    elinewidth=0.5, mew=0, zorder=4 if row.marker == '*' else 3)
        if isinstance(row.label, str):
            hs.append(plt.Line2D([], [], marker=row.marker, ls='', color=row.colour, ms=5 if row.marker == '*' else 3,
                                 label=row.label))
    r = np.corrcoef(A.perfect_share, A.axis_share)[0, 1]
    ax.text(0.97, 0.03, f'Pearson r = {r:.2f}\n({len(A)} settings)', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=5.0, color=INK2)
    ax.set_xlim(0, 0.34); ax.set_ylim(0, 1.3); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("perfect predictor's axis share\n(the data's, mean $r_{zg}^2$)")
    ax.set_ylabel("learned models' axis share")
    hgrid(ax)
    ax.legend(handles=hs, loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=5.0, handletextpad=0.2,
              borderaxespad=0.2, frameon=False, handlelength=1.0, labelspacing=0.2, ncol=2, columnspacing=0.6)


DEP_NAME = {'ridge': 'ridge (per gene)', 'random_forest': 'random forest', 'mlp': 'MLP (global)', 'knn': 'k-NN',
            'pca_regression': 'PCA regression', 'omics_axis': 'omics-predicted axis', 'tissue_axis': 'tissue axis mean',
            'tissue_gene_mean': 'tissue-by-gene mean'}


DEP_LABEL_OFFSET = {'pca_regression': ((-4, 2), 'right'), 'knn': ((-4, -6), 'right'), 'random_forest': ((1, -8), 'left'),
                    'mlp': ((4, 2), 'left'), 'ridge': ((4, 1), 'left'), 'tissue_gene_mean': ((0, -9), 'center')}


def panel_dep_capture(ax):
    """Gene dependency: axis capture against selective capture per model (105c), with lineage-adjusted S (tick)."""
    Mo = pd.read_csv(DEP / 'models.csv')
    Mo = Mo[Mo.config == 'primary']
    ax.plot([0, 1.4], [0, 1.4], color=GRAY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for _, row in Mo.iterrows():
        learned = row.kind == 'learned'
        if learned and pd.notna(row.selective_capture_lin):
            ax.vlines(row.axis_capture, row.selective_capture_lin, row.selective_capture, color=LIGHT_GRAY, lw=0.5, zorder=2)
            ax.scatter([row.axis_capture], [row.selective_capture_lin], s=9, marker='_', color=INK, lw=0.7, zorder=3)
        mk, col = ('o', BLUE) if learned else (('x', BASE_COL) if row.model == 'tissue_gene_mean' else ('P', BASE_COL))
        ax.scatter([row.axis_capture], [row.selective_capture], s=14, marker=mk, color=col, lw=0.8 if mk == 'x' else 0,
                   zorder=4)
        off, ha = DEP_LABEL_OFFSET.get(row.model, ((3, 2), 'left'))
        ax.annotate(DEP_NAME.get(row.model, row.model), (row.axis_capture, row.selective_capture), xytext=off,
                    textcoords='offset points', fontsize=5.0, color=INK2, ha=ha)
    ax.set_xlim(0, 1.45); ax.set_ylim(-0.02, 0.42)
    ax.set_xlabel("axis capture\n(ratio to perfect predictor's axis term)")
    ax.set_ylabel('selective capture')
    ax.text(0.97, 0.97, 'perfect predictor at (1, 1)', transform=ax.transAxes, ha='right', va='top', fontsize=5.0,
            color=INK)
    hgrid(ax)


def panel_dep_predictability(ax):
    """Genes that pass a per-gene 'predictability' threshold (r >= 0.3), and those whose axis term is at least half of
    r, per model (105c)."""
    Pr = pd.read_csv(DEP / 'predictability.csv')
    Pr = Pr[(Pr.config == 'primary') & (Pr.threshold == 0.3)]
    order = [m for m in ['ridge', 'mlp', 'random_forest', 'pca_regression', 'knn', 'omics_axis'] if m in set(Pr.model)]
    Pr = Pr.set_index('model').loc[order].reset_index()
    y = np.arange(len(Pr))
    ax.barh(y, Pr.n_pass_raw, color=LIGHT_BLUE, height=0.6, label='r $\\geq$ 0.3')
    ax.barh(y, Pr.n_axis_term_ge_half_r, color=BLUE, height=0.6, label='of which axis term $\\geq$ r/2')
    ax.errorbar(Pr.n_pass_raw, y, xerr=[Pr.n_pass_raw - Pr.n_pass_raw_lo, Pr.n_pass_raw_hi - Pr.n_pass_raw], fmt='none',
                ecolor=INK2, elinewidth=0.5, capsize=1.2)
    ax.set_yticks(y); ax.set_yticklabels([DEP_NAME.get(m, m) for m in Pr.model], fontsize=5.0)
    ax.invert_yaxis()
    ax.set_xlabel('genes "predictable" at r $\\geq$ 0.3\n(of %d active genes)' % int(Pr.genes_scored.max()))
    xgrid(ax)
    ax.legend(loc='lower right', fontsize=5.0, frameon=False, handlelength=1.0)


DEP_AXIS_WITH = [('ScreenROCAUC', 'screen ROC AUC'), ('ScreenNNMD', 'screen NNMD'),
                 ('ScreenMedianEssentialDepletion', 'median essential depletion'), ('growth_rate', 'growth rate'),
                 ('ScreenDoublingTime', 'doubling time'), ('CasActivity', 'Cas9 activity'),
                 ('omics_predicted_axis_oof', 'omics-predicted axis (out of fold)'), ('all_gene_axis', 'all-gene axis')]


def panel_dep_axis(ax):
    """What the general-dependency axis tracks: Pearson r with screen quality, growth and the predicted axis (105c)."""
    Ax = pd.read_csv(DEP / 'axis.csv')
    Ax = Ax[(Ax.axis == 'active_axis') & (Ax.statistic == 'pearson_r')].set_index('with')
    keep = [(k, lab) for k, lab in DEP_AXIS_WITH if k in Ax.index]
    v = Ax.loc[[k for k, _ in keep]]
    y = np.arange(len(v))
    ax.errorbar(v.value, y, xerr=[v.value - v.lo, v.hi - v.value], fmt='o', ms=3, color=ORANGE, ecolor=ORANGE,
                elinewidth=0.6)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in keep], fontsize=5.0)
    ax.invert_yaxis()
    ax.set_xlim(-0.7, 0.7)
    ax.set_xlabel('Pearson r with the gene-dependency axis')
    xgrid(ax)


def ed_figure4():
    """Extended Data Fig. 4 (formerly Extended Data Fig. 3): how much general sensitivity the data contain, by dataset and endpoint, and the same
    decomposition in CRISPR gene-dependency prediction."""
    fig = plt.figure(figsize=(W, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.75, hspace=0.62)
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, 'a', x=-0.36, y=1.12); panel_axis_weight(ax)
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, 'b', x=-0.3, y=1.12); panel_dep_capture(ax)
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, 'c', x=-0.55, y=1.08); panel_dep_predictability(ax)
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, 'd', x=-0.62, y=1.08); panel_dep_axis(ax)
    save(fig, 'extended_data_fig4_generality')


# (figure3 of the round-5 layout: replaced by the regrouped figures below)


# ================================================================== Supplementary: worked example
def sfigure_example():
    """Trametinib in CTRP: profile, general-sensitivity axis and selective residual by BRAF/RAS hotspot status, and
    what a per-drug correlation means for picking cell lines."""
    from scipy.stats import norm
    z = pd.read_parquet(RESULTS / 'decomposition' / 'z_CTRP.parquet')
    r = pd.read_parquet(RESULTS / 'decomposition' / 'resid_CTRP.parquet')
    g = pd.read_parquet(RESULTS / 'decomposition' / 'fitness_CTRP.parquet').fitness
    h = pd.read_parquet(PROCESSED / 'features' / 'hotspot.parquet')
    d = 'DPC-006587'
    df = pd.DataFrame({'z': z.loc[d], 'r': r.loc[d], 'g': g}).dropna()
    df['g'] = (df.g - df.g.mean()) / df.g.std()
    mut = h.reindex(df.index)
    grp = np.where(mut['BRAF'] > 0, 'BRAF', np.where((mut['NRAS'] > 0) | (mut['KRAS'] > 0), 'NRAS/KRAS', 'neither'))
    grp = np.where(mut['BRAF'].isna(), 'no data', grp)
    df['grp'] = grp
    cols = {'BRAF': ORANGE, 'NRAS/KRAS': BLUE, 'neither': LIGHT_GRAY, 'no data': '#e9e8e2'}
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.4), gridspec_kw=dict(width_ratios=[1, 0.8, 1], wspace=0.5))
    ax = axs[0]; panel_label(ax, 'a', x=-0.25)
    for k in ['no data', 'neither', 'NRAS/KRAS', 'BRAF']:
        q = df[df.grp == k]
        ax.scatter(q.g, q.z, s=6, color=cols[k], lw=0, label=f'{k} ({len(q)})', zorder=3 if k == 'BRAF' else 2)
    b1, b0 = np.polyfit(df.g, df.z, 1)
    xs = np.linspace(df.g.min(), df.g.max(), 10)
    ax.plot(xs, b0 + b1 * xs, color=INK, lw=0.9)
    rr = np.corrcoef(df.g, df.z)[0, 1]
    ax.set_xlabel('general-sensitivity axis $g$ (s.d.)'); ax.set_ylabel('trametinib sensitivity $z$ (CTRP)')
    ax.text(0.03, 0.97, f'$r_{{zg}}$ = {rr:.2f}; axis share {rr ** 2:.2f}', transform=ax.transAxes, va='top', fontsize=6)
    ax.legend(loc='lower right', fontsize=5.2, handletextpad=0.1, borderaxespad=0.1)
    hgrid(ax)
    ax = axs[1]; panel_label(ax, 'b', x=-0.3)
    order = ['BRAF', 'NRAS/KRAS', 'neither']
    for k, gname in enumerate(order):
        for j, col in enumerate(['z', 'r']):
            v = df.loc[df.grp == gname, col].values
            x = k + (j - 0.5) * 0.36 + RNG.uniform(-0.07, 0.07, len(v))
            ax.scatter(x, v, s=3, color=cols[gname] if j else LIGHT_GRAY, lw=0, alpha=0.8)
            ax.plot([k + (j - 0.5) * 0.36 - 0.12, k + (j - 0.5) * 0.36 + 0.12], [np.median(v)] * 2, color=INK, lw=1.1)
    ax.set_xticks(range(3)); ax.set_xticklabels(order, fontsize=6)
    ax.set_ylabel('left: profile $z$; right: selective residual')
    ax.axhline(0, color=GRAY, lw=0.6); hgrid(ax)
    ax = axs[2]; panel_label(ax, 'c', x=-0.25)
    rs = np.linspace(0, 0.6, 61)
    for frac, ls in [(0.1, '-'), (0.25, (0, (3, 2)))]:
        c = norm.ppf(1 - frac)
        gain = norm.pdf(c) / frac
        ax.plot(rs, rs * gain, color=INK, lw=1.0, ls=ls, label=f'top {int(frac * 100)}% of lines by prediction')
    for v, lab in [(0.13, 'best published model, selective'), (0.25, 'selective-target ridge'),
                   (0.42, 'best published model, raw')]:
        ax.axvline(v, color=GRAY, lw=0.5)
        ax.text(v - 0.012, 1.07, lab, fontsize=5.0, color=INK2, va='top', ha='right', rotation=90)
    ax.set_xlabel('per-drug correlation of the prediction')
    ax.set_ylabel('mean response of the selected lines\n(s.d. above average)')
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.1); hgrid(ax)
    ax.legend(loc='lower right', fontsize=5.2, handletextpad=0.3)
    save(fig, 'supp_figure_worked_example')


# ============================================================================ regrouped layout (2026-10-16)
# One Results section per main figure, at most five panels, 180 mm wide; technical panels in Extended Data.

def figure2():
    """Fig. 2: the axis carries most per-drug accuracy in both benchmarks, and models amplify it."""
    fig = plt.figure(figsize=(W, 5.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.62,
                          left=0.08, right=0.98, top=0.93, bottom=0.09)
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, 'a', x=-0.17, y=1.08); panel_leaderboard(ax)
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, 'b', x=-0.25, y=1.08); panel_improve_share(ax)
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, 'c', x=-0.17, y=1.08); panel_improve_floor(ax)
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, 'd', x=-0.25, y=1.08); panel_amplification(ax)
    save(fig, 'figure2_amplification')


def figure3():
    """Fig. 3: axis-adjusted scores predict which models keep drug-specific accuracy elsewhere."""
    fig = plt.figure(figsize=(W, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 0.85, 1.25], wspace=0.6, left=0.08, right=0.98, top=0.8,
                          bottom=0.2)
    ax = fig.add_subplot(gs[0]); panel_label(ax, 'a', x=-0.38, y=1.12)
    panel_transfer_scatter(ax, 'raw', 'raw per-drug correlation in CTRP v2\n(median over drugs)', 'raw_median')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.84), fontsize=5.0, frameon=False, handletextpad=0.1)
    ax = fig.add_subplot(gs[1]); panel_label(ax, 'b', x=-0.38, y=1.12)
    panel_transfer_scatter(ax, 'S', 'selective capture in CTRP v2', 'S')
    ax = fig.add_subplot(gs[2]); panel_label(ax, 'c', x=-0.62, y=1.12); panel_transfer_forest(ax)
    save(fig, 'figure3_transfer')


def figure4():
    """Fig. 4: identical predictions scored in another dataset: processing, endpoint and tested range."""
    R = target_rows()
    fig = plt.figure(figsize=(W, 7.4))
    ax = fig.add_axes([0.395, 0.385, 0.245, 0.565]); panel_label(ax, 'a', x=-1.5, y=1.045); panel_retention(ax, R)
    ax.text(-1.43, 1.012, 'Same CTRPv2-trained predictions, scored in another dataset or processing',
            transform=ax.transAxes, fontsize=6.6, color=INK, va='bottom')
    ax = fig.add_axes([0.665, 0.385, 0.255, 0.565]); panel_label(ax, 'b', x=-0.05, y=1.045); panel_change(ax, R)
    ax = fig.add_axes([0.09, 0.065, 0.36, 0.215]); panel_label(ax, 'c', x=-0.2, y=1.12); panel_retrain_terms(ax)
    ax = fig.add_axes([0.62, 0.065, 0.32, 0.215]); panel_label(ax, 'd', x=-0.27, y=1.12); panel_retrain_transfer(ax)
    save(fig, 'figure4_cross_dataset')


def figure5():
    """Fig. 5: what is selective, and how much models capture (formerly Fig. 4)."""
    fig = plt.figure(figsize=(W, 6.6))
    ax = fig.add_axes([0.275, 0.575, 0.233, 0.375]); panel_label(ax, 'a', x=-1.15, y=1.04); panel_classes(ax)
    ax = fig.add_axes([0.784, 0.575, 0.196, 0.375]); panel_label(ax, 'b', x=-1.15, y=1.04); panel_c2(ax)
    ax = fig.add_axes([0.085, 0.075, 0.36, 0.38]); panel_label(ax, 'c', x=-0.2, y=1.04); panel_biomarker(ax)
    ax = fig.add_axes([0.6, 0.075, 0.375, 0.34]); panel_label(ax, 'd', x=-0.3, y=1.1); panel_crossdataset(ax)
    save(fig, 'figure5_ceiling')


def ed_figure1():
    """Extended Data Fig. 1: known-answer simulation of the estimators (formerly Fig. 5d-f)."""
    fig = plt.figure(figsize=(W, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.9, 0.95], wspace=0.55, left=0.08, right=0.98, top=0.88,
                          bottom=0.2)
    ax = fig.add_subplot(gs[0]); panel_label(ax, 'a', x=-0.3, y=1.06); panel_sim_equal(ax)
    ax = fig.add_subplot(gs[1]); panel_label(ax, 'b', x=-0.35, y=1.06); panel_sim_recovery(ax)
    ax = fig.add_subplot(gs[2]); panel_label(ax, 'c', x=-0.35, y=1.06); panel_sim_leakage(ax)
    save(fig, 'extended_data_fig1_simulation')


def ed_figure3():
    """Extended Data Fig. 3: each leaderboard model's mean per-drug correlation split into the two terms (formerly
    Fig. 2b)."""
    fig = plt.figure(figsize=(4.6, 4.9))
    ax = fig.add_axes([0.42, 0.1, 0.4, 0.84]); panel_decomposition(ax)
    save(fig, 'extended_data_fig3_leaderboard')


def ed_figure5():
    """Extended Data Fig. 5: proportional attenuation: identical predictions in GDSC1 and in IMPROVE's other datasets
    (formerly Figs. 2c and 3c)."""
    fig = plt.figure(figsize=(W, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.0], wspace=0.45, left=0.09, right=0.97, top=0.82, bottom=0.17)
    ax = fig.add_subplot(gs[0]); panel_label(ax, 'a', x=-0.3, y=1.1); panel_paired(ax)
    ax = fig.add_subplot(gs[1]); panel_label(ax, 'b', x=-0.25, y=1.1); panel_improve_transfer(ax)
    save(fig, 'extended_data_fig5_attenuation')



if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', 'ed1', 'ed2', 'ed3', 'ed4', 'ed5', 'ed6']
    for w in which:
        fn = globals().get(f'ed_figure{w[2:]}') if w.startswith('ed') else globals().get(f'figure{w}')
        if fn is None:
            print('figure', w, 'not defined'); continue
        RNG = np.random.default_rng(20261015)   # each figure as if drawn by its own invocation (99i checks this)
        fn()
        print('figure', w, 'done')
