"""Main and supplementary figures (PDF + PNG) from the saved results. Palette follows the validated
reference instance (blue / orange / aqua categorical, single-hue blue sequential, ink-coloured text)."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import FIGURES, PROCESSED, RESULTS

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GRAY, LIGHT_GRAY = '#898781', '#c3c2b7'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e1e0d9'
SEQ = LinearSegmentedColormap.from_list('blue_seq', ['#f4f8fd', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
LABEL = {'PRISM': 'PRISM', 'CTRP': 'CTRP', 'GDSC1': 'GDSC1', 'GDSC2': 'GDSC2', 'gCSI': 'gCSI', 'NCI60': 'NCI-60'}
DISC = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
ALL6 = DISC + ['gCSI', 'NCI60']

plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5, 'axes.edgecolor': LIGHT_GRAY,
    'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'pdf.fonttype': 42,
    'savefig.dpi': 300, 'figure.dpi': 150})


def panel_label(ax, s, x=-0.18, y=1.06):
    """Panel letter, 8 pt bold (Nature Methods: 5-7 pt text labels, 8 pt bold panel letters)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left', color=INK)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


FIG_W = 7.08                     # 180 mm, the journal's maximum figure width


def fit_width(fig):
    """Narrow the canvas until the drawn figure (tight bounding box plus the save padding) is at most 180 mm wide; text
    sizes are unchanged, the axes shrink. 99i_source_data.py applies the same step before rendering."""
    target = FIG_W - 2 * plt.rcParams['savefig.pad_inches'] - 0.003
    for _ in range(6):
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        if bb.width <= target:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(w - (bb.width - target) - 0.004, h)
    print(f'WARNING: drawn width {bb.width * 25.4:.1f} mm still exceeds 180 mm')


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fit_width(fig)
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.png', bbox_inches='tight')
    plt.close(fig)


def jload(p):
    return json.loads((RESULTS / p).read_text())


# ------------------------------------------------------------------ Figure 1
def figure1():
    fit = jload('fitness_axis/summary.json')
    pairs = pd.DataFrame(fit['pairs'])
    stats = pd.read_csv(RESULTS / 'decomposition' / 'drug_screen_stats.csv')
    fig = plt.figure(figsize=(FIG_W, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.05, 0.95], wspace=0.78, hspace=0.62)

    # (a) schematic with a synthetic matrix
    ax = fig.add_subplot(gs[0, 0]); ax.axis('off'); panel_label(ax, 'a', x=-0.05)
    rng = np.random.default_rng(1)
    g = np.sort(rng.normal(size=12))
    load = np.r_[np.linspace(1.0, 0.3, 6)]
    sel = rng.normal(scale=0.6, size=(6, 12)); sel[1, :3] += 2.2; sel[4, 8:11] += 2.0
    fitpart = load[:, None] * g[None, :]
    z = fitpart + sel
    mats = [(z, 'within-drug\nprofiles $z$'), (fitpart, 'loading x\nfitness axis'), (sel, 'selective\nresidual $r$')]
    for k, (m, t) in enumerate(mats):
        a = ax.inset_axes([0.0 + k * 0.36, 0.34, 0.28, 0.46])
        a.imshow(m, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(LIGHT_GRAY); sp.set_linewidth(0.5)
        a.set_title(t, fontsize=6.3, color=INK, pad=3)
        if k == 0:
            a.set_ylabel('drugs', fontsize=6.3, color=INK2); a.set_xlabel('cell lines', fontsize=6.3, color=INK2)
    ax.text(0.315, 0.57, '=', transform=ax.transAxes, ha='center', va='center', fontsize=7, color=INK2)
    ax.text(0.675, 0.57, '+', transform=ax.transAxes, ha='center', va='center', fontsize=7, color=INK2)
    ax.text(0.0, 0.02, 'fitness axis = leave-drug-out mean of $z$\nover every drug in the same screen;\n'
            'selective residual = $z$ after regressing\nit on that axis', transform=ax.transAxes,
            fontsize=5.8, color=INK2, va='bottom')

    # (b) fitness-axis agreement heatmap; diagonal = split-half reliability
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, 'b')
    n = len(ALL6); M = np.full((n, n), np.nan)
    for i, s in enumerate(ALL6):
        M[i, i] = fit['per_screen'][s]['split_half'][0]
    for r in pairs.itertuples():
        i, j = ALL6.index(r.a), ALL6.index(r.b)
        M[i, j] = M[j, i] = r.corr
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.6,
                    color='white' if v > 0.62 else INK, fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([LABEL[s] for s in ALL6], rotation=45, ha='right'); ax.set_yticklabels([LABEL[s] for s in ALL6])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.axhline(3.5, color='white', lw=1.5); ax.axvline(3.5, color='white', lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.8, length=2)
    cb.set_label('fitness-axis correlation', fontsize=6)
    ax.set_title('diagonal: split-half reliability', fontsize=6, color=INK2, pad=3)

    # (c) per-drug fitness share by screen (active drugs)
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, 'c')
    data, labels = [], []
    for s in DISC:
        d = stats[(stats.screen == s) & (stats.frac_auc_below_0p8 >= 0.1)].fitness_r2.values
        data.append(d); labels.append(LABEL[s])
    bp = ax.boxplot(data, widths=0.42, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=GRAY, lw=0.7),
                    capprops=dict(color=GRAY, lw=0.7), boxprops=dict(lw=0))
    for p in bp['boxes']:
        p.set_facecolor('#b7d3f6')
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(labels, fontsize=5.3)
    ax.set_ylabel('fitness share of a drug\'s\nprofile ($R^2$, active drugs)')
    ax.set_ylim(0, 0.8); hgrid(ax)
    for k, d in enumerate(data):
        ax.text(k + 1.25, np.median(d), f'{np.median(d):.2f}', ha='left', va='center', fontsize=5.8, color=INK)

    rob = jload('robustness/summary.json'); es = jload('endpoint_sensitivity/summary.json'); rs = jload('prism_rescreen/summary.json')
    # (d) agreement ladder: same screen (split-half) > same-lab re-screen > other laboratories
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, 'd', x=-0.3)
    ep = pd.DataFrame(es['pairs'])
    cross = ep[~ep.same_screen]
    auc_pairs = cross[~cross.a.str.contains('lnIC50|GR_AOC') & ~cross.b.str.contains('lnIC50|GR_AOC')].axis_corr.values
    ic_pairs = cross[(cross.a.str.contains('lnIC50') | cross.b.str.contains('lnIC50')) & ~cross.a.str.contains('GR_AOC')
                     & ~cross.b.str.contains('GR_AOC')].axis_corr.values
    within = [v['split_half'][0] for k, v in fit['per_screen'].items() if k != 'gCSI']
    groups = [('split-\nhalf', within, INK), ('re-\nscreen', [rs['fitness_axis_corr']], BLUE),
              ('labs\n(AUC)', auc_pairs, ORANGE), ('labs\n(IC$_{50}$)', ic_pairs, AQUA)]
    for k, (lab, vals, col) in enumerate(groups):
        x = np.full(len(vals), k) + np.random.default_rng(k).uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=12, color=col, edgecolor='white', lw=0.4, zorder=3)
        ax.plot([k - 0.25, k + 0.25], [np.median(vals)] * 2, color=INK, lw=1.2, zorder=4)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=5.6)
    ax.set_ylabel('agreement of fitness axes ($r$)'); ax.set_ylim(-0.15, 1.05); hgrid(ax)
    ax.axhline(0, color=GRAY, lw=0.6)

    # (e) share of each axis loading on a factor common to all screens (triads)
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, 'e', x=-0.3)
    sa, si = rob['shared_fraction_auc'], rob['shared_fraction_with_ic50']
    items = [(LABEL.get(k, k), v, ORANGE) for k, v in sa.items()] + \
            [(k.replace('_lnIC50', ' IC$_{50}$'), v, AQUA) for k, v in si.items() if 'lnIC50' in k]
    for k, (lab, v, col) in enumerate(items):
        ax.errorbar(k, v['lambda2_median'], yerr=[[v['lambda2_median'] - v['ci95'][0]], [v['ci95'][1] - v['lambda2_median']]],
                    fmt='o', ms=4, color=col, elinewidth=0.8)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], rotation=40, ha='right', fontsize=5.8)
    ax.set_ylabel('share of the axis shared\nacross screens (triads)'); ax.set_ylim(-0.25, 1.0); hgrid(ax)
    ax.axhline(0, color=GRAY, lw=0.6)

    # (f) association with measured doubling time (gCSI)
    ax = fig.add_subplot(gs[1, 2]); panel_label(ax, 'f', x=-0.35)
    dt = rob['doubling_time']
    order = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2', 'NCI60', 'gCSI', 'GDSC1_lnIC50', 'GDSC2_lnIC50', 'gCSI_GR_AOC']
    names = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2', 'NCI-60', 'gCSI', 'GDSC1 IC$_{50}$', 'GDSC2 IC$_{50}$', 'gCSI GR']
    vals = [dt[k]['spearman_vs_doubling_time'] for k in order]
    cols = [ORANGE] * 6 + [AQUA, AQUA, GRAY]
    ax.barh(range(len(vals)), vals, color=cols, height=0.6)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=5.8); ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=0.6); ax.set_xlabel('Spearman with doubling time\n(gCSI measurements)')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, 'figure1_fitness_axis')


# ------------------------------------------------------------------ Figure 2
DREVAL_NAMES = {'ridge_sel_lm': 'kernel ridge, selective target', 'ridge_raw_lm': 'kernel ridge, raw target',
                'fitness_only': 'axis only (one value per cell line)', 'NaiveTissueDrugMeanPredictor': 'tissue x drug mean',
                'NaiveTissueMeanPredictor': 'tissue mean', 'NaiveMeanEffectsPredictor': 'tissue + drug effects'}
DREVAL_COL = {'naive': GRAY, 'classical': '#86b6ef', 'neural/factorization': BLUE, 'this study': AQUA}
DREVAL_AGNOSTIC = ['fitness_only', 'NaiveTissueMeanPredictor', 'NaiveMeanEffectsPredictor']
SHORT = {'fitness_only': 'axis only'}


def panel_own_models(ax):
    """Cell-blind predictors trained in one discovery screen, scored within and across screens."""
    s = pd.read_csv(RESULTS / 'models' / 'score_summary.csv')
    models = [('fitness_measured', 'measured\nfitness'), ('fitness_omics', 'omics\nfitness'),
              ('lineage_raw', 'lineage\nonly'), ('ridge_raw', 'ridge\n(raw)'),
              ('ridge_sel', 'ridge\n(selective)')]
    within = s[s['train'] == s['eval']].groupby('model').raw.median()
    cross_raw = s[s['train'] != s['eval']].groupby('model').raw.median()
    cross_sel = s[s['train'] != s['eval']].groupby('model').sel.median()
    x = np.arange(len(models)); w = 0.24
    v1 = [within[m] for m, _ in models]; v2 = [cross_raw[m] for m, _ in models]; v3 = [cross_sel[m] for m, _ in models]
    ax.bar(x - w - 0.02, v1, w, color=LIGHT_GRAY, label='within screen, per-drug correlation')
    ax.bar(x, v2, w, color=BLUE, label='other screens, per-drug correlation')
    ax.bar(x + w + 0.02, v3, w, color=ORANGE, label='other screens, selective correlation')
    for xi, (a_, b_, c_) in enumerate(zip(v1, v2, v3)):
        ax.text(xi - w - 0.02, a_ + 0.008, f'{a_:.2f}', ha='center', fontsize=5.6, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in models])
    ax.set_ylabel('median correlation with held-out cell lines'); ax.set_ylim(0, 0.52); hgrid(ax)
    ax.legend(loc='upper right', ncol=1, handlelength=1.0, borderaxespad=0.1, fontsize=6)
    ax.set_title('Cell-blind predictors trained in one screen', loc='left', color=INK)


def panel_global_models(ax):
    """Global tree and neural-network models under the common split designs."""
    g = pd.read_csv(RESULTS / 'global_model' / 'scores.csv')
    m = pd.read_csv(RESULTS / 'global_mlp' / 'scores.csv')
    order = [('random_pairs', 'random pairs'), ('cell_blind', 'cell-blind'), ('drug_blind', 'drug-blind'),
             ('cross_screen', 'other screen')]
    def agg_of(df, model):
        d = df[df.model == model].copy()
        d['split2'] = d.split.where(~d.split.str.startswith('cross_from'), 'cross_screen')
        return d.groupby('split2').agg(raw=('per_drug_raw', 'median'), sel=('per_drug_sel', 'median'), glob=('global_r', 'median'))
    A, B = agg_of(g, 'global_gbt'), agg_of(m, 'global_mlp')
    x = np.arange(len(order)); w = 0.19
    specs = [(A, 'raw', -1.5, BLUE, 'trees, per-drug'), (B, 'raw', -0.5, '#86b6ef', 'neural net, per-drug'),
             (A, 'sel', 0.5, ORANGE, 'trees, selective'), (B, 'sel', 1.5, '#f4a582', 'neural net, selective')]
    for tab, col, off, colr, lab in specs:
        ax.bar(x + off * (w + 0.01), [tab.loc[k, col] for k, _ in order], w, color=colr, label=lab)
    add = g[(g.model == 'drug_plus_cell_mean') & (g.split == 'random_pairs')].per_drug_raw.median()
    xa = 0 - 1.5 * (w + 0.01)
    ax.plot([xa - w / 2 - 0.03, xa + w / 2 + 0.03], [add, add], color=INK, lw=1.3, zorder=5)
    ax.annotate('drug + cell means\n(no features)', xy=(xa + w / 2, add), xytext=(0.62, 0.31), fontsize=5.6, color=INK2,
                arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(r {A.loc[k, "glob"]:.2f}/{B.loc[k, "glob"]:.2f})'.replace('0.', '.') for k, l in order], fontsize=6)
    ax.text(1.0, -0.2, 'r: global Pearson, trees/neural net', transform=ax.transAxes, ha='right', fontsize=5.4, color=INK2)
    ax.set_ylabel('median over drugs and screens'); ax.set_ylim(0, 0.62); hgrid(ax)
    ax.legend(loc='upper right', handlelength=1.0, ncol=1, fontsize=6)
    ax.set_title('Global models on raw AUC (trees / neural net)', loc='left', color=INK)


def dreval_summary():
    S = pd.read_csv(RESULTS / 'dreval' / 'model_summary.csv')
    obs_r2 = jload('dreval/run.json')['median_fitness_r2_observed']
    fm = S.set_index('model').loc['fitness_measured', 'raw']
    S = S[(S.drugs > 0) & ~S.model.isin(['SVR', 'fitness_measured', 'ridge_raw_expr', 'ridge_sel_expr', 'ridge_raw', 'ridge_sel'])]
    S = S.rename(columns={'class': 'klass'})
    return S, fm, obs_r2


def marker(m):
    return '^' if m in DREVAL_AGNOSTIC else 'o'


def panel_dreval_scatter(ax):
    """Leaderboard models: per-drug (raw) versus selective correlation within CTRPv2."""
    S, fm, _ = dreval_summary()
    for r in S.itertuples():
        col = DREVAL_COL[r.klass]
        ax.errorbar(r.raw, r.selective, xerr=[[r.raw - r.raw_lo], [r.raw_hi - r.raw]],
                    yerr=[[r.selective - r.selective_lo], [r.selective_hi - r.selective]], fmt=marker(r.model), ms=4.2,
                    color=col, mec='white', mew=0.5, elinewidth=0.5, ecolor=col, zorder=3)
    ax.axvline(fm, color=GRAY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(fm + 0.008, 0.12, 'measured fitness axis (not cell-blind)', rotation=90, ha='left', va='center', fontsize=5.4,
            color=INK2)
    label = {'ridge_sel_lm': (0.014, 0.0, 'left', 'center'), 'ridge_raw_lm': (0.0, 0.016, 'center', 'bottom'),
             'EnsembleMF': (-0.014, 0.004, 'right', 'center'), 'RandomForest': (0.014, -0.002, 'left', 'center'),
             'NaiveTissueDrugMeanPredictor': (-0.014, 0.004, 'right', 'center'),
             'fitness_only': (0.0, -0.05, 'center', 'center')}
    for m, (dx, dy, ha, va) in label.items():
        r = S[S.model == m]
        if len(r):
            ax.text(r.raw.iloc[0] + dx, r.selective.iloc[0] + dy, SHORT.get(m, DREVAL_NAMES.get(m, m)), fontsize=5.4,
                    color=INK2, ha=ha, va=va, zorder=4)
    r = S[S.model == 'fitness_only']
    ax.annotate('', xy=(r.raw.iloc[0], r.selective.iloc[0] - 0.006), xytext=(r.raw.iloc[0], r.selective.iloc[0] - 0.043),
                arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.5))
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlim(0.0, 0.6); ax.set_ylim(-0.062, 0.275); hgrid(ax)
    ax.set_xlabel('per-drug correlation (median over drugs)'); ax.set_ylabel('selective correlation (median over drugs)')
    for cls, lab in [('naive', 'naive baseline'), ('classical', 'classical ML'), ('neural/factorization', 'neural / factorization'),
                     ('this study', 'kernel ridge (this study)')]:
        ax.scatter([], [], s=14, color=DREVAL_COL[cls], label=lab)
    ax.scatter([], [], s=14, marker='^', color=GRAY, label='drug-agnostic')
    ax.legend(loc='upper left', fontsize=5.6, handletextpad=0.1, borderaxespad=0.1)
    ax.set_title('DrEval leaderboard re-scored (CTRPv2, unseen cell lines)', loc='left', color=INK)


def panel_dreval_collapse(ax):
    """Predicted profiles collapse onto one cell-line axis."""
    S, _, obs_r2 = dreval_summary()
    for r in S.dropna(subset=['prediction_fitness_r2']).itertuples():
        ax.scatter(r.prediction_fitness_r2, r.selective, s=20, marker=marker(r.model), color=DREVAL_COL[r.klass],
                   edgecolor='white', lw=0.5, zorder=3)
    ax.axvline(obs_r2, color=GRAY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(obs_r2 + 0.012, 0.222, 'observed responses', ha='left', va='center', fontsize=5.4, color=INK2)
    lab2 = {'ridge_sel_lm': (0.02, 0.0, 'left'), 'ridge_raw_lm': (0.02, 0.0, 'left'), 'EnsembleMF': (0.02, 0.006, 'left'),
            'NaiveTissueDrugMeanPredictor': (0.0, 0.013, 'center')}
    ax.annotate('axis only, tissue mean, Lasso,\nElasticNet, MultiViewRandomForest', xy=(0.995, 0.012), xytext=(0.985, -0.042),
                ha='right', va='center', fontsize=5.2, color=INK2, arrowprops=dict(arrowstyle='-', color=GRAY, lw=0.5))
    for m, (dx, dy, ha) in lab2.items():
        r = S[S.model == m]
        if len(r) and np.isfinite(r.prediction_fitness_r2.iloc[0]):
            ax.text(r.prediction_fitness_r2.iloc[0] + dx, r.selective.iloc[0] + dy, SHORT.get(m, DREVAL_NAMES.get(m, m)),
                    fontsize=5.4, color=INK2, ha=ha, va='center')
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlim(0, 1.04); ax.set_ylim(-0.062, 0.275); hgrid(ax)
    ax.set_xlabel('shared-axis share of predicted profiles\n(median $R^2$ on the other drugs\' mean prediction)')
    ax.set_ylabel('selective correlation (median over drugs)')
    ax.set_title('Predictions collapse onto one cell-line axis', loc='left', color=INK)


def panel_dreval_decomposition(ax):
    """Per-drug correlation of each leaderboard model split into fitness-mediated and selective terms."""
    S, _, _ = dreval_summary()
    T = S.dropna(subset=['mean_raw']).sort_values('mean_raw', ascending=False)
    x = np.arange(len(T))
    sel_term = T.mean_selective_term.clip(lower=0)
    ax.bar(x, T.mean_mediated, width=0.66, color=LIGHT_GRAY, label='through the fitness axis')
    ax.bar(x, sel_term, bottom=T.mean_mediated, width=0.66, color=ORANGE, label='drug-specific (selective)')
    for xi, r in zip(x, T.itertuples()):
        ax.text(xi, max(r.mean_raw, r.mean_mediated) + 0.006, f'{100 * r.mediated_share:.0f}', ha='center', fontsize=5.2,
                color=INK2)
        ax.scatter(xi, -0.028, s=12, marker=marker(r.model), color=DREVAL_COL[r.klass], lw=0, clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels([DREVAL_NAMES.get(m, m) for m in T.model], rotation=40, ha='right', fontsize=5.6)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.set_xlim(-0.6, len(T) - 0.4); ax.set_ylim(0, 0.5)
    ax.set_ylabel('mean per-drug correlation')
    hgrid(ax)
    ax.legend(loc='upper right', fontsize=6, handlelength=1.0, borderaxespad=0.1)
    ax.text(len(T) - 0.4, 0.375, 'numbers: % of the per-drug correlation\ncarried by the fitness axis', ha='right', va='top',
            fontsize=5.6, color=INK2)
    ax.set_title('Per-drug correlation of each model: fitness-mediated and selective terms', loc='left', color=INK,
                 fontsize=6.8)



XLAB = RESULTS / 'dreval_crosslab_leaderboard'
TARGET_LABEL = {'CTRPv1': 'CTRPv1\nsame lab', 'CCLE': 'CCLE', 'GDSC2': 'GDSC2\nother lab,\nsame readout',
                'GDSC1': 'GDSC1\nother lab,\nother readout'}


def panel_crosslab_terms(ax, target='GDSC1'):
    """Same predictions scored in CTRPv2 and in another laboratory: fitness-mediated and selective terms."""
    S = pd.read_csv(XLAB / 'summary.csv')
    S = S[(S.target == target) & S['class'].isin(['classical', 'neural/factorization'])]
    lim = float(np.ceil(max(S.mediated_within.max(), S.mediated_cross.max()) * 20 + 0.5) / 20)
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.6, zorder=1)
    ax.plot([0, lim], [0, lim / 2], color=GRAY, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.text(lim * 0.72, lim * 0.36 - 0.012, 'half retained', ha='left', va='top', fontsize=5.4, color=INK2)
    ax.text(lim - 0.005, lim - 0.03, 'fully retained', ha='right', va='top', fontsize=5.4, color=INK2, rotation=0)
    ax.scatter(S.mediated_within, S.mediated_cross, s=16, color=LIGHT_GRAY, edgecolor=INK2, lw=0.4, zorder=3,
               label='through the fitness axis')
    ax.scatter(S.selterm_within, S.selterm_cross, s=16, color=ORANGE, edgecolor='white', lw=0.4, zorder=3,
               label='drug-specific (selective)')
    ax.set_xlim(-0.02, lim); ax.set_ylim(-0.02, lim)
    ax.set_xlabel('term of mean per-drug correlation,\nscored in CTRPv2 (training laboratory)')
    ax.set_ylabel(f'same predictions, scored in {target}')
    ax.legend(loc='upper left', fontsize=5.8, handletextpad=0.1, borderaxespad=0.1)
    hgrid(ax)
    ax.set_title(f'{len(S)} published models, same cell lines and drugs', loc='left', color=INK)


def panel_crosslab_retention(ax):
    """Retention of the fitness-mediated and selective terms by evaluation laboratory (median and IQR over models)."""
    S = pd.read_csv(XLAB / 'summary.csv')
    S = S[S['class'].isin(['classical', 'neural/factorization'])]
    targets = ['CTRPv1', 'GDSC2', 'GDSC1']
    w = 0.34
    for k, t in enumerate(targets):
        d = S[S.target == t]
        for off, col, val, keep in [(-w / 2 - 0.01, LIGHT_GRAY, 'mediated_retained', d.mediated_within > 0.05),
                                    (w / 2 + 0.01, ORANGE, 'selterm_retained', d.selterm_within > 0.03)]:
            v = d.loc[keep, val]
            q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
            ax.bar(k + off, q2, w, color=col, edgecolor=INK2 if col == LIGHT_GRAY else 'none', lw=0.4)
            ax.plot([k + off, k + off], [q1, q3], color=INK, lw=0.8)
    ax.axhline(1, color=GRAY, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(len(targets))); ax.set_xticklabels([TARGET_LABEL[t] for t in targets], fontsize=5.3)
    ax.set_ylabel('cross-laboratory / training-laboratory\n(median over models, IQR)')
    ax.set_ylim(0, 1.35); hgrid(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=LIGHT_GRAY, edgecolor=INK2, lw=0.4, label='fitness-mediated term'),
                       Patch(facecolor=ORANGE, label='selective term')],
              loc='upper right', fontsize=5.6, handlelength=1.0, borderaxespad=0.1)
    ax.set_title('Retained in another screen', loc='left', color=INK)


def figure2():
    fig = plt.figure(figsize=(FIG_W, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.62)
    top = gs[0].subgridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.32)
    ax = fig.add_subplot(top[0]); panel_label(ax, 'a', x=-0.12); panel_dreval_scatter(ax)
    ax = fig.add_subplot(top[1]); panel_label(ax, 'b', x=-0.16); panel_crosslab_terms(ax)
    bot = gs[1].subgridspec(1, 2, width_ratios=[2.2, 1.1], wspace=0.3)
    ax = fig.add_subplot(bot[0]); panel_label(ax, 'c', x=-0.07); panel_dreval_decomposition(ax)
    ax = fig.add_subplot(bot[1]); panel_label(ax, 'd', x=-0.3); panel_crosslab_retention(ax)
    save(fig, 'figure2_benchmarks')


def sfigure4():
    """This study's own within-screen predictors and global models; collapse of leaderboard predictions."""
    fig = plt.figure(figsize=(FIG_W, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.15], wspace=0.3, hspace=0.5)
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, 'a', x=-0.1); panel_own_models(ax)
    ax = fig.add_subplot(gs[0, 1]); panel_label(ax, 'b', x=-0.14); panel_global_models(ax)
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, 'c', x=-0.1); panel_dreval_collapse(ax)
    save(fig, 'supp_figure4_own_models')


# ------------------------------------------------------------------ Figure 3
def figure3():
    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0)
    cs = pd.read_csv(RESULTS / 'drug_classes' / 'class_summary.csv', index_col=0)
    fig = plt.figure(figsize=(FIG_W, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.7)

    ax = fig.add_subplot(gs[0]); panel_label(ax, 'a', x=-0.2)
    hi = dt['class'].isin(['MAPK (RAF/MEK/ERK)', 'EGFR/ERBB', 'MDM2–p53'])
    ax.scatter(dt.loc[~hi, 'rho_raw'], dt.loc[~hi, 'rho_sel'], s=9, color=LIGHT_GRAY, edgecolor='white', lw=0.4,
               label='other mechanisms')
    for cls, col in [('MAPK (RAF/MEK/ERK)', BLUE), ('EGFR/ERBB', ORANGE), ('MDM2–p53', AQUA)]:
        d = dt[dt['class'] == cls]
        ax.scatter(d.rho_raw, d.rho_sel, s=14, color=col, edgecolor='white', lw=0.5, label=cls, zorder=3)
    lim = [-0.2, 0.8]
    ax.plot(lim, lim, color=GRAY, lw=0.6, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('cross-screen correlation of raw profiles'); ax.set_ylabel('cross-screen correlation of\nselective residuals')
    ax.legend(loc='upper left', handletextpad=0.2, borderaxespad=0.1)
    r = np.corrcoef(dt.rho_raw, dt.rho_sel)[0, 1]
    ax.text(0.97, 0.04, f'{len(dt)} drugs in ≥2 discovery screens\nmedian raw {dt.rho_raw.median():.2f}, selective {dt.rho_sel.median():.2f}',
            transform=ax.transAxes, ha='right', fontsize=5.8, color=INK2)
    hgrid(ax)

    ax = fig.add_subplot(gs[1]); panel_label(ax, 'b', x=-0.55)
    cs = cs.sort_values('median_rho_sel')
    y = np.arange(len(cs))
    for k, (cls, row) in enumerate(cs.iterrows()):
        vals = dt.loc[dt['class'] == cls, 'rho_sel'].values
        ax.scatter(vals, np.full(len(vals), k) + np.random.default_rng(k).uniform(-0.18, 0.18, len(vals)),
                   s=5, color='#86b6ef', lw=0, zorder=2)
        ax.plot([row.median_rho_sel] * 2, [k - 0.32, k + 0.32], color=INK, lw=1.4, zorder=3)
        ax.text(0.86, k, f'{row.frac_reproducible * 100:.0f}%', va='center', fontsize=5.8, color=INK2)
    ax.text(0.86, len(cs) - 0.2, 'reproducible\n(ρ ≥ 0.3)', fontsize=5.6, color=INK2, va='bottom')
    ax.set_yticks(y); ax.set_yticklabels([f'{c} ({int(n)})' for c, n in zip(cs.index, cs.drugs)])
    ax.axvline(0.3, color=GRAY, lw=0.6, zorder=1)
    ax.set_xlim(-0.25, 0.85); ax.set_xlabel('cross-screen selective correlation (mean over screen pairs)')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, 'figure3_transferable_selectivity')


# ------------------------------------------------------------------ Figure 4
def figure4():
    ce = jload('ceiling/summary.json')
    voi = pd.read_csv(RESULTS / 'value_of_information' / 'l1000_two_way.csv')
    inc = pd.read_csv(RESULTS / 'value_of_information' / 'l1000_incremental.csv')
    fig = plt.figure(figsize=(FIG_W, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 1.0, 1.25], wspace=0.75)

    ax = fig.add_subplot(gs[0]); panel_label(ax, 'a', x=-0.5)
    rel = ce['screen_reliability']
    for k, s in enumerate(DISC):
        for cls, col, dx in [('all', BLUE, -0.12), ('reproducible', ORANGE, 0.12)]:
            v = rel.get(f'{s}|{cls}')
            if v:
                ax.errorbar(k + dx, v['reliability'], yerr=[[v['reliability'] - v['ci95'][0]], [v['ci95'][1] - v['reliability']]],
                            fmt='o', ms=4, color=col, ecolor=col, elinewidth=0.8, capsize=0,
                            label=('drugs in ≥3 screens' if cls == 'all' else 'reproducible drugs') if k == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[s] for s in DISC], rotation=30, ha='right')
    ax.set_ylabel('reliability of one screen\'s selective profile\n(share of variance shared across screens)')
    ax.set_ylim(0, 0.7); hgrid(ax); ax.legend(loc='upper left', handletextpad=0.1, borderaxespad=0.0)

    ax = fig.add_subplot(gs[1]); panel_label(ax, 'b', x=-0.5)
    c2 = ce['captured_fraction_c2']
    rows = [('fitness_omics', 'omics fitness'), ('lineage_raw', 'lineage only'), ('target_prior', 'target dependency'),
            ('ridge_raw', 'omics ridge (raw)'), ('ridge_sel', 'omics ridge (selective)'),
            ('ridge_crispr', '+ CRISPR'), ('replicate', 'one other screen\n(measured)')]
    for k, (m, lab) in enumerate(rows):
        for cls, col, dy in [('all', BLUE, 0.15), ('reproducible', ORANGE, -0.15)]:
            v = c2.get(f'{m}|{cls}')
            if v:
                ax.errorbar(v['c2'], k + dy, xerr=[[v['c2'] - v['ci95'][0]], [v['ci95'][1] - v['c2']]], fmt='o', ms=3.6,
                            color=col, ecolor=col, elinewidth=0.8, capsize=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows]); ax.invert_yaxis()
    ax.set_xlabel('share of reproducible selective variance\ncaptured ($c^2$, cell-blind)')
    ax.set_xlim(-0.01, 0.42); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.scatter([], [], color=BLUE, s=10, label='drugs in ≥3 screens'); ax.scatter([], [], color=ORANGE, s=10, label='reproducible drugs')
    ax.legend(loc='upper right', handletextpad=0.1, borderaxespad=0.0)

    ax = fig.add_subplot(gs[2]); panel_label(ax, 'c', x=-0.12)
    panel_crosslab_selectivity(ax)
    save(fig, 'figure4_ceiling')


def panel_crosslab_selectivity(ax):
    """Cross-laboratory selective correlation of every leaderboard model and of this study's per-drug kernel ridge."""
    S = pd.read_csv(XLAB / 'summary.csv')
    ref = pd.read_csv(RESULTS / 'dreval_crossstudy' / 'summary.csv')
    keep = S['class'].isin(['classical', 'neural/factorization', 'naive']) | S.model.isin(['ridge_sel_lm', 'ridge_raw_lm'])
    S = S[keep & (S.model != 'SVR')]
    order = S[S.target == 'GDSC2'].sort_values('sel_cross').model.tolist()
    y = {m: i for i, m in enumerate(order)}
    for t, mk, col_off in [('GDSC1', 'o', -0.17), ('GDSC2', 's', 0.17)]:
        d = S[S.target == t]
        for r in d.itertuples():
            colr = AQUA if r.model.startswith('ridge') else (GRAY if r.model.startswith('Naive') else BLUE)
            ax.scatter(r.sel_cross, y[r.model] + col_off, s=11, marker=mk, color=colr, lw=0, zorder=3)
        ceil = ref[(ref.algorithm == 'ref:CTRPv2_measured') & (ref.target == t) & (ref['mode'] == 'LCO')].sel_cross.iloc[0]
        ax.axvline(ceil, color=INK2, lw=0.7, ls=(0, (3, 2)) if t == 'GDSC1' else (0, (1, 1.5)))
        ax.text(ceil + 0.005, 2.0 if t == 'GDSC1' else 6.5, f'CTRPv2 measured\nresponse ({t})', fontsize=5.0, color=INK2,
                va='center')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DREVAL_NAMES.get(m, m) for m in order], fontsize=5.0)
    ax.axvline(0, color=GRAY, lw=0.5)
    ax.set_xlim(-0.03, 0.45)
    ax.set_xlabel('selective correlation in another laboratory\n(median over drugs; CTRPv2-trained, unseen cell lines)')
    ax.scatter([], [], s=11, marker='o', color=INK2, label='scored in GDSC1')
    ax.scatter([], [], s=11, marker='s', color=INK2, label='scored in GDSC2')
    ax.scatter([], [], s=11, color=BLUE, label='published model'); ax.scatter([], [], s=11, color=GRAY, label='naive baseline')
    ax.scatter([], [], s=11, color=AQUA, label='per-drug kernel ridge\n(this study)')
    ax.legend(loc='lower right', fontsize=5.3, handletextpad=0.1, borderaxespad=0.1)
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)


# ------------------------------------------------------------------ Supplementary: value of information
def sfigure5():
    """Value of information: measured vs generated vs retrieved responses; generic programs; complementarity."""
    voi = pd.read_csv(RESULTS / 'value_of_information' / 'l1000_two_way.csv')
    rb = pd.read_csv(RESULTS / 'value_of_information' / 'retrieval_baselines.csv')
    inc = pd.read_csv(RESULTS / 'value_of_information' / 'l1000_incremental.csv')
    fig = plt.figure(figsize=(FIG_W, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.9, 0.8], wspace=0.55)
    cols4 = [BLUE, ORANGE, AQUA, GRAY]

    def ci(v):
        return [float(x) for x in str(v).strip('()[]').split(',')]
    ax = fig.add_subplot(gs[0]); panel_label(ax, 'a', x=-0.62)
    rows = [('measured', 'measured L1000 response', voi), ('xpert_released_generator', 'XPert, released (training-exposed)', voi),
            ('released_generator', 'TranSiGen, released (training-exposed)', voi),
            ('heldout_chemical_only_generator', 'TranSiGen, chemistry held out', voi),
            ('heldout_joint_generator', 'TranSiGen, chemistry + cell held out', voi),
            ('nearest_chemistry', 'nearest training chemistry (measured)', rb),
            ('same_target', 'shared-target training drugs (measured)', rb)]
    for k, (f, lab, tab) in enumerate(rows):
        for j, sname in enumerate(DISC):
            r = tab[(tab.features == f) & (tab['eval'] == sname)]
            if not len(r):
                continue
            lo, hi = ci(r.map_ci95.iloc[0]); m = r.map_r.iloc[0]
            ax.errorbar(m, k + (j - 1.5) * 0.15, xerr=[[m - lo], [hi - m]], fmt='o', ms=3.0, color=cols4[j], elinewidth=0.7,
                        label=LABEL[sname] if k == 0 else None)
    ax.axhspan(4.5, 6.5, color='#f4f8fd', zorder=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=6); ax.invert_yaxis()
    ax.axvline(0, color=GRAY, lw=0.6); ax.set_xlim(-0.15, 0.62)
    ax.set_xlabel('drug-specific correlation with selective residual')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1]); panel_label(ax, 'b', x=-0.32)
    ks = ['measured', 'measured_minus_1pc', 'measured_minus_3pc', 'measured_minus_10pc']
    for j, sname in enumerate(DISC):
        vals = [rb[(rb.features == k) & (rb['eval'] == sname)].map_r.iloc[0] for k in ks]
        ax.plot(range(len(ks)), vals, '-o', color=cols4[j], ms=3.5, lw=1.4, label=LABEL[sname])
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(['all', '-1 PC', '-3 PCs', '-10 PCs'])
    ax.set_xlabel('measured response, global\nprograms removed'); ax.set_ylabel('correlation with selective residual')
    ax.set_ylim(0, 0.62); hgrid(ax)
    ax.legend(title='evaluated in', loc='upper right', ncol=2, fontsize=5.4, title_fontsize=5.6, handletextpad=0.2,
              columnspacing=0.5, borderaxespad=0.1)

    ax = fig.add_subplot(gs[2]); panel_label(ax, 'c', x=-0.42)
    for j, sname in enumerate(DISC):
        d = inc[inc['eval'] == sname]
        ax.scatter(d.basal_r, d.combined_r, s=16, color=cols4[j], edgecolor='white', lw=0.4, zorder=3)
    ax.plot([0, 0.6], [0, 0.6], color=GRAY, lw=0.6)
    ax.set_xlim(0.1, 0.6); ax.set_ylim(0.1, 0.6)
    ax.set_xlabel('basal-omics model'); ax.set_ylabel('basal model + measured response')
    ax.text(0.12, 0.56, f'partial r of measured\n| basal: {inc.measured_given_basal.median():.2f}', fontsize=5.6, color=INK2, va='top')
    hgrid(ax)
    save(fig, 'supp_figure5_value_of_information')


# ------------------------------------------------------------------ Figure 5
def figure5():
    """Pre-registered confirmation in five untouched screens and transfer to primary AML and PDX-derived samples."""
    sets = [('gCSI', 'gCSI'), ('NCI60', 'NCI-60'), ('CCLE', 'CCLE'), ('FIMM', 'FIMM'), ('GRAY', 'GRAY')]
    H = {k: jload(f'confirmation_{k}/hypotheses.json') for k, _ in sets}
    fig = plt.figure(figsize=(FIG_W, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.15, 0.95], height_ratios=[1.0, 0.95], wspace=0.95, hspace=0.62)

    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, 'a', x=-0.42)
    hyps = ['H1', 'H2', 'H3', 'H4', 'H5a', 'H5b']
    for j, (k, lab) in enumerate(sets):
        h = H[k]
        h4p = h['H4']['median_difference'] >= 0.15 and (h['H4']['wilcoxon_p_one_sided'] is not None and h['H4']['wilcoxon_p_one_sided'] < 0.05)
        status = [h['H1']['pass'], h.get('H2', {}).get('pass'), h['H3'].get('pass'), h4p, h['H5'].get('pass_a'), h['H5'].get('pass_b')]
        for i, st in enumerate(status):
            if st is None:
                ax.text(j, i, 'n.e.', ha='center', va='center', fontsize=5.5, color=GRAY)
            else:
                ax.scatter(j, i, s=60, marker='o' if st else 'o', color=BLUE if st else 'white', edgecolor=BLUE, lw=1.2)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[1] for s in sets], rotation=30, ha='right')
    ax.set_yticks(range(len(hyps))); ax.set_yticklabels(['H1 fitness axis\nscreen-specific', 'H2 agreement\nis selective',
                                                          'H3 discovery class\npredicts agreement', 'H4 fitness does\nnot transfer',
                                                          'H5a model > 0', 'H5b measurement\n> model'], fontsize=5.6)
    ax.invert_yaxis(); ax.set_xlim(-0.6, len(sets) - 0.4)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('filled: passed; open: not met; n.e.: not estimable', fontsize=5.6, color=INK2, loc='left')

    ax = fig.add_subplot(gs[0, 1:]); panel_label(ax, 'b', x=-0.1)
    pos = 0; ticks = []
    for k, lab in sets:
        sc = pd.read_csv(RESULTS / f'confirmation_{k}' / 'scores.csv')
        for cls, col in [('non_reproducible', LIGHT_GRAY), ('reproducible', BLUE)]:
            v = sc[(sc.family == 'rep_sel_cons') & (sc['class'] == cls)].sel.dropna().values
            if len(v):
                x = pos + np.random.default_rng(int(pos * 10)).uniform(-0.15, 0.15, len(v))
                ax.scatter(x, v, s=5, color=col, lw=0, alpha=0.8, zorder=2)
                ax.plot([pos - 0.25, pos + 0.25], [np.median(v)] * 2, color=INK, lw=1.2, zorder=3)
            pos += 0.8
        ticks.append(pos - 1.2); pos += 0.6
    ax.set_xticks(ticks); ax.set_xticklabels([s[1] for s in sets], fontsize=6)
    ax.axhline(0, color=GRAY, lw=0.6); hgrid(ax)
    ax.set_ylabel('agreement of discovery consensus\nwith untouched screen (selective)')
    ax.scatter([], [], s=12, color=LIGHT_GRAY, label='non-reproducible in discovery')
    ax.scatter([], [], s=12, color=BLUE, label='reproducible in discovery')
    ax.legend(loc='lower left', fontsize=5.6, handletextpad=0.1, borderaxespad=0.1)

    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, 'c', x=-0.42)
    panel_patient_transfer(ax, 'confirmation_BeatAML', None,
                           'cell-line-to-patient transfer\n(BeatAML AML samples, selective)', top=5)
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, 'd', x=-0.3)
    panel_patient_transfer(ax, 'confirmation_PDX_Bruna', None,
                           'cell-line-to-PDX transfer\n(breast PDX-derived cells, selective)', top=5, gap=0.055)
    ax = fig.add_subplot(gs[1, 2]); panel_label(ax, 'e', x=-0.3)
    panel_transfer_summary(ax)
    save(fig, 'figure5_confirmation')


def generic_names():
    """Lower-case generic drug name where the compound table has one (as in the supplementary tables)."""
    import re
    c = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    out = {}
    for k, r in c.iterrows():
        names = [n for n in str(r.all_names).split('|') if n and n != 'nan']
        generic = [n for n in names if re.fullmatch(r'[a-z][a-z\-]+', n)]
        if generic:
            out[k] = generic[0]
        elif re.fullmatch(r'\d{2,7}-\d{2}-\d', str(r['name'])) and names:
            out[k] = names[0]
        else:
            out[k] = str(r['name'])
    return out


def panel_patient_transfer(ax, folder, show, ylabel, top=0, gap=0.03):
    """Selective transfer of frozen cell-line-trained predictions by discovery reproducibility class."""
    ba = pd.read_csv(RESULTS / folder / 'scores.csv')
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound').name
    gen = generic_names()
    b = ba[ba.family == 'sel']
    for k, (cls, col, lab) in enumerate([('non_reproducible', LIGHT_GRAY, 'non-\nreproducible'), ('intermediate', '#86b6ef', 'inter-\nmediate'),
                                         ('reproducible', BLUE, 'reproducible')]):
        v = b[b['class'] == cls]
        x = k + np.random.default_rng(k).uniform(-0.16, 0.16, len(v))
        ax.scatter(x, v.sel, s=9, color=col, lw=0, zorder=2)
        ax.plot([k - 0.28, k + 0.28], [v.sel.median()] * 2, color=INK, lw=1.2, zorder=3)
        if cls == 'reproducible':
            if show is None:
                best = v.sort_values('sel', ascending=False).head(top)
                show = {str(comp.get(c, '')): gen.get(c, str(comp.get(c, ''))) for c in best.compound}
            placed = []
            for xi, (_, r) in sorted(zip(x, v.iterrows()), key=lambda t: -t[1][1].sel):
                nm = show.get(str(comp.get(r.compound, '')))
                if nm:
                    y = r.sel
                    while any(abs(y - q) < gap for q in placed):
                        y -= gap
                    placed.append(y)
                    ax.text(k + 0.3, y, nm, fontsize=5.2, color=INK2, va='center')
                    ax.plot([xi, k + 0.28], [r.sel, y], color=LIGHT_GRAY, lw=0.4)
    ax.set_xticks(range(3)); ax.set_xticklabels(['non-\nreproducible', 'inter-\nmediate', 'reproducible'], fontsize=6)
    ax.axhline(0, color=GRAY, lw=0.6); hgrid(ax)
    ax.set_ylabel(ylabel)


def panel_transfer_summary(ax):
    """Median selective transfer of reproducible versus non-reproducible drugs in the two patient-derived sets."""
    sets = [('confirmation_BeatAML', 'BeatAML\n(H6, H7)'), ('confirmation_PDX_Bruna', 'PDX_Bruna\n(H8, H9)')]
    for k, (folder, lab) in enumerate(sets):
        b = pd.read_csv(RESULTS / folder / 'scores.csv')
        b = b[b.family == 'sel']
        for off, cls, col in [(-0.17, 'non_reproducible', LIGHT_GRAY), (0.17, 'reproducible', BLUE)]:
            v = b[b['class'] == cls].sel.dropna()
            m = v.median()
            bs = np.median(v.values[np.random.default_rng(1).integers(0, len(v), (2000, len(v)))], axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ax.bar(k + off, m, 0.3, color=col)
            ax.plot([k + off] * 2, [lo, hi], color=INK, lw=0.8)
    ax.set_xticks(range(len(sets))); ax.set_xticklabels([s[1] for s in sets], fontsize=6)
    ax.axhline(0, color=GRAY, lw=0.6); hgrid(ax)
    ax.set_ylabel('median selective transfer\n(95% drug-bootstrap interval)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=LIGHT_GRAY, label='non-reproducible'), Patch(facecolor=BLUE, label='reproducible')],
              loc='upper left', fontsize=5.6, handlelength=1.0, borderaxespad=0.1)


# ------------------------------------------------------------------ Supplementary figures
def sfigure1():
    """Fitness axes against CRISPR-inferred growth rate, and their lineage structure."""
    axes_df = pd.read_csv(RESULTS / 'fitness_axis' / 'fitness_axes.csv', index_col=0)
    growth = pd.read_csv(PROCESSED / 'features' / 'growth_rate.csv', index_col=0).growth_rate.dropna()
    fa = jload('fitness_axis/summary.json')['per_screen']
    fig, axs = plt.subplots(2, 3, figsize=(FIG_W, 4.4))
    for ax, s in zip(axs.flat, ALL6):
        f = axes_df[s].dropna(); c = f.index.intersection(growth.index)
        ax.scatter(growth[c], f[c], s=5, color=BLUE, lw=0, alpha=0.6)
        ax.set_title(f'{LABEL[s]}  (Spearman {fa[s]["growth_rate_spearman"]:.2f}, n={len(c)})', loc='left', color=INK, fontsize=6.8)
        ax.set_xlabel('CRISPR-inferred growth rate'); ax.set_ylabel('general-sensitivity axis (mean z)')
        hgrid(ax)
    fig.tight_layout()
    save(fig, 'supp_figure1_growth_rate')


def sfigure2():
    """Tahoe-100M measured responses versus selective residuals (pooled within-drug design)."""
    t = pd.read_csv(RESULTS / 'value_of_information' / 'tahoe_pooled.csv')
    feats = [('measured_response', 'measured response\n(drug-blind ridge)'), ('response_strength', 'response strength'),
             ('abundance_24h', '24-h cell abundance'), ('basal_expression', 'basal expression\n(drug-blind ridge)')]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for k, (f, lab) in enumerate(feats):
        for j, s in enumerate(DISC):
            r = t[(t.features == f) & (t['eval'] == s)]
            if len(r):
                ax.scatter(r.ridge_map_vs_selective.iloc[0], k + (j - 1.5) * 0.15, s=14, color=[BLUE, ORANGE, AQUA, GRAY][j],
                           label=LABEL[s] if k == 0 else None)
    rep = t[t.features.str.startswith('replicate_')].groupby('eval').ridge_map_vs_selective.median()
    for j, s in enumerate(DISC):
        ax.scatter(rep[s], len(feats) + (j - 1.5) * 0.15, s=14, marker='s', color=[BLUE, ORANGE, AQUA, GRAY][j])
    ax.set_yticks(range(len(feats) + 1)); ax.set_yticklabels([l for _, l in feats] + ['viability in another\nscreen (replicate)'])
    ax.invert_yaxis(); ax.axvline(0, color=GRAY, lw=0.6)
    ax.set_xlabel('pooled within-drug correlation with selective residual')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(title='evaluated in', loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=6, title_fontsize=6)
    save(fig, 'supp_figure2_tahoe')


def sfigure3():
    """DrEval re-analysis: paired per-drug comparison, reproducible drugs and feature variants of the kernel ridge."""
    per = pd.read_csv(RESULTS / 'dreval' / 'per_drug.csv.gz', dtype={'drug': str})
    S = pd.read_csv(RESULTS / 'dreval' / 'model_summary.csv').rename(columns={'class': 'klass'})
    fig = plt.figure(figsize=(FIG_W, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.35, 0.95], wspace=0.45)

    ax = fig.add_subplot(gs[0]); panel_label(ax, 'a', x=-0.3)
    W = per.pivot_table(index='drug', columns='model', values='selective')
    d = W[['ridge_sel_lm', 'EnsembleMF']].dropna()
    ax.scatter(d.EnsembleMF, d.ridge_sel_lm, s=5, color=BLUE, lw=0, alpha=0.7)
    ax.plot([-0.4, 0.8], [-0.4, 0.8], color=GRAY, lw=0.6)
    ax.set_xlim(-0.3, 0.75); ax.set_ylim(-0.3, 0.75)
    ax.set_xlabel('EnsembleMF (best published)'); ax.set_ylabel('kernel ridge, selective target\n(landmark genes)')
    ax.text(0.73, -0.27, f'{len(d)} drugs; ridge higher for {100 * (d.ridge_sel_lm > d.EnsembleMF).mean():.0f}%',
            ha='right', fontsize=5.6, color=INK2)
    ax.set_title('per-drug selective correlation', loc='left', color=INK, fontsize=6.8)
    hgrid(ax)

    ax = fig.add_subplot(gs[1]); panel_label(ax, 'b', x=-0.16)
    T = S[(S.drugs > 0) & ~S.model.isin(['SVR', 'fitness_measured', 'ridge_raw_expr', 'ridge_sel_expr', 'ridge_raw', 'ridge_sel'])]
    T = T.sort_values('selective', ascending=False)
    x = np.arange(len(T))
    ax.scatter(x, T.selective, s=12, color=LIGHT_GRAY, label='all drugs', zorder=3)
    ax.scatter(x, T.reproducible_selective, s=12, color=BLUE, label='drugs reproducible in discovery', zorder=3)
    for xi, (a_, b_) in enumerate(zip(T.selective, T.reproducible_selective)):
        ax.plot([xi, xi], [a_, b_], color=LIGHT_GRAY, lw=0.6, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([SHORT.get(m, DREVAL_NAMES.get(m, m)) for m in T.model], rotation=60, ha='right',
                                         fontsize=5.0)
    ax.axhline(0, color=GRAY, lw=0.6); hgrid(ax)
    ax.set_ylabel('selective correlation (median)')
    ax.legend(loc='upper right', fontsize=5.6, handletextpad=0.1)

    ax = fig.add_subplot(gs[2]); panel_label(ax, 'c', x=-0.3)
    V = S.set_index('model')
    groups = [('multi-\nomics', ''), ('expression\n(top 5,000)', '_expr'), ('expression\n(landmark)', '_lm')]
    w = 0.36
    for k, (lab, sfx) in enumerate(groups):
        ax.bar(k - w / 2, V.loc['ridge_raw' + sfx, 'selective'], w, color='#86b6ef', label='raw target' if k == 0 else None)
        ax.bar(k + w / 2, V.loc['ridge_sel' + sfx, 'selective'], w, color=AQUA, label='selective target' if k == 0 else None)
    ax.axhline(V.loc['EnsembleMF', 'selective'], color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(-0.55, 2.55)
    ax.text(2.6, V.loc['EnsembleMF', 'selective'], 'best\npublished', ha='left', va='center', fontsize=5.4, color=INK2,
            clip_on=False)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=5.6)
    ax.set_ylabel('selective correlation (median)'); ax.set_ylim(0, 0.32); hgrid(ax)
    ax.legend(loc='upper left', fontsize=5.6, handlelength=1.0)
    ax.set_title('kernel ridge features', loc='left', color=INK, fontsize=6.8)
    save(fig, 'supp_figure3_dreval')


def sfigure6():
    """DrEval's full results: same predictions scored in CTRPv2 and in another screen, for every test mode."""
    S = pd.read_csv(RESULTS / 'dreval_crossstudy' / 'summary.csv')
    S = S[~S.algorithm.str.startswith('Naive') & ~S.algorithm.str.startswith('ref:')]
    modes = [('LCO', 'unseen cell lines'), ('LTO', 'unseen tissues'), ('LPO', 'random pairs'), ('LDO', 'unseen drugs')]
    targets = ['CTRPv1', 'GDSC2', 'GDSC1']
    fig, axs = plt.subplots(1, 4, figsize=(FIG_W, 2.5), sharey=True)
    for ax, (mode, lab) in zip(axs, modes):
        for k, t in enumerate(targets):
            d = S[(S['mode'] == mode) & (S.target == t)]
            for off, a, b, col in [(-0.2, 'raw_within', 'raw_cross', BLUE), (0.2, 'sel_within', 'sel_cross', ORANGE)]:
                ax.scatter([k + off - 0.06] * len(d), d[a], s=6, color=col, alpha=0.35, lw=0)
                ax.scatter([k + off + 0.06] * len(d), d[b], s=6, color=col, lw=0)
                ax.plot([k + off - 0.12, k + off], [d[a].median()] * 2, color=INK2, lw=1.0)
                ax.plot([k + off, k + off + 0.12], [d[b].median()] * 2, color=INK, lw=1.4)
        ax.set_xticks(range(len(targets))); ax.set_xticklabels(targets, fontsize=6)
        ax.axhline(0, color=GRAY, lw=0.5); hgrid(ax)
        ax.set_title(f'{mode} ({lab})', loc='left', color=INK, fontsize=6.6)
    axs[0].set_ylabel('median over drugs (each dot a model)')
    axs[0].scatter([], [], s=8, color=BLUE, label='per-drug (raw)'); axs[0].scatter([], [], s=8, color=ORANGE, label='selective')
    axs[0].legend(loc='upper right', fontsize=5.6, handletextpad=0.1)
    fig.text(0.5, -0.04, 'left of each pair: scored in CTRPv2 (faded); right: same predictions scored in the named screen',
             ha='center', fontsize=6, color=INK2)
    fig.tight_layout()
    save(fig, 'supp_figure6_crossstudy_modes')


def sfigure7():
    """DrEval's own normalized Pearson (drug and tissue means removed), leaderboard models and this study's references."""
    R = pd.read_csv(RESULTS / 'dreval' / 'normalized_metric.csv')
    R = R[~R.model.isin(['SVR'])].sort_values('normalized_pearson')
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    y = np.arange(len(R))
    cols = [AQUA if c == 'this study' else (GRAY if c == 'naive' else BLUE) for c in R['class']]
    ax.barh(y, R.normalized_pearson, xerr=R.split_sd, color=cols, error_kw=dict(lw=0.6, ecolor=INK2))
    names = {'fitness_only': 'axis only (full scale)', 'fitness_increment': 'axis only (increment)',
             'fitness_increment_tissue_centred': 'axis only (within tissue)',
             'ridge_raw_lm': 'kernel ridge, raw target (landmark genes)', 'ridge_sel_lm': 'kernel ridge, selective target (landmark genes)',
             'ridge_raw_expr': 'kernel ridge, raw target (5,000 genes)', 'ridge_sel_expr': 'kernel ridge, selective target (5,000 genes)',
             'ridge_raw': 'kernel ridge, raw target (multi-omics)', 'ridge_sel': 'kernel ridge, selective target (multi-omics)'}
    ax.set_yticks(y); ax.set_yticklabels([names.get(m, DREVAL_NAMES.get(m, m)) for m in R.model], fontsize=5.4)
    ax.axvline(0, color=GRAY, lw=0.5)
    ax.set_xlabel("DrEval normalized Pearson (mean over 10 splits, $\\pm$ s.d.)")
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, 'supp_figure7_dreval_normalized')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', 's1', 's2', 's3', 's4', 's5', 's6', 's7']
    for w in which:
        globals()[f's{"figure" + w[1:]}' if w.startswith('s') else f'figure{w}']()
        print('figure', w, 'done')
