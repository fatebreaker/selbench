"""Revision (round 1), fitness-axis workstream, part 1: active versus inactive drugs (R2-1, R1-M1) and lineage /
growth-pattern structure of the axes (R2 minor 1-2). Exploratory; no frozen file is touched.

Definitions follow the paper:
* Discovery screens (PRISM, CTRP, GDSC1, GDSC2; sensitivity = -AUC): z matrices of 02_decompose.py
  (>=30 cells per drug, >=20 drugs per cell, rank-normal per drug), read from results/decomposition/z_*.parquet.
* Activity (02_decompose.activity_stats / 14_collect_numbers): a drug is ACTIVE in a screen if >=10% of its measured
  cell lines have AUC < 0.8 (drug_screen_stats.frac_auc_below_0p8 >= 0.1), else INACTIVE.
* Fitness axis of a screen (11_fitness_axis.py): per cell line, mean z over the drugs that define the axis.
  Variants: all drugs; active drugs; inactive drugs; and, per screen pair, drugs measured AND active in both screens.
  Subset axes require >=10 contributing drugs per cell line.
* Split-half reliability: random drug halves of the variant's own drug set, Spearman-Brown, median of 200 splits.
  Disattenuated r = r / sqrt(rel_a rel_b) with the reliabilities of the same variant.
* Triad shared fraction (20_robustness.shared_fraction): lambda_s^2 = r_st r_su / r_tu over triads with >=30 common
  lines and |r_tu| >= 0.1; median over triads; 95% interval from 200 cell bootstraps.
* Lineage: OncotreeLineage (DepMap 24Q4 Model.csv); lineage R^2 = squared correlation of the axis with its lineage
  means (one-way ANOVA eta^2, as in 11_fitness_axis) and its adjusted version; suspension effect as in 11 (GrowthPattern
  'Suspension' minus all other lines), converted to s.d. units (divided by the axis s.d.), plus Cohen's d of
  suspension versus 'Adherent' lines only.

Outputs (results/revision_axis/): activity_fraction.csv, axis_variants.csv, axis_pairs_activity.csv,
triad_shared_fraction.csv, lineage_by_screen.csv, axis_pairs_lineage.csv, axes_native.csv, summary_45.json.
Helpers defined here are imported by scripts 46-48.
"""
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import itertools  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common import PROCESSED, RESULTS, load_models, rank_normal, write_json  # noqa: E402

OUT = RESULTS / 'revision_axis'
DEC = RESULTS / 'decomposition'
DISC = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']
CONF = ['gCSI', 'gCSI_GRAOC', 'NCI60', 'CCLE', 'FIMM', 'GRAY']
N_SPLITS, N_BOOT, N_TRIAD_BOOT = 200, 1000, 200
MIN_SUBSET_DRUGS_PER_CELL = 10
ACTIVE_FRAC, AUC_CUT = 0.1, 0.8


# ----------------------------------------------------------------------------------------------------------- helpers
def corr(a, b, min_n=20):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def residualize_on(y, X):
    """OLS residual of y on X (X without intercept column; intercept added)."""
    X = np.c_[np.ones(len(y)), X]
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def partial_corr(a, b, Xcov):
    """Correlation of a and b after OLS removal of covariates Xcov (n x k), complete cases only."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    Xcov = np.asarray(Xcov, float).reshape(len(a), -1)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(Xcov).all(axis=1)
    if m.sum() < 20:
        return np.nan
    return corr(residualize_on(a[m], Xcov[m]), residualize_on(b[m], Xcov[m]))


def rn_matrix(M):
    return pd.DataFrame(np.vstack([rank_normal(r) for r in M.values]), index=M.index, columns=M.columns)


def build_z(long, value='sens', drug='compound', cell='depmap_id', min_cells=30, min_unique=10, min_drugs_per_cell=None):
    """Rank-normal drug x cell matrix. Without min_drugs_per_cell: the confirmation-screen rule of 11_fitness_axis
    (>=30 cells and >=10 distinct values per drug). With it: additionally the 02_decompose cell filter, then drugs
    re-filtered to >=min_cells."""
    M = long.pivot_table(index=drug, columns=cell, values=value, aggfunc='mean')
    M = M.loc[M.notna().sum(axis=1) >= min_cells]
    M = M.loc[[d for d in M.index if M.loc[d].dropna().nunique() >= min_unique]]
    if min_drugs_per_cell:
        M = M.loc[:, M.notna().sum(axis=0) >= min_drugs_per_cell]
        M = M.loc[M.notna().sum(axis=1) >= min_cells]
    return M, rn_matrix(M)


def axis_of(Z, drugs=None, min_drugs=None):
    """Per-cell mean z over `drugs` (all rows if None); cells with fewer than min_drugs contributing drugs -> NaN."""
    S = Z if drugs is None else Z.loc[Z.index.intersection(pd.Index(drugs))]
    ax = S.mean(axis=0)
    if min_drugs:
        ax[S.notna().sum(axis=0) < min_drugs] = np.nan
    return ax


def split_half(Z, rng, drugs=None, cells=None, n=N_SPLITS):
    """Median Spearman-Brown split-half reliability of the axis (random drug halves) and its 2.5/97.5 percentiles."""
    S = Z if drugs is None else Z.loc[Z.index.intersection(pd.Index(drugs))]
    if cells is not None:
        S = S.loc[:, S.columns.intersection(pd.Index(cells))]
    d = np.array(S.index)
    if len(d) < 4:
        return np.nan, np.nan, np.nan
    V = S.values
    h = len(d) // 2
    out = []
    for _ in range(n):
        p = rng.permutation(len(d))
        with warnings.catch_warnings():   # cells without any drug in one half -> NaN (excluded)
            warnings.simplefilter('ignore', RuntimeWarning)
            a, b = np.nanmean(V[p[:h]], axis=0), np.nanmean(V[p[h:]], axis=0)
        r = corr(a, b)
        out.append(2 * r / (1 + r) if np.isfinite(r) else np.nan)
    out = np.array(out)
    return float(np.nanmedian(out)), float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def loo_axis(Z):
    """Leave-one-drug-out mean z per cell, as a drug x cell matrix aligned to Z (02_decompose.loo_fitness)."""
    v = Z.values
    ok = np.isfinite(v)
    tot, cnt = np.nansum(v, axis=0), ok.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        G = (tot[None, :] - np.where(ok, v, 0.0)) / (cnt[None, :] - ok)
    return pd.DataFrame(G, index=Z.index, columns=Z.columns)


def fitness_r2(Z, G):
    """Per-drug squared correlation of z with its own (leave-out) axis row."""
    out = {}
    for i, d in enumerate(Z.index):
        out[d] = corr(Z.values[i], G.values[i], min_n=10) ** 2
    return pd.Series(out)


def residuals(Z, G):
    """OLS residual of each drug's z on its axis row (selective profile)."""
    R = np.full(Z.shape, np.nan)
    for i in range(Z.shape[0]):
        z, g = Z.values[i], G.values[i]
        m = np.isfinite(z) & np.isfinite(g)
        if m.sum() < 10:
            continue
        X = np.c_[np.ones(m.sum()), g[m]]
        R[i, m] = z[m] - X @ np.linalg.lstsq(X, z[m], rcond=None)[0]
    return pd.DataFrame(R, index=Z.index, columns=Z.columns)


def boot_ci(x, y, rng, n=N_BOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 20:
        return np.nan, np.nan
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def shared_fraction(axes, rng, n_boot=N_TRIAD_BOOT, min_cells=30, min_rtu=0.1):
    """One-factor triad estimate of each axis's squared loading on a factor shared across screens
    (20_robustness.shared_fraction, with per-triad output)."""
    out, rows = {}, []
    names = list(axes)
    for s in names:
        ests, boots = [], []
        for t, u in itertools.combinations([x for x in names if x != s], 2):
            cells = axes[s].dropna().index.intersection(axes[t].dropna().index).intersection(axes[u].dropna().index)
            if len(cells) < min_cells:
                continue
            a, b, c = (axes[k][cells].values for k in (s, t, u))
            rst, rsu, rtu = corr(a, b), corr(a, c), corr(b, c)
            if not np.isfinite(rtu) or abs(rtu) < min_rtu:
                continue
            ests.append(rst * rsu / rtu)
            rows.append({'screen': s, 't': t, 'u': u, 'cells': len(cells), 'r_st': rst, 'r_su': rsu, 'r_tu': rtu,
                         'lambda2': rst * rsu / rtu})
            bb = []
            for _ in range(n_boot):
                i = rng.integers(0, len(cells), len(cells))
                x1, x2, x3 = corr(a[i], b[i]), corr(a[i], c[i]), corr(b[i], c[i])
                bb.append(x1 * x2 / x3 if abs(x3) > 0.05 else np.nan)
            boots.append(bb)
        if ests:
            B = np.nanmedian(np.array(boots), axis=0)
            out[s] = {'triads': len(ests), 'lambda2_median': float(np.median(ests)),
                      'ci95': [float(np.nanpercentile(B, 2.5)), float(np.nanpercentile(B, 97.5))]}
    return out, pd.DataFrame(rows)


def load_discovery_z():
    return {s: pd.read_parquet(DEC / f'z_{s}.parquet') for s in DISC}


def load_confirmation_long(name):
    fn = {'gCSI_GRAOC': 'gCSI_GRAOC_long.csv.gz'}.get(name, f'{name}_long.csv.gz')
    return pd.read_csv(PROCESSED / 'confirmation' / fn)


def confirmation_z(name):
    """11_fitness_axis.confirmation_z: >=30 cells per drug, >=10 distinct values; no cell filter."""
    M, Z = build_z(load_confirmation_long(name))
    return M, Z


def cell_annotations():
    m = load_models().drop_duplicates('ModelID').set_index('ModelID')
    return m.OncotreeLineage.fillna('Unknown'), m.GrowthPattern.fillna('Unknown')


def lineage_centre(ax, lineage):
    lin = lineage.reindex(ax.index).fillna('Unknown')
    return ax - ax.groupby(lin).transform('mean')


# ------------------------------------------------------------------------------------------------------------ main
def activity_tables(Zd):
    stats = pd.read_csv(DEC / 'drug_screen_stats.csv')
    act, rows = {}, []
    for s in DISC:
        st = stats[stats.screen == s].set_index('compound')
        assert set(st.index) == set(Zd[s].index), s
        a = set(st.index[st.frac_auc_below_0p8 >= ACTIVE_FRAC])
        act[s] = a
        rows.append({'screen': s, 'drugs': len(st), 'active': len(a), 'inactive': len(st) - len(a),
                     'frac_inactive': 1 - len(a) / len(st), 'median_fitness_r2_all_drugs': float(st.fitness_r2.median()),
                     'median_fitness_r2_active_drugs': float(st.loc[list(a)].fitness_r2.median()),
                     'median_fitness_r2_inactive_drugs': float(st.drop(index=list(a)).fitness_r2.median())})
    return act, pd.DataFrame(rows), stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4501)
    Zd = load_discovery_z()
    act, act_tab, stats = activity_tables(Zd)
    act_tab.to_csv(OUT / 'activity_fraction.csv', index=False)
    print(act_tab.round(3).to_string(index=False), flush=True)

    # ---------------- confirmation screens (axes and activity where an AUC-type endpoint exists) ----------------
    Zc, Mc = {}, {}
    for name in CONF:
        Mc[name], Zc[name] = confirmation_z(name)
    # gCSI: sens = -mean viability (an AUC-type summary on the 0-1 viability scale) -> same activity rule
    conf_active = {'gCSI': set(Mc['gCSI'].index[((-Mc['gCSI']) < AUC_CUT).sum(axis=1) / Mc['gCSI'].notna().sum(axis=1) >= ACTIVE_FRAC])}
    for name in ['CCLE', 'FIMM', 'GRAY']:   # AAC = 1 - normalized AUC
        M = Mc[name]
        conf_active[name] = set(M.index[((1 - M) < AUC_CUT).sum(axis=1) / M.notna().sum(axis=1) >= ACTIVE_FRAC])

    # ---------------- axis variants, reliabilities and fitness shares ----------------
    axes, var_rows, rel = {}, [], {}
    for s in DISC:
        Z = Zd[s]
        G = loo_axis(Z)
        r2_all = fitness_r2(Z, G)
        A = sorted(act[s]); I = sorted(set(Z.index) - act[s])
        for variant, drugs in [('all', None), ('active', A), ('inactive', I)]:
            ax = axis_of(Z, drugs, None if drugs is None else MIN_SUBSET_DRUGS_PER_CELL)
            axes[f'{s}|{variant}'] = ax
            rel[f'{s}|{variant}'] = split_half(Z, rng, drugs)
            row = {'screen': s, 'variant': variant, 'drugs': len(Z) if drugs is None else len(drugs),
                   'cells': int(ax.notna().sum()), 'split_half': rel[f'{s}|{variant}'][0],
                   'split_half_lo': rel[f'{s}|{variant}'][1], 'split_half_hi': rel[f'{s}|{variant}'][2]}
            if variant == 'all':
                row['median_r2_all_drugs_on_axis'] = float(r2_all.median())
                row['median_r2_active_drugs_on_axis'] = float(r2_all.loc[A].median())
            else:   # every drug regressed on the variant axis (leave-one-out for drugs that define it)
                Sv = Z.loc[drugs]
                tot, cnt = Sv.sum(axis=0, min_count=1).fillna(0).values, Sv.notna().sum(axis=0).values
                Gv = np.tile(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), (len(Z), 1))
                ok = np.isfinite(Z.values)
                inset = Z.index.isin(drugs)
                with np.errstate(invalid='ignore', divide='ignore'):
                    loo = (tot[None, :] - np.where(ok, Z.values, 0)) / (cnt[None, :] - ok)
                Gv[inset] = loo[inset]
                r2v = fitness_r2(Z, pd.DataFrame(Gv, index=Z.index, columns=Z.columns))
                row['median_r2_all_drugs_on_axis'] = float(r2v.median())
                row['median_r2_active_drugs_on_axis'] = float(r2v.loc[A].median())
            var_rows.append(row)
        print(s, 'done variants', flush=True)
    for name in CONF:
        Z = Zc[name]
        axes[f'{name}|all'] = axis_of(Z)
        rel[f'{name}|all'] = split_half(Z, rng)
        r2 = fitness_r2(Z, loo_axis(Z))
        row = {'screen': name, 'variant': 'all', 'drugs': len(Z), 'cells': int(Z.shape[1]), 'split_half': rel[f'{name}|all'][0],
               'split_half_lo': rel[f'{name}|all'][1], 'split_half_hi': rel[f'{name}|all'][2],
               'median_r2_all_drugs_on_axis': float(r2.median())}
        if name in conf_active:
            A = sorted(conf_active[name] & set(Z.index))
            row['median_r2_active_drugs_on_axis'] = float(r2.loc[A].median()) if A else np.nan
            row['active_drugs'] = len(A)
            if 4 <= len(A) < len(Z):
                axes[f'{name}|active'] = axis_of(Z, A, min(MIN_SUBSET_DRUGS_PER_CELL, max(2, len(A) // 2)))
                rel[f'{name}|active'] = split_half(Z, rng, A)
                var_rows.append({'screen': name, 'variant': 'active', 'drugs': len(A), 'cells': int(axes[f'{name}|active'].notna().sum()),
                                 'split_half': rel[f'{name}|active'][0], 'split_half_lo': rel[f'{name}|active'][1],
                                 'split_half_hi': rel[f'{name}|active'][2]})
        var_rows.append(row)
        print(name, 'done', flush=True)
    var_tab = pd.DataFrame(var_rows)
    var_tab.to_csv(OUT / 'axis_variants.csv', index=False)
    pd.DataFrame(axes).to_csv(OUT / 'axes_native.csv')

    # ---------------- pairwise agreement of discovery axes by activity ----------------
    prow = []
    for a, b in itertools.combinations(DISC, 2):
        cells = Zd[a].columns.intersection(Zd[b].columns)
        for variant in ['all', 'active', 'inactive']:
            x, y = axes[f'{a}|{variant}'].reindex(cells), axes[f'{b}|{variant}'].reindex(cells)
            r = corr(x, y); lo, hi = boot_ci(x, y, rng)
            ra, rb = rel[f'{a}|{variant}'][0], rel[f'{b}|{variant}'][0]
            prow.append({'a': a, 'b': b, 'variant': variant, 'cells': int((x.notna() & y.notna()).sum()),
                         'drugs_a': int(var_tab.query('screen==@a and variant==@variant').drugs.iloc[0]),
                         'drugs_b': int(var_tab.query('screen==@b and variant==@variant').drugs.iloc[0]),
                         'r': r, 'ci_lo': lo, 'ci_hi': hi, 'rel_a': ra, 'rel_b': rb, 'disattenuated': r / np.sqrt(ra * rb)})
        common = Zd[a].index.intersection(Zd[b].index)
        for variant, drugs in [('shared_all', common), ('shared_active_both', sorted(set(common) & act[a] & act[b])),
                               ('shared_inactive_both', sorted(set(common) - act[a] - act[b]))]:
            drugs = list(drugs)
            if len(drugs) < 4:
                prow.append({'a': a, 'b': b, 'variant': variant, 'drugs_a': len(drugs), 'drugs_b': len(drugs)})
                continue
            md = min(MIN_SUBSET_DRUGS_PER_CELL, max(2, len(drugs) // 2))
            x = axis_of(Zd[a].loc[drugs, cells], None, md)
            y = axis_of(Zd[b].loc[drugs, cells], None, md)
            r = corr(x, y); lo, hi = boot_ci(x, y, rng)
            ra = split_half(Zd[a].loc[drugs, cells], rng)[0]
            rb = split_half(Zd[b].loc[drugs, cells], rng)[0]
            prow.append({'a': a, 'b': b, 'variant': variant, 'cells': int((x.notna() & y.notna()).sum()),
                         'drugs_a': len(drugs), 'drugs_b': len(drugs), 'r': r, 'ci_lo': lo, 'ci_hi': hi,
                         'rel_a': ra, 'rel_b': rb, 'disattenuated': r / np.sqrt(ra * rb) if ra > 0 and rb > 0 else np.nan})
        print(a, b, 'pairs done', flush=True)
    pairs = pd.DataFrame(prow)
    pairs.to_csv(OUT / 'axis_pairs_activity.csv', index=False)
    print(pairs.round(3).to_string(index=False), flush=True)

    # ---------------- triad shared fractions ----------------
    # each set: screen -> axis variant used for that screen
    sets = {
        'all-drug axes (paper)': {s: 'all' for s in ['PRISM', 'CTRP', 'GDSC1', 'GDSC2', 'gCSI', 'NCI60']},
        'active-drug axes (discovery + gCSI active; NCI-60 all)': {**{s: 'active' for s in DISC + ['gCSI']}, 'NCI60': 'all'},
        'active-drug axes, NCI-60 excluded': {s: 'active' for s in DISC + ['gCSI']},
        'all-drug axes, NCI-60 excluded': {s: 'all' for s in DISC + ['gCSI']},
        'inactive-drug discovery axes (gCSI, NCI-60 all)': {**{s: 'inactive' for s in DISC}, 'gCSI': 'all', 'NCI60': 'all'},
    }
    trow, tdet = [], []
    for setname, spec in sets.items():
        res, det = shared_fraction({s: axes[f'{s}|{v}'].dropna() for s, v in spec.items()}, np.random.default_rng(4502))
        for s, v in res.items():
            variant = spec[s]
            relv = rel.get(f'{s}|{variant}', (np.nan,))[0]
            trow.append({'axis_set': setname, 'screen': s, 'variant': variant, 'triads': v['triads'], 'lambda2': v['lambda2_median'],
                         'ci_lo': v['ci95'][0], 'ci_hi': v['ci95'][1], 'split_half': relv,
                         'share_of_reliable_variance': v['lambda2_median'] / relv if relv and np.isfinite(relv) else np.nan})
        tdet.append(det.assign(axis_set=setname))
    tri = pd.DataFrame(trow)
    tri.to_csv(OUT / 'triad_shared_fraction.csv', index=False)
    pd.concat(tdet).to_csv(OUT / 'triad_details.csv', index=False)
    print(tri.round(3).to_string(index=False), flush=True)

    # ---------------- lineage and growth pattern (Task 6) ----------------
    lineage, growth = cell_annotations()
    lnic = {}
    long = pd.read_csv(PROCESSED / 'discovery_long.csv.gz')
    long['neg_ln_ic50'] = -long.ln_ic50
    for s in ['GDSC1', 'GDSC2']:
        _, z = build_z(long[long.screen == s], 'neg_ln_ic50', min_unique=1, min_drugs_per_cell=20)
        lnic[s] = z
        axes[f'{s}_lnIC50|all'] = axis_of(z)
        rel[f'{s}_lnIC50|all'] = split_half(z, rng)
    pd.DataFrame(axes).to_csv(OUT / 'axes_native.csv')
    lin_rows = []
    screen_axes = [f'{s}|all' for s in DISC] + [f'{s}|active' for s in DISC] + ['GDSC1_lnIC50|all', 'GDSC2_lnIC50|all'] + \
                  [f'{c}|all' for c in CONF]
    for key in screen_axes:
        ax = axes[key].dropna()
        lin = lineage.reindex(ax.index).fillna('Unknown')
        gp = growth.reindex(ax.index).fillna('Unknown')
        lm = ax.groupby(lin).transform('mean')
        eta2 = corr(ax, lm, min_n=5) ** 2
        k, n = lin.nunique(), len(ax)
        susp = gp == 'Suspension'
        adh = gp == 'Adherent'
        sd = ax.std()
        diff = ax[susp].mean() - ax[~susp].mean() if susp.sum() else np.nan
        pooled = np.sqrt(((susp.sum() - 1) * ax[susp].var() + (adh.sum() - 1) * ax[adh].var()) / (susp.sum() + adh.sum() - 2)) if susp.sum() > 1 and adh.sum() > 1 else np.nan
        lin_rows.append({'axis': key, 'cells': n, 'lineages': k, 'lineage_r2': eta2,
                         'lineage_r2_adj': 1 - (1 - eta2) * (n - 1) / (n - k) if n > k else np.nan,
                         'axis_sd': sd, 'n_suspension': int(susp.sum()), 'n_adherent': int(adh.sum()),
                         'n_other': int((~susp & ~adh).sum()),
                         'susp_minus_rest_raw': diff, 'susp_minus_rest_sd_units': diff / sd if susp.sum() else np.nan,
                         'susp_minus_adherent_raw': ax[susp].mean() - ax[adh].mean() if susp.sum() and adh.sum() else np.nan,
                         'cohens_d_susp_vs_adherent': (ax[susp].mean() - ax[adh].mean()) / pooled if np.isfinite(pooled) else np.nan})
    lin_tab = pd.DataFrame(lin_rows)
    lin_tab.to_csv(OUT / 'lineage_by_screen.csv', index=False)
    print(lin_tab.round(3).to_string(index=False), flush=True)

    # cross-screen axis agreement after lineage centring
    Zall = {**{f'{s}|all': Zd[s] for s in DISC}, **{f'{c}|all': Zc[c] for c in CONF},
            'GDSC1_lnIC50|all': lnic['GDSC1'], 'GDSC2_lnIC50|all': lnic['GDSC2']}
    rel_lc = {}
    for key, Z in Zall.items():   # split-half reliability of the lineage-centred axis
        V = Z.values; d = len(Z); h = d // 2; out = []
        lin = lineage.reindex(Z.columns).fillna('Unknown')
        r_rng = np.random.default_rng(4503)
        for _ in range(N_SPLITS):
            p = r_rng.permutation(d)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                a = pd.Series(np.nanmean(V[p[:h]], axis=0), index=Z.columns)
                b = pd.Series(np.nanmean(V[p[h:]], axis=0), index=Z.columns)
            a = a - a.groupby(lin).transform('mean'); b = b - b.groupby(lin).transform('mean')
            r = corr(a, b); out.append(2 * r / (1 + r))
        rel_lc[key] = float(np.nanmedian(out))
    lrows = []
    keys = list(Zall)
    for a, b in itertools.combinations(keys, 2):
        if a.split('|')[0].split('_')[0] == b.split('|')[0].split('_')[0]:
            continue   # same screen, different endpoint
        xa, xb = axes[a].dropna(), axes[b].dropna()
        cells = xa.index.intersection(xb.index)
        if len(cells) < 30:
            continue
        lin = lineage.reindex(cells).fillna('Unknown')
        r = corr(xa[cells], xb[cells])
        ca, cb = lineage_centre(xa, lineage)[cells], lineage_centre(xb, lineage)[cells]
        r_within = corr(ca, cb)
        D = pd.get_dummies(lin, drop_first=True).values.astype(float)
        r_part = partial_corr(xa[cells].values, xb[cells].values, D)
        lrows.append({'a': a, 'b': b, 'cells': len(cells), 'lineages': lin.nunique(), 'r': r,
                      'disattenuated': r / np.sqrt(rel[a][0] * rel[b][0]),
                      'r_lineage_centred_within_screen': r_within,
                      'r_partial_lineage_shared_cells': r_part,
                      'rel_a_centred': rel_lc[a], 'rel_b_centred': rel_lc[b],
                      'disattenuated_lineage_centred': r_within / np.sqrt(rel_lc[a] * rel_lc[b])})
    lpairs = pd.DataFrame(lrows)
    lpairs.to_csv(OUT / 'axis_pairs_lineage.csv', index=False)
    print(lpairs.round(3).to_string(index=False), flush=True)

    write_json(OUT / 'summary_45.json', {
        'activity': act_tab.to_dict(orient='records'),
        'confirmation_active_drugs': {k: len(v & set(Zc[k].index)) for k, v in conf_active.items()},
        'confirmation_drugs': {k: int(Zc[k].shape[0]) for k in CONF},
        'axis_variants': var_tab.to_dict(orient='records'),
        'pairs': pairs.to_dict(orient='records'),
        'triads': tri.to_dict(orient='records'),
        'lineage': lin_tab.to_dict(orient='records'),
        'lineage_pairs': lpairs.to_dict(orient='records'),
        'reliability': {k: list(v) for k, v in rel.items()},
        'reliability_lineage_centred': rel_lc,
        'settings': {'n_splits': N_SPLITS, 'n_boot': N_BOOT, 'n_triad_boot': N_TRIAD_BOOT,
                     'min_subset_drugs_per_cell': MIN_SUBSET_DRUGS_PER_CELL, 'active_rule': '>=10% of lines AUC<0.8',
                     'seeds': [4501, 4502, 4503]}})


if __name__ == '__main__':
    main()
