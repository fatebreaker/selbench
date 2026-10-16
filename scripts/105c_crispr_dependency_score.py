"""Round 5 (second domain): score the CRISPR gene-dependency benchmark (105b) exactly as the drug analysis (Box 1).

What: per gene and predictor, the exact split of the per-gene correlation r_pz = r_pg r_zg + sqrt((1 - r_pg^2)(1 -
r_zg^2)) r_pz.g (p OOF prediction, z the gene's rank-normal measured profile, g its leave-gene-out active-gene axis;
all pooled over the 10 folds), and per predictor the Box 1 summaries with two-way (cell line x gene) bootstrap
intervals (B = 1,000; cell weights inside every per-gene correlation, gene multiplicities when pooling; the same draws
for every predictor, so differences and pooled summaries are paired):
  axis capture A = sum r_pg r_zg / sum r_zg^2; selective capture S = sum (r_pz - r_pg r_zg) / sum (1 - r_zg^2);
  model axis share = sum r_pg r_zg / sum r_pz; perfect-predictor share = mean r_zg^2; share ratio; alignment ratio
  (mean r_pg / mean r_zg)  -- selbench.box1.capture;
  median raw and selective r; lineage-adjusted selective score r(p, z | g, lineage indicators) and lineage-adjusted
  selective capture -- scripts/91_lineage.py's engine (weighted moments under every draw), lineages with >=5 lines
  (the 12 lines of smaller lineages pooled into one 'Other (<5 lines)' indicator);
  the gene-agnostic floor's median raw r over the best learned model's (ratio of medians, per draw);
  the tissue-by-gene mean's median selective r against each learned model's (differences, per draw);
  rank intervals by median raw and by median selective r (box1.rank_intervals) and the rank shifts.
Target prioritization: per predictor and threshold t in {0.3, 0.4}, genes with raw r >= t, the fraction whose axis
term is at least half of r, and how many fall below t on the selective term (r_pz - r_pg r_zg) alone (also on the
partial correlation r_pz.g); intervals from the same draws.
The dependency axis: correlations (Pearson, Spearman; cell-line bootstrap intervals) of the measured active-gene axis,
the all-gene axis, the common-essential axis and the OOF omics-predicted axis with the CRISPR-inferred growth rate,
screen QC (Chronos model efficacy; Achilles screen NNMD, ROC AUC, Cas activity, doubling time, median essential
depletion), variance explained by lineage, screen library and jointly; split-half reliability (box1.axis_split_half).
Configurations: primary (active genes, active-gene axis), sd02 (sensitivity i: all non-common-essential genes with
SD >= 0.2, active-gene axis, models retrained on that set), allgene_axis (sensitivity ii: active genes scored against
the all-gene axis; same draws as primary, so the differences are paired).
Validation (summary.json 'checks'): engine point estimates against selbench.split_terms, box1.lineage_terms and
selbench.oracle_fitness_share; box1.capture against 91.reps.
Inputs: results/revision_round5/crispr_dependency/{inputs, predictions} (105b); data/raw/depmap24q4 QC files (105a).
Outputs (results/revision_round5/crispr_dependency/): per_gene.csv.gz, models.csv (primary), pooled.csv,
sensitivity.csv, predictability.csv, ranks.csv, axis.csv, summary.json; predictions/manifest.csv (the OOF prediction
files of 105b, cells x genes, with SHA-256; folds in inputs/cells.csv).
Usage: ../../.venv/bin/python 105c_crispr_dependency_score.py [--workers 12] [--boot 1000]
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for _p in [str(HERE), str(PROJECT / 'selbench')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import selbench  # noqa: E402
import box1  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


os.environ['CRISPR_BLAS_THREADS'] = '1'             # 105b reads it at import
lib = _load('lib77', '77_round2_lib.py')          # draws, summ, wmedian_rows (read-only reuse)
lin = _load('lin91', '91_lineage.py')             # engine, reps, summ_basic (read-only reuse)
bench = _load('b105', '105b_crispr_dependency_benchmark.py')

OUT = bench.OUT
INP, PRED = bench.INP, bench.PRED
DM = bench.DM
LEARNED, AGNOSTIC, TISSUE = bench.LEARNED, bench.AGNOSTIC, bench.TISSUE
MODELS = LEARNED + AGNOSTIC + TISSUE
NO_LINEAGE = {'tissue_gene_mean', 'tissue_axis'}  # in the span of the lineage indicators up to fold variation
STATS = lin.STATS
SEED = 105
THRESH = (0.3, 0.4)
CONFIGS = {'primary': ('active', 'G_active', 'in_active'), 'sd02': ('sd02', 'G_active', 'in_sd02'),
           'allgene_axis': ('active', 'G_all', 'in_active')}
DRUG_REFERENCE = {
    'source': 'task statement and HANDOFF.md (drug-response analysis)',
    'leaderboard_axis_capture_A': 0.93, 'leaderboard_selective_capture_S': 0.051,
    'model_axis_share_over_perfect_share_range_nine_settings': [2.4, 4.4],
    'leaderboard_perfect_share': 0.29, 'GDSC2_AUC_perfect_share': 0.10,
    'tissue_by_drug_mean': 'matches all but one published model on selectivity (leaderboard: only EnsembleMF exceeds)',
    'drug_agnostic_predictor': 'beats many published models on raw r (axis-only 0.41 vs EnsembleMF 0.42 on the '
                               'leaderboard); floor/best 0.33 (GDSC2 AUC) vs 0.75-0.96 (other cells)',
    'axis_split_half': '0.98-0.99 within a screen'}

_G = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def kind(m):
    return 'learned' if m in LEARNED else ('gene-agnostic' if m in AGNOSTIC else 'tissue-by-gene baseline')


# ----------------------------------------------------------------------------------------------------------------
# engine (91's weighted-moment engine, run over gene chunks in forked workers)
def _chunk(idx):
    Z, G, Pm, tis, W = _G['Z'], _G['G'], _G['Pm'], _G['tissue'], _G['W']
    Zc = Z.iloc[idx]
    out, models, drugs, ncell, ntis = lin.engine(Zc, G.loc[Zc.index], {m: P.loc[Zc.index] for m, P in Pm.items()},
                                                 tis, W, None, mask='common')
    return idx, out, ncell, ntis


def run_engine(Z, G, Pm, tissue, W, workers):
    _G.update({'Z': Z, 'G': G, 'Pm': Pm, 'tissue': tissue, 'W': W})
    chunks = [c for c in np.array_split(np.arange(len(Z)), max(4 * workers, 1)) if len(c)]
    R, M, D = W.shape[0], len(Pm), len(Z)
    arrs = {s: np.full((R, M, D), np.nan, np.float32) for s in STATS}
    ncell = np.zeros((M, D), int)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork')) as ex:
        for idx, out, nc, _ in ex.map(_chunk, chunks):
            for s in STATS:
                arrs[s][:, :, idx] = out[s]
            ncell[:, idx] = nc
    return arrs, ncell


# ----------------------------------------------------------------------------------------------------------------
# summaries
def stack(arrs, ix, V):
    a = {k: np.concatenate([arrs[k][:, i].astype(float) for i in ix], axis=1) for k in STATS}
    return a, np.concatenate([V] * len(ix), axis=1)


def group_stats(arrs, ix, V, lineage=True):
    """Per-draw summaries for models ix (pooled when several): box1.capture + 91.reps (lineage-adjusted)."""
    a, Vc = stack(arrs, ix, V)
    cap = box1.capture(a['raw'], a['r_pg'], a['r_zg'], sel=a['sel'], weights=Vc)
    st, ok, okl = lin.reps(a, Vc)
    out = {k: np.asarray(v, float) for k, v in cap.items() if k not in ('bound', 'drugs')}
    for k in ['median_raw', 'median_selective', 'mean_selective', 'median_selective_lin', 'mean_selective_lin',
              'selective_capture_lin', 'covariate_share', 'perfect_covariate_share']:
        out[k] = np.asarray(st[k], float)
    if not lineage:
        for k in ['median_selective_lin', 'mean_selective_lin', 'selective_capture_lin']:
            out[k] = np.full_like(out[k], np.nan)
    chk = max(float(np.nanmax(np.abs(cap[k] - st[k]))) for k in ['axis_capture', 'selective_capture', 'axis_share',
                                                                   'perfect_share'])
    return out, int(ok[0].sum()), chk


def summ_row(st):
    row = {}
    for k, v in st.items():
        row.update(lin.summ_basic(v, k) if k in lin.R2_TYPE else lib.summ(v, k))
    return row


def predictability(arrs, i, V, t):
    raw, rpg, rzg, sel = (arrs[k][:, i].astype(float) for k in ('raw', 'r_pg', 'r_zg', 'sel'))
    med = rpg * rzg
    selterm = raw - med
    ok = np.isfinite(raw) & np.isfinite(med) & np.isfinite(sel)
    w = np.where(ok, V, 0.0)
    pas = ok & (raw >= t)
    with np.errstate(invalid='ignore', divide='ignore'):
        n_pass = (w * pas).sum(1)
        out = {'n_pass_raw': n_pass,
               'frac_axis_term_ge_half_r': (w * (pas & (med >= 0.5 * raw))).sum(1) / n_pass,
               'n_axis_term_ge_half_r': (w * (pas & (med >= 0.5 * raw))).sum(1),
               'n_below_on_selective_term': (w * (pas & (selterm < t))).sum(1),
               'frac_below_on_selective_term': (w * (pas & (selterm < t))).sum(1) / n_pass,
               'n_below_on_partial': (w * (pas & (sel < t))).sum(1),
               'frac_below_on_partial': (w * (pas & (sel < t))).sum(1) / n_pass,
               'n_pass_selective_term': (w * (ok & (selterm >= t))).sum(1),
               'n_pass_partial': (w * (ok & (sel >= t))).sum(1), 'genes_scored': w.sum(1)}
    return out, pas[0]


def score_config(cfg, Z, G, Pm, tissue, W, V, genes, workers):
    t0 = time.time()
    models = list(Pm)
    arrs, ncell = run_engine(Z, G, Pm, tissue, W, workers)
    log(cfg, 'engine', f'{time.time() - t0:.0f}s')
    rows, per, pools, preds, ranks, checks = [], [], [], [], [], {}
    gstats = {}
    for i, m in enumerate(models):
        st, n, chk = group_stats(arrs, [i], V, lineage=m not in NO_LINEAGE)
        gstats[m] = st
        checks[f'capture_vs_reps_{m}'] = chk
        rows.append({'config': cfg, 'model': m, 'kind': kind(m), 'genes': n, **summ_row(st)})
    learned = [m for m in LEARNED if m in models]
    best = max(learned, key=lambda m: gstats[m]['median_raw'][0])
    tb = 'tissue_gene_mean'
    for r in rows:
        m = r['model']
        st = gstats[m]
        with np.errstate(invalid='ignore', divide='ignore'):
            if tb in gstats and m != tb:
                d = st['median_selective'] - gstats[tb]['median_selective']
                r.update(lib.summ(d, 'median_selective_minus_tissue_gene_mean'))
                lo, hi = r['median_selective_minus_tissue_gene_mean_lo'], r['median_selective_minus_tissue_gene_mean_hi']
                r['vs_tissue_gene_mean_selective'] = 'above' if lo > 0 else ('below' if hi < 0 else 'matches')
                r.update(lib.summ(st['mean_selective'] - gstats[tb]['mean_selective'],
                                  'mean_selective_minus_tissue_gene_mean'))
            for ag in AGNOSTIC:
                if ag in gstats and m != ag:
                    d = st['median_raw'] - gstats[ag]['median_raw']
                    r.update(lib.summ(d, f'median_raw_minus_{ag}'))
                    lo, hi = r[f'median_raw_minus_{ag}_lo'], r[f'median_raw_minus_{ag}_hi']
                    r[f'vs_{ag}_raw'] = 'above' if lo > 0 else ('below' if hi < 0 else 'matches')
            r.update(lib.summ(st['median_raw'] / gstats[best]['median_raw'], 'median_raw_over_best_learned'))
        r['best_learned_by_median_raw'] = best
    # pooled groups
    for gname, gm in [('learned', learned), ('gene_agnostic', [m for m in AGNOSTIC if m in models])]:
        ix = [models.index(m) for m in gm]
        st, n, chk = group_stats(arrs, ix, V, lineage=gname == 'learned')
        checks[f'capture_vs_reps_pool_{gname}'] = chk
        prow = {'config': cfg, 'group': gname, 'models': len(ix), 'model_list': ';'.join(gm), 'model_gene_pairs': n,
                **summ_row(st)}
        if gname == 'learned':
            meds = np.vstack([gstats[m]['median_selective'] for m in gm])
            prow.update(lib.summ(np.median(meds, 0) - gstats[tb]['median_selective'],
                                 'median_over_models_median_selective_minus_tissue_gene_mean'))
            prow['n_learned_median_selective_above_tissue_gene_mean_point'] = int(
                (meds[:, 0] > gstats[tb]['median_selective'][0]).sum())
            for ag in AGNOSTIC:
                prow.update(lib.summ(gstats[ag]['median_raw'] / gstats[best]['median_raw'],
                                     f'{ag}_median_raw_over_best_learned'))
            prow['best_learned_by_median_raw'] = best
            prow['n_learned_median_raw_below_omics_axis_point'] = int(sum(
                gstats[m]['median_raw'][0] < gstats['omics_axis']['median_raw'][0] for m in gm))
        pools.append(prow)
    # rank intervals and shifts (all predictors, and learned only)
    for scope, ms in [('all_predictors', models), ('learned', learned)]:
        rk = {}
        for score in ['median_raw', 'median_selective']:
            S = np.column_stack([gstats[m][score] for m in ms])
            rk[score] = box1.rank_intervals(S, ms).set_index('model')
        for m in ms:
            a, b = rk['median_raw'].loc[m], rk['median_selective'].loc[m]
            ranks.append({'config': cfg, 'scope': scope, 'model': m, 'kind': kind(m),
                          'median_raw': a.score, 'rank_raw': a['rank'], 'rank_raw_lo': a.rank_lo,
                          'rank_raw_hi': a.rank_hi, 'resolved_raw': a.resolved_vs_others,
                          'median_selective': b.score, 'rank_selective': b['rank'], 'rank_selective_lo': b.rank_lo,
                          'rank_selective_hi': b.rank_hi, 'resolved_selective': b.resolved_vs_others,
                          'rank_shift_selective_minus_raw': b['rank'] - a['rank']})
    # predictability
    pass_sets = {}
    for i, m in enumerate(models):
        for t in THRESH:
            st, pas = predictability(arrs, i, V, t)
            pass_sets[(m, t)] = set(np.asarray(genes)[pas])
            row = {'config': cfg, 'model': m, 'kind': kind(m), 'threshold': t}
            for k, v in st.items():
                row.update(lib.summ(v, k))
            preds.append(row)
    for row in preds:
        bp = pass_sets[(best, row['threshold'])]
        mine = pass_sets[(row['model'], row['threshold'])]
        row['best_learned'] = best
        row['n_overlap_with_best_learned_pass'] = len(bp & mine)
    # perfect-predictor reference rows: raw = 1 for every gene; axis term r_zg^2 >= 1/2 r iff r_zg^2 >= 0.5
    rzg = arrs['r_zg'][:, 0].astype(float)
    okp = np.isfinite(rzg)
    w = np.where(okp, V, 0.0)
    for t in THRESH:
        row = {'config': cfg, 'model': 'perfect', 'kind': 'perfect predictor', 'threshold': t}
        row.update(lib.summ(w.sum(1), 'n_pass_raw'))
        with np.errstate(invalid='ignore', divide='ignore'):
            row.update(lib.summ((w * (rzg ** 2 >= 0.5)).sum(1) / w.sum(1), 'frac_axis_term_ge_half_r'))
            row.update(lib.summ((w * ((1 - rzg ** 2) < t)).sum(1), 'n_below_on_selective_term'))
        preds.append(row)
    # per-gene point table (+ perfect predictor)
    sym = {g: bench.symbol(g) for g in genes}
    for i, m in enumerate(models):
        for j, g in enumerate(genes):
            if not np.isfinite(arrs['raw'][0, i, j]):
                continue
            raw, rpg, rzg0 = (float(arrs[k][0, i, j]) for k in ('raw', 'r_pg', 'r_zg'))
            per.append({'config': cfg, 'model': m, 'kind': kind(m), 'gene': g, 'symbol': sym[g],
                        'n_cells': int(ncell[i, j]), 'raw': raw, 'alignment_r_pg': rpg, 'loading_r_zg': rzg0,
                        'axis_term': rpg * rzg0, 'selective_term': raw - rpg * rzg0,
                        'selective_r_pz_g': float(arrs['sel'][0, i, j]),
                        'selective_lineage': float(arrs['sel_lin'][0, i, j]) if m not in NO_LINEAGE else np.nan,
                        'covariate_term_lineage': float(arrs['cov_term'][0, i, j]),
                        'r2_z_axis_lineage': float(arrs['r2z_lin'][0, i, j])})
    for j, g in enumerate(genes):
        r0 = float(arrs['r_zg'][0, 0, j])
        if np.isfinite(r0):
            per.append({'config': cfg, 'model': 'perfect', 'kind': 'perfect predictor', 'gene': g, 'symbol': sym[g],
                        'n_cells': int(ncell[0, j]), 'raw': 1.0, 'alignment_r_pg': r0, 'loading_r_zg': r0,
                        'axis_term': r0 ** 2, 'selective_term': 1 - r0 ** 2, 'selective_r_pz_g': 1.0,
                        'selective_lineage': 1.0, 'covariate_term_lineage': float(arrs['r2z_lin'][0, 0, j]),
                        'r2_z_axis_lineage': float(arrs['r2z_lin'][0, 0, j])})
    # validation of the engine against selbench / box1 on a sample of genes
    rng = np.random.default_rng(0)
    sample = rng.choice(len(genes), size=min(60, len(genes)), replace=False)
    d_split, d_lin = 0.0, 0.0
    tis = tissue.reindex(Z.columns).values
    for j in sample:
        g = genes[j]
        z, gg = Z.loc[g].values.astype(float), G.loc[g].reindex(Z.columns).values.astype(float)
        for i, m in enumerate(models):
            p = Pm[m].loc[g].reindex(Z.columns).values.astype(float)
            mk = np.isfinite(z) & np.isfinite(gg) & np.isfinite(p)
            st = selbench.split_terms(p[mk], z[mk], gg[mk])
            d_split = max(d_split, abs(st['raw'] - arrs['raw'][0, i, j]), abs(st['alignment'] - arrs['r_pg'][0, i, j]),
                          abs(st['loading'] - arrs['r_zg'][0, i, j]), abs(st['selective'] - arrs['sel'][0, i, j]))
            if m not in NO_LINEAGE:
                lt = box1.lineage_terms(p, z, gg, tis)
                d_lin = max(d_lin, abs(lt['selective_lineage'] - arrs['sel_lin'][0, i, j]),
                            abs(lt['covariate_term'] - arrs['cov_term'][0, i, j]))
    checks['engine_vs_selbench_split_terms_max_abs'] = d_split
    checks['engine_vs_box1_lineage_terms_max_abs'] = d_lin
    checks['perfect_share_vs_selbench_oracle_fitness_share'] = abs(
        selbench.oracle_fitness_share(Z, G) - gstats[models[0]]['perfect_share'][0])
    log(cfg, 'summaries', f'{time.time() - t0:.0f}s', 'best', best)
    return {'rows': rows, 'per': per, 'pools': pools, 'preds': preds, 'ranks': ranks, 'checks': checks,
            'gstats': gstats, 'best': best, 'arrs': arrs, 'runtime_s': time.time() - t0}


# ----------------------------------------------------------------------------------------------------------------
# axis analysis
def cell_boot_corr(x, y, B=1000, seed=0, method='pearson'):
    from scipy.stats import rankdata
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if method == 'spearman':
        x, y = rankdata(x), rankdata(y)
    r0 = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.corrcoef(x[i], y[i])[0, 1])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r0, float(lo), float(hi), int(m.sum())


def r2_ols(y, X):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[m], X[m]
    X1 = np.c_[np.ones(len(y)), X]
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ b
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    k = np.linalg.matrix_rank(X1) - 1
    adj = 1 - (1 - r2) * (len(y) - 1) / max(len(y) - k - 1, 1)
    return float(r2), float(adj), int(len(y)), int(k)


def r2_boot(y, X, B=1000, seed=0):
    """In-sample R^2 and adjusted R^2 with cell-line bootstrap intervals for the adjusted R^2: percentile and basic
    (2 x point - percentiles; duplicated lines in a draw inflate in-sample R^2, as in 91's summ_basic)."""
    rng = np.random.default_rng(seed)
    r0 = r2_ols(y, X)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        bs.append(r2_ols(y[i], X[i])[1])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r0, float(lo), float(hi), float(2 * r0[1] - hi), float(2 * r0[1] - lo)


def dummies(lab, drop_most_frequent=True):
    lab = pd.Series(lab).astype(str)
    D = pd.get_dummies(lab).astype(float)
    if drop_most_frequent:
        D = D.drop(columns=lab.value_counts().index[0])
    return D.values


def qc_covariates(cells):
    eff = pd.read_csv(DM / 'CRISPRInferredModelEfficacy.csv', index_col=0)
    gr = pd.read_csv(DM / 'CRISPRInferredModelGrowthRate.csv', index_col=0)
    lib_ = gr.notna().apply(lambda r: '+'.join(c.replace('Achilles-', '') for c in gr.columns if r[c]), axis=1)
    qc = pd.read_csv(DM / 'AchillesScreenQCReport.csv')
    qc = qc[qc.ModelID.isin(cells)]
    qci = qc[qc.CanInclude.astype(str).str.lower() == 'true'] if 'CanInclude' in qc else qc
    qci = qci if qci.ModelID.nunique() >= 0.5 * len(cells) else qc
    cols = ['ScreenNNMD', 'ScreenROCAUC', 'CasActivity', 'ScreenDoublingTime', 'ScreenMedianEssentialDepletion',
            'ScreenMedianNonessentialDepletion']
    q = qci.groupby('ModelID')[cols].mean()
    cov = pd.DataFrame(index=pd.Index(cells))
    cov['growth_rate'] = gr.mean(axis=1).reindex(cells)
    cov['model_efficacy'] = eff.mean(axis=1).reindex(cells)
    for c in cols:
        cov[c] = q[c].reindex(cells)
    cov['library'] = lib_.reindex(cells).fillna('unknown')
    return cov


def axis_analysis(cel, workers):
    t0 = time.time()
    cells = list(cel.index)
    ax = pd.read_csv(INP / 'cell_axes.csv', index_col=0).reindex(cells)
    pred_ax = pd.read_parquet(PRED / 'active' / 'omics_axis_value.parquet').value.reindex(cells)
    tis_ax = pd.read_parquet(PRED / 'active' / 'tissue_axis_value.parquet').value.reindex(cells)
    axes = {'active_axis': ax.axis_active, 'all_gene_axis': ax.axis_all_genes,
            'common_essential_axis': ax.axis_common_essentials, 'omics_predicted_axis_oof': pred_ax,
            'tissue_axis_oof': tis_ax}
    cov = qc_covariates(cells)
    rows = []
    lin_d = dummies(cel.lineage_pooled.values)
    lib_d = dummies(cov.library.values)
    num = ['growth_rate', 'model_efficacy', 'ScreenNNMD', 'ScreenROCAUC', 'CasActivity', 'ScreenDoublingTime',
           'ScreenMedianEssentialDepletion', 'ScreenMedianNonessentialDepletion']
    for name, a in axes.items():
        y = a.values.astype(float)
        for other, b in list(axes.items()) + [(c, cov[c]) for c in num]:
            if other == name:
                continue
            for meth in ['pearson', 'spearman']:
                r, lo, hi, n = cell_boot_corr(y, b.values.astype(float), method=meth)
                rows.append({'axis': name, 'statistic': f'{meth}_r', 'with': other, 'value': r, 'lo': lo, 'hi': hi,
                             'n_lines': n})
        for lab, X in [('lineage (>=5 lines; small pooled)', lin_d), ('screen library', lib_d),
                       ('growth rate', cov[['growth_rate']].values),
                       ('screen QC (efficacy, NNMD, ROC AUC, Cas activity, median depletions)',
                        cov[['model_efficacy', 'ScreenNNMD', 'ScreenROCAUC', 'CasActivity',
                             'ScreenMedianEssentialDepletion', 'ScreenMedianNonessentialDepletion']].values),
                       ('lineage + library + growth + QC + doubling time',
                        np.c_[lin_d, lib_d, cov[num].values])]:
            (r2, adj, n, k), plo, phi, blo, bhi = r2_boot(y, X)
            rows.append({'axis': name, 'statistic': 'r2', 'with': lab, 'value': r2, 'lo': np.nan, 'hi': np.nan,
                         'n_lines': n, 'k': k})
            rows.append({'axis': name, 'statistic': 'adjusted_r2', 'with': lab, 'value': adj, 'lo': blo, 'hi': bhi,
                         'interval': 'basic bootstrap', 'lo_percentile': plo, 'hi_percentile': phi, 'n_lines': n,
                         'k': k})
    # split-half reliability: active-gene axis (benchmark Z) and all-gene axis (universe Z, rebuilt as in 105b)
    Z = pd.read_parquet(INP / 'Z.parquet')
    genes = pd.read_csv(INP / 'genes.csv')
    act = list(genes.loc[genes.in_active, 'gene'])
    sh_act = box1.axis_split_half(Z.loc[act], n_splits=200, seed=0)
    ce = pd.read_csv(DM / 'CRISPRGeneEffect.csv', index_col=0).loc[cells, list(genes.gene)]
    ZU = pd.DataFrame(np.vstack([selbench.rank_normal(-ce[g].values) for g in ce.columns]), index=ce.columns,
                      columns=cells)
    sh_all = box1.axis_split_half(ZU, n_splits=200, seed=0)
    chk_all = float(np.nanmax(np.abs(ZU.mean(axis=0).values - ax.axis_all_genes.values)))
    for name, sh in [('active_axis', sh_act), ('all_gene_axis', sh_all)]:
        rows.append({'axis': name, 'statistic': 'split_half_reliability', 'with': 'random halves of the genes',
                     'value': sh[0], 'lo': sh[1], 'hi': sh[2], 'n_lines': len(cells)})
    # the axis's share of the measured per-gene variance (perfect-predictor share) under each axis
    Ga = pd.read_parquet(INP / 'G_active.parquet')
    Gall = pd.read_parquet(INP / 'G_all.parquet')
    for name, G in [('active_axis', Ga), ('all_gene_axis', Gall)]:
        rows.append({'axis': name, 'statistic': 'perfect_share_mean_r_zg2_active_genes', 'with': 'active genes',
                     'value': selbench.oracle_fitness_share(Z.loc[act], G.loc[act]), 'lo': np.nan, 'hi': np.nan,
                     'n_lines': len(cells)})
    T = pd.DataFrame(rows)
    log('axis analysis', f'{time.time() - t0:.0f}s')
    return T, {'all_gene_axis_rebuild_max_abs_diff': chk_all, 'qc_lines': int(cov.ScreenNNMD.notna().sum()),
               'library_counts': cov.library.value_counts().to_dict(), 'runtime_s': time.time() - t0}


# ----------------------------------------------------------------------------------------------------------------
def load_predictions(setting, genes, cells):
    Pm = {}
    for m in MODELS:
        f = PRED / setting / f'{m}.parquet'
        P = pd.read_parquet(f).reindex(index=cells, columns=genes)
        Pm[m] = P.T.astype(float)                       # genes x cells
    return Pm


def oof_manifest(cel):
    """Manifest of the OOF prediction files (cells x genes; folds in inputs/cells.csv) with SHA-256 for audit."""
    import hashlib
    rows = []
    for f in sorted(PRED.glob('*/*.parquet')):
        P = pd.read_parquet(f)
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        rows.append({'file': str(f.relative_to(PRED)), 'rows_cells': P.shape[0], 'columns': P.shape[1],
                     'finite_values': int(np.isfinite(P.values.astype(float)).sum()), 'sha256': h,
                     'cells_in_folds_csv': int(P.index.isin(cel.index).sum())})
    T = pd.DataFrame(rows)
    T.to_csv(PRED / 'manifest.csv', index=False)
    return len(T)


def headline(summ, models, pooled, ranks, axis_T):
    """Compact headline block (primary configuration unless named), every number with its 95% interval."""
    c = summ['configs']['primary']
    M = models[models.config == 'primary'].set_index('model')
    L = M.loc[LEARNED]
    iv = lambda r, k: [float(r[k]), float(r[f'{k}_lo']), float(r[f'{k}_hi'])]  # noqa: E731
    P = pooled.set_index(['config', 'group'])
    pl = P.loc[('primary', 'learned')]
    A = axis_T.set_index(['axis', 'statistic', 'with'])
    ax = lambda a, st, w: [float(A.loc[(a, st, w), 'value']), float(A.loc[(a, st, w), 'lo']),  # noqa: E731
                           float(A.loc[(a, st, w), 'hi'])]
    R = ranks[(ranks.config == 'primary') & (ranks.scope == 'all_predictors')].set_index('model')
    pr = c['predictability']
    best = c['best_learned_by_median_raw']
    out = {
        'format': '[estimate, 95% lo, 95% hi]; two-way (cell line x gene) bootstrap B=1000 unless noted',
        'genes_active': c['genes'], 'cells': c['cells'], 'best_learned_by_median_raw': best,
        'perfect_share': iv(pl, 'perfect_share'),
        'pooled_learned': {k: iv(pl, k) for k in ['axis_capture', 'selective_capture', 'axis_share', 'share_ratio',
                                                  'alignment_ratio', 'median_raw', 'median_selective',
                                                  'median_selective_lin', 'selective_capture_lin']},
        'pooled_learned_selective_capture_lin_basic_interval': [float(pl['selective_capture_lin_lo_basic']),
                                                                float(pl['selective_capture_lin_hi_basic'])],
        'per_model': {m: {k: iv(L.loc[m], k) for k in ['axis_capture', 'selective_capture', 'axis_share',
                                                       'share_ratio', 'median_raw', 'median_selective',
                                                       'selective_capture_lin']} for m in LEARNED},
        'learned_ranges_point': {k: [float(L[k].min()), float(L[k].max())] for k in
                                 ['axis_capture', 'selective_capture', 'axis_share', 'share_ratio', 'median_raw',
                                  'median_selective', 'selective_capture_lin']},
        'floor_over_best_median_raw': {ag: iv(pl, f'{ag}_median_raw_over_best_learned') for ag in AGNOSTIC},
        'gene_agnostic_median_raw': {ag: iv(M.loc[ag], 'median_raw') for ag in AGNOSTIC},
        'learned_with_median_raw_below_omics_axis': int((L.vs_omics_axis_raw == 'below').sum()),
        'tissue_gene_mean_median_selective': iv(M.loc['tissue_gene_mean'], 'median_selective'),
        'tissue_gene_mean_selective_capture': iv(M.loc['tissue_gene_mean'], 'selective_capture'),
        'learned_minus_tissue_gene_mean_median_selective': {
            m: iv(L.loc[m], 'median_selective_minus_tissue_gene_mean') + [L.loc[m, 'vs_tissue_gene_mean_selective']]
            for m in LEARNED},
        'rank_shift_selective_minus_raw': {m: float(R.loc[m, 'rank_shift_selective_minus_raw']) for m in R.index},
        'ranks_raw_selective': {m: [float(R.loc[m, 'rank_raw']), float(R.loc[m, 'rank_selective'])] for m in R.index},
        'predictability_best_learned': {t: {k: [v[k], v['lo'], v['hi']] for k, v in pr[f'{best}|r>={t}'].items()
                                            if isinstance(v, dict)} for t in THRESH},
        'predictability_omics_axis': {t: {k: [v[k], v['lo'], v['hi']] for k, v in pr[f'omics_axis|r>={t}'].items()
                                          if isinstance(v, dict)} for t in THRESH},
        'perfect_predictor_frac_genes_axis_term_ge_half': [v for v in
                                                           pr['perfect|r>=0.3']['frac_axis_term_ge_half_r'].values()],
        'loading_r_zg': c['loading_r_zg'],
        'axis': {'split_half_active': ax('active_axis', 'split_half_reliability', 'random halves of the genes'),
                 'split_half_all_genes': ax('all_gene_axis', 'split_half_reliability', 'random halves of the genes'),
                 'r_active_vs_all_gene_axis': ax('active_axis', 'pearson_r', 'all_gene_axis'),
                 'r_growth_rate': ax('active_axis', 'pearson_r', 'growth_rate'),
                 'r_screen_roc_auc': ax('active_axis', 'pearson_r', 'ScreenROCAUC'),
                 'r_screen_nnmd': ax('active_axis', 'pearson_r', 'ScreenNNMD'),
                 'r_doubling_time': ax('active_axis', 'pearson_r', 'ScreenDoublingTime'),
                 'adj_r2_lineage': ax('active_axis', 'adjusted_r2', 'lineage (>=5 lines; small pooled)'),
                 'adj_r2_library': ax('active_axis', 'adjusted_r2', 'screen library'),
                 'adj_r2_growth': ax('active_axis', 'adjusted_r2', 'growth rate'),
                 'adj_r2_screen_qc': ax('active_axis', 'adjusted_r2',
                                        'screen QC (efficacy, NNMD, ROC AUC, Cas activity, median depletions)'),
                 'adj_r2_all': ax('active_axis', 'adjusted_r2', 'lineage + library + growth + QC + doubling time'),
                 'r_predicted_vs_measured': ax('active_axis', 'pearson_r', 'omics_predicted_axis_oof'),
                 'predicted_axis_adj_r2_lineage': ax('omics_predicted_axis_oof', 'adjusted_r2',
                                                     'lineage (>=5 lines; small pooled)'),
                 'note': 'cell-line bootstrap (B=1000); adjusted R^2 with basic bootstrap intervals'},
        'sensitivity': {}}
    for cfg in ['sd02', 'allgene_axis']:
        if ('%s' % cfg, 'learned') in P.index:
            r = P.loc[(cfg, 'learned')]
            out['sensitivity'][cfg] = {k: iv(r, k) for k in ['axis_capture', 'selective_capture', 'axis_share',
                                                             'perfect_share', 'share_ratio', 'median_raw',
                                                             'median_selective', 'selective_capture_lin']}
            out['sensitivity'][cfg]['genes'] = summ['configs'][cfg]['genes']
    dr = DRUG_REFERENCE
    sr = out['pooled_learned']['share_ratio']
    out['comparison_with_drugs'] = {
        'axis_capture': [out['pooled_learned']['axis_capture'][0], dr['leaderboard_axis_capture_A']],
        'selective_capture': [out['pooled_learned']['selective_capture'][0], dr['leaderboard_selective_capture_S']],
        'A_over_S': [out['pooled_learned']['axis_capture'][0] / out['pooled_learned']['selective_capture'][0],
                     dr['leaderboard_axis_capture_A'] / dr['leaderboard_selective_capture_S']],
        'share_ratio_pooled_vs_drug_range': [sr, dr['model_axis_share_over_perfect_share_range_nine_settings']],
        'share_ratio_pooled_within_drug_range': bool(2.4 <= sr[0] <= 4.4),
        'learned_share_ratio_above_1_all': bool((L.share_ratio_lo > 1).all()),
        'perfect_share': [out['perfect_share'][0], dr['leaderboard_perfect_share'], dr['GDSC2_AUC_perfect_share']]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    a = ap.parse_args()
    t_all = time.time()
    cel = pd.read_csv(INP / 'cells.csv', index_col=0)
    cells = list(cel.index)
    genes_t = pd.read_csv(INP / 'genes.csv')
    Z_all = pd.read_parquet(INP / 'Z.parquet')
    axes = {'G_active': pd.read_parquet(INP / 'G_active.parquet'), 'G_all': pd.read_parquet(INP / 'G_all.parquet')}
    tissue = cel.lineage_pooled
    res = {}
    for cfg in a.configs:
        setting, axname, col = CONFIGS[cfg]
        genes = list(genes_t.loc[genes_t[col], 'gene'])
        Z = Z_all.loc[genes, cells]
        G = axes[axname].loc[genes, cells]
        Pm = load_predictions(setting, genes, cells)
        W, V = lib.draws(len(cells), len(genes), SEED, n_boot=a.boot)
        res[cfg] = score_config(cfg, Z, G, Pm, tissue, W, V, genes, a.workers)
        res[cfg]['V'] = V
        res[cfg]['genes'] = genes
    # tables
    models = pd.DataFrame([r for c in res.values() for r in c['rows']])
    pooled = pd.DataFrame([r for c in res.values() for r in c['pools']])
    per = pd.DataFrame([r for c in res.values() for r in c['per']])
    preds = pd.DataFrame([r for c in res.values() for r in c['preds']])
    ranks = pd.DataFrame([r for c in res.values() for r in c['ranks']])
    models[models.config == 'primary'].to_csv(OUT / 'models.csv', index=False)
    pooled.to_csv(OUT / 'pooled.csv', index=False)
    per.to_csv(OUT / 'per_gene.csv.gz', index=False, compression='gzip')
    preds.to_csv(OUT / 'predictability.csv', index=False)
    ranks.to_csv(OUT / 'ranks.csv', index=False)
    # sensitivity: key statistics per configuration (models and the learned pool) + paired differences (ii - primary)
    keys = ['axis_capture', 'selective_capture', 'axis_share', 'perfect_share', 'share_ratio', 'alignment_ratio',
            'median_raw', 'median_selective', 'median_selective_lin', 'selective_capture_lin']
    srows = []
    for cfg, c in res.items():
        for r in c['rows']:
            srows.append({'config': cfg, 'unit': r['model'], 'kind': r['kind'],
                          **{k2: r.get(k2) for k in keys for k2 in (k, f'{k}_lo', f'{k}_hi')}})
        for r in c['pools']:
            srows.append({'config': cfg, 'unit': f"pooled_{r['group']}", 'kind': 'pool',
                          **{k2: r.get(k2) for k in keys for k2 in (k, f'{k}_lo', f'{k}_hi')}})
    if 'primary' in res and 'allgene_axis' in res:
        lix = [i for i, m in enumerate(MODELS) if m in LEARNED]
        sp, _, _ = group_stats(res['primary']['arrs'], lix, res['primary']['V'])
        sa, _, _ = group_stats(res['allgene_axis']['arrs'], lix, res['allgene_axis']['V'])
        d = {'config': 'allgene_axis_minus_primary (paired draws)', 'unit': 'pooled_learned', 'kind': 'difference'}
        for k in keys:
            d.update(lib.summ(sa[k] - sp[k], k))
        srows.append(d)
    sens = pd.DataFrame(srows)
    sens.to_csv(OUT / 'sensitivity.csv', index=False)
    axis_T, axis_info = axis_analysis(cel, a.workers)
    axis_T.to_csv(OUT / 'axis.csv', index=False)
    n_oof = oof_manifest(cel)
    # summary
    summ = {'runtime_s_score': None, 'bootstrap': {'B': a.boot, 'seed': SEED, 'type': 'two-way (cell line x gene)'},
            'counts': json.loads((INP / 'prepare.json').read_text()), 'drug_reference': DRUG_REFERENCE,
            'axis_info': axis_info, 'prediction_files': n_oof, 'checks': {}, 'configs': {}}
    train_meta = {}
    for f in sorted(PRED.glob('*/train_*.json')):
        j = json.loads(f.read_text())
        train_meta[f.stem] = {'runtime_s': j['runtime_s'], 'genes': j['genes'], 'cells': j['cells'],
                              'models': j['models'],
                              'per_fold': [{m: {k: v for k, v in h.items() if k not in ('cv_mse',)}
                                            for m, h in fd['models'].items()} for fd in j['folds']]}
    summ['training'] = train_meta
    for cfg, c in res.items():
        summ['checks'][cfg] = c['checks']
        M = models[models.config == cfg].set_index('model')
        P = pooled[pooled.config == cfg].set_index('group')
        best = c['best']
        def pick(r, k):
            o = {k: r[k], 'lo': r.get(f'{k}_lo'), 'hi': r.get(f'{k}_hi')}
            if f'{k}_lo_basic' in r.index:
                o.update({'lo_basic': r.get(f'{k}_lo_basic'), 'hi_basic': r.get(f'{k}_hi_basic')})
            return o
        d = {'genes': len(c['genes']), 'cells': len(cells), 'best_learned_by_median_raw': best,
             'runtime_s': c['runtime_s']}
        d['pooled_learned'] = {k: pick(P.loc['learned'], k) for k in keys + ['mean_raw', 'cf_share_ratio']}
        d['pooled_gene_agnostic'] = {k: pick(P.loc['gene_agnostic'], k) for k in ['axis_capture', 'selective_capture',
                                                                                   'axis_share', 'median_raw']}
        d['per_model'] = {m: {k: pick(M.loc[m], k) for k in keys + ['median_raw_over_best_learned']}
                          for m in M.index}
        d['floor_over_best'] = {ag: pick(P.loc['learned'], f'{ag}_median_raw_over_best_learned') for ag in AGNOSTIC}
        d['tissue_gene_mean_vs_learned_median_selective'] = {
            m: {**pick(M.loc[m], 'median_selective_minus_tissue_gene_mean'),
                'verdict': M.loc[m, 'vs_tissue_gene_mean_selective']} for m in LEARNED}
        d['omics_axis_vs_learned_median_raw'] = {
            m: {**pick(M.loc[m], 'median_raw_minus_omics_axis'), 'verdict': M.loc[m, 'vs_omics_axis_raw']}
            for m in LEARNED}
        R = ranks[(ranks.config == cfg) & (ranks.scope == 'all_predictors')].set_index('model')
        d['ranks_all_predictors'] = R[['rank_raw', 'rank_raw_lo', 'rank_raw_hi', 'rank_selective', 'rank_selective_lo',
                                       'rank_selective_hi', 'rank_shift_selective_minus_raw']].to_dict(orient='index')
        Pr = preds[(preds.config == cfg)].set_index(['model', 'threshold'])
        d['predictability'] = {}
        for m in [best, 'omics_axis', 'tissue_axis', 'perfect']:
            for t in THRESH:
                r = Pr.loc[(m, t)]
                d['predictability'][f'{m}|r>={t}'] = {k: pick(r, k) for k in
                                                     ['n_pass_raw', 'frac_axis_term_ge_half_r',
                                                      'n_below_on_selective_term', 'frac_below_on_selective_term',
                                                      'n_below_on_partial', 'n_pass_selective_term']
                                                     if f'{k}_lo' in r.index and pd.notna(r.get(k))}
                if 'n_overlap_with_best_learned_pass' in r.index and pd.notna(r['n_overlap_with_best_learned_pass']):
                    d['predictability'][f'{m}|r>={t}']['n_overlap_with_best_learned_pass'] = \
                        r['n_overlap_with_best_learned_pass']
        pg = per[(per.config == cfg) & (per.model == 'perfect')]
        d['loading_r_zg'] = {'median': float(pg.loading_r_zg.median()), 'mean': float(pg.loading_r_zg.mean()),
                             'frac_negative': float((pg.loading_r_zg < 0).mean()),
                             'frac_abs_gt_0.3': float((pg.loading_r_zg.abs() > 0.3).mean()),
                             'frac_r_zg2_ge_0.5': float((pg.loading_r_zg ** 2 >= 0.5).mean())}
        bg = per[(per.config == cfg) & (per.model == best) & (per.raw > 0)]
        d['best_learned_per_gene_axis_term_over_raw_median_where_raw_gt_0'] = float((bg.axis_term / bg.raw).median())
        summ['configs'][cfg] = d
    A = axis_T.set_index(['axis', 'statistic', 'with'])
    summ['axis'] = {f'{i[0]}|{i[1]}|{i[2]}': {'value': r.value, 'lo': r.lo, 'hi': r.hi}
                    for i, r in A.iterrows()}
    summ['design_choices'] = [
        'Sensitivity s = -Chronos gene effect; rank-normal profiles pooled over all folds (Box 1 step 1).',
        'Axis: leave-gene-out mean of the other active genes (>=10 contributing), box1.active_axis; genes outside the '
        'active set (sd02) receive the full active mean (as box1.active_axis does for inactive drugs).',
        'Active genes are the benchmark (1,077 <= 3,000: no SD cap).',
        'Features: top-5,000 variance expression genes chosen on the training lines of each fold (as 58\'s DepMap '
        'variant); standardized on training lines.',
        'Lineage-adjusted score: 91\'s engine with OncotreeLineage; lineages with <5 benchmark lines pooled into one '
        '"Other (<5 lines)" label; not reported for tissue_gene_mean and tissue_axis (in the span of the lineage '
        'indicators up to fold-to-fold variation, 91\'s note).',
        'Tissue-by-gene mean uses the full OncotreeLineage labels (min 1 training line, else gene mean), as '
        'selbench.tissue_drug_mean_baseline.',
        'Gene-agnostic floor: omics_axis (primary, the analogue of FitnessOnlyRidge); tissue_axis (drug-agnostic '
        'tissue mean) reported beside it.',
        'Predictability: "falls below threshold on the selective term alone" = selective term r_pz - r_pg r_zg < t '
        'among genes with r_pz >= t; also reported for the partial correlation r_pz.g < t.',
        'sd02 models are retrained on the sd02 gene set (multi-output models depend on the output set).']
    summ['headline'] = headline(summ, models, pooled, ranks, axis_T)
    summ['runtime_s_score'] = time.time() - t_all
    prep = summ['counts'].get('runtime_s', 0.0)
    summ['runtime_s_total'] = {'prepare': prep, 'training_sum_of_runs': sum(v['runtime_s'] for v in train_meta.values()),
                               'score': summ['runtime_s_score'],
                               'note': 'training runs: CPU models and the GPU MLP ran as separate processes, partly '
                                       'concurrently'}
    bench.write_json(OUT / 'summary.json', summ)
    log('done', f'{time.time() - t_all:.0f}s')


if __name__ == '__main__':
    main()
