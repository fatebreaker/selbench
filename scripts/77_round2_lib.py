"""Shared engine for the round-2 revision analyses (scripts 78-89; mock review round 2).

Two-way (cell line x drug) bootstrap engine for per-drug correlation decompositions, as in selbench.two_way_bootstrap
and scripts 42/59: every replicate b draws multinomial cell-line weights W[b] (applied inside every per-drug
correlation, so drugs that share cell lines stay dependent) and multinomial drug multiplicities V[b] (applied when
summing or taking medians over drugs). Row 0 of W and V is all ones (point estimate). The measured profiles z, the
axes g and the predictions are held at their full-sample values (as in 42, 59, 68). The same draws are shared by every
model of a setting, so differences between models and pooled summaries are paired.

Per drug and model, from weighted moments:
  raw r(p, z), alignment r_pg, loading r_zg, mediated r_pg r_zg, selective term raw - r_pg r_zg,
  selective r(p, z | g); in paired mode for source S ('_w', within) and target T ('_c', cross):
  selT = r(p, z | own axis), selB = r(p, z | g_S, g_T) (symmetric: also computed within S).

Data set-ups reuse the project's loaders read-only: 41/42 (DrEval leaderboard and processing versions), 66 (harmonized
curves), 59 (drevalpy retraining out-of-fold predictions), 68 (IMPROVE predictions, replicate-free option).
Outputs of the round-2 scripts go to results/revision_round2/<topic>/.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RES = PROJECT / 'results'
OUT2 = RES / 'revision_round2'
for _p in [str(HERE), str(PROJECT / 'selbench')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import selbench  # noqa: E402

B = 1000
TOL = 1e-10
MIN_CELLS = 30
_MODS = {}


def load_module(name, file):
    """Import a project script read-only (registered in sys.modules so that forked workers can unpickle)."""
    if name in _MODS:
        return _MODS[name]
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODS[name] = mod
    return mod


def out_dir(topic):
    d = OUT2 / topic
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return [conv(x) for x in o.tolist()]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    Path(path).write_text(json.dumps(conv(obj), indent=1))


# ----------------------------------------------------------------------------------------------------------------
# draws and interval helpers
def draws(n_cells, n_drugs, seed, n_boot=B):
    """W: (n_boot+1) x n_cells cell weights; V: (n_boot+1) x n_drugs drug multiplicities; row 0 = ones.
    Cells are drawn first, then drugs (the order used by 42, 59 and 66)."""
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones(n_cells), rng.multinomial(n_cells, np.full(n_cells, 1.0 / n_cells), size=n_boot)]).astype(float)
    V = np.vstack([np.ones(n_drugs), rng.multinomial(n_drugs, np.full(n_drugs, 1.0 / n_drugs), size=n_boot)]).astype(float)
    return W, V


def ci(rep, level=95):
    rep = np.asarray(rep, float)
    b = rep[1:][np.isfinite(rep[1:])]
    if len(b) < 10:
        return np.nan, np.nan
    a = (100 - level) / 2
    lo, hi = np.percentile(b, [a, 100 - a])
    return float(lo), float(hi)


def summ(rep, name, level=95):
    """{name: point (draw 0), name_lo, name_hi} from a replicate vector (draw 0 = point estimate)."""
    rep = np.asarray(rep, float)
    lo, hi = ci(rep, level)
    return {name: float(rep[0]), f'{name}_lo': lo, f'{name}_hi': hi}


def wmedian_rows(X, V):
    """Weighted median over the last axis. X: (R, D) values (NaN ignored), V: (R, D) integer multiplicities."""
    R = X.shape[0]
    out = np.full(R, np.nan)
    for b in range(R):
        ok = np.isfinite(X[b]) & (V[b] > 0)
        if not ok.any():
            continue
        x, w = X[b][ok], V[b][ok]
        o = np.argsort(x, kind='mergesort')
        x, cw = x[o], np.cumsum(w[o])
        n = cw[-1]
        lo = np.searchsorted(cw, (n - 1) // 2, side='right')
        hi = np.searchsorted(cw, n // 2, side='right')
        out[b] = 0.5 * (x[lo] + x[hi])
    return out


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok], kind='mergesort')]
    m = len(order)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


# ----------------------------------------------------------------------------------------------------------------
# correlation algebra on (co)variances (broadcasting)
def _corr(c, va, vb):
    with np.errstate(invalid='ignore', divide='ignore'):
        return c / np.sqrt(va * vb)


def pcor1(vx, vy, cxy, cx, cy, vc):
    """Partial correlation of x and y given one covariate c."""
    with np.errstate(invalid='ignore', divide='ignore'):
        rxx, ryy, rxy = vx - cx * cx / vc, vy - cy * cy / vc, cxy - cx * cy / vc
        out = rxy / np.sqrt(rxx * ryy)
    return np.where((rxx > TOL * vx) & (ryy > TOL * vy), out, np.nan)


def pcor2(vx, vy, cxy, cx1, cx2, cy1, cy2, v1, v2, c12):
    """Partial correlation of x and y given two covariates (as 42_crosslab_scores.pcor2)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        det = v1 * v2 - c12 * c12
        qxx = (cx1 * cx1 * v2 - 2 * cx1 * cx2 * c12 + cx2 * cx2 * v1) / det
        qyy = (cy1 * cy1 * v2 - 2 * cy1 * cy2 * c12 + cy2 * cy2 * v1) / det
        qxy = (cx1 * cy1 * v2 - (cx1 * cy2 + cx2 * cy1) * c12 + cx2 * cy2 * v1) / det
        rxx, ryy, rxy = vx - qxx, vy - qyy, cxy - qxy
        out = rxy / np.sqrt(rxx * ryy)
    return np.where((det > 1e-9 * v1 * v2) & (rxx > TOL * vx) & (ryy > TOL * vy), out, np.nan)


def _std(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.full_like(x, np.nan)


def within_stats(z, g, P, W):
    """One drug, single screen. z, g: (n,), P: (n, M) predictions, W: (R, n) cell weights.
    Returns dict of (R, M): raw, r_pg, r_zg, sel."""
    z, g = _std(z), _std(g)
    P = np.column_stack([_std(P[:, j]) for j in range(P.shape[1])])
    Q = np.concatenate([np.column_stack([z, g, z * z, g * g, z * g]), P, P * P, P * z[:, None], P * g[:, None]], axis=1)
    E = (W @ Q) / W.sum(1)[:, None]
    M = P.shape[1]
    mz, mg, ezz, egg, ezg = (E[:, i] for i in range(5))
    o = 5
    mp, epp, epz, epg = E[:, o:o + M], E[:, o + M:o + 2 * M], E[:, o + 2 * M:o + 3 * M], E[:, o + 3 * M:o + 4 * M]
    vz, vg, czg = (ezz - mz ** 2)[:, None], (egg - mg ** 2)[:, None], (ezg - mz * mg)[:, None]
    vp = epp - mp ** 2
    cpz, cpg = epz - mp * mz[:, None], epg - mp * mg[:, None]
    raw, rpg = _corr(cpz, vp, vz), _corr(cpg, vp, vg)
    rzg = np.broadcast_to(_corr(czg, vz, vg), raw.shape)
    return {'raw': raw, 'r_pg': rpg, 'r_zg': np.array(rzg), 'sel': pcor1(vp, vz, cpz, cpg, czg, vg)}


def paired_stats(zs, gs, zt, gt, P, W, identity=False):
    """One drug, matched cells, identical predictions scored against S (within, '_w') and T (cross, '_c').
    Returns dict of (R, M): raw, r_pg, r_zg, selT (own axis), selB (both axes) for w and c."""
    S = np.column_stack([_std(zs), _std(gs), _std(zt), _std(gt)])
    P = np.column_stack([_std(P[:, j]) for j in range(P.shape[1])])
    a = [0, 1, 2, 3, 0, 2, 0, 2, 1]
    b = [0, 1, 2, 3, 1, 3, 3, 1, 3]
    Q = np.concatenate([S, S[:, a] * S[:, b], P, P * P] + [P * S[:, [k]] for k in range(4)], axis=1)
    E = (W @ Q) / W.sum(1)[:, None]
    M = P.shape[1]
    m = E[:, :4]
    sh = E[:, 4:13]
    cov = {(a[k], b[k]): (sh[:, k] - m[:, a[k]] * m[:, b[k]])[:, None] for k in range(9)}

    def cs(i, j):
        return cov[(i, j)] if (i, j) in cov else cov[(j, i)]
    o = 13
    mp, epp = E[:, o:o + M], E[:, o + M:o + 2 * M]
    vp = epp - mp ** 2
    cp = [E[:, o + (2 + k) * M:o + (3 + k) * M] - mp * m[:, [k]] for k in range(4)]
    out = {}
    for side, zi, gi, go in [('w', 0, 1, 3), ('c', 2, 3, 1)]:
        vz, vg = cs(zi, zi), cs(gi, gi)
        raw = _corr(cp[zi], vp, vz)
        rpg = _corr(cp[gi], vp, vg)
        rzg = np.broadcast_to(_corr(cs(zi, gi), vz, vg), raw.shape)
        out[f'raw_{side}'], out[f'r_pg_{side}'], out[f'r_zg_{side}'] = raw, rpg, np.array(rzg)
        out[f'selT_{side}'] = pcor1(vp, vz, cp[zi], cp[gi], cs(zi, gi), vg)
        if identity:
            out[f'selB_{side}'] = out[f'selT_{side}']
        else:
            out[f'selB_{side}'] = pcor2(vp, vz, cp[zi], cp[gi], cp[go], cs(zi, gi), cs(zi, go), vg, cs(go, go),
                                        cs(gi, go))
    return out


# ----------------------------------------------------------------------------------------------------------------
# paired accumulation over drugs (per-model sums of every statistic, with 42's per-stat-pair masks)
PAIR_STATS = {  # name: (within stat, cross stat)
    'raw': ('raw_w', 'raw_c'), 'mediated': ('med_w', 'med_c'), 'selective_term': ('selterm_w', 'selterm_c'),
    'r_pg': ('r_pg_w', 'r_pg_c'), 'r_zg': ('r_zg_w', 'r_zg_c'), 'rzg2': ('rzg2_w', 'rzg2_c'),
    'selT': ('selT_w', 'selT_c'),
    'selBOTH_asym': ('selT_w', 'selB_c'),      # as in 42/43: within = r(p, z_S | g_S)
    'selBOTH_sym': ('selB_w', 'selB_c'),       # symmetric: within = r(p, z_S | g_S, g_T)
}


class PairedAccumulator:
    """Per-model, per-draw V-weighted sums over drugs of every PAIR_STATS pair (sum_w, sum_c, n), plus sums of a
    reference pseudo-model's cross selective-both and raw on each model's drugs (ref_*), and per-drug point rows."""

    def __init__(self, models, R, ref=None):
        self.models = list(models)
        self.ix = {m: i for i, m in enumerate(self.models)}
        self.R = R
        self.ref = ref
        M = len(self.models)
        self.S = {k: {kind: np.zeros((R, M)) for kind in ('sum_w', 'sum_c', 'n')} for k in PAIR_STATS}
        self.refsum = {k: np.zeros((R, M)) for k in ('ref_selB_c', 'mod_selB_c', 'ref_raw_c', 'mod_raw_c', 'n_ref')}
        self.rows = []

    def add(self, drug, n_cells, st, models_here, v):
        """st: dict stat -> (R, M_here) for models_here; v: (R,) drug multiplicities."""
        st = dict(st)
        for side in ('w', 'c'):
            st[f'med_{side}'] = st[f'r_pg_{side}'] * st[f'r_zg_{side}']
            st[f'selterm_{side}'] = st[f'raw_{side}'] - st[f'med_{side}']
            st[f'rzg2_{side}'] = st[f'r_zg_{side}'] ** 2
        cols = [self.ix[m] for m in models_here]
        vv = v[:, None]
        for k, (a, b) in PAIR_STATS.items():
            xa, xb = st[a], st[b]
            ok = np.isfinite(xa) & np.isfinite(xb)
            self.S[k]['sum_w'][:, cols] += np.where(ok, xa, 0.0) * vv
            self.S[k]['sum_c'][:, cols] += np.where(ok, xb, 0.0) * vv
            self.S[k]['n'][:, cols] += ok * vv
        if self.ref is not None and self.ref in models_here:
            j = models_here.index(self.ref)
            rb, rr = st['selB_c'][:, [j]], st['raw_c'][:, [j]]
            ok = np.isfinite(st['selB_c']) & np.isfinite(rb) & np.isfinite(st['raw_c']) & np.isfinite(rr)
            self.refsum['ref_selB_c'][:, cols] += np.where(ok, rb, 0.0) * vv
            self.refsum['mod_selB_c'][:, cols] += np.where(ok, st['selB_c'], 0.0) * vv
            self.refsum['ref_raw_c'][:, cols] += np.where(ok, rr, 0.0) * vv
            self.refsum['mod_raw_c'][:, cols] += np.where(ok, st['raw_c'], 0.0) * vv
            self.refsum['n_ref'][:, cols] += ok * vv
        keys = ['raw_w', 'raw_c', 'r_pg_w', 'r_pg_c', 'r_zg_w', 'r_zg_c', 'selT_w', 'selT_c', 'selB_w', 'selB_c']
        for jj, m in enumerate(models_here):
            self.rows.append({'model': m, 'drug': drug, 'n_cells': int(n_cells), **{k: float(st[k][0, jj]) for k in keys}})

    def save(self, path, meta):
        arrs = {f'{k}|{kind}': v.astype(np.float64) for k, d in self.S.items() for kind, v in d.items()}
        arrs.update({f'ref|{k}': v for k, v in self.refsum.items()})
        np.savez_compressed(path, models=np.array(self.models), meta=json.dumps(meta), **arrs)


def load_paired(path):
    z = np.load(path, allow_pickle=False)
    models = list(z['models'])
    meta = json.loads(str(z['meta']))
    S = {}
    for k in z.files:
        if '|' in k:
            a, kind = k.split('|', 1)
            S.setdefault(a, {})[kind] = z[k]
    return models, meta, S


def pooled(S, models, group, stat, kind):
    """Sum over the models of `group` (a list of names) of a per-model sum array -> (R,)."""
    ix = [models.index(m) for m in group if m in models]
    return S[stat][kind][:, ix].sum(1)


# ----------------------------------------------------------------------------------------------------------------
# set-ups
def leaderboard_setup(need_versions=None, harmonized=False):
    """DrEval leaderboard (CTRPv2 CurveCurator ln IC50, leave-cell-line-out) exactly as 42/66: predictions on the 739
    cells predicted by every model, decompositions of the requested versions, and 42's two-way draws."""
    sc = load_module('sc42', '42_crosslab_scores.py')
    cd = sc.cd
    V = cd.versions()
    hv, ranges = ({}, None)
    if harmonized:
        hr = load_module('hr66', '66_harmonized_range.py')
        hv, ranges = hr.build_versions()
    allv = {**V, **hv}
    need = set(need_versions or []) | {cd.REFERENCE, cd.AUC_REFERENCE}
    dec = {}
    for v in sorted(need):
        Z, G, _, _ = selbench.decompose(allv[v])
        dec[v] = (Z, G)
    P, mclass, lb = cd.predictions()
    cells = None
    for M_ in P.values():
        cells = M_.columns if cells is None else cells.intersection(M_.columns)
    U = np.array(sorted(cells))
    P = {m: M_.reindex(columns=U) for m, M_ in P.items()}
    all_drugs = sorted(dec[cd.REFERENCE][0].index.union(dec[cd.AUC_REFERENCE][0].index))
    W, Vd = draws(len(U), len(all_drugs), sc.SEED)
    published22 = [m for m in P if mclass[m] in ('classical', 'neural/factorization')]
    ag = pd.read_csv(cd.OUT / 'drug_agnostic_check.csv').set_index('model')
    return {'sc': sc, 'cd': cd, 'versions': allv, 'dec': dec, 'P': P, 'mclass': mclass, 'lb': set(lb), 'U': U,
            'all_drugs': all_drugs, 'drug_ix': {d: i for i, d in enumerate(all_drugs)}, 'W': W, 'V': Vd,
            'published22': published22, 'published21': [m for m in published22 if m != 'SVR'],
            'lbm': [m for m in P if m in set(lb)], 'agnostic': ag, 'ranges': ranges}


MEASURED, FITNESS = 'ref_CTRPv2_measured', 'ref_CTRPv2_fitness_ldo'


def leaderboard_pair(L, ref, tgt, acc_models=None):
    """Paired scores of one leaderboard pair (reference version `ref` of CTRPv2, target version `tgt`) on 42's cells:
    per drug, cells with finite profile and axis in both versions and a prediction from every leaderboard model."""
    Zs, Gs = L['dec'][ref]
    Zt, Gt = L['dec'][tgt]
    U = L['U']
    identity = ref == tgt
    models = list(L['P']) + [MEASURED, FITNESS]
    acc = PairedAccumulator(models, L['W'].shape[0], ref=MEASURED)
    mats = {k: M.reindex(columns=U) for k, M in [('zs', Zs), ('gs', Gs), ('zt', Zt), ('gt', Gt)]}
    pseudo = {MEASURED: Zs.reindex(columns=U), FITNESS: Gs.reindex(columns=U)}
    drugs = [d for d in Zs.index.intersection(Zt.index) if d in L['drug_ix']]
    n_used = 0
    for d in drugs:
        cols = {k: mats[k].loc[d].values.astype(float) for k in mats}
        m0 = np.all([np.isfinite(v) for v in cols.values()], axis=0)
        for m in L['lbm']:
            Pm = L['P'][m]
            if d in Pm.index:
                m0 &= np.isfinite(Pm.loc[d].values)
        if m0.sum() < MIN_CELLS:
            continue
        idx = np.where(m0)[0]
        use, pc = [], []
        for m in models:
            src = pseudo[m] if m in pseudo else L['P'][m]
            if d not in src.index:
                continue
            p = src.loc[d].values[idx].astype(float)
            if not np.all(np.isfinite(p)) or np.std(p) < 1e-12:
                continue
            use.append(m)
            pc.append(p)
        st = paired_stats(cols['zs'][idx], cols['gs'][idx], cols['zt'][idx], cols['gt'][idx], np.column_stack(pc),
                          L['W'][:, idx], identity=identity)
        acc.add(d, len(idx), st, use, L['V'][:, L['drug_ix'][d]])
        n_used += 1
    return acc, n_used


def retrain_setup(setting, models=None):
    """drevalpy retraining (59): out-of-fold predictions of the panel models, the training dataset's responses and
    the other datasets' responses with the same endpoint. `models`: restrict the panel (default: 59's main panel)."""
    rt = load_module('rt59', '59_retrain_score.py')
    ds, ep = rt.SETTINGS[setting]
    resp = rt.responses()
    oof, rawscale, incomplete = rt.load_oof(setting)
    allm = sorted(oof)
    if models is None:
        models = [m for m in allm if not m.endswith('Dm')]
    models = [m for m in models if m in oof]
    cells = sorted(set.intersection(*[set(oof[m].cell) for m in models]))
    Pfull = {m: oof[m][oof[m].cell.isin(cells)].pivot_table(index='drug', columns='cell', values='prediction')
             for m in models}
    obs = resp[(resp.dataset == ds) & (resp.endpoint == ep)][['drug', 'cell', 'tissue', 'response']]
    return {'rt': rt, 'setting': setting, 'dataset': ds, 'endpoint': ep, 'resp': resp, 'models': models,
            'cells': cells, 'P': Pfull, 'obs': obs, 'incomplete': incomplete, 'rawscale': rawscale}


def improve_setup(replicate_free=True, need_cross=True):
    """IMPROVE CSA (68): within and cross predictions; replicate-free by default (pairs whose replicate rows fell in
    different test splits removed; 68 --exclude-split-replicates)."""
    im = load_module('im68', '68_improve_csa.py')
    im.extract()
    clean = im.drop_split_replicates if replicate_free else (lambda w: w)
    obs = im.responses()
    dec = {ds: im.decompose(obs[ds]) for ds in im.DATASETS}
    P_within, P_cross, floors = {}, {}, {}
    DRUG, CELL = im.DRUG, im.CELL
    for model in im.MODELS:
        for src in im.DATASETS:
            w = im.load_preds(src, src, model)
            if w is None:
                continue
            w = clean(w)
            w = w.sort_values('split', kind='stable').drop_duplicates([DRUG, CELL])
            P_within[(model, src)] = w.assign(pred=-w.auc_pred).pivot_table(index=DRUG, columns=CELL, values='pred')
            if src not in floors:
                floors[src] = im.floor_predictions(w)
            if not need_cross:
                continue
            key = w.set_index([DRUG, CELL]).split
            for trg in im.DATASETS:
                if trg == src:
                    continue
                c = im.load_preds(src, trg, model)
                if c is None:
                    continue
                c = c.set_index([DRUG, CELL])
                common = c.index.intersection(key.index)
                cc = c.loc[common]
                cc = cc[cc.split.values == key.reindex(cc.index).values]
                cc = cc[~cc.index.duplicated()]
                P_cross[(model, src, trg)] = (cc.reset_index().assign(pred=lambda x: -x.auc_pred)
                                              .pivot_table(index=DRUG, columns=CELL, values='pred'))
    return {'im': im, 'obs': obs, 'dec': dec, 'P_within': P_within, 'P_cross': P_cross, 'floors': floors,
            'models': list(im.MODELS), 'datasets': list(im.DATASETS)}


def improve_source_measurement(I, src, trg, template):
    """68's 'source_measurement' reference: the source's measured response on pairs measured in both datasets,
    restricted to pairs the models predicted (template: one model's cross prediction matrix)."""
    obs = I['obs']
    both = obs[src].merge(obs[trg][['drug', 'cell']], on=['drug', 'cell'])
    Pref = both.pivot_table(index='drug', columns='cell', values='response')
    return Pref.where(template.reindex(index=Pref.index, columns=Pref.columns).notna())


# ----------------------------------------------------------------------------------------------------------------
# a generic within-screen engine (full per-drug arrays; used for capture and rankings)
def within_engine(Z, G, Pm, W, V, common_mask=True, rank_models=()):
    """Z, G: drug x cell (same columns as the weight columns); Pm: {model: drug x cell, columns aligned to Z};
    W: (R, n_cells); V: (R, n_drugs aligned to Z.index). common_mask: per drug, cells where every model predicting the
    drug is finite (59 within); otherwise each model on its own finite cells (68 within).
    Returns {stat: (R, M, D) float32} for raw, r_pg, r_zg, sel; plus models, drugs, n_cells (M, D)."""
    models = list(Pm)
    drugs = list(Z.index)
    R, M, D = W.shape[0], len(models), len(drugs)
    out = {s: np.full((R, M, D), np.nan, np.float32) for s in ('raw', 'r_pg', 'r_zg', 'sel')}
    ncell = np.zeros((M, D), int)
    Zv, Gv = Z.values.astype(float), G.reindex(index=Z.index, columns=Z.columns).values.astype(float)
    Pv = {m: Pm[m].reindex(index=Z.index, columns=Z.columns).values.astype(float) for m in models}
    for j, d in enumerate(drugs):
        z, g = Zv[j], Gv[j]
        base = np.isfinite(z) & np.isfinite(g)
        have = [m for m in models if np.isfinite(Pv[m][j]).any()]
        if not have:
            continue
        if common_mask:
            mk = base.copy()
            for m in have:
                mk &= np.isfinite(Pv[m][j])
            groups = [(mk, have)]
        else:
            groups = [(base & np.isfinite(Pv[m][j]), [m]) for m in have]
        for mk, ms in groups:
            if mk.sum() < MIN_CELLS:
                continue
            idx = np.where(mk)[0]
            keep = [m for m in ms if np.std(Pv[m][j][idx]) >= 1e-12]
            if not keep:
                continue
            st = within_stats(z[idx], g[idx], np.column_stack([Pv[m][j][idx] for m in keep]), W[:, idx])
            for jj, m in enumerate(keep):
                i = models.index(m)
                for s in out:
                    out[s][:, i, j] = st[s][:, jj]
                ncell[i, j] = len(idx)
    return out, models, drugs, ncell
