"""Round 2, task 7 (R2-Major 1, R1-M3a-c; editor item 2): details of the harmonization ladder.

(i)  ln IC50 on the same drugs. Each ln IC50 target scored on exactly the drugs scored in its range-censored version
     (58 for GDSC1): unharmonized (CurveCurator ln IC50, each dataset's own panel and axes), shared panel only, and
     shared panel + censoring to the common range (80_paired_engine.py; 21 published models; 42's two-way draws).
(ii) Training-side and target-side scores at each step (median over the 21 published models of per-model median
     per-drug r; 42/66 summaries): full panel, shared panel, common range (AUC), and the ln IC50 steps.
(iii) The range step decomposed. For each target T and each drug shared with CTRP v2, every CTRP v2 curve (DrEval's
     CurveCurator fits, as in 66) is summarized as mean fitted viability over (a) the common window [lo_c, hi_c],
     (b) CTRP v2's high-concentration tail beyond T's range [hi_c, hi_CTRPv2] and (c) its low tail [lo_CTRPv2, lo_c]
     (per-drug median tested ranges; tails used where they span >= 0.5 log10 units), besides (d) the full tested range
     (CurveCurator AUC). On the drugs with a high tail: (1) correlations across cell lines of the cell-level axes (mean
     rank-normal profile over the drugs) of every CTRP v2 summary with T's common-window and full-range axes;
     (2) which summary carries the CTRP v2-specific axis component, i.e. the part of CTRP v2's full-range axis not
     explained by T's full-range axis (correlation of that residual with the window and tail axes); (3) whether the
     high-dose phenotype is reproduced by a second screen that tested those concentrations: CTRP v1 curves summarized
     over the same tail and window (same institution, so an upper bound on cross-institution reproducibility);
     (4) how the leaderboard predictions score against each CTRP v2 summary (median raw and selective r, axis share).
     Two-way bootstrap (cell weights; drug multiplicities in the axis means), B = 1,000.
Outputs (results/revision_round2/harmonization/): lnic50_same_drugs.csv, training_side_steps.csv,
range_axes_correlations.csv, range_specific_component.csv, range_tail_reproducibility.csv, range_model_scores.csv,
summary.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

PAIRED = lib.OUT2 / 'paired'
OUT = lib.out_dir('harmonization')
TAIL_MIN = 0.5
MIN_DRUGS_AXIS = 10
SEED = 20261061
TARGETS = ['GDSC1', 'GDSC2', 'CTRPv1', 'CCLE']


# ----------------------------------------------------------------------------------------------------------------
def part_i():
    ix = pd.read_csv(PAIRED / 'pairs.csv')
    lb = ix[ix.benchmark == 'leaderboard'].set_index('pair')
    rows = []
    per = pd.read_csv(PAIRED / 'per_drug_leaderboard.csv.gz', dtype={'drug': str})
    pub = None
    for T in TARGETS:
        for lab, pair in [('unharmonized, all drugs', f'CTRPv2|CC|lnIC50->{T}|CC|lnIC50'),
                          ('unharmonized, range-censored drugs', f'CTRPv2|CC|lnIC50->{T}|CC|lnIC50@RLdrugs'),
                          ('shared panel, all drugs', f'CTRPv2|SL-{T}|lnIC50->{T}|SL|lnIC50'),
                          ('shared panel, range-censored drugs', f'CTRPv2|SL-{T}|lnIC50->{T}|SL|lnIC50@RLdrugs'),
                          ('shared panel + range-censored', f'CTRPv2|RL-{T}|lnIC50->{T}|RL|lnIC50')]:
            r = lb.loc[pair]
            models, meta, S = lib.load_paired(PAIRED / 'arrays' / f'leaderboard__{r.k}.npz')
            gm = r.pool_main.split(';')
            pub = gm
            g = lambda st, kind: lib.pooled(S, models, gm, st, kind)  # noqa: E731
            n = g('raw', 'n')
            with np.errstate(invalid='ignore', divide='ignore'):
                st = {'mediated_retained': g('mediated', 'sum_c') / g('mediated', 'sum_w'),
                      'alignment_retained': g('r_pg', 'sum_c') / g('r_pg', 'sum_w'),
                      'loading_retained': g('r_zg', 'sum_c') / g('r_zg', 'sum_w'),
                      'raw_mean_within': g('raw', 'sum_w') / n, 'raw_mean_cross': g('raw', 'sum_c') / n,
                      'raw_loss': (g('raw', 'sum_w') - g('raw', 'sum_c')) / n,
                      'share_of_loss': (g('mediated', 'sum_w') - g('mediated', 'sum_c')) / (g('raw', 'sum_w') - g('raw', 'sum_c')),
                      'excess_over_within_share': (g('mediated', 'sum_w') - g('mediated', 'sum_c')) /
                      (g('raw', 'sum_w') - g('raw', 'sum_c')) - g('mediated', 'sum_w') / g('raw', 'sum_w'),
                      'd_selBOTH_paper': (g('selBOTH_asym', 'sum_c') - g('selBOTH_asym', 'sum_w')) / g('selBOTH_asym', 'n')}
            p = per[(per.pair == pair) & per.model.isin(gm)]
            med_w = p.groupby('model').raw_w.median().median()
            med_c = p.groupby('model').raw_c.median().median()
            row = {'target': T, 'version': lab, 'pair': pair, 'drugs': int(p.drug.nunique()),
                   'median_raw_within_point': float(med_w), 'median_raw_cross_point': float(med_c)}
            for k, v in st.items():
                row.update(lib.summ(v, k))
            rows.append(row)
    return pd.DataFrame(rows)


def part_ii():
    a = pd.read_csv(lib.RES / 'revision_crosslab' / 'summary_by_target.csv')
    b = pd.read_csv(lib.RES / 'revision_crosslab' / 'harmonized_range' / 'summary_by_target.csv')
    S = pd.concat([a, b])
    S = S[(S.subset == 'all') & (S.cellset == 'own')].set_index('pair')
    rows = []
    for T in TARGETS:
        for step, pair in [('AUC: full panel, own ranges', f'CTRPv2|CC|AUC->{T}|CC|AUC'),
                           ('AUC: shared panel, own ranges', f'CTRPv2|SP-{T}|AUC->{T}|SP|AUC'),
                           ('AUC: shared panel, common range', f'CTRPv2|HR-{T}|AUC->{T}|HR|AUC'),
                           ('ln IC50: full panel', f'CTRPv2|CC|lnIC50->{T}|CC|lnIC50'),
                           ('ln IC50: shared panel', f'CTRPv2|SL-{T}|lnIC50->{T}|SL|lnIC50'),
                           ('ln IC50: shared panel, range-censored', f'CTRPv2|RL-{T}|lnIC50->{T}|RL|lnIC50')]:
            if pair not in S.index:
                continue
            r = S.loc[pair]
            rows.append({'target': T, 'step': step, 'pair': pair, 'drugs': int(r.drugs),
                         'training_side_median_raw': r.raw_median_within, 'training_side_lo': r.raw_median_within_lo,
                         'training_side_hi': r.raw_median_within_hi, 'target_side_median_raw': r.raw_median_cross,
                         'target_side_lo': r.raw_median_cross_lo, 'target_side_hi': r.raw_median_cross_hi,
                         'mediated_retained': r.mediated_retained, 'mediated_retained_lo': r.mediated_retained_lo,
                         'mediated_retained_hi': r.mediated_retained_hi, 'raw_loss_mean': r.raw_loss,
                         'raw_loss_lo': r.raw_loss_lo, 'raw_loss_hi': r.raw_loss_hi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
def summaries(frame, drugs, rng_lo, rng_hi):
    """Long table (drug, cell, response = -mean viability over [rng_lo[d], rng_hi[d]])."""
    hr = lib.load_module('hr66', '66_harmonized_range.py')
    x = frame[frame.pubchem_id.isin(drugs)].copy()
    lo = rng_lo.reindex(x.pubchem_id).values
    hi = rng_hi.reindex(x.pubchem_id).values
    x['v'] = hr.mean_viability(x, lo, hi)
    x = x.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).v.mean()
    return pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values, 'response': -x.v.values})


def own_auc(frame, drugs):
    x = frame[frame.pubchem_id.isin(drugs)].groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).AUC_curvecurator.mean()
    return pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values, 'response': -x.AUC_curvecurator.values})


def axis_reps(Z, cells, W, V):
    """Cell-level axis (mean rank-normal profile over drugs, drug multiplicities V) on `cells`: (R, n_cells)."""
    M = Z.reindex(columns=cells).values.astype(float)
    ok = np.isfinite(M)
    A = (V @ np.where(ok, M, 0.0)) / np.maximum(V @ ok, 1e-12)
    cnt = ok.sum(0)
    A[:, cnt < MIN_DRUGS_AXIS] = np.nan
    return A


def wcorr(a, b, W):
    ok = np.isfinite(a) & np.isfinite(b)
    w = np.where(ok, W, 0.0)
    a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    n = w.sum(1)
    ma, mb = (w * a).sum(1) / n, (w * b).sum(1) / n
    ca = (w * (a - ma[:, None]) ** 2).sum(1)
    cb = (w * (b - mb[:, None]) ** 2).sum(1)
    cab = (w * (a - ma[:, None]) * (b - mb[:, None])).sum(1)
    return cab / np.sqrt(ca * cb)


def wresid(y, x, W):
    """Weighted residual of y on x (per replicate) -> (R, n)."""
    ok = np.isfinite(y) & np.isfinite(x)
    w = np.where(ok, W, 0.0)
    y0, x0 = np.where(ok, y, 0.0), np.where(ok, x, 0.0)
    n = w.sum(1)
    my, mx = (w * y0).sum(1) / n, (w * x0).sum(1) / n
    b = (w * (y0 - my[:, None]) * (x0 - mx[:, None])).sum(1) / (w * (x0 - mx[:, None]) ** 2).sum(1)
    return np.where(ok, y - my[:, None] - b[:, None] * (x - mx[:, None]), np.nan)


def part_iii():
    hr = lib.load_module('hr66', '66_harmonized_range.py')
    S = hr.curves('CTRPv2')
    rS = S.groupby('pubchem_id')[['lo', 'hi']].median()
    V1 = hr.curves('CTRPv1')
    r1 = V1.groupby('pubchem_id')[['lo', 'hi']].median()
    L = lib.leaderboard_setup()
    pub = L['published21']
    corr_rows, spec_rows, rep_rows, score_rows = [], [], [], []
    for k, T in enumerate(['GDSC1', 'GDSC2', 'CCLE']):
        D = hr.curves(T)
        rT = D.groupby('pubchem_id')[['lo', 'hi']].median()
        sh = rS.index.intersection(rT.index)
        lo_c = np.maximum(rS.loc[sh, 'lo'], rT.loc[sh, 'lo'])
        hi_c = np.minimum(rS.loc[sh, 'hi'], rT.loc[sh, 'hi'])
        keep = sh[(hi_c - lo_c) >= 1.0]
        lo_c, hi_c = lo_c[keep], hi_c[keep]
        htail = keep[(rS.loc[keep, 'hi'] - hi_c) >= TAIL_MIN]
        ltail = keep[(lo_c - rS.loc[keep, 'lo']) >= TAIL_MIN]
        drugs = htail
        vers = {'CTRPv2 window': summaries(S, drugs, lo_c, hi_c),
                'CTRPv2 high tail': summaries(S, drugs, hi_c, rS['hi']),
                'CTRPv2 full range': own_auc(S, drugs),
                f'{T} window': summaries(D, drugs, lo_c, hi_c),
                f'{T} full range': own_auc(D, drugs)}
        lt = [d for d in drugs if d in set(ltail)]
        if len(lt) >= MIN_DRUGS_AXIS:
            vers['CTRPv2 low tail'] = summaries(S, lt, rS['lo'], lo_c)
        # CTRP v1 over the same window and tail (drugs whose CTRP v1 range reaches the CTRP v2 tail)
        # CTRP v1 covers CTRP v2's top concentrations but starts higher: window [max(lo_c, lo_v1), hi_c] (>= 1 log10)
        lo1 = pd.Series({d: max(lo_c[d], r1.loc[d, 'lo']) for d in drugs if d in r1.index})
        v1d = [d for d in lo1.index if r1.loc[d, 'hi'] >= rS.loc[d, 'hi'] - 0.3 and hi_c[d] - lo1[d] >= 1.0]
        if len(v1d) >= MIN_DRUGS_AXIS:
            vers['CTRPv1 window'] = summaries(V1, v1d, lo1, hi_c)
            vers['CTRPv1 high tail'] = summaries(V1, v1d, hi_c, rS['hi'])
            vers['CTRPv2 window (v1 drugs)'] = summaries(S, v1d, lo1, hi_c)
            vers['CTRPv2 high tail (v1 drugs)'] = summaries(S, v1d, hi_c, rS['hi'])
        dec = {v: lib.selbench.decompose(x) for v, x in vers.items()}
        cells = sorted(set.intersection(*[set(d[0].columns) for v, d in dec.items() if not v.startswith('CTRPv1')
                                          and '(v1 drugs)' not in v]))
        dl = sorted(set.intersection(*[set(dec[v][0].index) for v in dec if not v.startswith('CTRPv1')
                                       and '(v1 drugs)' not in v and v != 'CTRPv2 low tail']))
        W, Vd = lib.draws(len(cells), len(dl), SEED + k)
        AX = {v: axis_reps(dec[v][0].reindex(index=dl), cells, W * 0 + 1, Vd) for v in dec
              if not v.startswith('CTRPv1') and '(v1 drugs)' not in v and v != 'CTRPv2 low tail'}
        if 'CTRPv2 low tail' in dec:      # drugs without a low tail are missing rows (excluded from the means)
            AX['CTRPv2 low tail'] = axis_reps(dec['CTRPv2 low tail'][0].reindex(index=dl), cells, None, Vd)
        names = list(AX)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                c = wcorr(AX[a], AX[b], W)
                corr_rows.append({'target': T, 'axis_a': a, 'axis_b': b, 'drugs': len(dl), 'cells': len(cells),
                                  **lib.summ(c, 'r')})
        # CTRPv2-specific component of the full-range axis (residual on T's full-range axis)
        res = wresid(AX['CTRPv2 full range'], AX[f'{T} full range'], W)
        shared = AX['CTRPv2 full range'] - res
        for comp, arr in [('CTRPv2-specific (residual on target full-range axis)', res),
                          ('shared with target (fitted part)', shared)]:
            for v in ['CTRPv2 window', 'CTRPv2 high tail', 'CTRPv2 low tail']:
                if v in AX:
                    spec_rows.append({'target': T, 'component': comp, 'with_axis': v, **lib.summ(wcorr(arr, AX[v], W), 'r')})
        # partial: does the high tail add target-shared signal beyond the window?
        rt = wresid(AX['CTRPv2 high tail'], AX['CTRPv2 window'], W)
        spec_rows.append({'target': T, 'component': 'high tail residual on window', 'with_axis': f'{T} full range',
                          **lib.summ(wcorr(rt, AX[f'{T} full range'], W), 'r')})
        spec_rows.append({'target': T, 'component': 'high tail residual on window', 'with_axis': 'CTRPv2 full range',
                          **lib.summ(wcorr(rt, AX['CTRPv2 full range'], W), 'r')})
        # R^2 of CTRPv2's full-range axis on window and high tail
        for v in ['CTRPv2 window', 'CTRPv2 high tail']:
            spec_rows.append({'target': T, 'component': 'variance of CTRPv2 full-range axis explained (r^2)', 'with_axis': v,
                              **lib.summ(wcorr(AX['CTRPv2 full range'], AX[v], W) ** 2, 'r')})
        # reproducibility of the tail in CTRP v1
        if 'CTRPv1 window' in dec:
            c1 = sorted(set(dec['CTRPv1 window'][0].columns) & set(dec['CTRPv2 window (v1 drugs)'][0].columns))
            d1 = sorted(set(dec['CTRPv1 window'][0].index) & set(dec['CTRPv2 window (v1 drugs)'][0].index)
                        & set(dec['CTRPv1 high tail'][0].index) & set(dec['CTRPv2 high tail (v1 drugs)'][0].index))
            W1, V1d = lib.draws(len(c1), len(d1), SEED + 100 + k)
            A1 = {v: axis_reps(dec[v][0].reindex(index=d1), c1, None, V1d)
                  for v in ['CTRPv1 window', 'CTRPv1 high tail', 'CTRPv2 window (v1 drugs)', 'CTRPv2 high tail (v1 drugs)']}
            for a, b in [('CTRPv2 window (v1 drugs)', 'CTRPv1 window'), ('CTRPv2 high tail (v1 drugs)', 'CTRPv1 high tail'),
                         ('CTRPv2 window (v1 drugs)', 'CTRPv2 high tail (v1 drugs)'), ('CTRPv1 window', 'CTRPv1 high tail')]:
                rep_rows.append({'target_range_from': T, 'axis_a': a, 'axis_b': b, 'drugs': len(d1), 'cells': len(c1),
                                 **lib.summ(wcorr(A1[a], A1[b], W1), 'r')})
            # per-drug profile agreement CTRP v1 vs v2 in the window and in the tail (median over drugs)
            for lab, a, b in [('window', 'CTRPv2 window (v1 drugs)', 'CTRPv1 window'),
                              ('high tail', 'CTRPv2 high tail (v1 drugs)', 'CTRPv1 high tail')]:
                Za, Zb = dec[a][0], dec[b][0]
                rr = []
                for d in d1:
                    x, y = Za.loc[d].reindex(c1).values, Zb.loc[d].reindex(c1).values
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() >= 30:
                        rr.append(np.corrcoef(x[m], y[m])[0, 1])
                rep_rows.append({'target_range_from': T, 'axis_a': f'per-drug profile agreement CTRP v2 vs v1, {lab}',
                                 'axis_b': 'median over drugs', 'drugs': len(rr), 'cells': len(c1),
                                 'r': float(np.median(rr)) if rr else np.nan, 'r_lo': np.nan, 'r_hi': np.nan})
        # leaderboard predictions scored against each CTRP v2 summary (within, same drugs and cells)
        U = L['U']
        for v in ['CTRPv2 window', 'CTRPv2 high tail', 'CTRPv2 full range'] + (['CTRPv2 low tail'] if 'CTRPv2 low tail' in dec else []):
            Z, G = dec[v][0], dec[v][1]
            Zu, Gu = Z.reindex(columns=U), G.reindex(columns=U)
            dd = [d for d in Zu.index if d in L['drug_ix']]
            Wl, Vl = lib.draws(len(U), len(dd), SEED + 200 + k)
            Pm = {m: L['P'][m].reindex(index=dd, columns=U) for m in pub}
            arrs, models, drugs, nc = lib.within_engine(Zu.reindex(index=dd), Gu.reindex(index=dd), Pm, Wl, Vl,
                                                        common_mask=True)
            meds_raw = np.vstack([lib.wmedian_rows(arrs['raw'][:, i], Vl) for i in range(len(models))])
            meds_sel = np.vstack([lib.wmedian_rows(arrs['sel'][:, i], Vl) for i in range(len(models))])
            ok = np.isfinite(arrs['raw']) & np.isfinite(arrs['r_pg'])
            w = np.where(ok, Vl[:, None, :], 0)
            share = (w * np.where(ok, arrs['r_pg'] * arrs['r_zg'], 0)).sum((1, 2)) / (w * np.where(ok, arrs['raw'], 0)).sum((1, 2))
            rz2 = (w * np.where(ok, arrs['r_zg'] ** 2, 0)).sum((1, 2)) / w.sum((1, 2))
            score_rows.append({'target_range_from': T, 'version': v, 'drugs': int(np.isfinite(arrs['raw'][0]).any(0).sum()),
                               **lib.summ(np.nanmedian(meds_raw, 0), 'median_raw'),
                               **lib.summ(np.nanmedian(meds_sel, 0), 'median_selective'),
                               **lib.summ(share, 'axis_share_pooled'), **lib.summ(rz2, 'perfect_share')})
        print('range step', T, len(drugs), 'high-tail drugs', len(lt), 'low-tail', flush=True)
    return (pd.DataFrame(corr_rows), pd.DataFrame(spec_rows), pd.DataFrame(rep_rows), pd.DataFrame(score_rows))


def main():
    t0 = time.time()
    A = part_i()
    A.to_csv(OUT / 'lnic50_same_drugs.csv', index=False, float_format='%.5g')
    B = part_ii()
    B.to_csv(OUT / 'training_side_steps.csv', index=False, float_format='%.5g')
    C, Sp, Rp, Sc = part_iii()
    C.to_csv(OUT / 'range_axes_correlations.csv', index=False, float_format='%.5g')
    Sp.to_csv(OUT / 'range_specific_component.csv', index=False, float_format='%.5g')
    Rp.to_csv(OUT / 'range_tail_reproducibility.csv', index=False, float_format='%.5g')
    Sc.to_csv(OUT / 'range_model_scores.csv', index=False, float_format='%.5g')
    lib.write_json(OUT / 'summary.json', {'seconds': round(time.time() - t0), 'tail_min_log10': TAIL_MIN,
                                          'min_drugs_axis': MIN_DRUGS_AXIS})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 200)
    for nm, X in [('i', A), ('ii', B), ('iii corr', C), ('iii specific', Sp), ('iii reproducibility', Rp), ('iii scores', Sc)]:
        print('====', nm)
        print(X.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
