"""Paired within- versus cross-laboratory re-scoring of DrEval's full published results (Zenodo 23159774, all_results).

DrEval trained every model on CTRPv2 (CurveCurator ln IC50) and released out-of-fold predictions for CTRPv2 itself
and, without retraining, for CCLE, CTRPv1, GDSC1 and GDSC2, under four test modes (LPO, LCO, LDO, LTO). For each drug,
on the cell lines measured in both CTRPv2 and a target dataset T, the SAME out-of-fold prediction (from the CTRPv2 run)
is scored against CTRPv2's measurement and against T's measurement (identical drugs, cell lines and predictions). Per drug: raw correlation, selective (partial)
correlation given the evaluation dataset's leave-drug-out fitness axis, and the exact split
raw = r_pg r_zg + selective term. Any change between the two scores is due to the laboratory alone.

References: CTRPv2's measured fitness axis (the source laboratory's drug-agnostic cell ranking) and CTRPv2's measured
response itself (a replicate measurement in another laboratory: the cross-study ceiling of each drug).
Outputs: results/dreval_crossstudy/{per_drug.csv.gz, summary.csv, run.json}. Responses are oriented so that higher
means more sensitive (ln IC50 negated).
"""
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import PROJECT, RESULTS, write_json  # noqa: E402

sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402

SRC = PROJECT / 'data' / 'dreval' / 'all_results_parquet'
OUT = RESULTS / 'dreval_crossstudy'
TARGETS = ['CTRPv1', 'CCLE', 'GDSC1', 'GDSC2']
MODES = ['LCO', 'LTO', 'LPO', 'LDO']
CONSTANT = {'NaivePredictor', 'NaiveDrugMeanPredictor'}


def observed(prefix):
    """Union of measured responses over the test-mode files of one evaluation setting (drug x cell, sensitivity)."""
    parts = [pd.read_parquet(SRC / f'{prefix}__{m}.parquet', columns=['pubchem_id', 'cellosaurus_id', 'y_true'])
             for m in MODES if (SRC / f'{prefix}__{m}.parquet').exists()]
    o = pd.concat(parts).groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).y_true.mean()
    return o.rename(columns={'pubchem_id': 'drug', 'cellosaurus_id': 'cell'}).assign(response=lambda t: -t.y_true)


def _terms(p, z, g):
    c = np.corrcoef(np.vstack([p, z, g]))
    r_pz, r_pg, r_zg = c[0, 1], c[0, 2], c[1, 2]
    den = np.sqrt(max((1 - r_pg ** 2) * (1 - r_zg ** 2), 0))
    return {'raw': r_pz, 'selective': (r_pz - r_pg * r_zg) / den if den > 1e-9 else np.nan,
            'alignment': r_pg, 'mediated': r_pg * r_zg, 'selective_term': r_pz - r_pg * r_zg}


def score(P, Z, G, cells):
    """Per-drug raw, selective and fitness-mediated scores against one screen."""
    rows = []
    for d in P.index.intersection(Z.index):
        p = P.loc[d].reindex(cells).values.astype(float)
        z = Z.loc[d].reindex(cells).values.astype(float)
        g = G.loc[d].reindex(cells).values.astype(float)
        m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
        if m.sum() < 30 or np.std(p[m]) < 1e-12:
            continue
        rows.append({'drug': d, 'n': int(m.sum()), **_terms(p[m], z[m], g[m])})
    return pd.DataFrame(rows)


def score_pair(P, Zs, Gs, Zt, Gt, cells):
    """Score the same predictions against two screens on, for each drug, the SAME cell lines (measured in both)."""
    within, cross = [], []
    for d in P.index.intersection(Zs.index).intersection(Zt.index):
        p = P.loc[d].reindex(cells).values.astype(float)
        zs, gs = Zs.loc[d].reindex(cells).values.astype(float), Gs.loc[d].reindex(cells).values.astype(float)
        zt, gt = Zt.loc[d].reindex(cells).values.astype(float), Gt.loc[d].reindex(cells).values.astype(float)
        m = np.isfinite(p) & np.isfinite(zs) & np.isfinite(gs) & np.isfinite(zt) & np.isfinite(gt)
        if m.sum() < 30 or np.std(p[m]) < 1e-12:
            continue
        within.append({'drug': d, 'n': int(m.sum()), **_terms(p[m], zs[m], gs[m])})
        cross.append({'drug': d, 'n': int(m.sum()), **_terms(p[m], zt[m], gt[m])})
    return pd.DataFrame(within), pd.DataFrame(cross)


def run(args):
    mode, target = args
    src = observed('predictions')
    tgt = observed(f'cross-study-{target}')
    Zs, Gs, Rs, _ = selbench.decompose(src)
    Zt, Gt, Rt, _ = selbench.decompose(tgt)
    pred = pd.read_parquet(SRC / f'predictions__{mode}.parquet')
    pred = pred.groupby(['algorithm', 'pubchem_id', 'cellosaurus_id'], as_index=False).y_pred.mean()
    drugs = Zs.index.intersection(Zt.index)
    cells = Zs.columns.intersection(Zt.columns)
    out = []
    preds = {a: (-g.pivot(index='pubchem_id', columns='cellosaurus_id', values='y_pred')).reindex(index=drugs)
             for a, g in pred.groupby('algorithm') if a not in CONSTANT}
    # references: the source laboratory's measured fitness axis and its measured response (cross-laboratory ceiling)
    gs = Zs.mean(axis=0)
    preds['ref:CTRPv2_fitness_axis'] = pd.DataFrame(np.tile(gs.reindex(cells).values, (len(drugs), 1)), index=drugs, columns=cells)
    preds['ref:CTRPv2_measured'] = Zs.reindex(index=drugs, columns=cells)
    for a, P in preds.items():
        tw, tc = score_pair(P, Zs, Gs, Zt, Gt, cells)
        for where, t in [('within', tw), ('cross', tc)]:
            if len(t):
                out.append(t.assign(algorithm=a, mode=mode, target=target, evaluated_in=where))
    return pd.concat(out) if out else pd.DataFrame(), {'mode': mode, 'target': target, 'paired_drugs': len(drugs),
                                                       'paired_cells': len(cells)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(m, t) for m in MODES for t in TARGETS if (SRC / f'predictions__{m}.parquet').exists()]
    per, info = [], []
    with ProcessPoolExecutor(int(os.environ.get('WORKERS', 6))) as ex:
        for t, i in ex.map(run, jobs):
            per.append(t); info.append(i)
            print(i, flush=True)
    per = pd.concat(per, ignore_index=True)
    per.to_csv(OUT / 'per_drug.csv.gz', index=False)
    # summaries over drugs scored in both laboratories for each model
    key = ['algorithm', 'mode', 'target']
    w = per.pivot_table(index=key + ['drug'], columns='evaluated_in',
                        values=['raw', 'selective', 'mediated', 'selective_term']).dropna()
    w.columns = [f'{v}_{e}' for v, e in w.columns]
    w = w.reset_index()
    S = w.groupby(key).agg(drugs=('drug', 'size'),
                           raw_within=('raw_within', 'median'), raw_cross=('raw_cross', 'median'),
                           sel_within=('selective_within', 'median'), sel_cross=('selective_cross', 'median'),
                           mediated_within=('mediated_within', 'mean'), mediated_cross=('mediated_cross', 'mean'),
                           selterm_within=('selective_term_within', 'mean'), selterm_cross=('selective_term_cross', 'mean'),
                           meanraw_within=('raw_within', 'mean'), meanraw_cross=('raw_cross', 'mean')).reset_index()
    S['raw_drop'] = S.meanraw_within - S.meanraw_cross
    S['share_of_drop_from_fitness'] = (S.mediated_within - S.mediated_cross) / S.raw_drop
    S['mediated_retained'] = S.mediated_cross / S.mediated_within
    S['selterm_retained'] = S.selterm_cross / S.selterm_within
    S.to_csv(OUT / 'summary.csv', index=False)
    write_json(OUT / 'run.json', {'jobs': info})
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
    cols = key + ['drugs', 'raw_within', 'raw_cross', 'sel_within', 'sel_cross', 'share_of_drop_from_fitness',
                  'mediated_retained', 'selterm_retained']
    print(S[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
