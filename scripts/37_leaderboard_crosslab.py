"""Paired within- versus cross-laboratory scores of the 27 DrEval leaderboard models and this study's references.

The leaderboard (Zenodo 23159774, leaderboard_Oct26) released leave-cell-line-out predictions on CTRPv2 (CurveCurator
ln IC50; identical values to the CTRPv2 data of DrEval's full results). Each out-of-fold prediction is scored against
CTRPv2's measurement and against the measurement of the same drug and cell line in another laboratory's screen as
curated by DrEval (CTRPv1, CCLE, GDSC1, GDSC2; taken from the cross-study files of the full results), exactly as in
35_dreval_crossstudy.py. This study's references (fitness-only ridge and per-drug kernel ridge on the raw or selective
target, trained on the same leaderboard splits; 28_dreval_rescore.py) are scored on the same drugs and cell lines.
Outputs: results/dreval_crosslab_leaderboard/{per_drug.csv.gz, summary.csv, paired_selective_cross.csv, run.json}.
"""
import importlib.util
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common import PROJECT, RESULTS, write_json  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


cs = load('cs', '35_dreval_crossstudy.py')
rs = load('rs', '28_dreval_rescore.py')
OUT = RESULTS / 'dreval_crosslab_leaderboard'
B = 2000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261014)
    cid = pd.read_csv(PROJECT / 'data' / 'dreval' / 'cell_ids.csv', dtype=str).set_index('cell_line_name').cellosaurus_id
    lb = rs.load_leaderboard()
    refs = pd.read_csv(RESULTS / 'dreval' / 'reference_predictions.csv.gz', dtype={'drug': str})
    models = {m: d[['drug', 'cell', 'prediction']] for m, d in lb.items() if m not in rs.CONSTANT}
    for m in rs.OURS:
        models[m] = refs[refs.model == m][['drug', 'cell', 'prediction']]
    P = {m: d.assign(cell=d.cell.map(cid)).dropna(subset=['cell']).groupby(['drug', 'cell']).prediction.mean().unstack()
         for m, d in models.items()}
    Zs, Gs, _, _ = cs.selbench.decompose(cs.observed('predictions'))
    published = [m for m in P if m not in rs.OURS and m not in rs.NAIVE]
    rows, info = [], {}
    for target in cs.TARGETS:
        Zt, Gt, _, _ = cs.selbench.decompose(cs.observed(f'cross-study-{target}'))
        drugs = Zs.index.intersection(Zt.index)
        cells = Zs.columns.intersection(Zt.columns)
        for m in P:                                  # cells predicted by every model
            cells = cells.intersection(P[m].columns)
        info[target] = {'paired_drugs': len(drugs), 'paired_cells': len(cells)}
        for m, M in P.items():
            tw, tc = cs.score_pair(M.reindex(index=drugs), Zs, Gs, Zt, Gt, cells)
            for where, t in [('within', tw), ('cross', tc)]:
                if len(t):
                    rows.append(t.assign(model=m, target=target, evaluated_in=where))
        print(target, info[target], flush=True)
    per = pd.concat(rows, ignore_index=True)
    per['class'] = per.model.map(rs.model_class)
    per.to_csv(OUT / 'per_drug.csv.gz', index=False)
    w = per.pivot_table(index=['model', 'target', 'drug'], columns='evaluated_in',
                        values=['raw', 'selective', 'mediated', 'selective_term']).dropna()
    w.columns = [f'{v}_{e}' for v, e in w.columns]
    w = w.reset_index()
    S = w.groupby(['model', 'target']).agg(
        drugs=('drug', 'size'), raw_within=('raw_within', 'median'), raw_cross=('raw_cross', 'median'),
        sel_within=('selective_within', 'median'), sel_cross=('selective_cross', 'median'),
        mediated_within=('mediated_within', 'mean'), mediated_cross=('mediated_cross', 'mean'),
        selterm_within=('selective_term_within', 'mean'), selterm_cross=('selective_term_cross', 'mean'),
        meanraw_within=('raw_within', 'mean'), meanraw_cross=('raw_cross', 'mean')).reset_index()
    S['raw_drop'] = S.meanraw_within - S.meanraw_cross
    S['share_of_drop_from_fitness'] = (S.mediated_within - S.mediated_cross) / S.raw_drop
    S['mediated_retained'] = S.mediated_cross / S.mediated_within
    S['selterm_retained'] = S.selterm_cross / S.selterm_within
    S['class'] = S.model.map(rs.model_class)
    S.to_csv(OUT / 'summary.csv', index=False)
    # drug-for-drug: this study's selective-target ridge versus every published model, on cross-laboratory selectivity
    comp = []
    for target in cs.TARGETS:
        X = w[w.target == target].pivot_table(index='drug', columns='model', values='selective_cross')
        for ours in ['ridge_sel_lm', 'ridge_sel_expr', 'ridge_raw_lm']:
            for m in published:
                if ours not in X or m not in X:
                    continue
                x = (X[ours] - X[m]).dropna()
                if len(x) < 15:
                    continue
                bs = np.median(x.values[rng.integers(0, len(x), (B, len(x)))], axis=1)
                comp.append({'target': target, 'ours': ours, 'model': m, 'drugs': len(x), 'median_diff': float(x.median()),
                             'lo': float(np.percentile(bs, 2.5)), 'hi': float(np.percentile(bs, 97.5)),
                             'frac_better': float((x > 0).mean()), 'wilcoxon_p': float(wilcoxon(x).pvalue)})
    pd.DataFrame(comp).to_csv(OUT / 'paired_selective_cross.csv', index=False)
    write_json(OUT / 'run.json', info)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
    for target in ['GDSC1', 'GDSC2', 'CTRPv1']:
        print('==', target)
        print(S[S.target == target].sort_values('sel_cross', ascending=False)[
            ['model', 'class', 'drugs', 'raw_within', 'raw_cross', 'sel_within', 'sel_cross', 'share_of_drop_from_fitness',
             'mediated_retained', 'selterm_retained']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
