"""Validation of the extended selbench against the published DrEval re-scoring, and a demonstration of the protocol
components on the leaderboard (revision; Reviewer 1 M7, Reviewer 3 M8 and minor 8). Read-only on all existing files.

1. Reproduction. selbench.paired_scores_matrix, on exactly the inputs of scripts/37_leaderboard_crosslab.py, is
   compared with its per-drug table (results/dreval_crosslab_leaderboard/per_drug.csv.gz): every model, target screen
   and evaluation; maximum absolute differences are reported.
2. Model-axis diagnostic. selbench.model_axis_scores(scale='rank_normal') on the inputs of
   scripts/40_model_axis_adjusted.py (CTRPv2 leaderboard, common cells) is compared with
   results/dreval/model_axis_adjusted.csv; the z-scored default is reported next to it.
3. Demonstration (exploratory; not part of the submitted analyses). For the 21 published models pooled (as in
   scripts/38_crosslab_uncertainty.py: classical and neural/factorization models without SVR) and GDSC1/GDSC2/CTRPv1:
   two-way (drug x cell) and drug-only bootstrap intervals of the pooled retention statistics; per-model median
   selective scores under the evaluation axis, under both axes and with the model-axis diagnostic; and the leakage
   floor of the drug-agnostic predictors (fitness_only, NaiveTissueMeanPredictor, NaiveMeanEffectsPredictor).
Outputs: results/revision_repro/selbench_validation/{reproduction.csv, model_axis_ctrpv2.csv, demo_bootstrap.csv,
demo_models.csv, summary.json}. Run from scripts/ with ../../.venv-dreval/bin/python.
"""
import importlib.util
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = PROJECT / 'results'
OUT = RESULTS / 'revision_repro' / 'selbench_validation'
sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402

SEED = 20261014
N_BOOT = int(os.environ.get('N_BOOT', 1000))
AGNOSTIC = ['fitness_only', 'NaiveTissueMeanPredictor', 'NaiveMeanEffectsPredictor']


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cs = load('cs', '35_dreval_crossstudy.py')
    rs = load('rs', '28_dreval_rescore.py')
    assert Path(cs.selbench.__file__).resolve() == Path(selbench.__file__).resolve()
    ma_note = ('max_abs_diff_* compare with results/dreval/model_axis_adjusted.csv (rank-normal model axis; models of '
               'script 40 only); drug-agnostic predictors are added with the z-scored default')
    summary = {'selbench': str(Path(selbench.__file__).resolve()), 'n_boot': N_BOOT, 'seed': SEED}

    # ---- inputs exactly as in scripts/37_leaderboard_crosslab.py
    cid = pd.read_csv(PROJECT / 'data' / 'dreval' / 'cell_ids.csv', dtype=str).set_index('cell_line_name').cellosaurus_id
    lb = rs.load_leaderboard()
    refs = pd.read_csv(RESULTS / 'dreval' / 'reference_predictions.csv.gz', dtype={'drug': str})
    models = {m: d[['drug', 'cell', 'prediction']] for m, d in lb.items() if m not in rs.CONSTANT}
    for m in rs.OURS:
        models[m] = refs[refs.model == m][['drug', 'cell', 'prediction']]
    P = {m: d.assign(cell=d.cell.map(cid)).dropna(subset=['cell']).groupby(['drug', 'cell']).prediction.mean().unstack()
         for m, d in models.items()}
    Zs, Gs, _, _ = selbench.decompose(cs.observed('predictions'))
    published = [m for m in P if rs.model_class(m) in ('classical', 'neural/factorization') and m != 'SVR']
    summary['published_models'] = published

    # ---- 1. reproduction of the published per-drug table
    ref = pd.read_csv(RESULTS / 'dreval_crosslab_leaderboard' / 'per_drug.csv.gz', dtype={'drug': str})
    ref['evaluated_in'] = ref.evaluated_in.map({'within': 'source', 'cross': 'target'})
    rows, tables, screens = [], {}, {}
    for target in cs.TARGETS:
        Zt, Gt, _, _ = selbench.decompose(cs.observed(f'cross-study-{target}'))
        drugs = Zs.index.intersection(Zt.index)
        cells = Zs.columns.intersection(Zt.columns)
        for m in P:
            cells = cells.intersection(P[m].columns)
        screens[target] = (Zt, Gt, drugs, cells)
        for m, M in P.items():
            T = selbench.paired_scores_matrix(M.reindex(index=drugs), Zs, Gs, Zt, Gt, cells)
            tables[(m, target)] = T
            R = ref[(ref.model == m) & (ref.target == target)]
            J = T.merge(R, on=['drug', 'evaluated_in'], how='outer', suffixes=('', '_pub'), indicator=True)
            row = {'model': m, 'target': target, 'rows_selbench': len(T), 'rows_published': len(R),
                   'rows_matched': int((J._merge == 'both').sum())}
            for k in ['n', 'raw', 'selective', 'alignment', 'mediated', 'selective_term']:
                d = (J[k] - J[f'{k}_pub']).abs()
                row[f'maxdiff_{k}'] = float(d.max()) if len(d.dropna()) else np.nan
            rows.append(row)
        print('reproduced', target, flush=True)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / 'reproduction.csv', index=False)
    diffcols = [c for c in rep.columns if c.startswith('maxdiff_')]
    summary['reproduction'] = {
        'model_target_pairs': int(len(rep)),
        'all_rows_matched': bool((rep.rows_selbench == rep.rows_published).all()
                                 and (rep.rows_matched == rep.rows_published).all()),
        'max_abs_diff': float(np.nanmax(rep[diffcols].values))}

    # ---- 2. model-axis diagnostic, CTRPv2 leaderboard (inputs as in scripts/40_model_axis_adjusted.py)
    obs = pd.concat([d[['drug', 'cell', 'response']] for d in lb.values()]).groupby(['drug', 'cell'], as_index=False).response.mean()
    pred_cells = [set(d.cell) for m, d in lb.items() if m not in rs.NAIVE]
    common = sorted(set.intersection(*pred_cells) & set(refs.cell))
    Z, G, _, _ = selbench.decompose(obs[obs.cell.isin(common)])
    pub40 = pd.read_csv(RESULTS / 'dreval' / 'model_axis_adjusted.csv')
    pub40 = pub40[pub40['where'] == 'CTRPv2'].set_index('model')
    ma = []
    for m, d in models.items():
        if m not in pub40.index and m not in AGNOSTIC:
            continue
        Pm = d[d.cell.isin(common)].pivot_table(index='drug', columns='cell', values='prediction')
        A = selbench.model_axis_scores(Pm, Z, G, cells=pd.Index(common), scale='rank_normal')
        B = selbench.model_axis_scores(Pm, Z, G, cells=pd.Index(common), scale='zscore')
        ma.append({'model': m, 'class': rs.model_class(m), 'drugs': int(len(A)),
                   'selective_pub': float(pub40.loc[m, 'selective']) if m in pub40.index else np.nan,
                   'adjusted_pub': float(pub40.loc[m, 'adjusted']) if m in pub40.index else np.nan,
                   'complete_rows': float((Pm.reindex(columns=common).notna().mean(axis=1) == 1).mean()),
                   'selective_selbench': float(A.selective.median()), 'adjusted_rank_selbench': float(A.adjusted.median()),
                   'adjusted_zscore_selbench': float(B.adjusted.median())})
    ma = pd.DataFrame(ma)
    ma.to_csv(OUT / 'model_axis_ctrpv2.csv', index=False)
    summary['model_axis_ctrpv2'] = {
        'note': ma_note, 'models': int(len(ma)),
        'max_abs_diff_selective_median': float((ma.selective_pub - ma.selective_selbench).abs().max()),
        'max_abs_diff_adjusted_median_rank': float((ma.adjusted_pub - ma.adjusted_rank_selbench).abs().max()),
        'models_compared_with_script40': int(ma.selective_pub.notna().sum()),
        'drug_agnostic': ma[ma.model.isin(AGNOSTIC)].set_index('model')[
            ['selective_selbench', 'adjusted_rank_selbench', 'adjusted_zscore_selbench', 'complete_rows']].to_dict('index')}
    print('model axis done', flush=True)

    # ---- 3. demonstration on the cross-laboratory leaderboard
    boot, per_model = [], []
    for target in ['GDSC1', 'GDSC2', 'CTRPv1']:
        Zt, Gt, drugs, cells = screens[target]
        Pp = {m: P[m].reindex(index=drugs) for m in published}
        for mode in ['two_way', 'drug_only']:
            s, _ = selbench.two_way_bootstrap(Pp, Zs, Gs, Zt, Gt, n_boot=N_BOOT, seed=SEED, cells=cells,
                                              drug_only=(mode == 'drug_only'))
            boot.append(s.reset_index().rename(columns={'index': 'statistic'}).assign(target=target, bootstrap=mode))
        for m in published + [a for a in AGNOSTIC if a in P] + ['ridge_sel_lm', 'ridge_raw_lm']:
            T = tables[(m, target)]
            Mm = P[m].reindex(index=drugs)
            # model-axis diagnostic in the target screen, given both screens' axes (z-scored model axis)
            Gs_rows = selbench._axis_rows(Zs, Gs, list(Zt.index))
            Ztm = Zt.reindex(index=drugs, columns=cells).where(Zs.reindex(index=drugs, columns=cells).notna())
            D = selbench.model_axis_scores(Mm, Ztm, Gt, cells=cells, extra_axes={'source_axis': Gs_rows})
            tt, ts = T[T.evaluated_in == 'target'], T[T.evaluated_in == 'source']
            per_model.append({'target': target, 'model': m, 'class': rs.model_class(m), 'drugs': int(tt.drug.nunique()),
                              'raw_source': float(ts.raw.median()), 'raw_target': float(tt.raw.median()),
                              'sel_source': float(ts.selective.median()), 'sel_target': float(tt.selective.median()),
                              'sel_both_target': float(tt.selective_both.median()),
                              'adjusted_both_target': float(D.adjusted.median()) if len(D) else np.nan})
        print('demo', target, round(time.time() - t0), 's', flush=True)
    boot = pd.concat(boot, ignore_index=True)
    boot.to_csv(OUT / 'demo_bootstrap.csv', index=False)
    pm = pd.DataFrame(per_model)
    pm.to_csv(OUT / 'demo_models.csv', index=False)
    for target in ['GDSC1', 'GDSC2', 'CTRPv1']:
        x = pm[(pm.target == target) & pm.model.isin(published)]
        a = pm[(pm.target == target) & pm.model.isin(AGNOSTIC)]
        summary[f'demo_{target}'] = {
            'published_median_sel_target': float(x.sel_target.median()),
            'published_median_sel_both_target': float(x.sel_both_target.median()),
            'published_median_sel_source': float(x.sel_source.median()),
            'leakage_floor_target_axis': float(a.sel_target.max()),
            'leakage_floor_both_axes': float(a.sel_both_target.max()),
            'leakage_floor_model_axis': float(a.adjusted_both_target.max()),
            'drug_agnostic': a.set_index('model')[['sel_target', 'sel_both_target', 'adjusted_both_target']].to_dict('index')}
    summary['seconds'] = round(time.time() - t0, 1)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    pd.set_option('display.width', 220)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
