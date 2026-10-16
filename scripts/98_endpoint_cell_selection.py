"""Endpoint or cell selection? (round 3; Reviewers 1 and 2, concern 1)

CurveCurator ln IC50 is undefined for most resistant pairs (61% of GDSC2 pairs), so ln IC50 profiles are scored on
fewer, more sensitive cell lines than AUC profiles. To separate the endpoint from this cell selection, the out-of-fold
predictions of DrEval's four learned models trained on GDSC2 AUC are scored against GDSC2 AUC on all pairs and on the
pairs where ln IC50 is defined (each version decomposed on its own pairs, as the ln IC50 analysis is). The models
trained on GDSC2 ln IC50 are scored as in the paper. The perfect-predictor reference of ln IC50 is also computed with
undefined values tied at the resistant end (Box 1's censoring rule).

Pooled over the four models: axis share (sum axis term / sum raw), perfect-predictor share (mean r_zg^2), axis capture
and selective capture (box1.capture). Intervals: drug bootstrap (B = 1,000; cells fixed).
Output: results/revision_round2/endpoint_cells/{summary.csv, per_drug.csv.gz, run.json}.
Round 3 (editor item 1): --dataset CTRPv2 repeats the check for CTRP v2 once its AUC run exists: the four learned
models retrained on CTRP v2 CurveCurator AUC (drevalpy) scored on all AUC pairs and on the pairs where ln IC50 is
defined, against DrEval's released CTRP v2 ln IC50 predictions of the same four models (the 2 x 2's ln IC50 cell) and
the retrained EnsembleMF on ln IC50; outputs summary_CTRPv2.csv, per_drug_CTRPv2.csv.gz, run_CTRPv2.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402
import box1  # noqa: E402

spec = importlib.util.spec_from_file_location('sc59', HERE / '59_retrain_score.py')
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

OUT = PROJECT / 'results' / 'revision_round2' / 'endpoint_cells'
LEARNED = ['EnsembleMF', 'GradientBoosting', 'SimpleNeuralNetwork', 'RandomForest']
B, SEED, MIN_CELLS = 1000, 20261015, 30


def terms(P, Z, G):
    rows = []
    for d in Z.index.intersection(P.index):
        p = P.loc[d].reindex(Z.columns).values.astype(float)
        z, g = Z.loc[d].values.astype(float), G.loc[d].values.astype(float)
        m = np.isfinite(p) & np.isfinite(z) & np.isfinite(g)
        if m.sum() < MIN_CELLS or np.std(p[m]) < 1e-12:
            continue
        t = selbench.split_terms(p[m], z[m], g[m])
        rows.append({'drug': d, 'n': int(m.sum()), 'raw': t['raw'], 'r_pg': t['alignment'], 'r_zg': t['loading'],
                     'mediated': t['mediated'], 'selective_term': t['selective_term'], 'selective': t['selective']})
    return pd.DataFrame(rows)


def pooled(T):
    c = box1.capture(T.raw.values, T.r_pg.values, T.r_zg.values)
    return {'axis_share': float(T.mediated.sum() / T.raw.sum()), 'perfect_share': float((T.r_zg ** 2).mean()),
            'axis_capture': float(c['axis_capture']), 'selective_capture': float(c['selective_capture']),
            'median_cells': float(T.n.median()), 'drugs': int(T.drug.nunique())}


def boot(T, rng):
    drugs = T.drug.unique()
    idx = {d: np.where(T.drug.values == d)[0] for d in drugs}
    draws = []
    for _ in range(B):
        pick = rng.choice(drugs, len(drugs), replace=True)
        draws.append(pooled(T.iloc[np.concatenate([idx[d] for d in pick])]))
    D = pd.DataFrame(draws)
    return {f'{k}_{q}': float(np.percentile(D[k], p)) for k in ['axis_share', 'perfect_share', 'axis_capture',
                                                               'selective_capture'] for q, p in [('lo', 2.5), ('hi', 97.5)]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='GDSC2', choices=['GDSC2', 'CTRPv2'])
    ap.add_argument('--models', nargs='*', default=None, help='restrict the learned models (provisional runs)')
    ap.add_argument('--suffix', default=None)
    a = ap.parse_args()
    ds = a.dataset
    global LEARNED
    if a.models:
        LEARNED = [m for m in LEARNED if m in a.models]
    sfx = ('' if ds == 'GDSC2' else f'_{ds}') if a.suffix is None else a.suffix
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    resp = sc.responses()
    auc = resp[(resp.dataset == ds) & (resp.endpoint == 'auc')][['drug', 'cell', 'response']]
    ic = resp[(resp.dataset == ds) & (resp.endpoint == 'lnic50')][['drug', 'cell', 'response']]
    oof_auc, _, _ = sc.load_oof(f'{ds}_AUC')
    oof_ic, _, _ = sc.load_oof(f'{ds}_LNIC50')
    ic_models = LEARNED if ds == 'GDSC2' else [f'LB:{m}' for m in LEARNED]   # CTRP v2 ln IC50: DrEval's released
    out, per_drug = [], []
    versions = []
    cells_auc = sorted(set.intersection(*[set(oof_auc[m].cell) for m in LEARNED]))
    cells_ic = sorted(set.intersection(*[set(oof_ic[m].cell) for m in ic_models]))
    a_all = auc[auc.cell.isin(cells_auc)]
    defined = ic[['drug', 'cell']].drop_duplicates()
    a_def = a_all.merge(defined, on=['drug', 'cell'])
    versions.append(('AUC-trained, scored on AUC, all pairs', a_all, oof_auc, LEARNED))
    versions.append(('AUC-trained, scored on AUC, ln IC50-defined pairs only', a_def, oof_auc, LEARNED))
    versions.append(('ln IC50-trained, scored on ln IC50 (defined pairs)', ic[ic.cell.isin(cells_ic)], oof_ic, ic_models))
    if ds == 'CTRPv2':
        # cells common to both runs, so that the AUC and ln IC50 versions differ only in the scored pairs/endpoint
        common = sorted(set(cells_auc) & set(cells_ic))
        versions.append(('AUC-trained, scored on AUC, all pairs, cells of both runs', a_all[a_all.cell.isin(common)],
                         oof_auc, LEARNED))
        versions.append(('AUC-trained, scored on AUC, ln IC50-defined pairs, cells of both runs',
                         a_def[a_def.cell.isin(common)], oof_auc, LEARNED))
        versions.append(('ln IC50-trained, scored on ln IC50, cells of both runs', ic[ic.cell.isin(common)], oof_ic,
                         ic_models))
        if 'EnsembleMF' in oof_ic:
            versions.append(('EnsembleMF retrained on ln IC50, scored on ln IC50', ic[ic.cell.isin(cells_ic)], oof_ic,
                             ['EnsembleMF']))
            versions.append(('EnsembleMF trained on AUC, scored on AUC, all pairs', a_all, oof_auc, ['EnsembleMF']))
    for label, obs, oof, mods in versions:
        Z, G, _, _ = selbench.decompose(obs)
        T = pd.concat([terms(oof[m].pivot_table(index='drug', columns='cell', values='prediction'), Z, G).assign(model=m)
                       for m in mods], ignore_index=True)
        per_drug.append(T.assign(version=label))
        out.append({'version': label, 'pairs': int(len(obs)), **pooled(T), **boot(T, rng)})
    # ln IC50 with undefined pairs tied at the resistant end (perfect-predictor reference only: models trained on
    # ln IC50 predict defined pairs only)
    tied = a_all[['drug', 'cell']].merge(ic, on=['drug', 'cell'], how='left')
    lowest = tied.groupby('drug').response.transform('min')
    tied['response'] = tied.response.fillna(lowest - 1.0)          # below every defined value: most resistant, tied
    Zt, Gt, _, r2 = selbench.decompose(tied)
    out.append({'version': 'ln IC50 with undefined pairs tied at the resistant end (reference only)',
                'pairs': int(len(tied)), 'perfect_share': float(np.nanmean(list(r2.values())) if isinstance(r2, dict)
                                                                 else np.nanmean(r2))})
    S = pd.DataFrame(out)
    S.to_csv(OUT / f'summary{sfx}.csv', index=False)
    pd.concat(per_drug, ignore_index=True).to_csv(OUT / f'per_drug{sfx}.csv.gz', index=False, float_format='%.6g')
    (OUT / f'run{sfx}.json').write_text(json.dumps({'B': B, 'seed': SEED, 'models': LEARNED, 'ic_models': ic_models,
                                                    'dataset': ds, 'cells_auc': len(cells_auc),
                                                    'cells_ic': len(cells_ic)}, indent=1))
    pd.set_option('display.width', 250)
    print(S.round(3).to_string())


if __name__ == '__main__':
    main()
