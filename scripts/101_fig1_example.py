"""Worked example for the headline figure (Fig. 1a): one drug of DrEval's leaderboard (CTRP v2 CurveCurator ln IC50,
leave-cell-line-out), on the cells every model predicts (42's 739 cells, ln IC50 defined).

Onalespib (HSP90 inhibitor; PubChem 11955716) was chosen because EnsembleMF and the drug-agnostic omics-predicted axis
reach almost the same per-drug correlation on it with different compositions (per_drug_scores.csv.gz, CTRP v2 self
pair). For each predictor the per-drug correlation is split exactly into the axis term r_pg r_zg and the selective
term (selbench.split_terms); the perfect predictor (the measured profile) carries r_zg^2 and 1 - r_zg^2.
Output: results/revision_round4/fig1_example/{cells.csv, terms.csv}.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

DRUG, NAME = '11955716', 'onalespib'
MODELS = {'EnsembleMF': 'EnsembleMF (best released model)', 'fitness_only': 'omics-predicted axis (drug-agnostic)',
          'NaiveTissueDrugMeanPredictor': 'tissue-by-drug mean'}
OUT = lib.RES / 'revision_round4' / 'fig1_example'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = lib.leaderboard_setup()
    Z, G = L['dec'][L['cd'].REFERENCE]
    U = L['U']
    z, g = Z.loc[DRUG].reindex(U).values.astype(float), G.loc[DRUG].reindex(U).values.astype(float)
    m = np.isfinite(z) & np.isfinite(g)
    for k in L['lbm']:
        if DRUG in L['P'][k].index:
            m &= np.isfinite(L['P'][k].loc[DRUG].values.astype(float))
    cells = pd.DataFrame({'cell': U[m], 'z': z[m], 'g': g[m]})
    rows = []
    t = lib.selbench.split_terms(z[m], z[m], g[m])
    rows.append({'predictor': 'perfect predictor (measured profile)', 'raw': 1.0, 'axis_term': t['loading'] ** 2,
                 'selective_term': 1 - t['loading'] ** 2, 'r_pg': t['loading'], 'r_zg': t['loading']})
    for k, lab in MODELS.items():
        p = L['P'][k].loc[DRUG].values.astype(float)[m]
        cells[k] = p
        t = lib.selbench.split_terms(p, z[m], g[m])
        rows.append({'predictor': lab, 'raw': t['raw'], 'axis_term': t['mediated'], 'selective_term': t['selective_term'],
                     'r_pg': t['alignment'], 'r_zg': t['loading'], 'selective_r': t['selective']})
    T = pd.DataFrame(rows).assign(drug=DRUG, name=NAME, cells=int(m.sum()))
    cells.to_csv(OUT / 'cells.csv', index=False, float_format='%.6g')
    T.to_csv(OUT / 'terms.csv', index=False, float_format='%.6g')
    print(T.round(3).to_string())


if __name__ == '__main__':
    main()
