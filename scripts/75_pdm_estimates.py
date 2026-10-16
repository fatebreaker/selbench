"""Amendment 3 estimates for the paper (descriptive; run after the registered tests; uses the frozen 72_pdm_confirm.py
read-only).

For the registered run (orientation as registered, misoriented for fit_auc datasets) and the corrected run (deviation 4),
per discovery class (reproducible, non-reproducible) and per dataset (all, liver, sarcoma):
  * the registered estimand: precision-weighted pooled Fisher z (weights n - 4) with a patient-bootstrap 95% interval
    (patients resampled within datasets, 2,000 draws);
  * random effects over drugs (DerSimonian-Laird, Hartung-Knapp interval), as for BeatAML and PDX_Bruna.
Outputs: results/revision_transfer/{pdm_meta_analysis.csv, pdm_tests.csv}; results/confirmation_PDM_corrected/scores.csv
(per-unit selective transfer in the format of the BeatAML/PDX score tables).
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROCESSED, RESULTS

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pdm_confirm_ro', HERE / '72_pdm_confirm.py')
C = importlib.util.module_from_spec(spec); sys.modules['pdm_confirm_ro'] = C; spec.loader.exec_module(C)
OUT = RESULTS / 'revision_transfer'
RUNS = {'registered (orientation as registered)': (PROCESSED / 'pdm', RESULTS / 'confirmation_PDM'),
        'corrected orientation (deviation 4)': (PROCESSED / 'pdm_corrected', RESULTS / 'confirmation_PDM_corrected')}


def datasets(base):
    cls = C.classes()
    out = []
    for ds in C.DATASETS:
        path = base / ds / 'patient_long.csv.gz'
        if not path.exists():
            continue
        built = C.build(path)
        if built is None or built[0].shape[0] < C.MIN_DRUGS_PER_DATASET:
            continue
        preds = {f: C.patient_predictions(ds, f) for f in ['sel', 'raw']}
        D = C.Dataset(ds, *built, preds, cls)
        if D.units:
            out.append(D)
    return out


def main():
    rows, tests = [], []
    for name, (base, res_dir) in RUNS.items():
        sets = datasets(base)
        rng = np.random.default_rng(C.SEED + 1)
        counts = [np.stack([np.bincount(rng.integers(0, len(D.cols), len(D.cols)), minlength=len(D.cols))
                            for _ in range(C.N_BOOT)]) for D in sets]
        boot = [D.boot_terms(c, 'sel') for D, c in zip(sets, counts)]
        units = pd.read_csv(res_dir / 'units.csv')
        u = units[units.family == 'sel']
        for scope in ['all'] + [D.ds for D in sets]:
            for cl in ['reproducible', 'non_reproducible']:
                keep = [(D, b) for D, b in zip(sets, boot) if scope in ('all', D.ds)]
                R = np.hstack([b['selective'] for _, b in keep]); N = np.hstack([b['n'] for _, b in keep])
                mask = np.array([x['class'] == cl for D, _ in keep for x in D.units])
                obs = u[(u['class'] == cl) & ((u.dataset == scope) if scope != 'all' else True)]
                if mask.sum() == 0 or obs.empty:
                    continue
                est = C.pooled_z(obs.selective.values, obs.n.values)
                tb = C.wpooled_rows(R, N, mask)
                z = C.fz(obs.selective.values); se = 1 / np.sqrt(obs.n.values - 4)
                re = C.random_effects(z, se)
                rows.append({'analysis': name, 'scope': scope, 'class': cl, 'units': int(len(obs)),
                             'compounds': int(obs.compound.nunique()), 'pooled_r': float(np.tanh(est)),
                             'ci95_lo': float(np.tanh(np.nanpercentile(tb, 2.5))),
                             'ci95_hi': float(np.tanh(np.nanpercentile(tb, 97.5))),
                             're_pooled_r': re.get('pooled_r'), 're_ci95_hksj': re.get('ci95_hksj'),
                             're_tau2': re.get('tau2')})
        h = json.loads((res_dir / 'hypotheses.json').read_text())
        for k in ['H10', 'H11']:
            tests.append({'analysis': name, 'hypothesis': k, 'permutation_p': h[k].get('permutation_p_one_sided'),
                          'pass': h[k].get('pass'), 'secondary_p': h[k].get('secondary_wilcoxon_p_one_sided',
                                                                            h[k].get('secondary_mwu_p_one_sided'))})
    pd.DataFrame(rows).to_csv(OUT / 'pdm_meta_analysis.csv', index=False)
    pd.DataFrame(tests).to_csv(OUT / 'pdm_tests.csv', index=False)
    u = pd.read_csv(RESULTS / 'confirmation_PDM_corrected' / 'units.csv')
    u.rename(columns={'selective': 'sel'}).to_csv(RESULTS / 'confirmation_PDM_corrected' / 'scores.csv', index=False)
    print(pd.DataFrame(rows).drop(columns=['re_tau2']).round(3).to_string())
    print(pd.DataFrame(tests).to_string())


if __name__ == '__main__':
    main()
