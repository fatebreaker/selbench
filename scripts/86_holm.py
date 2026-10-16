"""Round 2, task 8 (R3-M5, R2-M3; editor item 5): Holm correction over the 25 registered tests.

Family: the 23 registered tests with a P value in results/revision_transfer/holm_registered_tests.csv plus the two
Amendment 3 tests (organoids), which the amendment registered for inclusion in the Holm summary:
  H10 and H11 from the registered run (results/confirmation_PDM/hypotheses.json: H10 P = 1.0, H11 P = 0.95) or from
  the corrected run after the documented orientation correction (results/confirmation_PDM_corrected: H10 P = 1/10,001
  permutation floor, reported as P <= 1e-4; H11 P = 0.047).
Variants
  (a) registered P values, registered PDM run;          (b) registered P values, corrected PDM run;
  (c) unit-level: every listed unit-level or clustered P substituted, i.e. the two-way bootstrap tail for H3 and H5b
      (B = 1,000 two-way draws in 56_confirmation_estimation.py; a tail of 0 is set to 1/B = 0.001, i.e. P < 1/B),
      the patient-level P for BeatAML H6/H7 and the tumour-level P (T1, G1) for PDX H8/H9 (the 'Unit-level P' column
      of Supplementary Table S28), with the registered or the corrected PDM run;
  (d) as (c) but with the least favourable unit-level variant for H6-H9 (patient-block permutation; max over
      tumour-level G1/G2 tests: 56's 'robust_p_or_boot_tail');
  each also with NCI-60 excluded (its P values are declared not interpretable).
Outputs (results/revision_round2/holm/): holm_25.csv (per test, every variant), holm_summary.csv, run.json.
"""
import sys

sys.dont_write_bytecode = True
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.out_dir('holm')
B_TAIL = 1000            # 56_confirmation_estimation.B_CELL: two-way draws behind the H3/H5b tails
ALPHA = 0.05


def main():
    H = pd.read_csv(lib.RES / 'revision_transfer' / 'holm_registered_tests.csv')
    reg = json.loads((lib.RES / 'confirmation_PDM' / 'hypotheses.json').read_text())
    cor = json.loads((lib.RES / 'confirmation_PDM_corrected' / 'hypotheses.json').read_text())
    T = H[['screen', 'hypothesis', 'n', 'registered_p', 'p_for_holm', 'robust_p_or_boot_tail', 'unit_level_p']].copy()
    # the 'Unit-level P' column of Table S28: unit_level_p for H6-H9, else the two-way bootstrap tail (H3, H5b)
    T['listed_unit_p'] = T.unit_level_p.where(T.unit_level_p.notna(), T.robust_p_or_boot_tail)
    T['least_favourable_unit_p'] = T.robust_p_or_boot_tail
    pdm = []
    for h in ['H10', 'H11']:
        rp = float(reg[h]['permutation_p_one_sided'])
        cp = float(cor[h]['permutation_p_one_sided'])
        pdm.append({'screen': 'PDM (organoids)', 'hypothesis': h,
                    'n': int(cor[h].get('units', cor[h].get('units_reproducible', 0)) or 0),
                    'registered_p': rp, 'p_for_holm': rp, 'corrected_run_p': cp,
                    'robust_p_or_boot_tail': np.nan, 'unit_level_p': np.nan})
    P = pd.concat([T, pd.DataFrame(pdm)], ignore_index=True)
    P['corrected_run_p'] = P.corrected_run_p.where(P.corrected_run_p.notna(), P.p_for_holm)
    # unit-level P: tails of 0 -> 1/B (P < 1/B); PDM tests are already patient-level permutation tests
    with np.errstate(invalid='ignore'):
        lu = P.listed_unit_p.copy()
        P['listed_unit_p_used'] = np.where(lu.notna(), np.maximum(lu.fillna(0), 1.0 / B_TAIL), np.nan)
        lf = P.least_favourable_unit_p.copy()
        P['least_fav_unit_p_used'] = np.where(lf.notna(), np.maximum(lf.fillna(0), 1.0 / B_TAIL), np.nan)
    # H6-H9: the least favourable variant is never below the registered P (as in 56)
    m69 = P.hypothesis.isin(['H6', 'H7', 'H8', 'H9'])
    P.loc[m69, 'least_fav_unit_p_used'] = np.maximum(P.loc[m69, 'least_fav_unit_p_used'], P.loc[m69, 'p_for_holm'])
    variants = {}
    for run, col in [('registered PDM run', 'p_for_holm'), ('corrected PDM run', 'corrected_run_p')]:
        base = P[col].astype(float).values.copy()
        variants[f'(a/b) registered P, {run}'] = base
        u = base.copy()
        has = P.listed_unit_p_used.notna().values
        u[has] = P.listed_unit_p_used.values[has]
        variants[f'(c) listed unit-level P, {run}'] = u
        lfv = base.copy()
        has2 = P.least_fav_unit_p_used.notna().values
        lfv[has2] = P.least_fav_unit_p_used.values[has2]
        variants[f'(d) least favourable unit-level P, {run}'] = lfv
    rows = []
    out = P.copy()
    for name, p in variants.items():
        for excl in [False, True]:
            keep = (P.screen != 'NCI60').values if excl else np.ones(len(P), bool)
            adj = np.full(len(P), np.nan)
            adj[keep] = lib.holm(p[keep])
            tag = name + (' | NCI-60 excluded' if excl else '')
            out[f'p_used | {tag}'] = np.where(keep, p, np.nan)
            out[f'holm | {tag}'] = adj
            surv = (adj < ALPHA) & keep
            for fam_n, fam in [(25, np.ones(len(P), bool)), (23, ~P.screen.eq('PDM (organoids)').values)]:
                k = keep & fam
                adj2 = np.full(len(P), np.nan)
                adj2[k] = lib.holm(p[k])
                rows.append({'variant': name, 'nci60_excluded': excl, 'family': f'{fam_n} tests' if not excl else
                             f'{int(k.sum())} tests (NCI-60 excluded)', 'tests': int(k.sum()),
                             'survive_holm_0.05': int(((adj2 < ALPHA) & k).sum()),
                             'fail': '; '.join(f'{s} {h}' for s, h, a, kk in zip(P.screen, P.hypothesis, adj2, k)
                                               if kk and not a < ALPHA)})
    out.to_csv(OUT / 'holm_25.csv', index=False, float_format='%.6g')
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'holm_summary.csv', index=False)
    lib.write_json(OUT / 'run.json', {'B_tail': B_TAIL, 'alpha': ALPHA,
                                      'H10_H11_registered': {h: reg[h]['permutation_p_one_sided'] for h in ['H10', 'H11']},
                                      'H10_H11_corrected': {h: cor[h]['permutation_p_one_sided'] for h in ['H10', 'H11']},
                                      'H10_permutations': cor['H10'].get('n_permutations')})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 200)
    print(S.to_string(index=False))


if __name__ == '__main__':
    main()
