"""Round 3 (editor item 8; R1-m6): validate selbench/box1.py's rank intervals and expected retention against the
project's results, so that every step of Box 1 has a library implementation.

  l  rank_intervals       88_rank_intervals.py's raw and selective leaderboard rankings (22 models, 42's draws):
                          point rank and 95% rank interval of every model must equal rankings/rank_intervals.csv
  m  triad_reliability    84_source_reference.py's dataset-level reproducible shares over all drugs (DrEval CC
                          ln IC50 and AUC families; point estimates, row 0): source_reference/
                          triad_reliability_all_drugs.csv
     expected_retention   sqrt(R_T / R_S) against every point estimate in source_reference/expected_retention.csv
  tests                   pytest test_selbench.py test_box1.py
Outputs (results/revision_round2/box1_validation/): validation_r3b.csv, tests_r3b.txt, summary_r3b.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import itertools  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('lib77', HERE / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)
import box1  # noqa: E402  (selbench/ is on sys.path via lib77)

OUT = lib.out_dir('box1_validation')
ROWS = []


def check(component, name, expected, got, tol):
    ok = bool(np.isclose(expected, got, atol=tol, rtol=0) or (np.isnan(expected) and np.isnan(got)))
    ROWS.append({'component': component, 'check': name, 'expected': float(expected), 'box1': float(got),
                 'abs_diff': float(abs(expected - got)), 'tolerance': tol, 'pass': ok})


def ranks():
    m88 = lib.load_module('ri88', '88_rank_intervals.py')
    L = lib.leaderboard_setup()
    models = sorted(m for m in L['P'] if L['mclass'][m] in ('classical', 'neural/factorization'))
    arrs, drugs = m88.within_arrays(L, models)
    common = np.all(np.isfinite(arrs['raw'][0]) & np.isfinite(arrs['sel'][0]), axis=0)
    cd_idx = np.where(common)[0]
    Vd = L['V'][:, [L['drug_ix'][drugs[j]] for j in cd_idx]]
    ref = pd.read_csv(lib.OUT2 / 'rankings' / 'rank_intervals.csv')
    for name, s in [('raw', 'raw'), ('selective', 'sel')]:
        X = arrs[s][:, :, cd_idx].astype(float)
        S = np.column_stack([lib.wmedian_rows(X[:, i], Vd) for i in range(len(models))])
        T = box1.rank_intervals(S, models).set_index('model')
        r = ref[ref.metric == name].set_index('model')
        for m in models:
            for col in ['rank', 'rank_lo', 'rank_hi']:
                check('l rank_intervals', f'{name} {m} {col}', r.loc[m, col], T.loc[m, col], 1e-9)


def triads():
    m84 = lib.load_module('sr84', '84_source_reference.py')
    cd = lib.load_module('cd41', '41_crosslab_data.py')
    V = cd.versions()
    ref = pd.read_csv(lib.OUT2 / 'source_reference' / 'triad_reliability_all_drugs.csv')
    orig = m84.lib.draws                     # 84 holds its own instance of 77_round2_lib
    m84.lib.draws = lambda nc, nd, seed, n_boot=None: orig(nc, nd, seed, n_boot=0)    # point estimates only (row 0)
    try:
        for ep in ['lnIC50', 'AUC']:
            fam = f'DrEval CC {ep}'
            Rs = {ds: lib.selbench.decompose(V[f'{ds}|CC|{ep}'])[2] for ds in cd.DATASETS}
            corr, drugs, Vd, ce = m84.pair_corrs(Rs, m84.SEED)
            get = lambda d, a, b: corr.get((d, a, b)) or corr.get((d, b, a))  # noqa: E731
            for ds in Rs:
                R84, n84 = m84.triad_R(corr, drugs, Vd, ce, ds, drugs)
                tri = {'drug': [], 'r_fa': [], 'r_fb': [], 'r_ab': [], 'n': []}
                for d in drugs:
                    others = sorted({k[1] for k in corr if k[0] == d} | {k[2] for k in corr if k[0] == d})
                    for a, b in itertools.combinations([o for o in others if o != ds], 2):
                        fa, fb, ab = get(d, ds, a), get(d, ds, b), get(d, a, b)
                        if fa is None or fb is None or ab is None:
                            continue
                        tri['drug'].append(d); tri['n'].append(min(fa[1], fb[1], ab[1]))
                        tri['r_fa'].append(float(fa[0][0])); tri['r_fb'].append(float(fb[0][0]))
                        tri['r_ab'].append(float(ab[0][0]))
                if not tri['drug']:
                    continue
                Rb, nb = box1.triad_reliability(**tri)
                check('m triad_reliability', f'{fam} {ds} vs 84.triad_R', float(R84[0]), Rb, 1e-9)
                r = ref[(ref.family == fam) & (ref.dataset == ds)]
                if len(r):
                    check('m triad_reliability', f'{fam} {ds} vs triad_reliability_all_drugs.csv',
                          float(r.R_triad.iloc[0]), Rb, 5e-5)
                    check('m triad_reliability', f'{fam} {ds} drugs', float(r.drugs.iloc[0]), nb, 0)
    finally:
        m84.lib.draws = orig


def retention():
    E = pd.read_csv(lib.OUT2 / 'source_reference' / 'expected_retention.csv').drop_duplicates(['pair'])
    for _, r in E.iterrows():
        check('m expected_retention', f'{r.benchmark} {r.pair}', r.expected_retention,
              float(box1.expected_retention(r.R_source, r.R_target)), 5e-5)


def main():
    ranks()
    triads()
    retention()
    T = pd.DataFrame(ROWS)
    T.to_csv(OUT / 'validation_r3b.csv', index=False, float_format='%.6g')
    t = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'test_selbench.py', 'test_box1.py'],
                       cwd=lib.PROJECT / 'selbench', capture_output=True, text=True)
    (OUT / 'tests_r3b.txt').write_text(t.stdout[-2000:])
    summ = {'checks': int(len(T)), 'passed': int(T['pass'].sum()),
            'by_component': T.groupby('component')['pass'].agg(['size', 'sum']).astype(int).to_dict('index'),
            'max_abs_diff': T.groupby('component').abs_diff.max().to_dict(),
            'tests': t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.stderr[-300:]}
    (OUT / 'summary_r3b.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    if not T['pass'].all():
        print(T[~T['pass']].to_string())


if __name__ == '__main__':
    main()
