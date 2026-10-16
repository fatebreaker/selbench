"""Power of the reliability-adjusted attenuation test across a range of departures, and its calibration under noise
shared by all drugs of a cell line (2026-10-16; a reviewer asked for power beyond the two large departures of 100b and
for the shared cell-level noise that 100b omits).

The known-answer generator and test of 100b_known_answer_simulation.py are used unchanged (imported read-only; only its
second-screen function is wrapped). The target screen differs from the source by
  axis_loss@rho       an axis-specific loss: the target's axis correlates rho with the source's (100b: rho = 0.6)
  selective_loss@rho  a selective-specific loss: the selective component correlates rho (100b: rho = 0.6)
  shared_noise@f      proportional attenuation (k = 0.75, as 100b's null) with a fraction f of the target's noise
                      variance shared by every drug of a cell line (a plate- or batch-like cell effect); f = 0.12 and
                      0.34 span the shared replicate noise measured in DrEval's screens (Discussion)
rho in 0.95, 0.9, 0.8, 0.7. A test rejects when the raw loss is clear and the 95% interval of the reliability-adjusted
excess excludes 0 (adj_reject_81, as in the paper and 100b). 100 replicates per setting, 150 drugs x 500 cell lines.
Outputs: results/revision_round6/attenuation_power/{replicates.csv.gz, power.csv, summary.json}.
Usage (from scripts/): ../../.venv/bin/python 107_attenuation_power.py [--reps 100] [--workers 24]
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
sys.dont_write_bytecode = True
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('sim100b', HERE / '100b_known_answer_simulation.py')
m = importlib.util.module_from_spec(_spec)
sys.modules['sim100b'] = m
_spec.loader.exec_module(m)

OUT = m.lib.RES / 'revision_round6' / 'attenuation_power'
SEED = 20261016
RHOS = [0.95, 0.9, 0.8, 0.7]
SHARED = [0.12, 0.34]
SETTINGS = ([f'axis_loss@{r}' for r in RHOS] + [f'selective_loss@{r}' for r in RHOS] +
            [f'shared_noise@{f}' for f in SHARED])
_orig_second_screen = m.second_screen


def second_screen(sc, scen, rng, k_att=0.75, rho=0.6):
    """100b's second screen with the departure size as a parameter, and the shared cell-level noise null."""
    if '@' not in scen:
        return _orig_second_screen(sc, scen, rng, k_att=k_att, rho=rho)
    kind, val = scen.split('@')
    val = float(val)
    if kind in ('axis_loss', 'selective_loss'):
        return _orig_second_screen(sc, kind, rng, k_att=k_att, rho=val)
    assert kind == 'shared_noise', scen
    D, N = sc['y'].shape
    l_, u, s, a, t = sc['l'], sc['u'], sc['s'], sc['a'], sc['t']
    var_sig = l_ ** 2 * 1.0 + m.PAR['var_u'] + m.PAR['var_s']
    ve = (var_sig + m.PAR['var_e']) / k_att ** 2 - var_sig               # 100b's attenuation null: noise only
    eta = rng.standard_normal(N)                                          # one shared effect per cell line
    yt = (l_[:, None] * a + u[:, t] + s + rng.standard_normal((D, N)) * np.sqrt((1 - val) * ve)[:, None]
          + np.sqrt(val * ve)[:, None] * eta[None, :])
    return sc['y'], yt, sc


m.second_screen = second_screen          # attenuation_rep looks the function up in its module at call time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m.G['tissues'] = m.tissues()
    eqf = m.OUT / 'eq_sigma.json'
    if eqf.exists():
        m.G['eq_sigma'] = json.loads(eqf.read_text())
    tasks = [(s, 150, 500, SEED + 1000 * i + j) for i, s in enumerate(SETTINGS) for j in range(a.reps)]
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('fork')) as pool:
        for k, r in enumerate(pool.map(m.attenuation_rep, tasks, chunksize=1)):
            rows += r
            if (k + 1) % 100 == 0:
                print(f'{k + 1}/{len(tasks)} replicates, {time.time() - t0:.0f}s', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'replicates.csv.gz', index=False, float_format='%.6g')
    R['kind'] = R.scenario.str.split('@').str[0]
    R['param'] = R.scenario.str.split('@').str[1].astype(float)
    P = R.groupby(['kind', 'param']).agg(
        replicates=('seed', 'size'), loss_clear_rate=('loss_clear', 'mean'), reject_rate=('adj_reject_81', 'mean'),
        reject_positive_rate=('adj_reject_81_positive', 'mean'), reject_negative_rate=('adj_reject_81_negative', 'mean'),
        unadjusted_reject_rate=('reject_81', 'mean'), median_adj_excess=('adj_excess', 'median'),
        median_axis_retained=('adj_axis_retained', 'median'), median_selterm_retained=('adj_selterm_retained', 'median'),
        median_rho_target=('rho_target', 'median')).reset_index()
    P.to_csv(OUT / 'power.csv', index=False, float_format='%.4g')
    summ = {'settings': SETTINGS, 'replicates_per_setting': a.reps, 'panel': '150 drugs x 500 cell lines',
            'rejection_rule': 'clear raw loss and 95% interval of the reliability-adjusted excess excluding 0',
            'power': {f"{r.kind}@{r.param:g}": float(r.reject_rate) for r in P.itertuples()},
            'seconds': round(time.time() - t0)}
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1))
    print(P.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
