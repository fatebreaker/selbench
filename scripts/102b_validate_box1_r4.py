"""Round 4 (editor item 8; R1-M3, R3-M3c, R1-m13): validate selbench/box1.py's axis-leakage functions and the
general-sensitivity aliases against the project's results, and run every selbench unit test.

  n  axis_leakage_bound, corrected_selective_capture
       (1) arithmetic: reproduce 100b's predicted leakage (with the split-half and with the latent reliability) and
           its corrected S in results/revision_round4/simulation/fig_5_axis_reliability_leakage.csv (40 rows) and the
           predicted leakage and corrected S of every model in realdata_leakage_correction.csv (95 rows);
       (2) documented properties, on the simulation (latent reliability, 10 panel sizes x 4 predictors):
           exact for the drug-agnostic predictor (|leakage - formula| <= 0.05 x formula + 0.002, i.e. within Monte
           Carlo error); an upper bound for the predictors with drug-specific signal (balanced, axis-heavy); negative
           leakage and a zero formula for the selective-only predictor; the full two-term leakage
           A m (1-rho)/[(1-m) rho] - S_lat m_lat (1-rho)/(1-m) reproduces the observed leakage (|diff| <= 0.002);
           the correction with the split-half reliability reduces the error against the latent S in every row;
       (3) informational: the formula with the split-half reliability against the drug-agnostic predictor's leakage
           (it understates it with few drugs, where the split-half estimate overstates the latent reliability);
           drug-agnostic predictors in the leave-cell-line-out benchmarks (observed S against the formula);
       (4) pair splits: IMPROVE's drug-agnostic cell-mean floor (observed S against the formula; expected to fail,
           as documented) and the ValueError for split='pair' and 'drug'.
  o  aliases: box1.axis / general_sensitivity_axis / axis_baseline / oracle_axis_share are selbench's functions.
  tests  pytest test_selbench.py test_box1.py
Outputs (results/revision_round2/box1_validation/): validation_r4.csv, tests_r4.txt, summary_r4.json (new files; the
earlier validation files are not touched).
"""
import sys

sys.dont_write_bytecode = True
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)
import box1  # noqa: E402  (selbench/ is on sys.path via lib77)
import selbench  # noqa: E402

OUT = lib.RES / 'revision_round2' / 'box1_validation'
SIM = lib.RES / 'revision_round4' / 'simulation'
SIG = 5              # the CSVs carry 5 significant digits (float_format '%.5g')


def half_ulp(x):
    x = abs(float(x))
    return 0.0 if x == 0 or not np.isfinite(x) else 0.5 * 10 ** (np.floor(np.log10(x)) - (SIG - 1))


def rounding_tol(fn, inputs, expected):
    """Largest change of fn over every +/- half-unit (5th significant digit) perturbation of its inputs, plus the
    rounding of the expected value: the discrepancy that CSV rounding alone can produce."""
    import itertools
    f0 = fn(*inputs)
    dev = 0.0
    for signs in itertools.product([-1, 1], repeat=len(inputs)):
        x = [v + s_ * half_ulp(v) for v, s_ in zip(inputs, signs)]
        dev = max(dev, abs(fn(*x) - f0))
    return dev + half_ulp(expected) + 1e-12


def main():
    for f in ['validation_r4.csv', 'tests_r4.txt', 'summary_r4.json']:
        assert not (OUT / f).exists() or '--overwrite' in sys.argv, f'{f} exists'
    rows = []

    def add(check, item, src, key, expected, value, status, tol='', note=''):
        rows.append({'component': 'n' if check != 'aliases' else 'o', 'check': check, 'item': item, 'source': src,
                     'key': key, 'expected': expected, 'box1': value,
                     'abs_diff': (abs(value - expected) if isinstance(value, float) and isinstance(expected, float) else ''),
                     'tolerance': tol, 'status': status, 'note': note})

    # (1)-(3) simulation
    F = pd.read_csv(SIM / 'fig_5_axis_reliability_leakage.csv')
    src = 'revision_round4/simulation/fig_5_axis_reliability_leakage.csv'
    for r in F.itertuples():
        key = f'D={r.D}|{r.predictor}'
        b_sh = box1.axis_leakage_bound(r.A_design, r.perfect_share_design, r.axis_split_half)
        b_lat = box1.axis_leakage_bound(r.A_design, r.perfect_share_design, r.axis_latent_reliability)
        c_sh = box1.corrected_selective_capture(r.S_design, r.A_design, r.perfect_share_design, r.axis_split_half)
        for item, exp, val, fn, inp in [
                ('formula, split-half rho = predicted_leakage_split_half', r.predicted_leakage_split_half, b_sh,
                 box1.axis_leakage_bound, (r.A_design, r.perfect_share_design, r.axis_split_half)),
                ('formula, latent rho = predicted_leakage_latent_rel', r.predicted_leakage_latent_rel, b_lat,
                 box1.axis_leakage_bound, (r.A_design, r.perfect_share_design, r.axis_latent_reliability)),
                ('corrected S, split-half rho = S_corrected_split_half', r.S_corrected_split_half, c_sh,
                 box1.corrected_selective_capture, (r.S_design, r.A_design, r.perfect_share_design, r.axis_split_half))]:
            tol = rounding_tol(fn, inp, exp)
            add('arithmetic', item, src, key, float(exp), float(val), 'pass' if abs(val - exp) <= tol else 'FAIL',
                f'{tol:.2g} (input rounding)')
        leak = float(r.leakage)
        if r.predictor == 'agnostic':
            tol = 0.05 * abs(b_lat) + 0.002
            add('property: exact for drug-agnostic', 'observed leakage = formula (latent rho)', src, key, leak, float(b_lat),
                'pass' if abs(leak - b_lat) <= tol else 'FAIL', f'{tol:.4f}', f'S_latent {r.S_latent:.4f}')
            add('info: split-half rho', 'observed leakage vs formula (split-half rho)', src, key, leak, float(b_sh), 'info',
                note=f'split-half {r.axis_split_half:.3f} vs latent {r.axis_latent_reliability:.3f}; formula/observed '
                     f'{b_sh / leak:.3f}' + (' (understates: not a bound)' if b_sh < leak else ''))
        elif r.predictor in ('balanced', 'axis_heavy'):
            add('property: upper bound otherwise', 'observed leakage <= formula (latent rho)', src, key, leak, float(b_lat),
                'pass' if leak <= b_lat else 'FAIL', note=f'S_latent {r.S_latent:.4f}; formula - observed {b_lat - leak:.4f}')
            add('info: split-half rho', 'observed leakage vs formula (split-half rho)', src, key, leak, float(b_sh), 'info',
                note='bound holds' if leak <= b_sh else 'understates (split-half rho too high)')
        else:
            add('property: selective-only', 'observed leakage < 0 and formula ~ 0', src, key, leak, float(b_lat),
                'pass' if (leak < 0 and abs(b_lat) < 1e-3) else 'FAIL')
        m_lat = float(r.perfect_share_latent)
        two = b_lat - r.S_latent * m_lat * (1 - r.axis_latent_reliability) / (1 - r.perfect_share_design)
        add('property: two-term leakage', 'observed leakage = A m(1-rho)/[(1-m)rho] - S_lat m_lat (1-rho)/(1-m)', src, key,
            leak, float(two), 'pass' if abs(two - leak) <= 0.002 else 'FAIL', '0.002')
        e_unc, e_cor = abs(r.S_design - r.S_latent), abs(c_sh - r.S_latent)
        add('property: correction reduces error', '|corrected S - S_latent| < |S - S_latent| (split-half rho)', src, key,
            float(e_unc), float(e_cor), 'pass' if e_cor < e_unc else 'FAIL', note=f'corrected error {c_sh - r.S_latent:+.4f}')
    # (1), (3), (4) real data
    Rd = pd.read_csv(SIM / 'realdata_leakage_correction.csv')
    src = 'revision_round4/simulation/realdata_leakage_correction.csv'
    for r in Rd.itertuples():
        key = f'{r.benchmark}|{r.setting}|{r.model}'
        b = box1.axis_leakage_bound(r.A, r.perfect_share, r.axis_split_half)          # arithmetic (split='cell')
        c = box1.corrected_selective_capture(r.S, r.A, r.perfect_share, r.axis_split_half)
        note = 'pair-split predictions: arithmetic only (box1 refuses split="pair")' if r.benchmark == 'improve' else ''
        t1 = rounding_tol(box1.axis_leakage_bound, (r.A, r.perfect_share, r.axis_split_half), r.predicted_leakage)
        t2 = rounding_tol(box1.corrected_selective_capture, (r.S, r.A, r.perfect_share, r.axis_split_half), r.S_corrected)
        add('arithmetic', 'formula = predicted_leakage', src, key, float(r.predicted_leakage), float(b),
            'pass' if abs(b - r.predicted_leakage) <= t1 else 'FAIL', f'{t1:.2g} (input rounding)', note)
        add('arithmetic', 'corrected S = S_corrected', src, key, float(r.S_corrected), float(c),
            'pass' if abs(c - r.S_corrected) <= t2 else 'FAIL', f'{t2:.2g} (input rounding)', note)
        if isinstance(r.drug_agnostic, str) and r.benchmark != 'improve':
            add('info: leave-cell-line-out drug-agnostic', 'observed S vs formula', src, key, float(r.S), float(b), 'info',
                note=f'drug-agnostic ({r.drug_agnostic}); observed - formula {r.S - b:+.4f}')
        if r.benchmark == 'improve' and r.model == 'floor_cell_mean':
            add('pair split: documented failure', 'IMPROVE cell-mean floor (drug-agnostic): observed S vs formula', src,
                key, float(r.S), float(b), 'documented failure (formula over-predicts)' if b > r.S else 'not reproduced',
                note=f'axis split-half {r.axis_split_half:.3f}; formula {b:.4f} vs observed {r.S:.4f} '
                     f'(ratio {b / r.S:.1f})')
    for split in ['pair', 'drug']:
        for fn, args in [(box1.axis_leakage_bound, (1.0, 0.2, 0.9)), (box1.corrected_selective_capture, (0.01, 1.0, 0.2, 0.9))]:
            try:
                fn(*args, split=split)
                ok = False
            except ValueError:
                ok = True
            add('pair split: refused', f'{fn.__name__}(split={split!r}) raises ValueError', 'box1', split, '', '',
                'pass' if ok else 'FAIL')
    # (o) aliases
    for a, s in [('axis', 'fitness_axis'), ('general_sensitivity_axis', 'fitness_axis'), ('axis_baseline', 'fitness_baseline'),
                 ('oracle_axis_share', 'oracle_fitness_share')]:
        add('aliases', f'box1.{a} is selbench.{s}', 'box1/selbench', a, '', '',
            'pass' if getattr(box1, a) is getattr(selbench, s) else 'FAIL')
    # the leaderboard axis via the alias equals 77's decomposition axis
    L = lib.leaderboard_setup()
    Z, Gm = L['dec'][L['cd'].REFERENCE]
    Ga = box1.axis(Z)
    d = float(np.nanmax(np.abs(Ga.reindex(index=Gm.index, columns=Gm.columns).values - Gm.values)))
    add('aliases', 'box1.axis(Z) = leaderboard decomposition axis (CTRP v2 CurveCurator ln IC50)', '77.leaderboard_setup',
        'leaderboard', 0.0, d, 'pass' if d < 1e-10 else 'FAIL', '1e-10')
    V = pd.DataFrame(rows)
    V.to_csv(OUT / 'validation_r4.csv', index=False, float_format='%.6g')
    t = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_selbench.py', 'test_box1.py'],
                       cwd=lib.PROJECT / 'selbench', capture_output=True, text=True)
    (OUT / 'tests_r4.txt').write_text(t.stdout + t.stderr)
    counts = V.groupby(['check', 'status']).size().reset_index(name='n')
    summ = {'checks': int(len(V)), 'pass': int((V.status == 'pass').sum()), 'fail': int((V.status == 'FAIL').sum()),
            'info': int((V.status == 'info').sum()),
            'documented_failure': int(V.status.str.startswith('documented failure').sum()),
            'not_reproduced': int((V.status == 'not reproduced').sum()),
            'by_check': {f'{r.check} | {r.status}': int(r.n) for r in counts.itertuples()},
            'tests_returncode': t.returncode, 'tests_tail': (t.stdout.strip().splitlines() or [''])[-1]}
    lib.write_json(OUT / 'summary_r4.json', summ)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 120)
    print(counts.to_string(index=False))
    print(V[V.status.isin(['FAIL', 'not reproduced']) | V.status.str.startswith('documented') | V.check.str.startswith('info')][
        ['check', 'key', 'expected', 'box1', 'status', 'note']].to_string(index=False))
    print(summ)


if __name__ == '__main__':
    main()
