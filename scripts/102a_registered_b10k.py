"""Round 4 (editor item 5; R3-M4, R2-M5): the near-threshold bootstrap-tail P values of the registered layer with
B = 10,000, and the Holm families recomputed with them.

The five near-threshold P values (R3-M4: "gCSI H3 0.010, gCSI H5b 0.010, CCLE H5b 0.015, FIMM H3 0.024 and GRAY H5b
0.006"; Monte Carlo SE about 0.003 at B = 1,000) are the two-way (cell x drug) bootstrap tails that
56_confirmation_estimation.py computes for H3 (P(median reproducible - median non-reproducible <= 0)) and H5b
(P(median paired measurement - model difference <= 0)), with B_CELL = 1,000 two-way draws, and that 86_holm.py enters
as the 'listed' and 'least favourable' unit-level P of H3 and H5b (a tail of 0 entered as 1/B).

Part 'tails'. 56.analyse_screen (imported read-only; the same code path) is re-run for gCSI, CCLE, FIMM and GRAY with
  (i)  the original seed and B_CELL = 1,000: a regression check that must reproduce the stored tails exactly;
  (ii) seed offset +1000 (SEED = 20262017) and B_CELL = 10,000: the new P values (primary);
  (iii) seed offset +2000 and B_CELL = 10,000: an independent replicate, to show the Monte Carlo agreement.
  Everything else in analyse_screen is unchanged except B_H1 (the H1 bootstrap, computed after H3/H5b from the same
  generator, so it cannot affect them; set to 20 to save time, and its output is not used). NCI-60 (tails 0 of 1,000,
  entered by 86 at the 1/B floor; its P values are declared not interpretable) is re-run the same way (--with-nci60;
  its H4 stage holds B_CELL x 39,682 draws, about 8 GB): at the floor, its Holm decision in the 25-test families
  depends on B (Monte Carlo margin below), so it is included.
Part 'holm'. 86_holm.py's families are rebuilt (the same rules, re-implemented here because 86 reads fixed files;
  checked to reproduce results/revision_round2/holm/holm_summary.csv exactly with the B = 1,000 tails) with the B =
  10,000 tails substituted (tail of 0 -> 1/10,000): registered P, listed unit-level P and least favourable unit-level P;
  registered and corrected organoid (PDM) runs; 25 tests and 21 tests (NCI-60 excluded) (and 23/19 without PDM).
  Also a Monte Carlo margin for every other resampling-based P in the families (binomial SE at its own number of draws:
  would its Holm decision change at +/- 3 SE?).
Outputs: results/revision_round4/holm_b10k/ (tails_b10k.csv, regression_check.csv, holm_25_b10k.csv,
  holm_summary_b10k.csv, decision_changes.csv, h6_h8_h10.csv, mc_margin.csv, run.json). results/revision_round2/holm/
  is read, never written.
Usage: 102a_registered_b10k.py --part tails holm
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location('lib77', HERE / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.RES / 'revision_round4' / 'holm_b10k'
SCREENS = ['gCSI', 'CCLE', 'FIMM', 'GRAY']
SCREENS_ALL = SCREENS + ['NCI60']
NEAR = [('gCSI', 'H3'), ('gCSI', 'H5b'), ('CCLE', 'H5b'), ('FIMM', 'H3'), ('GRAY', 'H5b')]
B_NEW = 10000
B_OLD = 1000
OFFSETS = {'primary': 1000, 'replicate': 2000}
ALPHA = 0.05
WORKERS = 3
M56 = None


def m56():
    global M56
    if M56 is None:
        spec = importlib.util.spec_from_file_location('ce56', HERE / '56_confirmation_estimation.py')
        M56 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M56)
    return M56


def tail_task(args):
    """One screen under (seed, B): 56.analyse_screen with module constants set; returns H3/H5b tails and intervals."""
    screen, seed, B, label = args
    m = m56()
    m.SEED, m.B_CELL, m.B_H1 = seed, B, 20
    t0 = time.time()
    res = m.analyse_screen(screen)
    est = pd.DataFrame(res['rows'])
    out = []
    h3 = est[(est.hypothesis == 'H3') & est.quantity.str.startswith('difference')]
    if len(h3):
        out.append({'screen': screen, 'hypothesis': 'H3', 'label': label, 'seed': seed, 'B': B,
                    'estimate': float(h3.estimate.iloc[0]), 'ci95_two_way': json.dumps([round(x, 4) for x in h3.ci95_two_way_boot.iloc[0]]),
                    'p_tail': float(h3.p_boot_diff_le_0_two_way.iloc[0])})
    h5 = est[(est.hypothesis == 'H5b') & est.quantity.str.contains('difference')]
    if len(h5):
        out.append({'screen': screen, 'hypothesis': 'H5b', 'label': label, 'seed': seed, 'B': B,
                    'estimate': float(h5.estimate.iloc[0]), 'ci95_two_way': json.dumps([round(x, 4) for x in h5.ci95_two_way_boot.iloc[0]]),
                    'p_tail': float(res['h5_p_boot_diff_le_0_two_way'])})
    for r in out:
        r['seconds'] = round(time.time() - t0)
    print(screen, label, [(r['hypothesis'], r['p_tail']) for r in out], f'{time.time() - t0:.0f}s', flush=True)
    return out


def part_tails(screens):
    base = m56().SEED
    tasks = [(s, base, B_OLD, 'regression (original seed, B = 1,000)') for s in screens]
    tasks += [(s, base + off, B_NEW, f'{lab} (seed + {off}, B = 10,000)') for lab, off in OFFSETS.items() for s in screens]
    rows = []
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context('fork')) as ex:
        for r in ex.map(tail_task, tasks):
            rows += r
    T = pd.DataFrame(rows)
    stored = pd.read_csv(lib.RES / 'revision_transfer' / 'holm_registered_tests.csv')
    stored = stored.set_index(['screen', 'hypothesis']).robust_p_or_boot_tail
    est = pd.read_csv(lib.RES / 'revision_transfer' / 'confirmation_estimates.csv')
    reg = T[T.label.str.startswith('regression')].copy()
    reg['stored_p_tail'] = [float(stored.get((s, h), np.nan)) for s, h in zip(reg.screen, reg.hypothesis)]
    reg['stored_ci95_two_way'] = [est[(est.screen == s) & (est.hypothesis == h) &
                                      (est.quantity.str.startswith('difference') if h == 'H3' else est.quantity.str.contains('difference'))]
                                  .ci95_two_way_boot.iloc[0] for s, h in zip(reg.screen, reg.hypothesis)]
    reg['registered_test'] = reg.stored_p_tail.notna()        # CCLE H3 is not a registered test (no P in 56's table)
    reg['identical'] = ((reg.p_tail - reg.stored_p_tail).abs() < 1e-12) | ~reg.registered_test
    reg['ci_identical'] = [json.loads(a) == json.loads(b) for a, b in zip(reg.ci95_two_way, reg.stored_ci95_two_way)]
    reg.to_csv(OUT / 'regression_check.csv', index=False)
    new = T[~T.label.str.startswith('regression')].copy()
    new['near_threshold'] = [(s, h) in NEAR for s, h in zip(new.screen, new.hypothesis)]
    new['p_B1000_stored'] = [float(stored.get((s, h), np.nan)) for s, h in zip(new.screen, new.hypothesis)]
    new['p_used'] = np.maximum(new.p_tail, 1.0 / new.B)       # 86's rule: a tail of 0 enters as 1/B
    new['mc_se'] = np.sqrt(new.p_used * (1 - new.p_used) / new.B)
    new.to_csv(OUT / 'tails_b10k.csv', index=False)
    print(reg.to_string(index=False))
    print(new.to_string(index=False))
    return reg, new


# ----------------------------------------------------------------------------------------------------------------
def table(tails=None):
    """86's per-test table; `tails` {(screen, hyp): P} replaces the bootstrap tails of H3/H5b (and B_tail)."""
    H = pd.read_csv(lib.RES / 'revision_transfer' / 'holm_registered_tests.csv')
    reg = json.loads((lib.RES / 'confirmation_PDM' / 'hypotheses.json').read_text())
    cor = json.loads((lib.RES / 'confirmation_PDM_corrected' / 'hypotheses.json').read_text())
    T = H[['screen', 'hypothesis', 'n', 'registered_p', 'p_for_holm', 'robust_p_or_boot_tail', 'unit_level_p']].copy()
    T['B_tail'] = np.where(T.hypothesis.isin(['H3', 'H5b']), B_OLD, np.nan)
    if tails:
        for (s, h), (p, B) in tails.items():
            k = (T.screen == s) & (T.hypothesis == h)
            assert k.sum() == 1, (s, h)
            T.loc[k, 'robust_p_or_boot_tail'] = p
            T.loc[k, 'B_tail'] = B
    T['listed_unit_p'] = T.unit_level_p.where(T.unit_level_p.notna(), T.robust_p_or_boot_tail)
    T['least_favourable_unit_p'] = T.robust_p_or_boot_tail
    pdm = []
    for h in ['H10', 'H11']:
        pdm.append({'screen': 'PDM (organoids)', 'hypothesis': h,
                    'n': int(cor[h].get('units', cor[h].get('units_reproducible', 0)) or 0),
                    'registered_p': float(reg[h]['permutation_p_one_sided']),
                    'p_for_holm': float(reg[h]['permutation_p_one_sided']),
                    'corrected_run_p': float(cor[h]['permutation_p_one_sided']),
                    'robust_p_or_boot_tail': np.nan, 'unit_level_p': np.nan, 'B_tail': np.nan})
    P = pd.concat([T, pd.DataFrame(pdm)], ignore_index=True)
    P['corrected_run_p'] = P.corrected_run_p.where(P.corrected_run_p.notna(), P.p_for_holm)
    floor = 1.0 / P.B_tail.where(P.B_tail.notna(), B_OLD)
    with np.errstate(invalid='ignore'):
        lu = P.listed_unit_p
        P['listed_unit_p_used'] = np.where(lu.notna(), np.maximum(lu.fillna(0), floor), np.nan)
        lf = P.least_favourable_unit_p
        P['least_fav_unit_p_used'] = np.where(lf.notna(), np.maximum(lf.fillna(0), floor), np.nan)
    m69 = P.hypothesis.isin(['H6', 'H7', 'H8', 'H9'])
    P.loc[m69, 'least_fav_unit_p_used'] = np.maximum(P.loc[m69, 'least_fav_unit_p_used'], P.loc[m69, 'p_for_holm'])
    return P


def variants(P):
    out = {}
    for run, col in [('registered PDM run', 'p_for_holm'), ('corrected PDM run', 'corrected_run_p')]:
        base = P[col].astype(float).values.copy()
        out[f'(a/b) registered P, {run}'] = base
        u = base.copy()
        has = P.listed_unit_p_used.notna().values
        u[has] = P.listed_unit_p_used.values[has]
        out[f'(c) listed unit-level P, {run}'] = u
        lfv = base.copy()
        has2 = P.least_fav_unit_p_used.notna().values
        lfv[has2] = P.least_fav_unit_p_used.values[has2]
        out[f'(d) least favourable unit-level P, {run}'] = lfv
    return out


def families(P):
    """{(variant, family label): (mask, p)}; families: 25 tests, 21 (NCI-60 excluded), 23 and 19 (without PDM)."""
    fam = {}
    for name, p in variants(P).items():
        for excl in [False, True]:
            keep = (P.screen != 'NCI60').values if excl else np.ones(len(P), bool)
            for with_pdm in [True, False]:
                k = keep & (np.ones(len(P), bool) if with_pdm else ~P.screen.eq('PDM (organoids)').values)
                fam[(name, int(k.sum()), excl)] = (k, p)
    return fam


def holm_all(P):
    rows, per = [], []
    for (name, n, excl), (k, p) in families(P).items():
        adj = np.full(len(P), np.nan)
        adj[k] = lib.holm(p[k])
        label = f'{n} tests' if not excl else f'{n} tests (NCI-60 excluded)'
        rows.append({'variant': name, 'nci60_excluded': excl, 'family': label, 'tests': n,
                     'survive_holm_0.05': int(((adj < ALPHA) & k).sum()),
                     'fail': '; '.join(f'{s} {h}' for s, h, a, kk in zip(P.screen, P.hypothesis, adj, k) if kk and not a < ALPHA)})
        for i in np.where(k)[0]:
            per.append({'variant': name, 'family': label, 'tests': n, 'screen': P.screen[i], 'hypothesis': P.hypothesis[i],
                        'p_used': float(p[i]), 'holm_adjusted': float(adj[i]), 'survives': bool(adj[i] < ALPHA)})
    return pd.DataFrame(rows), pd.DataFrame(per)


def mc_margin(P, Pd):
    """For every resampling-based P not recomputed here: binomial MC SE at its number of draws; would its Holm
    decision change if it moved by +/- 3 SE (others fixed)?"""
    s54 = json.loads((lib.RES / 'revision_transfer' / '54_summary.json').read_text())
    nperm = int(s54['settings']['n_perm'])
    pdx = pd.DataFrame(s54['pdx']['tests'])
    t1 = pdx[pdx.analysis.str.startswith('T1') & (pdx.grouping == 'G1')].iloc[0]
    t1b = pdx[pdx.analysis.str.startswith('T1') & (pdx.grouping == 'G2')].iloc[0]
    mc = {}   # (screen, hyp, which) -> (P, draws, source)
    for s, h in [('FIMM', 'H5b'), ('GRAY', 'H3'), ('NCI60', 'H3'), ('NCI60', 'H5b')]:
        mc[(s, h, 'tail')] = None
    # least favourable H8/H9: max of tumour-level tests; the permutation ones carry MC error (5,000 permutations)
    for h, cols in [('H8', ['H8_wilcoxon_p', 'H8_perm_p_meanz']), ('H9', ['H9_mwu_p', 'H9_perm_p_meanz'])]:
        vals = {f'{c}|G1': float(t1[c]) for c in cols} | {f'{c}|G2': float(t1b[c]) for c in cols}
        arg = max(vals, key=vals.get)
        mc[('PDX_Bruna', h, 'least_fav')] = (vals[arg], nperm if 'perm' in arg else None, arg)
    bea = pd.DataFrame(s54['beataml']['tests']).iloc[0]
    for h in ['H6', 'H7']:
        mc[('BeatAML', h, 'least_fav')] = (float(bea[f'{h}_perm_p_patient_blocks']), nperm, f'{h}_perm_p_patient_blocks '
                                           '(entered as max with the registered P)')
    reg = json.loads((lib.RES / 'confirmation_PDM' / 'hypotheses.json').read_text())
    cor = json.loads((lib.RES / 'confirmation_PDM_corrected' / 'hypotheses.json').read_text())
    for h in ['H10', 'H11']:
        mc[('PDM (organoids)', h, 'registered run')] = (float(reg[h]['permutation_p_one_sided']), 10000, 'permutation (registered run)')
        mc[('PDM (organoids)', h, 'corrected run')] = (float(cor[h]['permutation_p_one_sided']), 10000, 'permutation (corrected run)')
    rows = []
    for (name, n, excl), (k, p) in families(P).items():
        adj0 = np.full(len(P), np.nan)
        adj0[k] = lib.holm(p[k])
        for i in np.where(k)[0]:
            s, h = P.screen[i], P.hypothesis[i]
            draws, src = None, None
            if h in ('H3', 'H5b') and np.isfinite(P.B_tail[i]) and ('unit-level' in name):
                draws, src = int(P.B_tail[i]), f'two-way bootstrap tail (B = {int(P.B_tail[i])})'
            elif s == 'PDM (organoids)':
                key = (s, h, 'registered run' if 'registered PDM run' in name else 'corrected run')
                draws, src = mc[key][1], mc[key][2]
            elif h in ('H6', 'H7', 'H8', 'H9') and name.startswith('(d)'):
                v, dr, sc_ = mc[(s, h, 'least_fav')]
                if abs(v - p[i]) < 1e-12 and dr:
                    draws, src = dr, sc_
            if not draws:
                continue
            pv = max(p[i], 1.0 / draws)
            se = np.sqrt(pv * (1 - pv) / draws)
            flips = []
            for dlt in (-3 * se, 3 * se):
                q = p.copy()
                q[i] = min(1.0, max(1.0 / draws, p[i] + dlt))
                adj = np.full(len(P), np.nan)
                adj[k] = lib.holm(q[k])
                ch = [f'{P.screen[j]} {P.hypothesis[j]}' for j in np.where(k)[0] if (adj[j] < ALPHA) != (adj0[j] < ALPHA)]
                flips.append('; '.join(ch))
            rows.append({'variant': name, 'family': f'{n} tests' + (' (NCI-60 excluded)' if excl else ''), 'screen': s,
                         'hypothesis': h, 'p_used': float(p[i]), 'draws': draws, 'source': src, 'mc_se': se,
                         'holm_adjusted': float(adj0[i]), 'survives': bool(adj0[i] < ALPHA),
                         'decisions_changed_at_minus_3se': flips[0], 'decisions_changed_at_plus_3se': flips[1]})
    return pd.DataFrame(rows)


def part_holm():
    # regression: the rules re-implemented here reproduce 86's summary with the stored B = 1,000 tails
    P0 = table()
    S0, D0 = holm_all(P0)
    ref = pd.read_csv(lib.OUT2 / 'holm' / 'holm_summary.csv')
    cmp_ = S0.merge(ref, on=['variant', 'nci60_excluded', 'family', 'tests'], suffixes=('', '_86'))
    assert len(cmp_) == len(ref), (len(cmp_), len(ref))
    assert (cmp_['survive_holm_0.05'] == cmp_['survive_holm_0.05_86']).all() and (cmp_.fail == cmp_.fail_86.fillna('')).all()
    tl = pd.read_csv(OUT / 'tails_b10k.csv')
    reg = pd.read_csv(OUT / 'regression_check.csv')
    assert reg.identical.all(), 'B = 1,000 regression run does not reproduce the stored tails'
    registered = set(zip(P0.screen, P0.hypothesis))                 # CCLE H3 has no registered test
    tl = tl[[(s, h) in registered for s, h in zip(tl.screen, tl.hypothesis)]]
    prim = tl[tl.label.str.startswith('primary')]
    tails = {(r.screen, r.hypothesis): (float(r.p_tail), int(r.B)) for r in prim.itertuples()}
    P1 = table(tails)
    S1, D1 = holm_all(P1)
    rep = tl[tl.label.str.startswith('replicate')]
    P2 = table({(r.screen, r.hypothesis): (float(r.p_tail), int(r.B)) for r in rep.itertuples()})
    S2, D2 = holm_all(P2)
    out = P1.copy()
    for name, p in variants(P1).items():
        out[f'p_used | {name}'] = p
    out.to_csv(OUT / 'holm_25_b10k.csv', index=False, float_format='%.6g')
    S = S1.merge(S0[['variant', 'family', 'survive_holm_0.05', 'fail']], on=['variant', 'family'], suffixes=('', '_B1000'))
    S = S.merge(S2[['variant', 'family', 'survive_holm_0.05']].rename(columns={'survive_holm_0.05': 'survive_replicate_seed'}),
                on=['variant', 'family'])
    S['changed_vs_B1000'] = S['survive_holm_0.05'] != S['survive_holm_0.05_B1000']
    S.to_csv(OUT / 'holm_summary_b10k.csv', index=False)
    # per-test decision changes
    D = D1.merge(D0[['variant', 'family', 'screen', 'hypothesis', 'p_used', 'holm_adjusted', 'survives']],
                 on=['variant', 'family', 'screen', 'hypothesis'], suffixes=('', '_B1000'))
    D = D.merge(D2[['variant', 'family', 'screen', 'hypothesis', 'holm_adjusted', 'survives']].rename(
        columns={'holm_adjusted': 'holm_adjusted_replicate_seed', 'survives': 'survives_replicate_seed'}),
        on=['variant', 'family', 'screen', 'hypothesis'])
    D['decision_changed'] = D.survives != D.survives_B1000
    D.to_csv(OUT / 'holm_per_test_b10k.csv', index=False, float_format='%.6g')
    D[D.decision_changed | (D.survives != D.survives_replicate_seed)].to_csv(OUT / 'decision_changes.csv', index=False,
                                                                             float_format='%.6g')
    hh = D[D.hypothesis.isin(['H6', 'H8', 'H10']) & D.tests.isin([25, 21])]
    hh.to_csv(OUT / 'h6_h8_h10.csv', index=False, float_format='%.6g')
    M = mc_margin(P1, D1)
    M.to_csv(OUT / 'mc_margin.csv', index=False, float_format='%.6g')
    lib.write_json(OUT / 'run.json', {
        'B_tail_new': B_NEW, 'B_tail_old': B_OLD, 'seed_56': int(m56().SEED) if M56 else 20261017,
        'seed_offsets': OFFSETS, 'screens_rerun': sorted(set(tl.screen)), 'near_threshold_tests': [f'{s} {h}' for s, h in NEAR],
        'nci60': ('re-run with B = 10,000' if 'NCI60' in set(tl.screen) else 'not re-run; tails 0/1,000 entered as 1/1,000 (as in 86)'),
        'regression_identical': bool(reg.identical.all()),
        'reproduces_86_summary_with_B1000': True,
        'tails_primary': {f'{s} {h}': p for (s, h), (p, _) in tails.items()},
        'any_decision_changed_vs_B1000': bool(D.decision_changed.any()),
        'any_decision_differs_between_seeds': bool((D.survives != D.survives_replicate_seed).any()),
        'counts': {f"{r.variant} | {r.family}": {'B10k': int(r['survive_holm_0.05']), 'B1000': int(r['survive_holm_0.05_B1000']),
                                                 'replicate_seed': int(r.survive_replicate_seed)} for _, r in S.iterrows()}})
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 120)
    print(S[['variant', 'family', 'survive_holm_0.05', 'survive_holm_0.05_B1000', 'survive_replicate_seed', 'changed_vs_B1000']].to_string(index=False))
    print(hh[['variant', 'family', 'screen', 'hypothesis', 'p_used', 'holm_adjusted', 'survives', 'holm_adjusted_B1000', 'survives_B1000']].to_string(index=False))
    print('decision changes:', int(D.decision_changed.sum()))
    print(M[(M.decisions_changed_at_minus_3se != '') | (M.decisions_changed_at_plus_3se != '')].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', nargs='*', default=['tails', 'holm'])
    ap.add_argument('--with-nci60', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if 'tails' in a.part:
        part_tails(SCREENS_ALL if a.with_nci60 else SCREENS)
    if 'holm' in a.part:
        part_holm()
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
