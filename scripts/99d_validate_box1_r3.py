"""Round 3, task 3 (editor item 8; R2-M3, R1-m6, R3-m12): validate selbench/box1.py's censoring rule and portable
activity rule against the project's results, and run every selbench unit test.

  j  censor_resistant + selbench.decompose   GDSC2 CurveCurator ln IC50 with undefined pairs (curve never crosses
     50%) tied at the resistant end, on every screened pair (GDSC2 AUC pairs of the retraining panel cells): the
     perfect-predictor share must equal 98_endpoint_cell_selection.py's tied reference (endpoint_cells/summary.csv,
     0.175). Informative: the same rule for CTRP v2 and GDSC1 (all AUC pairs as the screened set) and, from DrEval's
     curve file, GDSC2 with IC50s beyond the highest tested concentration also tied.
  k  active_drugs_rule (AUC default)  vs the paper's active sets: decomposition/drug_screen_stats.csv
     (frac_auc_below_0p8 >= 0.1; PRISM, CTRP, GDSC1, GDSC2), symmetric difference 0; AAC default (> 0.2) vs 45's
     confirmation-screen sets (CCLE, FIMM, GRAY: 1 - AAC < 0.8) and gCSI (mean viability < 0.8), counts as in
     revision_axis/summary_45.json. Informative: the ln IC50 default (IC50 below the highest tested concentration,
     undefined IC50s counted as non-responding) against the AUC default on DrEval's GDSC2 curves; the GR AOC default
     on gCSI against gCSI's viability AUC rule.
  tests  pytest test_selbench.py test_box1.py
Outputs (results/revision_round2/box1_validation/): validation_r3.csv, tests_r3.txt, summary_r3.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)
import box1  # noqa: E402  (selbench/ is on sys.path via lib77)

OUT = lib.out_dir('box1_validation')
ROWS, INFO = [], {}
LEARNED = ['EnsembleMF', 'GradientBoosting', 'SimpleNeuralNetwork', 'RandomForest']


def record(component, function, reference, value, ref_value, tol, note=''):
    d = abs(float(value) - float(ref_value))
    ROWS.append({'component': component, 'function': function, 'reference': reference, 'value': float(value),
                 'reference_value': float(ref_value), 'abs_diff': d, 'tolerance': tol, 'passed': bool(d <= tol),
                 'note': note})
    print(f'{component} {function:60s} {value:.6f} vs {ref_value:.6f} {"PASS" if d <= tol else "FAIL"}', flush=True)


def perfect_share(long):
    _, _, _, r2 = lib.selbench.decompose(long)
    return float(np.nanmean(r2.values)), int(len(r2))


def v_censoring():
    rt = lib.load_module('rt59', '59_retrain_score.py')
    resp = rt.responses()
    get = lambda ds, ep: resp[(resp.dataset == ds) & (resp.endpoint == ep)][['drug', 'cell', 'response']]  # noqa: E731
    oof, _, _ = rt.load_oof('GDSC2_AUC')
    cells = sorted(set.intersection(*[set(oof[m].cell) for m in LEARNED]))
    a_all = get('GDSC2', 'auc')
    a_all = a_all[a_all.cell.isin(cells)]
    ic = get('GDSC2', 'lnic50')
    tied = box1.censor_resistant(ic, screened=a_all[['drug', 'cell']])
    ps, nd = perfect_share(tied[['drug', 'cell', 'response']])
    S98 = pd.read_csv(lib.OUT2 / 'endpoint_cells' / 'summary.csv')
    ref = S98[S98.version.str.startswith('ln IC50 with undefined pairs tied')].perfect_share.iloc[0]
    record('j', 'censor_resistant: GDSC2 ln IC50 perfect-predictor share (tied)',
           'endpoint_cells/summary.csv (98)', ps, ref, 1e-12,
           f'{int(tied.censored.sum())} of {len(tied)} screened pairs censored ({tied.censored.mean():.3f}); {nd} drugs')
    # informative: other datasets (screened = all AUC pairs), with and without ties
    for ds in ['CTRPv2', 'GDSC1', 'GDSC2']:
        A, I = get(ds, 'auc'), get(ds, 'lnic50')
        t = box1.censor_resistant(I, screened=A[['drug', 'cell']])
        INFO[f'{ds}_lnIC50_perfect_share_defined_only'] = perfect_share(I)[0]
        INFO[f'{ds}_lnIC50_perfect_share_tied'] = perfect_share(t[['drug', 'cell', 'response']])[0]
        INFO[f'{ds}_lnIC50_censored_fraction'] = float(t.censored.mean())
        INFO[f'{ds}_AUC_perfect_share'] = perfect_share(A)[0]
    # informative: DrEval's GDSC2 curve file, IC50 beyond the highest tested concentration also tied
    hr = lib.load_module('hr66', '66_harmonized_range.py')
    cv = hr.curves('GDSC2')
    long = pd.DataFrame({'drug': cv.pubchem_id.values, 'cell': cv.cellosaurus_id.values,
                         'response': -cv.LN_IC50_curvecurator.values,
                         'limit': -np.log(cv.max_dose_M.values * 1e6)})        # LN_IC50_curvecurator: ln(uM)
    long = long.groupby(['drug', 'cell'], as_index=False).agg(response=('response', 'mean'), limit=('limit', 'mean'))
    u = box1.censor_resistant(long[['drug', 'cell', 'response']])
    b = box1.censor_resistant(long, resistant_limit='limit')
    INFO['GDSC2_curves_pairs'] = int(len(long))
    INFO['GDSC2_curves_undefined_fraction'] = float(u.censored.mean())
    INFO['GDSC2_curves_undefined_or_beyond_range_fraction'] = float(b.censored.mean())
    INFO['GDSC2_curves_perfect_share_undefined_tied'] = perfect_share(u[['drug', 'cell', 'response']])[0]
    INFO['GDSC2_curves_perfect_share_undefined_and_beyond_range_tied'] = perfect_share(b[['drug', 'cell', 'response']])[0]
    return cv


def v_activity(cv):
    DEC = lib.RES / 'decomposition'
    st = pd.read_csv(DEC / 'drug_screen_stats.csv', dtype={'compound': str})
    long = pd.read_csv(lib.PROJECT / 'data' / 'processed' / 'discovery_long.csv.gz', usecols=['screen', 'compound',
                                                                                              'depmap_id', 'auc'])
    for s in ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']:
        Z = pd.read_parquet(DEC / f'z_{s}.parquet')
        auc = long[long.screen == s].pivot_table(index='compound', columns='depmap_id', values='auc', aggfunc='mean')
        auc = auc.reindex(index=Z.index, columns=Z.columns)
        act = set(box1.active_drugs_rule(auc, 'auc'))
        ref = set(st[(st.screen == s) & (st.frac_auc_below_0p8 >= 0.1)].compound)
        record('k', f'active_drugs_rule AUC default ({s}): symmetric difference', 'decomposition/drug_screen_stats.csv (02/45)',
               len(act ^ ref), 0, 0, f'{len(act)} active of {len(Z)} drugs; paper {len(ref)}')
    a45 = lib.load_module('ax45', '45_axis_activity_lineage.py')
    import json
    s45 = json.loads((lib.RES / 'revision_axis' / 'summary_45.json').read_text())['confirmation_active_drugs']
    for name, ep in [('CCLE', 'aac'), ('FIMM', 'aac'), ('GRAY', 'aac'), ('gCSI', 'auc')]:
        M, _ = a45.confirmation_z(name)
        X = M if ep == 'aac' else -M                    # gCSI: sens = -mean viability; AAC screens: sens = AAC
        act = set(box1.active_drugs_rule(X, ep))
        if ep == 'aac':
            ref = set(M.index[((1 - M) < 0.8).sum(axis=1) / M.notna().sum(axis=1) >= 0.1])
        else:
            ref = set(M.index[((-M) < 0.8).sum(axis=1) / M.notna().sum(axis=1) >= 0.1])
        record('k', f'active_drugs_rule {ep.upper()} default ({name}): symmetric difference vs 45 rule',
               '45_axis_activity_lineage.py conf_active', len(act ^ ref), 0, 0,
               f'{len(act)} active; summary_45 counts {s45.get(name)} on the axis drugs')
    # informative: ln IC50 default vs AUC default on DrEval's GDSC2 curves (same pairs)
    pv = lambda col: cv.pivot_table(index='pubchem_id', columns='cellosaurus_id', values=col, aggfunc='mean')  # noqa: E731
    aucM = pv('AUC_curvecurator')
    icM = pv('LN_IC50_curvecurator').reindex(index=aucM.index, columns=aucM.columns)
    mxM = np.log(pv('max_dose_M') * 1e6).reindex(index=aucM.index, columns=aucM.columns)
    scr = aucM.notna()
    a_auc = set(box1.active_drugs_rule(aucM, 'auc'))
    a_ic = set(box1.active_drugs_rule(icM, 'ln_ic50', max_tested=mxM, screened=scr))
    a_ic_naive = set(box1.active_drugs_rule(icM, 'ln_ic50', max_tested=mxM))
    INFO['GDSC2_curves_drugs'] = int(len(aucM))
    INFO['GDSC2_active_AUC_default'] = len(a_auc)
    INFO['GDSC2_active_lnIC50_default'] = len(a_ic)
    INFO['GDSC2_active_lnIC50_without_screened_mask'] = len(a_ic_naive)
    INFO['GDSC2_lnIC50_vs_AUC_jaccard'] = len(a_auc & a_ic) / max(1, len(a_auc | a_ic))
    INFO['GDSC2_lnIC50_vs_AUC_both'] = len(a_auc & a_ic)
    INFO['GDSC2_lnIC50_only'] = len(a_ic - a_auc)
    INFO['GDSC2_AUC_only'] = len(a_auc - a_ic)
    # informative: GR AOC default on gCSI vs gCSI's viability AUC rule
    try:
        Mg, _ = a45.confirmation_z('gCSI')
        Mgr, _ = a45.confirmation_z('gCSI_GRAOC')
        a_v = set(box1.active_drugs_rule(-Mg, 'auc'))
        a_gr = set(box1.active_drugs_rule(Mgr, 'gr_aoc'))
        INFO['gCSI_active_viability_AUC'] = len(a_v)
        INFO['gCSI_active_GR_AOC'] = len(a_gr)
        INFO['gCSI_GR_vs_AUC_jaccard'] = len(a_v & a_gr) / max(1, len(a_v | a_gr))
        INFO['gCSI_GR_AOC_range'] = [float(np.nanmin(Mgr.values)), float(np.nanmax(Mgr.values))]
    except Exception as e:  # noqa: BLE001
        INFO['gCSI_GR_error'] = repr(e)


def tests():
    sel = lib.PROJECT / 'selbench'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'test_selbench.py',
                        'test_box1.py'], cwd=str(sel), capture_output=True, text=True, env=env)
    (OUT / 'tests_r3.txt').write_text(r.stdout + r.stderr)
    last = [line for line in (r.stdout + r.stderr).splitlines() if 'passed' in line or 'failed' in line]
    return {'returncode': r.returncode, 'summary': last[-1] if last else ''}


def main():
    t0 = time.time()
    cv = v_censoring()
    v_activity(cv)
    T = pd.DataFrame(ROWS)
    T.to_csv(OUT / 'validation_r3.csv', index=False, float_format='%.10g')
    tr = tests()
    lib.write_json(OUT / 'summary_r3.json', {'seconds': round(time.time() - t0), 'checks': int(len(T)),
                                             'passed': int(T.passed.sum()), 'failed': list(T[~T.passed].function),
                                             'unit_tests': tr, 'informative': INFO,
                                             'paper_activity_rule': box1.PAPER_ACTIVITY_RULE})
    pd.set_option('display.width', 250)
    print(T[['component', 'function', 'value', 'reference_value', 'passed', 'note']].to_string(index=False))
    print(tr)
    print(INFO)


if __name__ == '__main__':
    main()
