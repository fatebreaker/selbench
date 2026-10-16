"""Revision workstream (cross-laboratory, processing-controlled): data layer.

Reads DrEval's curated response files (data/dreval/{CTRPv2,CTRPv1,GDSC1,GDSC2,CCLE}.zip) and builds every "target
version" of the paired design: the same dataset processed several ways (CurveCurator ln IC50 and AUC as re-fitted by
DrEval, and the data provider's own summary: CTRP area_under_curve, Sanger LN_IC50 and AUC, CCLE ActArea). Every
version is oriented so that higher = more sensitive (sensitivity = -ln IC50, -area under the viability curve,
+ActArea), and each orientation is verified empirically against -LN_IC50_curvecurator.

Also: ln IC50 censoring per dataset (R2 minor 6), CTRPv2 drug activity (R1 minor 9), mechanism classes of the DrEval
drugs for leave-class-out axes (editor item 3), cell-line tissue labels (lineage adjustment), the leaderboard and
reference predictions keyed by Cellosaurus ID, and an exact check of which predictors are drug-agnostic.
Duplicated drug-cell rows (exact copies in CTRPv2; 23 differing pairs in CCLE) are averaged, as in
35_dreval_crossstudy.observed().

Other scripts import these functions (scripts 42-44). Run directly to write the data-level outputs to
results/revision_crosslab/: versions.csv, orientation_check.csv, censoring.csv, censoring_by_drug.csv.gz,
drug_activity_ctrpv2.csv, drug_classes.csv, cell_tissue.csv, drug_agnostic_check.csv, data_checks.json.
"""
import sys

sys.dont_write_bytecode = True  # never write bytecode caches next to the project's scripts
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import zipfile  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from common import PROCESSED, PROJECT, RESULTS, norm_name, write_json  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = load('rs', '28_dreval_rescore.py')
dclass = load('dclass', '12_drug_classes.py')
selbench = rs.selbench

DRE = PROJECT / 'data' / 'dreval'
OUT = RESULTS / 'revision_crosslab'
DATASETS = ['CTRPv2', 'CTRPv1', 'GDSC1', 'GDSC2', 'CCLE']
REFERENCE = 'CTRPv2|CC|lnIC50'          # training endpoint of every DrEval leaderboard model
AUC_REFERENCE = 'CTRPv2|CC|AUC'         # uncensored reference used for the censoring control
CC = [('LN_IC50_curvecurator', -1, 'CC', 'lnIC50', 'CurveCurator (DrEval re-fit)'),
      ('AUC_curvecurator', -1, 'CC', 'AUC', 'CurveCurator (DrEval re-fit)')]
# (column, sign to sensitivity, processing tag, endpoint, processing description)
ENDPOINTS = {
    'CTRPv2': CC + [('area_under_curve', -1, 'CTRP', 'AUC', 'provider fit (CTRP area_under_curve)')],
    'CTRPv1': CC + [('area_under_curve', -1, 'CTRP', 'AUC', 'provider fit (CTRP area_under_curve)')],
    'GDSC1': CC + [('LN_IC50', -1, 'Sanger', 'lnIC50', 'provider fit (Sanger LN_IC50)'),
                   ('AUC', -1, 'Sanger', 'AUC', 'provider fit (Sanger AUC)')],
    'GDSC2': CC + [('LN_IC50', -1, 'Sanger', 'lnIC50', 'provider fit (Sanger LN_IC50)'),
                   ('AUC', -1, 'Sanger', 'AUC', 'provider fit (Sanger AUC)')],
    'CCLE': CC + [('ActArea', 1, 'CCLE', 'ActArea', 'provider fit (CCLE ActArea, area above the curve)')],
}
LAB = {'CTRPv2': 'Broad', 'CTRPv1': 'Broad', 'GDSC1': 'Sanger', 'GDSC2': 'Sanger', 'CCLE': 'Broad/Novartis'}
ACTIVE_AUC, ACTIVE_FRAC = 0.8, 0.10      # active in CTRPv2: >= 10% of cell lines with AUC_curvecurator < 0.8
UNCLASSED = 'Other / non-oncology'       # fallback bin of 12_drug_classes, not a mechanism: keeps the standard axis


def version_key(dataset, proc, endpoint):
    return f'{dataset}|{proc}|{endpoint}'


def version_table():
    rows = []
    for ds, eps in ENDPOINTS.items():
        for col, sign, proc, ep, desc in eps:
            rows.append({'version': version_key(ds, proc, ep), 'dataset': ds, 'laboratory': LAB[ds], 'column': col,
                         'sign_to_sensitivity': sign, 'processing': proc, 'endpoint': ep, 'description': desc})
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def read_raw(dataset):
    """Curated DrEval response table of one dataset, one row per drug-cell pair (duplicates averaged)."""
    cols = sorted({c for c, *_ in ENDPOINTS[dataset]})
    with zipfile.ZipFile(DRE / f'{dataset}.zip') as z:
        d = pd.read_csv(z.open(f'{dataset}/{dataset}.csv'), dtype={'pubchem_id': str},
                        usecols=['cellosaurus_id', 'cell_line_name', 'pubchem_id', 'drug_name', 'tissue'] + cols,
                        low_memory=False)
    meta = d.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False)[['cell_line_name', 'drug_name', 'tissue']].first()
    vals = d.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False)[cols].mean()
    out = meta.merge(vals, on=['pubchem_id', 'cellosaurus_id'])
    out.attrs['raw_rows'] = len(d)
    return out


def versions():
    """{version key: long table (drug, cell, response), higher = more sensitive}."""
    out = {}
    for ds, eps in ENDPOINTS.items():
        raw = read_raw(ds)
        for col, sign, proc, ep, _ in eps:
            t = raw[['pubchem_id', 'cellosaurus_id', col]].dropna()
            out[version_key(ds, proc, ep)] = pd.DataFrame({'drug': t.pubchem_id.values, 'cell': t.cellosaurus_id.values,
                                                           'response': sign * t[col].values})
    return out


def orientation_check():
    """Median over drugs of the per-drug Spearman correlation of each oriented endpoint with -LN_IC50_curvecurator."""
    rows = []
    for ds, eps in ENDPOINTS.items():
        raw = read_raw(ds)
        ref = -raw.LN_IC50_curvecurator
        for col, sign, proc, ep, _ in eps:
            s = sign * raw[col]
            rho = []
            for _, idx in raw.groupby('pubchem_id').groups.items():
                a, b = s.loc[idx].values, ref.loc[idx].values
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 30 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                    rho.append(spearmanr(a[ok], b[ok])[0])
            rho = np.array(rho)
            rows.append({'version': version_key(ds, proc, ep), 'dataset': ds, 'column': col, 'sign_to_sensitivity': sign,
                         'drugs': len(rho), 'median_spearman_with_neg_lnIC50_cc': float(np.median(rho)),
                         'frac_drugs_positive': float(np.mean(rho > 0)), 'orientation_ok': bool(np.median(rho) > 0)})
    return pd.DataFrame(rows)


def censoring():
    """How much of each dataset lacks a CurveCurator ln IC50, and is the missingness informative?"""
    rows, per = [], []
    for ds in DATASETS:
        raw = read_raw(ds)
        miss = raw.LN_IC50_curvecurator.isna()
        auc = raw.AUC_curvecurator
        row = {'dataset': ds, 'raw_rows': raw.attrs.get('raw_rows'), 'drug_cell_pairs': len(raw),
               'frac_missing_lnIC50_cc': float(miss.mean()), 'frac_missing_AUC_cc': float(auc.isna().mean()),
               'median_AUC_cc_lnIC50_missing': float(auc[miss].median()),
               'median_AUC_cc_lnIC50_present': float(auc[~miss].median()),
               'mannwhitney_p_AUC_missing_vs_present': float(mannwhitneyu(auc[miss].dropna(), auc[~miss].dropna()).pvalue)}
        for col, _, proc, ep, _ in ENDPOINTS[ds][2:]:
            row[f'frac_missing_{proc}_{ep}'] = float(raw[col].isna().mean())
        g = raw.assign(m=miss).groupby('pubchem_id')
        fr = g.m.mean()
        row['median_over_drugs_frac_missing_lnIC50_cc'] = float(fr.median())
        rows.append(row)
        t = pd.DataFrame({'dataset': ds, 'drug': fr.index, 'pairs': g.size().values, 'frac_missing_lnIC50_cc': fr.values,
                          'median_AUC_cc': g.AUC_curvecurator.median().values})
        per.append(t)
    return pd.DataFrame(rows), pd.concat(per, ignore_index=True)


def activity():
    """CTRPv2 activity per drug: fraction of measured cell lines with AUC_curvecurator < 0.8 (all CTRPv2 cells)."""
    raw = read_raw('CTRPv2')
    g = raw.dropna(subset=['AUC_curvecurator']).groupby('pubchem_id').AUC_curvecurator
    t = pd.DataFrame({'drug': g.size().index, 'cells_with_AUC_cc': g.size().values,
                      'frac_AUC_cc_below_0p8': g.apply(lambda s: float((s < ACTIVE_AUC).mean())).values})
    t['active_ctrpv2'] = t.frac_AUC_cc_below_0p8 >= ACTIVE_FRAC
    return t


def drug_classes():
    """Mechanism class of every DrEval drug (pubchem CID), for leave-class-out axes.

    strict (as specified): CID -> discovery compound via 28_dreval_rescore.id_maps() (InChIKey), class from
    results/drug_classes/drug_table.csv. extended: CID mapping, else a unique normalized-name match to the discovery
    compound synonyms; class from drug_table.csv, else 12_drug_classes.classify() of the compound's target/MOA text.
    In both, 'Other / non-oncology' (the classifier's fallback bin) and unmapped drugs keep the standard axis.
    """
    _, dmap = rs.id_maps()
    comp = pd.read_csv(PROCESSED / 'compounds.csv', dtype=str).set_index('compound')
    syn = {}
    for c, row in comp.iterrows():
        for n in [row['name']] + str(row['all_names']).split('|'):
            k = norm_name(n)
            if k:
                syn.setdefault(k, set()).add(c)
    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0)
    text = comp.targets.fillna('') + ' ' + comp.moa.fillna('') + ' ' + comp.target_or_mechanism.fillna('')
    names = []
    for ds in DATASETS:
        with zipfile.ZipFile(DRE / f'{ds}.zip') as z:
            names.append(pd.read_csv(z.open(f'{ds}/drug_names.csv'), dtype=str).assign(dataset=ds))
    names = pd.concat(names)
    rows = []
    for cid, g in names.groupby('pubchem_id'):
        by_name = {next(iter(syn[k])) for k in map(norm_name, g.drug_name) if k in syn and len(syn[k]) == 1}
        dpc_cid = dmap.get(cid)
        dpc_name = next(iter(by_name)) if len(by_name) == 1 else None
        dpc_ext = dpc_cid if isinstance(dpc_cid, str) else dpc_name
        cls_strict = dt['class'].get(dpc_cid) if isinstance(dpc_cid, str) else None
        cls_ext = None
        if isinstance(dpc_ext, str):
            cls_ext = dt['class'].get(dpc_ext)
            if cls_ext is None and text.get(dpc_ext, '').strip():
                cls_ext = dclass.classify(text[dpc_ext])
        rows.append({'drug': cid, 'drug_name': g.drug_name.iloc[0], 'datasets': '|'.join(sorted(g.dataset.unique())),
                     'dpc_strict': dpc_cid, 'class_strict': cls_strict, 'dpc_extended': dpc_ext, 'class_extended': cls_ext,
                     'name_vs_cid_conflict': bool(isinstance(dpc_cid, str) and dpc_name is not None and dpc_name != dpc_cid)})
    t = pd.DataFrame(rows)
    for v in ['strict', 'extended']:
        c = t[f'class_{v}']
        t[f'lco_class_{v}'] = c.where(c.notna() & (c != UNCLASSED))
    return t


def cell_tissue():
    t = pd.concat([read_raw(ds)[['cellosaurus_id', 'tissue']] for ds in DATASETS]).dropna().drop_duplicates()
    assert t.cellosaurus_id.is_unique, 'tissue labels must agree across DrEval files'
    return t.set_index('cellosaurus_id').tissue


def predictions():
    """Leaderboard (CTRPv2-trained, leave-cell-line-out) and reference predictions, higher = more sensitive.

    Returns ({model: drug x cell (Cellosaurus) matrix}, {model: class}, leaderboard dict with split labels).
    The two naive predictors that are constant within a drug (NaivePredictor, NaiveDrugMeanPredictor) are excluded.
    """
    cid = pd.read_csv(DRE / 'cell_ids.csv', dtype=str).set_index('cell_line_name').cellosaurus_id
    lb = rs.load_leaderboard()
    refs = pd.read_csv(RESULTS / 'dreval' / 'reference_predictions.csv.gz', dtype={'drug': str})
    long = {m: d[['drug', 'cell', 'prediction']] for m, d in lb.items() if m not in rs.CONSTANT}
    for m in rs.OURS:
        long[m] = refs[refs.model == m][['drug', 'cell', 'prediction']]
    P = {}
    for m, d in long.items():
        d = d.assign(cell=d.cell.map(cid))
        assert d.cell.notna().all(), m
        P[m] = d.groupby(['drug', 'cell']).prediction.mean().unstack()
    return P, {m: rs.model_class(m) for m in P}, lb


def agnostic_check(P, lb):
    """Is a predictor drug-agnostic? exact: identical value for every drug at each cell line. within split: for
    leave-cell-line-out models, the per-drug profiles are exactly collinear within each CV split (additive
    drug + cell models whose drug offsets differ between splits)."""
    rows = []
    for m, M in P.items():
        v = M.values
        spread = np.nanmax(v, axis=0) - np.nanmin(v, axis=0)
        sd = np.nanstd(v)
        exact = bool(np.nanmax(spread) <= 1e-9 * max(sd, 1e-12))
        row = {'model': m, 'class': rs.model_class(m), 'max_per_cell_spread_across_drugs': float(np.nanmax(spread)),
               'prediction_sd': float(sd), 'exactly_drug_agnostic': exact}
        if m in lb:
            d = lb[m]
            worst = 1.0
            for _, g in d.groupby('split'):
                S = g.pivot_table(index='drug', columns='cell', values='prediction')
                r = S.notna().sum(axis=1).idxmax()
                pr = S.loc[r].values
                for x in S.values:
                    ok = np.isfinite(x) & np.isfinite(pr)
                    if ok.sum() >= 10 and np.std(x[ok]) > 1e-12 and np.std(pr[ok]) > 1e-12:
                        worst = min(worst, float(np.corrcoef(x[ok], pr[ok])[0, 1]))
            row['min_within_split_profile_corr'] = worst
            row['drug_agnostic_within_split'] = bool(worst >= 1 - 1e-9)
        else:
            row['min_within_split_profile_corr'] = np.nan
            row['drug_agnostic_within_split'] = exact
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vt = version_table()
    V = versions()
    vt['pairs'] = vt.version.map(lambda k: len(V[k]))
    vt['drugs'] = vt.version.map(lambda k: V[k].drug.nunique())
    vt['cells'] = vt.version.map(lambda k: V[k].cell.nunique())
    vt.to_csv(OUT / 'versions.csv', index=False)
    ori = orientation_check()
    ori.to_csv(OUT / 'orientation_check.csv', index=False)
    assert ori.orientation_ok.all(), ori
    cen, cen_d = censoring()
    cen.to_csv(OUT / 'censoring.csv', index=False)
    cen_d.to_csv(OUT / 'censoring_by_drug.csv.gz', index=False)
    act = activity()
    act.to_csv(OUT / 'drug_activity_ctrpv2.csv', index=False)
    cls = drug_classes()
    cls.to_csv(OUT / 'drug_classes.csv', index=False)
    tis = cell_tissue()
    tis.rename('tissue').reset_index().to_csv(OUT / 'cell_tissue.csv', index=False)
    P, mclass, lb = predictions()
    ag = agnostic_check(P, lb)
    ag.to_csv(OUT / 'drug_agnostic_check.csv', index=False)

    # consistency checks: the zip's CurveCurator ln IC50 equals DrEval's evaluation data and the leaderboard response
    checks = {}
    allres = rs.PROJECT / 'data' / 'dreval' / 'all_results_parquet'
    for ds, prefix in [('CTRPv2', 'predictions'), ('CTRPv1', 'cross-study-CTRPv1'), ('GDSC1', 'cross-study-GDSC1'),
                       ('GDSC2', 'cross-study-GDSC2'), ('CCLE', 'cross-study-CCLE')]:
        o = pd.concat([pd.read_parquet(allres / f'{prefix}__{m}.parquet', columns=['pubchem_id', 'cellosaurus_id', 'y_true'])
                       for m in ['LCO', 'LTO', 'LPO', 'LDO']]).groupby(['pubchem_id', 'cellosaurus_id']).y_true.mean()
        z = read_raw(ds).dropna(subset=['LN_IC50_curvecurator']).set_index(['pubchem_id', 'cellosaurus_id']).LN_IC50_curvecurator
        j = pd.concat([o.rename('a'), z.rename('b')], axis=1)
        checks[f'{ds}_lnIC50_cc_vs_all_results'] = {'pairs_all_results': int(len(o)), 'pairs_zip': int(len(z)),
                                                    'pairs_only_in_one': int(j.isna().any(axis=1).sum()),
                                                    'max_abs_diff': float((j.a - j.b).abs().max())}
    cid = pd.read_csv(DRE / 'cell_ids.csv', dtype=str).set_index('cell_line_name').cellosaurus_id
    obs = pd.concat([d[['drug', 'cell', 'response']] for d in lb.values()]).groupby(['drug', 'cell']).response.mean()
    obs.index = pd.MultiIndex.from_arrays([obs.index.get_level_values(0), obs.index.get_level_values(1).map(cid)])
    z = read_raw('CTRPv2').dropna(subset=['LN_IC50_curvecurator']).set_index(['pubchem_id', 'cellosaurus_id']).LN_IC50_curvecurator
    checks['leaderboard_response_vs_zip'] = {'pairs': int(len(obs)),
                                             'max_abs_diff_after_sign_flip': float((obs + z.reindex(obs.index)).abs().max())}
    checks['active_drugs_ctrpv2'] = {'rule': f'>= {ACTIVE_FRAC:.0%} of cell lines with AUC_curvecurator < {ACTIVE_AUC}',
                                     'active': int(act.active_ctrpv2.sum()), 'drugs': int(len(act))}
    for v in ['strict', 'extended']:
        checks[f'classes_{v}'] = {ds: int(cls[cls.datasets.str.contains(ds) & cls[f'lco_class_{v}'].notna()].shape[0])
                                  for ds in DATASETS}
    checks['classes_drugs_per_dataset'] = {ds: int(cls.datasets.str.contains(ds).sum()) for ds in DATASETS}
    checks['exactly_drug_agnostic'] = ag[ag.exactly_drug_agnostic].model.tolist()
    checks['drug_agnostic_within_split_only'] = ag[ag.drug_agnostic_within_split & ~ag.exactly_drug_agnostic].model.tolist()
    write_json(OUT / 'data_checks.json', checks)
    pd.set_option('display.width', 250)
    print(vt.to_string(index=False))
    print(ori.round(3).to_string(index=False))
    print(cen.round(3).to_string(index=False))
    print(ag.round(6).to_string(index=False))
    print(checks)


if __name__ == '__main__':
    main()
