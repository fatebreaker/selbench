"""Revision (round 1), fitness-axis workstream, part 2: the axis under a harmonized curve fitter (R2-9, R1-M1, R3-M4c).

DrEval (Zenodo 23159774) re-fitted CTRPv2, CTRPv1, GDSC1, GDSC2 and CCLE with one pipeline (CurveCurator). From each
curated file (<name>/<name>.csv) we take, per (cellosaurus_id, pubchem_id), the CurveCurator ln IC50 and AUC and the
data providers' own summaries (GDSC: Sanger LN_IC50 and AUC; CTRP v1/v2: provider area under the curve as normalized
by DrEval, column AUC; CCLE: ActArea, and ln of the provider IC50, which is capped at 8 uM). Sensitivity = -value
(ActArea: +value). Duplicate measurements are averaged.

Per dataset and endpoint: selbench.decompose (>=30 cells and >=10 distinct values per drug; rank-normal per drug;
leave-drug-out axis), median fitness share r_zg^2, and the screen axis = per-cell mean z (>=10 contributing drugs),
with split-half reliability (random drug halves, Spearman-Brown, median of 200). Drug sets: all drugs; active drugs
(>=10% of lines with CurveCurator AUC < 0.8 in that dataset); per pair, shared drugs and shared drugs active in both.
Pairwise agreement: Pearson over shared cell lines, disattenuated by the split-halves of the same variant (computed on
the shared cells for pair-specific drug sets; not reported if a reliability is < 0.5 or the drug set has < 8 drugs);
per-drug raw (z) and selective (residual) agreement of shared drugs. A cell's axis value needs >=10 contributing
drugs (half the drug set if it has fewer than 20 drugs).
Within a dataset: agreement of axes from different fitters/summaries of the same raw data. CurveCurator ln IC50 is
missing when no IC50 is reached (response-dependent censoring, R2 minor 6); it is tabulated.

Outputs (results/revision_axis/): dreval_datasets.csv, dreval_axis_pairs.csv, dreval_perdrug_pairs.csv,
dreval_within_screen.csv, dreval_censoring.csv, dreval_axes_cellosaurus.csv, dreval_axes_depmap.csv, summary_46.json.
"""
import os

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import importlib  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import PROJECT, RESULTS, load_models, write_json  # noqa: E402

sys.path.insert(0, str(PROJECT / 'selbench'))
import selbench  # noqa: E402

H = importlib.import_module('45_axis_activity_lineage')
OUT = RESULTS / 'revision_axis'
DRE = PROJECT / 'data' / 'dreval'
DATASETS = ['CTRPv2', 'CTRPv1', 'GDSC1', 'GDSC2', 'CCLE']
LAB = {'CTRPv2': 'Broad', 'CTRPv1': 'Broad', 'GDSC1': 'Sanger', 'GDSC2': 'Sanger', 'CCLE': 'Novartis'}
# endpoint -> dataset -> (column, sign, transform)
ENDPOINTS = {
    'CC_lnIC50': {d: ('LN_IC50_curvecurator', -1, None) for d in DATASETS},
    'CC_AUC': {d: ('AUC_curvecurator', -1, None) for d in DATASETS},
    'prov_lnIC50': {'GDSC1': ('LN_IC50', -1, None), 'GDSC2': ('LN_IC50', -1, None), 'CCLE': ('IC50 (µM)', -1, 'log')},
    'prov_AUC': {'CTRPv2': ('AUC', -1, None), 'CTRPv1': ('AUC', -1, None), 'GDSC1': ('AUC', -1, None),
                 'GDSC2': ('AUC', -1, None), 'CCLE': ('ActArea', 1, None)},
}
MIN_AXIS_DRUGS = 10
MIN_REL_DISATT, MIN_DRUGS_DISATT = 0.5, 8   # below these, disattenuation is not reported (unstable)
SEED = 4601


def min_drugs(n):
    """Contributing drugs required for a cell's axis value: 10, or half the drug set if it has fewer than 20."""
    return min(MIN_AXIS_DRUGS, max(2, n // 2))


def disatt(r, ra, rb, n_drugs):
    if not (np.isfinite(r) and ra >= MIN_REL_DISATT and rb >= MIN_REL_DISATT and n_drugs >= MIN_DRUGS_DISATT):
        return np.nan
    return r / np.sqrt(ra * rb)


def load_dataset(name):
    cols = ['cellosaurus_id', 'pubchem_id', 'LN_IC50_curvecurator', 'AUC_curvecurator']
    extra = sorted({v[0] for e in ENDPOINTS.values() for d, v in e.items() if d == name} - set(cols))
    with zipfile.ZipFile(DRE / f'{name}.zip') as z:
        d = pd.read_csv(z.open(f'{name}/{name}.csv'), usecols=cols + extra, dtype={'pubchem_id': str})
    return name, d


def endpoint_long(d, col, sign, transform):
    v = d[col].astype(float)
    if transform == 'log':
        v = np.log(v.where(v > 0))
    out = pd.DataFrame({'drug': d.pubchem_id, 'cell': d.cellosaurus_id, 'response': sign * v}).dropna()
    return out.groupby(['drug', 'cell'], as_index=False).response.mean()


def decompose_one(args):
    name, ep, long, active = args
    Z, G, R, r2 = selbench.decompose(long)
    rng = np.random.default_rng(SEED + DATASETS.index(name) * 10 + list(ENDPOINTS).index(ep))
    rel = H.split_half(Z, rng)
    act = sorted(set(active) & set(Z.index))
    rel_act = H.split_half(Z, rng, act) if len(act) >= 4 else (np.nan,) * 3
    return name, ep, Z, R, r2, rel, rel_act, act


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    with ProcessPoolExecutor(5) as ex:
        data = dict(ex.map(load_dataset, DATASETS))
    # censoring of the CurveCurator ln IC50 (R2 minor 6)
    cens = []
    for name, d in data.items():
        g = d.groupby(['cellosaurus_id', 'pubchem_id'], as_index=False)[['LN_IC50_curvecurator', 'AUC_curvecurator']].mean()
        miss = g.LN_IC50_curvecurator.isna()
        per_drug = miss.groupby(g.pubchem_id).mean()
        cens.append({'dataset': name, 'pairs': len(g), 'frac_lnIC50_missing': float(miss.mean()),
                     'median_ccAUC_missing': float(g.AUC_curvecurator[miss].median()),
                     'median_ccAUC_present': float(g.AUC_curvecurator[~miss].median()),
                     'drugs': int(g.pubchem_id.nunique()), 'median_drug_frac_missing': float(per_drug.median()),
                     'drugs_over_50pct_missing': int((per_drug > 0.5).sum())})
    cens = pd.DataFrame(cens)
    cens.to_csv(OUT / 'dreval_censoring.csv', index=False)
    print(cens.round(3).to_string(index=False), flush=True)

    # activity per dataset from the (uncensored) CurveCurator AUC
    active = {}
    for name, d in data.items():
        g = d.groupby(['pubchem_id', 'cellosaurus_id']).AUC_curvecurator.mean()
        frac = (g < 0.8).groupby(level=0).mean()
        active[name] = set(frac.index[frac >= 0.1])

    jobs = [(name, ep, endpoint_long(data[name], *ENDPOINTS[ep][name]), active[name])
            for ep in ENDPOINTS for name in DATASETS if name in ENDPOINTS[ep]]
    res = {}
    with ProcessPoolExecutor(6) as ex:
        for name, ep, Z, R, r2, rel, rel_act, act in ex.map(decompose_one, jobs):
            res[(name, ep)] = {'Z': Z, 'R': R, 'r2': r2, 'rel': rel, 'rel_act': rel_act, 'act': act}
            print(name, ep, Z.shape, round(rel[0], 3), round(float(r2.median()), 3), flush=True)

    # ---------------- per dataset x endpoint ----------------
    rows, axes = [], {}
    for (name, ep), v in res.items():
        Z = v['Z']
        ax = H.axis_of(Z, None, min_drugs(Z.shape[0]))
        axa = H.axis_of(Z, v['act'], min_drugs(len(v['act']))) if len(v['act']) >= 4 else None
        axes[f'{name}|{ep}|all'] = ax
        if axa is not None:
            axes[f'{name}|{ep}|active'] = axa
        rows.append({'dataset': name, 'lab': LAB[name], 'endpoint': ep, 'drugs': Z.shape[0], 'cells_in_matrix': Z.shape[1],
                     'cells_axis': int(ax.notna().sum()), 'measured_pairs': int(Z.notna().values.sum()),
                     'split_half': v['rel'][0], 'split_half_lo': v['rel'][1], 'split_half_hi': v['rel'][2],
                     'median_fitness_r2': float(v['r2'].median()),
                     'median_fitness_r2_active': float(v['r2'].reindex(v['act']).median()) if v['act'] else np.nan,
                     'active_drugs': len(v['act']), 'frac_inactive': 1 - len(v['act']) / Z.shape[0],
                     'split_half_active': v['rel_act'][0]})
    dtab = pd.DataFrame(rows).sort_values(['endpoint', 'dataset'])
    dtab.to_csv(OUT / 'dreval_datasets.csv', index=False)
    print(dtab.round(3).to_string(index=False), flush=True)
    A = pd.DataFrame(axes)
    A.to_csv(OUT / 'dreval_axes_cellosaurus.csv')
    mod = load_models()[['ModelID', 'RRID']].dropna().drop_duplicates('RRID').set_index('RRID').ModelID
    Ad = A.copy()
    Ad.index = Ad.index.map(mod)
    Ad = Ad[Ad.index.notna()]
    Ad = Ad[~Ad.index.duplicated()]
    Ad.to_csv(OUT / 'dreval_axes_depmap.csv')

    # ---------------- pairwise between datasets (same endpoint) ----------------
    prow, drow = [], []
    for ep in ENDPOINTS:
        names = [n for n in DATASETS if (n, ep) in res]
        for a, b in itertools.combinations(names, 2):
            va, vb = res[(a, ep)], res[(b, ep)]
            Za, Zb = va['Z'], vb['Z']
            cells = Za.columns.intersection(Zb.columns)
            common = Za.index.intersection(Zb.index)
            base = {'endpoint': ep, 'a': a, 'b': b, 'same_lab': LAB[a] == LAB[b], 'shared_cells': len(cells),
                    'shared_drugs': len(common)}
            # all-drug and active-drug axes of each dataset
            for variant in ['all', 'active']:
                ka, kb = f'{a}|{ep}|{variant}', f'{b}|{ep}|{variant}'
                if ka not in axes or kb not in axes:
                    continue
                x, y = axes[ka].reindex(cells), axes[kb].reindex(cells)
                r = H.corr(x, y)
                lo, hi = H.boot_ci(x, y, rng)
                ra = va['rel'][0] if variant == 'all' else va['rel_act'][0]
                rb = vb['rel'][0] if variant == 'all' else vb['rel_act'][0]
                na_, nb_ = (Za.shape[0], Zb.shape[0]) if variant == 'all' else (len(va['act']), len(vb['act']))
                prow.append({**base, 'variant': variant, 'drugs_a': na_, 'drugs_b': nb_,
                             'cells': int((x.notna() & y.notna()).sum()), 'r': r, 'ci_lo': lo, 'ci_hi': hi,
                             'rel_a': ra, 'rel_b': rb, 'disattenuated': disatt(r, ra, rb, min(na_, nb_))})
            # shared-drug axes
            act_both = sorted(set(common) & set(va['act']) & set(vb['act']))
            for variant, drugs in [('shared_all', list(common)), ('shared_active_both', act_both)]:
                if len(drugs) < 4:
                    prow.append({**base, 'variant': variant, 'drugs_a': len(drugs), 'drugs_b': len(drugs)})
                    continue
                md = min_drugs(len(drugs))
                x = H.axis_of(Za.loc[drugs, cells], None, md)
                y = H.axis_of(Zb.loc[drugs, cells], None, md)
                r = H.corr(x, y)
                lo, hi = H.boot_ci(x, y, rng)
                ra = H.split_half(Za.loc[drugs, cells], rng)[0]
                rb = H.split_half(Zb.loc[drugs, cells], rng)[0]
                prow.append({**base, 'variant': variant, 'drugs_a': len(drugs), 'drugs_b': len(drugs),
                             'cells': int((x.notna() & y.notna()).sum()), 'r': r, 'ci_lo': lo, 'ci_hi': hi,
                             'rel_a': ra, 'rel_b': rb, 'disattenuated': disatt(r, ra, rb, len(drugs))})
            # per-drug agreement of shared drugs (raw z and selective residual), identical cells
            for d in common:
                za, zb = Za.loc[d, cells].values, Zb.loc[d, cells].values
                rra, rrb = va['R'].loc[d, cells].values, vb['R'].loc[d, cells].values
                m = np.isfinite(za) & np.isfinite(zb)
                drow.append({'endpoint': ep, 'a': a, 'b': b, 'drug': d, 'n': int(m.sum()), 'raw': H.corr(za, zb),
                             'selective': H.corr(rra, rrb), 'active_a': d in va['act'], 'active_b': d in vb['act']})
            print(ep, a, b, 'done', flush=True)
    pairs = pd.DataFrame(prow)
    pairs.to_csv(OUT / 'dreval_axis_pairs.csv', index=False)
    per = pd.DataFrame(drow)
    per.to_csv(OUT / 'dreval_perdrug_pairs.csv', index=False)
    pd.set_option('display.width', 250)
    print(pairs.round(3).to_string(index=False), flush=True)
    pdsum = per.groupby(['endpoint', 'a', 'b']).agg(drugs=('drug', 'size'), median_n=('n', 'median'),
                                                     median_raw=('raw', 'median'), median_sel=('selective', 'median')).reset_index()
    print(pdsum.round(3).to_string(index=False), flush=True)

    # ---------------- within dataset: fitters and summaries of the same raw data ----------------
    wrow = []
    for name in DATASETS:
        eps = [ep for ep in ENDPOINTS if (name, ep) in res]
        for e1, e2 in itertools.combinations(eps, 2):
            for variant in ['all', 'active']:
                k1, k2 = f'{name}|{e1}|{variant}', f'{name}|{e2}|{variant}'
                if k1 not in axes or k2 not in axes:
                    continue
                x, y = axes[k1], axes[k2]
                cells = x.dropna().index.intersection(y.dropna().index)
                r = H.corr(x[cells], y[cells])
                r1 = res[(name, e1)]['rel'][0] if variant == 'all' else res[(name, e1)]['rel_act'][0]
                r2_ = res[(name, e2)]['rel'][0] if variant == 'all' else res[(name, e2)]['rel_act'][0]
                # per-drug agreement of the two summaries of the same measurements
                Z1, Z2 = res[(name, e1)]['Z'], res[(name, e2)]['Z']
                dd = Z1.index.intersection(Z2.index)
                cc = Z1.columns.intersection(Z2.columns)
                pdr = [H.corr(Z1.loc[d, cc].values, Z2.loc[d, cc].values) for d in dd] if variant == 'all' else []
                # same raw measurements: errors are shared, so r is not disattenuated here (it can exceed 1)
                wrow.append({'dataset': name, 'endpoint_1': e1, 'endpoint_2': e2, 'variant': variant, 'cells': len(cells),
                             'r': r, 'rel_1': r1, 'rel_2': r2_,
                             'r2_1': float(res[(name, e1)]['r2'].median()), 'r2_2': float(res[(name, e2)]['r2'].median()),
                             'median_perdrug_r_same_drug': float(np.nanmedian(pdr)) if pdr else np.nan})
    wtab = pd.DataFrame(wrow)
    wtab.to_csv(OUT / 'dreval_within_screen.csv', index=False)
    print(wtab.round(3).to_string(index=False), flush=True)

    # ---------------- sanity: DrEval provider axes vs the paper's discovery axes (identity check of id mapping) ----
    nat = pd.read_csv(OUT / 'axes_native.csv', index_col=0)
    chk = []
    for disc, key in [('GDSC1|all', 'GDSC1|prov_AUC|all'), ('GDSC2|all', 'GDSC2|prov_AUC|all'), ('CTRP|all', 'CTRPv2|prov_AUC|all'),
                      ('CTRP|all', 'CTRPv2|CC_AUC|all'), ('GDSC1_lnIC50|all', 'GDSC1|prov_lnIC50|all'),
                      ('GDSC2_lnIC50|all', 'GDSC2|prov_lnIC50|all'), ('GDSC1|all', 'GDSC1|CC_lnIC50|all'),
                      ('GDSC2|all', 'GDSC2|CC_lnIC50|all'), ('GDSC1|all', 'GDSC1|CC_AUC|all'), ('GDSC2|all', 'GDSC2|CC_AUC|all'),
                      ('CCLE|all', 'CCLE|prov_AUC|all'), ('CCLE|all', 'CCLE|CC_AUC|all')]:
        if disc in nat.columns and key in Ad.columns:
            c = nat[disc].dropna().index.intersection(Ad[key].dropna().index)
            chk.append({'paper_axis': disc, 'dreval_axis': key, 'cells': len(c), 'r': H.corr(nat[disc][c], Ad[key][c])})
    chk = pd.DataFrame(chk)
    chk.to_csv(OUT / 'dreval_vs_paper_axes.csv', index=False)
    print(chk.round(3).to_string(index=False), flush=True)

    write_json(OUT / 'summary_46.json', {
        'datasets': dtab.to_dict(orient='records'), 'censoring': cens.to_dict(orient='records'),
        'pairs': pairs.to_dict(orient='records'), 'perdrug_summary': pdsum.to_dict(orient='records'),
        'within_screen': wtab.to_dict(orient='records'), 'paper_axis_check': chk.to_dict(orient='records'),
        'settings': {'min_axis_drugs': MIN_AXIS_DRUGS, 'seed': SEED, 'active_rule': '>=10% of lines CurveCurator AUC<0.8',
                     'decompose': 'selbench.decompose (>=30 cells, >=10 distinct values per drug)'}})


if __name__ == '__main__':
    main()
