"""Harmonized-range AUC: one curve fitter and one concentration range on both sides of the paired design
(editor item 1; R1-M1 "an AUC/AAC recomputed on a harmonised concentration range").

DrEval's curated files carry, for every measurement, CurveCurator's fitted four-parameter log-logistic curve
(pEC50, slope, front, back) and the tested dose range. Their AUC_curvecurator is the mean fitted viability over the
tested log10-dose range; the reconstruction below reproduces it to 1e-9 (checked on load).

For each target dataset T (CTRPv1, GDSC1, GDSC2, CCLE) and each drug shared with CTRPv2, the common range is
[max(lo_S, lo_T), min(hi_S, hi_T)], where lo and hi are the per-drug median log10 minimum and maximum tested doses in
each dataset. Drugs whose common range spans < 1 log10 unit are dropped. Every measurement of a retained drug, in
CTRPv2 and in T, is summarized by its fitted curve's mean viability over that common range (sensitivity = -value).
This gives two new versions per target, 'CTRPv2|HR-<T>|AUC' and '<T>|HR|AUC', which differ only in the measurements.

The paired scoring, two-way bootstrap (same seed, cells and drug universe as 42_crosslab_scores.py, so the draws are
identical) and summaries reuse scripts 42 and 43 unchanged. Outputs: results/revision_crosslab/harmonized_range/.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # so that forked workers can unpickle functions of this module by reference
    spec.loader.exec_module(mod)
    return mod


sc = load('sc42', '42_crosslab_scores.py')
sm = load('sm43', '43_crosslab_summaries.py')
cd = sc.cd
OUT = cd.OUT / 'harmonized_range'
TARGETS = ['CTRPv1', 'GDSC1', 'GDSC2', 'CCLE']
MIN_WIDTH = 1.0          # log10 units of common concentration range
GRID = np.linspace(0, 1, 101)


def curves(ds):
    cols = ['cellosaurus_id', 'pubchem_id', 'pEC50_curvecurator', 'Slope', 'Front', 'Back', 'AUC_curvecurator',
            'min_dose_M', 'max_dose_M', 'LN_IC50_curvecurator']
    with zipfile.ZipFile(cd.DRE / f'{ds}.zip') as z:
        d = pd.read_csv(z.open(f'{ds}/{ds}.csv'), usecols=cols, dtype={'pubchem_id': str}, low_memory=False)
    d = d.dropna(subset=[c for c in cols if c != 'LN_IC50_curvecurator'])
    d = d[(d.min_dose_M > 0) & (d.max_dose_M > d.min_dose_M)]
    d['lo'], d['hi'] = np.log10(d.min_dose_M), np.log10(d.max_dose_M)
    chk = d.sample(min(5000, len(d)), random_state=1)
    rec = mean_viability(chk, chk.lo.values, chk.hi.values)
    err = np.abs(rec - chk.AUC_curvecurator.values)
    q99 = float(np.nanpercentile(err, 99))
    assert np.nanmedian(err) < 1e-4 and q99 < 1e-2, (ds, np.nanmedian(err), q99)  # stored precision; step-like curves
    print(ds, 'AUC reconstruction: median abs error', f'{np.nanmedian(err):.1e}', '99th percentile', f'{q99:.1e}',
          'max', f'{np.nanmax(err):.1e}', flush=True)
    return d


def mean_viability(d, lo, hi):
    """Mean of the fitted curve f(x) = (front - back) / (1 + 10^(slope (x + pEC50))) + back over [lo, hi] (log10 M)."""
    X = lo[:, None] + (hi - lo)[:, None] * GRID[None, :]
    pec, s, f, b = (d[c].values[:, None] for c in ['pEC50_curvecurator', 'Slope', 'Front', 'Back'])
    with np.errstate(over='ignore'):
        Y = (f - b) / (1 + 10 ** (s * (X + pec))) + b
    return (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(Y, GRID, axis=1)


def build_versions():
    S = curves('CTRPv2')
    rS = S.groupby('pubchem_id')[['lo', 'hi']].median()
    versions, ranges = {}, []
    for T in TARGETS:
        D = curves(T)
        rT = D.groupby('pubchem_id')[['lo', 'hi']].median()
        shared = rS.index.intersection(rT.index)
        lo = np.maximum(rS.loc[shared, 'lo'], rT.loc[shared, 'lo'])
        hi = np.minimum(rS.loc[shared, 'hi'], rT.loc[shared, 'hi'])
        rng = pd.DataFrame({'lo': lo, 'hi': hi})
        rng['width'] = rng.hi - rng.lo
        keep = rng[rng.width >= MIN_WIDTH]
        for side, frame, key in [('S', S, f'CTRPv2|HR-{T}|AUC'), ('T', D, f'{T}|HR|AUC')]:
            x = frame[frame.pubchem_id.isin(keep.index)].copy()
            x['v'] = mean_viability(x, keep.loc[x.pubchem_id, 'lo'].values, keep.loc[x.pubchem_id, 'hi'].values)
            x = x.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).v.mean()
            versions[key] = pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values,
                                          'response': -x.v.values})
        # shared drug panel, CurveCurator ln IC50 (the leaderboard's endpoint; censored values stay missing)
        for side, frame, key in [('S', S, f'CTRPv2|SL-{T}|lnIC50'), ('T', D, f'{T}|SL|lnIC50')]:
            x = frame[frame.pubchem_id.isin(keep.index)].dropna(subset=['LN_IC50_curvecurator'])
            x = x.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).LN_IC50_curvecurator.mean()
            versions[key] = pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values,
                                          'response': -x.LN_IC50_curvecurator.values})
        # shared drug panel, ln IC50 censored to the common concentration range (equal censoring rule on both sides)
        for side, frame, key in [('S', S, f'CTRPv2|RL-{T}|lnIC50'), ('T', D, f'{T}|RL|lnIC50')]:
            x = frame[frame.pubchem_id.isin(keep.index)].dropna(subset=['LN_IC50_curvecurator']).copy()
            l10 = x.LN_IC50_curvecurator.values / np.log(10) - 6     # LN_IC50 is ln(micromolar); log10 molar IC50
            lo_, hi_ = keep.loc[x.pubchem_id, 'lo'].values, keep.loc[x.pubchem_id, 'hi'].values
            x = x[(l10 >= lo_) & (l10 <= hi_)]
            x = x.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).LN_IC50_curvecurator.mean()
            versions[key] = pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values,
                                          'response': -x.LN_IC50_curvecurator.values})
        # shared drug panel only, each dataset's own tested range (separates the panel from the concentration range)
        for side, frame, key in [('S', S, f'CTRPv2|SP-{T}|AUC'), ('T', D, f'{T}|SP|AUC')]:
            x = frame[frame.pubchem_id.isin(keep.index)]
            x = x.groupby(['pubchem_id', 'cellosaurus_id'], as_index=False).AUC_curvecurator.mean()
            versions[key] = pd.DataFrame({'drug': x.pubchem_id.values, 'cell': x.cellosaurus_id.values,
                                          'response': -x.AUC_curvecurator.values})
        for d_, r in rng.iterrows():
            ranges.append({'target': T, 'drug': d_, 'lo_common': r.lo, 'hi_common': r.hi, 'width': r.width,
                           'lo_ctrpv2': rS.loc[d_, 'lo'], 'hi_ctrpv2': rS.loc[d_, 'hi'], 'lo_target': rT.loc[d_, 'lo'],
                           'hi_target': rT.loc[d_, 'hi'], 'kept': bool(r.width >= MIN_WIDTH)})
        print(T, 'shared', len(shared), 'kept', len(keep), flush=True)
    return versions, pd.DataFrame(ranges)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    hv, ranges = build_versions()
    ranges.to_csv(OUT / 'ranges.csv', index=False)
    cls = pd.read_csv(cd.OUT / 'drug_classes.csv', dtype={'drug': str})
    strict = cls.set_index('drug').lco_class_strict.dropna()
    ext = cls.set_index('drug').lco_class_extended.dropna()
    tissue = pd.read_csv(cd.OUT / 'cell_tissue.csv').set_index('cellosaurus_id').tissue
    act = pd.read_csv(cd.OUT / 'drug_activity_ctrpv2.csv', dtype={'drug': str})
    ag = pd.read_csv(cd.OUT / 'drug_agnostic_check.csv')
    V = cd.versions()
    base = {k: V[k] for k in [cd.REFERENCE, cd.AUC_REFERENCE]}   # only to rebuild the identical drug universe
    dec = {}
    for v, long in list(base.items()) + list(hv.items()):
        Z, Gm, _, r2 = sc.selbench.decompose(long)
        GL, _ = sc.leave_class_out(Z, Gm, strict)
        GX, _ = sc.leave_class_out(Z, Gm, ext)
        dec[v] = {'Z': Z, 'G': Gm, 'GL': GL, 'GX': GX}
        print(v, Z.shape, f'median axis share {float(r2.median()):.3f}', flush=True)
    P, mclass, lb = cd.predictions()
    cells = None
    for M_ in P.values():
        cells = M_.columns if cells is None else cells.intersection(M_.columns)
    U = np.array(sorted(cells))
    P = {m: M_.reindex(columns=U) for m, M_ in P.items()}
    axes = {m: sc.model_axis(M_.loc[M_.notna().sum(axis=1) >= sc.MIN_CELLS]) for m, M_ in P.items()}
    refs = [f'CTRPv2|{fam}-{T}|AUC' for T in TARGETS for fam in ('HR', 'SP')] + [f'CTRPv2|{f}-{T}|lnIC50' for T in TARGETS for f in ('SL', 'RL')]
    pseudo_axes = {}
    for ref in refs:
        Zr, Gr = dec[ref]['Z'].reindex(columns=U), dec[ref]['G'].reindex(columns=U)
        pseudo_axes[ref] = {sc.MEASURED: sc.model_axis(Zr), sc.FITNESS: sc.model_axis(Gr)}
    all_drugs = sorted(dec[cd.REFERENCE]['Z'].index.union(dec[cd.AUC_REFERENCE]['Z'].index))
    rng = np.random.default_rng(sc.SEED)
    B = 1000
    Wc = np.vstack([np.ones(len(U)), rng.multinomial(len(U), np.full(len(U), 1 / len(U)), size=B)]).astype(float)
    Vd = np.vstack([np.ones(len(all_drugs)), rng.multinomial(len(all_drugs), np.full(len(all_drugs), 1 / len(all_drugs)),
                                                              size=B)]).astype(float)
    published22 = [m for m in P if mclass[m] in ('classical', 'neural/factorization')]
    sc.G.update({'dec': dec, 'cells': U, 'P': P, 'axes': axes, 'pseudo_axes': pseudo_axes, 'common': {},
                 'tissue': tissue, 'W': Wc, 'V': Vd, 'drug_ix': {d: i for i, d in enumerate(all_drugs)},
                 'active': set(act[act.active_ctrpv2].drug), 'models': list(P), 'leaderboard': set(lb),
                 'published22': published22, 'published21': [m for m in published22 if m != 'SVR'],
                 'agnostic': set(ag[ag.exactly_drug_agnostic].model) | {sc.FITNESS}})
    pairs = [{'pair': f'CTRPv2|{fam}-{T}|AUC->{T}|{fam}|AUC', 'reference': f'CTRPv2|{fam}-{T}|AUC',
              'target': f'{T}|{fam}|AUC', 'cellset': 'own',
              'family': {'HR': 'shared panel, common range AUC', 'SP': 'shared panel, own range AUC'}[fam]}
             for T in TARGETS for fam in ('HR', 'SP')]
    pairs += [{'pair': f'CTRPv2|{f}-{T}|lnIC50->{T}|{f}|lnIC50', 'reference': f'CTRPv2|{f}-{T}|lnIC50',
               'target': f'{T}|{f}|lnIC50', 'cellset': 'own',
               'family': {'SL': 'shared panel ln IC50', 'RL': 'shared panel + range ln IC50'}[f]}
              for T in TARGETS for f in ('SL', 'RL')]
    per, pooled, medmed, msum, comp, info = [], [], [], [], [], []
    with ProcessPoolExecutor(6, mp_context=mp.get_context('fork')) as ex:
        for r in ex.map(sc.run_pair, pairs):
            per.append(r[0]); pooled += r[1]; medmed += r[2]; msum += r[3]; comp += r[4]; info.append(r[5])
            print(r[5], f'{time.time() - t0:.0f}s', flush=True)
    per = pd.concat(per, ignore_index=True)
    per['class'] = per.model.map(lambda m: mclass.get(m, 'reference'))
    per.to_csv(OUT / 'per_drug_scores.csv.gz', index=False, float_format='%.6g')
    pd.DataFrame(msum).to_csv(OUT / 'model_medians.csv', index=False)
    pd.DataFrame(comp).to_csv(OUT / 'paired_comparisons.csv', index=False)
    info = pd.DataFrame(info)
    info['dataset'] = info.target.str.split('|').str[0]
    info['laboratory'] = info.dataset.map(cd.LAB)
    info['processing'] = info.target.str.split('|').str[1].map({'HR': 'CC common range', 'SP': 'CC own range',
                                                                 'SL': 'CC ln IC50 shared panel',
                                                                 'RL': 'CC ln IC50 shared panel + range'})
    info['endpoint'] = np.where(info.target.str.endswith('lnIC50'), 'lnIC50', 'AUC')
    info['identity'] = False
    info.to_csv(OUT / 'pair_info.csv', index=False)
    R = {(n, s, st, k): v for n, s, st, k, v in pooled}
    MM = {(n, s, st): v for n, s, st, v in medmed}
    S = sm.summary_by_target(info, R, MM)
    S.to_csv(OUT / 'summary_by_target.csv', index=False)
    sm.tost(info, R).to_csv(OUT / 'tost_selectivity.csv', index=False)
    ms = pd.DataFrame(msum)
    fl = ms[(ms.stat == 'selB_c') & ms.model.isin(list(sm.AGNOSTIC)) & (ms.subset == 'all')]
    fl.to_csv(OUT / 'leakage_floors.csv', index=False)
    cols = ['pair', 'drugs', 'raw_median_within', 'raw_median_cross', 'mediated_retained', 'mediated_retained_lo',
            'mediated_retained_hi', 'alignment_ratio_r_pg', 'target_loading_ratio_r_zg', 'd_selBOTH', 'd_selBOTH_lo',
            'd_selBOTH_hi', 'share_of_loss_through_fitness']
    print(S[S.subset == 'all'][cols].round(3).to_string())
    print('floors', fl.groupby('pair')['median'].max().round(3).to_dict())
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
