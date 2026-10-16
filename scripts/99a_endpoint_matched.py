"""Round 3, task 1 (editor item 2; R1-M2, R2-M2c, R3-M3): endpoint-matched versus endpoint-switched cross-dataset
attributions on the DrEval leaderboard.

In round 2 the provider-AUC targets (GDSC1/GDSC2 Sanger AUC, CTRP v1 CTRP AUC, CCLE ActArea) were paired only with
the CTRP v2 CurveCurator ln IC50 reference, so their attributions mixed a change of endpoint with a change of
dataset. 80_paired_engine.py --parts leaderboard_matched adds the AUC-referenced pairings (CTRP v2 CurveCurator AUC ->
every provider-AUC target; CTRP v2 provider AUC -> every AUC-type target), on the same cells, models and two-way draws
(42's seed), and 81/82/83 score them like every other pair. This script classifies every leaderboard pair and
counts.

Classification of a pair (reference version of CTRP v2 -> target version):
  endpoint     'endpoint-matched' when both are ln IC50 or both are AUC-type (AUC, ActArea), else 'endpoint-switched'
  comparison   'cross-dataset' or 'within CTRP v2 (re-processing only)'
  processing   'same fitter' (CurveCurator both sides, or CTRP's provider fit both sides), 'different fitter',
               'harmonized' (66's shared panel / common range / censored ln IC50 versions)
  role         'primary matched' (ln IC50 targets with the ln IC50 reference; AUC-type targets with the CurveCurator
               AUC reference), 'harmonized', 'switched (round-2 pairing)', 'alternative reference (CTRP provider AUC)',
               'drug-matched variant' (@RLdrugs)
Statistics (pooled over the 21 published models; two-way bootstrap, B = 1,000; from 81, 82, 83):
  clear raw loss (lower 95% bound of the raw loss > 0), within share, share of the loss, excess (share of loss -
  within share) and the proportional-attenuation decision (rejected where the loss is clear and the excess interval
  excludes 0), R3's linear contrast C = (mbar_w tbar_c - tbar_w mbar_c) / rbar_w^2 (zero under proportional
  attenuation; no near-zero denominator), symmetric selective-both change, retained fraction and the one-sided
  non-inferiority test at 0.8 ('keeps >= 80%'), TOST at +-0.03 and +-0.05, whether the within selective-both
  interval includes 0 (ratio undefined there), and the leakage floor (Methods: largest median cross selective-both
  among the drug-agnostic predictors: omics-predicted axis, NaiveTissueMeanPredictor, NaiveMeanEffectsPredictor,
  MultiViewRandomForest; also with Lasso and ElasticNet, which are drug-agnostic within a split) against the
  published models' mean cross selective-both ('leakage-overlapping' when the floor is at least the models' value).
Also: the within-dataset axis share of the published models under the ln IC50 and AUC references (identity pairs on
all drugs, and the range over the cross-dataset pairs' drug sets).
Outputs (results/revision_round2/endpoint_matched/): pair_classification.csv, counts.json.
"""
import sys

sys.dont_write_bytecode = True
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

O2 = lib.OUT2
OUT = lib.out_dir('endpoint_matched')
FLOOR_MODELS = ['fitness_only', 'NaiveTissueMeanPredictor', 'NaiveMeanEffectsPredictor', 'MultiViewRandomForest']
FLOOR_MODELS_WIDE = FLOOR_MODELS + ['Lasso', 'ElasticNet']
HARM = {'shared panel, common range AUC', 'shared panel, own range AUC', 'shared panel ln IC50',
        'shared panel + range ln IC50'}


def etype(version):
    return 'lnIC50' if version.endswith('lnIC50') else 'AUC-type'


def proc(version):
    return version.split('|')[1]


def classify(r):
    src, tgt = r.source, r.target
    out = {'source_endpoint': etype(src), 'target_endpoint': etype(tgt)}
    out['endpoint'] = 'endpoint-matched' if out['source_endpoint'] == out['target_endpoint'] else 'endpoint-switched'
    out['comparison'] = ('cross-dataset' if src.split('|')[0] != tgt.split('|')[0]
                         else 'within CTRP v2 (re-processing only)')
    ps, pt = proc(src), proc(tgt)
    if r.family in HARM:
        out['processing'] = 'harmonized'
    elif ps == pt or (ps == 'CTRP' and pt == 'CTRP'):
        out['processing'] = 'same fitter'
    else:
        out['processing'] = 'different fitter'
    if '@RLdrugs' in r.pair:
        role = 'drug-matched variant'
    elif r.family in HARM:
        role = 'harmonized'
    elif r.family == 'CTRP provider AUC reference':
        role = 'alternative reference (CTRP provider AUC)'
    elif out['endpoint'] == 'endpoint-switched':
        role = 'switched (round-2 pairing)'
    else:
        role = 'primary matched'
    out['role'] = role
    out['in_round2_34'] = bool(r.k < 100 and '@RLdrugs' not in r.pair and out['comparison'] == 'cross-dataset'
                               and r.family in (HARM | {'lnIC50-reference', 'AUC-reference'}))
    return out


def leakage(per):
    """Per pair: largest median cross selective-both among the drug-agnostic predictors."""
    rows = []
    for pair, g in per.groupby('pair'):
        med = g.groupby('model').selB_c.median()
        f4 = med.reindex(FLOOR_MODELS).dropna()
        f6 = med.reindex(FLOOR_MODELS_WIDE).dropna()
        rows.append({'pair': pair, 'leakage_floor': float(f4.max()) if len(f4) else np.nan,
                     'leakage_floor_model': f4.idxmax() if len(f4) else '',
                     'leakage_floor_with_lasso_enet': float(f6.max()) if len(f6) else np.nan})
    return pd.DataFrame(rows)


def within_axis_share():
    """Published-21 within-dataset axis share (sum mediated / sum raw) under each CTRP v2 reference, identity pairs."""
    ix = pd.read_csv(O2 / 'paired' / 'pairs.csv')
    out = {}
    for ref in ['CTRPv2|CC|lnIC50', 'CTRPv2|CC|AUC']:
        r = ix[(ix.benchmark == 'leaderboard') & (ix.source == ref) & (ix.target == ref)].iloc[0]
        models, meta, S = lib.load_paired(O2 / 'paired' / 'arrays' / f'leaderboard__{r.k}.npz')
        gm = r.pool_main.split(';')
        mw = lib.pooled(S, models, gm, 'mediated', 'sum_w')
        rw = lib.pooled(S, models, gm, 'raw', 'sum_w')
        out[ref] = {**lib.summ(mw / rw, 'within_share'), 'drugs': int(r.drugs_used)}
    # CTRP v2 provider AUC: the within side of the provider-reference pairs (one pair is enough for all drugs: use
    # the CTRP v2 internal pair CTRPv2|CTRP|AUC -> CTRPv2|CC|AUC, which scores every drug)
    r = ix[(ix.benchmark == 'leaderboard') & (ix.source == 'CTRPv2|CTRP|AUC') & (ix.target == 'CTRPv2|CC|AUC')].iloc[0]
    models, meta, S = lib.load_paired(O2 / 'paired' / 'arrays' / f'leaderboard__{r.k}.npz')
    gm = r.pool_main.split(';')
    out['CTRPv2|CTRP|AUC'] = {**lib.summ(lib.pooled(S, models, gm, 'mediated', 'sum_w') /
                                         lib.pooled(S, models, gm, 'raw', 'sum_w'), 'within_share'),
                              'drugs': int(r.drugs_used), 'note': 'within side of CTRPv2|CTRP|AUC -> CTRPv2|CC|AUC'}
    return out


def fmt(x, lo=None, hi=None, nd=3):
    if x is None or not np.isfinite(x):
        return 'NA'
    if lo is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return f'{x:.{nd}f}'
    return f'{x:.{nd}f} ({lo:.{nd}f} to {hi:.{nd}f})'


def main():
    ix = pd.read_csv(O2 / 'paired' / 'pairs.csv')
    lb = ix[(ix.benchmark == 'leaderboard') & (ix.source != ix.target) & (ix.drugs_used > 0)].copy()
    A = pd.read_csv(O2 / 'attenuation' / 'attenuation_by_target.csv')
    A = A[(A.benchmark == 'leaderboard') & (A.group == 'published 21')]
    SB = pd.read_csv(O2 / 'symmetric_selboth' / 'selboth_by_target.csv')
    SB = SB[(SB.benchmark == 'leaderboard') & (SB.group == 'main')]
    EQ = pd.read_csv(O2 / 'equivalence' / 'equivalence_leaderboard_all.csv')
    EQ = EQ[EQ.selective_both == 'symmetric']
    per = pd.concat([pd.read_csv(O2 / 'paired' / f, usecols=['model', 'drug', 'pair', 'selB_c'], dtype={'drug': str})
                     for f in ['per_drug_leaderboard.csv.gz', 'per_drug_leaderboard_matched.csv.gz']],
                    ignore_index=True)
    LK = leakage(per)
    rows = []
    for _, r in lb.iterrows():
        a = A[A.pair == r.pair]
        s = SB[SB.pair == r.pair]
        e = EQ[EQ.pair == r.pair]
        if not (len(a) and len(s) and len(e)):
            print('missing', r.pair, len(a), len(s), len(e))
            continue
        a, s, e = a.iloc[0], s.iloc[0], e.iloc[0]
        row = {'k': int(r.k), 'pair': r.pair, 'reference': r.source, 'target': r.target, 'family': r.family,
               'drugs': int(r.drugs_used), **classify(r)}
        for k in ['raw_mean_within', 'raw_mean_cross', 'raw_loss', 'within_share', 'share_of_loss', 'excess',
                  'linear_contrast', 'retention_difference']:
            row[k], row[f'{k}_lo'], row[f'{k}_hi'] = a[k], a[f'{k}_lo'], a[f'{k}_hi']
        row['clear_raw_loss'] = bool(a.loss_clear)
        row['clear_raw_gain'] = bool(a.gain_clear)
        row['rejects_proportional'] = bool(a.rejects_proportional_excess)
        row['rejection_direction'] = ('axis lost disproportionately' if a.rejects_proportional_excess and a.excess > 0
                                      else 'selective term lost disproportionately' if a.rejects_proportional_excess
                                      else '')
        row['linear_contrast_excludes_0'] = bool(a.linear_ci_excludes_0)
        row['linear_direction'] = ('' if not a.linear_ci_excludes_0 else
                                   'axis lost disproportionately' if a.linear_contrast > 0 else
                                   'selective term lost disproportionately')
        row['reading'] = a.reading
        for k in ['within_sym', 'cross', 'change_sym']:
            row[f'selBOTH_{k}'], row[f'selBOTH_{k}_lo'], row[f'selBOTH_{k}_hi'] = s[k], s[f'{k}_lo'], s[f'{k}_hi']
        row['selBOTH_change_clear_loss'] = bool(s.change_sym_hi < 0)
        for k in ['retained_fraction', 'retained_fraction_lo', 'retained_fraction_hi', 'retained_fraction_lower95_onesided']:
            row[k] = e[k]
        row['keeps_80pct'] = bool(e['noninferior_f_0.8'])
        row['tost_0.03'] = bool(e['tost_equivalent_0.03'])
        row['tost_0.05'] = bool(e['tost_equivalent_0.05'])
        row['within_selBOTH_ci_includes_0'] = bool(e['within_ci95_includes_0'])
        lk = LK[LK.pair == r.pair]
        if len(lk):
            row['leakage_floor'] = float(lk.leakage_floor.iloc[0])
            row['leakage_floor_model'] = lk.leakage_floor_model.iloc[0]
            row['leakage_floor_with_lasso_enet'] = float(lk.leakage_floor_with_lasso_enet.iloc[0])
            row['leakage_overlapping'] = bool(row['leakage_floor'] >= row['selBOTH_cross'])
            row['leakage_overlapping_with_lasso_enet'] = bool(row['leakage_floor_with_lasso_enet'] >= row['selBOTH_cross'])
        rows.append(row)
    T = pd.DataFrame(rows)
    order = {'primary matched': 0, 'harmonized': 1, 'switched (round-2 pairing)': 2,
             'alternative reference (CTRP provider AUC)': 3, 'drug-matched variant': 4}
    T = T.sort_values(['comparison', 'role', 'target'], key=lambda c: c.map(order) if c.name == 'role' else c,
                      kind='stable')
    T.to_csv(OUT / 'pair_classification.csv', index=False, float_format='%.5g')

    # ---------------- counts
    cross = T[T.comparison == 'cross-dataset']

    def attribution_counts(g):
        c = g[g.clear_raw_loss]
        return {'versions': int(len(g)), 'clear_raw_loss': int(len(c)),
                'clear_loss_pairs': list(c.pair),
                'reject_axis_disproportionate': int((c.rejects_proportional & (c.excess > 0)).sum()),
                'reject_selective_disproportionate': int((c.rejects_proportional & (c.excess < 0)).sum()),
                'not_rejected': int((~c.rejects_proportional).sum()),
                'rejecting_pairs': {p: fmt(x, lo, hi) for p, x, lo, hi in
                                    zip(c[c.rejects_proportional].pair, c[c.rejects_proportional].excess,
                                        c[c.rejects_proportional].excess_lo, c[c.rejects_proportional].excess_hi)},
                'linear_contrast_clear_loss_axis': int((c.linear_contrast_excludes_0 & (c.linear_contrast > 0)).sum()),
                'linear_contrast_clear_loss_selective': int((c.linear_contrast_excludes_0 & (c.linear_contrast < 0)).sum()),
                'linear_contrast_all_versions_axis': int((g.linear_contrast_excludes_0 & (g.linear_contrast > 0)).sum()),
                'linear_contrast_all_versions_selective': int((g.linear_contrast_excludes_0 & (g.linear_contrast < 0)).sum())}

    def equiv_counts(g):
        keep = g[g.keeps_80pct]
        clean = keep[~keep.leakage_overlapping & ~keep.within_selBOTH_ci_includes_0]
        return {'versions': int(len(g)), 'keeps_80pct': int(len(keep)), 'keeps_80pct_pairs': list(keep.pair),
                'keeps_80pct_leakage_overlapping': int(keep.leakage_overlapping.sum()),
                'keeps_80pct_within_ci_includes_0': int(keep.within_selBOTH_ci_includes_0.sum()),
                'keeps_80pct_neither_flag': int(len(clean)), 'keeps_80pct_neither_flag_pairs': list(clean.pair),
                'tost_0.03': int(g['tost_0.03'].sum()), 'tost_0.03_pairs': list(g[g['tost_0.03']].pair),
                'tost_0.05': int(g['tost_0.05'].sum()),
                'selBOTH_clear_loss': int(g.selBOTH_change_clear_loss.sum()),
                'versions_leakage_overlapping': int(g.leakage_overlapping.sum()),
                'versions_within_ci_includes_0': int(g.within_selBOTH_ci_includes_0.sum())}
    sets = {
        'matched_primary_unharmonized': cross[cross.role == 'primary matched'],
        'matched_primary_unharmonized_lnIC50': cross[(cross.role == 'primary matched') & (cross.target_endpoint == 'lnIC50')],
        'matched_primary_unharmonized_AUC': cross[(cross.role == 'primary matched') & (cross.target_endpoint == 'AUC-type')],
        'matched_primary_unharmonized_round2_only': cross[(cross.role == 'primary matched') & cross.in_round2_34],
        'harmonized': cross[cross.role == 'harmonized'],
        'matched_primary_plus_harmonized': cross[cross.role.isin(['primary matched', 'harmonized'])],
        'matched_round2_26': cross[cross.in_round2_34 & (cross.endpoint == 'endpoint-matched')],
        'switched_round2_pairings': cross[cross.role == 'switched (round-2 pairing)'],
        'round2_34': cross[cross.in_round2_34],
        'alternative_CTRP_provider_reference': cross[cross.role == 'alternative reference (CTRP provider AUC)'],
        'within_CTRPv2_reprocessing': T[T.comparison != 'cross-dataset'],
    }
    counts = {'attribution': {k: attribution_counts(g) for k, g in sets.items()},
              'equivalence': {k: equiv_counts(g) for k, g in sets.items()}}
    # within-dataset axis share under each reference
    ws = within_axis_share()
    rng = {}
    for ref in ['CTRPv2|CC|lnIC50', 'CTRPv2|CC|AUC', 'CTRPv2|CTRP|AUC']:
        g = cross[(cross.reference == ref) & (cross.role != 'drug-matched variant')]
        rng[ref] = {'min': float(g.within_share.min()), 'max': float(g.within_share.max()), 'versions': int(len(g))}
    counts['within_axis_share'] = {'identity_all_drugs': ws, 'range_over_cross_pairs': rng}
    # R2's synthesis in one line: the axis is lost when predictions trained on an axis-heavy endpoint (ln IC50) are
    # scored against an axis-light one (AUC)
    sw = sets['switched_round2_pairings']
    sm = sets['matched_primary_unharmonized']
    counts['synthesis'] = {
        'switched_clear_loss_rejecting_axis': f"{attribution_counts(sw)['reject_axis_disproportionate']} of "
                                              f"{attribution_counts(sw)['clear_raw_loss']}",
        'matched_clear_loss_rejecting_axis': f"{attribution_counts(sm)['reject_axis_disproportionate']} of "
                                             f"{attribution_counts(sm)['clear_raw_loss']}"}
    lib.write_json(OUT / 'counts.json', counts)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_rows', 300)
    cols = ['pair', 'endpoint', 'comparison', 'role', 'raw_loss', 'raw_loss_lo', 'raw_loss_hi', 'within_share',
            'share_of_loss', 'excess', 'excess_lo', 'excess_hi', 'rejection_direction', 'linear_contrast',
            'linear_contrast_lo', 'linear_contrast_hi', 'selBOTH_change_sym', 'retained_fraction', 'keeps_80pct',
            'tost_0.03', 'leakage_floor', 'selBOTH_cross', 'leakage_overlapping']
    print(T[cols].round(3).to_string(index=False))
    import json
    print(json.dumps(counts, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
