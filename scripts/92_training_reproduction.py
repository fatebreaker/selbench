"""Round 2 (WS6b), task 3 (editor item 4, R1-M2): does this study's drevalpy pipeline (58) reproduce DrEval's released
CTRP v2 ln IC50 leaderboard predictions when it retrains the models itself?

Compared, on DrEval's published leave-cell-line-out splits (58 reads data/dreval/splits.zip for CTRPv2_LNIC50):
  EnsembleMF     retrained in all 10 splits (results/revision_retrain/drevalpy_runs/CTRPv2_LNIC50/CTRPv2/LCO/EnsembleMF)
                 vs the released leaderboard predictions (data/dreval/leaderboard_Oct26/LCO/EnsembleMF);
  RandomForest   retrained in the splits available (planned: 0-2; grid fixed at max_depth = 30, the value DrEval's
                 tuning chose in 10/10 leaderboard splits) vs the released predictions on the same splits' cells.
For each pair (retrained, released):
  split check    every test cell's split agrees between the two runs;
  per drug       Pearson and Spearman correlation between retrained and released predictions over the drug's test
                 cells (ln IC50 scale), and the RMS difference relative to the released predictions' SD; as a yardstick
                 the same agreement between two different released models (EnsembleMF vs RandomForest) on the same cells;
  scores         per-drug raw r(p, z), axis term r_pg r_zg, selective term (raw - axis term) and selective partial
                 r(p, z | g) against CTRP v2 CurveCurator ln IC50 (rank-normal z, leave-drug-out axis g from all
                 leaderboard test cells, as 59); both predictions scored on identical cells per drug; means over drugs,
                 axis share, medians, and retrained - released differences with two-way (cell x drug) bootstrap
                 intervals (B = 1,000, one draw shared by both); across-drug correlation of each per-drug score;
  DrEval metric  DrEval's normalized Pearson (per split, all test pairs, centred on the released
                 NaiveMeanEffectsPredictor; mean over splits; 59's implementation) for both.
Outputs (results/revision_round2/training_reproduction/): per_drug.csv.gz, agreement_summary.csv,
score_summary.csv, normalized_pearson.csv, summary.json.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_spec = importlib.util.spec_from_file_location('lib77', Path(__file__).resolve().parent / '77_round2_lib.py')
lib = importlib.util.module_from_spec(_spec)
sys.modules['lib77'] = lib
_spec.loader.exec_module(lib)

OUT = lib.out_dir('training_reproduction')
SEED = 20261092
RUN = lib.RES / 'revision_retrain' / 'drevalpy_runs' / 'CTRPv2_LNIC50' / 'CTRPv2' / 'LCO'


def read_model(md, cmap, rt):
    fs = sorted((md / 'predictions').glob('predictions_split_*.csv'))
    if not fs:
        return None
    d = rt.read_split_files(fs, cmap)
    return d


def oof(d):
    g = d.groupby(['drug', 'cell'], as_index=False).agg(pred=('predictions', 'mean'), split=('split', 'first'))
    return g


def main():
    t0 = time.time()
    rt = lib.load_module('rt59', '59_retrain_score.py')
    cmap = rt.cell_map('CTRPv2')
    resp = rt.responses()
    obs = resp[(resp.dataset == 'CTRPv2') & (resp.endpoint == 'lnic50')][['drug', 'cell', 'response']]
    LBD = rt.LB / 'LCO'
    raw = {}
    for name, md in [('retrained:EnsembleMF', RUN / 'EnsembleMF'), ('released:EnsembleMF', LBD / 'EnsembleMF'),
                     ('retrained:RandomForest', RUN / 'RandomForest'), ('released:RandomForest', LBD / 'RandomForest'),
                     ('released:NaiveMeanEffectsPredictor', LBD / 'NaiveMeanEffectsPredictor')]:
        d = read_model(md, cmap, rt)
        if d is not None:
            raw[name] = d
            print(name, 'splits', sorted(d.split.unique()), 'rows', len(d), flush=True)
    # decomposition: CTRP v2 ln IC50 on every released leaderboard test cell
    lb_cells = sorted(set(raw['released:EnsembleMF'].cell))
    Z, G, _, _ = lib.selbench.decompose(obs[obs.cell.isin(lb_cells)])
    agg = {k: oof(v) for k, v in raw.items()}
    comps = []
    for model in ['EnsembleMF', 'RandomForest']:
        a, b = f'retrained:{model}', f'released:{model}'
        if a not in agg:
            continue
        comps.append((model, a, b))
    # yardstick: two different released models on the same cells (EnsembleMF vs RandomForest)
    comps.append(('yardstick: released EnsembleMF vs released RandomForest', 'released:EnsembleMF',
                  'released:RandomForest'))
    per_rows, agr_rows, score_rows, norm_rows = [], [], [], []
    info = {}
    for k, (label, a, b) in enumerate(comps):
        A, Bm = agg[a], agg[b]
        splits = sorted(set(A.split) & set(Bm.split))
        if label.startswith('yardstick') and 'retrained:RandomForest' in agg:
            splits = sorted(set(agg['retrained:RandomForest'].split))     # same cells as the RF comparison
        A, Bm = A[A.split.isin(splits)], Bm[Bm.split.isin(splits)]
        # split agreement per cell
        sa, sb = A.groupby('cell').split.first(), Bm.groupby('cell').split.first()
        common_cells = sa.index.intersection(sb.index)
        split_agree = float((sa[common_cells] == sb[common_cells]).mean())
        cells_only_a, cells_only_b = len(set(sa.index) - set(sb.index)), len(set(sb.index) - set(sa.index))
        M = A.merge(Bm, on=['drug', 'cell'], suffixes=('_a', '_b'))
        rows = []
        for d, g in M.groupby('drug'):
            if len(g) < lib.MIN_CELLS or g.pred_a.std() < 1e-12 or g.pred_b.std() < 1e-12:
                continue
            rows.append({'comparison': label, 'drug': d, 'n_cells': len(g),
                         'pearson_pred': float(np.corrcoef(g.pred_a, g.pred_b)[0, 1]),
                         'spearman_pred': float(spearmanr(g.pred_a, g.pred_b).correlation),
                         'rms_diff_rel_sd': float(np.sqrt(np.mean((g.pred_a - g.pred_b) ** 2)) / g.pred_b.std()),
                         'mean_diff': float((g.pred_a - g.pred_b).mean())})
        P = pd.DataFrame(rows)
        agr_rows.append({'comparison': label, 'model_a': a, 'model_b': b, 'splits': ';'.join(map(str, splits)),
                         'cells_common': len(common_cells), 'cells_only_a': cells_only_a, 'cells_only_b': cells_only_b,
                         'split_assignment_agreement': split_agree, 'pairs_common': len(M),
                         'drugs': len(P), 'pearson_all_pairs': float(np.corrcoef(M.pred_a, M.pred_b)[0, 1]),
                         'median_drug_pearson': float(P.pearson_pred.median()),
                         'q10_drug_pearson': float(P.pearson_pred.quantile(0.1)),
                         'q90_drug_pearson': float(P.pearson_pred.quantile(0.9)),
                         'min_drug_pearson': float(P.pearson_pred.min()),
                         'frac_drugs_pearson_ge_0.9': float((P.pearson_pred >= 0.9).mean()),
                         'frac_drugs_pearson_ge_0.8': float((P.pearson_pred >= 0.8).mean()),
                         'median_drug_spearman': float(P.spearman_pred.median()),
                         'median_rms_diff_rel_sd': float(P.rms_diff_rel_sd.median())})
        # scores on identical cells per drug
        cells = sorted(set(M.cell) & set(Z.columns))
        Pa = M.pivot_table(index='drug', columns='cell', values='pred_a').reindex(columns=cells)
        Pb = M.pivot_table(index='drug', columns='cell', values='pred_b').reindex(columns=cells)
        drugs = sorted(set(Pa.index) & set(Z.index))
        Zs, Gs = Z.reindex(index=drugs, columns=cells), G.reindex(index=drugs, columns=cells)
        # sensitivity orientation (higher = more sensitive) as 59: prediction = -ln IC50
        Pm = {a: -Pa.reindex(index=drugs), b: -Pb.reindex(index=drugs)}
        W, V = lib.draws(len(cells), len(drugs), SEED + k)
        arr, models, dl, nc = lib.within_engine(Zs, Gs, Pm, W, V, common_mask=True)
        ok = np.all([np.isfinite(arr['raw'][0, i]) for i in range(2)], axis=0)
        okb = np.broadcast_to(ok, (W.shape[0], len(dl)))
        Vm = np.where(okb, V, 0.0)
        st = {}
        for i, m in enumerate(models):
            raw_, med = arr['raw'][:, i].astype(float), (arr['r_pg'][:, i] * arr['r_zg'][:, i]).astype(float)
            sel = arr['sel'][:, i].astype(float)
            n = Vm.sum(1)
            f = lambda X: (Vm * np.where(okb, X, 0.0)).sum(1)  # noqa: E731
            st[m] = {'mean_raw': f(raw_) / n, 'mean_axis_term': f(med) / n, 'mean_selective_term': f(raw_ - med) / n,
                     'mean_selective_partial': f(sel) / n, 'axis_share': f(med) / f(raw_),
                     'median_raw': lib.wmedian_rows(np.where(okb, raw_, np.nan), V),
                     'median_selective_partial': lib.wmedian_rows(np.where(okb, sel, np.nan), V)}
            for j, d in enumerate(dl):
                if ok[j]:
                    per_rows.append({'comparison': label, 'model': m, 'drug': d, 'n_cells': int(nc[i, j]),
                                     'raw': float(raw_[0, j]), 'axis_term': float(med[0, j]),
                                     'selective_term': float(raw_[0, j] - med[0, j]), 'selective_partial': float(sel[0, j]),
                                     'r_pg': float(arr['r_pg'][0, i, j]), 'r_zg': float(arr['r_zg'][0, i, j])})
        for key in st[a]:
            row = {'comparison': label, 'statistic': key, 'drugs': int(ok.sum()), 'cells': len(cells)}
            row.update(lib.summ(st[a][key], 'a'))
            row.update(lib.summ(st[b][key], 'b'))
            row.update(lib.summ(st[a][key] - st[b][key], 'a_minus_b'))
            # across-drug agreement of the per-drug score
            if key in ('mean_raw', 'mean_axis_term', 'mean_selective_term', 'mean_selective_partial'):
                X = {'mean_raw': arr['raw'], 'mean_selective_partial': arr['sel'],
                     'mean_axis_term': arr['r_pg'] * arr['r_zg'],
                     'mean_selective_term': arr['raw'] - arr['r_pg'] * arr['r_zg']}[key][0]
                row['across_drug_pearson'] = float(np.corrcoef(X[0][ok], X[1][ok])[0, 1])
                row['median_abs_drug_diff'] = float(np.median(np.abs(X[0][ok] - X[1][ok])))
            score_rows.append(row)
        info[label] = {'splits': splits, 'drugs_scored': int(ok.sum()), 'cells': len(cells)}
        # DrEval normalized Pearson (59): centred on the released NaiveMeanEffectsPredictor, per split
        nv = raw['released:NaiveMeanEffectsPredictor'][['drug', 'cell', 'predictions', 'split']].rename(
            columns={'predictions': 'naive'})
        for m in (a, b):
            x = raw[m][['drug', 'cell', 'predictions', 'response', 'split']].merge(nv, on=['drug', 'cell', 'split'])
            x = x[x.split.isin(splits)]
            per = {s: np.corrcoef(g.response - g.naive, g.predictions - g.naive)[0, 1] for s, g in x.groupby('split')}
            norm_rows.append({'comparison': label, 'model': m, 'splits': len(per),
                              'normalized_pearson_mean': float(np.mean(list(per.values()))),
                              **{f'split_{s}': float(v) for s, v in per.items()}})
        print(label, agr_rows[-1]['median_drug_pearson'], f'{time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(per_rows).to_csv(OUT / 'per_drug.csv.gz', index=False, float_format='%.6g')
    pd.DataFrame(agr_rows).to_csv(OUT / 'agreement_summary.csv', index=False, float_format='%.5g')
    S = pd.DataFrame(score_rows)
    S.to_csv(OUT / 'score_summary.csv', index=False, float_format='%.5g')
    N = pd.DataFrame(norm_rows)
    ev = pd.read_csv(rt.LB / 'evaluation_results.csv')
    ev['model'] = ev.algorithm.str.replace(r' \(.*\)$', '', regex=True)
    rep = ev.groupby('model')['Pearson: normalized'].mean()
    N['dreval_reported_all_splits'] = [rep.get(m.split(':', 1)[1], np.nan) for m in N.model]
    N.to_csv(OUT / 'normalized_pearson.csv', index=False, float_format='%.5g')
    lib.write_json(OUT / 'summary.json', {'seconds': round(time.time() - t0), 'B': lib.B, 'info': info,
                                          'available_splits': {k: sorted(map(int, v.split.unique())) for k, v in raw.items()}})
    pd.set_option('display.width', 250)
    print(pd.DataFrame(agr_rows).round(3).to_string(index=False))
    print(S[['comparison', 'statistic', 'a', 'b', 'a_minus_b', 'a_minus_b_lo', 'a_minus_b_hi', 'across_drug_pearson']]
          .round(3).to_string(index=False))
    print(N.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
