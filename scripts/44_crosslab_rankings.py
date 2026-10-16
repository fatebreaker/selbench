"""Revision workstream: leaderboard rankings under raw, DrEval-normalized and selectivity-aware metrics, with
critical-difference uncertainty (R1-M7, R3 minor 5, editor item 7).

The 22 published leaderboard models (classical + neural/factorization, SVR included) are re-ranked on the drugs that
every one of them predicts (PharmaFormer, Precily and PaccMann cover 466 drugs and DrugGNN 441, so medians over each
model's own drugs are not comparable). Rankings (1 = best), from per-drug point estimates of 42_crosslab_scores.py:

  raw_within        median per-drug Pearson within CTRPv2 (leave-cell-line-out)
  dreval_normalized DrEval's normalized Pearson as reported in results/dreval/normalized_metric.csv, and re-computed
                    per CV split on the common drug set (36_dreval_normalized.py's estimator, validated here)
  selT_within       median selective-T within CTRPv2
  adjusted_within   median model-axis-adjusted score within CTRPv2 (diagnostic; exactly drug-agnostic models score 0)
  selBOTH_<target>  median selective-BOTH in GDSC1, GDSC2 and CTRPv1 (CurveCurator ln IC50, matched cells); the
                    cross-laboratory ranking is the mean of the three ranks
  adjusted_<target> median model-axis-adjusted score in the same targets

Critical differences: Friedman test (tie-corrected) with blocks = drugs (per-drug metrics) or the 10 CV splits
(DrEval normalized metric), Nemenyi critical difference at alpha = 0.05 (studentized range / sqrt 2) and Conover's
pairwise t-test (Holm-adjusted). A move between two rankings is called significant when the model's order relative
to another model reverses and the mean-rank difference exceeds the critical difference under both metrics.
Outputs (results/revision_crosslab/): rankings.csv, ranking_correlations.csv, ranking_moves.csv,
critical_difference.csv, critical_difference_mean_ranks.csv, normalized_per_split.csv.
"""
import sys

sys.dont_write_bytecode = True
import os  # noqa: E402

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_v] = '1'
import importlib.util  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2, rankdata, spearmanr, studentized_range  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

from common import RESULTS, write_json  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nm36 = load('nm36', '36_dreval_normalized.py')
rs = nm36.rs
OUT = RESULTS / 'revision_crosslab'
REF = 'CTRPv2|CC|lnIC50'
IDENTITY = f'{REF}->{REF}'
CROSS = {'GDSC1': f'{REF}->GDSC1|CC|lnIC50', 'GDSC2': f'{REF}->GDSC2|CC|lnIC50', 'CTRPv1': f'{REF}->CTRPv1|CC|lnIC50'}
AGNOSTIC_EXACT = {'MultiViewRandomForest'}
ALPHA = 0.05


def friedman(X):
    """X: blocks x models (higher = better). Returns mean ranks (1 = best), chi2, P, Nemenyi CD, Conover CD and
    Holm-adjusted Conover pairwise P matrix."""
    N, k = X.shape
    Rk = np.vstack([rankdata(-row) for row in X])
    Rj = Rk.sum(axis=0)
    A1 = (Rk ** 2).sum()
    C1 = N * k * (k + 1) ** 2 / 4
    T1 = (k - 1) * ((Rj - N * (k + 1) / 2) ** 2).sum() / (A1 - C1)
    p = float(chi2.sf(T1, k - 1))
    q = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd_nem = float(q * np.sqrt(k * (k + 1) / (6 * N)))
    df = (N - 1) * (k - 1)
    se = np.sqrt(2 * (N * A1 - (Rj ** 2).sum()) / df)
    cd_con = float(tdist.ppf(1 - ALPHA / 2, df) * se / N)
    P = np.ones((k, k))
    iu = list(combinations(range(k), 2))
    raw = np.array([2 * tdist.sf(abs(Rj[i] - Rj[j]) / se, df) for i, j in iu])
    order = np.argsort(raw)
    adj = np.empty_like(raw)
    run = 0.0
    for r, o in enumerate(order):
        run = max(run, min(1.0, (len(raw) - r) * raw[o]))
        adj[o] = run
    for (i, j), a in zip(iu, adj):
        P[i, j] = P[j, i] = a
    return Rj / N, T1, p, cd_nem, cd_con, P


def per_split_normalized(published, drugs):
    """DrEval normalized Pearson per CV split (36_dreval_normalized.norm_pearson, unrolled per split)."""
    models = sorted(m for m in os.listdir(rs.LB / 'LCO') if (rs.LB / 'LCO' / m / 'predictions').exists())
    lb = {m: nm36.load_raw(m) for m in models}
    naive = lb['NaiveMeanEffectsPredictor'][['drug', 'cell', 'split', 'pred']].rename(columns={'pred': 'naive'})
    refs = pd.read_csv(RESULTS / 'dreval' / 'reference_predictions.csv.gz', dtype={'drug': str})
    common = sorted(set.intersection(*[set(d.cell) for m, d in lb.items() if m not in rs.NAIVE]) & set(refs.cell))
    obs = pd.concat([d[['drug', 'cell', 'y']] for d in lb.values()]).groupby(['drug', 'cell'], as_index=False).y.mean()
    obs_s = obs.rename(columns={'y': 'response'}).assign(response=lambda t: -t.response)
    Z, _, _, _ = rs.selbench.decompose(obs_s[obs_s.cell.isin(common)])
    rows = []
    for m in published:
        d = lb[m]
        for setname, dset in [('dreval_common_set', set(Z.index)), ('common_drugs', set(drugs))]:
            x = d[d.cell.isin(common) & d.drug.isin(dset)].merge(naive, on=['drug', 'cell', 'split'], how='inner')
            for k, g in x.groupby('split'):
                yt, yp = (g.y - g.naive).values, (g.pred - g.naive).values
                ok = np.isfinite(yt) & np.isfinite(yp)
                rows.append({'model': m, 'drug_set': setname, 'split': int(k), 'normalized_pearson': float(np.corrcoef(yt[ok], yp[ok])[0, 1])})
    return pd.DataFrame(rows)


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(OUT), help='input/output directory (testing)')
    OUT = Path(ap.parse_args().dir)
    per = pd.read_csv(OUT / 'per_drug_scores.csv.gz', dtype={'drug': str})
    published = sorted(per[per['class'].isin(['classical', 'neural/factorization'])].model.unique())
    assert len(published) == 22, published
    scores, blocks = {}, {}
    # within CTRPv2: drugs predicted by all 22 published models
    t = per[(per.pair == IDENTITY) & per.model.isin(published)]
    Wm = {s: t.pivot_table(index='drug', columns='model', values=s).reindex(columns=published)
          for s in ['raw_w', 'selT_w', 'adj_w']}
    Wm['adj_w'] = Wm['adj_w'].copy()
    for m in AGNOSTIC_EXACT:  # no drug-specific component: adjusted score structurally 0 where the model predicts
        Wm['adj_w'][m] = Wm['adj_w'][m].where(Wm['raw_w'][m].isna(), Wm['adj_w'][m].fillna(0.0))
    common_within = Wm['raw_w'].dropna().index.intersection(Wm['selT_w'].dropna().index)
    for name, s in [('raw_within', 'raw_w'), ('selT_within', 'selT_w'), ('adjusted_within', 'adj_w')]:
        X = Wm[s].reindex(index=common_within, columns=published)
        blocks[name] = X.dropna()
        scores[name] = X.median()
    common_cross = {}
    for tg, pair in CROSS.items():
        t = per[(per.pair == pair) & per.model.isin(published)]
        for name, s in [(f'selBOTH_{tg}', 'selB_c'), (f'adjusted_{tg}', 'adj_c'), (f'raw_cross_{tg}', 'raw_c')]:
            X = t.pivot_table(index='drug', columns='model', values=s).reindex(columns=published)
            if s == 'adj_c':
                rawc = t.pivot_table(index='drug', columns='model', values='raw_c').reindex(index=X.index, columns=published)
                for m in AGNOSTIC_EXACT:
                    X[m] = X[m].where(rawc[m].isna(), X[m].fillna(0.0))
            X = X.dropna()
            common_cross[name] = len(X)
            blocks[name] = X
            scores[name] = X.median()
    nrm = pd.read_csv(RESULTS / 'dreval' / 'normalized_metric.csv').set_index('model').normalized_pearson
    scores['dreval_normalized_reported'] = nrm.reindex(published)
    ns = per_split_normalized(published, common_within)
    ns.to_csv(OUT / 'normalized_per_split.csv', index=False)
    chk = ns[ns.drug_set == 'dreval_common_set'].groupby('model').normalized_pearson.mean()
    max_dev = float((chk - nrm.reindex(chk.index)).abs().max())
    assert max_dev < 1e-9, max_dev
    sp = ns[ns.drug_set == 'common_drugs'].pivot_table(index='split', columns='model', values='normalized_pearson')
    blocks['dreval_normalized_common_drugs'] = sp.reindex(columns=published)
    scores['dreval_normalized_common_drugs'] = sp.mean().reindex(published)
    S = pd.DataFrame(scores)
    ranks = S.rank(ascending=False, method='average')
    ranks['selBOTH_crosslab_meanrank'] = ranks[[f'selBOTH_{t}' for t in CROSS]].mean(axis=1)
    ranks['adjusted_crosslab_meanrank'] = ranks[[f'adjusted_{t}' for t in CROSS]].mean(axis=1)
    ranks['selBOTH_crosslab'] = ranks.selBOTH_crosslab_meanrank.rank(method='average')
    ranks['adjusted_crosslab'] = ranks.adjusted_crosslab_meanrank.rank(method='average')
    long = S.reset_index().melt(id_vars='model', var_name='ranking', value_name='score').merge(
        ranks.reset_index().melt(id_vars='model', var_name='ranking', value_name='rank'), on=['model', 'ranking'], how='outer')
    long['class'] = long.model.map(rs.model_class)
    long.to_csv(OUT / 'rankings.csv', index=False, float_format='%.6g')
    main_rankings = ['raw_within', 'dreval_normalized_reported', 'dreval_normalized_common_drugs', 'selT_within',
                     'adjusted_within', 'selBOTH_GDSC1', 'selBOTH_GDSC2', 'selBOTH_CTRPv1', 'selBOTH_crosslab',
                     'adjusted_crosslab', 'raw_cross_GDSC1']
    cor = []
    for a, b in combinations(main_rankings, 2):
        r = spearmanr(ranks[a], ranks[b])
        cor.append({'ranking_a': a, 'ranking_b': b, 'models': len(ranks), 'spearman_rho': float(r[0]), 'p': float(r[1])})
    cor = pd.DataFrame(cor)
    cor.to_csv(OUT / 'ranking_correlations.csv', index=False, float_format='%.6g')
    # critical differences
    cdrows, mr = [], []
    fr = {}
    for name, X in blocks.items():
        X = X.reindex(columns=published)
        meanr, T1, p, cdn, cdc, Pc = friedman(X.values)
        fr[name] = (pd.Series(meanr, index=published), cdn, pd.DataFrame(Pc, index=published, columns=published))
        cdrows.append({'ranking': name, 'blocks': X.shape[0], 'block_unit': 'CV split' if 'normalized' in name else 'drug',
                       'models': X.shape[1], 'friedman_chi2': T1, 'friedman_p': p, 'nemenyi_cd_mean_rank': cdn,
                       'conover_cd_mean_rank_unadjusted': cdc})
        for m in published:
            mr.append({'ranking': name, 'model': m, 'mean_rank': float(fr[name][0][m])})
    pd.DataFrame(cdrows).to_csv(OUT / 'critical_difference.csv', index=False, float_format='%.6g')
    pd.DataFrame(mr).to_csv(OUT / 'critical_difference_mean_ranks.csv', index=False, float_format='%.6g')
    # moves between rankings and whether they exceed the critical difference
    moves = []
    comps = [('raw_within', 'selT_within'), ('raw_within', 'selBOTH_GDSC1'), ('raw_within', 'selBOTH_GDSC2'),
             ('raw_within', 'selBOTH_CTRPv1'), ('dreval_normalized_common_drugs', 'selT_within'),
             ('dreval_normalized_common_drugs', 'selBOTH_GDSC1'), ('raw_within', 'adjusted_within'),
             ('raw_within', 'dreval_normalized_common_drugs'), ('selT_within', 'adjusted_within')]
    for a, b in comps:
        ma, cda, Pa = fr[a]
        mb, cdb, Pb = fr[b]
        for m in published:
            sig_n, sig_c, over = [], [], []
            for o in published:
                if o == m:
                    continue
                before, after = ma[m] - ma[o], mb[m] - mb[o]   # negative = m ranked better
                if np.sign(before) != np.sign(after) and before != 0 and after != 0:
                    over.append(o)
                    if abs(before) > cda and abs(after) > cdb:
                        sig_n.append(o)
                    if Pa.loc[m, o] < ALPHA and Pb.loc[m, o] < ALPHA:
                        sig_c.append(o)
            moves.append({'from': a, 'to': b, 'model': m, 'class': rs.model_class(m),
                          'rank_from': float(ranks.loc[m, a]) if a in ranks else np.nan,
                          'rank_to': float(ranks.loc[m, b]) if b in ranks else np.nan,
                          'mean_rank_from': float(ma[m]), 'mean_rank_to': float(mb[m]),
                          'order_reversals': len(over), 'reversals_beyond_nemenyi_cd_both': len(sig_n),
                          'reversals_conover_holm_both': len(sig_c), 'nemenyi_cd_from': cda, 'nemenyi_cd_to': cdb,
                          'significant_reversal_partners_nemenyi': '|'.join(sig_n)})
    mv = pd.DataFrame(moves)
    mv['rank_change'] = mv.rank_to - mv.rank_from
    mv.to_csv(OUT / 'ranking_moves.csv', index=False, float_format='%.6g')
    write_json(OUT / 'rankings_run.json', {'published_models': published, 'common_drugs_within': int(len(common_within)),
                                           'common_drugs_cross': common_cross,
                                           'normalized_reimplementation_max_abs_dev': max_dev})
    pd.set_option('display.width', 250)
    print(ranks[main_rankings].sort_values('raw_within').round(1).to_string())
    print(cor.round(3).to_string(index=False))
    print(pd.DataFrame(cdrows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
