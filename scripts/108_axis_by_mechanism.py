"""Round 8 (a colleague's comment): does removing general sensitivity also remove reproducible signal, and for which
mechanisms? EXPLORATORY and descriptive; it re-summarizes existing per-drug outputs and fits nothing.

For each of the 353 multi-screen drugs, 53_reproducibility_classes gives the curated mechanism class, the axis share
(median over screens of r_zg^2) and the mean cross-screen agreement over screen pairs of the raw profiles (rho_raw) and
of the selective residuals (rho_sel). The paired difference rho_raw - rho_sel is the cross-screen agreement that rests
on the general-sensitivity axis: positive when the axis-aligned part of a drug's profile is reproduced by another
screen, so that removing the axis removes reproducible signal.

Per curated class with >= 5 drugs: median axis share, median raw and selective agreement, the median paired difference
with a drug-bootstrap 95% interval (5,000 draws) and the share of drugs whose raw agreement exceeds the selective by more
than 0.05. Across all drugs: Spearman correlation of axis share with the paired difference (drug-bootstrap interval).
Outputs: results/revision_round8/mechanism_axis/{class_table.csv, drugs.csv, summary.json}.
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, write_json

OUT = RESULTS / 'revision_round8' / 'mechanism_axis'
N_BOOT = 5000
SEED = 10801
MIN_DRUGS = 5
MARGIN = 0.05


def boot_median(v, rng):
    b = np.median(rng.choice(v, size=(N_BOOT, len(v)), replace=True), axis=1)
    return np.percentile(b, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(RESULTS / 'revision_transfer' / 'drug_reproducibility_table.csv')
    d = d[['compound', 'name', 'new_class', 'fitness_r2', 'rho_raw', 'rho_sel', 'n_pairs']].rename(
        columns={'new_class': 'class', 'fitness_r2': 'axis_share'})
    d['raw_minus_sel'] = d.rho_raw - d.rho_sel
    d.to_csv(OUT / 'drugs.csv', index=False)

    rows = []
    for cls, g in d.groupby('class'):
        if len(g) < MIN_DRUGS:
            continue
        lo, hi = boot_median(g.raw_minus_sel.values, rng)
        rows.append({'class': cls, 'drugs': len(g), 'median_axis_share': float(g.axis_share.median()),
                     'median_rho_raw': float(g.rho_raw.median()), 'median_rho_sel': float(g.rho_sel.median()),
                     'median_raw_minus_sel': float(g.raw_minus_sel.median()), 'ci_lo': float(lo), 'ci_hi': float(hi),
                     'frac_raw_above_sel_by_margin': float((g.raw_minus_sel > MARGIN).mean())})
    K = pd.DataFrame(rows).sort_values('median_raw_minus_sel', ascending=False)
    K.to_csv(OUT / 'class_table.csv', index=False)

    rho, p = stats.spearmanr(d.axis_share, d.raw_minus_sel)
    idx = rng.integers(0, len(d), size=(2000, len(d)))
    bs = [stats.spearmanr(d.axis_share.values[i], d.raw_minus_sel.values[i])[0] for i in idx]
    lo, hi = boot_median(d.raw_minus_sel.values, rng)
    summary = {'seed': SEED, 'n_drugs': len(d), 'n_classes_shown': len(K), 'margin': MARGIN,
               'all_median_axis_share': float(d.axis_share.median()), 'all_median_rho_raw': float(d.rho_raw.median()),
               'all_median_rho_sel': float(d.rho_sel.median()), 'all_median_raw_minus_sel': float(d.raw_minus_sel.median()),
               'all_ci': [float(lo), float(hi)], 'all_frac_raw_above_sel_by_margin': float((d.raw_minus_sel > MARGIN).mean()),
               'spearman_axis_share_vs_difference': float(rho), 'spearman_p': float(p),
               'spearman_ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               'classes': K.to_dict(orient='records'),
               'status': 'exploratory; descriptive re-summary of 53_reproducibility_classes outputs'}
    write_json(OUT / 'summary.json', summary)
    print(K.round(3).to_string(index=False))
    print({k: v for k, v in summary.items() if k != 'classes'})


if __name__ == '__main__':
    main()
