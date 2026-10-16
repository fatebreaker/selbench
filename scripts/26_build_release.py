"""Build the selbench data release (selbench/release/): reliability-annotated drug table, selective profiles of
the discovery screens, fitness axes of all screens, the shared cell-line partition and confirmation results."""
import json

import numpy as np
import pandas as pd

from common import PROCESSED, PROJECT, RESULTS

REL = PROJECT / 'selbench' / 'release'
DISC = ['PRISM', 'CTRP', 'GDSC1', 'GDSC2']


def main():
    REL.mkdir(parents=True, exist_ok=True)
    dt = pd.read_csv(RESULTS / 'drug_classes' / 'drug_table.csv', index_col=0)
    tri = pd.read_csv(RESULTS / 'ceiling' / 'triad_terms.csv').dropna()
    tri = tri[tri.rho_bc > 0.1]
    tri['R'] = (tri.rho_ab * tri.rho_ac / tri.rho_bc).clip(0, 1)
    rel = tri.groupby(['compound', 'screen']).R.median().unstack()
    rel.columns = [f'reliability_{c}' for c in rel.columns]
    out = dt.rename(columns={'rho_sel': 'mean_selective_agreement', 'rho_raw': 'mean_raw_agreement'}).join(rel)
    out['reproducibility_class'] = np.where(out.mean_selective_agreement >= 0.3, 'reproducible',
                                            np.where(out.mean_selective_agreement < 0.15, 'non_reproducible', 'intermediate'))
    for ds in ['gCSI', 'NCI60', 'CCLE', 'FIMM', 'GRAY']:
        sc = pd.read_csv(RESULTS / f'confirmation_{ds}' / 'scores.csv')
        out[f'confirm_{ds}'] = sc[sc.family == 'rep_sel_cons'].set_index('compound').sel.reindex(out.index)
    ba = pd.read_csv(RESULTS / 'confirmation_BeatAML' / 'scores.csv')
    out['beataml_transfer'] = ba[ba.family == 'sel'].set_index('compound').sel.reindex(out.index)
    out.index.name = 'compound'
    out.to_csv(REL / 'drug_reproducibility.csv')
    for s in DISC:
        z = pd.read_parquet(RESULTS / 'decomposition' / f'z_{s}.parquet').stack().rename('z')
        r = pd.read_parquet(RESULTS / 'decomposition' / f'resid_{s}.parquet').stack().rename('selective')
        pd.concat([z, r], axis=1).reset_index().rename(columns={'level_0': 'compound', 'level_1': 'depmap_id'}).to_csv(
            REL / f'profiles_{s}.csv.gz', index=False, float_format='%.4f')
    pd.read_csv(RESULTS / 'fitness_axis' / 'fitness_axes.csv', index_col=0).to_csv(REL / 'fitness_axes.csv', float_format='%.4f')
    pd.read_csv(RESULTS / 'models' / 'cell_folds.csv').to_csv(REL / 'cell_folds.csv', index=False)
    pd.read_csv(PROCESSED / 'compounds.csv')[['compound', 'name', 'targets', 'smiles', 'inchikey14', 'screens']].to_csv(
        REL / 'compounds.csv', index=False)
    summary = {'drugs': int(len(out)), 'classes': out.reproducibility_class.value_counts().to_dict(),
               'files': sorted(p.name for p in REL.iterdir())}
    (REL / 'manifest.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(summary)


if __name__ == '__main__':
    main()
