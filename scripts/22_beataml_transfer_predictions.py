"""Frozen cell-line-to-patient transfer predictions for BeatAML (Amendment 1; written and run before any BeatAML
drug-response data were retrieved).

Features shared by both domains: expression of the DepMap top-variance genes that BeatAML also measures and
gene-level mutation indicators of the DepMap mutation-matrix genes (patients: protein-altering WES calls).
Each feature is standardized within its own domain (cell lines; patient samples), blocks weighted equally.
For every discovery compound, kernel ridge models (alpha by closed-form LOO) are fitted on all screened cell
lines to the measured discovery consensus raw (rep_raw_cons) or selective (rep_sel_cons) profile and applied
to every included AML RNA sample. A cell-blind check of the same reduced feature set on cell lines (shared
5-fold partition) is saved for interpretation.
Outputs: results/beataml_transfer/{pred_raw,pred_sel}.parquet, samples.csv, cellblind_check.csv.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import OWN_RAW, PROCESSED, RESULTS, write_json
from modeling import kernel_ridge_loo

FEAT = PROCESSED / 'features'
BA = OWN_RAW / 'beataml_inputs'
OUT = RESULTS / 'beataml_transfer'
PROTEIN_ALTERING = {'missense_variant', 'frameshift_variant', 'stop_gained', 'inframe_deletion', 'inframe_insertion',
                    'splice_acceptor_variant', 'splice_donor_variant', 'protein_altering_variant', 'start_lost', 'stop_lost'}
G = {}


def zs(df):
    sd = df.std().replace(0, 1).fillna(1)
    return ((df - df.mean()) / sd).fillna(0)


def init(payload):
    G.update(payload)


def fit(d):
    out = {}
    for kind in ['raw', 'sel']:
        y = G[kind].loc[d].reindex(G['cells']).values if d in G[kind].index else None
        if y is None:
            continue
        ok = np.where(np.isfinite(y))[0]
        if len(ok) < 25:
            continue
        pred, _ = kernel_ridge_loo(G['Kcc'][np.ix_(ok, ok)], y[ok], G['Kpc'][:, ok])
        out[kind] = pred
        # cell-blind check on cell lines with the reduced features
        oof = np.full(len(y), np.nan)
        for f in range(5):
            tr = ok[G['folds'][ok] != f]; te = np.where(G['folds'] == f)[0]
            if len(tr) >= 25 and len(te):
                oof[te], _ = kernel_ridge_loo(G['Kcc'][np.ix_(tr, tr)], y[tr], G['Kcc'][np.ix_(te, tr)])
        out[kind + '_oof'] = oof
    return d, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    smap = pd.read_excel(BA / 'beataml_waves1to4_sample_mapping.xlsx')
    keep = smap[(smap.rna_include_in_analysis == 'yes') & (smap.rna_control == 'No')]
    expr = pd.read_csv(BA / 'beataml_waves1to4_norm_exp_dbgap.txt', sep='\t')
    expr = expr.assign(m=expr.iloc[:, 4:].mean(axis=1)).sort_values('m', ascending=False).drop_duplicates('display_label')
    expr = expr.set_index('display_label').drop(columns=['stable_id', 'description', 'biotype', 'm'])
    samples = [s for s in keep.dbgap_rnaseq_sample if s in expr.columns]
    Pexp = expr[samples].T
    mut = pd.read_csv(BA / 'beataml_wes_wv1to4_mutations_dbgap.txt', sep='\t', usecols=['dbgap_sample_id', 'variant_classification', 'symbol'])
    mut = mut[mut.variant_classification.isin(PROTEIN_ALTERING)]
    dna2rna = keep.dropna(subset=['dbgap_dnaseq_sample']).set_index('dbgap_dnaseq_sample').dbgap_rnaseq_sample
    mut['rna'] = mut.dbgap_sample_id.map(dna2rna)
    has_dna = set(keep.dropna(subset=['dbgap_dnaseq_sample']).dbgap_rnaseq_sample)

    cexp = pd.read_parquet(FEAT / 'expression_top.parquet')
    cmut = pd.read_parquet(FEAT / 'mutation.parquet')
    genes = [g for g in cexp.columns if g in Pexp.columns]
    mgenes = list(cmut.columns)
    Pmut = pd.DataFrame(0.0, index=samples, columns=mgenes)
    for r in mut.dropna(subset=['rna']).itertuples():
        if r.symbol in Pmut.columns and r.rna in Pmut.index:
            Pmut.loc[r.rna, r.symbol] = 1.0
    Pmut.loc[[s for s in samples if s not in has_dna]] = np.nan          # no WES: mutation block set to the mean
    folds_df = pd.read_csv(RESULTS / 'models' / 'cell_folds.csv').set_index('depmap_id').fold
    cells = [c for c in folds_df.index if c in cexp.index and c in cmut.index]
    Ce = zs(cexp.loc[cells, genes]) / np.sqrt(len(genes)); Pe = zs(Pexp[genes]) / np.sqrt(len(genes))
    Cm = zs(cmut.loc[cells, mgenes]) / np.sqrt(len(mgenes)); Pm = zs(Pmut[mgenes]) / np.sqrt(len(mgenes))
    Xc = np.hstack([Ce.values, Cm.values]); Xp = np.hstack([Pe.values, Pm.values])
    Kcc = Xc @ Xc.T / 2; Kpc = Xp @ Xc.T / 2
    fz = pd.read_parquet(RESULTS / 'frozen_predictions' / 'rep_raw_cons.parquet')
    sz = pd.read_parquet(RESULTS / 'frozen_predictions' / 'rep_sel_cons.parquet')
    compounds = list(fz.index)
    payload = {'raw': fz, 'sel': sz, 'cells': pd.Index(cells), 'Kcc': Kcc, 'Kpc': Kpc,
               'folds': folds_df.reindex(cells).values}
    preds = {'raw': {}, 'sel': {}, 'raw_oof': {}, 'sel_oof': {}}
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    with ProcessPoolExecutor(32, initializer=init, initargs=(payload,)) as ex:
        for i, (d, o) in enumerate(ex.map(fit, compounds, chunksize=8)):
            for k, v in o.items():
                preds[k][d] = v
            if i % 400 == 0:
                print('fitted', i, '/', len(compounds), flush=True)
    for k in ['raw', 'sel']:
        pd.DataFrame(preds[k], index=samples).T.astype('float32').to_parquet(OUT / f'pred_{k}.parquet')
        pd.DataFrame(preds[k + '_oof'], index=cells).T.astype('float32').to_parquet(OUT / f'cellblind_{k}.parquet')
    keep[keep.dbgap_rnaseq_sample.isin(samples)].to_csv(OUT / 'samples.csv', index=False)
    write_json(OUT / 'run.json', {'samples': len(samples), 'samples_with_wes': len(has_dna & set(samples)),
                                  'expression_genes': len(genes), 'mutation_genes': len(mgenes), 'cells': len(cells),
                                  'compounds': {k: len(v) for k, v in preds.items()}})
    print('done', len(samples), len(genes), len(mgenes))


if __name__ == '__main__':
    main()
