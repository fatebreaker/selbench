"""Frozen cell-line-to-patient-derived-model transfer predictions for Amendment 3 (written and run before any response
value of these datasets was retrieved).

Inputs: only the CoderData v2.4 input files fetched by 69_fetch_pdm_inputs.py (samples, transcriptomics, drugs) and
frozen discovery products. Datasets: bladder (Lee et al. 2018 organoids), pancreatic (Tiriac et al. 2018 organoids),
liver (hepatocellular carcinoma organoids), sarcoma (Al Shihabi et al. 2024) and MPNST (3D micro-tissues and tumours).
Colorectal (expression for 9 samples) and Novartis PDXE (in vivo) are not used.

Drug identity (as Amendment 2): the InChIKey connectivity block of CoderData's canonical SMILES (largest neutralised
fragment, common.inchikey14) matches exactly one discovery compound; otherwise a unique normalised name
(06_load_confirmation.drug_index); otherwise 'PDM:<improve_drug_id>' (used only for the dataset's fitness axis);
ambiguous matches are excluded.

Transfer models (as for BeatAML and PDX_Bruna): for every mapped discovery compound, kernel ridge (alpha by
closed-form LOO) is fitted on all screened cell lines to the frozen discovery consensus selective (rep_sel_cons) or
raw (rep_raw_cons) profile, from expression of the DepMap top-variance genes that the dataset also measures
(log2(TPM + 1); each gene standardised within its own domain: cell lines; the dataset's samples). Genes are matched
by Entrez id through the DepMap 24Q4 expression header.
Outputs: results/pdm_transfer/<dataset>/{drug_map.csv, samples.csv, pred_sel.parquet, pred_raw.parquet};
results/pdm_transfer/run.json.
"""
import importlib.util
import os
import re
from pathlib import Path

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import OWN_RAW, PROCESSED, RESULTS, inchikey14, norm_name, write_json  # noqa: E402
from modeling import kernel_ridge_loo  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lc', HERE / '06_load_confirmation.py')
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
FEAT = PROCESSED / 'features'
INP = OWN_RAW / 'pdm_inputs'
OUT = RESULTS / 'pdm_transfer'
DATASETS = ['bladder', 'pancreatic', 'liver', 'sarcoma', 'mpnst']


def zs(df):
    sd = df.std().replace(0, 1).fillna(1)
    return ((df - df.mean()) / sd).fillna(0)


def symbol_to_entrez():
    head = (OWN_RAW / 'depmap24q4' / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv').open().readline().rstrip('\n')
    out = {}
    for col in head.split(',')[1:]:
        m = re.match(r'^(.*) \((\d+)\)$', col)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def drug_map(ds):
    drugs = pd.read_csv(INP / f'{ds}_drugs.tsv.gz', sep='\t', dtype=str).drop_duplicates('improve_drug_id')
    comp = pd.read_csv(PROCESSED / 'compounds.csv', dtype=str)
    key_to = {}
    for r in comp.dropna(subset=['inchikey14']).itertuples():
        for k in r.inchikey14.split('|'):
            key_to.setdefault(k, set()).add(r.compound)
    didx, _ = lc.drug_index()
    rows = []
    for r in drugs.itertuples():
        smi = r.canSMILES
        k = inchikey14(smi) if isinstance(smi, str) else None
        cands = key_to.get(k, set()) if k else set()
        if len(cands) == 1:
            c, how = next(iter(cands)), 'inchikey14'
        else:
            nc = didx.get(norm_name(str(r.chem_name)), set())
            if len(cands) > 1 or len(nc) > 1:
                c, how = None, 'ambiguous_excluded'
            elif len(nc) == 1:
                c, how = next(iter(nc)), 'unique_name'
            else:
                c, how = f'PDM:{r.improve_drug_id}', 'not_in_discovery'
        rows.append({'improve_drug_id': r.improve_drug_id, 'chem_name': r.chem_name, 'inchikey14': k, 'compound': c,
                     'method': how, 'candidates': '|'.join(sorted(cands))})
    return pd.DataFrame(rows)


def expression(ds, s2e):
    t = pd.read_csv(INP / f'{ds}_transcriptomics.csv.gz', usecols=['improve_sample_id', 'entrez_id', 'transcriptomics'])
    t = t.dropna()
    X = t.pivot_table(index='improve_sample_id', columns='entrez_id', values='transcriptomics', aggfunc='mean')
    X = np.log2(X.clip(lower=0) + 1)
    e2s = {v: k for k, v in s2e.items()}
    X.columns = [e2s.get(int(c)) for c in X.columns]
    X = X.loc[:, [c is not None for c in X.columns]]
    X = X.loc[:, ~X.columns.duplicated()]
    X.index = X.index.astype(str)
    return X


def main():
    s2e = symbol_to_entrez()
    cexp = pd.read_parquet(FEAT / 'expression_top.parquet')
    folds = pd.read_csv(RESULTS / 'models' / 'cell_folds.csv').set_index('depmap_id').fold
    cells = [c for c in folds.index if c in cexp.index]
    fz = pd.read_parquet(RESULTS / 'frozen_predictions' / 'rep_raw_cons.parquet')
    sz = pd.read_parquet(RESULTS / 'frozen_predictions' / 'rep_sel_cons.parquet')
    run = {}
    for ds in DATASETS:
        out = OUT / ds
        out.mkdir(parents=True, exist_ok=True)
        dm = drug_map(ds)
        dm.to_csv(out / 'drug_map.csv', index=False)
        X = expression(ds, s2e)
        samples = pd.read_csv(INP / f'{ds}_samples.csv', dtype=str).drop_duplicates('improve_sample_id')
        samples['has_expression'] = samples.improve_sample_id.isin(X.index)
        samples.to_csv(out / 'samples.csv', index=False)
        genes = [g for g in cexp.columns if g in X.columns and X[g].notna().mean() > 0.9]
        Ce = zs(cexp.loc[cells, genes]) / np.sqrt(len(genes))
        Pe = zs(X[genes]) / np.sqrt(len(genes))
        Kcc, Kpc = Ce.values @ Ce.values.T, Pe.values @ Ce.values.T
        compounds = sorted(c for c in dm.compound.dropna().unique() if not c.startswith('PDM:') and c in fz.index)
        npred = {}
        for kind, M in [('raw', fz), ('sel', sz)]:
            P = {}
            for d in compounds:
                if d not in M.index:
                    continue
                y = M.loc[d].reindex(cells).values
                ok = np.where(np.isfinite(y))[0]
                if len(ok) < 25:
                    continue
                P[d], _ = kernel_ridge_loo(Kcc[np.ix_(ok, ok)], y[ok], Kpc[:, ok])
            pd.DataFrame(P, index=X.index).T.astype('float32').to_parquet(out / f'pred_{kind}.parquet')
            npred[kind] = len(P)
        run[ds] = {'samples': int(len(samples)), 'samples_with_expression': int(len(X)), 'genes': len(genes),
                   'drugs': int(len(dm)), 'mapping': dm.method.value_counts().to_dict(),
                   'mapped_compounds': len(compounds), 'predicted': npred}
        print(ds, run[ds], flush=True)
    write_json(OUT / 'run.json', {'cells': len(cells), 'datasets': run})


if __name__ == '__main__':
    main()
