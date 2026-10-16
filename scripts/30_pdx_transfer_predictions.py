"""Frozen cell-line-to-PDX transfer predictions for PDX_Bruna (Amendment 2; written and run before any PDX_Bruna
drug-response value was retrieved).

Inputs are only the PDX_Bruna input members fetched by 29_fetch_pdx_inputs.py (model names, drug names and SMILES,
expression, GISTIC copy number) and frozen discovery products.

Drug identity (registered rule): a PDX drug maps to the discovery compound whose InChIKey connectivity block
(largest neutralised fragment, common.inchikey14) equals that of DrEval's canonical SMILES, if exactly one compound
matches; otherwise by a unique normalised name (06_load_confirmation.drug_index); otherwise it is kept as
'PDX:<pubchem_id>' (used only for the PDX fitness axis); ambiguous matches are excluded.

Transfer models (as for BeatAML, Amendment 1): for every mapped discovery compound, kernel ridge (alpha by closed-form
LOO) is fitted on all screened cell lines to the measured discovery consensus selective (rep_sel_cons) or raw
(rep_raw_cons) profile and applied to every PDX model. Primary features: expression of the DepMap top-variance genes
that PDX_Bruna also measures, each gene standardised within its own domain (cell lines; PDX models). Secondary
features: the same plus gene-level copy number of the DepMap copy-number genes that PDX_Bruna also measures
(blocks weighted equally, standardised within domain). A cell-blind check on cell lines (shared 5-fold partition)
is saved for interpretation.
Outputs: results/pdx_transfer/{drug_map.csv, models.csv, pred_sel.parquet, pred_raw.parquet, pred_sel_cn.parquet,
pred_raw_cn.parquet, cellblind_sel.parquet, cellblind_raw.parquet, run.json}.
"""
import importlib.util
import os
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
PDX = OWN_RAW / 'pdx_bruna_inputs'
OUT = RESULTS / 'pdx_transfer'


def zs(df):
    sd = df.std().replace(0, 1).fillna(1)
    return ((df - df.mean()) / sd).fillna(0)


def drug_map():
    names = pd.read_csv(PDX / 'drug_names.csv', dtype=str)
    smiles = pd.read_csv(PDX / 'drug_smiles.csv', dtype=str).drop_duplicates('pubchem_id').set_index('pubchem_id')
    comp = pd.read_csv(PROCESSED / 'compounds.csv', dtype=str)
    key_to = {}
    for r in comp.dropna(subset=['inchikey14']).itertuples():
        for k in r.inchikey14.split('|'):
            key_to.setdefault(k, set()).add(r.compound)
    didx, _ = lc.drug_index()
    rows = []
    for r in names.itertuples():
        smi = smiles.canonical_smiles.get(r.pubchem_id)
        k = inchikey14(smi) if isinstance(smi, str) else None
        cands = key_to.get(k, set()) if k else set()
        if len(cands) == 1:
            c, how = next(iter(cands)), 'inchikey14'
        else:
            nc = didx.get(norm_name(r.drug_name), set())
            if len(cands) > 1 or len(nc) > 1:
                c, how = None, 'ambiguous_excluded'
            elif len(nc) == 1:
                c, how = next(iter(nc)), 'unique_name'
            else:
                c, how = f'PDX:{r.pubchem_id}', 'not_in_discovery'
        rows.append({'pubchem_id': r.pubchem_id, 'drug_name': r.drug_name, 'inchikey14': k, 'compound': c, 'method': how,
                     'candidates': '|'.join(sorted(cands))})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dm = drug_map()
    dm.to_csv(OUT / 'drug_map.csv', index=False)
    models = pd.read_csv(PDX / 'cell_line_names.csv', dtype=str)
    pexp = pd.read_csv(PDX / 'gene_expression.csv').set_index('cell_line_name')
    pexp = pexp.loc[:, ~pexp.columns.duplicated()].apply(pd.to_numeric, errors='coerce')
    pcn = pd.read_csv(PDX / 'copy_number_variation_gistic.csv').set_index('cell_line_name')
    pcn = pcn.loc[:, ~pcn.columns.duplicated()].apply(pd.to_numeric, errors='coerce')
    samples = [s for s in models.cell_line_name if s in pexp.index]
    models.assign(has_expression=models.cell_line_name.isin(pexp.index),
                  has_copy_number=models.cell_line_name.isin(pcn.index)).to_csv(OUT / 'models.csv', index=False)

    cexp = pd.read_parquet(FEAT / 'expression_top.parquet')
    ccn = pd.read_parquet(FEAT / 'copy_number.parquet')
    folds = pd.read_csv(RESULTS / 'models' / 'cell_folds.csv').set_index('depmap_id').fold
    cells = [c for c in folds.index if c in cexp.index and c in ccn.index]
    genes = [g for g in cexp.columns if g in pexp.columns and pexp[g].notna().mean() > 0.9]
    cgenes = [g for g in ccn.columns if g in pcn.columns and pcn[g].notna().mean() > 0.9]
    Ce = zs(cexp.loc[cells, genes]) / np.sqrt(len(genes))
    Pe = zs(pexp.loc[samples, genes]) / np.sqrt(len(genes))
    Cc = zs(ccn.loc[cells, cgenes]) / np.sqrt(max(len(cgenes), 1))
    Pc = zs(pcn.reindex(samples)[cgenes]) / np.sqrt(max(len(cgenes), 1))
    kernels = {'': (Ce.values @ Ce.values.T, Pe.values @ Ce.values.T)}
    Xc, Xp = np.hstack([Ce.values, Cc.values]), np.hstack([Pe.values, Pc.values])
    kernels['_cn'] = (Xc @ Xc.T / 2, Xp @ Xc.T / 2)

    fz = pd.read_parquet(RESULTS / 'frozen_predictions' / 'rep_raw_cons.parquet')
    sz = pd.read_parquet(RESULTS / 'frozen_predictions' / 'rep_sel_cons.parquet')
    compounds = sorted(c for c in dm.compound.dropna().unique() if not c.startswith('PDX:') and c in fz.index)
    fv = folds.reindex(cells).values
    preds = {}
    for kind, M in [('raw', fz), ('sel', sz)]:
        for sfx, (Kcc, Kpc) in kernels.items():
            P = {}
            O = {}
            for d in compounds:
                if d not in M.index:
                    continue
                y = M.loc[d].reindex(cells).values
                ok = np.where(np.isfinite(y))[0]
                if len(ok) < 25:
                    continue
                P[d], _ = kernel_ridge_loo(Kcc[np.ix_(ok, ok)], y[ok], Kpc[:, ok])
                if sfx == '':
                    oof = np.full(len(y), np.nan)
                    for f in range(5):
                        tr = ok[fv[ok] != f]; te = np.where(fv == f)[0]
                        if len(tr) >= 25 and len(te):
                            oof[te], _ = kernel_ridge_loo(Kcc[np.ix_(tr, tr)], y[tr], Kcc[np.ix_(te, tr)])
                    O[d] = oof
            pd.DataFrame(P, index=samples).T.astype('float32').to_parquet(OUT / f'pred_{kind}{sfx}.parquet')
            if O:
                pd.DataFrame(O, index=cells).T.astype('float32').to_parquet(OUT / f'cellblind_{kind}.parquet')
            preds[f'{kind}{sfx}'] = len(P)
    write_json(OUT / 'run.json', {'pdx_models': len(models), 'models_with_expression': len(samples),
                                  'expression_genes': len(genes), 'copy_number_genes': len(cgenes), 'cells': len(cells),
                                  'pdx_drugs': len(dm), 'mapping': dm.method.value_counts().to_dict(),
                                  'mapped_compounds': len(compounds), 'predicted': preds})
    print(dm.method.value_counts().to_dict(), len(samples), len(genes), len(cgenes), preds)


if __name__ == '__main__':
    main()
