"""Inputs for the retraining workstream (revision round 1; editor items 4-5, Reviewer 1 M4/M5).

Writes results/revision_retrain/drevalpy_data/, the data directory of the drevalpy runs (58): for each DrEval dataset
(CTRPv2, CTRPv1, GDSC1, GDSC2, CCLE) the members drevalpy reads (<ds>.csv, cell_line_names.csv, drug_names.csv,
gene_expression.csv, drug_fingerprints/), extracted unchanged from data/dreval/<ds>.zip (Zenodo 23159774), plus meta/
(tissue_mapping.csv, gene_lists/) from data/dreval/meta.zip. Nothing is downloaded and data/ is not modified.

Writes results/revision_retrain/inputs/:
  responses.parquet          dataset, endpoint (lnic50 | auc), drug (PubChem id), cell (Cellosaurus id), tissue,
                             response = sensitivity (-LN_IC50_curvecurator or -AUC_curvecurator; higher = more
                             sensitive); duplicate drug-cell measurements averaged. Used for scoring (59).
  cell_names.csv             dataset, cell_line_name (drevalpy's cell identifier), cellosaurus_id, tissue.
  feat_depmap_expr.parquet   DepMap 24Q4 expression, top-5,000 variance genes (data/processed/features/
                             expression_top.parquet, log2 TPM+1), rows mapped to Cellosaurus ids via Model.csv RRID;
                             the feature-source control models of 58 ('Dm' suffix) read it.
  manifest.json              counts and checks, including that DrEval's split procedure (drevalpy
                             _leave_group_out_cv: row permutation with numpy default_rng(42), GroupKFold(10) over
                             cell_line_name) reproduces the published CTRPv2 leaderboard splits (data/dreval/splits.zip)
                             exactly, so drevalpy's own splits for the new settings follow the leaderboard's procedure.
Notes recorded in the manifest: DrEval's gene_expression.csv is identical across CTRPv2/CTRPv1/CCLE (one source) and
across GDSC1/GDSC2 (another source, with 4 duplicated cell lines that drevalpy averages), but differs between the two
families, so a model trained on one family and applied to the other family's feature file (drevalpy cross-study
prediction) sees a shifted feature distribution; the paired analysis therefore uses out-of-fold predictions only.
"""
import os
import shutil
import zipfile

for _v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '1')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from common import PROCESSED, PROJECT, RESULTS, load_models, write_json  # noqa: E402

DRE = PROJECT / 'data' / 'dreval'
OUT = RESULTS / 'revision_retrain'
INP = OUT / 'inputs'
DATASETS = ['CTRPv2', 'CTRPv1', 'GDSC1', 'GDSC2', 'CCLE']
ENDPOINTS = {'lnic50': 'LN_IC50_curvecurator', 'auc': 'AUC_curvecurator'}
N_FOLDS = 10
DREVALPY_DATA = OUT / 'drevalpy_data'
DREVALPY_MEMBERS = ['{ds}.csv', 'cell_line_names.csv', 'drug_names.csv', 'gene_expression.csv']


def atomic_parquet(df, path, **kw):
    tmp = path.with_suffix('.tmp')
    df.to_parquet(tmp, **kw)
    os.replace(tmp, path)


def read_zip_csv(dataset, member, **kw):
    with zipfile.ZipFile(DRE / f'{dataset}.zip') as z:
        with z.open(f'{dataset}/{member}') as f:
            return pd.read_csv(f, **kw)


def extract_drevalpy_data():
    """Extract the members drevalpy needs into results/revision_retrain/drevalpy_data (idempotent)."""
    info = {}
    for ds in DATASETS:
        dst = DREVALPY_DATA / ds
        with zipfile.ZipFile(DRE / f'{ds}.zip') as z:
            names = [m.format(ds=ds) for m in DREVALPY_MEMBERS]
            names += [n.split('/', 1)[1] for n in z.namelist()
                      if n.startswith(f'{ds}/drug_fingerprints/') and not n.endswith('/')]
            for n in names:
                out = dst / n
                if out.exists() and out.stat().st_size == z.getinfo(f'{ds}/{n}').file_size:
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(f'{ds}/{n}') as src, open(out, 'wb') as f:
                    shutil.copyfileobj(src, f, 1 << 20)
            info[ds] = len(names)
    with zipfile.ZipFile(DRE / 'meta.zip') as z:
        for n in z.namelist():
            if n.endswith('/') or not (n == 'meta/tissue_mapping.csv' or n.startswith('meta/gene_lists/')):
                continue
            out = DREVALPY_DATA / n
            if out.exists() and out.stat().st_size == z.getinfo(n).file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(out, 'wb') as f:
                shutil.copyfileobj(src, f, 1 << 20)
    gl = DREVALPY_DATA / 'meta' / 'gene_lists'
    if not (gl / 'landmark_genes.csv').exists():      # gene lists already extracted under data/dreval/meta
        shutil.copytree(DRE / 'meta' / 'gene_lists', gl, dirs_exist_ok=True)
    info['meta_files'] = len([p for p in (DREVALPY_DATA / 'meta').rglob('*') if p.is_file()])
    return info


def dreval_lco_folds(cell_line_names, n_splits=N_FOLDS, seed=42):
    """drevalpy.datasets.dataset._leave_group_out_cv (group = cell line): cell line name -> test fold."""
    g = np.asarray(cell_line_names).astype(str)
    g = g[np.random.default_rng(seed).permutation(np.arange(len(g)))]
    out = {}
    for k, (_, te) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(g)), groups=g)):
        for c in np.unique(g[te]):
            out[c] = k
    return pd.Series(out, name='fold')


def main():
    INP.mkdir(parents=True, exist_ok=True)
    man = {'datasets': {}, 'features': {}, 'notes': {}}
    man['drevalpy_data'] = extract_drevalpy_data()

    # responses (both endpoints), sensitivity orientation
    raw, long = {}, []
    for ds in DATASETS:
        d = read_zip_csv(ds, f'{ds}.csv', usecols=['cellosaurus_id', 'cell_line_name', 'pubchem_id', 'tissue',
                                                   *ENDPOINTS.values()], dtype={'pubchem_id': str})
        raw[ds] = d
        names = d.drop_duplicates(['cellosaurus_id', 'cell_line_name'])
        tissue = d.drop_duplicates('cellosaurus_id').set_index('cellosaurus_id').tissue
        info = {'rows': int(len(d)), 'cells': int(d.cellosaurus_id.nunique()), 'drugs': int(d.pubchem_id.nunique()),
                'cellosaurus_ids_with_several_names': int(names.cellosaurus_id.duplicated().sum())}
        for ep, col in ENDPOINTS.items():
            x = d.dropna(subset=[col]).groupby(['pubchem_id', 'cellosaurus_id'], as_index=False)[col].mean()
            x = x.rename(columns={'pubchem_id': 'drug', 'cellosaurus_id': 'cell'})
            x['response'] = -x[col]
            x['tissue'] = x.cell.map(tissue)
            long.append(x[['drug', 'cell', 'tissue', 'response']].assign(dataset=ds, endpoint=ep))
            info[ep] = {'finite_rows': int(d[col].notna().sum()), 'pairs': int(len(x)),
                        'drugs': int(x.drug.nunique()), 'cells': int(x.cell.nunique()),
                        'censored_share_of_rows': float(1 - d[col].notna().mean())}
        man['datasets'][ds] = info
        print(ds, info, flush=True)
    resp = pd.concat(long, ignore_index=True)[['dataset', 'endpoint', 'drug', 'cell', 'tissue', 'response']]
    atomic_parquet(resp, INP / 'responses.parquet', index=False)
    names = pd.concat([d.drop_duplicates(['cell_line_name', 'cellosaurus_id'])[['cell_line_name', 'cellosaurus_id',
                                                                               'tissue']].assign(dataset=ds)
                       for ds, d in raw.items()])
    assert not names.duplicated(['dataset', 'cell_line_name']).any()
    names[['dataset', 'cell_line_name', 'cellosaurus_id', 'tissue']].to_csv(INP / 'cell_names.csv', index=False)

    # DrEval's split procedure reproduces the published CTRPv2 leaderboard splits
    d = raw['CTRPv2'].dropna(subset=[ENDPOINTS['lnic50']]).reset_index(drop=True)
    proc = dreval_lco_folds(d.cell_line_name.values)
    pub = {}
    with zipfile.ZipFile(DRE / 'splits.zip') as z:
        for k in range(N_FOLDS):
            t = pd.read_csv(z.open(f'splits/cv_split_{k}_test.csv'), dtype={'pubchem_id': str})
            for c in t.cell_line_name.unique():
                pub[c] = k
    pub = pd.Series(pub)
    agree = float((proc.reindex(pub.index) == pub).mean())
    assert agree == 1.0, f'DrEval procedure does not reproduce the published CTRPv2 splits ({agree})'
    man['split_procedure_reproduces_published_CTRPv2_splits'] = {'cells': int(len(pub)), 'agreement': agree}

    # DepMap 24Q4 expression (feature-source control)
    mod = load_models()[['ModelID', 'RRID']].dropna().drop_duplicates('RRID').set_index('RRID').ModelID
    expr = pd.read_parquet(PROCESSED / 'features' / 'expression_top.parquet')
    all_cells = sorted(resp.cell.unique())
    mid = mod.reindex(all_cells)
    keep = [c for c, m in zip(all_cells, mid.values) if isinstance(m, str) and m in expr.index]
    dep = expr.loc[mid.reindex(keep).values]
    dep.index = pd.Index(keep, name='cell')
    dep = dep.loc[~dep.index.duplicated()]
    atomic_parquet(dep.astype('float32'), INP / 'feat_depmap_expr.parquet')
    man['features']['depmap_expr'] = {'cells': int(len(dep)), 'genes': int(dep.shape[1]),
                                      'source': 'data/processed/features/expression_top.parquet (DepMap 24Q4)'}

    # DrEval's expression files: identical within the CTRP/CCLE and GDSC families, different between them
    lm = pd.read_csv(DRE / 'meta' / 'gene_lists' / 'landmark_genes.csv').Symbol.tolist()
    ge = {}
    for ds in DATASETS:
        x = read_zip_csv(ds, 'gene_expression.csv', usecols=lambda c: c == 'cellosaurus_id' or c in set(lm))
        man['features'][f'dreval_expression_{ds}'] = {'rows': int(len(x)),
                                                     'duplicated_cell_rows': int(x.cellosaurus_id.duplicated().sum())}
        ge[ds] = x.groupby('cellosaurus_id').mean()
        dsc = set(raw[ds].cellosaurus_id)
        man['features'][f'dreval_expression_{ds}']['dataset_cells_with_expression'] = len(dsc & set(ge[ds].index))
        man['features'][f'dreval_expression_{ds}']['dataset_cells_with_depmap'] = len(dsc & set(dep.index))
    for a, b in [('CTRPv2', 'CTRPv1'), ('CTRPv2', 'CCLE'), ('GDSC1', 'GDSC2'), ('CTRPv2', 'GDSC2')]:
        com = ge[a].index.intersection(ge[b].index)
        g = ge[a].columns.intersection(ge[b].columns)
        diff = np.abs(ge[a].loc[com, g].values - ge[b].loc[com, g].values)
        man['notes'][f'expression_{a}_vs_{b}'] = {'shared_cells': int(len(com)), 'max_abs_diff': float(diff.max()),
                                                  'median_value_a': float(np.median(ge[a].values)),
                                                  'median_value_b': float(np.median(ge[b].values))}
    write_json(INP / 'manifest.json', man)
    print(man['split_procedure_reproduces_published_CTRPv2_splits'], man['notes'], sep='\n')


if __name__ == '__main__':
    main()
