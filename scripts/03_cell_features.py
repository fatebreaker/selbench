"""Basal cell-line feature blocks from DepMap 24Q4 (unsupervised selection only; no response data).

Blocks: expression (top-variance protein-coding genes), mutations (hotspot or damaging, genes mutated
in >=1% of profiled models), relative copy number (log2 ratio, top-variance genes), lineage one-hot, CRISPR gene effect
(top-variance genes, kept separate) and the CRISPR-inferred growth rate.
"""
import numpy as np
import pandas as pd

from common import OWN_RAW, PROCESSED, RAW, load_models, write_json

N_EXPR, N_CN, N_CRISPR = 5000, 1000, 3000


def strip_gene(cols):
    return [c.split(' (')[0] for c in cols]


def top_var(df, n):
    v = df.var(axis=0).sort_values(ascending=False)
    return df[v.index[:n]]


def main():
    out = PROCESSED / 'features'
    out.mkdir(parents=True, exist_ok=True)
    dm = OWN_RAW / 'depmap24q4'
    expr = pd.read_csv(dm / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv', index_col=0)
    expr.columns = strip_gene(expr.columns)
    expr = expr.loc[:, ~expr.columns.duplicated()]
    expr.astype('float32').to_parquet(out / 'expression_all.parquet')
    top_var(expr, N_EXPR).astype('float32').to_parquet(out / 'expression_top.parquet')

    hot = pd.read_csv(dm / 'OmicsSomaticMutationsMatrixHotspot.csv', index_col=0)
    dam = pd.read_csv(dm / 'OmicsSomaticMutationsMatrixDamaging.csv', index_col=0)
    hot.columns, dam.columns = strip_gene(hot.columns), strip_gene(dam.columns)
    hot = hot.loc[:, ~hot.columns.duplicated()]; dam = dam.loc[:, ~dam.columns.duplicated()]
    genes = sorted(set(hot.columns) | set(dam.columns))
    idx = sorted(set(hot.index) | set(dam.index))
    mut = (hot.reindex(index=idx, columns=genes).fillna(0) > 0) | (dam.reindex(index=idx, columns=genes).fillna(0) > 0)
    mut = mut.astype('float32')
    mut = mut.loc[:, mut.mean(axis=0) >= 0.01]
    mut.to_parquet(out / 'mutation.parquet')
    hot_bin = (hot > 0).astype('float32')
    hot_bin.loc[:, hot_bin.sum(axis=0) >= 3].to_parquet(out / 'hotspot.parquet')

    # relative gene-level copy number (ratio to ploidy); broader model coverage than ABSOLUTE calls
    cn = pd.read_csv(dm / 'OmicsCNGene.csv', index_col=0)
    cn.columns = strip_gene(cn.columns)
    cn = cn.loc[:, ~cn.columns.duplicated()]
    cn = np.log2(cn.astype('float32').clip(lower=0.05))
    cn = cn.loc[:, cn.notna().mean(axis=0) > 0.95]
    cn = cn.fillna(cn.median())
    top_var(cn, N_CN).to_parquet(out / 'copy_number.parquet')

    crispr = pd.read_csv(RAW / 'depmap24q4_outcomes' / 'CRISPRGeneEffect.csv', index_col=0)
    crispr.columns = strip_gene(crispr.columns)
    crispr = crispr.loc[:, ~crispr.columns.duplicated()]
    crispr.astype('float32').to_parquet(out / 'crispr_all.parquet')
    crispr_f = crispr.loc[:, crispr.notna().mean(axis=0) > 0.98]
    crispr_f = crispr_f.fillna(crispr_f.mean())
    top_var(crispr_f, N_CRISPR).astype('float32').to_parquet(out / 'crispr_top.parquet')

    models = load_models().set_index('ModelID')
    lin = pd.get_dummies(models.OncotreeLineage.fillna('Unknown')).astype('float32')
    lin.to_parquet(out / 'lineage.parquet')
    growth = pd.read_csv(dm / 'CRISPRInferredModelGrowthRate.csv', index_col=0)
    growth['growth_rate'] = growth.mean(axis=1)
    growth[['growth_rate']].to_csv(out / 'growth_rate.csv')
    summary = {'expression': list(expr.shape), 'mutation': list(mut.shape), 'copy_number_kept': N_CN,
               'crispr': list(crispr.shape), 'lineage': list(lin.shape), 'growth_rate_models': int(growth.shape[0])}
    write_json(out / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
