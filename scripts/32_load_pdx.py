"""Amendment 2 PDX_Bruna loader (written BEFORE retrieval; binds DrEval's response columns to the registered fields).

Input: data/raw/pdx_bruna_responses/PDX_Bruna.csv in DrEval's curated format (one row per model and drug; columns
include cell_line_name, pubchem_id, AUC_curvecurator and LN_IC50_curvecurator).
Primary endpoint: CurveCurator AUC (first present of AUC_curvecurator, AUC). Orientation: if the median over drugs
(>= 10 jointly measured models) of Spearman(AUC, LN_IC50_curvecurator) is positive, the AUC is an area under the
viability curve and sensitivity = -AUC; if negative, sensitivity = AUC; if no ln IC50 column exists, sensitivity = -AUC
(CurveCurator AUC is an area under the relative-viability curve; on DrEval's CTRPv2 the median Spearman is +0.75).
If no AUC column exists, the primary endpoint is -LN_IC50_curvecurator. Secondary endpoint: -LN_IC50_curvecurator.
Replicate rows (same model and drug) are averaged. Drugs map through the frozen results/pdx_transfer/drug_map.csv;
ambiguous drugs are excluded; drugs outside discovery are kept as 'PDX:<pubchem_id>' for the PDX fitness axis.
Outputs: data/processed/pdx_bruna/{pdx_long.csv.gz, pdx_lnic50_long.csv.gz, load_summary.json}.
"""
import numpy as np
import pandas as pd

from common import OWN_RAW, PROCESSED, RESULTS, write_json

SRC = OWN_RAW / 'pdx_bruna_responses' / 'PDX_Bruna.csv'
OUT = PROCESSED / 'pdx_bruna'


def first(cols, names):
    return next((c for c in names if c in cols), None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(SRC, dtype={'pubchem_id': str}, low_memory=False)
    dm = pd.read_csv(RESULTS / 'pdx_transfer' / 'drug_map.csv', dtype={'pubchem_id': str})
    dm = dm[dm.method != 'ambiguous_excluded'].set_index('pubchem_id').compound
    f['compound'] = f.pubchem_id.map(dm)
    f = f.dropna(subset=['compound'])
    auc = first(f.columns, ['AUC_curvecurator', 'AUC'])
    lnic = first(f.columns, ['LN_IC50_curvecurator', 'LN_IC50'])
    info = {'rows': len(f), 'auc_column': auc, 'ln_ic50_column': lnic}
    if auc is not None:
        sign = -1.0
        if lnic is not None:
            g = f.dropna(subset=[auc, lnic]).groupby('pubchem_id')
            rho = g.apply(lambda x: x[auc].corr(x[lnic], method='spearman') if len(x) >= 10 else np.nan).dropna()
            info['orientation_median_spearman_auc_lnic50'] = float(rho.median()) if len(rho) else None
            if len(rho) and rho.median() < 0:
                sign = 1.0
        f['sens'] = sign * pd.to_numeric(f[auc], errors='coerce')
        info['primary'] = f'{"-" if sign < 0 else "+"}{auc}'
    else:
        f['sens'] = -pd.to_numeric(f[lnic], errors='coerce')
        info['primary'] = f'-{lnic}'
    long = f.dropna(subset=['sens']).groupby(['compound', 'cell_line_name'], as_index=False).sens.mean()
    long.rename(columns={'cell_line_name': 'sample'}).to_csv(OUT / 'pdx_long.csv.gz', index=False)
    if lnic is not None:
        f['sens_ic'] = -pd.to_numeric(f[lnic], errors='coerce')
        li = f.dropna(subset=['sens_ic']).groupby(['compound', 'cell_line_name'], as_index=False).sens_ic.mean()
        li.rename(columns={'cell_line_name': 'sample', 'sens_ic': 'sens'}).to_csv(OUT / 'pdx_lnic50_long.csv.gz', index=False)
    info.update({'models': int(long.cell_line_name.nunique()), 'compounds': int(long.compound.nunique()),
                 'pairs': len(long)})
    write_json(OUT / 'load_summary.json', info)
    print(info)


if __name__ == '__main__':
    main()
