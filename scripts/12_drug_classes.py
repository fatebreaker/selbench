"""Which mechanisms carry transferable selectivity? Coarse mechanism classes from target/MOA
annotations; per class, discovery cross-screen selective agreement, within-screen fitness share and
confirmation agreement (consensus measurement vs gCSI / NCI-60). Output: results/drug_classes/.
"""
import re

import numpy as np
import pandas as pd

from common import PROCESSED, RESULTS, write_json

CLASSES = [  # first match wins; patterns are searched in targets + MOA text (upper case)
    ('MAPK (RAF/MEK/ERK)', r'\bBRAF\b|\bRAF1?\b|MAP2K|\bMEK\b|MEK1|MEK2|\bERK\b|MAPK1\b|MAPK3\b|RAF INHIBITOR|MEK INHIBITOR'),
    ('EGFR/ERBB', r'\bEGFR\b|ERBB2|ERBB3|ERBB4|\bHER2\b'),
    ('MDM2–p53', r'\bMDM2\b|MDM4'),
    ('PI3K/AKT/mTOR', r'PIK3|PI3K|\bAKT[123]?\b|\bMTOR\b|MTORC'),
    ('BCL2 family / IAP', r'\bBCL2\b|BCL2L1|BCL-XL|\bMCL1\b|\bXIAP\b|BIRC[2-7]|\bIAP\b|BCL-2'),
    ('DNA damage response', r'\bPARP[12]?\b|\bATR\b|\bATM\b|CHEK[12]|\bCHK[12]?\b|\bWEE1\b|PRKDC|DNA-?PK'),
    ('Topoisomerase', r'TOP1|TOP2|TOPOISOMERASE'),
    ('Antimetabolite', r'\bDHFR\b|\bTYMS\b|RRM[12]|ANTIMETABOLITE|DNA SYNTHESIS|NUCLEOSIDE|POL[ABDE]\b|RIBONUCLEOTIDE'),
    ('Mitosis / microtubule', r'TUBULIN|TUBB|TUBA|MICROTUBULE|\bKIF11\b|\bPLK[1-4]?\b|AURK|AURORA|KINESIN'),
    ('CDK', r'\bCDK[0-9]*\b'),
    ('Epigenetic (HDAC/BET/DNMT)', r'\bHDAC|\bBRD[234]\b|BROMODOMAIN|\bDNMT|EZH2|\bBET\b|KDM|DOT1L|PRMT'),
    ('Proteasome / HSP90', r'PSMB|PROTEASOME|HSP90'),
    ('Other kinase', r'\bALK\b|\bMET\b|FGFR|\bKIT\b|PDGFR|VEGFR|KDR|\bABL1?\b|\bSRC\b|FLT3|IGF1R|\bJAK[123]?\b|\bBTK\b|\bSYK\b|KINASE'),
]


def classify(text):
    t = text.upper()
    for name, pat in CLASSES:
        if re.search(pat, t):
            return name
    return 'Other / non-oncology'


def main():
    out = RESULTS / 'drug_classes'
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(PROCESSED / 'compounds.csv').set_index('compound')
    text = (comp.targets.fillna('') + ' ' + comp.moa.fillna('') + ' ' + comp.target_or_mechanism.fillna(''))
    cls = text.map(classify)
    pairs = pd.read_csv(RESULTS / 'decomposition' / 'cross_screen_pairs.csv')
    stats = pd.read_csv(RESULTS / 'decomposition' / 'drug_screen_stats.csv')
    rep = pairs.groupby('compound').agg(rho_sel=('rho_sel', 'mean'), rho_raw=('rho_raw', 'mean'), n_pairs=('rho_sel', 'size'))
    fit = stats.groupby('compound').fitness_r2.median()
    act = stats.groupby('compound').frac_auc_below_0p8.max()
    df = rep.join(fit).join(act)
    df['class'] = cls.reindex(df.index).fillna('Other / non-oncology')
    df['name'] = comp.name.reindex(df.index)
    for scr in ['gCSI', 'NCI60']:
        sc = pd.read_csv(RESULTS / f'confirmation_{scr}' / 'scores.csv')
        df[f'confirm_{scr}'] = sc[sc.family == 'rep_sel_cons'].set_index('compound').sel.reindex(df.index)
        df[f'model_{scr}'] = sc[sc.family == 'ridge_sel_cons'].set_index('compound').sel.reindex(df.index)
    df.to_csv(out / 'drug_table.csv')
    summ = df.groupby('class').agg(drugs=('rho_sel', 'size'), median_rho_sel=('rho_sel', 'median'),
                                   frac_reproducible=('rho_sel', lambda s: float((s >= 0.3).mean())),
                                   median_fitness_r2=('fitness_r2', 'median'),
                                   gcsi_n=('confirm_gCSI', 'count'), gcsi_median=('confirm_gCSI', 'median'),
                                   nci_n=('confirm_NCI60', 'count'), nci_median=('confirm_NCI60', 'median'))
    summ = summ.sort_values('median_rho_sel', ascending=False)
    summ.to_csv(out / 'class_summary.csv')
    allstats = {'multi_screen_drugs': int(len(df)), 'frac_reproducible': float((df.rho_sel >= 0.3).mean()),
                'frac_nonreproducible': float((df.rho_sel < 0.15).mean()),
                'top_drugs': df.sort_values('rho_sel', ascending=False).head(25)[['name', 'class', 'rho_sel']].reset_index().to_dict(orient='records')}
    write_json(out / 'summary.json', allstats)
    print(summ.round(3).to_string())
    print(allstats['frac_reproducible'], allstats['frac_nonreproducible'], len(df))


if __name__ == '__main__':
    main()
