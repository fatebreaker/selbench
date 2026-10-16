"""Source-specific loaders for the two confirmation screens (written after retrieval; binds file
columns to the registered fields and implements the registered identity rules, protocol section 2).

gCSI (GR dataset v1.3, metrics table): `meanviability` equals the mean relative cell count over
the tested concentrations (verified against gCSI_GRvalues: correlation 1.0, values ~0.7), i.e. a
normalised area *under* the relative-count curve, despite the vignette's "100*(1-...)" wording.
Registered rule: sensitivity = -area under the curve = -meanviability. Secondary: GR_AOC (higher =
more sensitive). Experiments are averaged per cell line and drug.
NCI-60 (CellMiner DTP raw GI50, unit -log10 M): sensitivity = reported value. Rows (experiments)
are averaged per NSC and cell line. Secondary: same data restricted to compounds passing CellMiner's
compound QC ("Failure reason" == "-").
Outputs: data/processed/confirmation/{gCSI,gCSI_GRAOC,NCI60,NCI60_cellminerQC}_long.csv.gz and
identity-mapping tables.
"""
import re
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

from common import OWN_RAW, PROCESSED, load_models, norm_cell, norm_name, write_json

CONF = OWN_RAW / 'confirmation'
OUT = PROCESSED / 'confirmation'
NCI_ALIASES = {'A549ATCC': 'A549', 'HL60TB': 'HL60', '7860': '786O', 'U251': 'U251MG', 'SR': 'SR786',
               'MDAMB435': 'MDAMB435S', 'OVCAR3': 'NIHOVCAR3'}
NCI_EXCLUDE = {'NCIADRRES', 'MDAN'}
GCSI_FIXED = {'crizotinib': 'DPC-001953', 'etoposide': 'DPC-002651'}


def drug_index():
    comp = pd.read_csv(PROCESSED / 'compounds.csv')
    portal = pd.read_csv(OWN_RAW / 'depmap24q4' / 'PortalCompounds.csv')
    merges = pd.read_csv(PROCESSED / 'compound_merges.csv').set_index('merged').into
    present = set(comp.compound)
    idx = defaultdict(set)
    for r in comp.itertuples():
        for n in str(r.all_names).split('|') + [r.name]:
            idx[norm_name(n)].add(r.compound)
    for r in portal.itertuples():
        key = merges.get(r.CompoundID, r.CompoundID)
        if key not in present:
            continue
        for n in [r.CompoundName] + (str(r.Synonyms).split(';') if isinstance(r.Synonyms, str) else []):
            idx[norm_name(n)].add(key)
    idx.pop('', None)
    return idx, comp.set_index('compound')


def cell_index():
    m = load_models()
    m['key'] = m.StrippedCellLineName.map(norm_cell)
    dup = set(m.key[m.key.duplicated()])
    return m[~m.key.isin(dup)].set_index('key').ModelID


def load_gcsi(didx, cidx):
    g = pd.read_csv(CONF / 'OUTPUT' / 'gCSI_GRmetrics_v1.3.tsv', sep='\t')
    drug_map, log = {}, []
    for name in sorted(g.DrugName.unique()):
        key = norm_name(name)
        cands = didx.get(key, set())
        if name.lower() in GCSI_FIXED:
            chosen, how = GCSI_FIXED[name.lower()], 'registered_choice'
        elif len(cands) == 1:
            chosen, how = next(iter(cands)), 'unique_name'
        elif not cands:
            chosen, how = f'gCSI:{name}', 'not_in_discovery'
        else:
            chosen, how = None, 'ambiguous_excluded'
        drug_map[name] = chosen
        log.append({'source_name': name, 'compound': chosen, 'method': how, 'candidates': '|'.join(sorted(cands))})
    cells = {n: cidx.get(norm_cell(n)) for n in g.CellLineName.unique()}
    g['compound'] = g.DrugName.map(drug_map)
    g['depmap_id'] = g.CellLineName.map(cells)
    kept = g.dropna(subset=['compound', 'depmap_id'])
    prim = kept.groupby(['compound', 'depmap_id'], as_index=False).meanviability.mean()
    prim['sens'] = -prim.pop('meanviability')
    sec = kept.groupby(['compound', 'depmap_id'], as_index=False).GR_AOC.mean().rename(columns={'GR_AOC': 'sens'})
    cell_log = pd.DataFrame({'source_name': list(cells), 'depmap_id': list(cells.values())})
    return prim, sec, pd.DataFrame(log), cell_log, {'rows': len(g), 'durations': sorted(g.TrtDuration.dropna().unique().tolist())}


def load_nci60(didx, cidx):
    xlsx = CONF / 'nci60_raw' / 'output' / 'DTP_NCI60_RAW.xlsx'
    if not xlsx.exists():
        zipfile.ZipFile(CONF / 'DTP_NCI60_RAW.zip').extractall(CONF / 'nci60_raw')
    raw = pd.read_excel(xlsx, header=9)
    meta_cols = ['NSC #', 'Drug name', 'FDA status', 'Mechanism of action', 'PubChem SID', 'Total probes',
                 'Total after quality control', 'Failure reason', 'Experiment name']
    cell_cols = [c for c in raw.columns if c not in meta_cols]
    cell_map, cell_log = {}, []
    for c in cell_cols:
        key = norm_cell(str(c).split(':', 1)[-1])
        key = NCI_ALIASES.get(key, key)
        mid = None if key in NCI_EXCLUDE else cidx.get(key)
        cell_map[c] = mid
        cell_log.append({'source_name': c, 'key': key, 'depmap_id': mid})
    raw['NSC'] = raw['NSC #'].astype(str).str.replace(r'\.0$', '', regex=True)
    qc_pass = raw.groupby('NSC')['Failure reason'].agg(lambda s: all(str(x).strip() == '-' for x in s))
    names = raw.groupby('NSC')['Drug name'].first()
    vals = raw[['NSC'] + cell_cols].copy()
    for c in cell_cols:
        vals[c] = pd.to_numeric(vals[c], errors='coerce')
    per_nsc = vals.groupby('NSC')[cell_cols].mean()
    n_cells = per_nsc.notna().sum(axis=1)
    # NSC -> discovery compound; several NSCs for one compound: most measured cells, then lowest NSC
    nsc_map, log = {}, []
    for nsc, name in names.items():
        cands = didx.get(norm_name(str(name)), set()) if isinstance(name, str) and name.strip() not in ('-', '') else set()
        if len(cands) == 1:
            nsc_map[nsc] = next(iter(cands)); how = 'unique_name'
        elif not cands:
            nsc_map[nsc] = f'NSC:{nsc}'; how = 'not_in_discovery'
        else:
            nsc_map[nsc] = None; how = 'ambiguous_excluded'
        log.append({'nsc': nsc, 'source_name': name, 'compound': nsc_map[nsc], 'method': how,
                    'candidates': '|'.join(sorted(cands)), 'n_cells': int(n_cells.get(nsc, 0)),
                    'cellminer_qc_pass': bool(qc_pass.get(nsc, False))})
    log = pd.DataFrame(log)
    disc = log[log.method == 'unique_name'].copy()
    disc['nsc_int'] = pd.to_numeric(disc.nsc, errors='coerce')
    disc = disc.sort_values(['compound', 'n_cells', 'nsc_int'], ascending=[True, False, True])
    keep_disc = set(disc.drop_duplicates('compound').nsc)
    log['used'] = log.nsc.isin(keep_disc) | (log.method == 'not_in_discovery')
    used = log[log.used]
    long = per_nsc.loc[used.nsc].rename(columns=cell_map)
    long = long.loc[:, [c for c in long.columns if isinstance(c, str) and c.startswith('ACH-')]]
    long.index = used.compound.values
    long = long.stack().rename('sens').reset_index().rename(columns={'level_0': 'compound', 'level_1': 'depmap_id'})
    long = long.groupby(['compound', 'depmap_id'], as_index=False).sens.mean()
    qc_comps = set(used.loc[used.cellminer_qc_pass, 'compound'])
    return long, long[long.compound.isin(qc_comps)], log, pd.DataFrame(cell_log)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    didx, comp = drug_index()
    cidx = cell_index()
    gp, gs, glog, gcl, ginfo = load_gcsi(didx, cidx)
    gp.to_csv(OUT / 'gCSI_long.csv.gz', index=False); gs.to_csv(OUT / 'gCSI_GRAOC_long.csv.gz', index=False)
    glog.to_csv(OUT / 'gCSI_drug_mapping.csv', index=False); gcl.to_csv(OUT / 'gCSI_cell_mapping.csv', index=False)
    nl, nq, nlog, ncl = load_nci60(didx, cidx)
    nl.to_csv(OUT / 'NCI60_long.csv.gz', index=False); nq.to_csv(OUT / 'NCI60_cellminerQC_long.csv.gz', index=False)
    nlog.to_csv(OUT / 'NCI60_drug_mapping.csv', index=False); ncl.to_csv(OUT / 'NCI60_cell_mapping.csv', index=False)
    summary = {
        'gCSI': {**ginfo, 'drug_methods': glog.method.value_counts().to_dict(),
                 'cells_mapped': int(gcl.depmap_id.notna().sum()), 'cells_total': len(gcl),
                 'pairs': len(gp), 'compounds': int(gp.compound.nunique())},
        'NCI60': {'nsc_total': len(nlog), 'drug_methods': nlog.method.value_counts().to_dict(),
                  'discovery_compounds_used': int(nlog[nlog.used & (nlog.method == 'unique_name')].compound.nunique()),
                  'cells_mapped': int(ncl.depmap_id.notna().sum()), 'cells_total': len(ncl),
                  'pairs': len(nl), 'compounds': int(nl.compound.nunique()),
                  'qc_compounds': int(nq.compound.nunique())},
    }
    write_json(OUT / 'load_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
