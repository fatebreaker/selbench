"""Harmonise the four discovery screens (PRISM secondary, CTRP v2, GDSC1, GDSC2) to DepMap
compound and model identifiers.

Compound identity: DepMap PortalCompounds (24Q4) is the primary cross-dataset key. GDSC drug IDs
missing from it are attached by exact InChIKey connectivity block, then by normalised name/synonym,
to compounds already present in PRISM or CTRP. Cells: DepMap ModelID via PRISM depmap_id, CTRP
cell-line name (stripped CCLE name) and GDSC Sanger model / COSMIC identifiers.

Outputs (data/processed): discovery_long.csv.gz, compounds.csv, cells.csv, harmonize_summary.json.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from common import (OWN_RAW, PROCESSED, RAW, inchikey14, inchikey25, load_models, norm_cell, norm_name,
                    sha256, write_json)


def split_ids(s):
    out = []
    for x in str(s).split(';'):
        ds, _, i = x.strip().partition(':')
        if i:
            out.append((ds, i))
    return out


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    models = load_models()
    portal = pd.read_csv(OWN_RAW / 'depmap24q4' / 'PortalCompounds.csv')
    sample_rows = [(r.CompoundID, ds, i) for r in portal.itertuples() for ds, i in split_ids(r.SampleIDs)]
    sample = pd.DataFrame(sample_rows, columns=['dpc', 'dataset', 'sample_id'])
    brd13 = sample[sample.dataset == 'BRD'].assign(short=lambda d: d.sample_id.str[:13])
    brd13 = brd13.drop_duplicates('short').set_index('short').dpc

    records = []          # long table rows
    cpd_info = defaultdict(lambda: {'names': set(), 'smiles': set(), 'targets': set(), 'moa': set(), 'ids': set()})

    # ---------------- PRISM secondary ----------------
    p = pd.read_csv(RAW / 'prism' / 'secondary-screen-dose-response-curve-parameters.csv',
                    usecols=['broad_id', 'depmap_id', 'screen_id', 'auc', 'r2', 'name', 'moa', 'target', 'smiles',
                             'passed_str_profiling'])
    p = p[p.passed_str_profiling.astype(str).str.upper() == 'TRUE']
    p['dpc'] = p.broad_id.str[:13].map(brd13)
    prism_unmapped = sorted(p.loc[p.dpc.isna(), 'name'].unique().tolist())
    p = p.dropna(subset=['dpc', 'auc'])
    # Recommended technical redo (MTS010) replaces earlier screens for the same compound/cell.
    prio = {'MTS010': 0, 'HTS002': 1, 'MTS006': 2, 'MTS005': 3}
    p['prio'] = p.screen_id.map(prio).fillna(9)
    best = p.groupby(['dpc', 'depmap_id']).prio.transform('min')
    p = p[p.prio == best]
    agg = p.groupby(['dpc', 'depmap_id'], as_index=False).agg(auc=('auc', 'median'), n_rec=('auc', 'size'))
    for r in agg.itertuples():
        records.append(('PRISM', r.dpc, r.depmap_id, r.auc, np.nan, r.n_rec))
    for r in p.drop_duplicates('broad_id').itertuples():
        info = cpd_info[r.dpc]
        info['names'].add(r.name); info['ids'].add('BRD:' + r.broad_id)
        if isinstance(r.smiles, str):
            info['smiles'].add(r.smiles.split(',')[0].strip())
        if isinstance(r.target, str):
            info['targets'].update(t.strip() for t in r.target.split(','))
        if isinstance(r.moa, str):
            info['moa'].update(t.strip() for t in r.moa.split(','))

    # ---------------- CTRP v2 (author-corrected AUC) ----------------
    c = pd.read_csv(RAW / 'ctrp_outcomes' / 'new-abs-auc-with-qc.txt', sep='\t')
    ccl = pd.read_csv(RAW / 'ctrp_metadata' / 'mirror_cell_names.txt', sep='\t', header=None,
                      names=['index_name', 'ccl_name', 'master_ccl_id', 'histology', 'tissue'])
    informer = pd.read_excel(RAW / 'ctrp_metadata' / 'informer_set.xlsx')
    ctrp_map = sample[sample.dataset == 'CTRP'].drop_duplicates('sample_id').set_index('sample_id').dpc
    c['dpc'] = c.master_cpd_id.astype(str).map(ctrp_map)
    c = c.merge(ccl[['master_ccl_id', 'ccl_name']], on='master_ccl_id', how='left')
    stripped = models.assign(key=models.StrippedCellLineName.map(norm_cell))
    dup_keys = set(stripped.key[stripped.key.duplicated()])
    name_to_model = stripped[~stripped.key.isin(dup_keys)].set_index('key').ModelID
    c['depmap_id'] = c.ccl_name.map(norm_cell).map(name_to_model)
    ctrp_cell_unmapped = sorted(c.loc[c.depmap_id.isna(), 'ccl_name'].dropna().unique().tolist())
    c = c.dropna(subset=['dpc', 'depmap_id', 'area_under_curve'])
    agg = c.groupby(['dpc', 'depmap_id'], as_index=False).agg(auc=('area_under_curve', 'mean'),
                                                             n_rec=('area_under_curve', 'size'))
    for r in agg.itertuples():
        records.append(('CTRP', r.dpc, r.depmap_id, r.auc, np.nan, r.n_rec))
    inf = informer.assign(dpc=informer.master_cpd_id.astype(str).map(ctrp_map)).dropna(subset=['dpc'])
    for r in inf.itertuples():
        info = cpd_info[r.dpc]
        info['names'].add(r.cpd_name); info['ids'].add(f'CTRP:{r.master_cpd_id}')
        if isinstance(r.cpd_smiles, str):
            info['smiles'].add(r.cpd_smiles)
        if isinstance(r.gene_symbol_of_protein_target, str):
            info['targets'].update(t.strip() for t in r.gene_symbol_of_protein_target.split(';'))

    # ---------------- GDSC1 / GDSC2 ----------------
    harm = pd.read_csv(RAW / 'gdsc' / 'harmonized_compounds.csv')
    harm_parent = harm.dropna(subset=['parent']).drop_duplicates('DRUG_ID').set_index('DRUG_ID').parent
    screened = pd.read_csv(RAW / 'gdsc' / 'screened_compounds_rel_8.4.csv').drop_duplicates('DRUG_ID').set_index('DRUG_ID')
    # lookup tables for compounds already present via PRISM/CTRP/Portal
    key_to_dpc, name_to_dpc = {}, defaultdict(set)
    for dpc, info in cpd_info.items():
        for s in info['smiles']:
            k = inchikey14(s)
            if k:
                key_to_dpc.setdefault(k, set()).add(dpc)
        for n in info['names']:
            name_to_dpc[norm_name(n)].add(dpc)
    for r in portal.itertuples():
        name_to_dpc[norm_name(r.CompoundName)].add(r.CompoundID)
        for syn in str(r.Synonyms).split(';') if isinstance(r.Synonyms, str) else []:
            name_to_dpc[norm_name(syn)].add(r.CompoundID)
    gdsc_model = models.dropna(subset=['SangerModelID']).drop_duplicates('SangerModelID').set_index('SangerModelID').ModelID
    cosmic_model = models.dropna(subset=['COSMICID']).assign(COSMICID=lambda d: d.COSMICID.astype(int))
    cosmic_model = cosmic_model.drop_duplicates('COSMICID').set_index('COSMICID').ModelID
    gdsc_mapping_log = []
    for screen, path in [('GDSC1', RAW / 'gdsc1' / 'GDSC1_fitted_dose_response_24Jul22.csv'),
                         ('GDSC2', RAW / 'gdsc' / 'GDSC2_fitted_dose_response_24Jul22.csv')]:
        g = pd.read_csv(path)
        portal_map = sample[sample.dataset == screen].drop_duplicates('sample_id').set_index('sample_id').dpc
        id_to_dpc, how = {}, {}
        for did, dname in g[['DRUG_ID', 'DRUG_NAME']].drop_duplicates('DRUG_ID').itertuples(index=False):
            dpc, method = portal_map.get(str(did)), 'depmap_portal'
            if dpc is None and did in harm_parent.index:
                cands = key_to_dpc.get(inchikey14(harm_parent[did]), set())
                if len(cands) == 1:
                    dpc, method = next(iter(cands)), 'inchikey14'
            if dpc is None:
                names = {norm_name(dname)}
                if did in screened.index and isinstance(screened.loc[did, 'SYNONYMS'], str):
                    names |= {norm_name(x) for x in screened.loc[did, 'SYNONYMS'].split(',')}
                cands = set().union(*[name_to_dpc.get(n, set()) for n in names if n])
                if len(cands) == 1:
                    dpc, method = next(iter(cands)), 'name'
            if dpc is None:
                dpc, method = f'{screen}:{did}', 'unique_to_screen'
            id_to_dpc[did], how[did] = dpc, method
            gdsc_mapping_log.append({'screen': screen, 'DRUG_ID': did, 'DRUG_NAME': dname, 'dpc': dpc, 'method': method})
        g['dpc'] = g.DRUG_ID.map(id_to_dpc)
        g['depmap_id'] = g.SANGER_MODEL_ID.map(gdsc_model)
        miss = g.depmap_id.isna()
        g.loc[miss, 'depmap_id'] = g.loc[miss, 'COSMIC_ID'].map(cosmic_model)
        g = g.dropna(subset=['depmap_id', 'AUC'])
        agg = g.groupby(['dpc', 'depmap_id'], as_index=False).agg(auc=('AUC', 'mean'), ln_ic50=('LN_IC50', 'mean'),
                                                                 n_rec=('AUC', 'size'))
        for r in agg.itertuples():
            records.append((screen, r.dpc, r.depmap_id, r.auc, r.ln_ic50, r.n_rec))
        for did, dname in g[['DRUG_ID', 'DRUG_NAME']].drop_duplicates('DRUG_ID').itertuples(index=False):
            info = cpd_info[id_to_dpc[did]]
            info['names'].add(dname); info['ids'].add(f'{screen}:{did}')
            if did in harm_parent.index:
                info['smiles'].add(harm_parent[did])
            if did in screened.index and isinstance(screened.loc[did, 'TARGET'], str):
                info['targets'].update(t.strip() for t in screened.loc[did, 'TARGET'].split(','))
                info['moa'].add(str(screened.loc[did, 'TARGET_PATHWAY']))

    long = pd.DataFrame(records, columns=['screen', 'compound', 'depmap_id', 'auc', 'ln_ic50', 'n_records'])

    # CTRP fixed-ratio combinations are not single agents.
    combos = {d for d, info in cpd_info.items()
              if any(':' in str(n) and 'MOL/MOL' in str(n).upper() for n in info['names'])}
    long = long[~long.compound.isin(combos)]
    # Merge records of one structure (identical InChIKey connectivity + stereo blocks) under one key.
    n_rows = long.compound.value_counts()
    ik = {}
    for d, info in cpd_info.items():
        if d in combos or d not in n_rows.index:
            continue
        keys = [k for k in (inchikey25(s) for s in sorted(info['smiles'])) if k]
        if keys:
            ik[d] = max(set(keys), key=keys.count)
    merge = {}
    for key, group in pd.Series(ik).groupby(lambda d: ik[d]):
        members = sorted(group.index, key=lambda d: (-n_rows.get(d, 0), d))
        for d in members[1:]:
            merge[d] = members[0]
    long['compound'] = long.compound.replace(merge)
    for d, canon in merge.items():
        for k in ('names', 'smiles', 'targets', 'moa', 'ids'):
            cpd_info[canon][k] |= cpd_info[d][k]
    long = long.groupby(['screen', 'compound', 'depmap_id'], as_index=False).agg(
        auc=('auc', 'mean'), ln_ic50=('ln_ic50', 'mean'), n_records=('n_records', 'sum'))
    pd.DataFrame(sorted(merge.items()), columns=['merged', 'into']).to_csv(PROCESSED / 'compound_merges.csv', index=False)
    long['sens'] = -long['auc']           # higher = more sensitive
    long.to_csv(PROCESSED / 'discovery_long.csv.gz', index=False)

    portal_idx = portal.set_index('CompoundID')
    rows = []
    present = set(long.compound)
    for dpc, info in cpd_info.items():
        if dpc not in present:
            continue
        info['names'] = {str(n) for n in info['names'] if isinstance(n, (str, int, float)) and str(n) != 'nan'}
        pname = portal_idx.CompoundName.get(dpc) if dpc in portal_idx.index else None
        ptargets = portal_idx.GeneSymbolOfTargets.get(dpc) if dpc in portal_idx.index else None
        targets = set(info['targets'])
        if isinstance(ptargets, str):
            targets |= {t.strip() for t in ptargets.split(';')}
        targets = sorted(t for t in targets if t and t.lower() not in ('nan', 'none', 'unknown'))
        smiles = sorted(info['smiles'])
        keys = sorted({k for k in (inchikey14(s) for s in smiles) if k})
        rows.append({'compound': dpc, 'name': pname or sorted(info['names'])[0], 'all_names': '|'.join(sorted(info['names'])),
                     'targets': ';'.join(targets), 'moa': '|'.join(sorted(m for m in info['moa'] if m and m != 'nan')),
                     'smiles': smiles[0] if smiles else '', 'inchikey14': '|'.join(keys), 'ids': '|'.join(sorted(info['ids'])),
                     'target_or_mechanism': portal_idx.TargetOrMechanism.get(dpc) if dpc in portal_idx.index else ''})
    comp = pd.DataFrame(rows).sort_values('compound')
    screens_per = long.groupby('compound').screen.agg(lambda s: '|'.join(sorted(set(s))))
    comp['screens'] = comp.compound.map(screens_per)
    comp.to_csv(PROCESSED / 'compounds.csv', index=False)

    cells = models[models.ModelID.isin(long.depmap_id)][['ModelID', 'CellLineName', 'StrippedCellLineName', 'OncotreeLineage',
                                                          'OncotreePrimaryDisease', 'OncotreeSubtype', 'SangerModelID',
                                                          'COSMICID', 'GrowthPattern']]
    cells.to_csv(PROCESSED / 'cells.csv', index=False)
    pd.DataFrame(gdsc_mapping_log).to_csv(PROCESSED / 'gdsc_compound_mapping.csv', index=False)

    per_screen = long.groupby('screen').agg(pairs=('sens', 'size'), compounds=('compound', 'nunique'),
                                            cells=('depmap_id', 'nunique'))
    write_json(PROCESSED / 'harmonize_summary.json', {
        'per_screen': per_screen.to_dict(orient='index'),
        'prism_unmapped_names': prism_unmapped,
        'ctrp_unmapped_cell_names': ctrp_cell_unmapped,
        'gdsc_mapping_methods': pd.DataFrame(gdsc_mapping_log).groupby(['screen', 'method']).size().to_dict(),
        'ctrp_combinations_excluded': len(combos), 'structure_merges': len(merge),
        'inputs_sha256': {str(pth.relative_to(RAW.parent.parent)): sha256(pth) for pth in [
            RAW / 'prism' / 'secondary-screen-dose-response-curve-parameters.csv',
            RAW / 'ctrp_outcomes' / 'new-abs-auc-with-qc.txt',
            RAW / 'gdsc1' / 'GDSC1_fitted_dose_response_24Jul22.csv',
            RAW / 'gdsc' / 'GDSC2_fitted_dose_response_24Jul22.csv']},
    })
    print(per_screen)


if __name__ == '__main__':
    main()
