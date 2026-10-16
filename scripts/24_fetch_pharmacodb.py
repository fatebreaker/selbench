"""Amendment 1: retrieve CCLE, FIMM and GRAY responses from the PharmacoDB GraphQL API and map identities by the
registered rules (written before retrieval). Endpoint: AAC (sensitivity = AAC), averaged per cell line and
compound. Outputs: data/raw/pharmacodb_responses/*.json, data/processed/confirmation/{CCLE,FIMM,GRAY}_long.csv.gz
and mapping tables.
"""
import importlib.util
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from common import OWN_RAW, PROCESSED, norm_cell, norm_name, write_json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lc', HERE / '06_load_confirmation.py')
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
API = 'https://pharmacodb.ca/graphql'
DATASETS = {'CCLE': 1, 'FIMM': 3, 'GRAY': 6}
CHOICES = {'crizotinib': 'DPC-001953', 'etoposide': 'DPC-002651', 'epothiloneb': 'DPC-002542'}
ALIASES = dict(lc.NCI_ALIASES, U87MGATCC='U87MG', JURKATE61='JURKAT')
RAWDIR = OWN_RAW / 'pharmacodb_responses'
OUT = PROCESSED / 'confirmation'


def gql(query, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(API, data=json.dumps({'query': query}).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def cell_key(name):
    k = norm_cell(re.sub(r'\[.*?\]', '', str(name)))
    return ALIASES.get(k, k)


def main():
    RAWDIR.mkdir(parents=True, exist_ok=True)
    didx, comp = lc.drug_index()
    cidx = lc.cell_index()
    summary = {}
    for ds, dsid in DATASETS.items():
        compounds = json.loads((OWN_RAW / 'pharmacodb_metadata' / f'{ds}_COMPOUND.json').read_text())['data']['type_tested_on_dataset']['list']
        rows = []
        for c in compounds:
            cache = RAWDIR / f'{ds}_{c["id"]}.json'
            if cache.exists():
                data = json.loads(cache.read_text())
            else:
                data = gql('{ experiments(compoundId: %d, all: true) { dataset { name } cell_line { name } compound { name } '
                           'profile { AAC IC50 } } }' % c['id'])
                cache.write_text(json.dumps(data))
            for e in data['data']['experiments']:
                if e['dataset']['name'] == ds and e['profile'] and e['profile']['AAC'] is not None:
                    rows.append({'compound_name': c['name'], 'cell_name': e['cell_line']['name'], 'aac': e['profile']['AAC'],
                                 'ic50': e['profile']['IC50']})
        raw = pd.DataFrame(rows)
        # identity mapping (registered rules)
        dmap, dlog = {}, []
        for name in sorted(raw.compound_name.unique()):
            key = norm_name(name); cands = didx.get(key, set())
            if key in CHOICES:
                dmap[name], how = CHOICES[key], 'registered_choice'
            elif len(cands) == 1:
                dmap[name], how = next(iter(cands)), 'unique_name'
            elif not cands:
                dmap[name], how = f'{ds}:{name}', 'not_in_discovery'
            else:
                dmap[name], how = None, 'ambiguous_excluded'
            dlog.append({'source_name': name, 'compound': dmap[name], 'method': how, 'candidates': '|'.join(sorted(cands))})
        cmap = {n: cidx.get(cell_key(n)) for n in raw.cell_name.unique()}
        raw['compound'] = raw.compound_name.map(dmap); raw['depmap_id'] = raw.cell_name.map(cmap)
        kept = raw.dropna(subset=['compound', 'depmap_id'])
        long = kept.groupby(['compound', 'depmap_id'], as_index=False).aac.mean().rename(columns={'aac': 'sens'})
        long.to_csv(OUT / f'{ds}_long.csv.gz', index=False)
        pd.DataFrame(dlog).to_csv(OUT / f'{ds}_drug_mapping.csv', index=False)
        pd.DataFrame({'source_name': list(cmap), 'depmap_id': list(cmap.values())}).to_csv(OUT / f'{ds}_cell_mapping.csv', index=False)
        summary[ds] = {'experiments': len(raw), 'pairs': len(long), 'compounds': int(long.compound.nunique()),
                       'discovery_compounds': int(sum(1 for d in long.compound.unique() if str(d).startswith(('DPC-', 'GDSC')))),
                       'cells_mapped': int(sum(v is not None for v in cmap.values())), 'cells_total': len(cmap),
                       'drug_methods': pd.DataFrame(dlog).method.value_counts().to_dict()}
        print(ds, summary[ds], flush=True)
    write_json(OUT / 'pharmacodb_load_summary.json', summary)


if __name__ == '__main__':
    main()
