"""Amendment 1 BeatAML loader (written after retrieval; binds columns to the registered fields only).

beataml_probit_curve_fits_v4_dbgap.txt: sample = dbgap_rnaseq_sample (restricted to the 671 frozen samples),
inhibitor = inhibitor, single agents only (type == 'single-agent' and no ' - ' / ' + ' in the name),
sensitivity = -auc, replicates averaged. Inhibitors map to discovery compounds by unique normalized names;
others are kept as 'BA:<name>' for the fitness axis. Secondary file: paper_inclusion == TRUE only.
Outputs: data/processed/beataml/beataml_long.csv.gz, beataml_paperinclusion_long.csv.gz, drug_mapping.csv.
"""
import importlib.util
from pathlib import Path

import pandas as pd

from common import OWN_RAW, PROCESSED, RESULTS, norm_name, write_json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lc', HERE / '06_load_confirmation.py')
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)


def main():
    out = PROCESSED / 'beataml'; out.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(OWN_RAW / 'beataml_responses' / 'beataml_probit_curve_fits_v4_dbgap.txt', sep='\t')
    samples = set(pd.read_csv(RESULTS / 'beataml_transfer' / 'samples.csv').dbgap_rnaseq_sample)
    f = f[(f['type'] == 'single-agent') & ~f.inhibitor.str.contains(r' - | \+ ', regex=True)]
    f = f[f.dbgap_rnaseq_sample.isin(samples)]
    didx, comp = lc.drug_index()
    dmap, log = {}, []
    for name in sorted(f.inhibitor.unique()):
        cands = didx.get(norm_name(name), set())
        if len(cands) == 1:
            dmap[name], how = next(iter(cands)), 'unique_name'
        elif not cands:
            dmap[name], how = f'BA:{name}', 'not_in_discovery'
        else:
            dmap[name], how = None, 'ambiguous_excluded'
        log.append({'source_name': name, 'compound': dmap[name], 'method': how, 'candidates': '|'.join(sorted(cands))})
    f['compound'] = f.inhibitor.map(dmap)
    f = f.dropna(subset=['compound', 'auc'])
    f['sens'] = -f.auc
    for name, sub in [('beataml_long', f), ('beataml_paperinclusion_long', f[f.paper_inclusion.astype(str).str.upper() == 'TRUE'])]:
        long = sub.groupby(['compound', 'dbgap_rnaseq_sample'], as_index=False).sens.mean().rename(columns={'dbgap_rnaseq_sample': 'sample'})
        long.to_csv(out / f'{name}.csv.gz', index=False)
    pd.DataFrame(log).to_csv(out / 'drug_mapping.csv', index=False)
    s = {'rows': len(f), 'samples': int(f.dbgap_rnaseq_sample.nunique()), 'inhibitors': int(f.inhibitor.nunique()),
         'methods': pd.DataFrame(log).method.value_counts().to_dict()}
    write_json(out / 'load_summary.json', s)
    print(s)


if __name__ == '__main__':
    main()
