"""Amendment 3: checks of 71_load_pdm.py and 72_pdm_confirm.py on SIMULATED responses (run before any real response
of these datasets was retrieved; no real outcome enters).

For each dataset, CoderData-format response tables are simulated for the real in vitro samples (organoids, 3D-MEDS,
xenograft-derived organoids) and the real drug identifiers: latent sensitivity = drug offset + patient axis (sd 1) +
signal x the dataset-standardised frozen patient-level pred_sel of the compound + noise (sd 1); fit_auc = logistic(-latent)
(an area under viability), fit_ic50 = -6 - latent + noise; 10% of pairs missing; 30% duplicated as replicates. The
simulated tables live only in a temporary directory. Runs: 40 null runs (signal 0) and 20 runs at signals 0.10 and
0.15, each with 1,000 permutations and 50 bootstrap draws (the registered run uses 10,000 and 2,000).
An edge-case run checks the registered rules for a
published_auc-only dataset, a dataset with too few drugs, a minority time point, non-numeric values and in vivo rows.
Outputs: results/pdm_simulation/{runs.csv, summary.json, edge_cases.json}.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, write_json

HERE = Path(__file__).resolve().parent
OUT = RESULTS / 'pdm_simulation'
IN_VITRO = ['patient derived organoid', '3D-MEDS', 'xenograft derived organoid']


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


L = _module('pdm_loader_sim', '71_load_pdm.py')
C = _module('pdm_confirm_sim', '72_pdm_confirm.py')


def simulate(root, signal, rng, edge=False):
    """edge=True adds the registered edge cases: pancreatic reports only published_auc; sarcoma has 5 drugs (dataset
    excluded); liver has a minority second time point and non-numeric values; bladder has in vivo xenograft rows with
    a categorical metric (excluded)."""
    src = root / 'raw'
    src.mkdir(parents=True, exist_ok=True)
    for ds in L.DATASETS:
        s = pd.read_csv(L.INP / f'{ds}_samples.csv', dtype=str).drop_duplicates('improve_sample_id')
        s = s[s.model_type.isin(IN_VITRO)].assign(patient=lambda x: x.common_name.map(L.patient_key))
        dm = pd.read_csv(L.RESULTS / 'pdm_transfer' / ds / 'drug_map.csv', dtype=str)
        Pp = C.patient_predictions(ds, 'sel')
        Pp = Pp.sub(Pp.mean(axis=1), axis=0).div(Pp.std(axis=1) + 1e-9, axis=0)
        axis = {p: rng.normal() for p in s.patient.unique()}
        rows = []
        if edge and ds == 'sarcoma':
            dm = dm.head(5)
        if edge and ds == 'bladder':
            vivo = pd.read_csv(L.INP / f'{ds}_samples.csv', dtype=str)
            for v in vivo[vivo.model_type == 'patient derived xenograft'].improve_sample_id.unique():
                rows.append(('simulated', v, dm.improve_drug_id.iloc[0], 'simulated', 21, 'days', 'mRECIST', 'PD'))
        for r in dm.itertuples():
            a = rng.normal()
            for smp in s.itertuples():
                if rng.random() < 0.1:
                    continue
                known = isinstance(r.compound, str) and r.compound in Pp.index and smp.patient in Pp.columns
                lat = a + axis[smp.patient] + signal * (Pp.loc[r.compound, smp.patient] if known else 0.0) + rng.normal()
                mets = [('fit_auc', 1 / (1 + np.exp(lat)) + rng.normal(scale=0.01)),
                        ('fit_ic50', -6 - lat + rng.normal(scale=0.3))]
                if edge and ds == 'pancreatic':
                    mets = [('published_auc', 1 / (1 + np.exp(lat)) + rng.normal(scale=0.01))]
                for _ in range(2 if rng.random() < 0.3 else 1):
                    for met, val in mets:
                        rows.append(('simulated', smp.improve_sample_id, r.improve_drug_id, 'simulated', 72, 'hours',
                                     met, val))
                if edge and ds == 'liver' and rng.random() < 0.2:
                    rows.append(('simulated', smp.improve_sample_id, r.improve_drug_id, 'simulated', 120, 'hours',
                                 'fit_auc', rng.random()))
                if edge and ds == 'liver' and rng.random() < 0.02:
                    rows.append(('simulated', smp.improve_sample_id, r.improve_drug_id, 'simulated', 72, 'hours',
                                 'fit_auc', 'NA'))
        pd.DataFrame(rows, columns=['source', 'improve_sample_id', 'improve_drug_id', 'study', 'time', 'time_unit',
                                    'dose_response_metric', 'dose_response_value']).to_csv(
            src / f'{ds}_experiments.tsv.gz', sep='\t', index=False)
    return src


def edge_cases():
    """Registered edge cases, once, with a planted signal of 0.15."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        L.SRC, L.OUT = simulate(root, 0.15, np.random.default_rng(999), edge=True), root / 'processed'
        L.main()
        info = json.loads((root / 'processed' / 'load_summary.json').read_text())
        res = C.main(base=root / 'processed', out=root / 'out')
    edge = {'pancreatic_primary': info['pancreatic']['primary'], 'liver_time': info['liver']['time'],
            'bladder_metrics_after_exclusion': info['bladder']['metrics_present'],
            'datasets': res['datasets'], 'H10': res['H10'], 'H11': res['H11']}
    assert edge['pancreatic_primary'] == '-published_auc' and edge['liver_time'] == '72 hours'
    assert 'mRECIST' not in edge['bladder_metrics_after_exclusion']
    assert any(d['dataset'] == 'sarcoma' and 'excluded' in d for d in res['datasets'])
    write_json(OUT / 'edge_cases.json', edge)
    print('edge cases', edge, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C.N_PERM, C.N_BOOT = 1000, 50
    edge_cases()
    runs = [(0.0, s) for s in range(1, 41)] + [(0.10, s) for s in range(101, 121)] + [(0.15, s) for s in range(201, 221)]
    rows = []
    for signal, seed in runs:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            L.SRC, L.OUT = simulate(root, signal, np.random.default_rng(seed)), root / 'processed'
            L.main()
            res = C.main(base=root / 'processed', out=root / 'out')
        rows.append({'signal': signal, 'seed': seed, 'h10_p': res['H10'].get('permutation_p_one_sided'),
                     'h10_estimate': res['H10'].get('pooled_selective_transfer'), 'h10_units': res['H10']['units'],
                     'h11_p': res['H11'].get('permutation_p_one_sided'),
                     'h10_wilcoxon_p': res['H10'].get('secondary_wilcoxon_p_one_sided')})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'runs.csv', index=False)
    summ = {str(s): {'runs': int(len(g)), 'h10_rejection_rate': float((g.h10_p < 0.05).mean()),
                     'h11_rejection_rate': float((g.h11_p < 0.05).mean()), 'mean_h10_estimate': float(g.h10_estimate.mean())}
            for s, g in df.groupby('signal')}
    write_json(OUT / 'summary.json', summ)
    print(summ)


if __name__ == '__main__':
    main()
