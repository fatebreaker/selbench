"""Amendment 3 loader for CoderData patient-derived-model responses (written BEFORE retrieval; binds CoderData's
response fields to the registered endpoint).

Input: data/raw/pdm_responses/<dataset>_experiments.tsv.gz (CoderData v2.4 long format: source, improve_sample_id,
improve_drug_id, study, time, time_unit, dose_response_metric, dose_response_value), retrieved by
73_fetch_pdm_responses.py after the freeze; sample tables and drug maps from the frozen inputs.

Registered rules:
* Samples. All samples except in vivo xenografts (model_type 'patient derived xenograft').
* Endpoint. The first metric present in the dataset among fit_auc (area under the fitted curve over one fixed
  concentration range, 1e-10 to 1e-4 M), published_auc and auc.
* Orientation. For fit_auc or auc, if fit_ic50 is present: if the median over drugs (>= 10 jointly measured samples)
  of Spearman(endpoint, fit_ic50) is positive, the endpoint is an area under the viability curve and sensitivity =
  -endpoint; if negative, sensitivity = +endpoint. For published_auc: the same rule against fit_auc where both are
  present for >= 10 samples of a drug; otherwise sensitivity = -published_auc (CoderData's published_auc is an area
  under a viability curve or a viability score). Without fit_ic50, sensitivity = -endpoint.
* Values. dose_response_value is read as a number; non-numeric values are dropped.
* Time and replicates. If several time points exist (missing time counts as one value), the most frequent one in the
  dataset is used; replicate rows are averaged.
* Drugs. Through the frozen results/pdm_transfer/<dataset>/drug_map.csv; ambiguous drugs are excluded; drugs outside
  discovery are kept as 'PDM:<id>' for the dataset's fitness axis.
* Units. The primary unit is the patient (patient_key): CoderData common_name without a trailing '_<digits>'
  (bladder SCBO-11_2 and SCBO-11_3 are further specimens of SCBO-11; sarcoma SARC0139_1) and, for sarcoma
  identifiers SARC<4 digits>-<digits>, without the trailing '-<digits>' (SARC0139-2 is a further specimen of
  SARC0139). Sensitivities are averaged over a patient's samples. Sample-level tables are written for the secondary
  analysis.
Outputs: data/processed/pdm/<dataset>/{patient_long.csv.gz, sample_long.csv.gz}, data/processed/pdm/load_summary.json.
"""
import re

import numpy as np
import pandas as pd

from common import OWN_RAW, PROCESSED, RESULTS, write_json

DATASETS = ['bladder', 'pancreatic', 'liver', 'sarcoma', 'mpnst']
SRC = OWN_RAW / 'pdm_responses'
INP = OWN_RAW / 'pdm_inputs'
OUT = PROCESSED / 'pdm'
PRIORITY = ['fit_auc', 'published_auc', 'auc']


def patient_key(name):
    name = re.sub(r'_\d+$', '', str(name))
    return re.sub(r'^(SARC\d{4})-\d+$', r'\1', name)


def median_spearman(w, a, b):
    rho = []
    for d, g in w.groupby('improve_drug_id'):
        g = g.dropna(subset=[a, b])
        if len(g) >= 10:
            rho.append(g[a].corr(g[b], method='spearman'))
    rho = [r for r in rho if np.isfinite(r)]
    return float(np.median(rho)) if rho else None


def load(ds):
    e = pd.read_csv(SRC / f'{ds}_experiments.tsv.gz', sep='\t', dtype={'improve_sample_id': str, 'improve_drug_id': str})
    s = pd.read_csv(INP / f'{ds}_samples.csv', dtype=str).drop_duplicates('improve_sample_id')
    vivo = set(s.loc[s.model_type == 'patient derived xenograft', 'improve_sample_id'])
    e = e[~e.improve_sample_id.isin(vivo)]
    e = e.assign(dose_response_value=pd.to_numeric(e.dose_response_value, errors='coerce')).dropna(
        subset=['dose_response_value'])
    info = {'rows': int(len(e)), 'metrics_present': sorted(e.dose_response_metric.unique().tolist())}
    if e.empty:
        info['excluded'] = 'no in vitro response rows'
        return info
    tkey = (e['time'].astype(str) if 'time' in e else pd.Series('nan', index=e.index)) + ' ' + \
        (e['time_unit'].astype(str) if 'time_unit' in e else pd.Series('nan', index=e.index))
    vc = tkey.value_counts()
    best = sorted(vc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    info['time_points'], info['time'] = vc.to_dict(), best
    e = e[tkey == best]
    w = e.pivot_table(index=['improve_sample_id', 'improve_drug_id'], columns='dose_response_metric',
                      values='dose_response_value', aggfunc='mean').reset_index()
    metric = next((m for m in PRIORITY if m in w.columns and w[m].notna().any()), None)
    if metric is None:
        info['excluded'] = 'no registered endpoint'
        return info
    sign = -1.0
    if metric in ('fit_auc', 'auc') and 'fit_ic50' in w.columns:
        rho = median_spearman(w, metric, 'fit_ic50')
        info['orientation_median_spearman_vs_fit_ic50'] = rho
        if rho is not None and rho < 0:
            sign = 1.0
    elif metric == 'published_auc' and 'fit_auc' in w.columns:
        rho = median_spearman(w, metric, 'fit_auc')
        info['orientation_median_spearman_vs_fit_auc'] = rho
        if rho is not None and rho < 0:
            sign = 1.0
    w['sens'] = sign * pd.to_numeric(w[metric], errors='coerce')
    info['primary'] = f'{"-" if sign < 0 else "+"}{metric}'
    dm = pd.read_csv(RESULTS / 'pdm_transfer' / ds / 'drug_map.csv', dtype=str)
    dm = dm[dm.method != 'ambiguous_excluded'].set_index('improve_drug_id').compound
    w['compound'] = w.improve_drug_id.map(dm)
    w = w.dropna(subset=['compound', 'sens'])
    sample = w.groupby(['compound', 'improve_sample_id'], as_index=False).sens.mean()
    pat = s.set_index('improve_sample_id').common_name.map(patient_key)
    sample['patient'] = sample.improve_sample_id.map(pat)
    patient = sample.dropna(subset=['patient']).groupby(['compound', 'patient'], as_index=False).sens.mean()
    out = OUT / ds
    out.mkdir(parents=True, exist_ok=True)
    patient.rename(columns={'patient': 'unit'}).to_csv(out / 'patient_long.csv.gz', index=False)
    sample.rename(columns={'improve_sample_id': 'unit'})[['compound', 'unit', 'sens']].to_csv(out / 'sample_long.csv.gz',
                                                                                             index=False)
    info.update({'samples': int(sample.improve_sample_id.nunique()), 'patients': int(patient.patient.nunique()),
                 'compounds': int(patient.compound.nunique()), 'patient_pairs': int(len(patient))})
    return info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds in DATASETS:
        if (SRC / f'{ds}_experiments.tsv.gz').exists():
            summary[ds] = load(ds)
            print(ds, summary[ds], flush=True)
    write_json(OUT / 'load_summary.json', summary)


if __name__ == '__main__':
    main()
