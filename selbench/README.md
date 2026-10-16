# selbench

Selectivity-aware evaluation of drug-response predictions: the reference implementation of the protocol in Box 1 of
"Drug-response models capture general drug sensitivity far better than drug-specific selectivity".

Per-drug correlations, the usual benchmark score, mix two things: how generally sensitive a cell line is (its mean
response to other drugs, the general-sensitivity axis) and how it responds to the drug in question. `selbench` splits
each per-drug correlation exactly into an axis term and a selective term, reports both against a perfect predictor,
and scores identical predictions against two datasets on matched cells so that cross-dataset changes can be
attributed.

Dependencies: numpy, pandas, scipy. Python >= 3.8.

## Quick start

```python
import pandas as pd
import selbench

obs = pd.read_csv('observed.csv')      # columns: drug, cell, response   (higher = more sensitive)
pred = pd.read_csv('predictions.csv')  # columns: drug, cell, prediction (higher = more sensitive)

table, summary = selbench.evaluate(pred, obs)   # per-drug raw and selective correlations, medians
```

## Protocol steps and functions

| Box 1 step | What to report | Functions |
|---|---|---|
| 1. Axis | leave-drug-out axis per evaluation dataset; its split-half reliability; active-drug and leave-class-out variants | `fitness_axis`, `selective_residuals`, `axis_reliability`, `axis_agreement` (the code keeps the study's early name "fitness axis") |
| 2. Scores | raw $r_{pz}$, selective $r_{pz\cdot g}$ and the exact split $r_{pz}=r_{pg}r_{zg}+\sqrt{(1-r_{pg}^2)(1-r_{zg}^2)}\,r_{pz\cdot g}$; the perfect-predictor axis share | `split_terms`, `partial_corr`, `selective_scores`, `oracle_fitness_share`, `summarize` |
| 3. Across datasets | identical predictions scored against two datasets on matched cells; selectivity given both axes | `paired_scores`, `paired_scores_matrix`, `paired_summary` |
| 4. Floors and baselines | drug-agnostic baselines on the same metric as the model (selective correlation against `leakage_floor`'s median-selective `floor`; pooled capture against `box1.matched_baselines`); tissue-by-drug and omics-axis baselines | `leakage_floor`, `matched_baselines` (box1), `drug_agnostic_predictions`, `tissue_drug_mean_baseline`, `tissue_mean_baseline`, `fitness_baseline` |
| 5. Diagnostic | selectivity given the model's own leave-drug-out axis | `model_axis`, `model_axis_scores`, `model_axis_diagnostic` |
| 6. Uncertainty | two-way (drug x cell line) bootstrap; reliability shrinkage | `two_way_bootstrap`, `paired_comparison`, `shrink_reliability`, `normalize_by_reliability` |

Further protocol components are in `box1.py` (see its docstrings):

| Component | Functions |
|---|---|
| Capture and references | `capture`, `corrected_reference`, `corrected_capture`, `lineage_capture` |
| Axis reliability and leakage | `axis_leakage_bound` ($A\,m(1-\rho)/[(1-m)\rho]$, a model-based estimate of the axis's leakage into selective capture under a classical-error model (the axis's error independent of predictions and profiles, $\rho$ its reliability for the latent axis): under those conditions exact for drug-agnostic predictors and a bound otherwise, but drug agnosticism alone does not ensure them; not a bound for small drug panels; invalid for pair splits; prefer metric-matched empirical baselines, `matched_baselines`), `corrected_selective_capture` |
| General-sensitivity names | `axis`, `general_sensitivity_axis` (= `fitness_axis`), `axis_baseline` (= `fitness_baseline`), `oracle_axis_share` (= `oracle_fitness_share`) |
| Across datasets | `selective_both` (symmetric), `pooled_change`, `attenuation` (proportional-attenuation test; pass the two axes' split-half reliabilities to adjust it), `retained_fraction_test`, `retained_fraction_bootstrap`, `two_way_draws` |
| Harmonization from fitted curves | `curve_frame`, `mean_viability`, `tested_ranges`, `common_window`, `summarize_curves`, `shared_panel`, `harmonize_pair` |
| Baselines and lineage | `biomarker_scores`, `tissue_indicators`, `lineage_terms`, `lineage_adjusted_scores` |
| Axis | `active_drugs`, `active_drugs_rule`, `activity_fraction` (endpoint-specific activity rules), `active_axis`, `axis_split_half`, `censored_rank_normal` and `censor_resistant` (censored or undefined values tied at the resistant end before rank-normalization) |
| Expected retention | `triad_reliability` (reproducible share of each dataset's selective variance from screen triads), `psi_overlap`, `expected_retention` ($\sqrt{R_T/R_S}$, the retention of a predictor of shared selectivity) |
| Multiplicity and intervals | `holm`, `interval`, `rank_intervals` (paired bootstrap rank intervals and resolved model pairs) |

## Conventions

Long tables have columns `drug`, `cell` and `response` (observed) or `prediction`. Matrices are DataFrames with drugs
as rows and cell lines as columns. Higher values mean more sensitive in both observations and predictions.

## Tests

```
python -m pytest test_selbench.py test_box1.py
```

`scripts/95_validate_box1.py`, `99d_validate_box1_r3.py` and `99g_validate_box1_rank_retention.py` check the library
against the study's own results (264 checks, plus 484 for the round-4 additions in `102b_validate_box1_r4.py`; `results/revision_round2/box1_validation/`).

`release/` holds the study's derived discovery profiles, axes, reproducibility classes and cell folds, so that
results can be checked without re-running the harmonization.
