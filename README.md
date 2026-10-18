# selbench

Code for the manuscript *Drug-response models capture general drug sensitivity far better than drug-specific
selectivity* (in preparation), and `selbench`, the evaluation library that implements the paper's protocol (Box 1).

Benchmarks rank drug-response models by how well they order cell lines for each drug. A cell line's general
sensitivity to all drugs can earn much of that score without any drug-specific information. `selbench` splits each
per-drug correlation exactly into a general-sensitivity (axis) term and a drug-specific (selective) term, reports both
as ratios to a perfect predictor's terms, compares them with drug-agnostic baselines on the same metric, and scores
identical predictions against two datasets on matched cells so that cross-dataset changes can be attributed.

## Layout

| Path | Contents |
|---|---|
| `selbench/` | The library: `selbench.py` (core scoring) and `box1.py` (the remaining Box 1 components), their tests, a README mapping each Box 1 step to its functions, and `pyproject.toml` |
| `scripts/` | The analysis pipeline, numbered in the order it was written (see below) |
| `env/` | Pinned environments: pip lock files, conda explicit lists, package inventories, a Dockerfile and `environment-analysis.yml` |

## The library

```bash
pip install ./selbench          # numpy, pandas, scipy; Python >= 3.8
cd selbench && pytest -q        # 38 tests
```

```python
import pandas as pd
import selbench

obs = pd.read_csv('observed.csv')      # columns: drug, cell, response   (higher = more sensitive)
pred = pd.read_csv('predictions.csv')  # columns: drug, cell, prediction (higher = more sensitive)
table, summary = selbench.evaluate(pred, obs)   # per-drug raw and selective correlations, medians
```

`selbench/README.md` lists the functions for each protocol step (axis, scores and captures, cross-dataset scoring,
drug-agnostic floors on the same metric, diagnostics, uncertainty).

## The pipeline (`scripts/`)

Scripts run from `scripts/` and expect the project layout around them: `data/` (inputs), `results/` (outputs),
`protocol/` (pre-registration and freeze manifests), `figures/` and `manuscript/` next to `scripts/`, and
`selbench/` as here. Most use Python 3.8 (`env/requirements-venv.lock.txt`); the DrEval re-analysis and retraining
use Python 3.11 with drevalpy 1.6.0 (`env/requirements-venv-dreval.lock.txt`); the neural global model uses the GPU
environment.

| Scripts | Purpose |
|---|---|
| 01–05, `run_discovery_pipeline.sh` | Harmonize the four discovery screens (PRISM, CTRP v2, GDSC1, GDSC2), decompose profiles into axis and selective parts, build cell features, train cell-blind models, write the frozen consensus predictions |
| 06–08d | Ceiling, confirmation loader, the frozen confirmatory analysis and the freeze scripts (one-time; they refuse to overwrite) |
| 09–12, 17–21, `run_mlp_gpus.sh` | Global and neural models, value of information, axis and drug-class analyses, endpoint, rescreen, simulation, robustness and biomarker checks |
| 13–16 | Figures, number collection, supplementary tables and document build |
| 22–26 | BeatAML transfer, PharmacoDB screens (CCLE, FIMM, GRAY), library release |
| 27–28, 34–37 | DrEval: download, leaderboard re-scoring, full results, cross-study scoring, normalized metric |
| 29–33 | PDX-derived cells (Amendment 2) |
| 38–44 | Cross-dataset uncertainty, model-axis adjustment, paired cross-dataset data, scores, summaries and rankings |
| 45–52 | Axis analyses (activity, lineage, harmonization, within-screen) and the ceiling (estimands, factor model, simulation, canonical biomarkers) |
| 53–56 | Reproducibility classes, patient-derived inference, transfer power, confirmation estimates |
| 57–63 | Retraining in drevalpy, regeneration of frozen predictions, library validation, reproducibility inventory |
| 64–67 | Revised figures, revision tables, harmonized-range versions, number collection |
| 68 | IMPROVE cross-study analysis |
| 69–75 | Patient-derived organoids (Amendment 3) |
| 76–99i | Capture, reliability, paired engine, attenuation and equivalence tests, harmonization, Holm, lineage, dataset-by-endpoint analysis, validation of `selbench` against the analyses, numbers ledger, source data |
| 100a–107 | Transfer prediction, known-answer simulation, Fig. 1 example, CRISPR gene dependency, DrEval normalized-metric decomposition, attenuation power |
| 108–109 | Cross-screen agreement carried by the general-sensitivity axis, by mechanism; what the cell-blind model adds to a canonical biomarker |
