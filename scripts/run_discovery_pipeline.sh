#!/usr/bin/env bash
# Discovery pipeline (screens already opened before this study): harmonise -> decompose -> features ->
# cell-blind models -> frozen consensus predictions (+ GDSC2-held-out rehearsal).
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
LOG=../logs
mkdir -p "$LOG"
step() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG/pipeline.log"; }

if [[ "${SKIP_HARMONIZE:-0}" != 1 ]]; then
  step harmonize; OPENBLAS_NUM_THREADS=8 $PY 01_harmonize_discovery.py > "$LOG/01_harmonize.log" 2>&1
  step decompose; OPENBLAS_NUM_THREADS=8 $PY 02_decompose.py > "$LOG/02_decompose.log" 2>&1
fi
if [[ "${SKIP_FEATURES:-0}" != 1 ]]; then
  step features; OPENBLAS_NUM_THREADS=8 $PY 03_cell_features.py > "$LOG/03_cell_features.log" 2>&1
fi
step models; OPENBLAS_NUM_THREADS=1 $PY 04_cell_blind_models.py --workers "${WORKERS:-36}" > "$LOG/04_models.log" 2>&1
step frozen; OPENBLAS_NUM_THREADS=1 $PY 05_consensus_predictions.py --out frozen_predictions --workers "${WORKERS:-30}" > "$LOG/05_frozen.log" 2>&1
step rehearsal; OPENBLAS_NUM_THREADS=1 $PY 05_consensus_predictions.py --screens PRISM CTRP GDSC1 \
  --out rehearsal_predictions_noGDSC2 --workers "${WORKERS:-30}" > "$LOG/05_rehearsal.log" 2>&1
step done
