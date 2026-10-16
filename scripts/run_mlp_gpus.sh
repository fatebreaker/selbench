#!/usr/bin/env bash
# Train the global neural drug-response model for every screen x split on free GPUs (one process each).
set -euo pipefail
cd "$(dirname "$0")"
GPUS=(${GPU_LIST:-2 3 4 6 7})
i=0
for sc in PRISM CTRP GDSC1 GDSC2; do
  for sp in random_pairs cell_blind drug_blind; do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g ../../.venv-gpu/bin/python 09b_mlp_global.py train --screen "$sc" --splits "$sp" \
      --device cuda:0 > "../logs/09b_train_${sc}_${sp}.log" 2>&1 &
    i=$((i + 1))
  done
done
wait
echo "all trainings finished"
