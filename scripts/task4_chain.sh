#!/bin/bash
# usage: task4_chain.sh final|provisional
set -e
MODE=$1
S=/data/home_beta/htran/generation/med_preferences/discovery/selectivity_ceiling/scripts
L=/data/home_beta/htran/generation/med_preferences/discovery/selectivity_ceiling/results/revision_round2/logs
LR=/data/home_beta/htran/generation/med_preferences/discovery/selectivity_ceiling/results/revision_retrain/logs
PY=/data/home_beta/htran/generation/med_preferences/discovery/.venv-dreval/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $S
T=r3_$MODE
echo "start $(date)"
$PY 59_retrain_score.py --settings GDSC2_AUC CTRPv2_LNIC50 GDSC2_LNIC50 CTRPv2_AUC > $LR/59_score_$T.log 2>&1; echo "59 done $(date)"
$PY 60_retrain_summary.py > $LR/60_summary_$T.log 2>&1; echo "60 done $(date)"
$PY 78_capture.py > $L/78_$T.log 2>&1; echo "78 done $(date)"
if [ "$MODE" = provisional ]; then
  $PY 80_paired_engine.py --parts retrain --allow-incomplete-ctrpv2auc > $L/80_retrain_$T.log 2>&1
else
  $PY 80_paired_engine.py --parts retrain > $L/80_retrain_$T.log 2>&1
fi
echo "80 done $(date)"
$PY 81_attenuation_null.py > $L/81_$T.log 2>&1; echo "81 done $(date)"
$PY 82_symmetric_selboth.py > $L/82_$T.log 2>&1; echo "82 done $(date)"
$PY 84_source_reference.py > $L/84_$T.log 2>&1; echo "84 done $(date)"
$PY 89_collect.py > $L/89_$T.log 2>&1; echo "89 done $(date)"
$PY 91_lineage.py --parts retrain --settings CTRPv2_AUC > $L/91_$T.log 2>&1; echo "91 done $(date)"
if [ "$MODE" = provisional ]; then
  $PY 97_two_by_two.py --models EnsembleMF GradientBoosting RandomForest --out two_by_two_provisional_3models > $L/97_$T.log 2>&1
  $PY 98_endpoint_cell_selection.py --dataset CTRPv2 --models EnsembleMF GradientBoosting RandomForest --suffix _CTRPv2_provisional_3models > $L/98_$T.log 2>&1
else
  $PY 97_two_by_two.py > $L/97_$T.log 2>&1
  $PY 97_two_by_two.py --models EnsembleMF GradientBoosting RandomForest --out two_by_two_3models > $L/97_${T}_3models.log 2>&1
  $PY 98_endpoint_cell_selection.py --dataset CTRPv2 > $L/98_$T.log 2>&1
fi
echo "97/98 done $(date)"
$PY 99b_retention_baselines.py > $L/99b_$T.log 2>&1; echo "99b done $(date)"
$PY 99f_collect_ws7.py > $L/99f_$T.log 2>&1; echo "99f done $(date)"
echo "CHAIN COMPLETE $MODE $(date)"
