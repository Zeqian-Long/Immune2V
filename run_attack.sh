#!/usr/bin/env bash
set -euo pipefail

# ====== config ======
export CUDA_VISIBLE_DEVICES=0
LOGDIR=logs
mkdir -p $LOGDIR

# ====== run ======
echo "[1/3] Preprocess data"
python ./run_attack/preprocess_data.py \
  2>&1 | tee $LOGDIR/preprocess.log

echo "[2/3] Run attack"
python ./run_attack/Immune-attack.py \
  2>&1 | tee $LOGDIR/attack.log

echo "[3/3] Test"
python ./run_attack/Immune-test.py \
  2>&1 | tee $LOGDIR/test.log

echo "All done."