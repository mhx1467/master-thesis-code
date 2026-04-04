#!/usr/bin/env bash
set -e

CONFIG="configs/baseline_2d/baseline_2d_rd_sweep.yaml"
CKPT_DIR="artifacts/checkpoints"

BASE_LAMBDA=0.001
BASE_RUN_NAME="baseline_2d_rd_lambda_${BASE_LAMBDA}"
BASE_CKPT="${CKPT_DIR}/${BASE_RUN_NAME}_best.pt"
BASE_EPOCHS=2

python scripts/train.py \
    --config ${CONFIG} \
    --run-name ${BASE_RUN_NAME} \
    --override-rd-lambda ${BASE_LAMBDA} \
    --override-epochs ${BASE_EPOCHS} \
    --override-lr 0.0001

FT_EPOCHS=1
FT_LR=0.00001

LAMBDAS=(0.005 0.01 0.05 0.1)

PREV_CKPT=${BASE_CKPT}

for L in "${LAMBDAS[@]}"; do
    RUN_NAME="baseline_2d_rd_lambda_${L}"
    CURRENT_CKPT="${CKPT_DIR}/${RUN_NAME}_best.pt"
    
    python scripts/train.py \
        --config ${CONFIG} \
        --run-name ${RUN_NAME} \
        --pretrained ${PREV_CKPT} \
        --override-rd-lambda ${L} \
        --override-epochs ${FT_EPOCHS} \
        --override-lr ${FT_LR}
        
    PREV_CKPT=${CURRENT_CKPT}
done
