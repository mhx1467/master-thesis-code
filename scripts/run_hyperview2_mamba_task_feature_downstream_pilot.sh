#!/usr/bin/env bash
set -euo pipefail
set -x

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <hyperview2_root> <downstream_checkpoint> <pretrained_compressor> [epochs] [feature_loss_weight] [prediction_loss_weight]" >&2
  exit 1
fi

HV2_ROOT="$1"
DOWNSTREAM_CKPT="$2"
PRETRAINED_CKPT="$3"
EPOCHS="${4:-30}"
FEATURE_LOSS_WEIGHT="${5:-}"
PREDICTION_LOSS_WEIGHT="${6:-}"

CONFIG="${CONFIG:-configs/mamba/hyperview2_prisma_hierarchical_spectral_mamba_ae_latent48_task_feature_rd.yaml}"
LABELS_CSV="${LABELS_CSV:-$HV2_ROOT/HYPERVIEW2/train_gt.csv}"
EXP_NAME="${EXP_NAME:-hyperview2_prisma_hierarchical_spectral_mamba_ae_latent48_task_feature_rd}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
CKPT="artifacts/checkpoints/${EXP_NAME}_best.pt"
OUT_DIR="artifacts/downstream/${RUN_NAME}_compression_eval"

TRAIN_ARGS=(
  .venv311/bin/python scripts/train_hyperview2_compressor.py "$HV2_ROOT"
  --config "$CONFIG" \
  --downstream-checkpoint "$DOWNSTREAM_CKPT" \
  --labels-csv "$LABELS_CSV" \
  --pretrained "$PRETRAINED_CKPT" \
  --run-name "$RUN_NAME" \
  --override-epochs "$EPOCHS"
)
if [[ -n "$FEATURE_LOSS_WEIGHT" ]]; then
  TRAIN_ARGS+=(--override-feature-loss-weight "$FEATURE_LOSS_WEIGHT")
fi
if [[ -n "$PREDICTION_LOSS_WEIGHT" ]]; then
  TRAIN_ARGS+=(--override-prediction-loss-weight "$PREDICTION_LOSS_WEIGHT")
fi

"${TRAIN_ARGS[@]}"

.venv311/bin/python scripts/evaluate_hyperview2_downstream_compression.py "$HV2_ROOT" \
  --downstream-checkpoint "$DOWNSTREAM_CKPT" \
  --compressor-checkpoint "$CKPT" \
  --labels-csv "$LABELS_CSV" \
  --split val \
  --batch-size 128 \
  --device cuda \
  --reconstruction-mode actual \
  --output-dir "$OUT_DIR"

echo "Saved HYPERVIEW2 Mamba task-feature downstream-compression outputs to: $OUT_DIR"
