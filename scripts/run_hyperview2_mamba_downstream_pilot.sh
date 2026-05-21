#!/usr/bin/env bash
set -euo pipefail
set -x

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <hyperview2_root> <downstream_checkpoint> [epochs]" >&2
  exit 1
fi

HV2_ROOT="$1"
DOWNSTREAM_CKPT="$2"
EPOCHS="${3:-120}"

CONFIG="${CONFIG:-configs/mamba/hyperview2_prisma_hierarchical_spectral_mamba_ae_latent48.yaml}"
LABELS_CSV="${LABELS_CSV:-$HV2_ROOT/HYPERVIEW2/train_gt.csv}"
EXP_NAME="${EXP_NAME:-hyperview2_prisma_hierarchical_spectral_mamba_ae_latent48}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
CKPT="artifacts/checkpoints/${EXP_NAME}_best.pt"
OUT_DIR="artifacts/downstream/${RUN_NAME}_compression_eval"

.venv311/bin/python scripts/train_hyperview2_compressor.py "$HV2_ROOT" \
  --config "$CONFIG" \
  --downstream-checkpoint "$DOWNSTREAM_CKPT" \
  --labels-csv "$LABELS_CSV" \
  --run-name "$RUN_NAME" \
  --override-epochs "$EPOCHS"

.venv311/bin/python scripts/evaluate_hyperview2_downstream_compression.py "$HV2_ROOT" \
  --downstream-checkpoint "$DOWNSTREAM_CKPT" \
  --compressor-checkpoint "$CKPT" \
  --labels-csv "$LABELS_CSV" \
  --split val \
  --batch-size 128 \
  --device cuda \
  --reconstruction-mode actual \
  --output-dir "$OUT_DIR"

echo "Saved HYPERVIEW2 Mamba downstream-compression outputs to: $OUT_DIR"
