#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET_ROOT [NUM_EVAL_SAMPLES] [EPOCHS] [PRETRAINED_CHECKPOINT] [LR]" >&2
  exit 2
fi

DATASET_ROOT="$1"
NUM_EVAL_SAMPLES="${2:-1149}"
EPOCHS="${3:-10}"
PRETRAINED="${4:-artifacts/checkpoints/spectral_tcn_delta_lossless_symbol_grid_easy_predictive_rescue_v2_best.pt}"
LR="${5:-0.00003}"

CONFIG="configs/tcn/spectral_tcn_delta_lossless_symbol_grid_entropy_ft.yaml"
RUN_NAME="spectral_tcn_delta_lossless_symbol_grid_entropy_ft"
CHECKPOINT="artifacts/checkpoints/${RUN_NAME}_best.pt"
OUT_DIR="artifacts/analysis/lossless_tcn_entropy_ft_$(date +%Y%m%d_%H%M%S)"
CODECS="spectral_delta_zstd,bitplane_spectral_delta_zstd,tcn_residual_zstd,bitplane_tcn_residual_zstd"

if [[ ! -f "$PRETRAINED" ]]; then
  echo "Missing pretrained checkpoint: $PRETRAINED" >&2
  echo "Run scripts/run_lossless_tcn_delta_rescue_v2.sh first or pass PRETRAINED_CHECKPOINT explicitly." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python scripts/train.py \
  --config "$CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --pretrained "$PRETRAINED" \
  --run-name "$RUN_NAME" \
  --override-epochs "$EPOCHS" \
  --override-lr "$LR"

python scripts/evaluate_lossless_codecs.py \
  "$DATASET_ROOT" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source tif \
  --split test \
  --difficulty easy \
  --num-samples "$NUM_EVAL_SAMPLES" \
  --device auto \
  --codecs "$CODECS" \
  --save-json "$OUT_DIR/eval_lossless_tcn_entropy_ft.json" \
  --save-csv "$OUT_DIR/eval_lossless_tcn_entropy_ft.csv"

python scripts/audit_lossless_tcn_protocol.py \
  "$DATASET_ROOT" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source tif \
  --split test \
  --difficulty easy \
  --num-samples 32 \
  --device auto \
  --require-residual-backend \
  --save-json "$OUT_DIR/audit_lossless_tcn_entropy_ft.json"

echo "Saved entropy-aware lossless TCN fine-tuning outputs to: $OUT_DIR"
