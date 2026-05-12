#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATASET_ROOT [NUM_EVAL_SAMPLES] [EPOCHS]" >&2
  exit 2
fi

DATASET_ROOT="$1"
NUM_EVAL_SAMPLES="${2:-256}"
EPOCHS="${3:-30}"
CONFIG="configs/tcn/spectral_tcn_lossless_symbol_grid.yaml"
RUN_NAME="spectral_tcn_lossless_symbol_grid_easy_predictive"
CHECKPOINT="artifacts/checkpoints/${RUN_NAME}_best.pt"
OUT_DIR="artifacts/analysis/lossless_tcn_$(date +%Y%m%d_%H%M%S)"
CODECS="raw_zlib,raw_lzma,raw_zstd,symbols_zlib,symbols_zstd,bitplane_symbols_zstd,spectral_delta_zlib,spectral_delta_zstd,bitplane_spectral_delta_zstd,tcn_residual_zlib,tcn_residual_zstd"

mkdir -p "$OUT_DIR"

python scripts/audit_lossless_tcn_protocol.py \
  "$DATASET_ROOT" \
  --config "$CONFIG" \
  --split test \
  --difficulty easy \
  --num-samples 4 \
  --device auto \
  --require-residual-backend \
  --save-json "$OUT_DIR/audit_symbol_grid_pretrain.json"

python scripts/train.py \
  --config "$CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --run-name "$RUN_NAME" \
  --override-epochs "$EPOCHS"

python scripts/evaluate_lossless_codecs.py \
  "$DATASET_ROOT" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --source data_npy \
  --split test \
  --difficulty easy \
  --num-samples "$NUM_EVAL_SAMPLES" \
  --device auto \
  --codecs "$CODECS" \
  --save-json "$OUT_DIR/eval_lossless_codecs.json" \
  --save-csv "$OUT_DIR/eval_lossless_codecs_summary.csv"

python scripts/audit_lossless_tcn_protocol.py \
  "$DATASET_ROOT" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --difficulty easy \
  --num-samples 32 \
  --device auto \
  --require-residual-backend \
  --save-json "$OUT_DIR/audit_lossless_tcn_protocol.json"

echo "Saved lossless pilot outputs to: $OUT_DIR"
