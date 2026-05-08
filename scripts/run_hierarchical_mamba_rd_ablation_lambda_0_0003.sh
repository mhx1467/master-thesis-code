#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT_ARG="${1:-${DATASET_ROOT:-}}"
if [[ -z "${DATASET_ROOT_ARG}" ]]; then
    echo "Usage: $0 <DATASET_ROOT>"
    echo "or set DATASET_ROOT before running."
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RD_LAMBDA="${RD_LAMBDA:-0.0003}"
LR="${LR:-0.00001}"
EPOCHS="${EPOCHS:-30}"
RUN_K1="${RUN_K1:-1}"
RUN_NO_SPATIAL="${RUN_NO_SPATIAL:-1}"
EVAL_AFTER="${EVAL_AFTER:-1}"

K1_MSE_CKPT="${K1_MSE_CKPT:-artifacts/checkpoints/hierarchical_spectral_mamba_ae_k1_recon_latent96_best.pt}"
NO_SPATIAL_MSE_CKPT="${NO_SPATIAL_MSE_CKPT:-artifacts/checkpoints/hierarchical_spectral_mamba_ae_no_spatial_recon_latent96_best.pt}"
NO_SPATIAL_FALLBACK_CKPT="artifacts/checkpoints/hierarchical_spectral_mamba_ae_k4_no_spatial_recon_latent96_best.pt"

if [[ ! -f "${NO_SPATIAL_MSE_CKPT}" && -f "${NO_SPATIAL_FALLBACK_CKPT}" ]]; then
    NO_SPATIAL_MSE_CKPT="${NO_SPATIAL_FALLBACK_CKPT}"
fi

K1_RD_CONFIG="configs/mamba/hierarchical_spectral_mamba_ae_k1_spatial_rd_lambda_0_01.yaml"
NO_SPATIAL_RD_CONFIG="configs/mamba/hierarchical_spectral_mamba_ae_k4_no_spatial_rd_lambda_0_01.yaml"

require_checkpoint() {
    local checkpoint_path="$1"
    local run_id="$2"
    local filename="$3"

    if [[ -f "${checkpoint_path}" ]]; then
        return 0
    fi

    echo "Missing MSE checkpoint: ${checkpoint_path}"
    echo "Download it first:"
    echo "  ${PYTHON_BIN} scripts/download_wandb_checkpoint.py ${run_id} --filename ${filename}"
    exit 2
}

train_and_eval() {
    local label="$1"
    local config_path="$2"
    local pretrained_path="$3"
    local experiment_name="$4"
    local run_name="${experiment_name}_ft_from_mse"
    local best_checkpoint="artifacts/checkpoints/${experiment_name}_best.pt"

    echo
    echo "======================================================="
    echo "RD ablation: ${label}"
    echo "Config:      ${config_path}"
    echo "Pretrained:  ${pretrained_path}"
    echo "Lambda:      ${RD_LAMBDA}"
    echo "LR:          ${LR}"
    echo "Epochs:      ${EPOCHS}"
    echo "Checkpoint:  ${best_checkpoint}"
    echo "======================================================="

    "${PYTHON_BIN}" scripts/train.py \
        --config "${config_path}" \
        --dataset-root "${DATASET_ROOT_ARG}" \
        --pretrained "${pretrained_path}" \
        --run-name "${run_name}" \
        --override-experiment-name "${experiment_name}" \
        --override-rd-lambda "${RD_LAMBDA}" \
        --override-epochs "${EPOCHS}" \
        --override-lr "${LR}"

    if [[ "${EVAL_AFTER}" == "1" ]]; then
        "${PYTHON_BIN}" scripts/evaluate.py \
            "${best_checkpoint}" \
            "${DATASET_ROOT_ARG}" \
            --split test \
            --difficulty easy \
            --batch-size 4 \
            --num-workers 4 \
            --save-json \
            --run-name "eval_${experiment_name}_test"
    fi
}

if [[ "${RUN_K1}" == "1" ]]; then
    require_checkpoint \
        "${K1_MSE_CKPT}" \
        "c6ek5olx" \
        "$(basename "${K1_MSE_CKPT}")"

    train_and_eval \
        "K1 + spatial conditioning" \
        "${K1_RD_CONFIG}" \
        "${K1_MSE_CKPT}" \
        "hierarchical_spectral_mamba_ae_k1_spatial_rd_lambda_0_0003"
fi

if [[ "${RUN_NO_SPATIAL}" == "1" ]]; then
    require_checkpoint \
        "${NO_SPATIAL_MSE_CKPT}" \
        "p1vnaolg" \
        "$(basename "${NO_SPATIAL_MSE_CKPT}")"

    train_and_eval \
        "K4 without spatial conditioning" \
        "${NO_SPATIAL_RD_CONFIG}" \
        "${NO_SPATIAL_MSE_CKPT}" \
        "hierarchical_spectral_mamba_ae_k4_no_spatial_rd_lambda_0_0003"
fi
