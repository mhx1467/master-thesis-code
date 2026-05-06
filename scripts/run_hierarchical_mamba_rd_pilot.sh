#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT_ARG="${1:-${DATASET_ROOT:-}}"
if [[ -z "${DATASET_ROOT_ARG}" ]]; then
    echo "Usage: $0 <DATASET_ROOT>"
    echo "or set DATASET_ROOT before running."
    exit 2
fi

K4_MSE_CKPT="${K4_MSE_CKPT:-artifacts/checkpoints/hierarchical_spectral_mamba_ae_recon_latent96_best.pt}"
K1_MSE_CKPT="${K1_MSE_CKPT:-artifacts/checkpoints/hierarchical_spectral_mamba_ae_k1_recon_latent96_best.pt}"
RUN_K1="${RUN_K1:-0}"

K4_RD_CONFIG="configs/mamba/hierarchical_spectral_mamba_ae_k4_spatial_rd_lambda_0_01.yaml"
K1_RD_CONFIG="configs/mamba/hierarchical_spectral_mamba_ae_k1_spatial_rd_lambda_0_01.yaml"

if [[ ! -f "${K4_MSE_CKPT}" ]]; then
    echo "Missing K4 MSE checkpoint: ${K4_MSE_CKPT}"
    echo "Download it first, for example:"
    echo "  python scripts/download_wandb_checkpoint.py 3frjy1j5 --filename $(basename "${K4_MSE_CKPT}")"
    exit 2
fi

python scripts/evaluate.py \
    "${K4_MSE_CKPT}" \
    "${DATASET_ROOT_ARG}" \
    --split test \
    --difficulty easy \
    --batch-size 4 \
    --num-workers 4 \
    --save-json \
    --run-name eval_hierarchical_spectral_mamba_ae_recon_latent96_test

python scripts/train.py \
    --config "${K4_RD_CONFIG}" \
    --dataset-root "${DATASET_ROOT_ARG}" \
    --pretrained "${K4_MSE_CKPT}" \
    --run-name hierarchical_spectral_mamba_ae_k4_spatial_rd_lambda_0_01_ft_from_mse

if [[ "${RUN_K1}" == "1" ]]; then
    if [[ ! -f "${K1_MSE_CKPT}" ]]; then
        echo "Missing K1 MSE checkpoint: ${K1_MSE_CKPT}"
        echo "Download it first, for example:"
        echo "  python scripts/download_wandb_checkpoint.py c6ek5olx --filename $(basename "${K1_MSE_CKPT}")"
        exit 2
    fi

    python scripts/train.py \
        --config "${K1_RD_CONFIG}" \
        --dataset-root "${DATASET_ROOT_ARG}" \
        --pretrained "${K1_MSE_CKPT}" \
        --run-name hierarchical_spectral_mamba_ae_k1_spatial_rd_lambda_0_01_ft_from_mse
fi
