#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/train/scripts/diffusionvl_qwenvl_finetune_causal_64_include_tr.sh"

usage() {
  cat <<'USAGE'
Usage:
  PRETRAINED_CHECKPOINT=/path/to/DiffusionVL-Qwen2.5VL-3B-causal \
  DATA_PATH=/path/to/train.yaml \
  IMAGE_FOLDER=/path/to/images \
  OUTPUT_DIR=/path/to/outputs \
  GPU_NUM=4 \
  RUN_NAME=pa-bdm-csl-64 \
  BD3LM_BLOCK_SIZE=64 \
  bash run_train.sh

Required:
  PRETRAINED_CHECKPOINT  Converted DiffusionVL-QwenVL checkpoint directory.
  DATA_PATH              Training JSON/JSONL/YAML path.

Optional:
  IMAGE_FOLDER           Root directory for relative image paths.
  OUTPUT_DIR             Output directory. Default: ./outputs.
  NUM_NODES              Number of nodes. Default: 1.
  GPU_NUM                GPUs per node. Default: 4.
  RUN_NAME               Run name and output subdirectory.
  BD3LM_BLOCK_SIZE       Candidate block size. Default: 64.
  MASTER_ADDR            Torchrun master address. Default: 127.0.0.1.
  MASTER_PORT            Torchrun master port. Default: 29199.
  RANK                   Node rank. Default: 0.
  REPORT_TO              wandb or none. Default: wandb.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${PRETRAINED_CHECKPOINT:?PRETRAINED_CHECKPOINT is required. Run 'bash run_train.sh --help' for an example.}"
: "${DATA_PATH:?DATA_PATH is required. Run 'bash run_train.sh --help' for an example.}"

export IMAGE_FOLDER="${IMAGE_FOLDER:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
export NUM_NODES="${NUM_NODES:-1}"
export GPU_NUM="${GPU_NUM:-4}"
export RUN_NAME="${RUN_NAME:-pa-bdm-csl-64}"
export BD3LM_BLOCK_SIZE="${BD3LM_BLOCK_SIZE:-64}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29199}"
export RANK="${RANK:-0}"
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-pa-bdm}"

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}"

echo "Launching PA-BDM training"
echo "  repo: ${REPO_ROOT}"
echo "  checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "  data: ${DATA_PATH}"
echo "  image folder: ${IMAGE_FOLDER:-<from data file or absolute paths>}"
echo "  output: ${OUTPUT_DIR}"
echo "  nodes/gpus: ${NUM_NODES}/${GPU_NUM}"
echo "  run name: ${RUN_NAME}"
echo "  block size: ${BD3LM_BLOCK_SIZE}"

cd "${REPO_ROOT}/train"
bash "${TRAIN_SCRIPT}" "${NUM_NODES}" "${GPU_NUM}" "${RUN_NAME}" "${BD3LM_BLOCK_SIZE}"
