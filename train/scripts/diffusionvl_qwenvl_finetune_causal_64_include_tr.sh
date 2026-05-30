#!/bin/bash
# Finetune script for DiffusionVL-QwenVL model
# Model type: diffusionvl_qwenvl (Qwen2.5-VL built-in vision tower + BD3-LM)

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

# ============================================
# TODO: Configure these paths before running
# ============================================

# Wandb configuration (optional, set REPORT_TO=none to disable)
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-pa-bdm}"
REPORT_TO="${REPORT_TO:-wandb}"

# Model checkpoint path. Convert Qwen2.5-VL to DiffusionVL-QwenVL format first.
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"

# Training data paths. DATA_PATH can be a JSON/JSONL file or a YAML mixture file.
DATA_PATH="${DATA_PATH:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${IMG_PATH:-}}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

# ============================================
# Training configuration
# ============================================
# Effective global batch size is:
# num_node * gpu_num * per_device_train_batch_size * gradient_accumulation_steps.
num_node=${1:-1}
gpu_num=${2:-4}
custom_run_name=${3:-"20260508-CSL-64"}
BD3LM_BLOCK_SIZE=${4:-64}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "=========================================="
echo "DiffusionVL-QwenVL Finetune (Qwen2.5-VL + BD3-LM)"
echo "=========================================="
echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"
echo "BD3LM Block Size: ${BD3LM_BLOCK_SIZE}"

if [[ -z "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "ERROR: PRETRAINED_CHECKPOINT is required." >&2
    exit 1
fi

if [[ -z "${DATA_PATH}" ]]; then
    echo "ERROR: DATA_PATH is required." >&2
    exit 1
fi

LLM_VERSION=${PRETRAINED_CHECKPOINT}
VISION_MODEL_VERSION=${PRETRAINED_CHECKPOINT}

echo "Checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "Data path: ${DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER:-<from data file or absolute paths>}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Report to: ${REPORT_TO}"

PROMPT_VERSION=qwen_2_5
BASE_RUN_NAME=${custom_run_name}

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type qwen_merger \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --image_aspect_ratio pad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "${OUTPUT_DIR}/$BASE_RUN_NAME" \
    --num_train_epochs 15 \
    --max_steps -1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 3 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --force_model_type "diffusionvl_qwenvl" \
    --bd3lm_block_aligned_eos True \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to ${REPORT_TO} \
    --dataloader_drop_last True \
    --attn_implementation eager \
    --use_conversation_mask False \
    --enable_bd3lm True \
    --bd3lm_block_size ${BD3LM_BLOCK_SIZE} \
    --save_total_limit 3
