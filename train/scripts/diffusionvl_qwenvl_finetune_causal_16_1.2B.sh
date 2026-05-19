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

# Wandb configuration (optional, set report_to="none" to disable)
export WANDB_MODE=offline
export WANDB_DIR="./wandb"  # TODO: Set your wandb directory
export WANDB_PROJECT="diffusionvl"

# Model checkpoint path - Qwen2.5-VL model
# TODO: Download Qwen2.5-VL-7B-Instruct from HuggingFace and convert to DiffusionVL format set the path
PRETRAINED_CHECKPOINT="/inspire/hdd/global_user/chaimingxu-240108540141/models/DiffusionVL-Qwen2.5VL-1.2B-causal"

# Training data paths
# TODO: Set your training data paths
# DATA_PATH="/inspire/hdd/global_user/chaimingxu-240108540141/Diffusion/LaViDa/data/pretrain/json/UniMer_filter_len_1000.json"
# IMAGE_FOLDER="/inspire/hdd/global_user/chaimingxu-240108540141/Diffusion/LaViDa/data/pretrain/UniMer"
DATA_PATH=/inspire/hdd/global_user/chaimingxu-240108540141/Diffusion/data_yaml/fr_ocr_tr_1000.yaml
IMG_PATH=/inspire/hdd/global_user/chaimingxu-240108540141/Diffusion/LaViDa/data/pretrain

# Output directory
# TODO: Set your output directory
OUTPUT_DIR="./outputs"

# ============================================
# Training configuration
# ============================================
# we use 4 node and 8 gpu per node and global batch size is 256
# num_node=$1
# gpu_num=$2
num_node=${1:-1}
gpu_num=${2:-4}
custom_run_name=${3:-"20260505-CSL-16-1.2B"}
BD3LM_BLOCK_SIZE=${4:-16}

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

LLM_VERSION=${PRETRAINED_CHECKPOINT}
VISION_MODEL_VERSION=${PRETRAINED_CHECKPOINT}

echo "Checkpoint: ${PRETRAINED_CHECKPOINT}"

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
    --mm_vision_tower_lr=5e-5 \
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
    --num_train_epochs 20 \
    --max_steps -1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --force_model_type "diffusionvl_qwenvl" \
    --bd3lm_block_aligned_eos True \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation eager \
    --use_conversation_mask False \
    --enable_bd3lm True \
    --bd3lm_block_size ${BD3LM_BLOCK_SIZE} \
    --save_total_limit 3
