# Training and Evaluation Guide

PA-BDM training is implemented in `train/llava/` and the main script requested for this project is:

```text
train/scripts/diffusionvl_qwenvl_finetune_causal_64_include_tr.sh
```

The root-level `run_train.sh` wraps this script and exposes all machine-specific paths as environment variables.

## Data Format

The data loader accepts:

- a single `.json` file;
- a single `.jsonl` file when referenced from a YAML file;
- a `.yaml` file that mixes multiple datasets.

Minimal JSON sample:

```json
{
  "id": "sample-000001",
  "image": "images/sample.png",
  "conversations": [
    {"from": "human", "value": "<image>\nFormula Recognition."},
    {"from": "gpt", "value": "\\frac{x}{y}"}
  ]
}
```

YAML mixture:

```yaml
datasets:
  - json_path: /path/to/formula_train.json
    image_root: /path/to/formula_images
    sampling_strategy: all
  - json_path: /path/to/table_train.jsonl
    image_root: /path/to/table_images
    sampling_strategy: random:50000
```

Sampling strategies supported by the code are `all`, `first:N`, `end:N`, `random:N`, and percentage forms such as `random:20%`.

## Training From Zero

1. Create the environment:

```bash
conda env create -f environment.yml
conda activate pa-bdm
bash init_env.sh
```

2. Convert Qwen2.5-VL to the DiffusionVL-QwenVL checkpoint format:

```bash
python scripts/diffusionvl_prepare/convert_qwen2.5vl_to_diffusionvl.py \
  --source_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dest_path /path/to/models/DiffusionVL-Qwen2.5VL-3B-causal
```

3. Launch training:

```bash
PRETRAINED_CHECKPOINT=/path/to/models/DiffusionVL-Qwen2.5VL-3B-causal \
DATA_PATH=/path/to/train_data.yaml \
IMAGE_FOLDER=/path/to/images \
OUTPUT_DIR=/path/to/outputs \
GPU_NUM=4 \
RUN_NAME=pa-bdm-csl-64 \
BD3LM_BLOCK_SIZE=64 \
bash run_train.sh
```

`IMAGE_FOLDER` may be omitted when every sample uses an absolute image path or when `image_root` is specified in the YAML file.

## Multi-Node Training

```bash
NUM_NODES=4 \
GPU_NUM=8 \
MASTER_ADDR=10.0.0.1 \
MASTER_PORT=29199 \
RANK=0 \
PRETRAINED_CHECKPOINT=/path/to/models/DiffusionVL-Qwen2.5VL-3B-causal \
DATA_PATH=/path/to/train_data.yaml \
IMAGE_FOLDER=/path/to/images \
bash run_train.sh
```

Set `RANK` separately on each node.

## Key Hyperparameters

The training launcher uses:

- `--force_model_type diffusionvl_qwenvl`
- `--enable_bd3lm True`
- `--bd3lm_block_aligned_eos True`
- `--bd3lm_block_size ${BD3LM_BLOCK_SIZE}`
- `--model_max_length 4096`
- `--per_device_train_batch_size 6`
- `--gradient_accumulation_steps 3`
- `--learning_rate 1e-5`
- `--num_train_epochs 15`
- `--attn_implementation eager`

The paper reports PA-BDM with causal intra-block denoising, CSL, and PPC. It commonly uses maximum candidate block size `32` and confidence threshold `0.95` unless otherwise specified. The provided `causal_64_include_tr` script defaults to block size `64`; set `BD3LM_BLOCK_SIZE=32` to match that paper default.

`model_max_length` must be divisible by `BD3LM_BLOCK_SIZE`; this is checked by `train/llava/train/train.py`.

## Logging

By default, `run_train.sh` sets:

```bash
WANDB_MODE=offline
WANDB_PROJECT=pa-bdm
REPORT_TO=wandb
```

Disable Weights & Biases:

```bash
REPORT_TO=none bash run_train.sh
```

## Evaluation

The DiffusionVL-QwenVL evaluation launcher is:

```bash
cd eval
bash scripts/diffusionvl_qwenvl.sh
```

Before running, edit:

- `MODEL_PATHS`
- `OUTPUT_PATH`
- `TASK_NAMES`
- `TOTAL_GPUS`
- `BLOCK_SIZE`
- `STEPS`

The paper evaluates:

- text recognition with Edit Distance;
- formula recognition with CDM;
- table recognition with TEDS;
- diagram recognition with graph F1 from parsed Mermaid graphs.

The main paper benchmarks cover DocLayNet/OmniDoc text, UniMER formula subsets, PubTableNet/FinTabNet tables, and FlowLearn diagrams.
