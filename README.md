<div align="center">

# Prefix-Adaptive Block Diffusion for Efficient Document Recognition

[![arXiv](https://img.shields.io/badge/arXiv-PA--BDM-b31b1b.svg)](https://arxiv.org/pdf/2605.16861)
<a href="https://github.com/SII-sc22mc/PA-BDM"><img src="https://img.shields.io/badge/GitHub-Repository-black?logo=github" alt="GitHub"></a>
<a href="https://huggingface.co/MingxuChai/PA-BDM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="Hugging Face"></a>

</div>

PA-BDM is a document-recognition model built on Qwen2.5-VL and block diffusion. It targets text, formula, table, and diagram recognition, and is designed to keep the flexible-length generation and KV-cache benefits of Block Diffusion Models while improving their structural consistency and inference speed.

The paper identifies two bottlenecks in standard BDM decoding: tokens are only committed after a full block is finished, and bidirectional intra-block denoising conflicts with left-to-right inter-block generation. PA-BDM addresses them with:

- **Causal intra-block denoising**: tokens inside a candidate block attend from prefix to suffix, matching autoregressive structural order.
- **Confidence-gated Structural Loss (CSL)**: training supervision focuses on the longest reliable masked prefix and avoids noisy gradients from unstable continuations.
- **Progressive Prefix Commitment (PPC)**: inference treats block size as a maximum candidate range, commits the longest reliable prefix into KV cache, and resets the unresolved suffix as a new candidate range.

## Repository Layout

```text
PA-BDM/
|-- infer.ipynb                                      # Notebook inference demo
|-- main.py                                         # CLI inference demo
|-- run_train.sh                                    # One-command training launcher
|-- requirements.txt                                # Pip dependencies except CUDA torch stack
|-- environment.yml                                 # Conda environment with CUDA torch stack
|-- init_env.sh                                     # Editable installs for train/ and eval/
|-- docs/
|   |-- INSTALLATION.md
|   |-- INFERENCE.md
|   `-- TRAINING_EVALUATION.md
|-- train/
|   |-- llava/                                      # PA-BDM / DiffusionVL training code
|   `-- scripts/
|       `-- diffusionvl_qwenvl_finetune_causal_64_include_tr.sh
`-- eval/
    |-- scripts/                                    # lmms-eval launchers
    `-- lmms-eval/
```

## Quick Start

### 1. Create Environment

Conda is recommended for GPU training:

```bash
conda env create -f environment.yml
conda activate pa-bdm
bash init_env.sh
```

Pip-only setup is also possible. Install the CUDA-matched PyTorch stack first, then install the remaining packages:

```bash
conda create -n pa-bdm python=3.10 -y
conda activate pa-bdm
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash init_env.sh
```

### 2. Download Models

For inference, download the released PA-BDM checkpoint:

```bash
huggingface-cli download MingxuChai/PA-BDM --local-dir /path/to/models/PA-BDM
```

For training from Qwen2.5-VL, convert the base model to the DiffusionVL-QwenVL format:

```bash
python scripts/diffusionvl_prepare/convert_qwen2.5vl_to_diffusionvl.py \
  --source_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dest_path /path/to/models/DiffusionVL-Qwen2.5VL-3B-causal
```

### 3. Run Inference

Notebook:

```bash
jupyter lab infer.ipynb
```

CLI:

```bash
python main.py \
  --model-path /path/to/models/PA-BDM \
  --image example_formula.jpg \
  --task formula \
  --gen-length 1024 \
  --steps 32 \
  --confidence-threshold 0.95
```

Task prompts:

- `text`: `Text Recognition.`
- `formula`: `Formula Recognition.`
- `table`: `Table Recognition.`
- `diagram`: `Diagram Recognition.`

### 4. Prepare Training Data

The loader accepts a single JSON/JSONL file or a YAML file that mixes multiple datasets. Each sample follows the LLaVA-style multimodal conversation format:

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

YAML example:

```yaml
datasets:
  - json_path: /path/to/formula_train.json
    image_root: /path/to/formula_images
    sampling_strategy: all
  - json_path: /path/to/table_train.jsonl
    image_root: /path/to/table_images
    sampling_strategy: random:50000
```

Supported sampling strategies are `all`, `first:N`, `end:N`, `random:N`, and percentage forms such as `random:20%`.

### 5. Run Training

The root launcher wraps `train/scripts/diffusionvl_qwenvl_finetune_causal_64_include_tr.sh` and exposes the paths as environment variables:

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

For multi-node training, set the standard torchrun variables:

```bash
NUM_NODES=4 \
GPU_NUM=8 \
MASTER_ADDR=10.0.0.1 \
MASTER_PORT=29199 \
RANK=0 \
bash run_train.sh
```

The paper reports experiments with PA-BDM-1.2B/3B, CSL and PPC confidence threshold `0.95`, and maximum candidate block size commonly set to `32` unless otherwise specified. This repository's `causal_64_include_tr` training script defaults to `BD3LM_BLOCK_SIZE=64`; set `BD3LM_BLOCK_SIZE=32` if you want the paper's default block-size setting.


## Documentation

| Document | Description |
| :--- | :--- |
| [Installation](docs/INSTALLATION.md) | Environment, model download, and editable package setup |
| [Training & Evaluation](docs/TRAINING_EVALUATION.md) | Data format, one-command training, and evaluation scripts |
| [Inference](docs/INFERENCE.md) | Notebook and CLI inference |

## Acknowledgements

This repo is mainly built on [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL), [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), [BD3LMs](https://github.com/kuleshov-group/bd3lms), and [DiffusionVL](https://arxiv.org/abs/2512.15713). We thank the authors for their open-source contributions.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{chai2026prefixadaptiveblockdiffusionefficient,
      title={Prefix-Adaptive Block Diffusion for Efficient Document Recognition},
      author={Mingxu Chai and Ziyu Shen and Chenyu Liu and Kaidi Zhang and Jiazheng Zhang and Dingwei Zhu and Zhiheng Xi and Ruoyu Chen and Jun Long and Jihua Kang and Tao Gui and Qi Zhang},
      year={2026},
      eprint={2605.16861},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.16861},
}
```
