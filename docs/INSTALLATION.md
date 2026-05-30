# Installation Guide

This guide sets up the environment used by PA-BDM training, evaluation, and inference.

## 1. Clone

```bash
git clone https://github.com/SII-sc22mc/PA-BDM.git
cd PA-BDM
```

## 2. Conda Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate pa-bdm
bash init_env.sh
```

`environment.yml` installs Python 3.10, PyTorch 2.6.0, CUDA 12.4 runtime packages, and the pip dependencies in `requirements.txt`.

## 3. Pip-Only Alternative

If you already manage CUDA/PyTorch yourself:

```bash
conda create -n pa-bdm python=3.10 -y
conda activate pa-bdm
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
bash init_env.sh
```

`init_env.sh` installs the local training package under `train/` and the evaluation package under `eval/lmms-eval/` in editable mode.

## 4. Download Inference Checkpoint

```bash
huggingface-cli download MingxuChai/PA-BDM --local-dir /path/to/models/PA-BDM
```

## 5. Prepare Training Checkpoint

The training script expects a DiffusionVL-QwenVL checkpoint. Convert a Qwen2.5-VL checkpoint first:

```bash
python scripts/diffusionvl_prepare/convert_qwen2.5vl_to_diffusionvl.py \
  --source_path Qwen/Qwen2.5-VL-3B-Instruct \
  --dest_path /path/to/models/DiffusionVL-Qwen2.5VL-3B-causal
```

You can also pass a local Qwen2.5-VL directory to `--source_path`.

## 6. Verify Imports

```bash
python - <<'PY'
import torch
import transformers
import llava
import lmms_eval
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available())
PY
```
