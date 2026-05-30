# Inference Guide

## Model

| Model | Base Model | Download |
| :--- | :--- | :--- |
| PA-BDM-3B | Qwen2.5-VL-3B | [Hugging Face](https://huggingface.co/MingxuChai/PA-BDM) |

```bash
huggingface-cli download MingxuChai/PA-BDM --local-dir /path/to/models/PA-BDM
```

## Notebook

The original demo is in `infer.ipynb`:

```bash
jupyter lab infer.ipynb
```

The notebook loads `AutoModelForCausalLM` and `AutoProcessor`, then calls:

```python
model.generate(
    inputs=inputs["input_ids"],
    images=inputs.get("pixel_values"),
    image_grid_thws=inputs.get("image_grid_thw"),
    gen_length=1024,
    steps=32,
    temperature=0.0,
    confidence_threshold=0.95,
)
```

## CLI

`main.py` provides the same single-image flow without opening a notebook:

```bash
python main.py \
  --model-path /path/to/models/PA-BDM \
  --image example_formula.jpg \
  --task formula \
  --gen-length 1024 \
  --steps 32 \
  --confidence-threshold 0.95
```

Available task presets:

| Task | Prompt |
| :--- | :--- |
| `text` | `Text Recognition.` |
| `formula` | `Formula Recognition.` |
| `table` | `Table Recognition.` |
| `diagram` | `Diagram Recognition.` |

Use a custom prompt when needed:

```bash
python main.py \
  --model-path /path/to/models/PA-BDM \
  --image /path/to/image.png \
  --prompt "Formula Recognition."
```

## Important Parameters

- `--gen-length`: maximum generated response length.
- `--steps`: denoising steps; the released notebook uses `32`.
- `--confidence-threshold`: PPC confidence threshold; the paper default is `0.95`.
- `--temperature`: usually `0.0` for deterministic recognition.
- `--dtype`: `bfloat16` by default.

For paper-style throughput reporting, run with batch size 1 and compute generated tokens per second. `main.py` prints elapsed time and approximate token throughput.
