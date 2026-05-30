#!/usr/bin/env python
import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


TASK_PROMPTS = {
    "text": "Text Recognition.",
    "formula": "Formula Recognition.",
    "table": "Table Recognition.",
    "diagram": "Diagram Recognition.",
}


def register_local_model(repo_root: Path) -> None:
    train_dir = repo_root / "train"
    if train_dir.exists():
        sys.path.insert(0, str(train_dir))
    try:
        import llava.model.language_model.llava_diffusionvl_qwenvl  # noqa: F401
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PA-BDM inference on one document image.")
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face repo id for PA-BDM.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS), default="formula", help="Recognition task prompt.")
    parser.add_argument("--prompt", default=None, help="Override the task prompt.")
    parser.add_argument("--device", default="cuda:0", help="Device, for example cuda:0 or cpu.")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--gen-length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32, help="Denoising steps. In the notebook this is set to the block size.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--output-file", default=None, help="Optional path to save decoded text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    register_local_model(repo_root)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    image = Image.open(args.image).convert("RGB")
    prompt = args.prompt or TASK_PROMPTS[args.task]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"<image>\n{prompt}"},
            ],
        }
    ]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=inputs["input_ids"],
            images=inputs.get("pixel_values"),
            image_grid_thws=inputs.get("image_grid_thw"),
            gen_length=args.gen_length,
            steps=args.steps,
            temperature=args.temperature,
            confidence_threshold=args.confidence_threshold,
            without=None,
        )
    elapsed = time.time() - start

    output_text = processor.decode(output_ids[0], skip_special_tokens=False).replace("<|im_end|>", "")
    token_count = len(processor.tokenizer.encode(output_text))

    print(output_text)
    print(f"\n[time] {elapsed:.2f}s")
    if elapsed > 0:
        print(f"[throughput] {token_count / elapsed:.2f} tokens/s")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
