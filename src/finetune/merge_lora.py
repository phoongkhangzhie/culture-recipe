#!/usr/bin/env python3
"""
Merge a LoRA adapter into its base model and save a single merged checkpoint.

After LoRA fine-tuning you have a small adapter directory alongside the base
weights. This script loads both, merges them, and writes a self-contained model
you can serve directly with vLLM or any other inference stack.

Standalone usage:
    python src/finetune/merge_lora.py --base-model Qwen/Qwen2.5-7B-Instruct ...

Via main.py:
    python main.py finetune merge-lora --base-model Qwen/Qwen2.5-7B-Instruct ...
"""

from __future__ import annotations

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register all merge-lora arguments onto an existing parser."""
    parser.add_argument("--base-model", required=True, metavar="NAME_OR_PATH",
                        help="HuggingFace model ID or path of the original base model.")
    parser.add_argument("--lora-dir", required=True, metavar="DIR",
                        help="Directory containing the LoRA adapter (output of train).")
    parser.add_argument("--output-dir", required=True, metavar="DIR",
                        help="Where to save the merged model.")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Save in bfloat16 (default: True).")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")


def run(args: argparse.Namespace) -> None:
    """Execute merge with an already-parsed args namespace."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {args.lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)

    print("Merging weights…")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model."
    )
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
