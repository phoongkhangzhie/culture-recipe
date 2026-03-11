#!/usr/bin/env python3
"""
Fine-tune a causal LM on culture-recipe training data using TRL's SFTTrainer.

Supports both full fine-tuning and LoRA (via PEFT). The training data must be a
JSONL file where each line has a "messages" key in the OpenAI chat format:

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}

Standalone usage:
    python src/finetune/train.py --model Qwen/Qwen2.5-7B-Instruct ...

Via main.py:
    python main.py finetune train --model Qwen/Qwen2.5-7B-Instruct ...

Multi-GPU (via accelerate launch):
    accelerate launch src/finetune/train.py --model ... --train-file ... --lora ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register all train arguments onto an existing parser."""

    # ---- Model ----
    parser.add_argument("--model", required=True, metavar="NAME_OR_PATH",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help=(
                            "Directory to save checkpoints and the final model. "
                            "If omitted, derived from --train-file: "
                            "checkpoints/<train-file-stem> (stripping a trailing '-train')."
                        ))

    # ---- Data ----
    parser.add_argument("--train-file", required=True, metavar="FILE",
                        help="Training JSONL file (one {messages: [...]} per line).")
    parser.add_argument("--val-file", default=None, metavar="FILE",
                        help="Optional validation JSONL file.")

    # ---- LoRA ----
    parser.add_argument("--lora", action="store_true",
                        help="Use LoRA (PEFT) instead of full fine-tuning.")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (default: 16).")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32).")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05).")
    parser.add_argument("--lora-target-modules", default="all-linear",
                        help="LoRA target modules (default: all-linear).")

    # ---- Training hyperparameters ----
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3).")
    parser.add_argument("--per-device-batch-size", type=int, default=2,
                        help="Per-device training batch size (default: 2).")
    parser.add_argument("--grad-accumulation-steps", type=int, default=8,
                        help="Gradient accumulation steps (default: 8).")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4).")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="Warmup ratio (default: 0.03).")
    parser.add_argument("--lr-scheduler", default="cosine",
                        help="LR scheduler type (default: cosine).")
    parser.add_argument("--max-seq-length", type=int, default=16384,
                        help="Maximum sequence length in tokens (default: 16384).")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01).")
    parser.add_argument("--optim", default="adamw_torch",
                        help="Optimizer name as accepted by HuggingFace TrainingArguments "
                             "(default: adamw_torch). Use 'adamw_bnb_8bit' for 8-bit AdamW "
                             "via bitsandbytes.")

    # ---- Precision / memory ----
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 (default: True).")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false",
                        help="Disable bfloat16 and use fp32.")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing (default: True).")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")

    # ---- Logging / saving ----
    parser.add_argument("--logging-steps", type=int, default=10)


def run(args: argparse.Namespace) -> None:
    """Execute training with an already-parsed args namespace."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # ---- Derive output dir ----
    if args.output_dir is None:
        stem = Path(args.train_file).stem
        if stem.endswith("-train"):
            stem = stem[: -len("-train")]
        args.output_dir = str(Path("checkpoints") / stem)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    # ---- LoRA config ----
    peft_config = None
    if args.lora:
        from peft import LoraConfig

        target = args.lora_target_modules
        if target != "all-linear":
            target = [m.strip() for m in target.split(",")]

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # ---- Datasets ----
    def load_jsonl(path):
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    train_records = load_jsonl(args.train_file)
    train_dataset = Dataset.from_list(train_records)

    eval_dataset = None
    if args.val_file:
        val_records = load_jsonl(args.val_file)
        eval_dataset = Dataset.from_list(val_records)

    # ---- Training config ----
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        optim=args.optim,
        bf16=args.bf16,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=eval_dataset is not None,
        report_to="none",
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
    )

    mode = "LoRA" if args.lora else "Full fine-tuning"
    print(f"\n=== culture-recipe fine-tuning ({mode}) ===")
    print(f"  Model:          {args.model}")
    print(f"  Train examples: {len(train_records)}")
    if eval_dataset:
        print(f"  Val examples:   {len(val_records)}")
    if args.lora:
        print(f"  LoRA rank:      {args.lora_r}  alpha: {args.lora_alpha}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.per_device_batch_size} × {args.grad_accumulation_steps} accum steps")
    print(f"  Output:         {args.output_dir}\n")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning (full or LoRA) on culture-recipe data."
    )
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
