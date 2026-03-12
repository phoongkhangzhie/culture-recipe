#!/usr/bin/env python3
"""
Evaluate a causal LM on the BLEnD multiple-choice-questions benchmark.

Dataset: nayeon212/BLEnD (multiple-choice-questions split, 306k rows)

Each row's `prompt` field is already a fully-formatted MCQ prompt that asks the
model to respond in JSON: {"answer_choice": "<letter>"}. `answer_idx` is the
gold label, one of "A", "B", "C", "D".

Usage:
    python evaluation/blend/eval.py --model <model_path> [options]

Via scripts/eval_blend.sh:
    NUM_GPUS=4 ./scripts/eval_blend.sh /path/to/model

Output:
    evaluation/blend/results/<model-name>/scores.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def parse_answer(text: str) -> str | None:
    """Extract a single letter (A-D) from the model's generated text.

    Tries JSON parsing first (the prompt asks for {"answer_choice": "X"}),
    then falls back to the first standalone A/B/C/D found in the text.
    """
    text = text.strip()

    # 1. JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("answer_choice", "answer", "choice"):
                val = obj.get(key, "")
                if isinstance(val, str):
                    letter = val.strip().upper()
                    if letter in ("A", "B", "C", "D"):
                        return letter
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Regex: first standalone letter A-D
    m = re.search(r'\b([A-D])\b', text.upper())
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on BLEnD multiple-choice questions using vLLM."
    )
    parser.add_argument("--model", required=True, metavar="NAME_OR_PATH",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help=(
                            "Directory to save results. "
                            "Defaults to evaluation/blend/results/<model-name>/."
                        ))
    parser.add_argument("--countries", nargs="+", default=None, metavar="COUNTRY",
                        help="Filter to specific countries (e.g. --countries US KR CN). "
                             "Default: all countries.")
    parser.add_argument("--max-examples", type=int, default=None, metavar="N",
                        help="Cap number of examples (useful for debugging).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, metavar="N",
                        help="Number of GPUs for tensor parallelism (default: 1).")
    parser.add_argument("--batch-size", type=int, default=512, metavar="N",
                        help="vLLM generation batch size (default: 512).")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens to generate per example (default: 50).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy).")
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # ---- Output dir ----
    if args.output_dir is None:
        args.output_dir = str(
            Path("evaluation/blend/results") / Path(args.model).name
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
    print("Loading nayeon212/BLEnD (multiple-choice-questions)…", flush=True)
    from datasets import load_dataset
    dataset = load_dataset("nayeon212/BLEnD", "multiple-choice-questions", split="test")

    if args.countries:
        country_set = set(args.countries)
        dataset = dataset.filter(lambda x: x["country"] in country_set)
        print(f"Filtered to {args.countries} → {len(dataset):,} examples", flush=True)

    if args.max_examples and args.max_examples < len(dataset):
        dataset = dataset.select(range(args.max_examples))
        print(f"Capped to {args.max_examples:,} examples", flush=True)

    print(f"Total examples: {len(dataset):,}", flush=True)

    # ---- Load vLLM ----
    print(f"Loading model: {args.model}", flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # ---- Inference ----
    prompts = [row["prompt"] for row in dataset]
    print(f"Running inference on {len(prompts):,} examples…", flush=True)

    all_outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        outputs = llm.generate(batch, sampling_params)
        all_outputs.extend(outputs)
        done = min(i + args.batch_size, len(prompts))
        print(f"  {done:,}/{len(prompts):,}", flush=True)

    # ---- Score ----
    examples = []
    correct_by_country: dict[str, int] = defaultdict(int)
    total_by_country: dict[str, int] = defaultdict(int)
    n_correct = 0
    n_unparseable = 0

    for row, output in zip(dataset, all_outputs):
        generated = output.outputs[0].text
        predicted = parse_answer(generated)
        gold = row["answer_idx"]
        is_correct = predicted == gold

        if predicted is None:
            n_unparseable += 1
        if is_correct:
            n_correct += 1
            correct_by_country[row["country"]] += 1
        total_by_country[row["country"]] += 1

        examples.append({
            "MCQID": row["MCQID"],
            "country": row["country"],
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "generated": generated,
        })

    n_total = len(examples)
    overall_acc = n_correct / n_total if n_total else 0.0

    by_country = {
        country: {
            "correct": correct_by_country[country],
            "total": total_by_country[country],
            "accuracy": correct_by_country[country] / total_by_country[country],
        }
        for country in sorted(total_by_country)
    }

    results = {
        "model": args.model,
        "dataset": "nayeon212/BLEnD",
        "config": "multiple-choice-questions",
        "filters": {"countries": args.countries},
        "summary": {
            "total": n_total,
            "correct": n_correct,
            "accuracy": overall_acc,
            "unparseable": n_unparseable,
        },
        "by_country": by_country,
        "examples": examples,
    }

    # ---- Save ----
    scores_path = out_dir / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- Print summary ----
    print(f"\n=== BLEnD MCQ Results ===")
    print(f"  Model:       {args.model}")
    print(f"  Total:       {n_total:,}")
    print(f"  Correct:     {n_correct:,}")
    print(f"  Accuracy:    {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    print(f"  Unparseable: {n_unparseable:,}")
    print(f"\nBy country:")
    for country, s in by_country.items():
        print(f"  {country:<20s} {s['accuracy'] * 100:6.2f}%  ({s['correct']}/{s['total']})")
    print(f"\nSaved to {scores_path}")


if __name__ == "__main__":
    main()
