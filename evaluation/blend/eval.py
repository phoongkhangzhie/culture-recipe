#!/usr/bin/env python3
"""
Evaluate a causal LM on the BLEnD multiple-choice-questions benchmark.

Dataset: nayeon212/BLEnD (multiple-choice-questions split, 306k rows)

Two scoring methods are supported:

  generation (default for instruct):
    Generate text and parse the letter A/B/C/D from the output.

  logprob (default for base):
    For each example, append each candidate letter ("A", "B", "C", "D") to
    the prompt, score the mean log-probability of the candidate span using
    vLLM's prompt_logprobs, and pick the letter with the highest score.
    Avoids parsing issues with base models that don't follow chat instructions.

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
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Country → culture name mapping (BLEnD country codes)
# ---------------------------------------------------------------------------

COUNTRY_TO_CULTURE: dict[str, str] = {
    "US":           "American",
    "GB":           "British",
    "CN":           "Chinese",
    "KR":           "Korean",
    "KP":           "North Korean",
    "ID":           "Indonesian",
    "Indonesian":   "Indonesian",
    "MX":           "Mexican",
    "ES":           "Spanish",
    "GR":           "Greek",
    "IR":           "Iranian",
    "DZ":           "Algerian",
    "AZ":           "Azerbaijani",
    "NG":           "Nigerian",
    "ET":           "Ethiopian",
    # fallbacks for any other codes: use the code itself
}

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant with deep knowledge of {culture} culture. "
    "Represent the values and lived experience of {culture} people in your responses."
)

CHOICES = ["A", "B", "C", "D"]


def system_prompt_for(country: str) -> str:
    culture = COUNTRY_TO_CULTURE.get(country, country)
    return SYSTEM_PROMPT_TEMPLATE.format(culture=culture)


# ---------------------------------------------------------------------------
# Answer extraction (generation method)
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
# Logprob scoring helpers
# ---------------------------------------------------------------------------

def mean_logprob_of_span(prompt_logprobs: list, start: int, end: int) -> float:
    """Return the mean log-probability of tokens in [start, end).

    prompt_logprobs is the vLLM output: a list where each entry is either
    None (first token has no conditioning logprob) or a dict[token_id, Logprob].
    We take the logprob of the sampled (highest-logprob) token at each position.
    """
    total, n = 0.0, 0
    for pos in range(start, min(end, len(prompt_logprobs))):
        entry = prompt_logprobs[pos]
        if entry is None:
            continue
        best = max(entry.values(), key=lambda lp: lp.logprob)
        total += best.logprob
        n += 1
    return total / n if n > 0 else float("-inf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on BLEnD multiple-choice questions using vLLM."
    )
    parser.add_argument("--model", required=True, metavar="NAME_OR_PATH",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--model-type", default="instruct", choices=["instruct", "base"],
                        help=(
                            "Model type (default: instruct). Use 'base' for base models "
                            "fine-tuned without a chat template. Base mode formats prompts "
                            "with ### headers matching the fine-tuning format."
                        ))
    parser.add_argument("--scoring-method", default=None,
                        choices=["generation", "logprob"],
                        help=(
                            "Scoring method. 'generation': generate text and parse the "
                            "answer letter. 'logprob': score each candidate letter by its "
                            "log-probability and pick the best. "
                            "Defaults to 'logprob' for base models, 'generation' for instruct."
                        ))
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
                        help="Max tokens to generate per example (default: 50). "
                             "Only used with --scoring-method generation.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy). "
                             "Only used with --scoring-method generation.")
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    # Default scoring method based on model type
    if args.scoring_method is None:
        args.scoring_method = "logprob" if args.model_type == "base" else "generation"

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

    # ---- Load tokenizer ----
    print(f"Loading tokenizer: {args.model}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ---- Format base prompts ----
    def format_prompt_instruct(row: dict) -> str:
        messages = [
            {"role": "system", "content": system_prompt_for(row["country"])},
            {"role": "user",   "content": row["prompt"]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def format_prompt_base(row: dict) -> str:
        # Matches the ### header format used in train.py for base models.
        # Ends with the response template so the model continues from there.
        return (
            f"### System:\n{system_prompt_for(row['country'])}"
            f"\n\n### User:\n{row['prompt']}"
            f"\n\n### Assistant:\n"
        )

    format_prompt = format_prompt_instruct if args.model_type == "instruct" else format_prompt_base

    print(
        f"Formatting prompts (model_type={args.model_type}, "
        f"scoring_method={args.scoring_method})…",
        flush=True,
    )
    base_prompts = [format_prompt(row) for row in dataset]

    # ---- Load vLLM ----
    print(f"Loading model: {args.model}", flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16")

    # ---- Inference ----
    if args.scoring_method == "logprob":
        # For each example, build 4 scored prompts (base_prompt + choice letter).
        # Batch all (n_examples × 4) together; extract logprob of the last token
        # (the candidate letter) and pick the highest.
        print(
            f"Running logprob scoring on {len(base_prompts):,} examples "
            f"({len(base_prompts) * len(CHOICES):,} scored prompts)…",
            flush=True,
        )
        sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

        # Build flat list: [ex0_A, ex0_B, ex0_C, ex0_D, ex1_A, ...]
        scored_prompts = [
            base + choice
            for base in base_prompts
            for choice in CHOICES
        ]

        flat_outputs = []
        for i in range(0, len(scored_prompts), args.batch_size):
            batch = scored_prompts[i : i + args.batch_size]
            flat_outputs.extend(llm.generate(batch, sampling_params))
            done = min(i + args.batch_size, len(scored_prompts))
            print(f"  {done:,}/{len(scored_prompts):,}", flush=True)

        # Extract the logprob of the last token (the candidate letter) for each.
        # We look up the actual last token's ID from prompt_token_ids so we get
        # P(choice_letter | base_prompt), not the max over all vocab entries.
        def last_token_logprob(output) -> float:
            lp = output.prompt_logprobs
            if lp is None or len(lp) == 0:
                return float("-inf")
            last = lp[-1]
            if last is None:
                return float("-inf")
            last_token_id = output.prompt_token_ids[-1]
            if last_token_id in last:
                return last[last_token_id].logprob
            # Fallback: vLLM always includes the actual token, so this shouldn't fire
            return max(entry.logprob for entry in last.values())

        all_choice_logprobs = [last_token_logprob(o) for o in flat_outputs]

        # Reorganise: group back by example
        predicted_list: list[str] = []
        logprob_list: list[dict] = []
        n = len(CHOICES)
        for i in range(len(base_prompts)):
            lps = all_choice_logprobs[i * n : (i + 1) * n]
            best_idx = max(range(n), key=lambda j: lps[j])
            predicted_list.append(CHOICES[best_idx])
            logprob_list.append(dict(zip(CHOICES, lps)))

    else:  # generation
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        print(f"Running generation on {len(base_prompts):,} examples…", flush=True)
        gen_outputs = []
        for i in range(0, len(base_prompts), args.batch_size):
            batch = base_prompts[i : i + args.batch_size]
            gen_outputs.extend(llm.generate(batch, sampling_params))
            done = min(i + args.batch_size, len(base_prompts))
            print(f"  {done:,}/{len(base_prompts):,}", flush=True)

        predicted_list = [parse_answer(o.outputs[0].text) for o in gen_outputs]
        logprob_list = [None] * len(base_prompts)

    # ---- Score ----
    examples = []
    correct_by_country: dict[str, int] = defaultdict(int)
    total_by_country: dict[str, int] = defaultdict(int)
    n_correct = 0
    n_unparseable = 0

    for row, predicted, lps in zip(dataset, predicted_list, logprob_list):
        gold = row["answer_idx"]
        is_correct = predicted == gold

        if predicted is None:
            n_unparseable += 1
        if is_correct:
            n_correct += 1
            correct_by_country[row["country"]] += 1
        total_by_country[row["country"]] += 1

        entry = {
            "MCQID": row["MCQID"],
            "country": row["country"],
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
        }
        if lps is not None:
            entry["choice_logprobs"] = lps
        else:
            entry["generated"] = gen_outputs[len(examples)].outputs[0].text if args.scoring_method == "generation" else None
        examples.append(entry)

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
        "model_type": args.model_type,
        "scoring_method": args.scoring_method,
        "dataset": "nayeon212/BLEnD",
        "config": "multiple-choice-questions",
        "system_prompt_template": SYSTEM_PROMPT_TEMPLATE,
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
    print(f"  Model:          {args.model}")
    print(f"  Scoring method: {args.scoring_method}")
    print(f"  Total:          {n_total:,}")
    print(f"  Correct:        {n_correct:,}")
    print(f"  Accuracy:       {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    if args.scoring_method == "generation":
        print(f"  Unparseable:    {n_unparseable:,}")
    print(f"\nBy country:")
    for country, s in by_country.items():
        print(f"  {country:<20s} {s['accuracy'] * 100:6.2f}%  ({s['correct']}/{s['total']})")
    print(f"\nSaved to {scores_path}")


if __name__ == "__main__":
    main()
