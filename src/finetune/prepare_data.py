#!/usr/bin/env python3
"""
Prepare training data from culture-recipe output files.

Walks one or more output directories, extracts the `messages` from every
approved (or all) example records, and writes a JSONL file where each line is:

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}

Standalone usage:
    python src/finetune/prepare_data.py --input-dirs ./output/japanese_english ...

Via main.py:
    python main.py finetune prepare-data --input-dirs ./output/japanese_english ...
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

# Registry: name -> fn(candidates: list[dict], args) -> list[float]
# Each function receives the full candidate list (with "_meta") across ALL
# groups and returns a parallel list of scores. Higher score = more preferred.
_STRATEGIES: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _STRATEGIES[name] = fn
        return fn
    return decorator


@_register("perplexity")
def _strategy_perplexity(candidates: list[dict], args) -> list[float]:
    """
    Score by mean NLL of assistant turns under the selection model (via vLLM).
    Higher NLL = model more surprised = harder example = more training value.
    Requires --selection-model.

    All candidates are scored in a single batched vLLM call for efficiency.
    Returns (scores, n_tokens_per_example) via args._scoring_details for logging.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = args.selection_model
    if not model_name:
        raise ValueError("--selection-model is required for the 'perplexity' strategy.")

    # Load lazily; cache on args so we only load once across all batches.
    if not hasattr(args, "_selection_model_cache"):
        print(f"Loading selection model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        args._selection_model_cache = (llm, tokenizer)

    llm, tokenizer = args._selection_model_cache

    # Build full tokenized sequences and record assistant token spans per example.
    prompts: list[str] = []
    assistant_spans: list[list[tuple[int, int]]] = []  # [(start, end), ...]

    for ex in candidates:
        messages = ex["messages"]

        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompts.append(tokenizer.decode(full_ids))

        spans = []
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prefix_ids = tokenizer.apply_chat_template(
                messages[:i],
                tokenize=True,
                add_generation_prompt=True,
            )
            turn_ids = tokenizer.apply_chat_template(
                messages[:i + 1],
                tokenize=True,
                add_generation_prompt=False,
            )
            start, end = len(prefix_ids), len(turn_ids)
            if end > start:
                spans.append((start, end))
        assistant_spans.append(spans)

    # Single batched call: prompt_logprobs=1 returns the log-prob of each
    # input token given all preceding tokens.
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(prompts, sampling_params)

    scores = []
    n_tokens_list = []
    for output, spans in zip(outputs, assistant_spans):
        logprobs = output.prompt_logprobs  # list[None | dict[int, Logprob]]
        if logprobs is None:
            scores.append(0.0)
            n_tokens_list.append(0)
            continue

        total_nll, n_tokens = 0.0, 0
        for start, end in spans:
            for pos in range(start, min(end, len(logprobs))):
                entry = logprobs[pos]
                if entry is None:
                    continue
                # entry is {token_id: Logprob}; pick the top (sampled) token
                best = max(entry.values(), key=lambda lp: lp.logprob)
                total_nll += -best.logprob
                n_tokens += 1

        scores.append(total_nll / n_tokens if n_tokens > 0 else 0.0)
        n_tokens_list.append(n_tokens)

    # Stash per-example token counts for the score logger
    args._scoring_n_tokens = n_tokens_list

    return scores


@_register("random")
def _strategy_random(candidates: list[dict], args) -> list[float]:
    """Randomly score candidates (useful for ablations / baselines)."""
    scores = [random.random() for _ in candidates]
    args._scoring_n_tokens = [None] * len(candidates)
    return scores


# ---------------------------------------------------------------------------
# Core record iteration
# ---------------------------------------------------------------------------

def iter_records(input_dirs: list[Path], approved_only: bool, min_score: float):
    """Yield (messages, metadata) for every qualifying record."""
    for dir_path in input_dirs:
        for json_path in sorted(dir_path.rglob("*.json")):
            if json_path.name in ("progress.json",) or "trace" in json_path.parts:
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            records = data.get("records", [])
            for record in records:
                verification = record.get("verification", {})
                score = verification.get("overall_score", 0.0)
                is_approved = verification.get("is_approved", False)

                if approved_only and not is_approved:
                    continue
                if score < min_score:
                    continue

                messages = record.get("example", {}).get("content", {}).get("messages")
                if not messages:
                    continue

                meta = {
                    "source_file": str(json_path),
                    "culture": data.get("culture"),
                    "dimension": data.get("dimension", {}).get("name"),
                    "overall_score": score,
                    "is_approved": is_approved,
                }
                yield messages, meta


# ---------------------------------------------------------------------------
# Top-k selection
# ---------------------------------------------------------------------------

def apply_topk(
    examples: list[dict],
    topk: int,
    strategy: str,
    args,
) -> tuple[list[dict], list[dict]]:
    """
    For each source-file group with more than `topk` candidates, score them
    with `strategy` and keep the top-k highest-scoring examples.
    Groups with <= topk candidates are kept as-is.

    Returns (selected_examples, score_log_entries) where each score_log_entry is:
        {
            "source_file", "culture", "dimension",
            "verification_score", "selection_score", "n_tokens_scored",
            "selected", "rank"
        }
    """
    score_fn = _STRATEGIES.get(strategy)
    if score_fn is None:
        raise ValueError(
            f"Unknown selection strategy '{strategy}'. "
            f"Available: {sorted(_STRATEGIES)}"
        )

    # Group by source file (each file = one dimension)
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        groups[ex["_meta"]["source_file"]].append(ex)

    # Separate groups that need scoring from those that don't
    needs_scoring: list[dict] = []
    group_slices: list[tuple[str, int, int]] = []
    passthrough: list[dict] = []

    for source_file, group in groups.items():
        if len(group) <= topk:
            passthrough.extend(group)
        else:
            start = len(needs_scoring)
            needs_scoring.extend(group)
            group_slices.append((source_file, start, len(needs_scoring)))

    selected = list(passthrough)
    score_log: list[dict] = []
    n_reduced = 0

    if needs_scoring:
        print(
            f"Scoring {len(needs_scoring)} candidates across "
            f"{len(group_slices)} dimension(s) with strategy='{strategy}'..."
        )
        all_scores = score_fn(needs_scoring, args)
        n_tokens_all = getattr(args, "_scoring_n_tokens", [None] * len(needs_scoring))

        for source_file, start, end in group_slices:
            group = needs_scoring[start:end]
            group_scores = all_scores[start:end]
            group_n_tokens = n_tokens_all[start:end]

            ranked = sorted(
                enumerate(zip(group_scores, group, group_n_tokens)),
                key=lambda x: x[1][0],
                reverse=True,
            )

            selected_indices = {orig_idx for orig_idx, _ in ranked[:topk]}

            for rank, (orig_idx, (score, ex, n_tok)) in enumerate(ranked):
                meta = ex["_meta"]
                score_log.append({
                    "source_file": meta["source_file"],
                    "culture": meta["culture"],
                    "dimension": meta["dimension"],
                    "verification_score": meta["overall_score"],
                    "selection_score": score,
                    "n_tokens_scored": n_tok,
                    "rank": rank + 1,
                    "selected": orig_idx in selected_indices,
                })

            selected.extend(ex for _, (_, ex, _) in ranked[:topk])
            n_reduced += 1

    if n_reduced:
        print(
            f"Top-k selection ({strategy}, k={topk}): "
            f"reduced {n_reduced} dimension(s) to at most {topk} example(s) each."
        )

    return selected, score_log


# ---------------------------------------------------------------------------
# Score log writer
# ---------------------------------------------------------------------------

def _dim_stats(scores: list[float]) -> dict:
    if not scores:
        return {}
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return {
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
    }


def write_score_log(
    path: Path,
    score_log: list[dict],
    strategy: str,
    model: str | None,
    topk: int,
) -> None:
    """Write a JSON score log with per-example entries and per-dimension summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-dimension summaries
    by_dim: dict[str, list[dict]] = defaultdict(list)
    for entry in score_log:
        key = f"{entry['culture']}::{entry['dimension']}"
        by_dim[key].append(entry)

    dimension_summaries = []
    for key, entries in sorted(by_dim.items()):
        all_scores = [e["selection_score"] for e in entries]
        selected_scores = [e["selection_score"] for e in entries if e["selected"]]
        culture, dimension = key.split("::", 1)
        dimension_summaries.append({
            "culture": culture,
            "dimension": dimension,
            "n_candidates": len(entries),
            "n_selected": sum(1 for e in entries if e["selected"]),
            "all_scores": _dim_stats(all_scores),
            "selected_scores": _dim_stats(selected_scores),
        })

    all_sel_scores = [e["selection_score"] for e in score_log if e["selected"]]
    all_ver_scores = [e["verification_score"] for e in score_log if e["selected"]]

    log = {
        "config": {
            "strategy": strategy,
            "model": model,
            "topk": topk,
        },
        "global_summary": {
            "total_candidates": len(score_log),
            "total_selected": sum(1 for e in score_log if e["selected"]),
            "selection_score": _dim_stats(all_sel_scores),
            "verification_score": _dim_stats(all_ver_scores),
        },
        "dimension_summaries": dimension_summaries,
        "examples": score_log,
    }

    path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"Wrote score log ({len(score_log)} entries) → {path}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register all prepare-data arguments onto an existing parser."""
    parser.add_argument(
        "--input-dirs", nargs="+", required=True, metavar="DIR",
        help="One or more output directories from culture-recipe runs.",
    )
    parser.add_argument(
        "--output", required=True, metavar="FILE",
        help=(
            "Path for the output JSONL inside prepared_output/ "
            "(e.g. prepared_output/japanese_english_train.jsonl)."
        ),
    )
    parser.add_argument(
        "--approved-only", action="store_true",
        help="Only include records where is_approved=true.",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0, metavar="SCORE",
        help="Minimum overall_score to include a record (default: 0.0).",
    )
    parser.add_argument(
        "--split", type=float, default=None, metavar="RATIO",
        help=(
            "If set, split into train/val files at this ratio (e.g. 0.9). "
            "Val file is written alongside the output file as <name>_val.jsonl."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling / random strategy (default: 42).",
    )
    # Top-k selection
    parser.add_argument(
        "--topk", type=int, default=None, metavar="K",
        help=(
            "For dimensions with more than K examples, keep only the top-K "
            "according to --selection-strategy. Dimensions with <= K examples "
            "are kept as-is. Omit to keep all examples."
        ),
    )
    parser.add_argument(
        "--selection-strategy",
        default="perplexity",
        choices=sorted(_STRATEGIES),
        metavar="STRATEGY",
        help=(
            f"Scoring strategy used with --topk. "
            f"Choices: {sorted(_STRATEGIES)}. Default: perplexity."
        ),
    )
    parser.add_argument(
        "--selection-model", default=None, metavar="NAME_OR_PATH",
        help=(
            "Model used for the 'perplexity' strategy (HuggingFace ID or local path). "
            "Loaded via vLLM for efficient batched scoring."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, metavar="N",
        help="Number of GPUs to use for tensor parallelism when loading the selection model via vLLM (default: 1).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute prepare-data with an already-parsed args namespace."""
    input_dirs = [Path(d) for d in args.input_dirs]
    for d in input_dirs:
        if not d.is_dir():
            raise ValueError(f"Not a directory: {d}")

    examples = []
    for messages, meta in iter_records(input_dirs, args.approved_only, args.min_score):
        examples.append({"messages": messages, "_meta": meta})

    if not examples:
        print("No examples found — check --input-dirs and filter flags.")
        return

    # Apply top-k selection before shuffling
    score_log = []
    if args.topk is not None:
        examples, score_log = apply_topk(examples, args.topk, args.selection_strategy, args)

    random.seed(args.seed)
    random.shuffle(examples)

    def strip_meta(items):
        return [{"messages": e["messages"]} for e in items]

    output_path = Path(args.output)

    if args.split is not None:
        n_train = max(1, int(len(examples) * args.split))
        train_items = examples[:n_train]
        val_items = examples[n_train:]
        val_path = output_path.with_name(output_path.stem + "_val" + output_path.suffix)
        write_jsonl(output_path, strip_meta(train_items))
        write_jsonl(val_path, strip_meta(val_items))
        print(
            f"Wrote {len(train_items)} train examples → {output_path}\n"
            f"Wrote {len(val_items)} val examples   → {val_path}"
        )
    else:
        write_jsonl(output_path, strip_meta(examples))
        print(f"Wrote {len(examples)} examples → {output_path}")

    # Write score log into prepared_output/scores/
    if score_log:
        scores_dir = output_path.parent / "scores"
        score_log_path = scores_dir / (output_path.stem + "_scores.json")
        write_score_log(
            score_log_path,
            score_log,
            strategy=args.selection_strategy,
            model=args.selection_model,
            topk=args.topk,
        )

    cultures = {e["_meta"]["culture"] for e in examples}
    dims = {e["_meta"]["dimension"] for e in examples}
    scores = [e["_meta"]["overall_score"] for e in examples]
    print(
        f"\nSummary:\n"
        f"  Cultures:   {len(cultures)}  ({', '.join(sorted(cultures))})\n"
        f"  Dimensions: {len(dims)}\n"
        f"  Avg score:  {sum(scores)/len(scores):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten culture-recipe output files into a training JSONL."
    )
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
