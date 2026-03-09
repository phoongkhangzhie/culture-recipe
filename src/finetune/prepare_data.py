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
import random
from pathlib import Path


def iter_records(input_dirs: list[Path], approved_only: bool, min_score: float):
    """Yield (messages, metadata) for every qualifying record."""
    for dir_path in input_dirs:
        for json_path in sorted(dir_path.rglob("*.json")):
            # Skip progress / trace files
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


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register all prepare-data arguments onto an existing parser."""
    parser.add_argument(
        "--input-dirs", nargs="+", required=True, metavar="DIR",
        help="One or more output directories from culture-recipe runs.",
    )
    parser.add_argument(
        "--output", required=True, metavar="FILE",
        help="Path for the output JSONL (e.g. train.jsonl).",
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
        help="Random seed for the train/val split (default: 42).",
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
