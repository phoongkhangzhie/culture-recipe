#!/usr/bin/env python3
"""
culture-recipe — unified CLI entry point.

Data generation:
  python main.py generate --culture Japanese --dimension power_distance
  python main.py generate --culture Japanese --all-dimensions --output-dir ./results/japanese
  python main.py generate --list-dimensions

Fine-tuning:
  python main.py finetune prepare-data --input-dirs ./output/japanese_english --output train.jsonl
  python main.py finetune train --model Qwen/Qwen2.5-7B-Instruct --train-file train.jsonl --lora
  python main.py finetune merge-lora --base-model Qwen/Qwen2.5-7B-Instruct --lora-dir ./checkpoints/lora --output-dir ./checkpoints/lora-merged
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.syntax import Syntax

from src.data_generation.agent import run_pipeline
from src.data_generation.models import GenerationParams, GenerationResult
from src.data_generation.taxonomy import CULTURAL_DIMENSIONS, get_dimension

console = Console()


# ---------------------------------------------------------------------------
# generate subcommand — argument parser
# ---------------------------------------------------------------------------

def _build_generate_parser(subparsers):
    gen = subparsers.add_parser(
        "generate",
        help="Generate culturally-aligned LLM training examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run the agentic data generation pipeline.",
        epilog="""\
Examples:
  python main.py generate --culture Japanese --dimension power_distance
  python main.py generate --culture Nigerian --dimension hospitality --output result.json
  python main.py generate --culture Japanese --all-dimensions --output-dir ./results/japanese
  python main.py generate --culture Brazilian --dimensions power_distance,collectivism --output-dir ./out
  python main.py generate --list-dimensions
""",
    )

    # ---- Target ----
    gen.add_argument("--culture", metavar="NAME",
                     help="Target culture, e.g. 'Japanese', 'Nigerian', 'Brazilian'")

    dim_group = gen.add_mutually_exclusive_group()
    dim_group.add_argument("--dimension", metavar="KEY",
                           help="Single cultural dimension key")
    dim_group.add_argument("--dimensions", metavar="KEY,KEY,...",
                           help="Comma-separated list of dimension keys to run sequentially")
    dim_group.add_argument("--all-dimensions", action="store_true",
                           help="Run all available dimensions sequentially (resumable)")

    # ---- Generation ----
    gen.add_argument("--language", default="English", metavar="LANG",
                     help="Language for the generated examples (default: English)")
    gen.add_argument("--topic", metavar="TEXT",
                     help="Optional topic hint — the agent may use this as inspiration")
    gen.add_argument("--implicit-culture", action="store_true",
                     help=(
                         "User messages written as a natural insider of the culture; "
                         "assistant responds with culturally shared assumptions."
                     ))

    # ---- Output ----
    out_group = gen.add_mutually_exclusive_group()
    out_group.add_argument("--output", metavar="FILE",
                           help="Save single-dimension result to a JSON file")
    out_group.add_argument("--output-dir", metavar="DIR",
                           help="Directory for multi-dimension results (one file per dimension)")

    # ---- Misc ----
    gen.add_argument("--verbose", action="store_true",
                     help="Show detailed progress including model output")
    gen.add_argument("--trace", action="store_true",
                     help="Record a full pipeline trace alongside each result")
    gen.add_argument("--list-dimensions", action="store_true",
                     help="Print all available cultural dimensions and exit")

    return gen


# ---------------------------------------------------------------------------
# finetune subcommand — argument parsers
# ---------------------------------------------------------------------------

def _build_finetune_parser(subparsers):
    from src.finetune import prepare_data, train, merge_lora

    ft = subparsers.add_parser(
        "finetune",
        help="Fine-tune a model on generated culture-recipe data.",
        description="Fine-tuning pipeline: prepare data, train (full FT or LoRA), merge adapter.",
    )
    ft_sub = ft.add_subparsers(dest="finetune_cmd", metavar="COMMAND")
    ft_sub.required = True

    # ---- prepare-data ----
    prep = ft_sub.add_parser(
        "prepare-data",
        help="Flatten output JSONs into a training JSONL.",
        description=prepare_data.__doc__,
    )
    prepare_data.add_arguments(prep)

    # ---- train ----
    tr = ft_sub.add_parser(
        "train",
        help="SFT fine-tuning (full or LoRA) on culture-recipe data.",
        description=train.__doc__,
    )
    train.add_arguments(tr)

    # ---- merge-lora ----
    mg = ft_sub.add_parser(
        "merge-lora",
        help="Merge a LoRA adapter into its base model.",
        description=merge_lora.__doc__,
    )
    merge_lora.add_arguments(mg)

    return ft


# ---------------------------------------------------------------------------
# generate helpers
# ---------------------------------------------------------------------------

def _print_dimensions() -> None:
    from rich.table import Table

    table = Table(title="Available Cultural Dimensions", show_header=True,
                  header_style="bold", border_style="dim")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Category", style="yellow")
    table.add_column("Description")

    for key, dim in CULTURAL_DIMENSIONS.items():
        table.add_row(key, dim.name, dim.category, dim.description)

    console.print(table)


def _result_payload(result: GenerationResult) -> dict:
    meta = {k: v for k, v in result.metadata.items() if k != "tracer"}
    return {
        "culture": result.culture,
        "dimension": result.dimension.model_dump(),
        "params": result.params.model_dump(),
        "records": [
            {
                "example": {
                    "content": r.example.content,
                    "cultural_elements": r.example.cultural_elements,
                },
                "verification": r.verification.model_dump(),
                "refinement_iterations": r.refinement_iterations,
            }
            for r in result.records
        ],
        "metadata": meta,
    }


def _save_result(result: GenerationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_result_payload(result), fh, ensure_ascii=False, indent=2)


def _save_trace(result: GenerationResult, path: Path) -> None:
    tracer = result.metadata.get("tracer")
    if tracer is not None:
        tracer.save(path)


_PROGRESS_FILE = "progress.json"


def _load_progress(output_dir: Path) -> dict:
    p = output_dir / _PROGRESS_FILE
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_progress(output_dir: Path, progress: dict) -> None:
    progress["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(output_dir / _PROGRESS_FILE, "w", encoding="utf-8") as fh:
        json.dump(progress, fh, ensure_ascii=False, indent=2)


def _display_result(result: GenerationResult) -> None:
    n = len(result.records)
    for i, record in enumerate(result.records):
        title = (
            f"[bold green]Generated Chat Training Example {i + 1}/{n}[/bold green]"
            if n > 1
            else "[bold green]Generated Chat Training Example[/bold green]"
        )
        console.print(
            Panel(
                Syntax(
                    json.dumps(record.example.content, ensure_ascii=False, indent=2),
                    "json", theme="monokai", word_wrap=True,
                ),
                title=title,
                border_style="green",
            )
        )

        if record.example.cultural_elements:
            console.print("\n[bold]Cultural Elements Incorporated:[/bold]")
            for elem in record.example.cultural_elements:
                console.print(f"  [dim]•[/dim] {elem}")

        v = record.verification
        score_colour = "green" if v.overall_score >= 7 else ("yellow" if v.overall_score >= 5 else "red")
        console.print("\n[bold]Quality Assessment:[/bold]")
        console.print(f"  Cultural Accuracy       {v.cultural_accuracy_score:5.1f}/10")
        console.print(f"  Linguistic Authenticity {v.linguistic_authenticity_score:5.1f}/10")
        console.print(f"  Dimension Relevance     {v.dimension_relevance_score:5.1f}/10")
        console.print(f"  Training Quality        {v.training_quality_score:5.1f}/10")
        console.print(
            f"  [bold]Overall Score         [{score_colour}]{v.overall_score:5.1f}/10"
            f"[/{score_colour}][/bold]"
        )
        if record.refinement_iterations:
            console.print(f"  Refinement iterations   {record.refinement_iterations}")
        if v.issues:
            console.print("\n[bold yellow]Remaining Issues:[/bold yellow]")
            for issue in v.issues:
                console.print(f"  • {issue}")


def _run_single(args, params: GenerationParams) -> None:
    try:
        dimension = get_dimension(args.dimension)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    lines = [
        f"[bold]Culture:[/bold]        {args.culture}",
        f"[bold]Dimension:[/bold]      {dimension.name}  [dim]({dimension.category})[/dim]",
        f"[bold]Language:[/bold]       {params.language}",
        f"[bold]Format:[/bold]         Multi-turn chat (agent chooses task & turns)",
    ]
    if params.topic:
        lines.append(f"[bold]Topic hint:[/bold]     {params.topic}")
    if params.implicit_culture:
        lines.append("[bold]Mode:[/bold]           Implicit culture (user queries carry no cultural markers)")
    console.print(Panel("\n".join(lines), title="[bold blue]culture-recipe[/bold blue]",
                        border_style="blue"))

    live_trace_path = None
    if args.trace and args.output:
        live_trace_path = str(
            Path(args.output).parent / f"{Path(args.output).stem}_trace.json"
        )
        console.print(f"[dim]Live trace → {live_trace_path} (tail -f to watch)[/dim]")

    try:
        result = run_pipeline(
            args.culture, dimension, params,
            verbose=args.verbose, trace=args.trace, trace_path=live_trace_path,
        )
    except Exception as exc:
        console.print(f"\n[red]Pipeline error:[/red] {exc}")
        if args.verbose:
            import traceback as tb
            console.print(tb.format_exc())
        sys.exit(1)

    console.print()
    _display_result(result)

    if args.output:
        output_path = Path(args.output)
        _save_result(result, output_path)
        console.print(f"\n[green]✓[/green] Result saved to [cyan]{output_path}[/cyan]")

        if args.trace:
            trace_path = output_path.parent / f"{output_path.stem}_trace.json"
            _save_trace(result, trace_path)
            console.print(f"[green]✓[/green] Trace saved to [cyan]{trace_path}[/cyan]")


def _run_multi(args, params: GenerationParams, dim_keys: list[str]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = _load_progress(output_dir)
    already_completed: set[str] = set(progress.get("completed", []))
    failed: set[str] = set(progress.get("failed", []))

    if not progress:
        progress = {
            "culture": args.culture,
            "language": params.language,
            "topic": params.topic,
            "dimensions": dim_keys,
            "completed": [],
            "failed": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_progress(output_dir, progress)

    pending = [k for k in dim_keys if k not in already_completed]
    skipped = len(already_completed.intersection(dim_keys))

    console.print(
        Panel(
            "\n".join([
                f"[bold]Culture:[/bold]    {args.culture}",
                f"[bold]Language:[/bold]   {params.language}",
                f"[bold]Format:[/bold]     Multi-turn chat (agent chooses task & turns)",
                f"[bold]Dimensions:[/bold] {len(dim_keys)} total  "
                f"[dim]({skipped} already done, {len(pending)} remaining)[/dim]",
                f"[bold]Output dir:[/bold] {output_dir}",
            ]),
            title="[bold blue]culture-recipe — multi-dimension run[/bold blue]",
            border_style="blue",
        )
    )

    if not pending:
        console.print("[green]All dimensions already completed. Nothing to do.[/green]")
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=False,
    ) as progress_bar:
        task = progress_bar.add_task("Dimensions", total=len(dim_keys), completed=skipped)

        for dim_key in pending:
            try:
                dimension = get_dimension(dim_key)
            except ValueError as exc:
                console.print(f"[red]Unknown dimension '{dim_key}': {exc}[/red]")
                failed.add(dim_key)
                progress_bar.advance(task)
                _save_progress(output_dir, {**progress,
                                            "completed": list(already_completed),
                                            "failed": list(failed)})
                continue

            progress_bar.update(task, description=f"[cyan]{dim_key}[/cyan]")
            console.print(
                f"\n[bold blue]→[/bold blue] [{list(dim_keys).index(dim_key)+1}/{len(dim_keys)}] "
                f"[cyan]{dim_key}[/cyan]  [dim]{dimension.name}[/dim]"
            )

            live_trace_path = None
            if args.trace:
                traces_dir = output_dir / "traces"
                traces_dir.mkdir(parents=True, exist_ok=True)
                live_trace_path = str(traces_dir / f"{dim_key}_trace.json")

            try:
                result = run_pipeline(
                    args.culture, dimension, params,
                    verbose=args.verbose, trace=args.trace, trace_path=live_trace_path,
                )
            except Exception as exc:
                console.print(f"  [red]✗ Failed:[/red] {exc}")
                if args.verbose:
                    import traceback as tb
                    console.print(tb.format_exc())
                failed.add(dim_key)
                progress_bar.advance(task)
                _save_progress(output_dir, {**progress,
                                            "completed": list(already_completed),
                                            "failed": list(failed)})
                continue

            result_path = output_dir / f"{dim_key}.json"
            _save_result(result, result_path)

            if args.trace:
                _save_trace(result, traces_dir / f"{dim_key}_trace.json")

            already_completed.add(dim_key)
            failed.discard(dim_key)
            progress_bar.advance(task)
            _save_progress(output_dir, {**progress,
                                        "completed": list(already_completed),
                                        "failed": list(failed)})

            n_rec = len(result.records)
            avg_score = sum(r.verification.overall_score for r in result.records) / n_rec
            score_colour = ("green" if avg_score >= 7
                            else "yellow" if avg_score >= 5 else "red")
            console.print(
                f"  [green]✓[/green] Saved  "
                f"[dim]{n_rec} example{'s' if n_rec > 1 else ''}  "
                f"avg score: [{score_colour}]{avg_score:.1f}/10[/{score_colour}][/dim]"
            )

    remaining_failed = [k for k in failed if k in dim_keys]
    console.print(
        f"\n[bold]Done.[/bold]  "
        f"[green]{len(already_completed.intersection(dim_keys))} completed[/green]"
        + (f"  [red]{len(remaining_failed)} failed[/red]" if remaining_failed else "")
    )
    if remaining_failed:
        console.print("[dim]Failed dimensions:[/dim] " + ", ".join(remaining_failed))
        console.print("[dim]Re-run the same command to retry failed dimensions.[/dim]")


def _run_generate(args) -> None:
    if args.list_dimensions:
        _print_dimensions()
        return

    if not args.culture:
        console.print("[red]Error:[/red] --culture is required.")
        sys.exit(1)

    if args.all_dimensions:
        dim_keys = list(CULTURAL_DIMENSIONS.keys())
    elif args.dimensions:
        dim_keys = [k.strip() for k in args.dimensions.split(",") if k.strip()]
    elif args.dimension:
        dim_keys = [args.dimension]
    else:
        console.print(
            "[red]Error:[/red] Specify --dimension, --dimensions, or --all-dimensions.\n"
            "Run with --list-dimensions to see available dimension keys."
        )
        sys.exit(1)

    is_multi = len(dim_keys) > 1 or args.all_dimensions or args.dimensions
    if is_multi and not args.output_dir:
        console.print("[red]Error:[/red] Multi-dimension runs require --output-dir DIR.")
        sys.exit(1)
    if not is_multi and args.output_dir:
        console.print(
            "[red]Error:[/red] --output-dir is for multi-dimension runs. "
            "Use --output FILE for a single dimension."
        )
        sys.exit(1)

    params = GenerationParams(
        language=args.language,
        topic=args.topic,
        implicit_culture=args.implicit_culture,
    )

    if is_multi:
        _run_multi(args, params, dim_keys)
    else:
        _run_single(args, params)


def _run_finetune(args) -> None:
    from src.finetune import prepare_data, train, merge_lora

    if args.finetune_cmd == "prepare-data":
        prepare_data.run(args)
    elif args.finetune_cmd == "train":
        train.run(args)
    elif args.finetune_cmd == "merge-lora":
        merge_lora.run(args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="culture-recipe",
        description="Generate culturally-aligned LLM training data and fine-tune models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Commands:
  generate      Run the agentic data generation pipeline
  finetune      Fine-tune a model on generated data

Run `python main.py <command> --help` for command-specific options.
""",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    _build_generate_parser(subparsers)
    _build_finetune_parser(subparsers)

    args = parser.parse_args()

    if args.command == "generate":
        _run_generate(args)
    elif args.command == "finetune":
        _run_finetune(args)


if __name__ == "__main__":
    main()
