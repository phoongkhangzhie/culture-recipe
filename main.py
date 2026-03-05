#!/usr/bin/env python3
"""
culture-recipe — CLI entry point.

Generates culturally-aligned LLM training examples via a multi-phase
agentic pipeline (research → generate → verify → refine).
"""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.agent import run_pipeline
from src.models import ExampleLength, ExampleType, GenerationParams, OutputFormat
from src.taxonomy import CULTURAL_DIMENSIONS, get_dimension

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing (stdlib only — no extra deps)
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="culture-recipe",
        description="Generate culturally-aligned LLM training examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --culture Japanese --dimension power_distance
  python main.py --culture Nigerian --dimension hospitality \\
      --example-type conversation --output-format sharegpt --language English
  python main.py --culture Brazilian --dimension individualism \\
      --language Portuguese --length long --output output.json
  python main.py --list-dimensions
""",
    )

    parser.add_argument(
        "--culture",
        metavar="NAME",
        help="Target culture, e.g. 'Japanese', 'Nigerian', 'Brazilian'",
    )
    parser.add_argument(
        "--dimension",
        metavar="KEY",
        help="Cultural dimension key (see --list-dimensions for options)",
    )
    parser.add_argument(
        "--language",
        default="English",
        metavar="LANG",
        help="Language for the generated example (default: English)",
    )
    parser.add_argument(
        "--example-type",
        default="conversation",
        choices=[e.value for e in ExampleType],
        metavar="TYPE",
        help=(
            f"Type of training example "
            f"({', '.join(e.value for e in ExampleType)}) "
            f"[default: conversation]"
        ),
    )
    parser.add_argument(
        "--output-format",
        default="openai",
        choices=[f.value for f in OutputFormat],
        metavar="FMT",
        help=(
            f"Output format "
            f"({', '.join(f.value for f in OutputFormat)}) "
            f"[default: openai]"
        ),
    )
    parser.add_argument(
        "--length",
        default="medium",
        choices=[l.value for l in ExampleLength],
        metavar="LEN",
        help="Length of the example (short | medium | long) [default: medium]",
    )
    parser.add_argument(
        "--topic",
        metavar="TEXT",
        help="Optional specific topic within the dimension",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=2,
        metavar="N",
        help="Number of dialogue turns for conversation examples [default: 2]",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Save the result to a JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress including model output",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Record a full pipeline trace (inputs, tool calls, thinking, usage) to a JSON file",
    )
    parser.add_argument(
        "--trace-output",
        metavar="FILE",
        help="Path for the trace JSON file (default: <output-stem>_trace.json or trace_<run-id>.json)",
    )
    parser.add_argument(
        "--list-dimensions",
        action="store_true",
        help="Print all available cultural dimensions and exit",
    )

    return parser


# ---------------------------------------------------------------------------
# Dimension listing
# ---------------------------------------------------------------------------

def _print_dimensions() -> None:
    table = Table(
        title="Available Cultural Dimensions",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold white")
    table.add_column("Category", style="yellow")
    table.add_column("Description")

    for key, dim in CULTURAL_DIMENSIONS.items():
        table.add_row(key, dim.name, dim.category, dim.description)

    console.print(table)


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def _display_result(result, params: GenerationParams) -> None:
    # Training example
    console.print(
        Panel(
            Syntax(
                json.dumps(result.example.content, ensure_ascii=False, indent=2),
                "json",
                theme="monokai",
                word_wrap=True,
            ),
            title=(
                f"[bold green]Generated {params.example_type.value.title()} Example"
                f" [{params.output_format.value}][/bold green]"
            ),
            border_style="green",
        )
    )

    # Cultural elements
    if result.example.cultural_elements:
        console.print("\n[bold]Cultural Elements Incorporated:[/bold]")
        for elem in result.example.cultural_elements:
            console.print(f"  [dim]•[/dim] {elem}")

    # Verification scores
    v = result.verification
    score_colour = (
        "green" if v.overall_score >= 7
        else "yellow" if v.overall_score >= 5
        else "red"
    )

    console.print("\n[bold]Quality Assessment:[/bold]")
    console.print(f"  Cultural Accuracy       {v.cultural_accuracy_score:5.1f}/10")
    console.print(f"  Linguistic Authenticity {v.linguistic_authenticity_score:5.1f}/10")
    console.print(f"  Dimension Relevance     {v.dimension_relevance_score:5.1f}/10")
    console.print(f"  Training Quality        {v.training_quality_score:5.1f}/10")
    console.print(
        f"  [bold]Overall Score         [{score_colour}]{v.overall_score:5.1f}/10"
        f"[/{score_colour}][/bold]"
    )

    if result.refinement_iterations:
        console.print(f"  Refinement iterations   {result.refinement_iterations}")

    if v.issues:
        console.print("\n[bold yellow]Remaining Issues:[/bold yellow]")
        for issue in v.issues:
            console.print(f"  • {issue}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_dimensions:
        _print_dimensions()
        return

    # Validate required args
    if not args.culture or not args.dimension:
        console.print(
            "[red]Error:[/red] --culture and --dimension are required.\n"
            "Run with --list-dimensions to see available dimension keys."
        )
        sys.exit(1)

    # Load dimension
    try:
        dimension = get_dimension(args.dimension)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # Build params
    params = GenerationParams(
        language=args.language,
        example_type=ExampleType(args.example_type),
        output_format=OutputFormat(args.output_format),
        length=ExampleLength(args.length),
        topic=args.topic,
        num_turns=args.num_turns,
    )

    # Header panel
    lines = [
        f"[bold]Culture:[/bold]        {args.culture}",
        f"[bold]Dimension:[/bold]      {dimension.name}  [dim]({dimension.category})[/dim]",
        f"[bold]Language:[/bold]       {params.language}",
        f"[bold]Example type:[/bold]   {params.example_type.value}",
        f"[bold]Output format:[/bold]  {params.output_format.value}",
        f"[bold]Length:[/bold]         {params.length.value}",
    ]
    if params.topic:
        lines.append(f"[bold]Topic:[/bold]          {params.topic}")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold blue]culture-recipe[/bold blue]",
            border_style="blue",
        )
    )

    # Run pipeline
    try:
        result = run_pipeline(
            culture=args.culture,
            dimension=dimension,
            params=params,
            verbose=args.verbose,
            trace=args.trace,
        )
    except Exception as exc:
        console.print(f"\n[red]Pipeline error:[/red] {exc}")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

    # Display
    console.print()
    _display_result(result, params)

    # Save result
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Exclude the tracer object from the JSON payload
        meta = {k: v for k, v in result.metadata.items() if k != "tracer"}
        payload = {
            "culture": result.culture,
            "dimension": result.dimension.model_dump(),
            "params": result.params.model_dump(),
            "example": {
                "example_type": result.example.example_type.value,
                "output_format": result.example.output_format.value,
                "content": result.example.content,
                "cultural_elements": result.example.cultural_elements,
            },
            "verification": result.verification.model_dump(),
            "metadata": {
                **meta,
                "refinement_iterations": result.refinement_iterations,
            },
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        console.print(
            f"\n[green]✓[/green] Result saved to [cyan]{output_path}[/cyan]"
        )

    # Save trace
    if args.trace:
        tracer = result.metadata.get("tracer")
        if tracer is not None:
            if args.trace_output:
                trace_path = Path(args.trace_output)
            elif args.output:
                stem = Path(args.output).stem
                trace_path = Path(args.output).parent / f"{stem}_trace.json"
            else:
                trace_path = Path(f"trace_{tracer.run_id[:8]}.json")

            tracer.save(trace_path)
            console.print(
                f"[green]✓[/green] Trace saved to [cyan]{trace_path}[/cyan]"
            )


if __name__ == "__main__":
    main()
