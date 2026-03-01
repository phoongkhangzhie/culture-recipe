"""
Main pipeline orchestrator.

Runs the four phases in sequence:
  1. Research  — gather cultural context via web search
  2. Generate  — produce a training example
  3. Verify    — score quality and cultural grounding
  4. Refine    — improve if score < threshold (loop up to max_refinement_iterations)
"""

from rich.console import Console

from config import config
from src.generator import generate_example, refine_example
from src.models import CultureDimension, GenerationParams, GenerationResult
from src.researcher import research_cultural_context
from src.verifier import verify_example

console = Console()


def run_pipeline(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    verbose: bool = False,
) -> GenerationResult:
    """
    Execute the full cultural training example generation pipeline.

    Returns a GenerationResult containing the final example, verification
    scores, research summary, and metadata.
    """

    # ------------------------------------------------------------------
    # Phase 1: Research
    # ------------------------------------------------------------------
    console.print(
        f"\n[bold blue]Phase 1[/bold blue]  Researching "
        f"[cyan]{culture}[/cyan] culture — "
        f"[yellow]{dimension.name}[/yellow]…"
    )

    research_context = research_cultural_context(
        culture, dimension, params, verbose=verbose
    )

    console.print(
        f"  [green]✓[/green] Research complete "
        f"([dim]{len(research_context):,} chars[/dim])"
    )

    # ------------------------------------------------------------------
    # Phase 2: Generate
    # ------------------------------------------------------------------
    console.print(
        "\n[bold blue]Phase 2[/bold blue]  Generating "
        f"[yellow]{params.example_type.value}[/yellow] training example…"
    )

    example = generate_example(
        culture, dimension, params, research_context, verbose=verbose
    )

    console.print(
        f"  [green]✓[/green] Example generated "
        f"([dim]{len(example.cultural_elements)} cultural elements[/dim])"
    )

    # ------------------------------------------------------------------
    # Phases 3 + 4: Verify → Refine loop
    # ------------------------------------------------------------------
    refinement_iterations = 0
    verification = None

    for attempt in range(config.max_refinement_iterations + 1):
        phase_label = "Phase 3" if attempt == 0 else "Phase 4"
        action = "Verifying" if attempt == 0 else "Re-verifying"

        console.print(
            f"\n[bold blue]{phase_label}[/bold blue]  "
            f"{action} example (attempt {attempt + 1})…"
        )

        verification = verify_example(
            culture, dimension, params, example, research_context, verbose=verbose
        )

        score = verification.overall_score
        score_colour = "green" if score >= 7 else ("yellow" if score >= 5 else "red")
        status = "[green]Approved[/green]" if verification.is_approved else "[yellow]Needs improvement[/yellow]"

        console.print(
            f"  [green]✓[/green] Score: "
            f"[{score_colour}]{score:.1f}/10[/{score_colour}] — {status}"
        )

        if verification.is_approved or attempt >= config.max_refinement_iterations:
            break

        # Refine
        refinement_iterations += 1
        console.print(
            f"\n[bold blue]Phase 4[/bold blue]  "
            f"Refining example (iteration {refinement_iterations})…"
        )

        if verbose and verification.issues:
            console.print("  Issues:")
            for issue in verification.issues:
                console.print(f"    • {issue}")

        example = refine_example(
            culture,
            dimension,
            params,
            example,
            verification.model_dump(),
            research_context,
            verbose=verbose,
        )

        console.print(
            f"  [green]✓[/green] Refinement {refinement_iterations} complete"
        )

    return GenerationResult(
        culture=culture,
        dimension=dimension,
        params=params,
        research_summary=research_context,
        example=example,
        verification=verification,
        refinement_iterations=refinement_iterations,
        metadata={
            "model": config.model,
            "quality_threshold": config.quality_threshold,
        },
    )
