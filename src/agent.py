"""
Autonomous cultural alignment training example generator.

Claude decides the workflow — it can research multiple times, generate,
verify, and refine, calling finish() when satisfied with the quality.

Tools exposed to the orchestrating agent:
  research_culture          — web-search-backed cultural context gathering
  generate_training_example — LLM-based example generation / refinement
  verify_training_example   — structured quality scoring
  finish                    — submit the final approved example
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
from rich.console import Console

from config import config
from src.generator import generate_example, refine_example
from src.models import (
    CultureDimension,
    GeneratedExample,
    GenerationParams,
    GenerationResult,
)
from src.researcher import research_cultural_context
from src.tracer import PipelineTracer
from src.verifier import verify_example

console = Console()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

def _get_tools() -> list[dict]:
    return [
        {
            "name": "research_culture",
            "description": (
                "Search the web for cultural context about the target culture and dimension. "
                "Call this at least once before generating, and again for more specific angles. "
                "All results are accumulated and used by subsequent tool calls."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "focus_query": {
                        "type": "string",
                        "description": (
                            "The specific cultural aspect to research, "
                            "e.g. 'Japanese gift-giving norms in business settings' "
                            "or 'power distance in Southeast Asian workplaces'."
                        ),
                    }
                },
                "required": ["focus_query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "generate_training_example",
            "description": (
                "Generate a cultural alignment training example using all research gathered so far. "
                "For the first attempt leave feedback empty. "
                "For subsequent attempts pass the issues and suggestions from the last verification "
                "so the example can be improved."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "feedback": {
                        "type": "string",
                        "description": (
                            "Issues and improvement suggestions from the last verification. "
                            "Leave empty for the initial generation."
                        ),
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "verify_training_example",
            "description": (
                "Evaluate the most recently generated cultural alignment training example. "
                "Returns cultural accuracy, linguistic authenticity, dimension relevance, "
                "and training quality scores (each 0–10), plus specific issues and suggestions. "
                "Always call this after generate_training_example."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "finish",
            "description": (
                f"Submit the final cultural alignment training example and end the session. "
                f"Call this once the example is approved (overall score ≥ {config.quality_threshold}/10) "
                f"or after exhausting all reasonable improvement attempts. "
                f"You must have called verify_training_example at least once before calling finish."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    ]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
) -> str:
    topic_line = f"\n  Topic:        {params.topic}" if params.topic else ""
    turns_line = (
        f"\n  Turns:        {params.num_turns}"
        if params.example_type.value == "conversation"
        else ""
    )
    return f"""\
You are an expert in cross-cultural communication and LLM training data curation.

Your task is to produce a **high-quality cultural alignment training example** — a \
training data sample that teaches language models to respond appropriately, \
authentically, and sensitively within a specific cultural context.

Target specification:
  Culture:      {culture}
  Dimension:    {dimension.name}
  Category:     {dimension.category}
  Description:  {dimension.description}
  Example type: {params.example_type.value}
  Format:       {params.output_format.value}
  Language:     {params.language}
  Length:       {params.length.value}{topic_line}{turns_line}

You have four tools:
  • research_culture           — gather cultural knowledge (call multiple times for different angles)
  • generate_training_example  — create the cultural alignment training example from your research
  • verify_training_example    — evaluate it for cultural accuracy and training quality
  • finish                     — submit your final approved example

Work autonomously. You decide how many times to research, generate, and verify.

A great cultural alignment training example must be:
  1. Culturally accurate — grounded in real norms, not stereotypes or caricatures
  2. Linguistically authentic — natural phrasing, idioms, and register for the culture
  3. Dimension-focused — the target dimension surfaces clearly but organically
  4. High training value — would genuinely teach an LLM about this culture

Stop and call finish() when overall quality score ≥ {config.quality_threshold}/10, \
or after making your best attempt.\
"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    verbose: bool = False,
    trace: bool = False,
) -> GenerationResult:
    """
    Run the autonomous cultural alignment training example generation agent.

    The orchestrating Claude decides when to research, generate, verify, and
    refine — calling finish() when it is satisfied with the result.
    """
    tracer: PipelineTracer | None = None
    if trace:
        tracer = PipelineTracer(
            culture=culture,
            dimension_key=dimension.name,
            params=params.model_dump(),
        )

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # ------------------------------------------------------------------
    # Mutable state shared with the tool executor (via closures)
    # ------------------------------------------------------------------
    accumulated_research: list[str] = []
    current_example: GeneratedExample | None = None
    current_verification = None
    generation_count: int = 0
    finished: bool = False

    # ------------------------------------------------------------------
    # Tool executor
    # ------------------------------------------------------------------
    def execute_tool(tool_name: str, tool_input: dict) -> Any:
        nonlocal current_example, current_verification, generation_count, finished

        # ---- research_culture ----------------------------------------
        if tool_name == "research_culture":
            focus_query = tool_input.get("focus_query", "")
            console.print(
                f"\n  [bold blue]→ research_culture[/bold blue]  "
                f"[dim]{focus_query}[/dim]"
            )
            if tracer:
                tracer.start_phase(
                    f"research_{len(accumulated_research) + 1}",
                    {
                        "culture": culture,
                        "dimension": dimension.name,
                        "focus_query": focus_query,
                    },
                )
            result = research_cultural_context(
                culture, dimension, params,
                verbose=verbose, tracer=tracer,
                focus_query=focus_query,
            )
            accumulated_research.append(result)
            if tracer:
                tracer.end_phase(
                    output=result[:500] + "…" if len(result) > 500 else result
                )
            console.print(
                f"  [green]✓[/green] Research done "
                f"([dim]{len(result):,} chars — "
                f"total context: {sum(len(r) for r in accumulated_research):,} chars[/dim])"
            )
            return result

        # ---- generate_training_example --------------------------------
        elif tool_name == "generate_training_example":
            feedback = tool_input.get("feedback", "")
            generation_count += 1
            action = (
                "Generating" if generation_count == 1
                else f"Regenerating (attempt {generation_count})"
            )
            console.print(
                f"\n  [bold blue]→ generate_training_example[/bold blue]  "
                f"[dim]{action}[/dim]"
            )
            combined_research = (
                "\n\n---\n\n".join(accumulated_research)
                or "(no research context yet)"
            )
            if tracer:
                tracer.start_phase(
                    f"generate_{generation_count}",
                    {
                        "example_type": params.example_type.value,
                        "generation_count": generation_count,
                        "has_feedback": bool(feedback),
                    },
                )

            if current_example is not None and current_verification is not None and feedback:
                # Refinement path — pass the previous verification scores/issues
                example = refine_example(
                    culture, dimension, params,
                    current_example,
                    current_verification.model_dump(),
                    combined_research,
                    verbose=verbose, tracer=tracer,
                )
            else:
                # Fresh generation
                example = generate_example(
                    culture, dimension, params, combined_research,
                    verbose=verbose, tracer=tracer,
                )

            current_example = example
            if tracer:
                tracer.end_phase(
                    output={
                        "content": example.content,
                        "cultural_elements": example.cultural_elements,
                    }
                )
            console.print(
                f"  [green]✓[/green] Example generated "
                f"([dim]{len(example.cultural_elements)} cultural elements[/dim])"
            )
            return {
                "content": example.content,
                "cultural_elements": example.cultural_elements,
            }

        # ---- verify_training_example ----------------------------------
        elif tool_name == "verify_training_example":
            if current_example is None:
                return {
                    "error": (
                        "No example has been generated yet. "
                        "Call generate_training_example first."
                    )
                }
            console.print("\n  [bold blue]→ verify_training_example[/bold blue]")
            combined_research = (
                "\n\n---\n\n".join(accumulated_research)
                or "(no research context)"
            )
            if tracer:
                tracer.start_phase(
                    f"verify_{generation_count}",
                    {"generation_count": generation_count},
                )
            verification = verify_example(
                culture, dimension, params, current_example, combined_research,
                verbose=verbose, tracer=tracer,
            )
            current_verification = verification
            if tracer:
                tracer.end_phase(output=verification.model_dump())
            score = verification.overall_score
            score_colour = "green" if score >= 7 else ("yellow" if score >= 5 else "red")
            status = (
                "[green]Approved[/green]"
                if verification.is_approved
                else "[yellow]Needs improvement[/yellow]"
            )
            console.print(
                f"  [green]✓[/green] Score: "
                f"[{score_colour}]{score:.1f}/10[/{score_colour}] — {status}"
            )
            return verification.model_dump()

        # ---- finish ---------------------------------------------------
        elif tool_name == "finish":
            if current_verification is None:
                return {
                    "error": (
                        "You must call verify_training_example before calling finish."
                    )
                }
            finished = True
            console.print("\n  [bold green]→ finish[/bold green]  Submitting final example.")
            return {"status": "completed"}

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------
    system_prompt = _build_system_prompt(culture, dimension, params)
    user_message = (
        f"Please produce a cultural alignment training example for "
        f"**{culture}** culture, dimension: **{dimension.name}**."
    )
    messages: list[dict] = [{"role": "user", "content": user_message}]

    # Safety cap: generous but finite
    max_iterations = config.max_refinement_iterations * 4 + 8

    console.print(
        f"\n[bold blue]Agent[/bold blue]  "
        f"[cyan]{culture}[/cyan] — [yellow]{dimension.name}[/yellow]  "
        f"[dim](up to {max_iterations} orchestrator turns)[/dim]"
    )

    if tracer:
        tracer.start_phase(
            "orchestrator",
            {"culture": culture, "dimension": dimension.name},
        )

    iteration = 0
    for iteration in range(max_iterations):
        response = client.messages.create(
            model=config.model,
            max_tokens=config.generation_max_tokens,
            system=system_prompt,
            thinking={"type": "adaptive"},
            tools=_get_tools(),
            messages=messages,
        )

        if tracer:
            tracer.increment_api_calls()
            tracer.add_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            from src.tracer import extract_trace_data
            extract_trace_data(response.content, tracer)

        if verbose:
            for block in response.content:
                if getattr(block, "type", None) == "text" and block.text.strip():
                    console.print(f"\n  [dim italic]{block.text.strip()}[/dim italic]")

        tool_use_blocks = [
            b for b in response.content if getattr(b, "type", None) == "tool_use"
        ]
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            break

        # Execute all tool calls in this turn
        tool_results = []
        for block in tool_use_blocks:
            result = execute_tool(
                block.name,
                json.loads(json.dumps(block.input, default=str)),
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": (
                        json.dumps(result, ensure_ascii=False, default=str)
                        if isinstance(result, (dict, list))
                        else str(result)
                    ),
                }
            )

        messages.append({"role": "user", "content": tool_results})

        if finished:
            break

    if tracer:
        tracer.end_phase(
            output={
                "iterations": iteration + 1,
                "generation_count": generation_count,
                "finished_cleanly": finished,
            }
        )

    # ------------------------------------------------------------------
    # Guard: ensure we have a usable result
    # ------------------------------------------------------------------
    if current_example is None:
        raise RuntimeError("Agent did not produce any training example.")
    if current_verification is None:
        # Verify one final time if the agent somehow skipped it
        combined_research = "\n\n---\n\n".join(accumulated_research) or ""
        current_verification = verify_example(
            culture, dimension, params, current_example, combined_research,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    metadata: dict = {
        "model": config.model,
        "quality_threshold": config.quality_threshold,
        "agentic_iterations": iteration + 1,
        "research_rounds": len(accumulated_research),
    }
    if tracer:
        metadata["tracer"] = tracer

    return GenerationResult(
        culture=culture,
        dimension=dimension,
        params=params,
        research_summary="\n\n---\n\n".join(accumulated_research),
        example=current_example,
        verification=current_verification,
        refinement_iterations=max(0, generation_count - 1),
        metadata=metadata,
    )
