"""
Autonomous cultural alignment training example generator.

Claude decides the workflow — it can research multiple times, generate,
verify, and refine, producing one or more examples per dimension before
calling finish() when satisfied with the coverage.

Tools exposed to the orchestrating agent:
  research_culture          — web-search-backed cultural context gathering
  generate_training_example — LLM-based example generation / refinement
  verify_training_example   — structured quality scoring
  commit_example            — archive current verified example, prepare for next
  finish                    — submit all generated examples and end the session
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
    ExampleRecord,
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
                "Generate a culturally-grounded multi-turn chat training example for cultural alignment "
                "using all research gathered so far. "
                "You choose the scenario and task — it can be anything (advice, planning, problem-solving, "
                "creative writing, navigating a social situation, etc.) as long as it arises naturally from "
                "the target culture and illustrates the specified cultural dimension. "
                "You also decide the number of conversation turns (typically 3–8). "
                "The purpose is to produce high-quality cultural alignment training data that teaches "
                "an LLM to respond authentically within this cultural context. "
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
                            "Leave empty for the initial generation or when starting a fresh example."
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
            "name": "commit_example",
            "description": (
                "Archive the current verified training example and prepare to generate another one. "
                "Call this after verify_training_example when you want to produce an additional example "
                "covering a different sub-aspect of the dimension (e.g., a different religious holiday, "
                "a different social context, a different scenario type). "
                "The current example is saved; you can then call generate_training_example again "
                "to create a fresh example for the next sub-aspect. "
                "Only call this if you genuinely need another example — do not call it and then "
                "immediately call finish without generating anything new."
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
                f"Submit all generated cultural alignment training examples and end the session. "
                f"Call this when you are satisfied with the examples produced. "
                f"The current verified example (if any) will be automatically included. "
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
    topic_line = f"\n  Topic hint:   {params.topic}" if params.topic else ""
    return f"""\
You are an expert in cross-cultural communication and LLM training data curation.

Your task is to produce **high-quality cultural alignment training examples** — \
multi-turn chat training samples that teach language models to respond appropriately, \
authentically, and sensitively within a specific cultural context.

Target specification:
  Culture:      {culture}
  Dimension:    {dimension.name}
  Category:     {dimension.category}
  Description:  {dimension.description}
  Language:     {params.language}
  Format:       Multi-turn chat (user ↔ AI assistant){topic_line}

You have five tools:
  • research_culture           — gather cultural knowledge (call multiple times for different angles)
  • generate_training_example  — create a cultural alignment training example from your research
  • verify_training_example    — evaluate it for cultural accuracy and training quality
  • commit_example             — archive the current verified example and prepare to generate another
  • finish                     — submit all generated examples

Work autonomously. You decide how many times to research, generate, verify, and how many examples to produce.

**Number of examples per dimension:**
Generate one example by default. Generate multiple examples when the dimension has distinct \
sub-aspects that each deserve their own scenario — for example:
  - A "religious observance" dimension might warrant separate examples for different major \
holidays or rituals.
  - A "workplace hierarchy" dimension might benefit from both a junior–senior and a \
cross-department scenario.
  - A "gift-giving etiquette" dimension might have business and personal contexts that differ \
enough to each deserve an example.
Do not generate multiple examples for trivial variations. Only add another example when it \
would genuinely teach something distinct about the culture.

To generate multiple examples: after verifying an approved example, call commit_example to \
archive it, then generate_training_example again for the next sub-aspect. Call finish when \
all examples are ready.

When generating, **you choose the task and scenario** for the chat. Pick any realistic \
situation in which a user turns to an AI assistant for help — advice, planning, creative \
writing, navigating a social or professional situation, language help, etc. The task \
must arise naturally from {culture} cultural life and let the {dimension.name} dimension \
surface organically. You also decide the number of turns (typically 3–8 exchanges) and \
the length of each assistant response — brief when brevity is appropriate, longer when \
explanation, nuance, or cultural context is genuinely needed.

A great cultural alignment training example must be:
  1. Culturally accurate — grounded in real norms, not stereotypes or caricatures
  2. Linguistically authentic — natural phrasing, idioms, and register for the culture
  3. Dimension-focused — the target dimension surfaces clearly but organically
  4. Appropriately sized — each assistant response is as long as the moment requires,
     never padded, never truncated
  5. High training value — would genuinely teach an LLM about this culture

Stop and call finish() when all examples meet the quality threshold (overall score \
≥ {config.quality_threshold}/10), or after making your best attempt.\
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

    The orchestrating Claude decides when to research, generate, verify, refine,
    and whether to produce multiple examples — calling finish() when done.
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
    generation_count: int = 0           # total generate calls across all examples
    current_example_gen_count: int = 0  # generate calls for the current example
    completed_records: list[ExampleRecord] = []
    finished: bool = False

    # ------------------------------------------------------------------
    # Tool executor
    # ------------------------------------------------------------------
    def execute_tool(tool_name: str, tool_input: dict) -> Any:
        nonlocal current_example, current_verification, generation_count, current_example_gen_count, finished

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
            current_example_gen_count += 1
            example_num = len(completed_records) + 1
            if current_example_gen_count == 1:
                action = f"Generating example {example_num}"
            else:
                action = f"Refining example {example_num} (attempt {current_example_gen_count})"
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
                        "generation_count": generation_count,
                        "example_num": example_num,
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
                # Fresh generation (first attempt or new example after commit)
                example = generate_example(
                    culture, dimension, params, combined_research,
                    verbose=verbose, tracer=tracer,
                )

            current_example = example
            current_verification = None  # invalidate previous verification
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

        # ---- commit_example -------------------------------------------
        elif tool_name == "commit_example":
            if current_verification is None:
                return {
                    "error": (
                        "No verified example to commit. "
                        "Call generate_training_example and verify_training_example first."
                    )
                }
            record = ExampleRecord(
                example=current_example,
                verification=current_verification,
                refinement_iterations=max(0, current_example_gen_count - 1),
            )
            completed_records.append(record)
            record_num = len(completed_records)
            score = current_verification.overall_score
            score_colour = "green" if score >= 7 else ("yellow" if score >= 5 else "red")
            # Reset state for next example
            current_example = None
            current_verification = None
            current_example_gen_count = 0
            console.print(
                f"\n  [bold blue]→ commit_example[/bold blue]  "
                f"Example {record_num} archived "
                f"([{score_colour}]{score:.1f}/10[/{score_colour}])"
            )
            return {
                "status": "committed",
                "examples_committed": record_num,
                "message": (
                    "Example archived. You may now call generate_training_example "
                    "to create another example, or call finish to submit all examples."
                ),
            }

        # ---- finish ---------------------------------------------------
        elif tool_name == "finish":
            if not completed_records and current_verification is None:
                return {
                    "error": (
                        "You must generate and verify at least one example before calling finish. "
                        "Call generate_training_example then verify_training_example first."
                    )
                }
            # Auto-commit any pending verified example
            if current_example is not None and current_verification is not None:
                completed_records.append(
                    ExampleRecord(
                        example=current_example,
                        verification=current_verification,
                        refinement_iterations=max(0, current_example_gen_count - 1),
                    )
                )
                current_example = None
                current_verification = None
                current_example_gen_count = 0
            finished = True
            n = len(completed_records)
            console.print(
                f"\n  [bold green]→ finish[/bold green]  "
                f"Submitting {n} example{'s' if n > 1 else ''}."
            )
            return {"status": "completed", "examples_submitted": n}

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------
    system_prompt = _build_system_prompt(culture, dimension, params)
    user_message = (
        f"Please produce cultural alignment training example(s) for "
        f"**{culture}** culture, dimension: **{dimension.name}**."
    )
    messages: list[dict] = [{"role": "user", "content": user_message}]

    # Safety cap: generous but finite
    max_iterations = config.max_refinement_iterations * 6 + 10

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
                "examples_produced": len(completed_records),
                "finished_cleanly": finished,
            }
        )

    # ------------------------------------------------------------------
    # Guard: ensure we have a usable result
    # ------------------------------------------------------------------
    if not completed_records and current_example is None:
        raise RuntimeError("Agent did not produce any training example.")

    # Commit any pending example (verified or not)
    if current_example is not None:
        if current_verification is None:
            combined_research = "\n\n---\n\n".join(accumulated_research) or ""
            current_verification = verify_example(
                culture, dimension, params, current_example, combined_research,
                verbose=verbose,
            )
        completed_records.append(
            ExampleRecord(
                example=current_example,
                verification=current_verification,
                refinement_iterations=max(0, current_example_gen_count - 1),
            )
        )

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    metadata: dict = {
        "model": config.model,
        "quality_threshold": config.quality_threshold,
        "agentic_iterations": iteration + 1,
        "research_rounds": len(accumulated_research),
        "examples_produced": len(completed_records),
    }
    if tracer:
        metadata["tracer"] = tracer

    return GenerationResult(
        culture=culture,
        dimension=dimension,
        params=params,
        research_summary="\n\n---\n\n".join(accumulated_research),
        records=completed_records,
        metadata=metadata,
    )
