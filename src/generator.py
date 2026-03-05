"""
Phase 2 — Generation
Phase 4 — Refinement

Both phases ask Claude to generate a training example (or an improved one)
and return two JSON code blocks in its text response:
  Block 1: the training example in the requested format
  Block 2: a JSON array of cultural elements incorporated

Adaptive thinking is enabled on all calls for maximum quality.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import anthropic

from config import config
from src.models import CultureDimension, GeneratedExample, GenerationParams
from src.prompts import (
    GENERATION_SYSTEM_PROMPT,
    get_generation_prompt,
    get_refinement_prompt,
)

if TYPE_CHECKING:
    from src.tracer import PipelineTracer


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json_blocks(text: str) -> list[dict | list]:
    """
    Extract all ```json ... ``` code blocks from a response string and parse them.
    Returns a list of parsed objects (dict or list).
    """
    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    blocks = []
    for raw in pattern.findall(text):
        try:
            blocks.append(json.loads(raw))
        except json.JSONDecodeError:
            # Skip malformed blocks
            pass
    return blocks


def _fallback_parse(text: str) -> dict:
    """
    Last-resort parser: find the first {...} object in the text.
    Used when no ```json block is present.
    """
    # Greedy match of a top-level JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"raw": text}


def _parse_generation_response(
    text: str,
) -> tuple[dict, list[str]]:
    """
    Parse the model's text response to extract:
      - content  (dict)  — the formatted training example
      - elements (list)  — the cultural elements
    """
    blocks = _extract_json_blocks(text)

    content: dict = {}
    elements: list[str] = []

    if len(blocks) >= 1 and isinstance(blocks[0], dict):
        content = blocks[0]
    if len(blocks) >= 2 and isinstance(blocks[1], list):
        elements = [str(e) for e in blocks[1]]

    # Fallbacks
    if not content:
        content = _fallback_parse(text)
    if not elements:
        # Try to find a plain bullet list of elements
        bullet_lines = re.findall(r"^[-*•]\s+(.+)", text, re.MULTILINE)
        if bullet_lines:
            elements = bullet_lines

    return content, elements


# ---------------------------------------------------------------------------
# Shared streaming call
# ---------------------------------------------------------------------------

def _stream_generation(
    user_prompt: str,
    verbose: bool = False,
    tracer: "PipelineTracer | None" = None,
) -> str:
    """
    Run a streaming generation call and return the full text response.
    """
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    with client.messages.stream(
        model=config.model,
        max_tokens=config.generation_max_tokens,
        system=GENERATION_SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        if verbose:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta and delta.type == "text_delta":
                        print(delta.text, end="", flush=True)
            print()

        response = stream.get_final_message()

    if tracer is not None:
        tracer.increment_api_calls()
        tracer.add_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        from src.tracer import extract_trace_data
        extract_trace_data(response.content, tracer)

    text_parts = [b.text for b in response.content if b.type == "text"]
    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_example(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    research_context: str,
    verbose: bool = False,
    tracer: "PipelineTracer | None" = None,
) -> GeneratedExample:
    """
    Phase 2: Generate a training example based on research context.
    """
    user_prompt = get_generation_prompt(culture, dimension, params, research_context)
    text = _stream_generation(user_prompt, verbose=verbose, tracer=tracer)
    content, elements = _parse_generation_response(text)

    return GeneratedExample(
        content=content,
        cultural_elements=elements,
    )


def refine_example(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    example: GeneratedExample,
    verification_dict: dict,
    research_context: str,
    verbose: bool = False,
    tracer: "PipelineTracer | None" = None,
) -> GeneratedExample:
    """
    Phase 4: Refine the example based on verification feedback.
    """
    user_prompt = get_refinement_prompt(
        culture, dimension, params, example.content, verification_dict, research_context
    )
    text = _stream_generation(user_prompt, verbose=verbose, tracer=tracer)
    content, elements = _parse_generation_response(text)

    return GeneratedExample(
        content=content,
        cultural_elements=elements,
    )
