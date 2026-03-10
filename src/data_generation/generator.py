"""
Phase 2 — Generation
Phase 4 — Refinement

Both phases ask the model to generate a training example (or an improved one)
and return two JSON code blocks in its text response:
  Block 1: the training example in the requested format
  Block 2: a JSON array of cultural elements incorporated
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import openai

from config import config
from .models import CultureDimension, GeneratedExample, GenerationParams
from .prompts import (
    GENERATION_SYSTEM_PROMPT,
    get_generation_prompt,
    get_refinement_prompt,
)

if TYPE_CHECKING:
    from .tracer import PipelineTracer


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
            pass
    return blocks


def _fallback_parse(text: str) -> dict:
    """
    Last-resort parser: find the first {...} object in the text.
    Used when no ```json block is present.
    """
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
        bullet_lines = re.findall(r"^[-*•]\s+(.+)", text, re.MULTILINE)
        if bullet_lines:
            elements = bullet_lines

    return content, elements


# ---------------------------------------------------------------------------
# Shared generation call
# ---------------------------------------------------------------------------

def _call_generation(
    user_prompt: str,
    verbose: bool = False,
    tracer: "PipelineTracer | None" = None,
) -> str:
    """
    Run a generation call via OpenAI-compatible API and return the text response.
    """
    client = openai.OpenAI(base_url=config.api_base_url, api_key="vllm")

    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.generation_max_tokens,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    if tracer is not None:
        tracer.increment_api_calls()
        tracer.add_usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    content = response.choices[0].message.content or ""
    if verbose:
        print(content)
    return content


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
    text = _call_generation(user_prompt, verbose=verbose, tracer=tracer)
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
    text = _call_generation(user_prompt, verbose=verbose, tracer=tracer)
    content, elements = _parse_generation_response(text)

    return GeneratedExample(
        content=content,
        cultural_elements=elements,
    )
