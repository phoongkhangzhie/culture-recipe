"""
Phase 3 — Verification.

Uses the OpenAI-compatible API (Ollama) with JSON-mode structured output to get
validated evaluation scores. The VerificationOutput Pydantic model is used to
parse and validate the model's JSON response.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import openai

from config import config
from .models import (
    CultureDimension,
    GeneratedExample,
    GenerationParams,
    VerificationOutput,
)
from .prompts import VERIFICATION_SYSTEM_PROMPT, get_verification_prompt

if TYPE_CHECKING:
    from .tracer import PipelineTracer

# JSON schema appended to the system prompt so the model knows what to output
_VERIFICATION_JSON_SCHEMA = """

Respond with a single valid JSON object (no markdown, no explanation) with exactly these fields:
{
  "cultural_accuracy_score": <float 0-10>,
  "linguistic_authenticity_score": <float 0-10>,
  "dimension_relevance_score": <float 0-10>,
  "training_quality_score": <float 0-10>,
  "overall_score": <float 0-10>,
  "cultural_elements_verified": [<string>, ...],
  "issues": [<string>, ...],
  "suggestions": [<string>, ...],
  "is_approved": <true|false>
}"""


def verify_example(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    example: GeneratedExample,
    research_context: str,
    verbose: bool = False,
    tracer: "PipelineTracer | None" = None,
) -> VerificationOutput:
    """
    Phase 3: Evaluate the training example and return structured scores.

    Uses JSON-mode output and manually validates against the VerificationOutput
    Pydantic model.
    """
    client = openai.OpenAI(base_url=config.api_base_url, api_key="vllm")

    user_prompt = get_verification_prompt(
        culture, dimension, params, example.content, research_context
    )

    if verbose:
        print("  [verifier] Evaluating cultural authenticity and training quality…")

    system_with_schema = VERIFICATION_SYSTEM_PROMPT + _VERIFICATION_JSON_SCHEMA

    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.verification_max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_with_schema},
            {"role": "user", "content": user_prompt},
        ],
    )

    if tracer is not None:
        tracer.increment_api_calls()
        tracer.add_usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    raw_content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        # Fallback: return a minimal passing-ish result rather than crashing
        data = {}

    def clamp(v: float) -> float:
        return max(0.0, min(10.0, float(v)))

    def _float(key: str) -> float:
        try:
            return clamp(data.get(key, 0))
        except (TypeError, ValueError):
            return 0.0

    def _strlist(key: str) -> list[str]:
        val = data.get(key, [])
        if isinstance(val, list):
            return [str(x) for x in val]
        return []

    return VerificationOutput(
        cultural_accuracy_score=_float("cultural_accuracy_score"),
        linguistic_authenticity_score=_float("linguistic_authenticity_score"),
        dimension_relevance_score=_float("dimension_relevance_score"),
        training_quality_score=_float("training_quality_score"),
        overall_score=_float("overall_score"),
        cultural_elements_verified=_strlist("cultural_elements_verified"),
        issues=_strlist("issues"),
        suggestions=_strlist("suggestions"),
        is_approved=bool(data.get("is_approved", False)),
    )
