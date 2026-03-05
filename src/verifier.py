"""
Phase 3 — Verification.

Uses client.messages.parse() with a Pydantic model to get structured,
validated evaluation scores from the model.

Adaptive thinking is enabled for careful, reasoned assessment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic

from config import config
from src.models import (
    CultureDimension,
    GeneratedExample,
    GenerationParams,
    VerificationOutput,
)
from src.prompts import VERIFICATION_SYSTEM_PROMPT, get_verification_prompt

if TYPE_CHECKING:
    from src.tracer import PipelineTracer


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

    Uses messages.parse() so the SDK validates the response against the
    VerificationOutput Pydantic model, enforcing all field constraints
    client-side even when the API doesn't support them server-side.
    """
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    user_prompt = get_verification_prompt(
        culture, dimension, params, example.content, research_context
    )

    if verbose:
        print("  [verifier] Evaluating cultural authenticity and training quality…")

    response = client.messages.parse(
        model=config.model,
        max_tokens=config.verification_max_tokens,
        system=VERIFICATION_SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_prompt}],
        output_format=VerificationOutput,
    )

    if tracer is not None:
        tracer.increment_api_calls()
        tracer.add_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        from src.tracer import extract_trace_data
        extract_trace_data(response.content, tracer)

    result: VerificationOutput = response.parsed_output

    # Clamp scores to [0, 10] in case the model drifts slightly
    def clamp(v: float) -> float:
        return max(0.0, min(10.0, v))

    return VerificationOutput(
        cultural_accuracy_score=clamp(result.cultural_accuracy_score),
        linguistic_authenticity_score=clamp(result.linguistic_authenticity_score),
        dimension_relevance_score=clamp(result.dimension_relevance_score),
        training_quality_score=clamp(result.training_quality_score),
        overall_score=clamp(result.overall_score),
        cultural_elements_verified=result.cultural_elements_verified,
        issues=result.issues,
        suggestions=result.suggestions,
        is_approved=result.is_approved,
    )
