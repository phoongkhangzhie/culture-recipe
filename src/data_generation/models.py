from typing import Any, Optional

from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    language: str = "English"
    topic: Optional[str] = None
    implicit_culture: bool = False


class CultureDimension(BaseModel):
    name: str
    category: str
    description: str
    keywords: list[str] = []


class VerificationOutput(BaseModel):
    """Structured output for the verification phase, returned via messages.parse()."""

    cultural_accuracy_score: float = Field(
        description="How accurately this reflects the culture without misrepresentation (0-10)"
    )
    linguistic_authenticity_score: float = Field(
        description="How natural and authentic the language use is for this culture (0-10)"
    )
    dimension_relevance_score: float = Field(
        description="How effectively the cultural dimension is illustrated (0-10)"
    )
    training_quality_score: float = Field(
        description="How valuable this is as LLM training data (0-10)"
    )
    overall_score: float = Field(
        description="Weighted overall quality score (0-10)"
    )
    cultural_elements_verified: list[str] = Field(
        description="Cultural elements that are accurately represented"
    )
    issues: list[str] = Field(
        description="Specific issues found with the example"
    )
    suggestions: list[str] = Field(
        description="Actionable suggestions for improvement"
    )
    is_approved: bool = Field(
        description="True if overall_score >= 7.0 and no critical issues"
    )


class GeneratedExample(BaseModel):
    """The processed multi-turn chat training example."""

    content: dict[str, Any]       # always {"messages": [...]}
    cultural_elements: list[str]


class ExampleRecord(BaseModel):
    """One complete generate → verify → (refine → verify)* cycle."""

    example: GeneratedExample
    verification: VerificationOutput
    refinement_iterations: int = 0


class GenerationResult(BaseModel):
    """Final output of the full generation pipeline."""

    culture: str
    dimension: CultureDimension
    params: GenerationParams
    research_summary: str
    records: list[ExampleRecord]   # one or more examples per dimension
    metadata: dict[str, Any] = Field(default_factory=dict)
