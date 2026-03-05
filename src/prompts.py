import json

from src.models import (
    CultureDimension,
    ExampleLength,
    ExampleType,
    GenerationParams,
    OutputFormat,
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """\
You are an expert cultural anthropologist and researcher with deep knowledge of \
cross-cultural studies, ethnography, and regional cultures worldwide.

Your task is to gather accurate, nuanced, and comprehensive information about \
cultural practices, values, and norms using web search.

When researching:
- Prioritise primary sources: cultural institutions, academic publications, \
  authentic cultural voices and journalists from within the culture.
- Document both traditional practices and contemporary expressions.
- Note regional and generational variations.
- Include specific, concrete examples of how the dimension manifests in daily life.
- Identify key vocabulary, phrases, or concepts important to understanding the culture.
- Avoid stereotypes while acknowledging genuine cultural patterns.
- Be thorough — search multiple angles to build a complete, nuanced picture.\
"""

GENERATION_SYSTEM_PROMPT = """\
You are an expert in cross-cultural LLM alignment and training data generation.

Your role is to create high-quality, culturally authentic training examples that \
help AI systems understand and appropriately respond within diverse cultural contexts.

Core principles:
1. Cultural authenticity — use realistic names, places, and scenarios from the culture.
2. Naturalness — cultural elements should be organic, not forced or caricatured.
3. Dimension embedding — the cultural dimension should surface through authentic \
   behaviour, dialogue, and context.
4. Linguistic precision — match appropriate formality levels and communication styles.
5. Training value — the example must effectively teach cultural patterns to an LLM.
6. Avoid harm — reflect genuine cultural depth, never harmful stereotypes.\
"""

VERIFICATION_SYSTEM_PROMPT = """\
You are an expert cultural evaluator specialising in assessing LLM training data \
for cultural authenticity and alignment quality.

Evaluate examples rigorously. Your assessment should reflect whether an LLM trained \
on this example would develop accurate, nuanced cultural understanding.

Scoring guide (0-10):
  0-3  = Poor / inaccurate / harmful
  4-5  = Adequate but shallow or generic
  6-7  = Good — culturally grounded with minor issues
  8-9  = Excellent — authentic, nuanced, highly valuable
  10   = Exceptional / publishable quality

Set is_approved = true ONLY if overall_score >= 7.0 and no critical issues.\
"""


# ---------------------------------------------------------------------------
# Helper: format schema for the requested output format
# ---------------------------------------------------------------------------

_FORMAT_SCHEMAS: dict[str, str] = {
    OutputFormat.OPENAI: """\
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}""",
    OutputFormat.OPENAI
    + "_preference": """\
{
  "prompt":   [{"role": "user", "content": "..."}],
  "chosen":   [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}""",
    OutputFormat.ALPACA: """\
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}""",
    OutputFormat.SHAREGPT: """\
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt",   "value": "..."}
  ]
}""",
    OutputFormat.RAW: '{"text": "..."}',
}


def _get_format_schema(params: GenerationParams) -> str:
    if (
        params.output_format == OutputFormat.OPENAI
        and params.example_type == ExampleType.PREFERENCE_PAIR
    ):
        return _FORMAT_SCHEMAS[OutputFormat.OPENAI + "_preference"]
    return _FORMAT_SCHEMAS[params.output_format]


# ---------------------------------------------------------------------------
# Research prompt
# ---------------------------------------------------------------------------

def get_research_prompt(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    focus_query: str = "",
) -> str:
    topic_clause = f" specifically around '{params.topic}'" if params.topic else ""
    focus_clause = f"\n\n**Research focus**: {focus_query}" if focus_query else ""
    return f"""\
Research {culture} culture's relationship with the following cultural dimension{topic_clause}:{focus_clause}

**Dimension**: {dimension.name}
**Description**: {dimension.description}
**Key concepts**: {', '.join(dimension.keywords)}

I need comprehensive cultural context to generate authentic LLM training examples. \
Please search for and synthesise:

1. **Core values and beliefs** — How does {culture} culture approach {dimension.name}? \
   What are the underlying values?
2. **Concrete examples** — Specific customs, practices, or behaviours that illustrate \
   this dimension in {culture} society.
3. **Daily life manifestations** — How does {dimension.name} show up in everyday \
   {culture} interactions (home, workplace, public spaces)?
4. **Language patterns** — Vocabulary, phrases, honorifics, or communication styles \
   related to this dimension.
5. **Social contexts** — Where this dimension is most visible (family, work, education, \
   religion, etc.).
6. **Historical and contemporary context** — How these values developed and any notable \
   shifts in recent generations.
7. **Regional/generational variations** — Important differences within {culture} culture.

Target language for the training example: {params.language}
Example type to be generated: {params.example_type.value}

Please synthesise your findings into a comprehensive cultural brief.\
"""


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

_LENGTH_GUIDE: dict[ExampleLength, str] = {
    ExampleLength.SHORT: "50–150 words per turn or response section",
    ExampleLength.MEDIUM: "150–350 words per turn or response section",
    ExampleLength.LONG: "350–700 words per turn or response section",
}

_TYPE_TEMPLATES: dict[ExampleType, str] = {
    ExampleType.CONVERSATION: (
        "a natural {num_turns}-turn dialogue between people in a {culture} cultural context"
    ),
    ExampleType.QA: (
        "a question-answer pair set in a {culture} cultural context, where the "
        "answer demonstrates culturally appropriate knowledge or behaviour"
    ),
    ExampleType.INSTRUCTION: (
        "an instruction-following task embedded in {culture} cultural context — "
        "the instruction and response should both reflect cultural norms"
    ),
    ExampleType.STORY: (
        "a short narrative or anecdote (2–4 paragraphs) depicting {culture} "
        "cultural values in action"
    ),
    ExampleType.PREFERENCE_PAIR: (
        "two assistant responses to the same user prompt — one culturally aligned "
        "with {culture} values (chosen) and one culturally misaligned or generic (rejected), "
        "with a brief rationale for each"
    ),
}


def get_generation_prompt(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    research_context: str,
) -> str:
    type_desc = _TYPE_TEMPLATES[params.example_type].format(
        culture=culture, num_turns=params.num_turns
    )
    format_schema = _get_format_schema(params)

    return f"""\
Using the cultural research below, generate an authentic training example for \
LLM cultural alignment.

## Cultural Research Context

{research_context}

---

## Generation Requirements

- **Culture**: {culture}
- **Dimension**: {dimension.name} — {dimension.description}
- **Language**: {params.language}
- **Example Type**: Create {type_desc}
- **Length per section**: {_LENGTH_GUIDE[params.length]}
{f"- **Specific topic**: {params.topic}" if params.topic else ""}

## Output Instructions

Return **two JSON code blocks** in your response:

**Block 1 — The training example** (follow this schema exactly):
```json
{format_schema}
```

**Block 2 — Cultural elements** (a JSON array of strings listing the specific \
cultural elements you incorporated):
```json
["element 1", "element 2", ...]
```

### Quality checklist before submitting:
- Cultural elements feel organic, not forced or stereotyped.
- Names, places, and scenarios are authentic to {culture} culture.
- The language register (formal/informal, direct/indirect) matches the cultural context.
- The dimension '{dimension.name}' is clearly — but naturally — expressed.
- The example would be genuinely useful for teaching an LLM about {culture} culture.\
"""


# ---------------------------------------------------------------------------
# Verification prompt
# ---------------------------------------------------------------------------

def get_verification_prompt(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    example_content: dict,
    research_context: str,
) -> str:
    excerpt = (
        research_context[:2500] + "\n[...truncated...]"
        if len(research_context) > 2500
        else research_context
    )
    return f"""\
Evaluate this LLM training example for cultural authenticity and training quality.

## Evaluation Context

- **Target Culture**: {culture}
- **Cultural Dimension**: {dimension.name} — {dimension.description}
- **Language**: {params.language}
- **Example Type**: {params.example_type.value}

## Cultural Research (Ground Truth)

{excerpt}

## Example to Evaluate

```json
{json.dumps(example_content, ensure_ascii=False, indent=2)}
```

## Scoring Instructions

Score each dimension 0–10 and provide:
1. **cultural_accuracy_score** — How accurately does this reflect {culture} culture's \
   approach to {dimension.name}? Penalise stereotypes or factual errors.
2. **linguistic_authenticity_score** — Is the language natural for this cultural \
   context? Would native speakers find it believable?
3. **dimension_relevance_score** — How effectively and naturally does this illustrate \
   {dimension.name}?
4. **training_quality_score** — Would a model trained on this develop accurate, \
   nuanced cultural understanding?
5. **overall_score** — Weighted score: \
   cultural_accuracy × 0.35 + linguistic_authenticity × 0.25 + \
   dimension_relevance × 0.25 + training_quality × 0.15

Also provide:
- **cultural_elements_verified** — Cultural elements that are accurately represented.
- **issues** — Specific problems (be precise, e.g. "The honorific used is incorrect \
  for this social context").
- **suggestions** — Actionable improvements.
- **is_approved** — true only if overall_score >= 7.0 and no critical issues.\
"""


# ---------------------------------------------------------------------------
# Refinement prompt
# ---------------------------------------------------------------------------

def get_refinement_prompt(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    example_content: dict,
    verification: dict,
    research_context: str,
) -> str:
    excerpt = (
        research_context[:1800] + "\n[...truncated...]"
        if len(research_context) > 1800
        else research_context
    )
    format_schema = _get_format_schema(params)
    issues_str = "; ".join(verification.get("issues", [])) or "None specified."
    suggestions_str = "; ".join(verification.get("suggestions", [])) or "None specified."

    return f"""\
Improve the following training example based on evaluation feedback.

## Original Example

```json
{json.dumps(example_content, ensure_ascii=False, indent=2)}
```

## Evaluation Feedback

| Dimension | Score |
|---|---|
| Cultural Accuracy | {verification.get('cultural_accuracy_score', 0):.1f}/10 |
| Linguistic Authenticity | {verification.get('linguistic_authenticity_score', 0):.1f}/10 |
| Dimension Relevance | {verification.get('dimension_relevance_score', 0):.1f}/10 |
| Training Quality | {verification.get('training_quality_score', 0):.1f}/10 |
| **Overall** | **{verification.get('overall_score', 0):.1f}/10** |

**Issues identified**: {issues_str}
**Suggestions**: {suggestions_str}

## Cultural Research Context

{excerpt}

## Requirements

- Culture: {culture} | Dimension: {dimension.name} | Language: {params.language}
- Format: {params.output_format.value} | Type: {params.example_type.value}
{f"- Topic: {params.topic}" if params.topic else ""}

## Task

Create an improved version that addresses every issue and implements the suggestions. \
Return the same two JSON code blocks as before:

**Block 1 — Improved training example**:
```json
{format_schema}
```

**Block 2 — Updated cultural elements list**:
```json
["element 1", "element 2", ...]
```\
"""
