import json

from .models import CultureDimension, GenerationParams


# ---------------------------------------------------------------------------
# System prompt template for generated training examples
# ---------------------------------------------------------------------------

def get_example_system_prompt(culture: str) -> str:
    """
    The system message embedded in every generated training example.
    Kept intentionally simple so the model focuses on the cultural content,
    not on crafting a system prompt.
    """
    return (
        f"You are a helpful AI assistant with deep knowledge of {culture} culture. "
        f"Represent the values and lived experience of {culture} people in your responses."
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

Your role is to create high-quality, culturally authentic multi-turn chat training \
examples that help AI systems understand and appropriately respond within diverse \
cultural contexts.

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

Please synthesise your findings into a comprehensive cultural brief.\
"""


# ---------------------------------------------------------------------------
# Generation prompt
# ---------------------------------------------------------------------------

def get_generation_prompt(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    research_context: str,
) -> str:
    topic_hint = (
        f"\n- **Topic hint** (optional starting point): {params.topic}"
        if params.topic
        else ""
    )
    implicit_mode_note = (
        f"\n- **Mode**: Implicit cultural context — see task instructions below"
        if params.implicit_culture
        else ""
    )
    implicit_task_block = (
        f"""\

**Implicit cultural context mode.**
Write user messages the way a natural insider of {culture} culture would write them — \
using culturally specific language, references, and norms organically, without \
explicitly announcing or performing their cultural background (e.g. avoid "As a {culture} \
person, I..." framing). Cultural markers should appear because they are natural to the \
speaker, not because they are signalling culture to an AI. \
The assistant should respond with culturally shared assumptions — as if both parties \
simply share that background — rather than explaining or reflecting the culture back at \
the user."""
        if params.implicit_culture
        else ""
    )
    return f"""\
Using the cultural research below, generate an authentic multi-turn chat training \
example for LLM cultural alignment.

## Cultural Research Context

{research_context}

---

## Generation Requirements

- **Culture**: {culture}
- **Dimension**: {dimension.name} — {dimension.description}
- **Language**: {params.language}
- **Format**: Multi-turn conversation between a user and an AI assistant{topic_hint}{implicit_mode_note}

## Your Task

Choose a realistic, culturally grounded scenario in which a user seeks help from an \
AI assistant. The task can be anything — advice, planning, problem-solving, creative \
writing, language help, navigating a social situation, etc. — as long as:

1. It arises naturally from {culture} cultural life.
2. The conversation authentically illustrates the **{dimension.name}** dimension.
3. Both user and assistant responses reflect culturally appropriate norms.
{implicit_task_block}

Decide how many turns best serve the scenario (typically 3–12 exchanges, favouring more turns). \
Longer conversations are strongly preferred — more turns let cultural norms emerge gradually \
and naturally across the dialogue, rather than being stated all at once.

Calibrate the length of **each individual assistant response** to what the moment calls for:
- **Brief and direct** when the answer is simple, factual, or the culture expects conciseness. \
  A one-line reply, a short acknowledgment, or a quick follow-up question is often the most \
  natural response. Do not pad it.
- **Longer and more developed** when the situation genuinely calls for explanation, cultural \
  context, emotional support, step-by-step guidance, or nuanced reasoning.
- **Formatted** (bullet lists, numbered steps, tables, headers) only when structure genuinely \
  aids comprehension — e.g. comparing options, listing steps, organising multiple pieces of \
  advice. Do not force structure where flowing prose or a single sentence is more natural.

Avoid padding or filler — every sentence must add value. A one-line reply and a multi-paragraph \
structured response can both be correct, depending on the scenario. Mixing short and long \
responses across turns is realistic and desirable.

## Output Instructions

Return **two JSON code blocks** in your response:

**Block 1 — The training example** (follow this schema exactly):
```json
{{
  "messages": [
    {{"role": "system", "content": "{get_example_system_prompt(culture)}"}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}
```
The `system` message is required and must be exactly as shown above. \
There must be at least two user/assistant exchanges.

**Block 2 — Cultural elements** (a JSON array listing what you incorporated):
```json
["element 1", "element 2", ...]
```

### Quality checklist before submitting:
- The scenario feels authentic, not contrived or stereotyped.
- Names, places, and references are specific to {culture}.
- The assistant's tone and style match culturally appropriate communication norms.
- The **{dimension.name}** dimension surfaces organically through the dialogue.
- Each assistant response is exactly as long as it needs to be — no padding, no truncation.
- The full exchange would genuinely teach an LLM about {culture} culture.
{f"- (Implicit mode) User messages use cultural language naturally, as an insider would — without announcing or performing their cultural identity." if params.implicit_culture else ""}\
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
Evaluate this multi-turn chat LLM training example for cultural authenticity and \
training quality.

## Evaluation Context

- **Target Culture**: {culture}
- **Cultural Dimension**: {dimension.name} — {dimension.description}
- **Language**: {params.language}
- **Format**: Multi-turn chat (user ↔ assistant)
{f"- **Mode**: Implicit cultural context — user messages should use cultural language naturally, as an insider would, without explicitly stating their background. Penalise any user turn that announces or performs cultural identity (e.g. 'As a {culture} person, I...'). The assistant should respond with culturally shared assumptions, not explanations directed at an outsider." if params.implicit_culture else ""}

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
    issues_str = "; ".join(verification.get("issues", [])) or "None specified."
    suggestions_str = "; ".join(verification.get("suggestions", [])) or "None specified."

    return f"""\
Improve the following multi-turn chat training example based on evaluation feedback.

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
- Format: Multi-turn chat (user ↔ assistant)
{f"- Topic hint: {params.topic}" if params.topic else ""}\
{f"""- Mode: Implicit cultural context — user messages use culturally specific language naturally, \
as an insider would, without announcing their background. The assistant responds with shared \
cultural assumptions rather than explaining the culture back to the user.""" if params.implicit_culture else ""}

## Task

Create an improved version that addresses every issue and implements the suggestions. \
You may adjust the scenario, number of turns, or dialogue content as needed. \
Calibrate the length of each individual assistant response to what the moment calls for — \
short and direct when the answer is simple or cultural norms expect brevity, longer when \
explanation, emotional support, or nuanced guidance is genuinely needed. \
Mix short and long responses across turns to keep the conversation natural. Avoid padding. \
Return the same two JSON code blocks as before:

**Block 1 — Improved training example**:
```json
{{
  "messages": [
    {{"role": "system", "content": "{get_example_system_prompt(culture)}"}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}
```

**Block 2 — Updated cultural elements list**:
```json
["element 1", "element 2", ...]
```\
"""
