# culture-recipe

An agentic system that generates culturally-aligned LLM training examples.

Given a **culture**, a **cultural dimension**, and **generation parameters**, the agent:

1. **Searches the web** for relevant cultural information (Anthropic web-search tool)
2. **Generates** a training example grounded in that research
3. **Verifies** quality and cultural accuracy with structured scoring
4. **Refines** the example iteratively until it meets the quality threshold

## Architecture

```
Input (culture, dimension, params)
        │
        ▼
┌───────────────┐   web_search   ┌──────────────────────────┐
│  Phase 1      │ ─────────────► │  Anthropic web_search     │
│  Research     │ ◄───────────── │  (server-side tool)       │
└───────┬───────┘                └──────────────────────────┘
        │ research_context (text)
        ▼
┌───────────────┐
│  Phase 2      │  claude-opus-4-6 + adaptive thinking
│  Generate     │  → JSON code blocks parsed from text response
└───────┬───────┘
        │ GeneratedExample
        ▼
┌───────────────┐
│  Phase 3      │  claude-opus-4-6 + adaptive thinking
│  Verify       │  → messages.parse() → VerificationOutput (Pydantic)
└───────┬───────┘
        │ score < threshold?
        ▼
┌───────────────┐
│  Phase 4      │  Repeat up to max_refinement_iterations times
│  Refine       │
└───────┬───────┘
        ▼
  GenerationResult (JSON)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Usage

```bash
# List all available cultural dimensions
python main.py --list-dimensions

# Minimal example
python main.py --culture Japanese --dimension power_distance

# Full options
python main.py \
  --culture Nigerian \
  --dimension hospitality \
  --language English \
  --example-type conversation \
  --output-format sharegpt \
  --length medium \
  --num-turns 3 \
  --topic "welcoming a first-time guest" \
  --output output.json \
  --verbose
```

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--culture` | — | Target culture (e.g. `Japanese`, `Nigerian`) |
| `--dimension` | — | Dimension key from `--list-dimensions` |
| `--language` | `English` | Language for the example |
| `--example-type` | `conversation` | `conversation`, `qa`, `instruction`, `story`, `preference_pair` |
| `--output-format` | `openai` | `openai`, `alpaca`, `sharegpt`, `raw` |
| `--length` | `medium` | `short`, `medium`, `long` |
| `--num-turns` | `2` | Dialogue turns (for `conversation` type) |
| `--topic` | — | Optional specific topic within the dimension |
| `--output` | — | Save result to a JSON file |
| `--verbose` | false | Stream model output to stdout |

## Available Dimensions

| Key | Name | Category |
|---|---|---|
| `power_distance` | Power Distance | Hofstede |
| `individualism` | Individualism vs Collectivism | Hofstede |
| `uncertainty_avoidance` | Uncertainty Avoidance | Hofstede |
| `long_term_orientation` | Long-term vs Short-term Orientation | Hofstede |
| `masculinity` | Masculinity vs Femininity | Hofstede |
| `indulgence` | Indulgence vs Restraint | Hofstede |
| `high_context_communication` | High-Context Communication | Communication |
| `low_context_communication` | Low-Context Communication | Communication |
| `family_values` | Family and Kinship | Social |
| `religious_practices` | Religious and Spiritual Practices | Social |
| `hospitality` | Hospitality and Guest Relations | Social |
| `food_culture` | Food and Culinary Culture | Social |
| `conflict_resolution` | Conflict Resolution and Harmony | Social |
| `social_greetings` | Social Greetings and Etiquette | Social |
| `work_ethics` | Work Ethics and Professional Culture | Work |
| `education_values` | Education and Learning Values | Education |
| `time_perception` | Time Perception and Punctuality | Social |
| `gender_roles` | Gender Roles and Equality | Social |

## Output Format Examples

**OpenAI** (`--output-format openai`)
```json
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Alpaca** (`--output-format alpaca`)
```json
{"instruction": "...", "input": "...", "output": "..."}
```

**ShareGPT** (`--output-format sharegpt`)
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

## Configuration

Edit [config.py](config.py) to tune thresholds and token budgets:

```python
quality_threshold = 7.0          # minimum overall score to approve (0–10)
max_refinement_iterations = 3    # maximum refine attempts per example
research_max_tokens = 8000
generation_max_tokens = 4000
verification_max_tokens = 2000
```

## Model

Uses **claude-opus-4-6** with `thinking: {type: "adaptive"}` on all phases.
The web-search tool (`web_search_20260209`) is fully server-side — no external
search API key required.
