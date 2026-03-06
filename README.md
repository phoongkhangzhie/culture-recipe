# culture-recipe

An agentic system that generates culturally-aligned LLM training examples.

Given a **culture**, a **cultural dimension**, and **generation parameters**, the agent:

1. **Searches the web** for relevant cultural information (DuckDuckGo, client-side)
2. **Generates** a training example grounded in that research
3. **Verifies** quality and cultural accuracy with structured scoring
4. **Refines** the example iteratively until it meets the quality threshold
5. **Repeats** for additional sub-aspects of the dimension if warranted

## Architecture

The agent decides its own workflow. It has five tools and chooses when and how
many times to call each one:

```
Input (culture, dimension, params)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrating Agent                      │
│                  (local model via vLLM)                     │
│                                                             │
│   ┌──────────────────┐   calls   ┌──────────────────────┐   │
│   │ research_culture │ ────────► │ DuckDuckGo search    │   │
│   │  (any # times)   │ ◄──────── │ (client-side tool)   │   │
│   └──────────────────┘  results  └──────────────────────┘   │
│                                                             │
│   ┌──────────────────────┐                                  │
│   │ generate_training_   │  fresh generation or refinement  │
│   │ example (any # times)│  depending on prior feedback     │
│   └──────────────────────┘                                  │
│                                                             │
│   ┌──────────────────────┐                                  │
│   │ verify_training_     │  JSON-mode → VerificationOutput  │
│   │ example (any # times)│  scores + issues + suggestions   │
│   └──────────────────────┘                                  │
│                                                             │
│   ┌─────────────────┐                                       │
│   │ commit_example  │  archive current example, start next  │
│   │ (optional, ≤5)  │  sub-aspect (e.g. different holiday)  │
│   └─────────────────┘                                       │
│                                                             │
│   ┌─────────────────┐                                       │
│   │     finish      │  submit all committed examples        │
│   └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  GenerationResult (one or more ExampleRecords)
```

A typical run might look like:
`research → research → generate → verify → generate (refine) → verify → commit → generate → verify → finish`

But the agent may research more, skip refinement if quality is already high, or
produce only one example — it decides based on the dimension's complexity.
Maximum of **5 examples per dimension** (configurable via `max_examples_per_dimension` in [config.py](config.py)).

## Setup

```bash
# Install dependencies
uv sync   # or: pip install -r requirements.txt

# Start vLLM (requires a GPU)
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# No API key required — everything runs locally
```

The pipeline connects to vLLM's OpenAI-compatible endpoint at
`http://localhost:8000/v1` by default. Override via environment variable:

```bash
export API_BASE_URL=http://localhost:8000/v1
```

## Usage

```bash
# List all available cultural dimensions
python main.py --list-dimensions

# Single dimension
python main.py --culture Japanese --dimension guest_hospitality

# Single dimension with output saved
python main.py --culture Nigerian --dimension hospitality --output result.json --verbose

# Multiple specific dimensions
python main.py --culture Brazilian \
  --dimensions power_distance,collectivism,guest_hospitality \
  --output-dir ./output/brazilian

# All 139 dimensions (resumable — skips already-completed ones)
python main.py --culture Japanese --all-dimensions --output-dir ./output/japanese
```

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--culture` | — | Target culture (e.g. `Japanese`, `Nigerian`) |
| `--dimension` | — | Single dimension key from `--list-dimensions` |
| `--dimensions` | — | Comma-separated list of dimension keys |
| `--all-dimensions` | — | Run all 139 dimensions sequentially |
| `--language` | `English` | Language for the generated examples |
| `--topic` | — | Optional topic hint for the agent |
| `--output` | — | Save single-dimension result to a JSON file |
| `--output-dir` | — | Directory for multi-dimension results (one file per dimension) |
| `--verbose` | false | Show detailed agent output |
| `--trace` | false | Record a full pipeline trace alongside each result |

## Running on SLURM

Use the provided [run.slurm](run.slurm) script. It starts vLLM in the background,
waits for Uvicorn's `"Application startup complete."` log line (confirming the model
is fully loaded), runs the pipeline, then shuts down vLLM cleanly.

```bash
# Single dimension
sbatch --export=ALL,CULTURE=Korean,DIMENSION=filial_piety,OUTPUT_DIR=./output/korean run.slurm

# All dimensions — edit run.slurm to use --all-dimensions
sbatch --export=ALL,CULTURE=Korean,OUTPUT_DIR=./output/korean run.slurm
```

vLLM logs are written to `logs/vllm-<jobid>.log` separately from the SLURM job log.

## Configuration

Edit [config.py](config.py) to change the model or tune token budgets:

```python
api_base_url = "http://localhost:8000/v1"   # vLLM endpoint
model = "Qwen/Qwen2.5-7B-Instruct"          # model served by vLLM
research_model = "Qwen/Qwen2.5-7B-Instruct" # can differ from main model

quality_threshold = 7.0            # minimum overall score to approve (0–10)
max_refinement_iterations = 3      # maximum refine attempts per example
max_examples_per_dimension = 5     # hard cap on examples the agent can produce
orchestrator_max_tokens = 2500   # token budget for the agentic loop
generation_max_tokens = 4000
verification_max_tokens = 2000
research_max_tokens = 8000
research_summary_max_tokens = 1500
```

## Output Format

Each result file contains one or more `ExampleRecord` objects — the agent decides
whether a dimension warrants multiple examples (e.g. separate examples for different
religious holidays within a single dimension).

```json
{
  "culture": "Japanese",
  "dimension": { "name": "guest_hospitality", ... },
  "params": { "language": "English" },
  "records": [
    {
      "example": {
        "content": { "messages": [ ... ] },
        "cultural_elements": ["omotenashi", "slippers", "green tea"]
      },
      "verification": {
        "overall_score": 8.5,
        "is_approved": true,
        "issues": [],
        "suggestions": []
      },
      "refinement_iterations": 1
    }
  ],
  "metadata": { "model": "Qwen/Qwen2.5-7B-Instruct", "agentic_iterations": 6 }
}
```

## Available Dimensions

Dimensions are drawn from the **CultureScope** taxonomy (3 layers, 7 categories, 20 Topic Aspects).
Each fine-grained dimension is a standalone queryable topic — **139 dimensions** in total.
Run `python main.py --list-dimensions` to see the full list.

### Layer 1 — Institutional Norms › Geography & Customs › Population and Geography

| Key | Name |
|---|---|
| `population_rank` | Population Rank |
| `population_distribution` | Population Distribution |
| `land_area_percentage` | Land Area and Geography |
| `main_regions` | Main Regions |
| `ethnicity` | Ethnicity and Racial Composition |
| `official_languages` | Official Languages |
| `widely_spoken_languages` | Widely Spoken Languages |
| `famous_rivers` | Famous Rivers and Waterways |
| `climate_geography` | Climate and Physical Geography |

### Layer 1 — Institutional Norms › Geography & Customs › Dates of Significance

| Key | Name |
|---|---|
| `national_holidays` | National Holidays |
| `religious_holidays` | Religious Holidays |
| `cultural_holidays` | Cultural Holidays and Festivals |
| `festival_origins` | Origin of Festivals |
| `festival_celebrations` | Festival Celebration Practices |
| `festival_symbols` | Festival Symbols |

### Layer 1 — Institutional Norms › Regulation & Policy › Transportation Rules

| Key | Name |
|---|---|
| `vehicle_movement_rules` | Vehicle Movement Rules |
| `traffic_signs_and_signals` | Traffic Signs and Signals |
| `pedestrian_rules` | Pedestrian and Non-Motorized Vehicle Rules |
| `motor_vehicle_driving_rules` | Motor Vehicle Driving Rules |
| `parking_regulations` | Parking Regulations |

### Layer 1 — Institutional Norms › Regulation & Policy › Data Format

| Key | Name |
|---|---|
| `date_order` | Date Order Convention |
| `date_separator` | Date and Number Separator |
| `year_format` | Year Format |
| `month_representation` | Month Representation |
| `zero_padding` | Zero-Padding in Dates |
| `natural_language_date` | Natural Language Date Expression |
| `calendar_system` | Calendar System |

### Layer 1 — Institutional Norms › Regulation & Policy › Measurement Unit

| Key | Name |
|---|---|
| `measurement_system` | Measurement System |
| `unit_localization` | Unit Localization |

### Layer 1 — Institutional Norms › Regulation & Policy › Financial Market Rules

| Key | Name |
|---|---|
| `financial_regulation_structure` | Financial Regulation Structure |
| `financial_market_type` | Financial Market Type |
| `financial_entity_types` | Financial Entity Types |
| `financial_access_licensing` | Financial Access and Licensing |
| `financial_conduct_compliance` | Financial Conduct and Compliance |
| `financial_capital_risk` | Capital Requirements and Risk Management |
| `monetary_payment_system` | Monetary and Payment System |
| `fintech_regulation` | FinTech Regulation |
| `financial_tax_accounting` | Financial Tax and Accounting Standards |
| `financial_international_alignment` | International Financial Alignment |
| `financial_stability_resolution` | Financial Stability and Resolution |

### Layer 2 — Behavioral Patterns › Personal Choices & Habits › Daily Life and Travel

| Key | Name |
|---|---|
| `payment_habits` | Payment Habits |
| `travel_habits` | Travel Habits |
| `bathing_habits` | Bathing and Hygiene Habits |
| `pet_raising_habits` | Pet-Raising Habits |
| `social_media_usage` | Social Media Usage Patterns |
| `marriage_customs` | Marriage Customs |

### Layer 2 — Behavioral Patterns › Personal Choices & Habits › Diet and Health Concept

| Key | Name |
|---|---|
| `eating_habits` | Eating Habits |
| `attitudes_lifecycle` | Attitudes Toward Birth, Aging, Illness, and Death |
| `traditional_vs_modern_medicine` | Traditional vs. Modern Medicine |

### Layer 2 — Behavioral Patterns › Personal Choices & Habits › Education and Knowledge

| Key | Name |
|---|---|
| `educational_practices` | Educational Practices |
| `teacher_student_relationship` | Teacher-Student Relationship |
| `academic_major_selection` | Academic Major Selection |
| `career_choices` | Career Choices and Aspirations |

### Layer 2 — Behavioral Patterns › Personal Choices & Habits › Art and Entertainment

| Key | Name |
|---|---|
| `diaspora_festive_activities` | Diaspora Festive Activities |
| `traditional_music_and_dance` | Traditional Music and Dance |
| `contemporary_art` | Contemporary Art and Culture |

### Layer 2 — Behavioral Patterns › Personal Choices & Habits › Personal Etiquette

| Key | Name |
|---|---|
| `basic_etiquette` | Basic Etiquette |
| `naming_convention` | Naming Convention |
| `name_origin` | Name Origin |
| `name_significance` | Name Significance |
| `punctuality_visiting` | Punctuality When Visiting |
| `shoe_etiquette` | Shoe Etiquette During a Visit |
| `guest_hospitality` | Hospitality Customs When Receiving Guests |
| `host_gift_customs` | Bringing Gifts for the Host |
| `seating_etiquette` | Seating Etiquette for Guests and Hosts |
| `serving_etiquette` | Serving Etiquette During a Visit |
| `leaving_food_etiquette` | Leaving Food After Eating |
| `salt_etiquette` | Using Salt While Eating |
| `meal_compliments` | Giving Compliments During a Meal |
| `right_hand_eating` | Eating with the Right Hand |
| `alcohol_etiquette` | Alcohol Etiquette |
| `pork_dietary_norms` | Pork and Dietary Taboos |
| `gift_giving_gesture` | Handing Gifts |
| `gifts_for_children` | Gifts for Children |
| `gift_opening_norms` | Opening Gifts |

### Layer 2 — Behavioral Patterns › Personal Choices & Habits › Etiquette and Courtesy

| Key | Name |
|---|---|
| `general_greeting_principles` | General Greeting Principles |
| `male_male_greetings` | Greetings Between Men |
| `female_female_greetings` | Greetings Between Women |
| `male_female_greetings` | Greetings Between Men and Women |
| `business_appointment_scheduling` | Business Appointment Scheduling |
| `business_dress_code` | Business Dress Code |
| `business_card_exchange` | Business Card Exchange |
| `business_network_building` | Business Network Building |
| `seniority_in_business` | Age and Experience in Business |
| `business_familiarity` | Familiarity Before Business Meetings |
| `business_socialization` | Socialization During Business Meetings |
| `meeting_duration` | Meeting Duration |
| `open_door_policy` | Open Door Policy in Business |
| `business_meeting_interruptions` | Interruptions During Business Meetings |
| `deference_to_seniority` | Deference to Senior in Business |
| `negotiation_style` | Negotiation Style |
| `business_decision_making` | Business Decision-Making |
| `business_bartering` | Bartering in Business |
| `private_business_meetings` | Private Meetings During Negotiations |
| `confrontation_avoidance_business` | Confrontation Avoidance in Business |
| `business_meeting_followup` | Business Meeting Follow-Up |
| `ongoing_negotiations` | Ongoing Negotiations After a Meeting |

### Layer 2 — Behavioral Patterns › Personal Choices & Habits › Communication Style

| Key | Name |
|---|---|
| `verbal_communication_style` | Verbal Communication Style |
| `indirect_communication` | Indirect Communication |
| `cultural_humour` | Cultural Humour |
| `physical_contact_norms` | Physical Contact Norms |
| `personal_space_norms` | Personal Space |
| `communication_gestures` | Gestures in Communication |
| `beckoning_gestures` | Beckoning Gestures |
| `eye_contact_norms` | Eye Contact Norms |

### Layer 2 — Behavioral Patterns › Personal Choices & Habits › Fixed Expressions in Language

| Key | Name |
|---|---|
| `linguistic_idioms` | Linguistic Idioms |
| `common_sayings` | Common Sayings |
| `proverbs` | Proverbs |
| `neologisms_and_abbreviations` | Neologisms and Abbreviations |

### Layer 3 — Core Values › Social Relationship and Structures › Family Dynamics

| Key | Name |
|---|---|
| `communal_living` | Communal Living |
| `parental_care_norms` | Parental Care |

### Layer 3 — Core Values › Social Relationship and Structures › Household Structures

| Key | Name |
|---|---|
| `patriarchal_structures` | Patriarchal Structures |
| `womens_family_roles` | Women's Roles in the Family |
| `household_social_interaction` | Social Interaction Within Households |
| `urban_rural_divide` | Urban-Rural Divide |
| `seniority_and_childhood` | Seniority and Childhood |

### Layer 3 — Core Values › Social Relationship and Structures › Gender Roles

| Key | Name |
|---|---|
| `male_dominance` | Male Dominance |
| `gender_social_compliance` | Gender Social Compliance |
| `gender_and_honour` | Gender and Honour |
| `changing_gender_attitudes` | Changing Attitudes Toward Gender |

### Layer 3 — Core Values › Values and Beliefs › Cultural Values

| Key | Name |
|---|---|
| `power_distance` | Power Distance |
| `collectivism` | Collectivism |
| `individualism` | Individualism |
| `achievement_motivation` | Motivation Toward Achievement and Success |
| `uncertainty_avoidance` | Uncertainty Avoidance |
| `long_term_orientation` | Long-Term Orientation |
| `cultural_indulgence` | Indulgence |

### Layer 3 — Core Values › Values and Beliefs › Religion

| Key | Name |
|---|---|
| `religious_beliefs_and_practices` | Religious Beliefs and Practices |

### Layer 3 — Core Values › Values and Beliefs › Do's and Don'ts

| Key | Name |
|---|---|
| `modest_dress_norms` | Modest Dress Norms |
| `informality_norms` | Informality Norms |
| `compliment_norms` | Compliment Norms |
| `cultural_acknowledgement` | Cultural Acknowledgement |
| `education_as_value` | Education as a Cultural Value |
| `cultural_insults` | Cultural Insults |
| `inappropriate_humor` | Inappropriate Humor |
| `political_criticism_norms` | Political Criticism Norms |
| `culturally_sensitive_topics` | Culturally Sensitive Topics |
| `ethnicity_assumptions` | Ethnicity Assumptions |
| `cultural_stereotyping` | Cultural Stereotyping |
