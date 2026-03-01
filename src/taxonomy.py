from src.models import CultureDimension

CULTURAL_DIMENSIONS: dict[str, CultureDimension] = {
    # Hofstede's Six Dimensions
    "power_distance": CultureDimension(
        name="Power Distance",
        category="Hofstede",
        description=(
            "The extent to which less powerful members of society accept and expect "
            "that power is distributed unequally"
        ),
        keywords=[
            "hierarchy",
            "authority",
            "respect for elders",
            "seniority",
            "deference",
            "status symbols",
            "boss-subordinate relations",
        ],
    ),
    "individualism": CultureDimension(
        name="Individualism vs Collectivism",
        category="Hofstede",
        description=(
            "The degree to which people integrate into groups versus focusing on "
            "personal goals and independent identity"
        ),
        keywords=[
            "in-group loyalty",
            "group harmony",
            "personal achievement",
            "self-reliance",
            "interdependence",
            "community obligations",
            "face",
        ],
    ),
    "uncertainty_avoidance": CultureDimension(
        name="Uncertainty Avoidance",
        category="Hofstede",
        description=(
            "A society's tolerance for ambiguity and its tendency to prefer "
            "structure and rules to manage uncertainty"
        ),
        keywords=[
            "rules",
            "structure",
            "risk aversion",
            "formality",
            "planning",
            "tradition",
            "anxiety about the unknown",
        ],
    ),
    "long_term_orientation": CultureDimension(
        name="Long-term vs Short-term Orientation",
        category="Hofstede",
        description=(
            "How much a society values long-standing traditions and thrift versus "
            "adaptability and quick results"
        ),
        keywords=[
            "perseverance",
            "thrift",
            "future planning",
            "tradition",
            "quick gratification",
            "adaptation",
            "saving face",
        ],
    ),
    "masculinity": CultureDimension(
        name="Masculinity vs Femininity",
        category="Hofstede",
        description=(
            "Distribution of emotional roles and social values between genders, "
            "contrasting competitiveness with caring"
        ),
        keywords=[
            "competition",
            "achievement",
            "assertiveness",
            "quality of life",
            "cooperation",
            "nurturing",
            "work-life balance",
        ],
    ),
    "indulgence": CultureDimension(
        name="Indulgence vs Restraint",
        category="Hofstede",
        description=(
            "The extent to which people try to control their desires and impulses, "
            "balancing free gratification against strict social norms"
        ),
        keywords=[
            "leisure",
            "enjoyment",
            "freedom of speech",
            "self-control",
            "modesty",
            "pessimism",
            "happiness norms",
        ],
    ),
    # Communication Styles
    "high_context_communication": CultureDimension(
        name="High-Context Communication",
        category="Communication",
        description=(
            "Communication where meaning is heavily embedded in context, "
            "relationships, and non-verbal cues rather than explicit words"
        ),
        keywords=[
            "indirect speech",
            "face-saving",
            "implicit meaning",
            "reading between lines",
            "non-verbal cues",
            "relationship-based",
            "silence",
        ],
    ),
    "low_context_communication": CultureDimension(
        name="Low-Context Communication",
        category="Communication",
        description=(
            "Communication style that relies on explicit, direct verbal expression "
            "with minimal reliance on implicit context"
        ),
        keywords=[
            "direct speech",
            "explicit meaning",
            "transparency",
            "written contracts",
            "verbal clarity",
            "task-focused",
            "frankness",
        ],
    ),
    # Social Structures
    "family_values": CultureDimension(
        name="Family and Kinship",
        category="Social",
        description=(
            "The role, importance, and structure of family relationships, "
            "including obligations and decision-making within the family unit"
        ),
        keywords=[
            "filial piety",
            "extended family",
            "family obligations",
            "ancestor veneration",
            "marriage customs",
            "elder care",
            "lineage",
        ],
    ),
    "religious_practices": CultureDimension(
        name="Religious and Spiritual Practices",
        category="Social",
        description=(
            "The role of religion and spirituality in shaping daily life, "
            "moral values, and social interactions"
        ),
        keywords=[
            "prayer",
            "festivals",
            "rituals",
            "sacred obligations",
            "community worship",
            "dietary laws",
            "spiritual beliefs",
        ],
    ),
    "hospitality": CultureDimension(
        name="Hospitality and Guest Relations",
        category="Social",
        description=(
            "Cultural norms around welcoming guests, showing generosity, "
            "and the social obligations between hosts and guests"
        ),
        keywords=[
            "guest honor",
            "food sharing",
            "generosity",
            "welcoming rituals",
            "reciprocity",
            "home as sacred space",
            "gift-giving",
        ],
    ),
    "food_culture": CultureDimension(
        name="Food and Culinary Culture",
        category="Social",
        description=(
            "The cultural significance of food, communal eating practices, "
            "dietary traditions, and food as identity and social ritual"
        ),
        keywords=[
            "communal eating",
            "dietary restrictions",
            "traditional cuisine",
            "meal rituals",
            "food as identity",
            "cooking roles",
            "table manners",
        ],
    ),
    "conflict_resolution": CultureDimension(
        name="Conflict Resolution and Harmony",
        category="Social",
        description=(
            "Cultural approaches to handling disagreements, preserving relationships, "
            "and restoring social harmony after disputes"
        ),
        keywords=[
            "face-saving",
            "mediation",
            "direct confrontation",
            "harmony maintenance",
            "third-party involvement",
            "avoidance",
            "reconciliation",
        ],
    ),
    "social_greetings": CultureDimension(
        name="Social Greetings and Etiquette",
        category="Social",
        description=(
            "Cultural norms around greetings, introductions, forms of address, "
            "and everyday social etiquette"
        ),
        keywords=[
            "bowing",
            "handshakes",
            "titles and honorifics",
            "physical contact norms",
            "eye contact",
            "personal space",
            "formal vs informal registers",
        ],
    ),
    # Work and Education
    "work_ethics": CultureDimension(
        name="Work Ethics and Professional Culture",
        category="Work",
        description=(
            "Cultural attitudes toward work, career advancement, professional "
            "relationships, and the balance between work and personal life"
        ),
        keywords=[
            "work-life balance",
            "overtime culture",
            "loyalty to employer",
            "workplace hierarchy",
            "professionalism norms",
            "entrepreneurship",
            "job security",
        ],
    ),
    "education_values": CultureDimension(
        name="Education and Learning Values",
        category="Education",
        description=(
            "Cultural attitudes toward education, academic achievement, "
            "learning styles, and the social role of credentials"
        ),
        keywords=[
            "academic pressure",
            "rote learning",
            "critical thinking",
            "respect for teachers",
            "practical vs theoretical",
            "credentials",
            "parental involvement",
        ],
    ),
    "time_perception": CultureDimension(
        name="Time Perception and Punctuality",
        category="Social",
        description=(
            "Cultural attitudes toward time management, punctuality, "
            "scheduling, and whether time is seen as linear or flexible"
        ),
        keywords=[
            "monochronic",
            "polychronic",
            "punctuality",
            "event-based time",
            "deadlines",
            "multi-tasking",
            "relationship over schedule",
        ],
    ),
    "gender_roles": CultureDimension(
        name="Gender Roles and Equality",
        category="Social",
        description=(
            "Cultural norms and expectations around gender roles, "
            "relationships, opportunities, and the evolving landscape of gender equality"
        ),
        keywords=[
            "gender equality",
            "traditional roles",
            "women in workforce",
            "masculinity norms",
            "femininity ideals",
            "gender dynamics",
            "division of labor",
        ],
    ),
}


def get_dimension(key: str) -> CultureDimension:
    """Get a CultureDimension by its key."""
    if key not in CULTURAL_DIMENSIONS:
        available = list(CULTURAL_DIMENSIONS.keys())
        raise ValueError(
            f"Unknown dimension '{key}'.\nAvailable dimensions: {available}"
        )
    return CULTURAL_DIMENSIONS[key]


def list_dimensions() -> dict[str, str]:
    """Return a mapping of dimension key → description."""
    return {k: v.description for k, v in CULTURAL_DIMENSIONS.items()}
