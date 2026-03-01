import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    model: str = "claude-opus-4-6"
    # Research phase
    research_max_tokens: int = 8000
    max_research_continuations: int = 5
    # Generation phase
    generation_max_tokens: int = 4000
    # Verification phase
    verification_max_tokens: int = 2000
    # Refinement loop
    quality_threshold: float = 7.0
    max_refinement_iterations: int = 3


config = Config()
