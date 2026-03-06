import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Local LLM server (vLLM / Ollama / LM Studio — any OpenAI-compatible endpoint)
    api_base_url: str = field(
        default_factory=lambda: os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
    )
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    research_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # Research phase
    research_max_tokens: int = 8000
    max_research_continuations: int = 2
    research_summary_max_tokens: int = 1500  # max tokens for each summarised research chunk
    # Orchestrator (agentic loop) — kept lower to limit token accumulation
    orchestrator_max_tokens: int = 2500
    # Generation phase
    generation_max_tokens: int = 4000
    # Verification phase
    verification_max_tokens: int = 2000
    # Refinement loop
    quality_threshold: float = 7.0
    max_refinement_iterations: int = 3
    max_examples_per_dimension: int = 5


config = Config()
