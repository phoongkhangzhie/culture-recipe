"""
Phase 1 — Research.

Uses DuckDuckGo as a client-side web search tool via OpenAI-compatible API
(Ollama). Runs a tool-use loop: the model calls web_search as needed, then
synthesises findings into a cultural brief.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import openai
from ddgs import DDGS

from config import config
from .models import CultureDimension, GenerationParams
from .prompts import RESEARCH_SYSTEM_PROMPT, get_research_prompt

if TYPE_CHECKING:
    from .tracer import PipelineTracer

# Web search tool definition (OpenAI function-calling format)
_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for information about a topic. "
            "Returns snippets from the top results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}


def _do_web_search(query: str, max_results: int = 5) -> str:
    """Execute a DuckDuckGo search and return formatted results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return "No results found."

    parts = []
    for r in results:
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        parts.append(f"**{title}**\n{href}\n{body}")
    return "\n\n---\n\n".join(parts)


def research_cultural_context(
    culture: str,
    dimension: CultureDimension,
    params: GenerationParams,
    verbose: bool = False,
    tracer: "PipelineTracer | None" = None,
    focus_query: str = "",
) -> str:
    """
    Search the web for cultural context and return a synthesised brief.

    Runs an OpenAI-compatible tool-use loop with DuckDuckGo as the search
    backend. The model decides how many searches to perform before synthesising.
    """
    client = openai.OpenAI(base_url=config.api_base_url, api_key="vllm")
    user_prompt = get_research_prompt(culture, dimension, params, focus_query=focus_query)

    messages: list[dict] = [
        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    def _create_with_backoff(**kwargs):
        delay = 60
        for attempt in range(5):
            try:
                return client.chat.completions.create(**kwargs)
            except openai.RateLimitError:
                if attempt == 4:
                    raise
                print(
                    f"  [research] Rate limit — waiting {delay}s (attempt {attempt + 1}/5)…",
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2, 300)

    last_content = ""

    for _iteration in range(config.max_research_continuations + 1):
        response = _create_with_backoff(
            model=config.research_model,
            max_tokens=config.research_max_tokens,
            tools=[_WEB_SEARCH_TOOL],
            tool_choice="auto",
            messages=messages,
        )

        if tracer is not None:
            tracer.increment_api_calls()
            tracer.add_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        choice = response.choices[0]

        # Capture any text the model produced
        if choice.message.content:
            last_content = choice.message.content

        # Build assistant message for history
        assistant_msg: dict = {"role": "assistant", "content": choice.message.content}
        if choice.message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls → model has synthesised its answer
        if not choice.message.tool_calls:
            break

        # Execute each tool call and add results to the conversation
        for tc in choice.message.tool_calls:
            if tc.function.name == "web_search":
                try:
                    args = json.loads(tc.function.arguments)
                    query = args.get("query", "")
                except (json.JSONDecodeError, AttributeError):
                    query = str(tc.function.arguments)

                if verbose:
                    print(f"\n  [web_search] Searching: {query}", flush=True)

                result = _do_web_search(query)

                if tracer is not None:
                    tracer.record_tool_call(
                        tool="web_search",
                        input_data={"query": query},
                        tool_use_id=tc.id,
                        result=result[:500],
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    return last_content if last_content else "Research summary unavailable."


# ---------------------------------------------------------------------------
# Research summariser
# ---------------------------------------------------------------------------

def summarize_research(
    raw_research: str,
    culture: str,
    dimension: CultureDimension,
) -> str:
    """
    Condense a raw research result into a compact cultural briefing.

    Uses the research model to keep it lightweight.
    """
    if not raw_research.strip():
        return raw_research

    client = openai.OpenAI(base_url=config.api_base_url, api_key="vllm")
    prompt = (
        f"You are condensing web-search research for a cultural training data project.\n\n"
        f"Culture: {culture}\n"
        f"Dimension: {dimension.name} — {dimension.description}\n\n"
        f"Summarise the research below into a concise briefing (≤400 words) that preserves "
        f"the most important cultural norms, practices, anecdotes, and nuances relevant to "
        f"the dimension. Keep specific examples and culturally distinctive details; drop "
        f"generic or off-topic content.\n\n"
        f"RESEARCH:\n{raw_research}"
    )

    delay = 60
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=config.research_model,
                max_tokens=config.research_summary_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or raw_research[:6000]
        except openai.RateLimitError:
            if attempt == 4:
                return raw_research[:6000]
            print(
                f"  [summarise] Rate limit — waiting {delay}s (attempt {attempt + 1}/5)…",
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, 300)

    return raw_research[:6000]
