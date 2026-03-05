"""
Phase 1 — Research.

Uses Anthropic's built-in web_search_20260209 server-side tool to gather
cultural context.  Handles pause_turn continuation for multi-round searches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anthropic

from config import config
from src.models import CultureDimension, GenerationParams
from src.prompts import RESEARCH_SYSTEM_PROMPT, get_research_prompt

if TYPE_CHECKING:
    from src.tracer import PipelineTracer

# The web-search tool declaration (server-side, no schema required)
_WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}


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

    Uses streaming so progress is visible; handles the pause_turn stop reason
    that occurs when the server-side search loop reaches its iteration limit.
    """
    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    user_prompt = get_research_prompt(culture, dimension, params, focus_query=focus_query)

    # Initial message history
    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    last_response = None

    for iteration in range(config.max_research_continuations + 1):
        if verbose and iteration > 0:
            print(f"  [research] Continuing search (round {iteration})…")

        with client.messages.stream(
            model=config.model,
            max_tokens=config.research_max_tokens,
            system=RESEARCH_SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            tools=[_WEB_SEARCH_TOOL],
            messages=messages,
        ) as stream:
            if verbose:
                _print_research_stream(stream)

            last_response = stream.get_final_message()

        if tracer is not None:
            tracer.increment_api_calls()
            tracer.add_usage(
                input_tokens=last_response.usage.input_tokens,
                output_tokens=last_response.usage.output_tokens,
            )
            from src.tracer import extract_trace_data
            extract_trace_data(last_response.content, tracer)

        if last_response.stop_reason == "end_turn":
            break

        if last_response.stop_reason == "pause_turn":
            # Server-side loop hit its limit — re-send to continue.
            # Do NOT add a new user message; the API resumes automatically
            # when it sees a trailing server_tool_use block.
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": last_response.content},
            ]
            continue

        # Any other stop reason — accept what we have
        break

    if last_response is None:
        return "Research could not be completed."

    text_parts = [
        block.text
        for block in last_response.content
        if block.type == "text"
    ]
    return "\n\n".join(text_parts) if text_parts else "Research summary unavailable."


# ---------------------------------------------------------------------------
# Verbose stream printer
# ---------------------------------------------------------------------------

def _print_research_stream(stream: anthropic.MessageStream) -> None:
    """Print text deltas and search-query notifications while streaming."""
    in_text = False
    for event in stream:
        etype = getattr(event, "type", None)

        if etype == "content_block_start":
            block = getattr(event, "content_block", None)
            if block is None:
                continue
            if block.type == "text":
                in_text = True
            elif block.type == "server_tool_use" and block.name == "web_search":
                # The query shows up in block.input once the block closes;
                # print a placeholder for now
                print("\n  [web_search] Searching…", flush=True)
                in_text = False

        elif etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and delta.type == "text_delta" and in_text:
                print(delta.text, end="", flush=True)

        elif etype == "content_block_stop":
            in_text = False

    print()  # trailing newline
