"""
Pipeline tracing — records inputs, tool calls, thinking, outputs, and token
usage for every phase of the culture-recipe pipeline.

Usage
-----
tracer = PipelineTracer(culture, dimension_key, params)
tracer.start_phase("research", {"culture": culture, "dimension": ...})
tracer.record_thinking("…")
tracer.record_tool_call("web_search", {"query": "…"}, result="…")
tracer.add_usage(input_tokens=1234, output_tokens=567)
tracer.end_phase(output="…")
tracer.save("run_trace.json")
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PhaseTrace:
    phase: str
    started_at: str
    ended_at: str | None = None
    duration_ms: float | None = None
    api_calls: int = 0
    input: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    output: Any = None
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "api_calls": self.api_calls,
            "input": self.input,
            "tool_calls": self.tool_calls,
            "thinking": self.thinking,
            "output": self.output,
            "usage": self.usage,
        }


# ---------------------------------------------------------------------------
# Main tracer
# ---------------------------------------------------------------------------

class PipelineTracer:
    """
    Collects per-phase trace data for one full pipeline run.

    Thread safety: not considered — the pipeline is single-threaded.
    """

    def __init__(
        self,
        culture: str,
        dimension_key: str,
        params: dict,
        live_path: "str | Path | None" = None,
    ) -> None:
        self.run_id: str = str(uuid.uuid4())
        self.started_at: str = _now_iso()
        self.culture: str = culture
        self.dimension_key: str = dimension_key
        self.params: dict = params
        self._phases: list[PhaseTrace] = []
        self._current: PhaseTrace | None = None
        # Track wall-clock start of current phase (monotonic is not serialisable)
        self._phase_start_ts: float | None = None
        # If set, the trace is flushed to this path after every end_phase call
        self._live_path: Path | None = Path(live_path) if live_path else None

    # ------------------------------------------------------------------
    # Phase lifecycle
    # ------------------------------------------------------------------

    def start_phase(self, phase: str, inputs: dict) -> None:
        """Begin recording a new pipeline phase."""
        import time
        self._current = PhaseTrace(phase=phase, started_at=_now_iso(), input=inputs)
        self._phase_start_ts = time.monotonic()

    def end_phase(self, output: Any = None) -> None:
        """Finalise the current phase and append it to the trace list."""
        if self._current is None:
            return
        import time
        self._current.ended_at = _now_iso()
        if self._phase_start_ts is not None:
            self._current.duration_ms = round(
                (time.monotonic() - self._phase_start_ts) * 1000, 1
            )
        self._current.output = output
        self._phases.append(self._current)
        self._current = None
        self._phase_start_ts = None
        if self._live_path is not None:
            self.save(self._live_path)

    # ------------------------------------------------------------------
    # Recording helpers (called while a phase is active)
    # ------------------------------------------------------------------

    def increment_api_calls(self) -> None:
        if self._current:
            self._current.api_calls += 1

    def record_thinking(self, thinking: str) -> None:
        if self._current and thinking:
            self._current.thinking.append(thinking)

    def record_tool_call(
        self,
        tool: str,
        input_data: Any,
        tool_use_id: str | None = None,
        result: Any = None,
    ) -> None:
        if self._current is None:
            return
        self._current.tool_calls.append(
            {
                "tool": tool,
                "tool_use_id": tool_use_id,
                "input": input_data,
                "result": result,
            }
        )

    def update_last_tool_result(self, tool_use_id: str, result: Any) -> None:
        """Attach a result to the most recent matching tool call."""
        if self._current is None:
            return
        for call in reversed(self._current.tool_calls):
            if call.get("tool_use_id") == tool_use_id:
                call["result"] = result
                return
        # Fallback: update the last call regardless
        if self._current.tool_calls:
            self._current.tool_calls[-1]["result"] = result

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        if self._current is None:
            return
        self._current.usage["input_tokens"] += input_tokens
        self._current.usage["output_tokens"] += output_tokens

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        total_in = sum(p.usage["input_tokens"] for p in self._phases)
        total_out = sum(p.usage["output_tokens"] for p in self._phases)
        total_calls = sum(p.api_calls for p in self._phases)

        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "culture": self.culture,
            "dimension_key": self.dimension_key,
            "params": self.params,
            "phases": [p.to_dict() for p in self._phases],
            "summary": {
                "total_api_calls": total_calls,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_tokens": total_in + total_out,
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Content-block extraction utility (OpenAI-compatible)
# ---------------------------------------------------------------------------

def extract_trace_data(content: Any, tracer: "PipelineTracer") -> None:
    """
    No-op stub retained for call-site compatibility.

    Tool calls and usage are recorded directly in the pipeline modules
    (researcher, generator, verifier, agent) via tracer.record_tool_call()
    and tracer.add_usage(). This function is no longer needed but kept to
    avoid import errors if called from legacy paths.
    """
    pass
