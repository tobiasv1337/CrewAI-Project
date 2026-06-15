from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

from crewai.hooks import (
    ToolCallHookContext,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
    unregister_after_tool_call_hook,
    unregister_before_tool_call_hook,
)


DEFAULT_RUNS_DIR = Path("logs/crew_runs")
DEFAULT_PREVIEW_CHARS = 4000


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


def safe_jsonable(value: Any) -> Any:
    """Convert arbitrary hook payloads into JSON-serializable data."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): safe_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [safe_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return safe_jsonable(value.model_dump())
    return str(value)


def preview_text(value: Any, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


@dataclass
class ToolCallSummary:
    call_id: int
    tool_name: str
    tool_input: dict[str, Any]
    output_preview: str
    output_chars: int
    output_truncated: bool


class ToolTraceRecorder:
    """Persist tool-call traces for one CrewAI kickoff."""

    def __init__(
        self,
        *,
        query: str,
        student_context: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        logs_root: Path | str = DEFAULT_RUNS_DIR,
        run_id: str | None = None,
        trace_full: bool = False,
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
    ) -> None:
        self.query = query
        self.student_context = student_context or ""
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.run_id = run_id or make_run_id()
        self.run_dir = Path(logs_root) / self.run_id
        self.trace_full = trace_full
        self.preview_chars = preview_chars
        self.tool_calls: list[ToolCallSummary] = []
        self._pending: list[dict[str, Any]] = []
        self._next_call_id = 1
        self._installed = False

    @property
    def trace_path(self) -> Path:
        return self.run_dir / "trace.jsonl"

    @property
    def answer_path(self) -> Path:
        return self.run_dir / "answer.md"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.md"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def ordered_tool_calls(self) -> list[ToolCallSummary]:
        return sorted(self.tool_calls, key=lambda call: call.call_id)

    def install(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        register_before_tool_call_hook(self.before_tool_call)
        register_after_tool_call_hook(self.after_tool_call)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        unregister_before_tool_call_hook(self.before_tool_call)
        unregister_after_tool_call_hook(self.after_tool_call)
        self._installed = False

    def before_tool_call(self, context: ToolCallHookContext) -> bool | None:
        started_at = utc_now()
        call = {
            "call_id": self._next_call_id,
            "tool_name": context.tool_name,
            "tool_input": safe_jsonable(context.tool_input),
            "agent_role": getattr(context.agent, "role", None),
            "task_name": getattr(context.task, "name", None),
            "task_description": preview_text(getattr(context.task, "description", None), 500),
            "started_at": started_at,
            "started_perf": perf_counter(),
        }
        self._next_call_id += 1
        self._pending.append(call)
        return None

    def after_tool_call(self, context: ToolCallHookContext) -> str | None:
        finished_at = utc_now()
        pending = self._pop_pending(context.tool_name)
        result_text = "" if context.tool_result is None else str(context.tool_result)
        output_preview = preview_text(result_text, self.preview_chars)
        output_value = result_text if self.trace_full else output_preview
        output_truncated = len(result_text) > self.preview_chars
        record = {
            "event": "tool_call",
            "call_id": pending.get("call_id", self._next_call_id),
            "tool_name": context.tool_name,
            "tool_input": pending.get("tool_input", safe_jsonable(context.tool_input)),
            "output": output_value,
            "output_preview": output_preview,
            "output_chars": len(result_text),
            "output_truncated": output_truncated,
            "agent_role": pending.get("agent_role") or getattr(context.agent, "role", None),
            "task_name": pending.get("task_name") or getattr(context.task, "name", None),
            "task_description": pending.get("task_description")
            or preview_text(getattr(context.task, "description", None), 500),
            "started_at": pending.get("started_at"),
            "finished_at": finished_at,
            "duration_ms": self._duration_ms(pending),
        }
        self._append_jsonl(record)
        self.tool_calls.append(
            ToolCallSummary(
                call_id=int(record["call_id"]),
                tool_name=str(record["tool_name"]),
                tool_input=dict(record["tool_input"] or {}),
                output_preview=output_preview,
                output_chars=len(result_text),
                output_truncated=output_truncated,
            )
        )
        return None

    def write_answer(self, answer: str, *, usage_metrics: Any = None, state: Any = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.answer_path.write_text(answer, encoding="utf-8")
        self.report_path.write_text(self._build_markdown_report(answer), encoding="utf-8")
        state_path: str | None = None
        if state is not None:
            self.state_path.write_text(
                json.dumps(safe_jsonable(state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            state_path = str(self.state_path)
        summary = {
            "run_id": self.run_id,
            "query": self.query,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "answer_chars": len(answer),
            "report_path": str(self.report_path),
            "state_path": state_path,
            "tool_call_count": len(self.tool_calls),
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "tool_input": call.tool_input,
                    "output_chars": call.output_chars,
                    "output_truncated": call.output_truncated,
                }
                for call in self.ordered_tool_calls
            ],
            "usage_metrics": safe_jsonable(usage_metrics),
            "created_at": utc_now(),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_markdown_report(self, answer: str) -> str:
        lines = [
            "# Moses Agent Run Report",
            "",
            "## Prompt",
            "",
            "### User Query",
            "",
            self.query or "(empty)",
            "",
            "### Student Context",
            "",
            self.student_context or "No student context supplied.",
            "",
            "## Run Configuration",
            "",
            f"- Model: `{self.model or 'default'}`",
            f"- Temperature: `{self.temperature if self.temperature is not None else 'default'}`",
            f"- top_p: `{self.top_p if self.top_p is not None else 'default'}`",
            f"- Trace mode: `{'full tool outputs' if self.trace_full else 'preview tool outputs'}`",
            "",
            "## Observable Execution Trace",
            "",
            (
                "CrewAI does not expose private chain-of-thought here. This section "
                "shows the auditable behavior instead: tool calls, parameters, and "
                "tool results."
            ),
            "",
        ]
        if not self.tool_calls:
            lines.extend(["No tool calls were captured.", ""])
        for call in self.ordered_tool_calls:
            lines.extend(
                [
                    f"### {call.call_id}. {call.tool_name}",
                    "",
                    "**Parameters**",
                    "",
                    "```json",
                    json.dumps(call.tool_input, ensure_ascii=False, indent=2, sort_keys=True),
                    "```",
                    "",
                    "**Result**",
                    "",
                    "```text",
                    call.output_preview,
                    "```",
                    "",
                    f"Result length: {call.output_chars} chars"
                    + ("; result truncated in this report." if call.output_truncated and not self.trace_full else "."),
                    "",
                ]
            )
        lines.extend(
            [
                "## Final Answer",
                "",
                answer.rstrip(),
                "",
            ]
        )
        return "\n".join(lines)

    def compact_summary_lines(self) -> list[str]:
        if not self.tool_calls:
            return ["No tool calls were captured."]
        lines = []
        for call in self.ordered_tool_calls:
            args = json.dumps(call.tool_input, ensure_ascii=False, sort_keys=True)
            lines.append(
                f"{call.call_id}. {call.tool_name} args={args} -> "
                f"{call.output_chars} chars"
                + (" (truncated in trace)" if call.output_truncated and not self.trace_full else "")
            )
        return lines

    def _pop_pending(self, tool_name: str) -> dict[str, Any]:
        for index in range(len(self._pending) - 1, -1, -1):
            if self._pending[index].get("tool_name") == tool_name:
                return self._pending.pop(index)
        return {}

    def _duration_ms(self, pending: dict[str, Any]) -> int | None:
        started_perf = pending.get("started_perf")
        if started_perf is None:
            return None
        return round((perf_counter() - float(started_perf)) * 1000)

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_jsonable(record), ensure_ascii=False) + "\n")


class NullToolTraceRecorder:
    run_dir: Path | None = None
    tool_calls: list[ToolCallSummary] = []
    state_path: Path | None = None

    def write_answer(self, answer: str, *, usage_metrics: Any = None, state: Any = None) -> None:
        del answer, usage_metrics, state

    def compact_summary_lines(self) -> list[str]:
        return ["Tool tracing disabled."]


@contextmanager
def capture_tool_traces(
    *,
    enabled: bool,
    query: str,
    student_context: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    logs_root: Path | str = DEFAULT_RUNS_DIR,
    trace_full: bool = False,
    run_id: str | None = None,
) -> Iterator[ToolTraceRecorder | NullToolTraceRecorder]:
    if not enabled:
        yield NullToolTraceRecorder()
        return

    recorder = ToolTraceRecorder(
        query=query,
        student_context=student_context,
        model=model,
        temperature=temperature,
        top_p=top_p,
        logs_root=logs_root,
        run_id=run_id,
        trace_full=trace_full,
    )
    recorder.install()
    try:
        yield recorder
    finally:
        recorder.uninstall()
