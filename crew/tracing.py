from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import threading
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
_TRACE_EVENT_STATE = threading.local()
TRACE_SUPPRESSED_SOURCE_ATTRIBUTE = "_tu_study_trace_suppressed"


@contextmanager
def suppress_trace_events() -> Iterator[None]:
    """Keep auxiliary LLM calls from being attributed to the active crew."""
    previous = bool(getattr(_TRACE_EVENT_STATE, "suppressed", False))
    _TRACE_EVENT_STATE.suppressed = True
    try:
        yield
    finally:
        _TRACE_EVENT_STATE.suppressed = previous


def suppress_trace_events_from(source: Any) -> None:
    """Exclude events emitted by an auxiliary LLM from the user-facing trace.

    CrewAI dispatches event-bus handlers on worker threads, so the thread-local
    ``suppress_trace_events`` flag is not visible there. Auxiliary LLM instances
    are instead marked at their source and filtered by the recorder handlers.
    """
    try:
        setattr(source, TRACE_SUPPRESSED_SOURCE_ATTRIBUTE, True)
    except Exception:
        # Tracing must never make an auxiliary LLM call fail.
        return


def trace_events_suppressed_for(source: Any) -> bool:
    return bool(getattr(_TRACE_EVENT_STATE, "suppressed", False)) or bool(
        getattr(source, TRACE_SUPPRESSED_SOURCE_ATTRIBUTE, False)
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


def safe_jsonable(value: Any) -> Any:
    """Convert arbitrary hook payloads into JSON-serializable data."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value):
        return safe_jsonable(asdict(value))
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
    agent_role: str | None = None
    agent_label: str = "Unknown Agent"
    task_name: str | None = None
    duration_ms: int | None = None
    source_system: str = "Other"
    status: str = "ok"
    badges: list[str] = field(default_factory=list)
    output: str | None = None


@dataclass
class AgentTraceGroup:
    agent_label: str
    agent_role: str | None
    source_system: str
    tool_calls: list[ToolCallSummary] = field(default_factory=list)
    duration_ms: int = 0
    status: str = "ok"


@dataclass
class TraceWorkbench:
    run_id: str | None
    run_dir: str | None
    groups: list[AgentTraceGroup]
    total_tool_calls: int
    source_flow: list[dict[str, Any]]
    artifacts: dict[str, str | None]
    disabled: bool = False

    def model_dump(self) -> dict[str, Any]:
        return safe_jsonable(self)


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
        run_label: str = "Moses Agent Run Report",
        on_event: Callable[[dict[str, Any]], None] | None = None,
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
        self.run_label = run_label
        self.on_event = on_event
        self.tool_calls: list[ToolCallSummary] = []
        self._pending: list[dict[str, Any]] = []
        self._next_call_id = 1
        self._next_event_id = 1
        self._started_perf = perf_counter()
        self._event_lock = threading.Lock()
        self._trace_lock = threading.Lock()
        self._event_bus: Any | None = None
        self._event_bus_handlers: list[tuple[type[Any], Callable[..., Any]]] = []
        self._installed = False

    @property
    def trace_path(self) -> Path:
        return self.run_dir / "trace.jsonl"

    @property
    def events_path(self) -> Path:
        """Append-only lifecycle journal written while a run is executing."""
        return self.run_dir / "events.jsonl"

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

    def emit_event(self, event: dict[str, Any]) -> None:
        """Persist and forward one public runtime event immediately."""
        self._emit_event(event)

    @property
    def ordered_tool_calls(self) -> list[ToolCallSummary]:
        return sorted(self.tool_calls, key=lambda call: call.call_id)

    def install(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        register_before_tool_call_hook(self.before_tool_call)
        register_after_tool_call_hook(self.after_tool_call)
        self._install_event_bus_handlers()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        unregister_before_tool_call_hook(self.before_tool_call)
        unregister_after_tool_call_hook(self.after_tool_call)
        self._uninstall_event_bus_handlers()
        self._installed = False

    def before_tool_call(self, context: ToolCallHookContext) -> bool | None:
        started_at = utc_now()
        with self._trace_lock:
            call_id = self._next_call_id
            self._next_call_id += 1
        call = {
            "call_id": call_id,
            "tool_name": context.tool_name,
            "tool_input": safe_jsonable(context.tool_input),
            "agent_role": getattr(context.agent, "role", None),
            "task_name": getattr(context.task, "name", None),
            "task_description": preview_text(getattr(context.task, "description", None), 500),
            "started_at": started_at,
            "started_perf": perf_counter(),
            "thread_id": threading.get_ident(),
        }
        with self._trace_lock:
            self._pending.append(call)
        self._emit_event(
            {
                "event": "tool_start",
                "run_id": self.run_id,
                "call_id": call["call_id"],
                "tool_name": call["tool_name"],
                "tool_input": call["tool_input"],
                "agent_role": call["agent_role"],
                "agent_label": agent_label_for_role(call["agent_role"]),
                "task_name": call["task_name"],
                "source_system": source_system_for_tool(str(call["tool_name"])),
                "status": "running",
                "badges": [source_system_for_tool(str(call["tool_name"]))],
                "started_at": call["started_at"],
            }
        )
        return None

    def after_tool_call(self, context: ToolCallHookContext) -> str | None:
        finished_at = utc_now()
        pending = self._pop_pending(context.tool_name)
        result_text = "" if context.tool_result is None else str(context.tool_result)
        output_preview = preview_text(result_text, self.preview_chars)
        output_truncated = len(result_text) > self.preview_chars
        record = {
            "event": "tool_call",
            "call_id": pending.get("call_id", self._next_call_id),
            "tool_name": context.tool_name,
            "tool_input": pending.get("tool_input", safe_jsonable(context.tool_input)),
            "output": result_text,
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
        summary = ToolCallSummary(
            call_id=int(record["call_id"]),
            tool_name=str(record["tool_name"]),
            tool_input=dict(record["tool_input"] or {}),
            output_preview=output_preview,
            output_chars=len(result_text),
            output_truncated=output_truncated,
            agent_role=record.get("agent_role"),
            agent_label=agent_label_for_role(record.get("agent_role")),
            task_name=record.get("task_name"),
            duration_ms=record.get("duration_ms"),
            source_system=source_system_for_tool(str(record["tool_name"])),
            status=status_for_tool_output(str(record["tool_name"]), output_preview),
            badges=badges_for_tool_output(str(record["tool_name"]), output_preview),
            output=result_text,
        )
        with self._trace_lock:
            self._append_jsonl(record)
            self.tool_calls.append(summary)
        self._emit_event(
            {
                "event": "tool_finish",
                "run_id": self.run_id,
                "tool_call": safe_jsonable(summary),
                "workbench": self.workbench().model_dump(),
                "finished_at": finished_at,
            }
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
            "events_path": str(self.events_path),
            "state_path": state_path,
            "tool_call_count": len(self.tool_calls),
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "tool_input": call.tool_input,
                    "output_chars": call.output_chars,
                    "output_truncated": call.output_truncated,
                    "agent_role": call.agent_role,
                    "agent_label": call.agent_label,
                    "task_name": call.task_name,
                    "duration_ms": call.duration_ms,
                    "source_system": call.source_system,
                    "status": call.status,
                    "badges": call.badges,
                }
                for call in self.ordered_tool_calls
            ],
            "workbench": self.workbench().model_dump(),
            "usage_metrics": safe_jsonable(usage_metrics),
            "created_at": utc_now(),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_markdown_report(self, answer: str) -> str:
        lines = [
            f"# {self.run_label}",
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
                    call.output if (self.trace_full and call.output is not None) else call.output_preview,
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
                f"{call.call_id}. {call.agent_label} / {call.tool_name} [{call.status}] args={args} -> "
                f"{call.output_chars} chars"
                + (" (truncated in trace)" if call.output_truncated and not self.trace_full else "")
            )
        return lines

    def workbench(self) -> TraceWorkbench:
        return build_trace_workbench(
            self.ordered_tool_calls,
            run_id=self.run_id,
            run_dir=self.run_dir,
        )

    def _pop_pending(self, tool_name: str) -> dict[str, Any]:
        thread_id = threading.get_ident()
        with self._trace_lock:
            for index, item in enumerate(self._pending):
                if item.get("tool_name") == tool_name and item.get("thread_id") == thread_id:
                    return self._pending.pop(index)
            for index, item in enumerate(self._pending):
                if item.get("tool_name") == tool_name:
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

    def _emit_event(self, event: dict[str, Any]) -> None:
        if bool(getattr(_TRACE_EVENT_STATE, "suppressed", False)):
            return
        with self._event_lock:
            event.setdefault("event_id", self._next_event_id)
            self._next_event_id += 1
            event.setdefault("run_id", self.run_id)
            event.setdefault("emitted_at", utc_now())
            event.setdefault("elapsed_ms", round((perf_counter() - self._started_perf) * 1000))
            safe_event = safe_jsonable(event)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe_event, ensure_ascii=False) + "\n")
        if self.on_event is not None:
            try:
                self.on_event(safe_event)
            except Exception:
                # UI callbacks must never break a CrewAI run.
                return

    def _install_event_bus_handlers(self) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.events.types.crew_events import (
                CrewKickoffCompletedEvent,
                CrewKickoffFailedEvent,
                CrewKickoffStartedEvent,
            )
            from crewai.events.types.llm_events import (
                LLMCallCompletedEvent,
                LLMCallFailedEvent,
                LLMCallStartedEvent,
            )
            from crewai.events.types.task_events import (
                TaskCompletedEvent,
                TaskFailedEvent,
                TaskStartedEvent,
            )
            from crewai.events.types.tool_usage_events import (
                ToolUsageErrorEvent,
                ToolUsageFinishedEvent,
                ToolUsageStartedEvent,
            )
        except Exception:
            return

        handlers: list[tuple[type[Any], Callable[..., Any]]] = [
            (CrewKickoffStartedEvent, self._on_crew_started),
            (CrewKickoffCompletedEvent, self._on_crew_completed),
            (CrewKickoffFailedEvent, self._on_crew_failed),
            (TaskStartedEvent, self._on_task_started),
            (TaskCompletedEvent, self._on_task_completed),
            (TaskFailedEvent, self._on_task_failed),
            (LLMCallStartedEvent, self._on_llm_started),
            (LLMCallCompletedEvent, self._on_llm_completed),
            (LLMCallFailedEvent, self._on_llm_failed),
            (ToolUsageStartedEvent, self._on_tool_usage_started),
            (ToolUsageFinishedEvent, self._on_tool_usage_finished),
            (ToolUsageErrorEvent, self._on_tool_usage_error),
        ]
        for event_type, handler in handlers:
            crewai_event_bus.on(event_type)(handler)
        self._event_bus = crewai_event_bus
        self._event_bus_handlers = handlers

    def _uninstall_event_bus_handlers(self) -> None:
        if self._event_bus is None:
            return
        for event_type, handler in self._event_bus_handlers:
            try:
                self._event_bus.off(event_type, handler)
            except Exception:
                continue
        self._event_bus = None
        self._event_bus_handlers = []

    def _on_crew_started(self, source: Any, event: Any) -> None:
        self._emit_event(
            {
                "event": "crew_started",
                "crew_name": getattr(event, "crew_name", None) or getattr(source, "name", None),
                "phase": "kickoff",
                "status": "running",
                "activity": "Crew kickoff started. The orchestrator is reading the request and planning delegation.",
            }
        )

    def _on_crew_completed(self, source: Any, event: Any) -> None:
        self._emit_event(
            {
                "event": "crew_completed",
                "crew_name": getattr(event, "crew_name", None) or getattr(source, "name", None),
                "phase": "answer",
                "status": "ok",
                "activity": "Crew completed. The orchestrator produced the final answer.",
                "total_tokens": getattr(event, "total_tokens", None),
            }
        )

    def _on_crew_failed(self, source: Any, event: Any) -> None:
        self._emit_event(
            {
                "event": "crew_failed",
                "crew_name": getattr(event, "crew_name", None) or getattr(source, "name", None),
                "phase": "failed",
                "status": "error",
                "activity": "Crew failed before a final answer was produced.",
                "error": str(getattr(event, "error", "") or ""),
            }
        )

    def _on_task_started(self, source: Any, event: Any) -> None:
        task = getattr(event, "task", None) or source
        agent_role = _agent_role_from_task(task)
        self._emit_event(
            {
                "event": "task_started",
                "agent_role": agent_role,
                "agent_label": agent_label_for_role(agent_role) if agent_role else "Orchestrator",
                "task_name": _task_name(task),
                "phase": "task",
                "status": "running",
                "activity": f"Task started: {_task_name(task)}",
            }
        )

    def _on_task_completed(self, source: Any, event: Any) -> None:
        task = getattr(event, "task", None) or source
        agent_role = _agent_role_from_task(task)
        output = getattr(event, "output", None)
        self._emit_event(
            {
                "event": "task_completed",
                "agent_role": agent_role,
                "agent_label": agent_label_for_role(agent_role) if agent_role else "Orchestrator",
                "task_name": _task_name(task),
                "phase": "task",
                "status": "ok",
                "activity": f"Task completed: {_task_name(task)}",
                "output_preview": preview_text(getattr(output, "raw", output), 500),
            }
        )

    def _on_task_failed(self, source: Any, event: Any) -> None:
        task = getattr(event, "task", None) or source
        agent_role = _agent_role_from_task(task)
        self._emit_event(
            {
                "event": "task_failed",
                "agent_role": agent_role,
                "agent_label": agent_label_for_role(agent_role) if agent_role else "Orchestrator",
                "task_name": _task_name(task),
                "phase": "task",
                "status": "error",
                "activity": f"Task failed: {_task_name(task)}",
                "error": str(getattr(event, "error", "") or ""),
            }
        )

    def _on_llm_started(self, source: Any, event: Any) -> None:
        if trace_events_suppressed_for(source):
            return
        agent_role = getattr(event, "agent_role", None)
        label = agent_label_for_role(agent_role)
        if label == "Unknown Agent":
            label = "Orchestrator"
        tools = getattr(event, "tools", None) or []
        tool_choices = [
            str(item.get("function", {}).get("name") or item.get("name") or "")
            for item in tools
            if isinstance(item, dict)
        ]
        task_name = getattr(event, "task_name", None)
        call_id = getattr(event, "call_id", None)
        self._emit_event(
            {
                "event": "llm_started",
                "agent_role": agent_role,
                "agent_label": label,
                "call_id": call_id,
                "model": getattr(event, "model", None),
                "task_name": task_name,
                "phase": "llm",
                "status": "running",
                "activity": (
                    f"{label} started an LLM reasoning step with "
                    f"{len(tool_choices)} available tool schema{'s' if len(tool_choices) != 1 else ''}."
                ),
                "tool_choices": tool_choices,
                "tools_count": len(tools),
            }
        )

    def _on_llm_completed(self, source: Any, event: Any) -> None:
        if trace_events_suppressed_for(source):
            return
        agent_role = getattr(event, "agent_role", None)
        label = agent_label_for_role(agent_role)
        if label == "Unknown Agent":
            label = "Orchestrator"
        self._emit_event(
            {
                "event": "llm_completed",
                "agent_role": agent_role,
                "agent_label": label,
                "call_id": getattr(event, "call_id", None),
                "model": getattr(event, "model", None),
                "task_name": getattr(event, "task_name", None),
                "phase": "llm",
                "status": "ok",
                "activity": "LLM step completed.",
                "usage": getattr(event, "usage", None),
                "finish_reason": getattr(event, "finish_reason", None),
            }
        )

    def _on_llm_failed(self, source: Any, event: Any) -> None:
        if trace_events_suppressed_for(source):
            return
        agent_role = getattr(event, "agent_role", None)
        label = agent_label_for_role(agent_role)
        if label == "Unknown Agent":
            label = "Orchestrator"
        self._emit_event(
            {
                "event": "llm_failed",
                "agent_role": agent_role,
                "agent_label": label,
                "call_id": getattr(event, "call_id", None),
                "model": getattr(event, "model", None),
                "task_name": getattr(event, "task_name", None),
                "phase": "llm",
                "status": "error",
                "activity": "LLM step failed.",
                "error": str(getattr(event, "error", "") or ""),
            }
        )

    def _on_tool_usage_started(self, source: Any, event: Any) -> None:
        self._emit_event(_tool_usage_event_dict(event, status="running", activity="Tool selected and execution started."))

    def _on_tool_usage_finished(self, source: Any, event: Any) -> None:
        data = _tool_usage_event_dict(event, status="ok", activity="Tool execution finished.")
        started_at = getattr(event, "started_at", None)
        finished_at = getattr(event, "finished_at", None)
        if started_at and finished_at:
            try:
                data["duration_ms"] = round((finished_at - started_at).total_seconds() * 1000)
            except Exception:
                pass
        data["output_preview"] = preview_text(getattr(event, "output", None), 500)
        self._emit_event(data)

    def _on_tool_usage_error(self, source: Any, event: Any) -> None:
        data = _tool_usage_event_dict(event, status="error", activity="Tool execution failed.")
        data["error"] = str(getattr(event, "error", "") or "")
        self._emit_event(data)


def _agent_role_from_task(task: Any) -> str | None:
    agent = getattr(task, "agent", None)
    role = getattr(agent, "role", None)
    return str(role) if role else None


def _task_name(task: Any) -> str:
    name = getattr(task, "name", None)
    if name:
        return str(name)
    description = str(getattr(task, "description", "") or "").strip()
    if not description:
        return "Unnamed task"
    return preview_text(description.replace("\n", " "), 80)


def _tool_usage_event_dict(event: Any, *, status: str, activity: str) -> dict[str, Any]:
    tool_name = str(getattr(event, "tool_name", "") or "Unknown Tool")
    agent_role = getattr(event, "agent_role", None)
    label = agent_label_for_role(agent_role)
    if label == "Unknown Agent" and "coworker" in tool_name.casefold():
        label = "Orchestrator"
    source_system = source_system_for_tool(tool_name)
    return {
        "event": f"tool_usage_{status}",
        "agent_role": str(agent_role) if agent_role else None,
        "agent_label": label,
        "task_name": getattr(event, "task_name", None),
        "tool_name": tool_name,
        "tool_input": safe_jsonable(getattr(event, "tool_args", None)),
        "source_system": source_system,
        "phase": "tool",
        "status": status,
        "activity": activity,
        "badges": [source_system],
    }


class NullToolTraceRecorder:
    run_dir: Path | None = None
    tool_calls: list[ToolCallSummary] = []
    state_path: Path | None = None

    def emit_event(self, event: dict[str, Any]) -> None:
        del event

    def write_answer(self, answer: str, *, usage_metrics: Any = None, state: Any = None) -> None:
        del answer, usage_metrics, state

    def compact_summary_lines(self) -> list[str]:
        return ["Tool tracing disabled."]

    def workbench(self) -> TraceWorkbench:
        return TraceWorkbench(
            run_id=None,
            run_dir=None,
            groups=[],
            total_tool_calls=0,
            source_flow=source_flow_for_calls([]),
            artifacts={},
            disabled=True,
        )


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
    run_label: str = "Moses Agent Run Report",
    on_event: Callable[[dict[str, Any]], None] | None = None,
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
        run_label=run_label,
        on_event=on_event,
    )
    recorder.install()
    try:
        yield recorder
    finally:
        recorder.uninstall()


def build_trace_workbench(
    tool_calls: list[ToolCallSummary],
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
) -> TraceWorkbench:
    groups_by_label: dict[str, AgentTraceGroup] = {}
    for call in tool_calls:
        group = groups_by_label.get(call.agent_label)
        if group is None:
            group = AgentTraceGroup(
                agent_label=call.agent_label,
                agent_role=call.agent_role,
                source_system=call.source_system,
            )
            groups_by_label[call.agent_label] = group
        group.tool_calls.append(call)
        if call.duration_ms:
            group.duration_ms += call.duration_ms
        group.status = combine_status(group.status, call.status)

    for group in groups_by_label.values():
        if group.status == "error":
            group.status = "ok"

    ordered_labels = [
        "Orchestrator",
        "Study Advisor",
        "Grade Optimization Specialist",
        "MOSES Module Researcher",
        "Degree Regulations Specialist",
        "ISIS Course Info Specialist",
        "Course Commitment Specialist",
        "Unknown Agent",
    ]
    groups = sorted(
        groups_by_label.values(),
        key=lambda item: (
            ordered_labels.index(item.agent_label) if item.agent_label in ordered_labels else len(ordered_labels),
            item.agent_label,
        ),
    )
    run_dir_text = str(run_dir) if run_dir is not None else None
    artifacts = {
        "report": str(Path(run_dir) / "report.md") if run_dir is not None else None,
        "trace": str(Path(run_dir) / "trace.jsonl") if run_dir is not None else None,
        "events": str(Path(run_dir) / "events.jsonl") if run_dir is not None else None,
        "state": str(Path(run_dir) / "state.json") if run_dir is not None else None,
        "summary": str(Path(run_dir) / "summary.json") if run_dir is not None else None,
    }
    return TraceWorkbench(
        run_id=run_id,
        run_dir=run_dir_text,
        groups=groups,
        total_tool_calls=len(tool_calls),
        source_flow=source_flow_for_calls(tool_calls),
        artifacts=artifacts,
    )


def load_trace_workbench(run_dir: Path | str) -> TraceWorkbench:
    path = Path(run_dir)
    trace_path = path / "trace.jsonl"
    calls: list[ToolCallSummary] = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            calls.append(tool_call_summary_from_event(event))
    return build_trace_workbench(calls, run_id=path.name, run_dir=path)


def tool_call_summary_from_event(event: dict[str, Any]) -> ToolCallSummary:
    tool_name = str(event.get("tool_name") or "Unknown Tool")
    output_preview = str(event.get("output_preview") or event.get("output") or "")
    agent_role = event.get("agent_role")
    return ToolCallSummary(
        call_id=int(event.get("call_id") or 0),
        tool_name=tool_name,
        tool_input=dict(event.get("tool_input") or {}),
        output_preview=output_preview,
        output_chars=int(event.get("output_chars") or len(output_preview)),
        output_truncated=bool(event.get("output_truncated")),
        agent_role=str(agent_role) if agent_role else None,
        agent_label=agent_label_for_role(agent_role),
        task_name=event.get("task_name"),
        duration_ms=event.get("duration_ms"),
        source_system=source_system_for_tool(tool_name),
        status=status_for_tool_output(tool_name, output_preview),
        badges=badges_for_tool_output(tool_name, output_preview),
        output=event.get("output"),
    )


def agent_label_for_role(role: object | None) -> str:
    text = str(role or "").casefold()
    if "orchestrator" in text:
        return "Orchestrator"
    if "study advisor" in text or "personal study advisor" in text:
        return "Study Advisor"
    if "grade optimization" in text or "grade optimizer" in text:
        return "Grade Optimization Specialist"
    if "moses" in text or "module researcher" in text:
        return "MOSES Module Researcher"
    if "degree regulations" in text or "regulations specialist" in text or "stupo" in text:
        return "Degree Regulations Specialist"
    if "isis" in text or "course information specialist" in text:
        return "ISIS Course Info Specialist"
    if "commitment" in text or "course commitment" in text:
        return "Course Commitment Specialist"
    return "Unknown Agent"


def source_system_for_tool(tool_name: str) -> str:
    text = tool_name.casefold()
    if "grade scenario" in text or "grade sensitivity" in text or "target grade" in text or "target-grade" in text:
        return "Grade Optimization"
    if "study plan" in text or "degree requirement" in text:
        return "Grade Manager"
    if "propose course actions" in text or "confirmation" in text or "commitment" in text:
        return "Course Commitment"
    if "degree regulation" in text or "regelstudienplan" in text or "stupo" in text:
        return "Degree Regulations"
    if "moses" in text:
        return "MOSES"
    if "isis" in text:
        return "ISIS"
    return "Other"


def status_for_tool_output(tool_name: str, output_preview: str) -> str:
    text = f"{tool_name}\n{output_preview}".casefold()
    error_markers = (
        "failed",
        "error",
        "could not",
        "write refused",
        "invalid",
    )
    warning_markers = (
        "access required",
        "requires enrollment key",
        "ambiguous",
        "not found",
        "no modules",
        "no matching",
        "denied access",
    )
    if any(marker in text for marker in error_markers):
        return "error"
    if any(marker in text for marker in warning_markers):
        return "warning"
    return "ok"


def badges_for_tool_output(tool_name: str, output_preview: str) -> list[str]:
    text = f"{tool_name}\n{output_preview}".casefold()
    badges: list[str] = [source_system_for_tool(tool_name)]
    if "temporarily enrolled by this tool call: yes" in text:
        badges.append("temporary enrollment")
    if "cleanup attempted: yes; succeeded: yes" in text:
        badges.append("cleanup succeeded")
    if "cleanup attempted: yes; succeeded: no" in text or "cleanup error" in text:
        badges.append("cleanup failed")
    if "requires enrollment key" in text or "enrollment key required" in text:
        badges.append("key required")
    if "write refused" in text:
        badges.append("write refused")
    if "add module to study plan" in tool_name.casefold() and "added `" in text:
        badges.append("study-plan write")
    if "propose course actions" in tool_name.casefold():
        badges.append("ui proposal")
    return _dedupe_strings(badges)


def source_flow_for_calls(tool_calls: list[ToolCallSummary]) -> list[dict[str, Any]]:
    active_sources = {call.source_system for call in tool_calls}
    return [
        {
            "source": "Grade Manager",
            "agent": "Study Advisor",
            "target": "Orchestrator",
            "active": "Grade Manager" in active_sources,
        },
        {
            "source": "Grade Optimization",
            "agent": "Grade Optimization Specialist",
            "target": "Orchestrator",
            "active": "Grade Optimization" in active_sources,
        },
        {
            "source": "MOSES",
            "agent": "MOSES Module Researcher",
            "target": "Orchestrator",
            "active": "MOSES" in active_sources,
        },
        {
            "source": "Degree Regulations",
            "agent": "Degree Regulations Specialist",
            "target": "Orchestrator",
            "active": "Degree Regulations" in active_sources,
        },
        {
            "source": "ISIS",
            "agent": "ISIS Course Info Specialist",
            "target": "Orchestrator",
            "active": "ISIS" in active_sources,
        },
        {
            "source": "Course Commitment",
            "agent": "Course Commitment Specialist",
            "target": "Student",
            "active": "Course Commitment" in active_sources,
        },
        {
            "source": "Orchestrator",
            "agent": "Final Answer",
            "target": "Student",
            "active": bool(tool_calls),
        },
    ]


def combine_status(left: str, right: str) -> str:
    order = {"ok": 0, "warning": 1, "error": 2}
    return right if order.get(right, 0) > order.get(left, 0) else left


def _dedupe_strings(values: list[str]) -> list[str]:
    result = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
