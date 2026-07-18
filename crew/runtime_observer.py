from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from crew.config.llm import (
    get_default_llm,
    resolve_study_assistant_observer_model,
    resolve_study_assistant_observer_timeout,
)
from crew.tracing import suppress_trace_events, suppress_trace_events_from


OBSERVER_EVENT_LIMIT = 10
OBSERVER_DELEGATION_LIMIT = 8
OBSERVER_IGNORED_EVENTS = {
    "agent_ready",
    "heartbeat",
    "llm_stream_chunk",
    "observer_started",
    "observer_progress",
    "observer_failed",
}

OBSERVER_AGENT_NAMES = (
    "Orchestrator",
    "Study Advisor",
    "Grade Optimization Specialist",
    "MOSES Module Researcher",
    "Degree Regulations Specialist",
    "ISIS Course Info Specialist",
    "Course Commitment Specialist",
)

_RAW_RUNTIME_MARKERS = (
    "llm call",
    "tool execution",
    "tool schema",
    "input keys",
    "call id",
    "crewai runtime",
    "crewai event",
    "exposed schemas",
)

_GERMAN_MARKERS = {
    "der",
    "die",
    "das",
    "den",
    "dem",
    "ich",
    "mein",
    "meine",
    "und",
    "wie",
    "kann",
    "möchte",
    "würde",
    "nicht",
    "note",
    "masterarbeit",
}
_ENGLISH_MARKERS = {
    "the",
    "this",
    "that",
    "student",
    "currently",
    "planning",
    "providing",
    "determining",
    "running",
    "request",
    "analysis",
}


class AgentProgressReport(BaseModel):
    """User-facing interpretation of one evidenced agent's current contribution."""

    agent: str = Field(description="Exact canonical agent name from agent_states.")
    state: Literal["active", "waiting", "completed", "blocked"] = Field(
        description="Current evidenced state of this agent."
    )
    summary: str = Field(
        description=(
            "One concise status sentence describing only this agent's evidenced current investigation "
            "or completed finding. Never answer the student's request or advise the student."
        )
    )


class RuntimeObserverReport(BaseModel):
    """Grounded global and per-agent interpretation of cumulative runtime state."""

    headline: str = Field(
        description=(
            "Concrete 3-8 word execution-status headline naming active or completed work; "
            "never a title for an answer or recommendation. Whole-run completion is legal only "
            "when run_stage is complete."
        )
    )
    detail: str = Field(
        description=(
            "At most three concise status sentences covering evidenced completed work, current agent work, "
            "and dependencies. Distinguish returned specialist responses from still-active work. Do not answer "
            "the student, recommend actions, or address the student directly."
        )
    )
    active_agent: str | None = Field(
        default=None,
        description="Primary active agent when one dominates, otherwise null.",
    )
    agent_updates: list[AgentProgressReport] = Field(
        default_factory=list,
        description="Exactly one update for every invoked agent supplied in agent_states.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="One to three compact internal trace facts supporting the report; not user-facing copy.",
    )


@dataclass(frozen=True)
class RuntimeObserverResult:
    model: str
    report: RuntimeObserverReport


def _bounded_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _canonical_agent_name(value: Any) -> str | None:
    raw = " ".join(str(value or "").replace("\n", " ").split())
    folded = raw.casefold()
    if not folded:
        return None
    if "runtime observer" in folded:
        return None
    if "orchestrator" in folded or "study assistant manager" in folded:
        return "Orchestrator"
    if "grade optimization" in folded:
        return "Grade Optimization Specialist"
    if "study advisor" in folded or "grade manager" in folded:
        return "Study Advisor"
    if "moses" in folded:
        return "MOSES Module Researcher"
    if "degree regulation" in folded or "stu po" in folded or "stupo" in folded:
        return "Degree Regulations Specialist"
    if "isis" in folded or "moodle" in folded:
        return "ISIS Course Info Specialist"
    if "course commitment" in folded or "commitment specialist" in folded:
        return "Course Commitment Specialist"
    return raw if raw in OBSERVER_AGENT_NAMES else None


def _event_agent_name(event: dict[str, Any]) -> str | None:
    call = event.get("tool_call") if isinstance(event.get("tool_call"), dict) else {}
    return _canonical_agent_name(
        event.get("agent_label")
        or call.get("agent_label")
        or event.get("agent_role")
        or call.get("agent_role")
    )


def _request_language(request: str) -> Literal["German", "English"]:
    words = set(re.findall(r"[a-zäöüß]+", request.casefold()))
    return "German" if len(words & _GERMAN_MARKERS) >= 2 else "English"


def _semantic_tool_input(value: Any, *, depth: int = 0) -> Any:
    """Keep useful tool parameters while redacting credentials and bounding context."""
    if depth > 2:
        return _bounded_text(value, 240)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:12]:
            name = str(key)
            if any(secret in name.casefold() for secret in ("password", "token", "secret", "cookie", "api_key")):
                result[name] = "[redacted]"
            else:
                result[name] = _semantic_tool_input(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_semantic_tool_input(item, depth=depth + 1) for item in list(value)[:12]]
    if isinstance(value, (str, bytes)):
        return _bounded_text(value, 360)
    return value


def _normalized_event(event: dict[str, Any]) -> dict[str, Any]:
    event_name = str(event.get("event") or "unknown")
    call = event.get("tool_call") if event_name == "tool_finish" else None
    call = call if isinstance(call, dict) else {}
    tool_input = call.get("tool_input") or event.get("tool_input") or {}
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    normalized = {
        "event": event_name,
        "agent": _event_agent_name(event),
        "status": call.get("status") or event.get("status") or "unknown",
        "activity": _bounded_text(event.get("activity"), 220),
        "tool": call.get("tool_name") or event.get("tool_name"),
        "source": call.get("source_system") or event.get("source_system"),
        "task": call.get("task_name") or event.get("task_name"),
        "goal": _bounded_text(
            call.get("task_description") or event.get("task_description"),
            520,
        ),
        "tool_input": _semantic_tool_input(tool_input) if tool_input else None,
        "output_preview": _bounded_text(
            call.get("output_preview") or event.get("output_preview"),
            600,
        ),
    }
    if event_name == "tool_start":
        normalized["delegated_to"] = tool_input.get("coworker")
        normalized["delegated_task"] = _bounded_text(tool_input.get("task"), 260)
    return {key: value for key, value in normalized.items() if value not in (None, "", [])}


def _delegation_snapshot(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    delegations: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, event in enumerate(events):
        event_name = str(event.get("event") or "")
        call = event.get("tool_call") if isinstance(event.get("tool_call"), dict) else {}
        tool_name = str(call.get("tool_name") or event.get("tool_name") or "")
        if "coworker" not in tool_name.casefold():
            continue
        call_id = str(call.get("call_id") or event.get("call_id") or f"delegation-{index}")
        tool_input = call.get("tool_input") or event.get("tool_input") or {}
        tool_input = tool_input if isinstance(tool_input, dict) else {}
        if call_id not in delegations:
            receiver = _canonical_agent_name(tool_input.get("coworker"))
            if receiver is None:
                continue
            delegations[call_id] = {
                "sender": _event_agent_name(event) or "Orchestrator",
                "receiver": receiver,
                "request": _bounded_text(
                    tool_input.get("task") or tool_input.get("question") or tool_input.get("request"),
                    900,
                ),
                "status": "active",
            }
            order.append(call_id)
        if event_name == "tool_finish":
            delegations[call_id].update(
                {
                    "status": call.get("status") or "completed",
                    "response": _bounded_text(call.get("output") or call.get("output_preview"), 1400),
                }
            )
    return [delegations[key] for key in order[-OBSERVER_DELEGATION_LIMIT:]]


def _agent_state_snapshot(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce noisy lifecycle events to one authoritative state per invoked agent."""
    states: dict[str, dict[str, Any]] = {}
    open_delegations: dict[str, str] = {}

    def ensure(agent: str) -> dict[str, Any]:
        return states.setdefault(
            agent,
            {
                "agent": agent,
                "state": "waiting",
                "goal": "",
                "current_activity": "",
                "latest_result": "",
                "response_ready": False,
            },
        )

    def has_open_delegation(agent: str) -> bool:
        return agent in open_delegations.values()

    for index, event in enumerate(events):
        event_name = str(event.get("event") or "")
        normalized = _normalized_event(event)
        call = event.get("tool_call") if isinstance(event.get("tool_call"), dict) else {}
        tool_name = str(call.get("tool_name") or event.get("tool_name") or "")
        tool_input = call.get("tool_input") or event.get("tool_input") or {}
        tool_input = tool_input if isinstance(tool_input, dict) else {}
        receiver = (
            _canonical_agent_name(tool_input.get("coworker"))
            if "coworker" in tool_name.casefold()
            else None
        )
        delegation_id = str(
            call.get("call_id")
            or event.get("call_id")
            or f"delegation-{index}"
        )

        if receiver and event_name == "tool_start":
            open_delegations[delegation_id] = receiver

        agent = normalized.get("agent")
        if agent:
            state = ensure(str(agent))
            if normalized.get("goal"):
                state["goal"] = normalized["goal"]
            if normalized.get("activity"):
                state["current_activity"] = normalized["activity"]
            if normalized.get("output_preview"):
                state["latest_result"] = normalized["output_preview"]

            if event_name in {
                "ui_run_started",
                "intent_classified",
                "route_execution_started",
                "crew_started",
                "task_started",
                "llm_started",
                "tool_start",
            }:
                state["state"] = "active"
            elif event_name == "task_completed":
                # A delegated task is not user-visible as complete until the
                # enclosing A2A tool emits its response-bearing tool_finish.
                if not has_open_delegation(str(agent)):
                    state["state"] = "completed"
            elif event_name in {"task_failed", "llm_failed"}:
                state["state"] = "blocked"

        if receiver:
            receiver_state = ensure(receiver)
            request = _bounded_text(
                tool_input.get("task") or tool_input.get("question") or tool_input.get("request"),
                620,
            )
            if request:
                receiver_state["goal"] = request
            if event_name == "tool_finish":
                open_delegations.pop(delegation_id, None)
                status = str(call.get("status") or "ok")
                result = _bounded_text(
                    call.get("output")
                    or call.get("output_preview")
                    or event.get("output")
                    or event.get("output_preview"),
                    800,
                )
                if result:
                    receiver_state["latest_result"] = result
                receiver_state["response_ready"] = bool(result)
                if has_open_delegation(receiver):
                    receiver_state["state"] = "active"
                else:
                    receiver_state["state"] = (
                        "completed" if status in {"ok", "completed", "success"} else "blocked"
                    )
            elif event_name in {"tool_start", "tool_usage_running"}:
                # Completion acknowledgements such as tool_usage_ok follow
                # tool_finish and must never reopen an already returned A2A call.
                receiver_state["state"] = "active"
                receiver_state["response_ready"] = False

    run_completed = any(event.get("event") == "crew_completed" for event in events)
    run_failed = any(event.get("event") in {"crew_failed", "ui_error"} for event in events)
    if run_completed:
        for state in states.values():
            state["state"] = "completed"
    elif run_failed and "Orchestrator" in states:
        states["Orchestrator"]["state"] = "blocked"

    return [
        {key: value for key, value in states[agent].items() if value not in (None, "", [])}
        for agent in OBSERVER_AGENT_NAMES
        if agent in states
    ]


def _compact_previous_report(report: Any) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    updates = []
    for item in report.get("agent_updates") or []:
        if not isinstance(item, dict):
            continue
        agent = _canonical_agent_name(item.get("agent"))
        summary = _bounded_text(item.get("summary"), 260)
        if agent and summary:
            updates.append({"agent": agent, "summary": summary})
    return {
        "headline": _bounded_text(report.get("headline"), 100),
        "detail": _bounded_text(report.get("detail"), 500),
        "agent_updates": updates,
    }


def runtime_observer_snapshot(
    events: list[dict[str, Any]],
    *,
    study_context: str | None = None,
) -> dict[str, Any]:
    """Build bounded cumulative state from trace, A2A history, and the prior observer report."""
    previous_report = next(
        (
            event.get("report")
            for event in reversed(events)
            if event.get("event") == "observer_progress" and isinstance(event.get("report"), dict)
        ),
        None,
    )
    relevant = [
        event
        for event in events
        if str(event.get("event") or "") not in OBSERVER_IGNORED_EVENTS
    ]
    intent = next(
        (
            event.get("intent")
            for event in reversed(relevant)
            if event.get("event") == "intent_classified" and isinstance(event.get("intent"), dict)
        ),
        None,
    )
    request = next(
        (
            _bounded_text(event.get("query"), 1200)
            for event in relevant
            if event.get("event") == "ui_run_started" and event.get("query")
        ),
        "",
    )
    agent_states = _agent_state_snapshot(relevant)
    run_completed = any(event.get("event") == "crew_completed" for event in relevant)
    run_failed = any(event.get("event") in {"crew_failed", "ui_error"} for event in relevant)
    active_agents = [
        str(item.get("agent"))
        for item in agent_states
        if item.get("state") == "active"
    ]
    completed_agents = [
        str(item.get("agent"))
        for item in agent_states
        if item.get("state") == "completed"
    ]
    blocked_agents = [
        str(item.get("agent"))
        for item in agent_states
        if item.get("state") == "blocked"
    ]
    active_specialists = [agent for agent in active_agents if agent != "Orchestrator"]
    completed_specialists = [agent for agent in completed_agents if agent != "Orchestrator"]
    if run_completed:
        stage = "complete"
    elif run_failed:
        stage = "failed"
    elif active_specialists:
        stage = "execution"
    elif completed_specialists and "Orchestrator" in active_agents:
        stage = "synthesis"
    elif any(item.get("agent") != "Orchestrator" for item in agent_states):
        stage = "execution"
    else:
        stage = "intake"
    delegations = _delegation_snapshot(relevant)
    return {
        "student_request": request,
        "output_language": _request_language(request),
        "run_stage": stage,
        "study_context": _bounded_text(study_context, 3200) if study_context else None,
        "intent": intent,
        "previous_observer_report": _compact_previous_report(previous_report),
        "agent_states": agent_states,
        "lifecycle": {
            "crew_completed": run_completed,
            "crew_failed": run_failed,
            "active_agents": active_agents,
            "completed_agents": completed_agents,
            "blocked_agents": blocked_agents,
            "responses_available_for": [
                str(item.get("agent"))
                for item in agent_states
                if item.get("response_ready") is True
            ],
        },
        "a2a_delegations": delegations,
        "recent_changes": [_normalized_event(event) for event in relevant[-OBSERVER_EVENT_LIMIT:]],
    }


def _split_sentences(value: Any) -> list[str]:
    text = " ".join(str(value or "").split())
    if not text:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", text)
        if sentence.strip()
    ]


def _contains_raw_runtime_copy(value: str) -> bool:
    folded = value.casefold()
    if any(marker in folded for marker in _RAW_RUNTIME_MARKERS):
        return True
    return bool(re.search(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b", value))


def _german_fallback_summary(agent: str, state: str) -> str:
    active = state == "active"
    summaries = {
        "Orchestrator": (
            "Koordiniert die Fachanalysen und führt ihre Ergebnisse zu einer Antwort zusammen.",
            "Hat die Fachanalysen koordiniert und ihre Ergebnisse in der finalen Antwort zusammengeführt.",
        ),
        "Study Advisor": (
            "Prüft den aktuellen Studienstand, offene Leistungen und den realistischen Abschlusszeitplan.",
            "Hat Studienstand, offene Leistungen und den Abschlusszeitplan geprüft.",
        ),
        "Grade Optimization Specialist": (
            "Vergleicht Notenszenarien und berechnet den Einfluss von Kurs- und Abschlussarbeitsnoten.",
            "Hat Notenszenarien und den Einfluss von Kurs- und Abschlussarbeitsnoten berechnet.",
        ),
        "MOSES Module Researcher": (
            "Prüft relevante Module, Lehrveranstaltungen und Prüfungsinformationen im MOSES-Katalog.",
            "Hat relevante Modul-, Lehrveranstaltungs- und Prüfungsinformationen geprüft.",
        ),
        "Degree Regulations Specialist": (
            "Prüft die verbindlichen Studien- und Prüfungsregeln für Abschluss, Gewichtung und Fristen.",
            "Hat die verbindlichen Regeln für Abschluss, Gewichtung und Fristen geprüft.",
        ),
        "ISIS Course Info Specialist": (
            "Prüft aktuelle Kursräume, Termine und Aufgaben in ISIS.",
            "Hat die relevanten ISIS-Kursinformationen geprüft.",
        ),
        "Course Commitment Specialist": (
            "Prüft mögliche Änderungen am Studienplan und bereitet zustimmungspflichtige Schritte vor.",
            "Hat mögliche Studienplanänderungen und zustimmungspflichtige Schritte geprüft.",
        ),
    }
    current, finished = summaries[agent]
    return current if active else finished


def _english_fallback_summary(agent: str, state: str) -> str:
    active = state == "active"
    summaries = {
        "Orchestrator": (
            "Coordinates the specialist analyses and combines their findings into one answer.",
            "Coordinated the specialist analyses and combined their findings in the final answer.",
        ),
        "Study Advisor": (
            "Reviews the current study record, outstanding work, and realistic completion timeline.",
            "Reviewed the study record, outstanding work, and completion timeline.",
        ),
        "Grade Optimization Specialist": (
            "Compares grade scenarios and calculates the impact of coursework and thesis outcomes.",
            "Calculated grade scenarios and the impact of coursework and thesis outcomes.",
        ),
        "MOSES Module Researcher": (
            "Checks relevant modules, courses, and assessment details in the MOSES catalog.",
            "Checked the relevant module, course, and assessment details.",
        ),
        "Degree Regulations Specialist": (
            "Checks the binding degree rules for completion, grade weighting, and deadlines.",
            "Checked the binding degree rules for completion, grade weighting, and deadlines.",
        ),
        "ISIS Course Info Specialist": (
            "Checks current ISIS course spaces, deadlines, and assignments.",
            "Checked the relevant ISIS course information.",
        ),
        "Course Commitment Specialist": (
            "Reviews possible study-plan changes and prepares steps that require approval.",
            "Reviewed possible study-plan changes and approval-dependent steps.",
        ),
    }
    current, finished = summaries[agent]
    return current if active else finished


def _fallback_detail(snapshot: dict[str, Any]) -> str:
    language = snapshot.get("output_language")
    stage = snapshot.get("run_stage")
    lifecycle = snapshot.get("lifecycle") if isinstance(snapshot.get("lifecycle"), dict) else {}
    active_agents = [str(item) for item in lifecycle.get("active_agents") or []]
    completed_agents = [str(item) for item in lifecycle.get("completed_agents") or []]
    completed_specialists = [agent for agent in completed_agents if agent != "Orchestrator"]
    request = str(snapshot.get("student_request") or "").casefold()
    grade_focus = any(token in request for token in ("note", "grade", "1,0", "1.0", "maximum", "bestmöglich"))
    planning_focus = any(token in request for token in ("master", "semester", "studienplan", "study plan", "finish", "fertig"))
    if language == "German":
        if stage == "complete":
            return "Die angeforderten Fachanalysen sind abgeschlossen und wurden in der finalen Antwort zusammengeführt."
        if stage == "failed":
            return "Die Koordination konnte nicht erfolgreich abgeschlossen werden. Der letzte verifizierte Zwischenstand bleibt sichtbar."
        if stage == "synthesis":
            return "Die Fachagenten haben ihre Responses zurückgeliefert. Der Orchestrator gleicht die Ergebnisse derzeit ab und erstellt daraus die Gesamtantwort."
        if stage == "intake":
            if grade_focus and planning_focus:
                return "Die Anfrage zur weiteren Masterplanung und zur bestmöglichen Abschlussnote wurde erfasst. Der Orchestrator bestimmt jetzt die benötigten Fachanalysen und Datenquellen."
            if planning_focus:
                return "Die Anfrage zur weiteren Studienplanung wurde erfasst. Der Orchestrator bestimmt jetzt die benötigten Fachanalysen und Datenquellen."
            return "Die Anfrage wurde erfasst. Der Orchestrator bestimmt jetzt die benötigten Fachagenten und Datenquellen."
        if completed_specialists and any(agent != "Orchestrator" for agent in active_agents):
            return "Erste Fachagenten haben ihre Responses zurückgeliefert, während weitere Analysen noch laufen. Der Orchestrator übernimmt nur verifizierte Ergebnisse in die Gesamtbewertung."
        return "Die zuständigen Fachagenten prüfen derzeit Studienstand, formale Vorgaben und relevante Szenarien. Noch ausstehende A2A-Responses gelten weiterhin als offen."
    if stage == "complete":
        return "The requested specialist analyses are complete and have been combined in the final answer."
    if stage == "failed":
        return "The coordination run could not complete successfully. The last verified intermediate state remains visible."
    if stage == "synthesis":
        return "The specialist responses have returned. The orchestrator is reconciling the findings and composing the overall answer."
    if stage == "intake":
        if grade_focus and planning_focus:
            return "The request about the next study-planning steps and the best achievable final grade has been captured. The orchestrator is selecting the required analyses and data sources."
        if planning_focus:
            return "The study-planning request has been captured. The orchestrator is selecting the required analyses and data sources."
        return "The request has been captured. The orchestrator is now selecting the required specialists and data sources."
    if completed_specialists and any(agent != "Orchestrator" for agent in active_agents):
        return "Some specialist responses have returned while other analyses are still active. The orchestrator incorporates only verified results into the overall assessment."
    return "The relevant specialists are reviewing the study record, formal rules, and requested scenarios. A2A work without a returned response remains open."


def _summary_matches_language(summary: str, language: str) -> bool:
    if language != "German":
        return True
    words = set(re.findall(r"[a-zäöüß]+", summary.casefold()))
    if words & (_GERMAN_MARKERS | {"hat", "prüft", "koordiniert", "vergleicht", "berechnet"}):
        return True
    return len(words & _ENGLISH_MARKERS) < 2


def _summary_describes_active_work(summary: str, language: str) -> bool:
    """Detect active-work wording that becomes stale once lifecycle state advances."""
    folded = " ".join(summary.casefold().split())
    if language == "German":
        return bool(
            re.search(
                r"\b(aktuell|derzeit|gerade|momentan|arbeitet|wertet|prüft|analysiert|"
                r"untersucht|vergleicht|berechnet|sucht|koordiniert|erstellt|bestimmt|"
                r"ermittelt|bereitet|formuliert|verdichtet|synthetisiert|wartet|führt)\b",
                folded,
            )
        )
    return bool(
        re.search(
            r"\b(currently|still|is reviewing|is checking|is comparing|is calculating|"
            r"is analyzing|is evaluating|is searching|is coordinating|reviews|checks|compares|"
            r"calculates|analyzes|evaluates|searches|coordinates|prepares|determines|"
            r"formulates|synthesizes|combines|is formulating|is synthesizing|is combining|is waiting|"
            r"(?:work|analysis|review|search|calculation) is in progress)\b",
            folded,
        )
    )


def _summary_describes_completed_work(summary: str, language: str) -> bool:
    folded = " ".join(summary.casefold().split())
    if language == "German":
        return bool(
            re.search(
                r"\b(hat|haben|wurde|wurden|abgeschlossen|beendet|fertiggestellt|"
                r"zurückgeliefert|durchgeführt|festgestellt)\b",
                folded,
            )
        )
    return bool(
        re.search(
            r"\b(has|have|completed|finished|reviewed|checked|calculated|established|returned)\b",
            folded,
        )
    )


def _claims_run_completion(value: str, language: str) -> bool:
    """Detect claims that the whole run is complete, not merely one specialist result."""
    folded = " ".join(value.casefold().split())
    if language == "German":
        return bool(
            re.search(
                r"\b(?:gesamte[nr]?|vollständige[nr]?|tiefgehende[nr]?|angeforderte[nr]?)?\s*"
                r"(?:analyse|fachanalysen?|auswertung|bearbeitung|koordination|untersuchung|"
                r"gesamtbewertung|prozess|lauf|anfrage)\b.{0,100}\b"
                r"(?:abgeschlossen|beendet|fertig)\b",
                folded,
            )
            or bool(re.search(r"\b(?:alle|sämtliche)\b.{0,100}\b(?:abgeschlossen|beendet|fertig)\b", folded))
        )
    return bool(
        re.search(
            r"\b(?:overall|complete|full|requested|deep)?\s*"
            r"(?:analysis|analyses|evaluation|work|coordination|investigation|process|run|request)\b"
            r".{0,100}\b(?:complete|completed|finished|done)\b",
            folded,
        )
        or re.search(r"\b(?:all|every)\b.{0,100}\b(?:complete|completed|finished|done)\b", folded)
    )


def _summary_matches_state(summary: str, state: str, language: str) -> bool:
    active_wording = _summary_describes_active_work(summary, language)
    completed_wording = _summary_describes_completed_work(summary, language)
    if state == "active":
        return not completed_wording or active_wording
    if state == "completed":
        return not active_wording
    if state == "waiting":
        return not active_wording and not completed_wording
    return True


def _looks_like_student_answer(value: str, language: str, *, headline: bool = False) -> bool:
    """Reject advice/final-answer prose from the status-only observer surface."""
    folded = " ".join(value.casefold().split())
    if language == "German":
        direct_address = bool(
            re.search(
                r"\b(sie|ihnen|ihr|ihre|ihren|ihrem|du|dir|dich|dein|deine|deinen|deinem)\b",
                folded,
            )
        )
        advisory = bool(
            re.search(
                r"\b(sollten|solltest|sollte|müssen|musst|können sie|kannst|empfehlung|"
                r"empfehlungen|empfohlen|konzentrieren sie sich|fokussieren sie sich)\b",
                folded,
            )
        )
        student_advice = bool(re.search(r"\b(?:der student|die studentin|studierende)\b.*\bsoll", folded))
    else:
        direct_address = bool(re.search(r"\b(you|your|yours)\b", folded))
        advisory = bool(
            re.search(
                r"\b(should|must|need to|recommend|recommendation|recommendations|"
                r"focus on|best course of action)\b",
                folded,
            )
        )
        student_advice = bool(re.search(r"\bthe student\b.*\bshould\b", folded))
    if direct_address or student_advice:
        return True
    return headline and advisory


def _detail_conflicts_with_agent_state(
    sentence: str,
    state_by_agent: dict[str, str],
    language: str,
) -> bool:
    folded = sentence.casefold()
    for agent, state in state_by_agent.items():
        if agent.casefold() not in folded:
            continue
        active_wording = _summary_describes_active_work(sentence, language)
        completed_wording = _summary_describes_completed_work(sentence, language)
        if state == "active" and completed_wording and not active_wording:
            return True
        if state == "completed" and active_wording:
            return True
        if state == "waiting" and (active_wording or completed_wording):
            return True
    return False


def _deterministic_evidence(snapshot: dict[str, Any]) -> list[str]:
    """Retain only lifecycle facts derived from trace state, never LLM-authored evidence."""
    facts = [f"run_stage={snapshot.get('run_stage') or 'intake'}"]
    states = [
        f"{item.get('agent')}:{item.get('state')}"
        for item in snapshot.get("agent_states") or []
        if isinstance(item, dict) and item.get("agent") and item.get("state")
    ]
    if states:
        facts.append("agent_states=" + ",".join(states))
    delegations = [
        f"{item.get('receiver')}:{item.get('status')}"
        for item in snapshot.get("a2a_delegations") or []
        if isinstance(item, dict) and item.get("receiver") and item.get("status")
    ]
    if delegations:
        facts.append("a2a=" + ",".join(delegations[-4:]))
    return [_bounded_text(fact, 180) for fact in facts[:3]]


def normalize_runtime_observer_report(
    report: RuntimeObserverReport | dict[str, Any],
    snapshot: dict[str, Any],
) -> RuntimeObserverReport:
    """Apply lifecycle truth, language/length limits, and UI-safe semantic fallbacks."""
    parsed = report if isinstance(report, RuntimeObserverReport) else RuntimeObserverReport.model_validate(report)
    state_by_agent = {
        str(item.get("agent")): str(item.get("state") or "waiting")
        for item in snapshot.get("agent_states") or []
        if isinstance(item, dict) and item.get("agent") in OBSERVER_AGENT_NAMES
    }
    language = str(snapshot.get("output_language") or "English")
    final = snapshot.get("run_stage") == "complete"

    supplied_updates: dict[str, str] = {}
    for item in parsed.agent_updates:
        agent = _canonical_agent_name(item.agent)
        summary = " ".join(item.summary.split())
        if (
            agent not in state_by_agent
            or not summary
            or _contains_raw_runtime_copy(summary)
            or _looks_like_student_answer(summary, language)
            or not _summary_matches_state(summary, state_by_agent[agent], language)
        ):
            continue
        sentence = _split_sentences(summary)[0] if _split_sentences(summary) else ""
        if sentence and _summary_matches_language(sentence, language):
            supplied_updates[agent] = _bounded_text(sentence, 320)

    previous_updates: dict[str, str] = {}
    previous = snapshot.get("previous_observer_report")
    for item in (previous or {}).get("agent_updates") or []:
        if not isinstance(item, dict):
            continue
        agent = _canonical_agent_name(item.get("agent"))
        summary = _bounded_text(item.get("summary"), 320)
        if (
            agent in state_by_agent
            and summary
            and not _contains_raw_runtime_copy(summary)
            and not _looks_like_student_answer(summary, language)
            and _summary_matches_language(summary, language)
            and _summary_matches_state(summary, state_by_agent[agent], language)
        ):
            previous_updates[agent] = summary

    updates = []
    for agent, state in state_by_agent.items():
        summary = supplied_updates.get(agent) or previous_updates.get(agent)
        if not summary or (final and re.search(r"\b(currently|planning|considering|providing|determining|running)\b", summary, re.I)):
            summary = (
                _german_fallback_summary(agent, state)
                if language == "German"
                else _english_fallback_summary(agent, state)
            )
        updates.append(AgentProgressReport(agent=agent, state=state, summary=summary))

    detail_sentences: list[str] = []
    seen: set[str] = set()
    for sentence in _split_sentences(parsed.detail):
        folded = re.sub(r"\W+", " ", sentence.casefold()).strip()
        if not folded or folded in seen or _contains_raw_runtime_copy(sentence):
            continue
        if _looks_like_student_answer(sentence, language):
            continue
        if _detail_conflicts_with_agent_state(sentence, state_by_agent, language):
            continue
        if not final and _claims_run_completion(sentence, language):
            continue
        if final and re.search(r"\b(currently|planning|considering|providing|determining|running)\b", sentence, re.I):
            continue
        if final and re.search(r"\b(student (?:is )?(?:seeking|asking|unsure|wants)|the request asks)\b", sentence, re.I):
            continue
        if not _summary_matches_language(sentence, language):
            continue
        seen.add(folded)
        detail_sentences.append(sentence)
        if len(detail_sentences) == 3:
            break
    detail = _bounded_text(" ".join(detail_sentences), 680) if detail_sentences else _fallback_detail(snapshot)

    headline = _bounded_text(parsed.headline, 100)
    if final:
        headline = "Analyse abgeschlossen" if language == "German" else "Analysis complete"
    elif (
        not headline
        or _contains_raw_runtime_copy(headline)
        or _looks_like_student_answer(headline, language, headline=True)
        or not _summary_matches_language(headline, language)
        or _detail_conflicts_with_agent_state(headline, state_by_agent, language)
        or _claims_run_completion(headline, language)
    ):
        if snapshot.get("run_stage") == "synthesis":
            headline = "Ergebnisse werden zusammengeführt" if language == "German" else "Synthesizing specialist results"
        else:
            headline = "Laufende Analyse" if language == "German" else "Analysis in progress"

    active_agent = _canonical_agent_name(parsed.active_agent)
    if active_agent not in state_by_agent or state_by_agent.get(active_agent) != "active":
        active_agent = next((agent for agent, state in state_by_agent.items() if state == "active"), None)

    evidence = _deterministic_evidence(snapshot)

    return RuntimeObserverReport(
        headline=headline,
        detail=detail,
        active_agent=active_agent,
        agent_updates=updates,
        evidence=evidence,
    )


def reconcile_runtime_observer_report(
    events: list[dict[str, Any]],
    report: RuntimeObserverReport | dict[str, Any],
) -> RuntimeObserverReport:
    """Reconcile a persisted report with newer lifecycle events, including run completion."""
    return normalize_runtime_observer_report(report, runtime_observer_snapshot(events))


def deterministic_runtime_observer_report(
    events: list[dict[str, Any]],
    *,
    study_context: str | None = None,
) -> RuntimeObserverReport:
    """Return an immediate semantic report when the background LLM is pending or unavailable."""
    snapshot = runtime_observer_snapshot(events, study_context=study_context)
    language = str(snapshot.get("output_language") or "English")
    stage = str(snapshot.get("run_stage") or "intake")
    if stage == "complete":
        headline = "Analyse abgeschlossen" if language == "German" else "Analysis complete"
    elif stage == "failed":
        headline = "Analyse unterbrochen" if language == "German" else "Analysis interrupted"
    elif stage == "synthesis":
        headline = "Ergebnisse werden zusammengeführt" if language == "German" else "Synthesizing specialist results"
    elif stage == "execution":
        headline = "Fachanalysen laufen" if language == "German" else "Specialist analysis in progress"
    else:
        headline = "Anfrage wird eingeordnet" if language == "German" else "Scoping the request"
    base = RuntimeObserverReport(
        headline=headline,
        detail=_fallback_detail(snapshot),
        active_agent=None,
        agent_updates=[],
        evidence=[],
    )
    return normalize_runtime_observer_report(base, snapshot)


def generate_runtime_observer_report(
    events: list[dict[str, Any]],
    *,
    observer_model: str | None = None,
    manager_model: str | None = None,
    specialist_model: str | None = None,
    study_context: str | None = None,
) -> RuntimeObserverResult:
    """Use the lightweight observer model to narrate cumulative global and per-agent progress."""
    model = resolve_study_assistant_observer_model(
        observer_model=observer_model,
        manager_model=manager_model,
        specialist_model=specialist_model,
    )
    llm = get_default_llm(
        model=model,
        temperature=0.1,
        timeout=resolve_study_assistant_observer_timeout(),
    )
    # Event-bus handlers run on CrewAI worker threads. Mark this distinct LLM
    # instance so those handlers cannot turn observer calls into new observed
    # lifecycle events and recursively schedule more observer calls.
    suppress_trace_events_from(llm)
    snapshot = runtime_observer_snapshot(events, study_context=study_context)
    with suppress_trace_events():
        report = llm.call(
            messages=[
                {
                    "role": "system",
                    "content": (
                    "You are a read-only Runtime Observer for a CrewAI multi-agent study workbench. /no_think "
                    "Your only task is execution-status telemetry. You are not the study assistant and must never "
                    "answer the student's request, give advice, recommend actions, or write a final-response preview. "
                    "Do not address the student as 'you', 'Sie', or 'du'. Maintain a cumulative status explanation "
                    "of what the agent system has demonstrably completed, is doing now, or is waiting for. "
                    "Write every user-facing field in output_language, which matches the student's request. "
                    "Treat agent_states as authoritative: copy each canonical agent name and state exactly, and never "
                    "describe a completed agent as active or an active agent as completed. response_ready is authoritative: "
                    "internal tool results do not mean that an A2A response has returned. The previous report is memory, "
                    "not text to repeat. "
                    "student_request, intent, and study_context identify the scope only. In particular, study_context "
                    "is background supplied to the crew, not evidence that an agent has analyzed or established it. "
                    "A completed finding may be reported only when it appears in latest_result, an A2A response, or a "
                    "recent tool output. Use only supported facts and never claim that an eligible agent was invoked "
                    "without evidence. "
                    "Translate runtime mechanics into domain meaning: explain what each invoked agent is trying to "
                    "learn, compare, validate, or calculate and which courses, requirements, grades, or deadlines matter. "
                    "Do not expose model names, raw task identifiers, tool function names, input-key lists, schema counts, "
                    "call IDs, or generic filler such as 'thinking', 'working', and 'gathering information'. "
                    "The overall detail must synthesize the most important progress change in at most three concise "
                    "sentences; do not concatenate one sentence per agent or restate the request. During intake, report "
                    "only that the request is being scoped or routed. During execution, prioritize evidenced new findings "
                    "and current dependencies. run_stage and lifecycle.crew_completed are absolute: unless run_stage is "
                    "complete, never say that the analysis, coordination, request, process, or run is completed, finished, "
                    "done, abgeschlossen, beendet, or fertig. During synthesis, state which specialist responses have "
                    "returned and that the Orchestrator is still combining them. At completion, report what agents established without turning those findings "
                    "into recommendations for the student, and never say work is still being planned. "
                    "Create exactly one update for every agent in agent_states. Each summary must be one concise domain-level "
                    "sentence about what that agent is investigating or has established. Omit all other agents. "
                    "Keep evidence compact and technical because it is retained only for diagnostics, not displayed to users."
                    ),
                },
                {
                    "role": "user",
                    "content": "Current cumulative runtime state:\n" + json.dumps(snapshot, ensure_ascii=False),
                },
            ],
            response_model=RuntimeObserverReport,
        )
    if not isinstance(report, RuntimeObserverReport):
        report = RuntimeObserverReport.model_validate(report)
    report = normalize_runtime_observer_report(report, snapshot)
    return RuntimeObserverResult(model=model, report=report)
