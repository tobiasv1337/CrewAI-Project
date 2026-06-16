from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import json
from pathlib import Path
import re
from typing import Any

import streamlit as st

from core.manager import DegreeManager
from core.models import Module
from core.persistence import load_modules
from core.registry import create_program, list_relevant_programs, list_selectable_programs
from crew.config.llm import DEFAULT_STUDY_ASSISTANT_MODEL
from crew.isis_client import IsisCredentials, MoodleRestClient, login_via_playwright_sync
from crew.tracing import TraceWorkbench, load_trace_workbench
from main import MultiAgentStudyAssistantRunResult, run_study_assistant_query


CHAT_HISTORY_KEY = "study_chat_messages_by_profile"
ISIS_SESSIONS_KEY = "study_chat_isis_sessions"
PENDING_PROMPT_KEY = "study_chat_pending_prompt"
DEFAULT_MANAGER_MODEL = ""
DEFAULT_TEMPERATURE = 0.2
AGENT_LANES = [
    "Orchestrator",
    "Study Advisor",
    "MOSES Module Researcher",
    "ISIS Course Info Specialist",
]


@dataclass(frozen=True)
class ChatRuntimeSettings:
    specialist_model: str | None
    manager_model: str | None
    temperature: float
    top_p: float | None
    trace_enabled: bool
    trace_full: bool
    verbose: bool
    cache: bool
    allow_temp_enrollment: bool
    planning_enabled: bool = False


def render_chat_page() -> None:
    inject_chat_css()
    profile_slug = str(st.session_state.get("active_profile") or "primary")
    profile_name = _active_profile_display_name(profile_slug)

    settings = _render_chat_sidebar(profile_slug)
    messages = get_profile_messages(profile_slug)

    st.markdown(
        f"""
        <section class="chat-hero">
          <div>
            <h1>Study Chat</h1>
            <p>Active profile: <strong>{html.escape(profile_name)}</strong>. Multi-agent answers are grounded in Grade Manager, MOSES, and ISIS tool traces.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not messages:
        _render_empty_state()

    for message in messages:
        _render_chat_message(message)

    pending_prompt = st.session_state.pop(PENDING_PROMPT_KEY, None)
    if isinstance(pending_prompt, dict) and pending_prompt.get("profile_slug") == profile_slug:
        prompt_text = str(pending_prompt.get("prompt") or "").strip()
        if prompt_text:
            _run_and_render_assistant_turn(profile_slug, prompt_text, settings)

    prompt = st.chat_input("Ask the TU Study Assistant...")
    if prompt:
        run_prompt = prompt.strip()
        if run_prompt:
            _append_message(profile_slug, {"role": "user", "content": run_prompt, "created_at": _now_iso()})
            with st.chat_message("user"):
                st.markdown(run_prompt)
            _run_and_render_assistant_turn(profile_slug, run_prompt, settings)


def _render_chat_sidebar(profile_slug: str) -> ChatRuntimeSettings:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Study Chat")
        _render_isis_account_panel(profile_slug)

        with st.expander("Agent runtime", expanded=False):
            specialist_model = st.text_input(
                "Specialist model",
                value=DEFAULT_STUDY_ASSISTANT_MODEL,
                help="Used by Study Advisor, MOSES Researcher, and ISIS Specialist.",
                key=f"chat_specialist_model_{profile_slug}",
            ).strip()
            manager_model = st.text_input(
                "Manager model",
                value=DEFAULT_MANAGER_MODEL,
                help="Optional override for the Orchestrator. Leave empty to use the specialist model.",
                key=f"chat_manager_model_{profile_slug}",
            ).strip()
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_TEMPERATURE,
                step=0.05,
                key=f"chat_temperature_{profile_slug}",
            )
            top_p_enabled = st.toggle("Set top_p", value=False, key=f"chat_top_p_enabled_{profile_slug}")
            top_p = (
                st.slider("top_p", min_value=0.1, max_value=1.0, value=0.9, step=0.05, key=f"chat_top_p_{profile_slug}")
                if top_p_enabled
                else None
            )
            trace_mode = st.segmented_control(
                "Tracing",
                ["Preview", "Full", "Disabled"],
                default="Preview",
                key=f"chat_trace_mode_{profile_slug}",
            )
            allow_temp_enrollment = st.toggle(
                "Temporary ISIS self-enrollment",
                value=True,
                help="When enabled, read-only ISIS tools may enroll briefly, inspect course information, then unenroll.",
                key=f"chat_temp_enrollment_{profile_slug}",
            )
            verbose = st.toggle("CrewAI verbose logs", value=False, key=f"chat_verbose_{profile_slug}")
            cache = st.toggle("CrewAI cache", value=True, key=f"chat_cache_{profile_slug}")
            st.toggle(
                "CrewAI Planning",
                value=False,
                disabled=True,
                help="Reserved for the later Flow/Planning iteration.",
                key=f"chat_planning_disabled_{profile_slug}",
            )

        if st.button("Clear this profile's chat", width="stretch", key=f"chat_clear_{profile_slug}"):
            _set_profile_messages(profile_slug, [])
            st.rerun()

    trace_mode = str(st.session_state.get(f"chat_trace_mode_{profile_slug}") or "Preview")
    return ChatRuntimeSettings(
        specialist_model=specialist_model or None,
        manager_model=manager_model or None,
        temperature=temperature,
        top_p=top_p,
        trace_enabled=trace_mode != "Disabled",
        trace_full=trace_mode == "Full",
        verbose=verbose,
        cache=cache,
        allow_temp_enrollment=allow_temp_enrollment,
    )


def _render_isis_account_panel(profile_slug: str) -> None:
    sessions = _isis_sessions()
    session = sessions.get(profile_slug) or {"mode": "env", "created_at": None, "client": None}
    mode = str(session.get("mode") or "env")
    created_at = session.get("created_at")

    st.markdown("#### ISIS access")
    if mode == "session" and session.get("client") is not None:
        st.success(f"Session login active. Age: {_age_label(created_at)}")
    else:
        st.info("Using environment fallback for ISIS credentials/token.")

    username_key = f"chat_isis_username_{profile_slug}"
    password_key = f"chat_isis_password_{profile_slug}"
    username = st.text_input("TUB account", key=username_key)
    password = st.text_input("Password", type="password", key=password_key)

    col_login, col_env = st.columns(2)
    if col_login.button("Login", width="stretch", key=f"chat_isis_login_{profile_slug}"):
        if not username.strip() or not password:
            st.warning("Enter both TUB account and password.")
        else:
            with st.spinner("Logging in to ISIS via Shibboleth..."):
                try:
                    credentials = IsisCredentials(username=username.strip(), password=password)
                    bundle = login_via_playwright_sync(credentials)
                    client = MoodleRestClient(wstoken=bundle.wstoken, cookies=bundle.cookies)
                    sessions[profile_slug] = {
                        "mode": "session",
                        "client": client,
                        "created_at": _now_iso(),
                    }
                    st.session_state[ISIS_SESSIONS_KEY] = sessions
                    st.session_state[password_key] = ""
                    st.success("ISIS session login is active for this profile.")
                except Exception as exc:
                    st.error(f"ISIS login failed: {exc}")

    if col_env.button("Use env", width="stretch", key=f"chat_isis_env_{profile_slug}"):
        sessions[profile_slug] = {"mode": "env", "client": None, "created_at": _now_iso()}
        st.session_state[ISIS_SESSIONS_KEY] = sessions
        st.toast("ISIS env fallback enabled for this profile.")

    if st.button("Clear ISIS session", width="stretch", key=f"chat_isis_clear_{profile_slug}"):
        sessions.pop(profile_slug, None)
        st.session_state[ISIS_SESSIONS_KEY] = sessions
        st.toast("ISIS session cleared.")


def _render_empty_state() -> None:
    examples = [
        "What ML modules can I still take next semester?",
        "How many credits are still missing in my degree?",
        "Check whether Reinforcement Learning fits my study plan.",
        "What deadlines are visible in my current ISIS courses?",
    ]
    st.markdown('<div class="chat-empty-grid">', unsafe_allow_html=True)
    for example in examples:
        st.markdown(f"<div class='chat-example'>{html.escape(example)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_message(message: dict[str, Any]) -> None:
    with st.chat_message(message.get("role", "assistant")):
        st.markdown(str(message.get("content") or ""))
        if message.get("role") == "assistant":
            workbench = message.get("workbench")
            if workbench:
                _render_trace_panel(workbench, expanded=False)
            pending_write = message.get("pending_write")
            if pending_write:
                _render_pending_write_card(str(message.get("profile_slug") or ""), pending_write)


def _run_and_render_assistant_turn(profile_slug: str, prompt: str, settings: ChatRuntimeSettings) -> None:
    events: list[dict[str, Any]] = []
    with st.chat_message("assistant"):
        status = st.status("Crew is coordinating agents...", expanded=True)
        live_placeholder = st.empty()

        def on_trace_event(event: dict[str, Any]) -> None:
            events.append(event)
            with live_placeholder.container():
                _render_live_trace(events)

        try:
            with status:
                _render_live_trace(events)
                result = _run_chat_query(
                    profile_slug=profile_slug,
                    prompt=prompt,
                    settings=settings,
                    on_trace_event=on_trace_event if settings.trace_enabled else None,
                )
            status.update(label="Crew finished", state="complete", expanded=False)
            st.markdown(result.answer.rstrip())
            workbench = _workbench_from_result(result)
            if workbench:
                _render_trace_panel(workbench, expanded=True)
            pending_write = extract_pending_study_plan_write(result, workbench)
            if pending_write:
                _render_pending_write_card(profile_slug, pending_write)
            if workbench and trace_has_successful_study_plan_write(workbench):
                refresh_streamlit_profile_state(profile_slug)
                st.success("Study plan data was updated and reloaded.")
            _append_message(
                profile_slug,
                {
                    "role": "assistant",
                    "content": result.answer,
                    "created_at": _now_iso(),
                    "trace_dir": str(result.trace_dir) if result.trace_dir else None,
                    "state_path": str(result.state_path) if result.state_path else None,
                    "tool_summary_lines": result.tool_summary_lines,
                    "workbench": workbench,
                    "pending_write": pending_write,
                    "profile_slug": profile_slug,
                },
            )
        except Exception as exc:
            status.update(label="Crew failed", state="error", expanded=True)
            content = f"Could not run the Study Assistant: `{exc}`"
            st.error(content)
            _append_message(profile_slug, {"role": "assistant", "content": content, "created_at": _now_iso()})


def _run_chat_query(
    *,
    profile_slug: str,
    prompt: str,
    settings: ChatRuntimeSettings,
    on_trace_event,
) -> MultiAgentStudyAssistantRunResult:
    return run_study_assistant_query(
        query=prompt,
        student_context=build_student_context(profile_slug),
        allow_temp_enrollment=settings.allow_temp_enrollment,
        profile_slug=profile_slug,
        isis_client=get_profile_isis_client(profile_slug),
        on_trace_event=on_trace_event,
        model=settings.specialist_model,
        manager_model=settings.manager_model,
        temperature=settings.temperature,
        top_p=settings.top_p,
        trace=settings.trace_enabled,
        trace_full=settings.trace_full,
        verbose=settings.verbose,
        cache=settings.cache,
    )


def _render_pending_write_card(profile_slug: str, action: dict[str, Any]) -> None:
    if not profile_slug:
        return
    module_query = str(action.get("module_query") or "").strip()
    term = str(action.get("term") or "").strip()
    if not module_query or not term:
        return
    area = str(action.get("area") or "").strip()
    program_key = str(action.get("program_key") or "").strip()

    st.markdown(
        f"""
        <div class="pending-write">
          <strong>Study-plan confirmation required</strong>
          <div>Module: <code>{html.escape(module_query)}</code></div>
          <div>Term: <code>{html.escape(term)}</code></div>
          <div>Area: <code>{html.escape(area or "auto")}</code></div>
          <div>Program: <code>{html.escape(program_key or "auto")}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    confirm_prompt = (
        "I explicitly confirm this study-plan write. "
        f"Add module {module_query} for {term}. "
        f"{'Use area ' + area + '. ' if area else ''}"
        f"{'Use program ' + program_key + '. ' if program_key else ''}"
        "Use confirmation_token CONFIRM_STUDY_PLAN_WRITE."
    )
    if st.button("Confirm and run Study Advisor write", type="primary", key=f"confirm_write_{profile_slug}_{module_query}_{term}"):
        _append_message(profile_slug, {"role": "user", "content": confirm_prompt, "created_at": _now_iso()})
        st.session_state[PENDING_PROMPT_KEY] = {"profile_slug": profile_slug, "prompt": confirm_prompt}
        st.rerun()


def _current_settings_from_state(profile_slug: str) -> ChatRuntimeSettings:
    trace_mode = str(st.session_state.get(f"chat_trace_mode_{profile_slug}") or "Preview")
    return ChatRuntimeSettings(
        specialist_model=str(st.session_state.get(f"chat_specialist_model_{profile_slug}") or DEFAULT_STUDY_ASSISTANT_MODEL).strip() or None,
        manager_model=str(st.session_state.get(f"chat_manager_model_{profile_slug}") or "").strip() or None,
        temperature=float(st.session_state.get(f"chat_temperature_{profile_slug}") or DEFAULT_TEMPERATURE),
        top_p=float(st.session_state[f"chat_top_p_{profile_slug}"]) if st.session_state.get(f"chat_top_p_enabled_{profile_slug}") else None,
        trace_enabled=trace_mode != "Disabled",
        trace_full=trace_mode == "Full",
        verbose=bool(st.session_state.get(f"chat_verbose_{profile_slug}", False)),
        cache=bool(st.session_state.get(f"chat_cache_{profile_slug}", True)),
        allow_temp_enrollment=bool(st.session_state.get(f"chat_temp_enrollment_{profile_slug}", True)),
    )


def _render_live_trace(events: list[dict[str, Any]]) -> None:
    workbench = live_workbench_from_events(events)
    st.markdown("<div class='live-label'>Live agent workbench</div>", unsafe_allow_html=True)
    _render_source_flow(workbench)
    _render_agent_lanes(workbench, live=True)


def _render_trace_panel(workbench: dict[str, Any], *, expanded: bool) -> None:
    with st.expander("Agent workbench and tool trace", expanded=expanded):
        _render_source_flow(workbench)
        _render_agent_lanes(workbench, live=False)
        _render_artifact_links(workbench)
        _render_tool_expanders(workbench)


def _render_source_flow(workbench: dict[str, Any]) -> None:
    flows = workbench.get("source_flow") or []
    if not flows:
        flows = _default_source_flow(active=False)
    pieces = []
    for item in flows:
        cls = "flow-step active" if item.get("active") else "flow-step"
        label = f"{item.get('source')} -> {item.get('agent')} -> {item.get('target')}"
        pieces.append(f"<div class='{cls}'>{html.escape(label)}</div>")
    st.markdown(f"<div class='source-flow'>{''.join(pieces)}</div>", unsafe_allow_html=True)


def _render_agent_lanes(workbench: dict[str, Any], *, live: bool) -> None:
    groups = _groups_by_label(workbench)
    cols = st.columns(4)
    for index, label in enumerate(AGENT_LANES):
        group = groups.get(label, {"agent_label": label, "tool_calls": [], "status": "idle", "duration_ms": 0})
        with cols[index]:
            status = str(group.get("status") or ("running" if live else "idle"))
            calls = group.get("tool_calls") or []
            st.markdown(
                f"""
                <div class="agent-lane {html.escape(status)}">
                  <div class="agent-lane-title">{html.escape(label)}</div>
                  <div class="agent-lane-meta">{len(calls)} tool call{'s' if len(calls) != 1 else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if calls:
                for call in calls[-4:]:
                    _render_tool_call_row(call)
            else:
                st.caption("Waiting for activity.")


def _render_tool_call_row(call: dict[str, Any]) -> None:
    badges = " ".join(f"<span class='trace-badge'>{html.escape(str(badge))}</span>" for badge in call.get("badges", []))
    duration = call.get("duration_ms")
    duration_text = f"{duration} ms" if duration is not None else str(call.get("status") or "running")
    st.markdown(
        f"""
        <div class="tool-row {html.escape(str(call.get('status') or 'ok'))}">
          <div class="tool-name">{html.escape(str(call.get('tool_name') or 'Tool'))}</div>
          <div class="tool-meta">{html.escape(duration_text)}</div>
          <div>{badges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_tool_expanders(workbench: dict[str, Any]) -> None:
    calls = [
        call
        for group in (workbench.get("groups") or [])
        for call in (group.get("tool_calls") or [])
    ]
    if not calls:
        st.caption("No tool calls captured.")
        return
    for call in calls:
        title = f"{call.get('call_id')}. {call.get('agent_label')} / {call.get('tool_name')}"
        with st.expander(title, expanded=False):
            st.caption(f"Status: {call.get('status')} | Source: {call.get('source_system')}")
            st.code(json.dumps(call.get("tool_input") or {}, ensure_ascii=False, indent=2), language="json")
            st.text(str(call.get("output_preview") or ""))


def _render_artifact_links(workbench: dict[str, Any]) -> None:
    artifacts = {key: value for key, value in (workbench.get("artifacts") or {}).items() if value}
    if not artifacts:
        return
    st.markdown("Trace artifacts:")
    for label, path in artifacts.items():
        st.code(f"{label}: {path}", language="text")


def live_workbench_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    calls_by_id: dict[int, dict[str, Any]] = {}
    run_id = None
    for event in events:
        run_id = event.get("run_id") or run_id
        if event.get("event") == "tool_start":
            call_id = int(event.get("call_id") or 0)
            calls_by_id[call_id] = {
                "call_id": call_id,
                "tool_name": event.get("tool_name"),
                "tool_input": event.get("tool_input") or {},
                "agent_role": event.get("agent_role"),
                "agent_label": event.get("agent_label") or "Unknown Agent",
                "task_name": event.get("task_name"),
                "source_system": event.get("source_system") or "Other",
                "status": "running",
                "badges": event.get("badges") or [],
                "output_preview": "",
                "output_chars": 0,
                "output_truncated": False,
            }
        elif event.get("event") == "tool_finish":
            call = dict(event.get("tool_call") or {})
            if call:
                calls_by_id[int(call.get("call_id") or 0)] = call

    groups: dict[str, dict[str, Any]] = {}
    for call in sorted(calls_by_id.values(), key=lambda item: int(item.get("call_id") or 0)):
        label = str(call.get("agent_label") or "Unknown Agent")
        group = groups.setdefault(
            label,
            {
                "agent_label": label,
                "agent_role": call.get("agent_role"),
                "source_system": call.get("source_system") or "Other",
                "tool_calls": [],
                "duration_ms": 0,
                "status": "ok",
            },
        )
        group["tool_calls"].append(call)
        group["status"] = _combine_status(str(group.get("status") or "ok"), str(call.get("status") or "ok"))
    active_sources = {str(call.get("source_system") or "Other") for call in calls_by_id.values()}
    return {
        "run_id": run_id,
        "run_dir": None,
        "groups": [groups[label] for label in AGENT_LANES if label in groups],
        "total_tool_calls": len(calls_by_id),
        "source_flow": _default_source_flow(active_sources=active_sources, active=bool(calls_by_id)),
        "artifacts": {},
    }


def _workbench_from_result(result: MultiAgentStudyAssistantRunResult) -> dict[str, Any] | None:
    if not result.trace_dir:
        return None
    workbench = load_trace_workbench(result.trace_dir)
    return workbench.model_dump() if isinstance(workbench, TraceWorkbench) else workbench


def extract_pending_study_plan_write(
    result: MultiAgentStudyAssistantRunResult,
    workbench: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if workbench:
        for call in _all_tool_calls(workbench):
            if call.get("tool_name") == "Add Module To Study Plan" and "write refused" in str(call.get("output_preview") or "").casefold():
                tool_input = call.get("tool_input") or {}
                module_query = str(tool_input.get("module_query") or "").strip()
                term = str(tool_input.get("term") or "").strip()
                if module_query and term:
                    return {
                        "module_query": module_query,
                        "term": term,
                        "area": tool_input.get("area"),
                        "program_key": tool_input.get("program_key"),
                    }
    return extract_pending_study_plan_write_from_text(result.answer)


def extract_pending_study_plan_write_from_text(text: str) -> dict[str, Any] | None:
    module_match = re.search(r"\b(?:MOSES\s*)?(?:module\s*)?(\d{4,6})\b", text, flags=re.I)
    term_match = re.search(r"\b(WS\s*\d{2}(?:/\d{2})?|SS\s*\d{2}|WiSe\s*\d{4}(?:/\d{2})?|SoSe\s*\d{4})\b", text, flags=re.I)
    if not module_match or not term_match:
        return None
    return {
        "module_query": module_match.group(1),
        "term": term_match.group(1).upper().replace("  ", " "),
        "area": None,
        "program_key": None,
    }


def trace_has_successful_study_plan_write(workbench: dict[str, Any]) -> bool:
    for call in _all_tool_calls(workbench):
        if "study-plan write" in call.get("badges", []):
            return True
    return False


def build_student_context(profile_slug: str) -> str:
    modules = list(st.session_state.get("modules") or [])
    program_view = st.session_state.get("program_view") or "All"
    relevant_programs = st.session_state.get("relevant_programs") or list_relevant_programs(modules)
    completed = sum(1 for module in modules if getattr(module, "state", None) and module.state.value == "Completed")
    in_progress = sum(1 for module in modules if getattr(module, "state", None) and module.state.value == "In Progress")
    planned = sum(1 for module in modules if getattr(module, "state", None) and module.state.value == "Planned")
    return "\n".join(
        [
            f"Active Grade Manager profile slug: {profile_slug}",
            f"Selected program view: {program_view}",
            f"Relevant degree programs: {', '.join(relevant_programs) if relevant_programs else 'none inferred'}",
            f"Loaded modules in UI state: {len(modules)}",
            f"State counts: Completed={completed}, In Progress={in_progress}, Planned={planned}",
            "Use Grade Manager tools as the source of truth for full module details.",
        ]
    )


def get_profile_isis_client(profile_slug: str) -> MoodleRestClient | None:
    session = _isis_sessions().get(profile_slug)
    if not session or session.get("mode") != "session":
        return None
    client = session.get("client")
    return client if isinstance(client, MoodleRestClient) else None


def refresh_streamlit_profile_state(profile_slug: str) -> None:
    modules = load_modules(profile_slug)
    relevant = list_relevant_programs(modules)
    st.session_state["modules"] = modules
    st.session_state["relevant_programs"] = relevant
    st.session_state["selectable_programs"] = list_selectable_programs(modules)
    st.session_state["managers"] = {key: DegreeManager(create_program(key)) for key in relevant}


def get_profile_messages(profile_slug: str) -> list[dict[str, Any]]:
    return list(_chat_store().get(profile_slug, []))


def _append_message(profile_slug: str, message: dict[str, Any]) -> None:
    messages = get_profile_messages(profile_slug)
    messages.append(message)
    _set_profile_messages(profile_slug, messages)


def _set_profile_messages(profile_slug: str, messages: list[dict[str, Any]]) -> None:
    store = _chat_store()
    store[profile_slug] = messages
    st.session_state[CHAT_HISTORY_KEY] = store


def _chat_store() -> dict[str, list[dict[str, Any]]]:
    store = st.session_state.get(CHAT_HISTORY_KEY)
    if not isinstance(store, dict):
        store = {}
        st.session_state[CHAT_HISTORY_KEY] = store
    return store


def _isis_sessions() -> dict[str, dict[str, Any]]:
    sessions = st.session_state.get(ISIS_SESSIONS_KEY)
    if not isinstance(sessions, dict):
        sessions = {}
        st.session_state[ISIS_SESSIONS_KEY] = sessions
    return sessions


def _groups_by_label(workbench: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(group.get("agent_label")): group for group in (workbench.get("groups") or [])}


def _all_tool_calls(workbench: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        call
        for group in (workbench.get("groups") or [])
        for call in (group.get("tool_calls") or [])
    ]


def _default_source_flow(*, active_sources: set[str] | None = None, active: bool = False) -> list[dict[str, Any]]:
    active_sources = active_sources or set()
    return [
        {"source": "Grade Manager", "agent": "Study Advisor", "target": "Orchestrator", "active": "Grade Manager" in active_sources},
        {"source": "MOSES", "agent": "MOSES Module Researcher", "target": "Orchestrator", "active": "MOSES" in active_sources},
        {"source": "ISIS", "agent": "ISIS Course Info Specialist", "target": "Orchestrator", "active": "ISIS" in active_sources},
        {"source": "Orchestrator", "agent": "Final Answer", "target": "Student", "active": active},
    ]


def _combine_status(left: str, right: str) -> str:
    order = {"idle": 0, "ok": 1, "running": 2, "warning": 3, "error": 4}
    return right if order.get(right, 0) > order.get(left, 0) else left


def _active_profile_display_name(profile_slug: str) -> str:
    profiles = st.session_state.get("profiles") or []
    profile = next((item for item in profiles if getattr(item, "slug", None) == profile_slug), None)
    return getattr(profile, "display_name", profile_slug)


def _age_label(created_at: object | None) -> str:
    if not created_at:
        return "unknown"
    try:
        created = datetime.fromisoformat(str(created_at))
    except ValueError:
        return "unknown"
    seconds = max(0, int((datetime.now() - created.replace(tzinfo=None)).total_seconds()))
    minutes = seconds // 60
    if minutes < 1:
        return "under 1 min"
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def inject_chat_css() -> None:
    st.markdown(
        """
        <style>
        .chat-hero {
            border-bottom: 1px solid rgba(217, 223, 232, 0.95);
            padding: 0.35rem 0 1rem 0;
            margin-bottom: 1rem;
        }
        .chat-hero h1 {
            font-size: 1.75rem;
            line-height: 1.2;
            margin: 0 0 0.25rem 0;
            letter-spacing: 0;
            color: #151922;
        }
        .chat-hero p {
            margin: 0;
            color: #536070;
            font-size: 0.95rem;
        }
        .chat-empty-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.5rem 0;
        }
        .chat-example {
            border: 1px solid #d8dee8;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.85rem;
            color: #2d3748;
            min-height: 4.25rem;
        }
        .live-label {
            font-weight: 760;
            color: #1f2937;
            margin: 0.25rem 0 0.5rem 0;
        }
        .source-flow {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.5rem;
            margin: 0.5rem 0 0.75rem 0;
        }
        .flow-step {
            border: 1px solid #d8dee8;
            background: #f7f9fc;
            color: #687386;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            font-size: 0.78rem;
            min-height: 3rem;
        }
        .flow-step.active {
            border-color: rgba(39, 125, 96, 0.45);
            background: #eef9f3;
            color: #155b42;
        }
        .agent-lane {
            border: 1px solid #d8dee8;
            background: #ffffff;
            border-radius: 8px;
            padding: 0.75rem;
            min-height: 4.4rem;
            margin-bottom: 0.5rem;
        }
        .agent-lane.running {
            border-color: #7093d8;
            background: #f3f7ff;
        }
        .agent-lane.warning {
            border-color: #d9a441;
            background: #fff8e8;
        }
        .agent-lane.error {
            border-color: #c94b5b;
            background: #fff1f3;
        }
        .agent-lane-title {
            font-weight: 760;
            font-size: 0.92rem;
            color: #1f2937;
        }
        .agent-lane-meta {
            color: #687386;
            font-size: 0.76rem;
            margin-top: 0.2rem;
        }
        .tool-row {
            border: 1px solid #e1e6ef;
            border-radius: 8px;
            padding: 0.55rem;
            margin: 0.45rem 0;
            background: #ffffff;
        }
        .tool-row.running {
            border-color: #7093d8;
        }
        .tool-row.warning {
            border-color: #d9a441;
        }
        .tool-row.error {
            border-color: #c94b5b;
        }
        .tool-name {
            font-size: 0.78rem;
            font-weight: 700;
            color: #1f2937;
            overflow-wrap: anywhere;
        }
        .tool-meta {
            color: #687386;
            font-size: 0.72rem;
            margin: 0.15rem 0 0.35rem 0;
        }
        .trace-badge {
            display: inline-block;
            border-radius: 999px;
            border: 1px solid #d8dee8;
            padding: 0.05rem 0.35rem;
            margin: 0.08rem 0.12rem 0.08rem 0;
            font-size: 0.68rem;
            color: #344054;
            background: #f8fafc;
        }
        .pending-write {
            border: 1px solid rgba(190, 54, 73, 0.35);
            background: #fff5f6;
            color: #3f1720;
            border-radius: 8px;
            padding: 0.85rem;
            margin-top: 0.75rem;
        }
        .pending-write code {
            background: rgba(255, 255, 255, 0.75);
            padding: 0.05rem 0.25rem;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
