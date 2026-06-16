from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import html
import json
from pathlib import Path
from queue import Empty, Queue
import re
import time
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

    settings = _render_chat_config_panel(profile_slug)
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
        _render_empty_state(profile_slug)

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


def _render_chat_config_panel(profile_slug: str) -> ChatRuntimeSettings:
    with st.expander("🛠️ Configuration & Connections", expanded=False):
        col_isis, col_agent = st.columns(2)

        with col_isis:
            _render_isis_account_panel(profile_slug)

        with col_agent:
            st.markdown("#### Agent runtime settings")
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

            col_temp, col_topp = st.columns(2)
            with col_temp:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=DEFAULT_TEMPERATURE,
                    step=0.05,
                    key=f"chat_temperature_{profile_slug}",
                )
            with col_topp:
                top_p_enabled = st.toggle("Set top_p", value=False, key=f"chat_top_p_enabled_{profile_slug}")
                top_p = (
                    st.slider("top_p", min_value=0.1, max_value=1.0, value=0.9, step=0.05, key=f"chat_top_p_{profile_slug}")
                    if top_p_enabled
                    else None
                )

            trace_mode = st.segmented_control(
                "Tracing level",
                ["Preview", "Full", "Disabled"],
                default="Preview",
                key=f"chat_trace_mode_{profile_slug}",
            )

            col_toggles1, col_toggles2 = st.columns(2)
            with col_toggles1:
                allow_temp_enrollment = st.toggle(
                    "Temp ISIS enrollment",
                    value=True,
                    help="When enabled, read-only ISIS tools may enroll briefly, inspect course information, then unenroll.",
                    key=f"chat_temp_enrollment_{profile_slug}",
                )
                verbose = st.toggle("CrewAI verbose logs", value=False, key=f"chat_verbose_{profile_slug}")
            with col_toggles2:
                cache = st.toggle("CrewAI cache", value=True, key=f"chat_cache_{profile_slug}")
                st.toggle(
                    "CrewAI Planning",
                    value=False,
                    disabled=True,
                    help="Reserved for the later Flow/Planning iteration.",
                    key=f"chat_planning_disabled_{profile_slug}",
                )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear this profile's chat history", type="secondary", key=f"chat_clear_{profile_slug}"):
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

    st.markdown("#### ISIS connection")
    if mode == "session" and session.get("client") is not None:
        st.success(f"Session login active. Age: {_age_label(created_at)}")
    else:
        st.info("Using environment fallback for ISIS credentials.")

    username_key = f"chat_isis_username_{profile_slug}"
    password_key = f"chat_isis_password_{profile_slug}"
    col_u, col_p = st.columns(2)
    with col_u:
        username = st.text_input("TUB account", key=username_key, placeholder="e.g. ab123")
    with col_p:
        password = st.text_input("Password", type="password", key=password_key, placeholder="••••••••")

    col_login, col_env, col_clear = st.columns(3)
    if col_login.button("Login", key=f"chat_isis_login_{profile_slug}", use_container_width=True):
        if not username.strip() or not password:
            st.warning("Enter both account and password.")
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
                    st.success("ISIS login active.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"ISIS login failed: {exc}")

    if col_env.button("Use env", key=f"chat_isis_env_{profile_slug}", use_container_width=True):
        sessions[profile_slug] = {"mode": "env", "client": None, "created_at": _now_iso()}
        st.session_state[ISIS_SESSIONS_KEY] = sessions
        st.toast("ISIS env fallback enabled.")
        st.rerun()

    if col_clear.button("Clear session", key=f"chat_isis_clear_{profile_slug}", use_container_width=True):
        sessions.pop(profile_slug, None)
        st.session_state[ISIS_SESSIONS_KEY] = sessions
        st.toast("ISIS session cleared.")
        st.rerun()


def _render_empty_state(profile_slug: str) -> None:
    examples = [
        ("🤖 What ML modules can I still take next semester?", "What ML modules can I still take next semester?"),
        ("🎓 How many credits are still missing in my degree?", "How many credits are still missing in my degree?"),
        ("🔄 Check whether Reinforcement Learning fits my study plan.", "Check whether Reinforcement Learning fits my study plan."),
        ("📅 What deadlines are visible in my current ISIS courses?", "What deadlines are visible in my current ISIS courses?"),
    ]
    st.markdown('<div class="empty-state-header">Suggested Questions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for idx, (label, prompt) in enumerate(examples):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if st.button(label, key=f"example_btn_{idx}_{profile_slug}", use_container_width=True):
                st.session_state[PENDING_PROMPT_KEY] = {"profile_slug": profile_slug, "prompt": prompt}
                st.rerun()


def _render_chat_message(message: dict[str, Any]) -> None:
    with st.chat_message(message.get("role", "assistant")):
        st.markdown(str(message.get("content") or ""))
        if message.get("role") == "assistant":
            workbench = message.get("workbench")
            if workbench:
                _render_trace_panel(workbench, expanded=False)
            proposals = message.get("proposals")
            if proposals:
                _render_proposals_panel(str(message.get("profile_slug") or ""), proposals)
            else:
                pending_write = message.get("pending_write")
                if pending_write:
                    legacy_prop = [{
                        "title": f"Module {pending_write.get('module_query')}",
                        "grade_manager": {
                            "type": "grade_manager",
                            "module_query": pending_write.get("module_query"),
                            "term": pending_write.get("term"),
                            "area": pending_write.get("area"),
                            "program_key": pending_write.get("program_key")
                        },
                        "isis": None
                    }]
                    _render_proposals_panel(str(message.get("profile_slug") or ""), legacy_prop)


def _run_and_render_assistant_turn(profile_slug: str, prompt: str, settings: ChatRuntimeSettings) -> None:
    events = initial_live_trace_events(prompt, settings)
    event_queue: Queue[dict[str, Any]] = Queue()
    student_context = build_student_context(profile_slug)
    isis_client = get_profile_isis_client(profile_slug)

    with st.chat_message("assistant"):
        status = st.status("Crew is coordinating agents...", expanded=True)
        live_placeholder = st.empty()

        def on_trace_event(event: dict[str, Any]) -> None:
            event_queue.put(event)

        try:
            with status:
                with live_placeholder.container():
                    _render_live_trace(events)
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="study-chat-crew") as executor:
                    future = executor.submit(
                        _run_chat_query,
                        profile_slug=profile_slug,
                        prompt=prompt,
                        settings=settings,
                        student_context=student_context,
                        isis_client=isis_client,
                        on_trace_event=on_trace_event if settings.trace_enabled else None,
                    )
                    last_render = 0.0
                    last_heartbeat = 0.0
                    while not future.done():
                        updated = _drain_trace_queue(event_queue, events)
                        now = time.monotonic()
                        if now - last_heartbeat >= 1.2:
                            _append_heartbeat_event(events)
                            updated = True
                            last_heartbeat = now
                        if updated or now - last_render >= 1.0:
                            with live_placeholder.container():
                                _render_live_trace(events)
                            last_render = now
                        time.sleep(0.2)
                    _drain_trace_queue(event_queue, events)
                    with live_placeholder.container():
                        _render_live_trace(events, completed=True)
                    result = future.result()
            status.update(label="Crew finished", state="complete", expanded=False)
            st.markdown(result.answer.rstrip())
            workbench = _workbench_from_result(result)
            if workbench:
                _render_trace_panel(workbench, expanded=True)

            # Extract and display new proposals
            proposals = extract_all_proposals(result, workbench)
            if proposals:
                _render_proposals_panel(profile_slug, proposals)

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
                    "proposals": proposals,
                    "profile_slug": profile_slug,
                },
            )
        except Exception as exc:
            events.append(
                {
                    "event": "ui_error",
                    "agent_label": "Orchestrator",
                    "status": "error",
                    "phase": "failed",
                    "activity": f"Crew run failed: {exc}",
                    "elapsed_ms": _elapsed_ms_from_events(events),
                }
            )
            with live_placeholder.container():
                _render_live_trace(events, completed=True)
            status.update(label="Crew failed", state="error", expanded=True)
            content = f"Could not run the Study Assistant: `{exc}`"
            st.error(content)
            _append_message(profile_slug, {"role": "assistant", "content": content, "created_at": _now_iso()})


def _run_chat_query(
    *,
    profile_slug: str,
    prompt: str,
    settings: ChatRuntimeSettings,
    student_context: str,
    isis_client: MoodleRestClient | None,
    on_trace_event,
) -> MultiAgentStudyAssistantRunResult:
    return run_study_assistant_query(
        query=prompt,
        student_context=student_context,
        allow_temp_enrollment=settings.allow_temp_enrollment,
        profile_slug=profile_slug,
        isis_client=isis_client,
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


def _resolve_module_name(module_query: str, workbench: dict[str, Any] | None) -> str:
    if not workbench:
        return f"Module {module_query}"
    groups = workbench.get("groups") or []
    for group in groups:
        for call in (group.get("tool_calls") or []):
            if call.get("tool_name") in {"get_tu_berlin_moses_module_details", "get_module_details"}:
                tool_input = call.get("tool_input") or {}
                if str(tool_input.get("module_query")).strip() == str(module_query).strip():
                    output = str(call.get("output_preview") or "")
                    first_line = output.split("\n")[0]
                    if first_line.startswith("#"):
                        return first_line.lstrip("#").strip()
    return f"Module {module_query}"


def extract_refused_proposals(workbench: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not workbench:
        return []
    proposals = []
    groups = workbench.get("groups") or []

    for group in groups:
        for call in (group.get("tool_calls") or []):
            if call.get("tool_name") == "Add Module To Study Plan":
                output = str(call.get("output_preview") or "").casefold()
                if "refused" in output or "token" in output:
                    tool_input = call.get("tool_input") or {}
                    module_query = str(tool_input.get("module_query") or "").strip()
                    term = str(tool_input.get("term") or "").strip()
                    if module_query and term:
                        if len(module_query) == 4 and 1900 <= int(module_query) <= 2099:
                            continue
                        module_name = _resolve_module_name(module_query, workbench)
                        proposals.append({
                            "type": "grade_manager",
                            "module_query": module_query,
                            "module_name": module_name,
                            "term": term,
                            "area": tool_input.get("area"),
                            "program_key": tool_input.get("program_key"),
                            "raw_call": call
                        })

    for group in groups:
        for call in (group.get("tool_calls") or []):
            if call.get("tool_name") == "Permanently Enroll In ISIS Course":
                output = str(call.get("output_preview") or "").casefold()
                if "refused" in output or "token" in output:
                    tool_input = call.get("tool_input") or {}
                    course_id = tool_input.get("course_id")
                    course_query = tool_input.get("course_query")
                    course_url = tool_input.get("course_url")
                    expected_title = tool_input.get("expected_title")

                    name = expected_title or course_query or (f"ISIS ID {course_id}" if course_id else "ISIS Course")
                    proposals.append({
                        "type": "isis",
                        "course_id": course_id,
                        "course_query": course_query,
                        "course_url": course_url,
                        "term_hint": tool_input.get("term_hint"),
                        "name": name,
                        "raw_call": call
                    })

    return proposals


def group_proposals(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    gm_props = [p for p in proposals if p["type"] == "grade_manager"]
    isis_props = [p for p in proposals if p["type"] == "isis"]

    matched_isis = set()

    for gm in gm_props:
        title = gm["module_name"]
        query = gm["module_query"]

        best_isis = None
        for isis in isis_props:
            if isis["name"] in matched_isis:
                continue

            isis_name = str(isis["name"]).casefold()
            isis_query = str(isis["course_query"] or "").casefold()
            gm_name = title.casefold()

            is_match = (
                (query and query in isis_name) or
                (query and isis["course_id"] and str(isis["course_id"]) in gm_name) or
                (gm_name and gm_name in isis_name) or
                (isis_name and isis_name in gm_name) or
                (isis_query and isis_query in gm_name)
            )

            if is_match:
                best_isis = isis
                matched_isis.add(isis["name"])
                break

        groups.append({
            "title": title,
            "grade_manager": gm,
            "isis": best_isis
        })

    for isis in isis_props:
        if isis["name"] not in matched_isis:
            groups.append({
                "title": isis["name"],
                "grade_manager": None,
                "isis": isis
            })

    return groups


def extract_all_proposals(result: MultiAgentStudyAssistantRunResult, workbench: dict[str, Any] | None) -> list[dict[str, Any]]:
    proposals = extract_refused_proposals(workbench)
    if proposals:
        return group_proposals(proposals)

    text_prop = extract_pending_study_plan_write_from_text(result.answer)
    if text_prop:
        module_query = text_prop["module_query"]
        module_name = _resolve_module_name(module_query, workbench)
        return [{
            "title": module_name,
            "grade_manager": {
                "type": "grade_manager",
                "module_query": module_query,
                "module_name": module_name,
                "term": text_prop["term"],
                "area": text_prop["area"],
                "program_key": text_prop["program_key"]
            },
            "isis": None
        }]
    return []


def _render_proposals_panel(profile_slug: str, proposals: list[dict[str, Any]]) -> None:
    if not profile_slug or not proposals:
        return

    st.markdown(
        """
        <div class="pending-write-card">
          <div class="pending-write-title">📋 Proposed Study Plan & ISIS Actions</div>
          <div class="pending-write-subtitle">Review the proposed course modifications. You can select actions to approve or decline.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confirmed_items = []

    for idx, group in enumerate(proposals):
        title = group["title"]
        gm = group["grade_manager"]
        isis = group["isis"]

        st.markdown(f"**Course: {html.escape(title)}**")

        col_gm, col_isis = st.columns(2)

        gm_checked = False
        isis_checked = False

        with col_gm:
            if gm:
                term = gm["term"]
                area = gm["area"] or "auto"
                gm_checked = st.checkbox(
                    f"Add to Grade Manager ({term}, Area: {area})",
                    value=True,
                    key=f"confirm_gm_{idx}_{profile_slug}"
                )
            else:
                st.caption("No Grade Manager write proposed.")

        with col_isis:
            if isis:
                course_id = isis["course_id"] or isis["course_query"] or isis["course_url"]
                isis_checked = st.checkbox(
                    f"Enroll on ISIS Moodle (ID/Query: {course_id})",
                    value=True,
                    key=f"confirm_isis_{idx}_{profile_slug}"
                )
            else:
                st.caption("No ISIS enrollment proposed.")

        confirmed_items.append({
            "title": title,
            "grade_manager": gm if gm_checked else None,
            "isis": isis if isis_checked else None,
            "gm_refused": gm,
            "isis_refused": isis
        })
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    feedback = st.text_area(
        "Additional instructions or adjustments (optional)",
        placeholder="e.g. 'I declined course X because I took it already. Suggest a machine learning course instead.'",
        key=f"proposal_feedback_{profile_slug}"
    )

    if st.button("Submit Decisions", type="primary", key=f"submit_proposals_{profile_slug}", use_container_width=True):
        approved_lines = []
        declined_lines = []

        for item in confirmed_items:
            title = item["title"]
            gm = item["grade_manager"]
            isis = item["isis"]
            gm_refused = item["gm_refused"]
            isis_refused = item["isis_refused"]

            if gm:
                area_str = f", Area: {gm['area']}" if gm["area"] else ""
                prog_str = f", Program: {gm['program_key']}" if gm["program_key"] else ""
                approved_lines.append(
                    f"- Grade Manager: Add module '{title}' (query: {gm['module_query']}) for term {gm['term']}{area_str}{prog_str}. "
                    f"Use confirmation_token {STUDY_PLAN_CONFIRMATION_TOKEN}."
                )
            elif gm_refused:
                declined_lines.append(f"- Grade Manager: Do NOT add module '{title}' (query: {gm_refused['module_query']}).")

            if isis:
                id_query = isis["course_id"] or isis["course_query"] or isis["course_url"]
                approved_lines.append(
                    f"- ISIS Moodle: Permanently enroll in course '{title}' (locator: {id_query}). "
                    f"Use confirmation_token {CONFIRMATION_TOKEN}."
                )
            elif isis_refused:
                id_query = isis_refused["course_id"] or isis_refused["course_query"] or isis_refused["course_url"]
                declined_lines.append(f"- ISIS Moodle: Do NOT enroll in course '{title}' (locator: {id_query}).")

        prompt_parts = ["The user reviewed the proposed study plan modifications and ISIS enrollments."]
        if approved_lines:
            prompt_parts.append("\n**Approved Actions:**")
            prompt_parts.extend(approved_lines)
        if declined_lines:
            prompt_parts.append("\n**Declined Actions:**")
            prompt_parts.extend(declined_lines)

        if feedback.strip():
            prompt_parts.append(f"\n**User feedback / instruction:**\n{feedback.strip()}")

        confirm_prompt = "\n".join(prompt_parts)

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


def _render_live_trace(events: list[dict[str, Any]], *, completed: bool = False) -> None:
    workbench = live_workbench_from_events(events)
    st.markdown(_compile_workbench_html(workbench, live=not completed), unsafe_allow_html=True)


def _render_trace_panel(workbench: dict[str, Any], *, expanded: bool) -> None:
    with st.expander("🔍 View Agent Workbench & Tool Trace", expanded=expanded):
        st.markdown(_compile_workbench_html(workbench, live=False), unsafe_allow_html=True)
        _render_artifact_links(workbench)
        _render_tool_expanders(workbench)


def _compile_workbench_html(workbench: dict[str, Any], live: bool = False) -> str:
    phases = workbench.get("phases") or []
    groups = {str(group.get("agent_label")): group for group in (workbench.get("groups") or [])}
    events = workbench.get("latest_events") or []
    total_calls = int(workbench.get("total_tool_calls") or 0)

    phases_html = ""
    for phase in phases:
        status = str(phase.get("status") or "idle")
        label = str(phase.get("label") or "")
        title = str(phase.get("title") or "")
        phases_html += f"""
        <div class="phase-node {status}">
          <div class="phase-num">{html.escape(label)}</div>
          <div class="phase-txt">{html.escape(title)}</div>
        </div>
        """

    flows = workbench.get("source_flow") or []
    flow_html = ""
    for item in flows:
        active_cls = "active" if item.get("active") else ""
        source = str(item.get("source") or "")
        agent = str(item.get("agent") or "")
        flow_html += f"""
        <div class="flow-card {active_cls}">
          <span class="flow-src">{html.escape(source)}</span>
          <span class="flow-connector">➔</span>
          <span class="flow-agt">{html.escape(agent)}</span>
        </div>
        """

    agents_html = ""
    for label in AGENT_LANES:
        group = groups.get(label, {
            "agent_label": label,
            "status": "idle",
            "activity": _default_activity_for_agent(label),
            "llm_calls": 0,
            "tool_calls": [],
            "source_system": _source_for_agent_label(label)
        })
        status = str(group.get("status") or "idle")
        activity = str(group.get("activity") or "Waiting for activity.")
        llm = int(group.get("llm_calls") or 0)
        calls = group.get("tool_calls") or []
        source = str(group.get("source_system") or _source_for_agent_label(label))

        theme_class = label.lower().replace(" ", "-")
        pulse_html = '<span class="pulse-indicator"></span>' if status == "running" or (live and status == "idle" and label == "Orchestrator") else ""

        tools_list_html = ""
        if calls:
            for call in calls:
                tool_name = str(call.get("tool_name") or "")
                duration = call.get("duration_ms")
                dur_str = f"{duration}ms" if duration is not None else "running"
                status_cls = str(call.get("status") or "ok")
                tools_list_html += f"""
                <div class="tool-item {status_cls}">
                  <span class="tool-lbl">⚙️ {html.escape(tool_name)}</span>
                  <span class="tool-dur">{html.escape(dur_str)}</span>
                </div>
                """
        else:
            tools_list_html = '<div class="no-tools">No tool calls yet.</div>'

        agents_html += f"""
        <div class="agent-card {status} {theme_class}">
          <div class="agent-card-header">
            <div>
              <div class="agent-card-name">{pulse_html}{html.escape(label)}</div>
              <div class="agent-card-source">{html.escape(source)}</div>
            </div>
            <span class="status-badge {status}">{status.upper()}</span>
          </div>
          <div class="agent-card-activity">{html.escape(activity)}</div>
          <div class="agent-card-stats">
            <span>{llm} LLM Calls</span>
            <span>{len(calls)} Tools</span>
          </div>
          <div class="agent-card-tools">
            {tools_list_html}
          </div>
        </div>
        """

    events_html = ""
    for event in events[-12:]:
        elapsed = _format_elapsed(event.get("elapsed_ms"))
        agent_label = str(event.get("agent_label") or "Crew")
        activity = str(event.get("activity") or event.get("event") or "")
        status_cls = str(event.get("status") or "ok")
        events_html += f"""
        <div class="log-line {status_cls}">
          <span class="log-time">[{html.escape(elapsed)}]</span>
          <span class="log-agent">&lt;{html.escape(agent_label)}&gt;</span>
          <span class="log-text">{html.escape(activity)}</span>
        </div>
        """

    title_label = "Live Agent Coordination Workbench" if live else "Agent Coordination Workbench & Trace"
    pulse_dot = '<span class="live-dot"></span>' if live else ""

    html_content = f"""
    <div class="workbench-container">
      <div class="workbench-header">
        <div class="workbench-title">{pulse_dot}{title_label}</div>
        <div class="workbench-summary">{total_calls} Total Tool Calls | {len(events)} Events</div>
      </div>

      <!-- Run Phases -->
      <div class="phases-timeline-container">
        <div class="phases-timeline-title">Run Phases</div>
        <div class="phases-timeline">
          {phases_html}
        </div>
      </div>

      <!-- Data Pipeline -->
      <div class="flow-pipeline-container">
        <div class="flow-pipeline-title">Active Data Pipeline</div>
        <div class="flow-pipeline">
          {flow_html}
        </div>
      </div>

      <!-- Agents Grid -->
      <div class="agents-grid">
        {agents_html}
      </div>

      <!-- Live Log Console -->
      <div class="console-container">
        <div class="console-header">
          <div class="console-title">Live Log Console</div>
          <div class="console-status">STREAMING</div>
        </div>
        <div class="console-body">
          {events_html}
        </div>
      </div>
    </div>
    """
    return html_content


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
    groups: dict[str, dict[str, Any]] = {
        label: {
            "agent_label": label,
            "agent_role": None,
            "source_system": _source_for_agent_label(label),
            "tool_calls": [],
            "events": [],
            "duration_ms": 0,
            "status": "idle",
            "activity": _default_activity_for_agent(label),
            "llm_calls": 0,
        }
        for label in AGENT_LANES
    }
    latest_events: list[dict[str, Any]] = []
    for event in events:
        run_id = event.get("run_id") or run_id
        event_name = str(event.get("event") or "")
        label = _event_agent_label(event)
        if label in groups:
            group = groups[label]
            group["events"].append(event)
            group["activity"] = _activity_from_event(event)
            group["status"] = _merge_group_status(str(group.get("status") or "idle"), str(event.get("status") or "ok"))
            group["agent_role"] = event.get("agent_role") or group.get("agent_role")
            if event_name == "llm_started":
                group["llm_calls"] = int(group.get("llm_calls") or 0) + 1
            if event.get("source_system"):
                group["source_system"] = event.get("source_system")
        if event_name not in {"heartbeat"}:
            latest_events.append(event)
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

    for call in sorted(calls_by_id.values(), key=lambda item: int(item.get("call_id") or 0)):
        label = str(call.get("agent_label") or "Unknown Agent")
        group = groups.setdefault(
            label,
            {
                "agent_label": label,
                "agent_role": call.get("agent_role"),
                "source_system": call.get("source_system") or "Other",
                "tool_calls": [],
                "events": [],
                "duration_ms": 0,
                "status": "idle",
                "activity": _default_activity_for_agent(label),
                "llm_calls": 0,
            },
        )
        group["tool_calls"].append(call)
        group["status"] = _combine_status(str(group.get("status") or "ok"), str(call.get("status") or "ok"))
        if call.get("status") == "running":
            group["activity"] = f"Running {call.get('tool_name') or 'tool'}."
        elif call.get("tool_name"):
            group["activity"] = f"Finished {call.get('tool_name')}."
    active_sources = {str(call.get("source_system") or "Other") for call in calls_by_id.values()}
    source_flow = _default_source_flow(active_sources=active_sources, active=bool(calls_by_id) or _has_event(events, "crew_started"))
    _activate_flow_for_lifecycle_events(source_flow, events)
    return {
        "run_id": run_id,
        "run_dir": None,
        "groups": [groups[label] for label in AGENT_LANES if label in groups],
        "total_tool_calls": len(calls_by_id),
        "source_flow": source_flow,
        "phases": _trace_phases_from_events(events, calls_by_id),
        "latest_events": latest_events[-12:],
        "artifacts": {},
    }


def initial_live_trace_events(prompt: str, settings: ChatRuntimeSettings) -> list[dict[str, Any]]:
    del prompt
    temp_status = "enabled" if settings.allow_temp_enrollment else "disabled"
    return [
        {
            "event": "ui_run_started",
            "event_id": 0,
            "elapsed_ms": 0,
            "agent_label": "Orchestrator",
            "phase": "kickoff",
            "status": "running",
            "activity": "Request accepted. Preparing hierarchical crew.",
        },
        {
            "event": "agent_ready",
            "event_id": 0,
            "elapsed_ms": 0,
            "agent_label": "Study Advisor",
            "phase": "ready",
            "status": "idle",
            "activity": "Ready for Grade Manager reads and confirmed study-plan writes.",
            "source_system": "Grade Manager",
        },
        {
            "event": "agent_ready",
            "event_id": 0,
            "elapsed_ms": 0,
            "agent_label": "MOSES Module Researcher",
            "phase": "ready",
            "status": "idle",
            "activity": "Ready for MOSES catalog and degree-structure lookups.",
            "source_system": "MOSES",
        },
        {
            "event": "agent_ready",
            "event_id": 0,
            "elapsed_ms": 0,
            "agent_label": "ISIS Course Info Specialist",
            "phase": "ready",
            "status": "idle",
            "activity": f"Ready for ISIS read-only inspection; temporary enrollment is {temp_status}.",
            "source_system": "ISIS",
        },
    ]


def _drain_trace_queue(event_queue: Queue[dict[str, Any]], events: list[dict[str, Any]]) -> bool:
    updated = False
    while True:
        try:
            events.append(event_queue.get_nowait())
            updated = True
        except Empty:
            return updated


def _append_heartbeat_event(events: list[dict[str, Any]]) -> None:
    elapsed_ms = _elapsed_ms_from_events(events)
    last = next((event for event in reversed(events) if event.get("event") != "heartbeat"), {})
    label = _event_agent_label(last)
    if label not in AGENT_LANES:
        label = "Orchestrator"
    activity = _heartbeat_activity(last)
    if events and events[-1].get("event") == "heartbeat":
        events[-1].update({"elapsed_ms": elapsed_ms, "agent_label": label, "activity": activity})
        return
    events.append(
        {
            "event": "heartbeat",
            "elapsed_ms": elapsed_ms,
            "agent_label": label,
            "phase": "heartbeat",
            "status": "running",
            "activity": activity,
        }
    )


def _elapsed_ms_from_events(events: list[dict[str, Any]]) -> int:
    elapsed = 0
    for event in reversed(events):
        try:
            elapsed = int(event.get("elapsed_ms") or 0)
            break
        except (TypeError, ValueError):
            continue
    return elapsed + 1200


def _source_for_agent_label(label: str) -> str:
    if label == "Study Advisor":
        return "Grade Manager"
    if label == "MOSES Module Researcher":
        return "MOSES"
    if label == "ISIS Course Info Specialist":
        return "ISIS"
    return "CrewAI"


def _default_activity_for_agent(label: str) -> str:
    return {
        "Orchestrator": "Preparing delegation.",
        "Study Advisor": "Waiting for Grade Manager work.",
        "MOSES Module Researcher": "Waiting for MOSES work.",
        "ISIS Course Info Specialist": "Waiting for ISIS work.",
    }.get(label, "Waiting for activity.")


def _event_agent_label(event: dict[str, Any]) -> str:
    label = str(event.get("agent_label") or "")
    if label in AGENT_LANES:
        return label
    role = str(event.get("agent_role") or "").casefold()
    if "study advisor" in role or "personal study advisor" in role:
        return "Study Advisor"
    if "moses" in role or "module researcher" in role:
        return "MOSES Module Researcher"
    if "isis" in role or "course information specialist" in role:
        return "ISIS Course Info Specialist"
    if event.get("event") in {"crew_started", "crew_completed", "crew_failed", "ui_run_started"}:
        return "Orchestrator"
    tool_name = str(event.get("tool_name") or "").casefold()
    if "coworker" in tool_name:
        return "Orchestrator"
    return label or "Orchestrator"


def _activity_from_event(event: dict[str, Any]) -> str:
    activity = str(event.get("activity") or "").strip()
    if activity:
        return activity
    event_name = str(event.get("event") or "")
    if event_name == "tool_start":
        return f"Running {event.get('tool_name') or 'tool'}."
    if event_name == "tool_finish":
        call = event.get("tool_call") or {}
        return f"Finished {call.get('tool_name') or 'tool'}."
    return event_name.replace("_", " ").strip().title() or "Working."


def _merge_group_status(current: str, incoming: str) -> str:
    if incoming in {"error", "warning", "running"}:
        return _combine_status(current, incoming)
    if current in {"error", "warning"} and incoming == "ok":
        return current
    return incoming or current


def _has_event(events: list[dict[str, Any]], name: str) -> bool:
    return any(event.get("event") == name for event in events)


def _activate_flow_for_lifecycle_events(flow: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    active_labels = {_event_agent_label(event) for event in events if event.get("status") == "running"}
    for item in flow:
        agent = str(item.get("agent") or "")
        if agent in active_labels or (agent == "Final Answer" and "Orchestrator" in active_labels):
            item["active"] = True


def _trace_phases_from_events(events: list[dict[str, Any]], calls_by_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    has_started = _has_event(events, "crew_started") or _has_event(events, "ui_run_started")
    has_llm = any(str(event.get("event") or "").startswith("llm_") for event in events)
    has_delegation = any("coworker" in str(call.get("tool_name") or "").casefold() for call in calls_by_id.values())
    has_tools = bool(calls_by_id)
    has_completed = _has_event(events, "crew_completed")
    has_failed = _has_event(events, "crew_failed") or _has_event(events, "ui_error")
    return [
        {"label": "1", "title": "Kickoff", "status": _phase_status(has_started, has_llm or has_tools or has_completed, has_failed)},
        {"label": "2", "title": "Manager reasoning", "status": _phase_status(has_llm, has_delegation or has_tools or has_completed, has_failed)},
        {"label": "3", "title": "Delegation", "status": _phase_status(has_delegation, has_tools or has_completed, has_failed)},
        {"label": "4", "title": "Tool work", "status": _phase_status(has_tools, has_completed, has_failed)},
        {"label": "5", "title": "Final answer", "status": "error" if has_failed else ("done" if has_completed else "idle")},
    ]


def _phase_status(started: bool, finished: bool, failed: bool) -> str:
    if failed:
        return "error"
    if finished:
        return "done"
    if started:
        return "active"
    return "idle"


def _heartbeat_activity(last_event: dict[str, Any]) -> str:
    event_name = str(last_event.get("event") or "")
    if event_name in {"tool_start", "tool_usage_running"}:
        return "Tool is still running."
    if event_name == "llm_started":
        return "LLM call is still running; waiting for the next tool or response."
    if event_name in {"ui_run_started", "crew_started"}:
        return "Orchestrator is still planning or starting the first delegation."
    return "Crew is still working; waiting for the next observable event."


def _format_elapsed(value: object) -> str:
    try:
        millis = int(value or 0)
    except (TypeError, ValueError):
        millis = 0
    seconds = millis / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = int(seconds % 60)
    return f"{minutes}m {remainder}s"


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
    normalized = text.casefold()
    write_markers = (
        "add module",
        "study-plan write",
        "study plan write",
        "confirm",
        "confirmation",
        "eintragen",
        "hinzufügen",
        "hinzufuegen",
        "in deinen plan",
        "in den plan",
    )
    if not any(marker in normalized for marker in write_markers):
        return None
    module_match = re.search(
        r"\b(?:MOSES\s*)?(?:module|modul|modulnummer|module\s*id|moses\s*id)\s*[:#-]?\s*(\d{4,6})\b",
        text,
        flags=re.I,
    )
    term_match = re.search(r"\b(WS\s*\d{2}(?:/\d{2})?|SS\s*\d{2}|WiSe\s*\d{4}(?:/\d{2})?|SoSe\s*\d{4})\b", text, flags=re.I)
    if not module_match or not term_match:
        return None
    module_query = module_match.group(1)
    if len(module_query) == 4 and 1900 <= int(module_query) <= 2099:
        return None
    return {
        "module_query": module_query,
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
        /* Workbench Container */
        .workbench-container {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background-color: #ffffff;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin: 1rem 0 1.5rem 0;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Header block */
        .workbench-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #f8fafc;
            padding: 0.85rem 1.25rem;
            border-bottom: 1px solid #e2e8f0;
        }
        .workbench-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1e293b;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .workbench-summary {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 500;
        }

        /* Green live dot animation */
        .live-dot {
            height: 8px;
            width: 8px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse-dot-key 1.5s infinite;
        }
        @keyframes pulse-dot-key {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }

        /* Timeline and phase classes */
        .phases-timeline-container {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid #f1f5f9;
        }
        .phases-timeline-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }
        .phases-timeline {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .phase-node {
            flex: 1;
            min-width: 130px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            background-color: #f8fafc;
            transition: all 0.2s ease-in-out;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .phase-node.idle {
            background-color: #f8fafc;
            color: #94a3b8;
            border-color: #e2e8f0;
        }
        .phase-node.active {
            border-color: #3b82f6;
            background-color: #eff6ff;
            color: #1e3a8a;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }
        .phase-node.done {
            border-color: #10b981;
            background-color: #ecfdf5;
            color: #065f46;
        }
        .phase-node.error {
            border-color: #ef4444;
            background-color: #fef2f2;
            color: #991b1b;
        }
        .phase-num {
            height: 18px;
            width: 18px;
            border-radius: 50%;
            background-color: rgba(0, 0, 0, 0.05);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 700;
        }
        .phase-node.active .phase-num {
            background-color: #3b82f6;
            color: #ffffff;
        }
        .phase-node.done .phase-num {
            background-color: #10b981;
            color: #ffffff;
        }
        .phase-node.error .phase-num {
            background-color: #ef4444;
            color: #ffffff;
        }
        .phase-txt {
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Flow pipeline styles */
        .flow-pipeline-container {
            padding: 0.75rem 1.25rem;
            background-color: #f8fafc;
            border-bottom: 1px solid #f1f5f9;
        }
        .flow-pipeline-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            color: #94a3b8;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .flow-pipeline {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
        }
        .flow-card {
            border: 1px solid #e2e8f0;
            background-color: #ffffff;
            border-radius: 6px;
            padding: 0.35rem 0.6rem;
            font-size: 0.72rem;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-weight: 500;
            color: #64748b;
        }
        .flow-card.active {
            border-color: #10b981;
            background-color: #ecfdf5;
            color: #047857;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(16, 185, 129, 0.05);
        }
        .flow-src {
            font-weight: 700;
            color: #334155;
        }
        .flow-card.active .flow-src {
            color: #065f46;
        }
        .flow-connector {
            color: #94a3b8;
        }
        .flow-agt {
            color: #475569;
        }
        .flow-card.active .flow-agt {
            color: #047857;
        }

        /* Agents grid */
        .agents-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1rem;
            padding: 1.25rem;
            background-color: #ffffff;
        }
        .agent-card {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            background-color: #ffffff;
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            transition: all 0.2s ease-in-out;
            position: relative;
        }
        .agent-card.running {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }
        .agent-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .agent-card-name {
            font-size: 0.85rem;
            font-weight: 700;
            color: #1e293b;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .agent-card-source {
            font-size: 0.68rem;
            color: #64748b;
            font-weight: 500;
            margin-top: 0.1rem;
        }
        .status-badge {
            font-size: 0.62rem;
            font-weight: 700;
            padding: 0.125rem 0.375rem;
            border-radius: 9999px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .status-badge.idle {
            background-color: #f1f5f9;
            color: #64748b;
            border: 1px solid #cbd5e1;
        }
        .status-badge.running {
            background-color: #eff6ff;
            color: #2563eb;
            border: 1px solid #93c5fd;
        }
        .status-badge.success, .status-badge.ok, .status-badge.done {
            background-color: #ecfdf5;
            color: #059669;
            border: 1px solid #a7f3d0;
        }
        .status-badge.error, .status-badge.failed {
            background-color: #fef2f2;
            color: #dc2626;
            border: 1px solid #fca5a5;
        }
        .agent-card-activity {
            font-size: 0.74rem;
            color: #475569;
            line-height: 1.35;
            min-height: 2.2rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .agent-card-stats {
            display: flex;
            gap: 0.5rem;
            font-size: 0.68rem;
            color: #64748b;
            font-weight: 500;
            border-top: 1px solid #f1f5f9;
            padding-top: 0.5rem;
        }
        .agent-card-stats span {
            background-color: #f8fafc;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
        }
        .agent-card-tools {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }
        .no-tools {
            font-size: 0.68rem;
            color: #94a3b8;
            font-style: italic;
        }

        /* Tool items inside agent cards */
        .tool-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.68rem;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            padding: 0.25rem 0.4rem;
        }
        .tool-item.running {
            border-color: #93c5fd;
            background-color: #eff6ff;
        }
        .tool-item.error {
            border-color: #fca5a5;
            background-color: #fef2f2;
        }
        .tool-lbl {
            font-weight: 600;
            color: #334155;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 140px;
        }
        .tool-dur {
            font-family: monospace;
            color: #64748b;
            font-size: 0.62rem;
        }

        /* Pulsing indicator for running agents */
        .pulse-indicator {
            height: 6px;
            width: 6px;
            background-color: #3b82f6;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
            animation: pulse-indicator-key 1.5s infinite;
        }
        @keyframes pulse-indicator-key {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 4px rgba(59, 130, 246, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
            }
        }

        /* Agent Theme Overrides */
        .agent-card.orchestrator {
            border-left: 3px solid #6366f1;
        }
        .agent-card.study-advisor {
            border-left: 3px solid #10b981;
        }
        .agent-card.moses-module-researcher {
            border-left: 3px solid #14b8a6;
        }
        .agent-card.isis-course-info-specialist {
            border-left: 3px solid #f59e0b;
        }

        /* Console Container (Developer Terminal style) */
        .console-container {
            background-color: #0f172a;
            border-radius: 8px;
            margin: 0.5rem 1.25rem 1.25rem 1.25rem;
            overflow: hidden;
            border: 1px solid #1e293b;
        }
        .console-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #1e293b;
            padding: 0.4rem 0.85rem;
            border-bottom: 1px solid #334155;
        }
        .console-title {
            color: #94a3b8;
            font-size: 0.72rem;
            font-family: monospace;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .console-status {
            color: #ef4444;
            font-family: monospace;
            font-size: 0.65rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.35rem;
            animation: blink-console-key 1.5s step-end infinite;
        }
        @keyframes blink-console-key {
            from, to { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .console-body {
            padding: 0.75rem;
            max-height: 220px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .log-line {
            font-family: Consolas, Monaco, "Lucida Console", "Liberation Mono", "DejaVu Sans Mono", monospace;
            font-size: 0.72rem;
            line-height: 1.4;
            display: flex;
            gap: 0.45rem;
            color: #cbd5e1;
        }
        .log-line.error {
            color: #fca5a5;
        }
        .log-line.warning {
            color: #fde047;
        }
        .log-time {
            color: #64748b;
            flex-shrink: 0;
        }
        .log-agent {
            color: #38bdf8;
            font-weight: 700;
            flex-shrink: 0;
        }
        .log-text {
            word-break: break-all;
        }

        /* Proposals panel styling */
        .pending-write-card {
            border: 1px solid #cbd5e1;
            background-color: #f8fafc;
            border-radius: 10px;
            padding: 1rem;
            margin: 1.25rem 0 0.85rem 0;
        }
        .pending-write-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .pending-write-subtitle {
            font-size: 0.8rem;
            color: #475569;
        }
        .divider {
            height: 1px;
            background-color: #e2e8f0;
            margin: 0.85rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
