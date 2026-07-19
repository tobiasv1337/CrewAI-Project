from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import html
import json
from math import ceil
import os
from pathlib import Path
from queue import Empty, Queue
import re
import time
from typing import Any

import streamlit as st
from markdown_it import MarkdownIt

from core.manager import DegreeManager
from core.models import Module
from core.persistence import load_modules
from core.registry import create_program, list_relevant_programs, list_selectable_programs
from crew.chat_models import ActionDecision, ChatMessage, CourseProposal
from crew.chat_persistence import (
    clear_chat_thread,
    load_chat_thread,
    reset_chat_thread,
    save_chat_thread,
    list_chat_threads,
)
from crew.config.llm import (
    resolve_study_assistant_manager_model,
    resolve_study_assistant_model,
    resolve_study_assistant_observer_model,
)
from crew.isis_client import IsisCredentials, MoodleRestClient, login_via_playwright_sync
from crew.runtime_observer import (
    RuntimeObserverResult,
    deterministic_runtime_observer_report,
    generate_runtime_observer_report,
    reconcile_runtime_observer_report,
)
from crew.semester_context import semester_reference_context
from crew.tools.proposal_tools import ProposalCourseInput, build_course_proposal
from crew.tracing import TraceWorkbench, load_trace_workbench
from main import MultiAgentStudyAssistantRunResult, run_study_assistant_query


CHAT_HISTORY_KEY = "study_chat_messages_by_profile"
ISIS_SESSIONS_KEY = "study_chat_isis_sessions"
PENDING_PROMPT_KEY = "study_chat_pending_prompt"
PROPOSAL_DECISION_PREFIX = "study_chat_course_decision"
DEFAULT_TEMPERATURE = 0.2
AGENT_LANES = [
    "Orchestrator",
    "Study Advisor",
    "Grade Optimization Specialist",
    "MOSES Module Researcher",
    "Degree Regulations Specialist",
    "ISIS Course Info Specialist",
    "Course Commitment Specialist",
]
HIERARCHICAL_ROUTES = {"deep_dive", "recommendation"}
HIERARCHICAL_SPECIALISTS = {
    label for label in AGENT_LANES if label != "Orchestrator"
}
ROUTE_AGENT_MAP = {
    "simple_grade_manager": "Study Advisor",
    "simple_grade_optimization": "Grade Optimization Specialist",
    "simple_moses": "MOSES Module Researcher",
    "simple_isis": "ISIS Course Info Specialist",
    "simple_degree_regulations": "Degree Regulations Specialist",
}
SOURCE_AGENT_MAP = {
    "grade_manager": "Study Advisor",
    "grade_optimization": "Grade Optimization Specialist",
    "moses": "MOSES Module Researcher",
    "degree_regulations": "Degree Regulations Specialist",
    "isis": "ISIS Course Info Specialist",
    "course_commitment": "Course Commitment Specialist",
}


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
    show_agent_chat: bool = False
    observer_model: str | None = None
    observer_enabled: bool = True
    observer_min_interval_seconds: float = 15.0


def _get_active_thread_id(profile_slug: str) -> str:
    key = f"active_thread_id_{profile_slug}"
    if key not in st.session_state:
        st.session_state[key] = "default"
    return st.session_state[key]


def render_chat_page() -> None:
    inject_chat_css()
    profile_slug = str(st.session_state.get("active_profile") or "primary")

    active_tid = _get_active_thread_id(profile_slug)
    thread = load_chat_thread(profile_slug, thread_id=active_tid)
    
    # Initialize UI decision state from on-disk active proposals if not present
    typed_proposals = [_proposal_model(item) for item in thread.active_proposals]
    cards = _course_cards_from_proposals([item for item in typed_proposals if item is not None])
    for card in cards:
        dec_key = _course_decision_key(profile_slug, card)
        if dec_key not in st.session_state:
            statuses = {action.status for action in card["actions"]}
            if "approved" in statuses:
                st.session_state[dec_key] = "accept"
            elif "declined" in statuses:
                st.session_state[dec_key] = "reject"
            else:
                st.session_state[dec_key] = "unsure"
        
        for action in card["actions"]:
            toggle_key = _course_action_toggle_key(profile_slug, card, action.action_id)
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = (action.status != "declined")

    messages = get_profile_messages(profile_slug, thread_id=active_tid)

    st.markdown(
        _clean_html(
            f"""
            <section class="chat-hero">
              <div class="chat-hero-main">
                <div class="chat-hero-kicker"><span></span>Multi-agent study assistant</div>
                <h1>Agent Coordination Workbench</h1>
                <p>Plan your studies, verify degree requirements, explore grade scenarios, and check course information. Inspect the live agent trace whenever you want to see how an answer was produced.</p>
              </div>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )

    # ── Session Selector & Actions ───────────────────────────────────────────
    threads = list_chat_threads(profile_slug)
    if not threads:
        threads = [load_chat_thread(profile_slug, thread_id="default")]
    
    def thread_label(t) -> str:
        if t.thread_id == "default":
            first_user = next((m.content for m in t.messages if m.role == "user"), None)
            if first_user:
                return f"Default: {first_user[:30]}" + ("..." if len(first_user) > 30 else "")
            return "Default Chat"
        first_user = next((m.content for m in t.messages if m.role == "user"), None)
        if first_user:
            return first_user[:35] + ("..." if len(first_user) > 35 else "")
        try:
            dt = datetime.fromisoformat(t.updated_at)
            return f"Chat on {dt.strftime('%b %d, %H:%M')}"
        except Exception:
            return f"Chat: {t.thread_id[:8]}"
            
    thread_options = {t.thread_id: thread_label(t) for t in threads}
    if active_tid not in thread_options:
        t_active = load_chat_thread(profile_slug, thread_id=active_tid)
        threads.insert(0, t_active)
        thread_options[active_tid] = thread_label(t_active)
        
    def on_new_chat(slug: str):
        new_t = reset_chat_thread(slug)
        st.session_state[f"active_thread_id_{slug}"] = new_t.thread_id
        st.session_state[f"chat_session_selector_{slug}"] = new_t.thread_id
        _set_profile_messages(slug, [])
        _clear_all_course_card_state(slug)

    def on_delete_chat(slug: str, tid: str):
        if tid == "default":
            clear_chat_thread(slug, thread_id="default")
        else:
            clear_chat_thread(slug, thread_id=tid)
            st.session_state[f"active_thread_id_{slug}"] = "default"
            st.session_state[f"chat_session_selector_{slug}"] = "default"
        _set_profile_messages(slug, [])
        _clear_all_course_card_state(slug)

    col_sel, col_new, col_runtime, col_del = st.columns([6, 1.15, 1.55, 1])
    with col_sel:
        selected_tid = st.selectbox(
            "Session Selector",
            options=list(thread_options.keys()),
            format_func=lambda tid: thread_options[tid],
            index=list(thread_options.keys()).index(active_tid),
            key=f"chat_session_selector_{profile_slug}",
            label_visibility="collapsed"
        )
        if selected_tid != active_tid:
            st.session_state[f"active_thread_id_{profile_slug}"] = selected_tid
            st.rerun()
            
    with col_new:
        st.button(
            "New chat",
            key=f"chat_new_btn_header_{profile_slug}",
            use_container_width=True,
            type="secondary",
            on_click=on_new_chat,
            args=(profile_slug,)
        )

    with col_runtime:
        settings = _render_chat_config_panel(profile_slug, active_tid=active_tid)
            
    with col_del:
        is_default = (active_tid == "default")
        btn_label = "Clear" if is_default else "Delete"
        st.button(
            btn_label,
            key=f"chat_del_btn_header_{profile_slug}",
            use_container_width=True,
            type="secondary",
            on_click=on_delete_chat,
            args=(profile_slug, active_tid)
        )
            
    st.markdown("<div class='chat-toolbar-spacer'></div>", unsafe_allow_html=True)

    if not messages:
        _render_empty_state(profile_slug)

    clear_pass = st.session_state.get("nm_clear_proposals_flag", False)
    run_active = (PENDING_PROMPT_KEY in st.session_state)

    last_assistant_idx = -1
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_assistant_idx = idx

    for idx, message in enumerate(messages):
        is_latest_assistant = (idx == last_assistant_idx)
        if message.get("role") == "assistant" and settings.show_agent_chat:
            _render_agent_interactions_inline(get_interactions_for_message(message), live=False)
        _render_chat_message(message, is_latest_assistant=is_latest_assistant, run_active=run_active)

    if messages and _message_error_detail(messages[-1]) and not run_active:
        _render_empty_state(profile_slug, recovery=True)

    pending_prompt = st.session_state.get(PENDING_PROMPT_KEY)
    if isinstance(pending_prompt, dict) and pending_prompt.get("profile_slug") == profile_slug and not clear_pass:
        st.session_state.pop(PENDING_PROMPT_KEY)
        prompt_text = str(pending_prompt.get("prompt") or "").strip()
        if prompt_text:
            proposal_decisions = pending_prompt.get("proposal_decisions") or []
            ui_decisions = [ActionDecision.model_validate(d) for d in (pending_prompt.get("ui_decisions") or [])]
            _run_and_render_assistant_turn(
                profile_slug,
                prompt_text,
                settings,
                display_prompt=str(pending_prompt.get("display_prompt") or prompt_text),
                proposal_decisions=proposal_decisions,
                ui_decisions=ui_decisions,
                approved_actions=pending_prompt.get("approved_actions") or [],
                thread_id=active_tid,
            )

    if thread.active_proposals and not run_active and not clear_pass:
        _render_proposals_panel(profile_slug, thread.active_proposals)

    prompt = st.chat_input("Ask the agent team, revise a proposal, or type 'apply selected'…")
    if prompt:
        run_prompt = prompt.strip()
        if run_prompt:
            proposal_decisions = _collect_course_card_decisions(profile_slug, thread.active_proposals)
            ui_decisions = _action_decisions_from_course_card_decisions(proposal_decisions)
            st.session_state[PENDING_PROMPT_KEY] = {
                "profile_slug": profile_slug,
                "prompt": run_prompt,
                "display_prompt": run_prompt,
                "proposal_decisions": proposal_decisions,
                "ui_decisions": [d.model_dump(mode="json") for d in ui_decisions],
                "approved_actions": [],
            }
            st.session_state["nm_clear_proposals_flag"] = True
            st.rerun()

    if clear_pass:
        st.session_state.pop("nm_clear_proposals_flag", None)
        st.rerun()


def _render_chat_config_panel(profile_slug: str, active_tid: str = "default") -> ChatRuntimeSettings:
    del active_tid
    with st.popover(
        "Runtime settings",
        icon=":material/tune:",
        use_container_width=True,
        help="Models, tracing, agent behavior, and ISIS access.",
    ):
        st.caption("Advanced controls for the current profile")
        tab_models, tab_execution, tab_isis = st.tabs(["Models", "Execution", "ISIS"])

        with tab_models:
            st.markdown("**Model roles**")
            configured_model = resolve_study_assistant_model()
            configured_manager_model = resolve_study_assistant_manager_model(
                specialist_model=configured_model
            )
            configured_observer_model = resolve_study_assistant_observer_model(
                manager_model=configured_manager_model,
                specialist_model=configured_model,
            )
            specialist_model = st.text_input(
                "Specialist model override",
                value="",
                placeholder=f".env default: {configured_model}",
                help=(
                    "Optional. Leave empty to use STUDY_ASSISTANT_MODEL from .env for all specialist agents."
                ),
                key=f"chat_specialist_model_{profile_slug}",
            ).strip()
            manager_model = st.text_input(
                "Manager model override",
                value="",
                placeholder=f".env default: {configured_manager_model}",
                help=(
                    "Optional. Leave empty to use STUDY_ASSISTANT_MANAGER_MODEL from .env, "
                    "then the specialist model."
                ),
                key=f"chat_manager_model_{profile_slug}",
            ).strip()
            observer_model = st.text_input(
                "Progress summary model override",
                value="",
                placeholder=f".env default: {configured_observer_model}",
                help=(
                    "Optional lightweight model for intent classification, proposal-decision interpretation, "
                    "and grounded live trace narration. It observes lifecycle evidence but never controls execution."
                ),
                key=f"chat_observer_model_{profile_slug}",
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

        with tab_execution:
            st.markdown("**Trace and execution**")
            trace_mode = st.segmented_control(
                "Tracing level",
                ["Preview", "Full", "Disabled"],
                default="Preview",
                key=f"chat_trace_mode_{profile_slug}",
            )

            allow_temp_enrollment = st.toggle(
                "Temporary ISIS enrollment",
                value=True,
                help="Read-only ISIS tools may enroll briefly, inspect course information, then unenroll.",
                key=f"chat_temp_enrollment_{profile_slug}",
            )
            observer_enabled = st.toggle(
                "LLM progress summaries",
                value=True,
                help=(
                    "Use the lightweight observer model to turn A2A delegation and tool lifecycle events "
                    "into cumulative overall and per-agent progress reports. The first update runs when post-intent "
                    "agent activity appears; later updates are throttled to the configured interval and require new evidence. "
                    "Immediate trace-derived summaries remain visible while the background LLM is pending."
                ),
                key=f"chat_observer_enabled_{profile_slug}",
            )
            env_default_interval = float(os.getenv("OBSERVER_MIN_INTERVAL_SECONDS", "15.0"))
            observer_min_interval_seconds = env_default_interval
            if observer_enabled:
                observer_min_interval_seconds = st.slider(
                    "LLM summary interval (seconds)",
                    min_value=5.0,
                    max_value=120.0,
                    value=env_default_interval,
                    step=5.0,
                    help="Minimum seconds to wait between updates to the live progress summary.",
                    key=f"chat_observer_min_interval_seconds_{profile_slug}",
                )
            show_agent_chat = st.toggle(
                "Show internal agent chat",
                value=False,
                help="Render observable Orchestrator-to-specialist delegation above the final answer.",
                key=f"chat_show_agent_chat_{profile_slug}",
            )
            planning_enabled = st.toggle(
                "CrewAI planning",
                value=False,
                help=(
                    "Add CrewAI's separate planning pass before the hierarchical manager starts. "
                    "The orchestrator already plans and delegates without this optional extra call."
                ),
                key=f"chat_planning_{profile_slug}",
            )
            cache = st.toggle("CrewAI cache", value=True, key=f"chat_cache_{profile_slug}")
            verbose = st.toggle("Verbose lifecycle logs", value=False, key=f"chat_verbose_{profile_slug}")

        with tab_isis:
            _render_isis_account_panel(profile_slug)
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
        planning_enabled=planning_enabled,
        show_agent_chat=show_agent_chat,
        observer_model=observer_model or None,
        observer_enabled=observer_enabled,
        observer_min_interval_seconds=observer_min_interval_seconds,
    )


def _render_isis_account_panel(profile_slug: str) -> None:
    sessions = _isis_sessions()
    session = sessions.get(profile_slug) or {"mode": "env", "created_at": None, "client": None}
    mode = str(session.get("mode") or "env")
    created_at = session.get("created_at")

    st.markdown("**ISIS connection**")
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


def _render_empty_state(profile_slug: str, *, recovery: bool = False) -> None:
    examples = [
        (
            "Plan my next semester",
            "Build a realistic plan for my next semester using my completed modules, degree requirements, current grade forecast, and modules currently offered in MOSES. Explain any assumptions.",
        ),
        (
            "Explore my path to a 1.7",
            "Can I still reach a final grade of 1.7? Compare realistic scenarios, identify the modules with the greatest impact, and explain the constraints.",
        ),
        (
            "Review upcoming deadlines",
            "Check my active Grade Manager courses in ISIS for upcoming deadlines and summarize what I should prioritize.",
        ),
    ]
    heading = "Try another question" if recovery else "What would you like to work on?"
    subtitle = (
        "The previous run is preserved above. Start a fresh request or use one of these examples."
        if recovery
        else "Ask in your own words, or start with an example."
    )
    with st.container(key="chat_prompt_starters"):
        st.markdown(
            _clean_html(
                f"""
                <section class="prompt-starters-heading">
                  <h2>{heading}</h2>
                  <p>{subtitle}</p>
                </section>
                """
            ),
            unsafe_allow_html=True,
        )
        columns = st.columns(len(examples))
        for idx, ((label, prompt), target_col) in enumerate(zip(examples, columns, strict=True)):
            with target_col:
                if st.button(
                    label,
                    key=f"example_btn_{idx}_{profile_slug}",
                    type="tertiary",
                    icon=":material/arrow_outward:",
                    icon_position="right",
                    use_container_width=True,
                ):
                    st.session_state[PENDING_PROMPT_KEY] = {"profile_slug": profile_slug, "prompt": prompt}
                    st.rerun()


def _highlight_json(json_str: str) -> str:
    """Return HTML-safe JSON with lightweight key/value syntax coloring."""

    def replace_key(match: re.Match[str]) -> str:
        return f'__K_START__"{match.group(1)}"__K_END__:'

    def replace_string(match: re.Match[str]) -> str:
        return f': __S_START__"{match.group(1)}"__S_END__'

    def replace_scalar(match: re.Match[str]) -> str:
        return f': __V_START__{match.group(1)}__V_END__'

    protected = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:', replace_key, json_str)
    protected = re.sub(r':\s*"([^"\\]*(?:\\.[^"\\]*)*)"', replace_string, protected)
    protected = re.sub(r':\s*(true|false|null|-?\d+(?:\.\d+)?)', replace_scalar, protected)
    escaped = html.escape(protected)
    return (
        escaped.replace("__K_START__", '<span style="color: #60a5fa; font-weight: 600;">')
        .replace("__K_END__", "</span>")
        .replace("__S_START__", '<span style="color: #10b981;">')
        .replace("__S_END__", "</span>")
        .replace("__V_START__", '<span style="color: #f43f5e; font-weight: 600;">')
        .replace("__V_END__", "</span>")
    )


def _safe_int(val: Any) -> int:
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (TypeError, ValueError):
        if isinstance(val, str):
            digits = "".join(ch for ch in val if ch.isdigit())
            if digits:
                return int(digits)
        return 0




def _message_error_detail(message: dict[str, Any]) -> str | None:
    metadata = message.get("metadata") or {}
    error = metadata.get("error") if isinstance(metadata, dict) else None
    if isinstance(error, dict) and str(error.get("detail") or "").strip():
        return str(error["detail"]).strip()
    content = str(message.get("content") or "")
    prefix = "Could not run the Study Assistant:"
    if content.startswith(prefix):
        return content[len(prefix) :].strip().strip("`")
    return None


def _render_chat_message(message: dict[str, Any], is_latest_assistant: bool = False, run_active: bool = False) -> None:
    del is_latest_assistant, run_active
    error_detail = _message_error_detail(message)
    if message.get("role") == "assistant":
        # Extract workbench from metadata or direct field
        workbench = message.get("workbench") or (message.get("metadata") or {}).get("workbench")
        if workbench:
            _render_trace_panel(workbench, expanded=error_detail is not None)
        elif message.get("trace_dir"):
            st.caption(f"📊 Trace artifacts: `{message.get('trace_dir')}`")

    with st.chat_message(message.get("role", "assistant")):
        if error_detail:
            st.markdown(
                _clean_html(
                    """
                    <div class="run-error-card">
                      <span class="run-error-mark">!</span>
                      <div>
                        <strong>Run interrupted before answer synthesis</strong>
                        <p>The trace above preserves the completed lifecycle events. Retry the prompt after inspecting the technical detail.</p>
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
            with st.expander("Technical failure detail", expanded=False):
                st.code(error_detail, language=None)
        else:
            st.markdown(str(message.get("content") or ""))


def _render_live_answer_status(placeholder: Any, events: list[dict[str, Any]]) -> str:
    """Render a true token preview when safe, otherwise show deterministic run state."""
    preview = _streaming_answer_preview(events)
    if preview:
        placeholder.markdown(f"{preview}\n\n▌")
        return preview

    label, detail = _live_run_status(events)
    observer_report = _latest_observer_report(events)
    observer_meta = (
        '<div class="answer-runtime-observer-label">Live coordination summary</div>'
        if observer_report
        else ""
    )
    placeholder.markdown(
        _clean_html(
            f"""
            <div class="answer-runtime-status">
              <span class="answer-runtime-spinner" aria-hidden="true"></span>
              <div>
                {observer_meta}
                <strong>{html.escape(label)}</strong>
                <span class="answer-runtime-detail">{html.escape(detail)}</span>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    return ""


def _live_run_status(events: list[dict[str, Any]]) -> tuple[str, str]:
    rate_limit_wait = _active_rate_limit_wait(events)
    if rate_limit_wait is not None:
        agent = _event_agent_label(rate_limit_wait)
        remaining = max(0, ceil(float(rate_limit_wait["retry_at_unix"]) - time.time()))
        return (
            "Waiting for the API rate limit",
            f"{agent} will retry in {remaining} seconds after the provider's requested cooldown.",
        )
    observer_report = _latest_observer_report(events)
    if observer_report:
        return (
            str(observer_report.get("headline") or "Coordination update"),
            str(observer_report.get("detail") or "The observer is interpreting the latest trace evidence."),
        )
    intent = _latest_intent(events)
    route = str((intent or {}).get("route") or "")
    last = next(
        (
            event
            for event in reversed(events)
            if event.get("event") not in {"heartbeat", "llm_stream_chunk", "agent_ready"}
        ),
        {},
    )
    event_name = str(last.get("event") or "")
    agent = _event_agent_label(last)
    if event_name == "observer_started":
        return "Updating the coordination summary", "New delegation and execution evidence is being interpreted in the background."
    if event_name == "observer_failed":
        return "Trace-derived progress summary", "The LLM enrichment is temporarily unavailable; verified lifecycle state remains visible."
    if event_name == "flow_turn_persisting":
        return "Finalizing run state", "Persisting the answer, trace metadata, and proposal state."
    if event_name == "intent_classification_started":
        return "Computing execution scope", "The intent router is selecting specialists and source boundaries."
    if event_name == "intent_classified":
        sources = ", ".join(str(item) for item in ((intent or {}).get("required_sources") or []))
        scope = f" Priority sources: {sources}." if sources else ""
        eligibility = (
            " All hierarchical specialists remain eligible for manager delegation."
            if route in HIERARCHICAL_ROUTES
            else ""
        )
        return f"Scope classified as {route or 'unclassified'}", f"Intent routing completed.{scope}{eligibility}"
    if event_name == "route_execution_started":
        return "Preparing the specialist workflow", "The orchestrator is applying the classified source scope and delegation strategy."
    if event_name == "crew_started":
        return "Specialist coordination started", "The orchestrator is preparing the first evidence-driven delegation."
    if event_name in {"task_started", "task_completed"}:
        if event_name == "task_started":
            return f"{agent} analysis started", _agent_card_fallback_activity(agent, {"status": "running"})
        return f"{agent} analysis completed", _agent_card_fallback_activity(agent, {"status": "ok"})
    if event_name == "tool_start":
        source = str(last.get("source_system") or "configured source")
        return f"{agent} is checking {source}", _agent_card_fallback_activity(agent, {"status": "running"})
    if event_name == "tool_finish":
        call = last.get("tool_call") if isinstance(last.get("tool_call"), dict) else {}
        source = str(call.get("source_system") or "configured source")
        return f"{agent} received {source} evidence", "The result is available for the next coordination step."
    if event_name == "llm_started":
        return f"{agent} is interpreting current evidence", _agent_card_fallback_activity(agent, {"status": "running"})
    if event_name == "llm_completed":
        return f"{agent} completed an analysis step", "The result is being incorporated into the next delegation or final synthesis."
    if route:
        return "Coordination in progress", "Awaiting the next verified specialist or tool result."
    return "CrewAI Flow accepted the turn", "Loading context and starting intent classification."


def _active_rate_limit_wait(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return an unfinished provider-directed cooldown, if the trace has one."""
    for event in reversed(events):
        event_name = str(event.get("event") or "")
        if event_name == "llm_rate_limit_retry_started":
            return None
        if event_name != "llm_rate_limit_wait":
            continue
        try:
            if float(event.get("retry_at_unix")) > time.time():
                return event
        except (TypeError, ValueError):
            return None
        return None
    return None


def _latest_observer_report(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        report = event.get("report")
        if event.get("event") == "observer_progress" and isinstance(report, dict):
            try:
                return reconcile_runtime_observer_report(events, report).model_dump(mode="json")
            except Exception:
                try:
                    return deterministic_runtime_observer_report(events).model_dump(mode="json")
                except Exception:
                    return None
    if not any(
        event.get("event") not in {
            "agent_ready",
            "heartbeat",
            "llm_stream_chunk",
            "observer_started",
            "observer_failed",
        }
        for event in events
    ):
        return None
    try:
        return deterministic_runtime_observer_report(events).model_dump(mode="json")
    except Exception:
        return None


def _observer_agent_updates_from_report(report: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    updates: dict[str, dict[str, str]] = {}
    for item in (report or {}).get("agent_updates") or []:
        if not isinstance(item, dict):
            continue
        label, _, _ = _clean_agent_label(str(item.get("agent") or ""))
        summary = " ".join(str(item.get("summary") or "").split())
        if label not in AGENT_LANES or not summary:
            continue
        updates[label] = {
            "summary": summary,
            "state": str(item.get("state") or "active"),
        }
    return updates


def _latest_observer_agent_updates(events: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return _observer_agent_updates_from_report(_latest_observer_report(events))


OBSERVER_TRIGGER_EVENTS = {
    "intent_classified",
    "route_execution_started",
    "crew_started",
    "task_started",
    "task_completed",
    "tool_start",
    "tool_finish",
    "llm_started",
    "llm_completed",
}
OBSERVER_INITIAL_EVIDENCE_EVENTS = {
    "route_execution_started",
    "crew_started",
    "task_started",
    "task_completed",
    "tool_start",
    "tool_finish",
    "llm_started",
    "llm_completed",
}
OBSERVER_MIN_INTERVAL_SECONDS = float(os.getenv("OBSERVER_MIN_INTERVAL_SECONDS", "15.0"))


def _observer_trigger_signature(events: list[dict[str, Any]]) -> str | None:
    relevant = [event for event in events if event.get("event") in OBSERVER_TRIGGER_EVENTS]
    intent_index = next(
        (
            index
            for index in range(len(relevant) - 1, -1, -1)
            if relevant[index].get("event") == "intent_classified"
        ),
        None,
    )
    if intent_index is None or not any(
        event.get("event") in OBSERVER_INITIAL_EVIDENCE_EVENTS
        for event in relevant[intent_index + 1 :]
    ):
        return None
    event = relevant[-1]
    call = event.get("tool_call") if isinstance(event.get("tool_call"), dict) else {}
    return ":".join(
        [
            str(len(relevant)),
            str(event.get("event") or ""),
            str(event.get("call_id") or call.get("call_id") or ""),
            str(_event_agent_label(event) or ""),
            str(event.get("status") or call.get("status") or ""),
        ]
    )


def _append_observer_result(
    events: list[dict[str, Any]],
    future: Future[RuntimeObserverResult],
) -> bool:
    if not future.done():
        return False
    try:
        result = future.result()
        report = result.report.model_dump(mode="json")
        events.append(
            {
                "event": "observer_progress",
                "agent_label": "Runtime Observer",
                "source_system": "Trace Observer",
                "phase": "observation",
                "status": "ok",
                "model": result.model,
                "report": report,
                "activity": f"{report['headline']}: {report['detail']}",
                "elapsed_ms": _elapsed_ms_from_events(events),
            }
        )
    except Exception as exc:
        events.append(
            {
                "event": "observer_failed",
                "agent_label": "Runtime Observer",
                "source_system": "Trace Observer",
                "phase": "observation",
                "status": "warning",
                "activity": "LLM observer unavailable; deterministic lifecycle status remains active.",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": _elapsed_ms_from_events(events),
            }
        )
    return True


def _latest_intent(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        intent = event.get("intent")
        if event.get("event") == "intent_classified" and isinstance(intent, dict):
            return intent
    return None


def _answer_agent_for_intent(intent: dict[str, Any] | None) -> str:
    route = str((intent or {}).get("route") or "")
    return {
        "simple_grade_manager": "Study Advisor",
        "simple_grade_optimization": "Grade Optimization Specialist",
        "simple_moses": "MOSES Module Researcher",
        "simple_isis": "ISIS Course Info Specialist",
        "simple_degree_regulations": "Degree Regulations Specialist",
    }.get(route, "Orchestrator")


def _streaming_answer_preview(events: list[dict[str, Any]]) -> str:
    """Return the current answer candidate from CrewAI's real LLM chunk stream.

    Tool-call chunks reset the candidate. This keeps planning/delegation calls out
    of the answer surface while allowing the final tool-free response to appear
    token by token.
    """
    target_agent = _answer_agent_for_intent(_latest_intent(events))
    buffer = ""
    for event in events:
        if event.get("event") != "llm_stream_chunk":
            continue
        if _event_agent_label(event) != target_agent:
            continue
        if str(event.get("chunk_type") or "text") == "tool_call":
            buffer = ""
            continue
        buffer += str(event.get("content") or "")
    return buffer if len(buffer.strip()) >= 24 else ""


def _animated_answer_chunks(answer: str, streamed_prefix: str = ""):
    """Preserve the answer-writing animation with or without true streaming."""
    text = answer.rstrip()
    prefix = streamed_prefix if streamed_prefix and text.startswith(streamed_prefix) else ""
    if prefix:
        yield prefix
    remainder = text[len(prefix) :]
    pieces = re.split(r"(\s+)", remainder)
    nonempty = [piece for piece in pieces if piece]
    delay = min(0.014, max(0.002, 1.6 / max(len(nonempty), 1)))
    for piece in nonempty:
        yield piece
        time.sleep(delay)


def _run_and_render_assistant_turn(
    profile_slug: str,
    prompt: str,
    settings: ChatRuntimeSettings,
    *,
    display_prompt: str | None = None,
    proposal_decisions: list[dict[str, Any]] | None = None,
    ui_decisions: list[ActionDecision] | None = None,
    approved_actions: list[dict[str, Any]] | list[ActionDecision] | None = None,
    thread_id: str = "default",
) -> None:
    events = initial_live_trace_events(prompt, settings)
    event_queue: Queue[dict[str, Any]] = Queue()
    student_context = build_student_context(profile_slug)
    proposal_context = _format_course_card_decisions_context(proposal_decisions or [])
    if proposal_context:
        student_context = f"{student_context}\n\n{proposal_context}"
    isis_client = get_profile_isis_client(profile_slug)

    with st.chat_message("user"):
        st.markdown(display_prompt or prompt)

    dialogue_placeholder = st.empty() if settings.show_agent_chat else None
    if dialogue_placeholder is not None:
        with dialogue_placeholder.container():
            _render_agent_interactions_inline(extract_agent_interactions(events), live=True)

    trace_placeholder = st.empty() if settings.trace_enabled else None

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        streamed_answer = _render_live_answer_status(answer_placeholder, events)

        def on_trace_event(event: dict[str, Any]) -> None:
            event_queue.put(event)

        observer_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime-observer")
            if settings.observer_enabled
            else None
        )
        observer_future: Future[RuntimeObserverResult] | None = None
        try:
            if trace_placeholder is not None:
                with trace_placeholder.container():
                    _render_live_trace(events)
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="study-chat-crew") as executor:
                future = executor.submit(
                    _run_chat_query,
                    profile_slug=profile_slug,
                    thread_id=thread_id,
                    prompt=prompt,
                    settings=settings,
                    student_context=student_context,
                    isis_client=isis_client,
                    on_trace_event=on_trace_event,
                    approved_actions=approved_actions or [],
                    ui_decisions=ui_decisions or [],
                )
                last_render = 0.0
                last_heartbeat = 0.0
                last_observer_started = -settings.observer_min_interval_seconds
                last_observer_signature: str | None = None
                while not future.done():
                    updated = _drain_trace_queue(event_queue, events)
                    now = time.monotonic()
                    if observer_future is not None and _append_observer_result(events, observer_future):
                        observer_future = None
                        updated = True

                    observer_signature = _observer_trigger_signature(events)
                    if (
                        observer_executor is not None
                        and observer_future is None
                        and observer_signature is not None
                        and observer_signature != last_observer_signature
                        and now - last_observer_started >= settings.observer_min_interval_seconds
                    ):
                        observer_model = resolve_study_assistant_observer_model(
                            observer_model=settings.observer_model,
                            manager_model=settings.manager_model,
                            specialist_model=settings.specialist_model,
                        )
                        observer_input = [dict(event) for event in events]
                        events.append(
                            {
                                "event": "observer_started",
                                "agent_label": "Runtime Observer",
                                "source_system": "Trace Observer",
                                "phase": "observation",
                                "status": "running",
                                "model": observer_model,
                                "activity": "Interpreting the latest delegation and tool lifecycle evidence.",
                                "elapsed_ms": _elapsed_ms_from_events(events),
                            }
                        )
                        observer_future = observer_executor.submit(
                            generate_runtime_observer_report,
                            observer_input,
                            observer_model=settings.observer_model,
                            manager_model=settings.manager_model,
                            specialist_model=settings.specialist_model,
                            study_context=student_context,
                        )
                        last_observer_signature = observer_signature
                        last_observer_started = now
                        updated = True

                    if now - last_heartbeat >= 1.2:
                        _append_heartbeat_event(events)
                        updated = True
                        last_heartbeat = now
                    if updated or now - last_render >= 1.0:
                        streamed_answer = _render_live_answer_status(answer_placeholder, events)
                        if dialogue_placeholder is not None:
                            with dialogue_placeholder.container():
                                _render_agent_interactions_inline(extract_agent_interactions(events), live=True)
                        if trace_placeholder is not None:
                            with trace_placeholder.container():
                                _render_live_trace(events)
                        last_render = now
                    time.sleep(0.2)
                _drain_trace_queue(event_queue, events)
                if observer_future is not None and _append_observer_result(events, observer_future):
                    observer_future = None
                if dialogue_placeholder is not None:
                    with dialogue_placeholder.container():
                        _render_agent_interactions_inline(extract_agent_interactions(events), live=False)
                if trace_placeholder is not None:
                    with trace_placeholder.container():
                        _render_live_trace(events, completed=True)
                result = future.result()

            if observer_executor is not None:
                if observer_future is not None:
                    observer_future.cancel()
                observer_executor.shutdown(wait=False, cancel_futures=True)

            if result.answer.strip():
                with answer_placeholder.container():
                    final_answer = result.answer.rstrip()
                    streamed_answer = _streaming_answer_preview(events) or streamed_answer
                    if streamed_answer == final_answer:
                        st.markdown(final_answer)
                    else:
                        st.write_stream(_animated_answer_chunks(final_answer, streamed_answer))
            else:
                answer_placeholder.markdown("*(No answer returned)*")
            # Build the completed workbench directly from live events to keep all rich details
            workbench = live_workbench_from_events(events, completed=True)
            if result.intent:
                workbench["intent"] = result.intent.model_dump(mode="json")
            if result.trace_dir:
                workbench["artifacts"] = {
                    "report": str(result.trace_dir / "report.md"),
                    "trace": str(result.trace_dir / "trace.jsonl"),
                    "events": str(result.trace_dir / "events.jsonl"),
                    "state": str(result.trace_dir / "state.json"),
                    "summary": str(result.trace_dir / "summary.json"),
                }
            agent_dialogue = extract_agent_interactions(events)
            if agent_dialogue:
                workbench["agent_dialogue"] = agent_dialogue

            # Persist the assistant message with workbench and trace info
            assistant_message = {
                "role": "assistant",
                "content": result.answer.rstrip(),
                "created_at": _now_iso(),
                "trace_dir": str(result.trace_dir) if result.trace_dir else None,
                "metadata": {"workbench": workbench, "agent_dialogue": agent_dialogue} if workbench else {},
            }
            previous_proposals = list(load_chat_thread(profile_slug, thread_id=thread_id).active_proposals)
            course_proposals = resolve_course_proposals(result, workbench)
            _update_or_append_assistant_message(profile_slug, assistant_message, proposals=course_proposals, thread_id=thread_id)
            if result.executed_actions:
                _clear_course_card_state(profile_slug, previous_proposals)

            proposals = _proposal_dicts(course_proposals)
            if proposals:
                _render_proposals_panel(profile_slug, proposals)

            if result.executed_actions or (workbench and trace_has_successful_study_plan_write(workbench)):
                refresh_streamlit_profile_state(profile_slug)
                st.success("Study plan data was updated and reloaded.")
            st.rerun()
        except Exception as exc:
            if observer_executor is not None:
                if observer_future is not None:
                    observer_future.cancel()
                observer_executor.shutdown(wait=False, cancel_futures=True)
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
            if trace_placeholder is not None:
                with trace_placeholder.container():
                    _render_live_trace(events, completed=True)
            error_detail = str(exc)
            content = "Run interrupted before answer synthesis. Inspect the preserved trace and retry the prompt."
            workbench = live_workbench_from_events(events, completed=True)
            error_message = {
                "role": "assistant",
                "content": content,
                "created_at": _now_iso(),
                "metadata": {
                    "workbench": workbench,
                    "error": {
                        "type": type(exc).__name__,
                        "detail": error_detail,
                    },
                },
            }
            st.error(content)
            _append_message(profile_slug, error_message, thread_id=thread_id)
            st.rerun()


def _run_chat_query(
    *,
    profile_slug: str,
    thread_id: str,
    prompt: str,
    settings: ChatRuntimeSettings,
    student_context: str,
    isis_client: MoodleRestClient | None,
    on_trace_event,
    approved_actions: list[dict[str, Any]] | list[ActionDecision] | None = None,
    ui_decisions: list[dict[str, Any]] | list[ActionDecision] | None = None,
) -> MultiAgentStudyAssistantRunResult:
    return run_study_assistant_query(
        query=prompt,
        student_context=student_context,
        allow_temp_enrollment=settings.allow_temp_enrollment,
        profile_slug=profile_slug,
        thread_id=thread_id,
        approved_actions=list(approved_actions or []),
        ui_decisions=list(ui_decisions or []),
        isis_client=isis_client,
        on_trace_event=on_trace_event,
        model=settings.specialist_model,
        manager_model=settings.manager_model,
        observer_model=settings.observer_model,
        planning_enabled=settings.planning_enabled,
        temperature=settings.temperature,
        top_p=settings.top_p,
        trace=settings.trace_enabled,
        trace_full=settings.trace_full,
        verbose=settings.verbose,
        cache=settings.cache,
        stream_answer=True,
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
            if call.get("tool_name") in {"Add Module To Study Plan", "Update Module In Study Plan", "Remove Module From Study Plan"}:
                output = str(call.get("output_preview") or "").casefold()
                if "refused" in output or "token" in output:
                    tool_input = call.get("tool_input") or {}
                    module_query = str(tool_input.get("module_query") or "").strip()
                    term = str(tool_input.get("target_term") or tool_input.get("term") or tool_input.get("current_term") or "").strip()
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


def resolve_course_proposals(
    result: MultiAgentStudyAssistantRunResult,
    workbench: dict[str, Any] | None,
) -> list[CourseProposal]:
    proposals: list[CourseProposal] = []
    seen: set[str] = set()

    for item in result.proposed_actions or []:
        proposal = _proposal_model(item)
        if proposal and proposal.proposal_id not in seen:
            proposals.append(proposal)
            seen.add(proposal.proposal_id)

    for proposal in extract_explicit_course_proposals(workbench):
        if proposal.proposal_id not in seen:
            proposals.append(proposal)
            seen.add(proposal.proposal_id)

    return proposals


def extract_explicit_course_proposals(workbench: dict[str, Any] | None) -> list[CourseProposal]:
    if not workbench:
        return []

    proposals: list[CourseProposal] = []
    seen: set[str] = set()
    for call in _all_tool_calls(workbench):
        if not _is_explicit_course_proposal_tool(call.get("tool_name")):
            continue
        tool_input = call.get("tool_input") or {}
        proposal_title = str(tool_input.get("proposal_title") or "").strip()
        proposal_summary = str(tool_input.get("proposal_summary") or "").strip()
        raw_courses = tool_input.get("courses") or []
        if not proposal_title or not isinstance(raw_courses, list) or not raw_courses:
            continue
        try:
            courses = [
                course if isinstance(course, ProposalCourseInput) else ProposalCourseInput.model_validate(course)
                for course in raw_courses
            ]
            proposal = build_course_proposal(
                proposal_title=proposal_title,
                proposal_summary=proposal_summary,
                courses=courses,
                source_agent=str(call.get("agent_label") or "Course Commitment Specialist"),
            )
        except Exception:
            continue
        if proposal.actions and proposal.proposal_id not in seen:
            proposals.append(proposal)
            seen.add(proposal.proposal_id)
    return proposals


def _is_explicit_course_proposal_tool(tool_name: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(tool_name or "").casefold()).strip("_")
    return normalized == "propose_course_actions_for_confirmation"


def _proposal_dicts(proposals: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for proposal in proposals or []:
        if isinstance(proposal, CourseProposal):
            result.append(proposal.model_dump(mode="json"))
        elif isinstance(proposal, dict):
            result.append(proposal)
        elif hasattr(proposal, "model_dump"):
            result.append(proposal.model_dump(mode="json"))
    return result


def _proposal_model(proposal: Any) -> CourseProposal | None:
    try:
        if isinstance(proposal, CourseProposal):
            return proposal
        if isinstance(proposal, dict) and "actions" in proposal:
            return CourseProposal.model_validate(proposal)
    except Exception:
        return None
    return None


def _render_proposals_panel(profile_slug: str, proposals: list[Any]) -> None:
    if not profile_slug or not proposals:
        return

    typed_proposals = [_proposal_model(item) for item in proposals]
    typed_proposals = [item for item in typed_proposals if item is not None]

    if not typed_proposals:
        _render_legacy_proposals_panel(profile_slug, proposals)
        return

    course_cards = _course_cards_from_proposals(typed_proposals)
    if not course_cards:
        return

    st.markdown(
        _clean_html(
            f"""
            <section class="course-choice-header">
              <div>
                <div class="course-choice-kicker">Pending Choices</div>
                <div class="course-choice-title">Suggested courses</div>
              </div>
              <span>{len(course_cards)} course{'s' if len(course_cards) != 1 else ''}</span>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(course_cards), 3):
        row_cards = course_cards[row_start : row_start + 3]
        columns = st.columns(len(row_cards), gap="medium")
        for column, card in zip(columns, row_cards):
            with column:
                _render_course_choice_card(profile_slug, card)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    proposal_decisions = _collect_course_card_decisions(profile_slug, typed_proposals)
    ui_decisions = _action_decisions_from_course_card_decisions(proposal_decisions)
    has_decisions = len(ui_decisions) > 0

    col_submit, col_clear, _ = st.columns([1.2, 1.2, 1.6])
    with col_submit:
        if st.button(
            "Confirm & Execute",
            key=f"submit_proposals_panel_{profile_slug}",
            type="primary",
            use_container_width=True,
            disabled=not has_decisions,
            help="Submit decisions for execution (adds accepted courses, removes rejected ones)."
        ):
            st.session_state[PENDING_PROMPT_KEY] = {
                "profile_slug": profile_slug,
                "prompt": "apply selected",
                "display_prompt": "Executing confirmed actions...",
                "proposal_decisions": proposal_decisions,
                "ui_decisions": [d.model_dump(mode="json") for d in ui_decisions],
                "approved_actions": [],
            }
            st.session_state["nm_clear_proposals_flag"] = True
            st.rerun()

    with col_clear:
        if st.button(
            "Clear Suggestions",
            key=f"clear_proposals_panel_btn_{profile_slug}",
            type="secondary",
            use_container_width=True,
            help="Discard these recommendations without executing."
        ):
            st.session_state[PENDING_PROMPT_KEY] = {
                "profile_slug": profile_slug,
                "prompt": "discard active proposals",
                "display_prompt": "Discarding suggestions...",
                "proposal_decisions": [],
                "ui_decisions": [],
                "approved_actions": [],
            }
            st.session_state["nm_clear_proposals_flag"] = True
            st.rerun()


def _course_cards_from_proposals(proposals: list[CourseProposal]) -> list[dict[str, Any]]:
    cards_by_key: dict[str, dict[str, Any]] = {}
    for proposal in proposals:
        for action in proposal.actions:
            if action.status not in {"proposed", "approved", "declined", "needs_clarification"}:
                continue
            key = _course_card_identity(action)
            card = cards_by_key.setdefault(
                key,
                {
                    "card_id": key,
                    "proposal_id": proposal.proposal_id,
                    "proposal_ids": [],
                    "proposal_title": proposal.title,
                    "proposal_summary": proposal.summary,
                    "course_title": action.course_title,
                    "source_agent": proposal.source_agent,
                    "actions": [],
                    "evidence": [],
                },
            )
            card["proposal_id"] = proposal.proposal_id
            if proposal.proposal_id not in card["proposal_ids"]:
                card["proposal_ids"].append(proposal.proposal_id)
            card["proposal_title"] = proposal.title or card["proposal_title"]
            card["proposal_summary"] = proposal.summary or card["proposal_summary"]
            card["source_agent"] = proposal.source_agent or card["source_agent"]
            existing_idx = next((idx for idx, existing in enumerate(card["actions"]) if existing.kind == action.kind), -1)
            if existing_idx >= 0:
                card["actions"][existing_idx] = action
            else:
                card["actions"].append(action)
            card["evidence"].extend(action.evidence or [])
            card["evidence"].extend(proposal.evidence or [])
    for card in cards_by_key.values():
        card["evidence"] = _dedupe_text(card["evidence"])
    return list(cards_by_key.values())


def _render_course_choice_card(profile_slug: str, card: dict[str, Any]) -> None:
    decision_key = _course_decision_key(profile_slug, card)
    selected = st.session_state.get(decision_key)
    selected_class = str(selected or "unsure")
    metadata = _course_card_metadata(card)
    evidence = " · ".join(str(item) for item in card.get("evidence", [])[:2])
    action_chips = "".join(
        f'<span class="course-choice-chip {_action_kind_class(action.kind)}">{html.escape(_action_kind_label(action.kind))}</span>'
        for action in card["actions"]
    )
    meta_html = "".join(
        f"""
        <div class="course-choice-meta-item">
          <span>{html.escape(label)}</span>
          <strong>{html.escape(value)}</strong>
        </div>
        """
        for label, value in metadata
    )
    st.markdown(
        _clean_html(
            f"""
            <section class="course-choice-card {html.escape(selected_class)}">
              <div class="course-choice-card-top">
                <span>{html.escape(str(card.get("proposal_title") or "Proposal"))}</span>
                <strong>{html.escape(str(card["course_title"]))}</strong>
              </div>
              <div class="course-choice-chips">{action_chips}</div>
              <div class="course-choice-meta">{meta_html}</div>
              {f'<div class="course-choice-evidence">{html.escape(evidence)}</div>' if evidence else ''}
            </section>
            """
        ),
        unsafe_allow_html=True,
    )

    button_cols = st.columns(3, gap="small")
    for col, decision, label in zip(button_cols, ["accept", "unsure", "reject"], ["Accept", "Unsure", "Decline"]):
        with col:
            if st.button(
                label,
                key=f"{decision_key}_{decision}",
                type=("primary" if selected == decision else "secondary"),
                use_container_width=True,
            ):
                st.session_state[decision_key] = decision
                st.rerun()

    action_toggle_actions = [
        action
        for action in card["actions"]
        if action.kind in {"grade_manager_add", "grade_manager_update", "grade_manager_remove", "isis_resolve", "isis_enroll"}
    ]
    if len(action_toggle_actions) > 1:
        toggle_cols = st.columns(len(action_toggle_actions), gap="small")
        for col, action in zip(toggle_cols, action_toggle_actions):
            toggle_key = _course_action_toggle_key(profile_slug, card, action.action_id)
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = True
            with col:
                st.toggle(
                    _action_kind_short_label(action.kind),
                    key=toggle_key,
                    help=_action_kind_label(action.kind),
                )


def _course_card_metadata(card: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    term = None
    area = None
    module = None
    isis_id = None
    for action in card["actions"]:
        if action.kind in {"grade_manager_add", "grade_manager_update", "grade_manager_remove"}:
            payload = action.grade_manager_payload or {}
            term = term or payload.get("target_term") or payload.get("term") or payload.get("current_term")
            area = area or payload.get("target_area") or payload.get("area") or payload.get("current_area")
            module = module or payload.get("module_query")
        elif action.kind in {"isis_resolve", "isis_enroll"}:
            payload = action.isis_payload or {}
            term = term or payload.get("term_hint")
            isis_id = isis_id or payload.get("course_id")
    if term:
        values.append(("Term", str(term)))
    if area:
        values.append(("Area", str(area)))
    if module:
        values.append(("Module", str(module)))
    if isis_id:
        values.append(("ISIS ID", str(isis_id)))
    return values[:4]


def _collect_course_card_decisions(profile_slug: str, proposals: list[Any]) -> list[dict[str, Any]]:
    typed_proposals = [_proposal_model(item) for item in proposals]
    cards = _course_cards_from_proposals([item for item in typed_proposals if item is not None])
    decisions: list[dict[str, Any]] = []
    for card in cards:
        selected = str(st.session_state.get(_course_decision_key(profile_slug, card)) or "unsure")
        action_items = []
        for action in card["actions"]:
            enabled = bool(st.session_state.get(_course_action_toggle_key(profile_slug, card, action.action_id), True))
            action_items.append(
                {
                    "action_id": action.action_id,
                    "kind": action.kind,
                    "enabled": enabled,
                }
            )
        decisions.append(
            {
                "card_id": card.get("card_id"),
                "proposal_id": card["proposal_id"],
                "proposal_ids": card.get("proposal_ids", []),
                "proposal_title": card["proposal_title"],
                "course_title": card["course_title"],
                "decision": selected if selected in {"accept", "reject", "unsure"} else "unsure",
                "actions": action_items,
            }
        )
    return decisions


def _action_decisions_from_course_card_decisions(decisions: list[dict[str, Any]]) -> list[ActionDecision]:
    action_decisions: list[ActionDecision] = []
    for item in decisions:
        decision = str(item.get("decision") or "unsure")
        if decision not in {"accept", "reject"}:
            continue
        for action in item.get("actions") or []:
            enabled = bool(action.get("enabled", True))
            action_decisions.append(
                ActionDecision(
                    action_id=str(action.get("action_id") or ""),
                    approved=(decision == "accept" and enabled),
                    feedback=(
                        f"Course card decision: {decision}"
                        if enabled or decision == "reject"
                        else "Course card action disabled by user."
                    ),
                )
            )
    return [decision for decision in action_decisions if decision.action_id]


def _format_course_card_decisions_context(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return ""
    compact = [
        {
            "course_title": item.get("course_title"),
            "decision": item.get("decision") or "unsure",
            "actions": [
                {
                    "kind": action.get("kind"),
                    "enabled": bool(action.get("enabled", True)),
                }
                for action in (item.get("actions") or [])
            ],
        }
        for item in decisions
    ]
    return "Structured course-card decision payload from the UI:\n" + json.dumps(compact, ensure_ascii=False, indent=2)


def _course_decision_key(profile_slug: str, card: dict[str, Any]) -> str:
    card_id = str(card.get("card_id") or f"{card.get('proposal_id', 'default')}:{card['course_title']}")
    return f"{PROPOSAL_DECISION_PREFIX}_{profile_slug}_{_slugify(card_id)}"


def _course_action_toggle_key(profile_slug: str, card: dict[str, Any], action_id: str) -> str:
    return f"{_course_decision_key(profile_slug, card)}_{action_id}_enabled"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return slug[:48] or "course"


def _dedupe_text(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result


def _course_card_identity(action: Any) -> str:
    grade_payload = getattr(action, "grade_manager_payload", None) or {}
    isis_payload = getattr(action, "isis_payload", None) or {}
    moses_identity = (
        grade_payload.get("moses_module_number")
        or isis_payload.get("moses_module_number")
        or _numeric_identity(grade_payload.get("module_query"))
    )
    if moses_identity:
        return f"moses:{_identity_text(moses_identity)}"
    title = _identity_text(getattr(action, "course_title", ""))
    if title:
        return f"title:{title}"
    course_id = isis_payload.get("course_id")
    if course_id:
        return f"isis-id:{_identity_text(course_id)}"
    course_url = isis_payload.get("course_url")
    if course_url:
        return f"isis-url:{_identity_text(course_url)}"
    course_query = isis_payload.get("course_query")
    if course_query:
        return f"isis-query:{_identity_text(course_query)}"
    return f"action:{getattr(action, 'action_id', 'unknown')}"


def _identity_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().strip().split())


def _numeric_identity(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if text.isdigit() else None


def _render_legacy_proposals_panel(profile_slug: str, proposals: list[Any]) -> None:
    st.warning("Legacy proposal format detected. Ask the assistant to regenerate course actions.")


def _action_kind_label(kind: str) -> str:
    if kind == "grade_manager_add":
        return "Study Plan Add"
    if kind == "grade_manager_update":
        return "Study Plan Update"
    if kind == "grade_manager_remove":
        return "Study Plan Remove"
    if kind == "isis_resolve":
        return "ISIS Resolve + Enroll"
    if kind == "isis_enroll":
        return "ISIS Enrollment"
    return str(kind).replace("_", " ").title()


def _action_kind_short_label(kind: str) -> str:
    if kind == "grade_manager_add":
        return "Plan"
    if kind == "grade_manager_update":
        return "Update"
    if kind == "grade_manager_remove":
        return "Remove"
    if kind == "isis_resolve":
        return "ISIS"
    if kind == "isis_enroll":
        return "ISIS"
    return str(kind).replace("_", " ").title()


def _action_kind_class(kind: str) -> str:
    if kind in {"grade_manager_add", "grade_manager_update", "grade_manager_remove"}:
        return "grade-manager"
    if kind in {"isis_resolve", "isis_enroll"}:
        return "isis"
    return "other"


def _current_settings_from_state(profile_slug: str) -> ChatRuntimeSettings:
    trace_mode = str(st.session_state.get(f"chat_trace_mode_{profile_slug}") or "Preview")
    env_default_interval = float(os.getenv("OBSERVER_MIN_INTERVAL_SECONDS", "15.0"))
    return ChatRuntimeSettings(
        specialist_model=str(st.session_state.get(f"chat_specialist_model_{profile_slug}") or "").strip() or None,
        manager_model=str(st.session_state.get(f"chat_manager_model_{profile_slug}") or "").strip() or None,
        temperature=float(st.session_state.get(f"chat_temperature_{profile_slug}") or DEFAULT_TEMPERATURE),
        top_p=float(st.session_state[f"chat_top_p_{profile_slug}"]) if st.session_state.get(f"chat_top_p_enabled_{profile_slug}") else None,
        trace_enabled=trace_mode != "Disabled",
        trace_full=trace_mode == "Full",
        verbose=bool(st.session_state.get(f"chat_verbose_{profile_slug}", False)),
        cache=bool(st.session_state.get(f"chat_cache_{profile_slug}", True)),
        allow_temp_enrollment=bool(st.session_state.get(f"chat_temp_enrollment_{profile_slug}", True)),
        planning_enabled=bool(st.session_state.get(f"chat_planning_{profile_slug}", False)),
        show_agent_chat=bool(st.session_state.get(f"chat_show_agent_chat_{profile_slug}", False)),
        observer_model=str(st.session_state.get(f"chat_observer_model_{profile_slug}") or "").strip() or None,
        observer_enabled=bool(st.session_state.get(f"chat_observer_enabled_{profile_slug}", True)),
        observer_min_interval_seconds=float(st.session_state.get(f"chat_observer_min_interval_seconds_{profile_slug}", env_default_interval)),
    )


def extract_agent_interactions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract observable manager-to-specialist delegation from trace events."""
    interactions: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for index, event in enumerate(events or []):
        event_name = str(event.get("event") or "")
        if event_name == "tool_start" and _is_coworker_tool(event.get("tool_name")):
            call_id = str(_safe_int(event.get("call_id")) or f"start-{index}")
            interaction = _interaction_from_tool_payload(
                tool_name=event.get("tool_name"),
                tool_input=event.get("tool_input"),
                sender=_event_agent_label(event),
                call_id=call_id,
                status="running",
                elapsed_ms=event.get("elapsed_ms"),
                started_at=event.get("started_at") or event.get("emitted_at"),
            )
            interactions[call_id] = interaction
            order.append(call_id)
        elif event_name == "tool_finish":
            call = dict(event.get("tool_call") or {})
            if not _is_coworker_tool(call.get("tool_name")):
                continue
            call_id = str(_safe_int(call.get("call_id")) or f"finish-{index}")
            existing = interactions.get(call_id)
            interaction = _interaction_from_tool_payload(
                tool_name=call.get("tool_name"),
                tool_input=call.get("tool_input"),
                sender=call.get("agent_label") or _event_agent_label(event),
                call_id=call_id,
                status=str(call.get("status") or "completed"),
                elapsed_ms=event.get("elapsed_ms"),
                started_at=call.get("started_at"),
                finished_at=event.get("finished_at") or event.get("emitted_at"),
                duration_ms=call.get("duration_ms"),
                response=call.get("output") or call.get("output_preview"),
            )
            if existing:
                existing.update(
                    {
                        key: value
                        for key, value in interaction.items()
                        if value is not None and value != "" and value != []
                    }
                )
                existing["status"] = "completed" if existing.get("status") in {"ok", "running"} else existing.get("status")
            else:
                interactions[call_id] = interaction
                order.append(call_id)
        elif event_name == "tool_call" and _is_coworker_tool(event.get("tool_name")):
            call_id = str(_safe_int(event.get("call_id")) or f"call-{index}")
            interaction = _interaction_from_tool_payload(
                tool_name=event.get("tool_name"),
                tool_input=event.get("tool_input"),
                sender=event.get("agent_label") or _event_agent_label(event),
                call_id=call_id,
                status=str(event.get("status") or "completed"),
                elapsed_ms=event.get("elapsed_ms"),
                started_at=event.get("started_at"),
                finished_at=event.get("finished_at"),
                duration_ms=event.get("duration_ms"),
                response=event.get("output") or event.get("output_preview"),
            )
            interactions[call_id] = interaction
            order.append(call_id)

    ordered_keys = list(dict.fromkeys(order))
    agent_updates = _latest_observer_agent_updates(events)
    ordered = [
        _normalize_interaction_status(interactions[key])
        for key in ordered_keys
        if key in interactions
    ]
    latest_for_receiver: dict[str, int] = {}
    for index, interaction in enumerate(ordered):
        latest_for_receiver[str(interaction.get("receiver") or "")] = index
    for index, interaction in enumerate(ordered):
        update = agent_updates.get(str(interaction.get("receiver") or ""))
        if update and latest_for_receiver.get(str(interaction.get("receiver") or "")) == index:
            interaction["observer_summary"] = update["summary"]
            interaction["observer_state"] = (
                "completed" if interaction.get("status") == "completed" else update["state"]
            )
    return ordered


def get_interactions_for_message(
    message: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if events:
        return extract_agent_interactions(events)

    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    stored = metadata.get("agent_dialogue") or message.get("agent_dialogue")
    if isinstance(stored, list):
        return [_normalize_interaction_status(item) for item in stored if isinstance(item, dict)]

    workbench = metadata.get("workbench") or message.get("workbench")
    if isinstance(workbench, dict):
        stored = workbench.get("agent_dialogue")
        if isinstance(stored, list):
            return [_normalize_interaction_status(item) for item in stored if isinstance(item, dict)]
        interactions = extract_agent_interactions_from_workbench(workbench)
        if interactions:
            return interactions

    return extract_agent_interactions(_load_events_from_trace_dir(message.get("trace_dir")))


def extract_agent_interactions_from_workbench(workbench: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not workbench:
        return []
    stored = workbench.get("agent_dialogue")
    if isinstance(stored, list):
        return [_normalize_interaction_status(item) for item in stored if isinstance(item, dict)]
    events: list[dict[str, Any]] = []
    for call in _all_tool_calls(workbench):
        if _is_coworker_tool(call.get("tool_name")):
            event = dict(call)
            event.setdefault("event", "tool_call")
            events.append(event)
    return extract_agent_interactions(events)


def _load_events_from_trace_dir(trace_dir: str | Path | None) -> list[dict[str, Any]]:
    if not trace_dir:
        return []
    run_dir = Path(trace_dir)
    events_path = run_dir / "events.jsonl"
    trace_path = events_path if events_path.exists() else run_dir / "trace.jsonl"
    if not trace_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _interaction_from_tool_payload(
    *,
    tool_name: Any,
    tool_input: Any,
    sender: Any,
    call_id: str,
    status: str,
    elapsed_ms: Any = None,
    started_at: Any = None,
    finished_at: Any = None,
    duration_ms: Any = None,
    response: Any = None,
) -> dict[str, Any]:
    input_data = _coerce_mapping(tool_input)
    sender_label, sender_avatar, sender_class = _clean_agent_label(str(sender or "Orchestrator"))
    receiver_label, receiver_avatar, receiver_class = _clean_agent_label(_receiver_from_tool_input(input_data))
    
    question_full = _delegation_request_from_tool_input(input_data)
    response_full = "" if response is None else str(response).strip()
    
    return {
        "call_id": call_id,
        "tool_name": str(tool_name or ""),
        "sender": sender_label,
        "sender_avatar": sender_avatar,
        "sender_class": sender_class,
        "receiver": receiver_label,
        "receiver_avatar": receiver_avatar,
        "receiver_class": receiver_class,
        "question": _preview_dialogue_text(question_full, limit=3000),
        "question_full": question_full,
        "response": _preview_dialogue_text(response_full, limit=5000),
        "response_full": response_full,
        "status": status,
        "elapsed_ms": _optional_int(elapsed_ms),
        "duration_ms": _optional_int(duration_ms),
        "started_at": str(started_at) if started_at else None,
        "finished_at": str(finished_at) if finished_at else None,
    }


def _is_coworker_tool(tool_name: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(tool_name or "").casefold()).strip("_")
    return "coworker" in normalized or normalized in {
        "delegate_work",
        "delegate_work_to_coworker",
        "ask_question_to_coworker",
    }


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {}


def _receiver_from_tool_input(tool_input: dict[str, Any]) -> str:
    for key in ("coworker", "coworker_name", "coworker_role", "recipient", "agent", "role"):
        value = str(tool_input.get(key) or "").strip()
        if value:
            return value
    raw = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
    for label in AGENT_LANES:
        if label != "Orchestrator" and label.casefold() in raw.casefold():
            return label
    return "Specialist Agent"


def _delegation_request_from_tool_input(tool_input: dict[str, Any]) -> str:
    request_parts: list[str] = []
    for key in ("question", "task", "request", "assignment", "message"):
        value = str(tool_input.get(key) or "").strip()
        if value and value not in request_parts:
            request_parts.append(value)
    if not request_parts:
        raw = str(tool_input.get("raw") or "").strip()
        if raw:
            request_parts.append(raw)
    context = str(tool_input.get("context") or "").strip()
    if context and not any(context in part for part in request_parts):
        request_parts.append(f"Context: {context}")
    return "\n\n".join(request_parts) or "Delegation request captured without readable text."


def _clean_agent_label(role_or_label: str) -> tuple[str, str, str]:
    text = str(role_or_label or "").strip()
    lowered = text.casefold()
    if "orchestrator" in lowered or "manager" in lowered:
        return "Orchestrator", "🧭", "orchestrator"
    if "study advisor" in lowered or "personal study advisor" in lowered:
        return "Study Advisor", "🎓", "study-advisor"
    if "grade optimization" in lowered or "grade optimizer" in lowered:
        return "Grade Optimization Specialist", "📈", "grade-optimization"
    if "moses" in lowered or "module researcher" in lowered:
        return "MOSES Module Researcher", "🔎", "moses"
    if "degree regulations" in lowered or "regulations specialist" in lowered or "stupo" in lowered:
        return "Degree Regulations Specialist", "📜", "degree-regulations"
    if "isis" in lowered or "course information specialist" in lowered:
        return "ISIS Course Info Specialist", "📚", "isis"
    if "commitment" in lowered:
        return "Course Commitment Specialist", "✅", "commitment"
    return text or "Specialist Agent", "🤖", "specialist"


def _normalize_interaction_status(interaction: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(interaction)
    status = str(normalized.get("status") or "").casefold()
    if status in {"ok", "complete", "completed", "success"}:
        normalized["status"] = "completed"
    elif status in {"error", "failed"}:
        normalized["status"] = "error"
    elif status == "warning":
        normalized["status"] = "warning"
    else:
        normalized["status"] = "running"
    normalized.setdefault("sender", "Orchestrator")
    normalized.setdefault("receiver", "Specialist Agent")
    normalized.setdefault("question", "")
    normalized.setdefault("response", "")
    if not normalized.get("sender_avatar"):
        sender, avatar, css_class = _clean_agent_label(str(normalized.get("sender") or "Orchestrator"))
        normalized["sender"] = sender
        normalized["sender_avatar"] = avatar
        normalized["sender_class"] = css_class
    if not normalized.get("receiver_avatar"):
        receiver, avatar, css_class = _clean_agent_label(str(normalized.get("receiver") or "Specialist Agent"))
        normalized["receiver"] = receiver
        normalized["receiver_avatar"] = avatar
        normalized["receiver_class"] = css_class
    return normalized


def _preview_dialogue_text(value: Any, *, limit: int) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n...[truncated {len(text) - limit} chars]"


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return _safe_int(value)


def _render_agent_interactions_inline(interactions: list[dict[str, Any]], *, live: bool = False) -> None:
    if not interactions:
        if live:
            with st.container(border=True):
                st.markdown("**Agent Team Collaboration**")
                st.caption("Waiting for the orchestrator to delegate work to a specialist agent.")
        return

    with st.container(border=True):
        title_suffix = " · live" if live else ""
        st.markdown(f"**Agent Team Collaboration{title_suffix}**")
        for interaction in interactions:
            item = _normalize_interaction_status(interaction)
            sender = str(item.get("sender") or "Orchestrator")
            receiver = str(item.get("receiver") or "Specialist Agent")
            duration = item.get("duration_ms")
            duration_label = f" · {duration} ms" if duration is not None else ""

            with st.chat_message(sender, avatar=str(item.get("sender_avatar") or "🧭")):
                st.caption(f"Delegation to {receiver}{duration_label}")
                st.markdown(str(item.get("question") or "Delegation request captured without readable text."))

            with st.chat_message(receiver, avatar=str(item.get("receiver_avatar") or "🤖")):
                status = str(item.get("status") or "running")
                observer_summary = str(item.get("observer_summary") or "").strip()
                if observer_summary:
                    st.caption("Result summary" if status == "completed" else "Progress summary")
                    st.markdown(observer_summary)
                if status == "completed":
                    st.caption(f"Response to {sender}")
                    st.markdown(str(item.get("response") or "Completed without a response preview."))
                elif status == "error":
                    st.caption(f"Response to {sender}")
                    st.error(str(item.get("response") or "The delegated work failed."))
                elif status == "warning":
                    st.caption(f"Response to {sender}")
                    st.warning(str(item.get("response") or "The delegated work completed with warnings."))
                else:
                    if not observer_summary:
                        st.caption(f"Working for {sender}")
                        st.info(f"Awaiting {receiver}'s traced response to this delegation.")


def _render_agent_interactions_panel(interactions: list[dict[str, Any]], *, live: bool = False) -> None:
    st.markdown(_compile_agent_dialogue_html(interactions, live=live), unsafe_allow_html=True)


def _render_dialogue_body_html(preview_text: str, full_text: str | None = None) -> str:
    md = MarkdownIt("gfm-like")
    preview_str = str(preview_text or "").strip()
    full_str = str(full_text or "").strip()
    
    if not full_str or len(full_str) <= len(preview_str):
        return md.render(preview_str) if preview_str else ""
        
    preview_html = md.render(preview_str)
    full_html = md.render(full_str)
    
    import uuid
    uniq = uuid.uuid4().hex[:6]
    
    return f"""
    <details class="agent-dialogue-expandable" id="details-{uniq}">
      <summary class="agent-dialogue-expand-trigger">
        <div class="agent-dialogue-preview-text">{preview_html}</div>
        <span class="agent-dialogue-expand-label show-more">[Show full message ({len(full_str) - len(preview_str)} more chars)]</span>
        <span class="agent-dialogue-expand-label show-less" style="display: none;">[Hide full message]</span>
      </summary>
      <div class="agent-dialogue-full-text">{full_html}</div>
    </details>
    """


def _compile_agent_dialogue_html(interactions: list[dict[str, Any]], *, live: bool = False) -> str:
    if not interactions:
        message = (
            "Waiting for the orchestrator to delegate work to a specialist agent."
            if live
            else "No agent-to-agent delegation was captured for this turn."
        )
        return _clean_html(
            f"""
            <div class="agent-dialogue-panel">
              <div class="agent-dialogue-header">
                <div class="agent-dialogue-title">Agent Dialogue</div>
                <div class="agent-dialogue-count">0 exchanges</div>
              </div>
              <div class="agent-dialogue-empty">{html.escape(message)}</div>
            </div>
            """
        )

    exchange_html = ""
    for interaction in interactions:
        item = _normalize_interaction_status(interaction)
        sender = str(item.get("sender") or "Orchestrator")
        receiver = str(item.get("receiver") or "Specialist Agent")
        sender_class = str(item.get("sender_class") or "orchestrator")
        receiver_class = str(item.get("receiver_class") or "specialist")
        sender_avatar = str(item.get("sender_avatar") or "🧭")
        receiver_avatar = str(item.get("receiver_avatar") or "🤖")
        
        question_html = _render_dialogue_body_html(
            item.get("question") or "Delegation request captured without readable text.",
            item.get("question_full")
        )
        response = item.get("response") or ""
        response_full = item.get("response_full")
        status = str(item.get("status") or "running")
        duration = item.get("duration_ms")
        duration_html = f'<span>{html.escape(str(duration))} ms</span>' if duration is not None else ""
        status_label = "working" if status == "running" else status
        observer_summary = str(item.get("observer_summary") or "").strip()
        observer_label = "Result summary" if status == "completed" else "Progress summary"
        observer_html = (
            '<div class="agent-dialogue-observer-summary">'
            f'<span>{observer_label}</span>'
            f'<p>{html.escape(observer_summary)}</p>'
            '</div>'
            if observer_summary
            else ""
        )

        if status in {"completed", "warning", "error"}:
            response_html = observer_html + _render_dialogue_body_html(
                response or "Completed without a response preview.",
                response_full,
            )
        else:
            response_html = observer_html or '<em>Delegated task active; awaiting its traced tool or response event.</em>'
        exchange_html += f"""
        <div class="agent-dialogue-pair">
          <div class="agent-dialogue-message request {html.escape(sender_class)}">
            <div class="agent-dialogue-avatar">{html.escape(sender_avatar)}</div>
            <div class="agent-dialogue-bubble">
              <div class="agent-dialogue-meta">
                <strong>{html.escape(sender)}</strong>
                <span>to {html.escape(receiver)}</span>
                {duration_html}
              </div>
              <div class="agent-dialogue-body">{question_html}</div>
            </div>
          </div>
          <div class="agent-dialogue-message response {html.escape(receiver_class)} {html.escape(status)}">
            <div class="agent-dialogue-avatar">{html.escape(receiver_avatar)}</div>
            <div class="agent-dialogue-bubble">
              <div class="agent-dialogue-meta">
                <strong>{html.escape(receiver)}</strong>
                <span>{html.escape(status_label)}</span>
              </div>
              <div class="agent-dialogue-body">{response_html}</div>
            </div>
          </div>
        </div>
        """

    live_badge = '<span class="agent-dialogue-live">LIVE</span>' if live else ""
    return _clean_html(
        f"""
        <div class="agent-dialogue-panel">
          <div class="agent-dialogue-header">
            <div class="agent-dialogue-title">Agent Dialogue {live_badge}</div>
            <div class="agent-dialogue-count">{len(interactions)} exchange{'s' if len(interactions) != 1 else ''}</div>
          </div>
          <div class="agent-dialogue-list">{exchange_html}</div>
        </div>
        """
    )


def _dialogue_text_html(text: str) -> str:
    # Kept as fallback for any external caller, but dialogue body html now renders markdown.
    md = MarkdownIt("gfm-like")
    return md.render(text or "")


def _render_live_trace(events: list[dict[str, Any]], *, completed: bool = False) -> None:
    workbench = live_workbench_from_events(events, completed=completed)
    _render_trace_panel(workbench, expanded=not completed, live=not completed)


def _render_trace_panel(workbench: dict[str, Any], *, expanded: bool, live: bool = False) -> None:
    title = "Agent Coordination Workbench & Trace" if not live else "Live Agent Coordination Workbench & Trace"
    with st.expander(title, expanded=expanded):
        dashboard_tab, dialogue_tab, raw_tab = st.tabs(
            ["Orchestration & Topology", "Internal Agent Chat (A2A)", "Raw Trace & Tool I/O"]
        )
        with dashboard_tab:
            st.markdown(_compile_workbench_html(workbench, live=live), unsafe_allow_html=True)
        with dialogue_tab:
            _render_agent_interactions_panel(extract_agent_interactions_from_workbench(workbench), live=live)
        with raw_tab:
            _render_raw_trace_tab(workbench, live=live)


def _render_raw_trace_tab(workbench: dict[str, Any], *, live: bool = False) -> None:
    groups = workbench.get("groups") or []
    events = workbench.get("latest_events") or []
    calls = _all_tool_calls(workbench)
    llm_calls = sum(int(group.get("llm_calls") or 0) for group in groups)
    total_tokens = sum(int(group.get("total_tokens") or 0) for group in groups)

    col_events, col_llm, col_tools, col_tokens = st.columns(4)
    col_events.metric("Trace events", int(workbench.get("event_count") or len(events)))
    col_llm.metric("LLM calls", llm_calls)
    col_tools.metric("Tool calls", len(calls))
    col_tokens.metric("Observed tokens", total_tokens or "—")

    st.caption(
        "Technical runtime view. Status is derived from CrewAI lifecycle events; tool inputs and outputs follow the configured trace redaction/preview level."
    )
    if calls:
        for call in sorted(calls, key=lambda item: _safe_int(item.get("call_id"))):
            label = (
                f"#{_safe_int(call.get('call_id'))} · {call.get('agent_label') or 'Unknown Agent'} · "
                f"{call.get('tool_name') or 'Unknown Tool'} · {call.get('status') or 'unknown'}"
            )
            with st.expander(label, expanded=False):
                st.markdown("**Input**")
                st.json(call.get("tool_input") or {}, expanded=1)
                st.markdown("**Output preview**")
                st.markdown(str(call.get("output_preview") or "*(No output captured yet.)*"))
    elif live:
        st.info("No tool call has been emitted yet. Intent classification or manager reasoning may still be running.")
    else:
        st.caption("No tool calls were captured for this route.")

    with st.expander("Latest normalized lifecycle events", expanded=False):
        st.json(events, expanded=1)


def _clean_html(html_str: str) -> str:
    return "\n".join(line.strip() for line in html_str.split("\n") if line.strip())


def _compact_workbench_topology(
    flows: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    *,
    live: bool,
) -> list[dict[str, Any]]:
    """Render topology as participants, not a repetitive lifecycle timeline."""
    order: list[str] = []
    latest: dict[str, dict[str, Any]] = {}
    for item in flows:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent") or "").strip()
        if not agent or agent == "Runtime Observer":
            continue
        if agent not in latest:
            order.append(agent)
        latest[agent] = dict(item)

    if "Orchestrator" in groups and "Orchestrator" not in latest:
        order.insert(0, "Orchestrator")
        latest["Orchestrator"] = {"agent": "Orchestrator"}
    if not live and "Final Answer" not in latest:
        order.append("Final Answer")
        latest["Final Answer"] = {"agent": "Final Answer", "status": "done"}

    compact: list[dict[str, Any]] = []
    for agent in order:
        item = latest[agent]
        group = groups.get(agent)
        if group:
            status = str(group.get("status") or item.get("status") or "idle")
            active = status == "running"
        elif agent == "Final Answer":
            status = "queued" if live else "done"
            active = not live
        else:
            status = str(item.get("status") or "idle")
            active = bool(item.get("active"))
        compact.append({"agent": agent, "status": status, "active": active})
    return compact


def _workbench_execution_sequence(
    workbench: dict[str, Any],
    groups: dict[str, dict[str, Any]],
    *,
    live: bool,
) -> tuple[list[dict[str, Any]], bool]:
    """Prefer the real A2A delegation order; fall back to a compact participant topology."""
    interactions = [
        item
        for item in (workbench.get("agent_dialogue") or [])
        if isinstance(item, dict)
    ]
    if not interactions:
        return (
            _compact_workbench_topology(
                workbench.get("topology") or workbench.get("source_flow") or [],
                groups,
                live=live,
            ),
            False,
        )

    sequence: list[dict[str, Any]] = [
        {
            "agent": "Orchestrator",
            "status": groups.get("Orchestrator", {}).get("status") or ("running" if live else "ok"),
            "active": str(groups.get("Orchestrator", {}).get("status") or "") == "running",
        }
    ]
    invocation_counts: dict[str, int] = {}
    for interaction in interactions:
        receiver, _, _ = _clean_agent_label(str(interaction.get("receiver") or ""))
        if receiver not in AGENT_LANES or receiver == "Orchestrator":
            continue
        invocation_counts[receiver] = invocation_counts.get(receiver, 0) + 1
        interaction_status = str(interaction.get("status") or "running")
        status = {
            "completed": "ok",
            "warning": "warning",
            "error": "error",
        }.get(interaction_status, "running")
        sequence.append(
            {
                "agent": receiver,
                "status": status,
                "active": status == "running",
                "invocation": invocation_counts[receiver],
            }
        )
    sequence.append(
        {
            "agent": "Final Answer",
            "status": "queued" if live else "done",
            "active": not live,
        }
    )
    return sequence, True


def _agent_card_fallback_activity(label: str, group: dict[str, Any]) -> str:
    status = str(group.get("status") or "idle")
    active = status == "running"
    copy = {
        "Orchestrator": (
            "Coordinating specialist work and assembling the final response.",
            "Specialist coordination and final synthesis completed.",
        ),
        "Study Advisor": (
            "Reviewing the study record, open requirements, and completion timeline.",
            "Study record, degree progress, and completion timeline reviewed.",
        ),
        "Grade Optimization Specialist": (
            "Calculating grade scenarios and sensitivity to remaining assessments.",
            "Grade scenarios and assessment sensitivity calculated.",
        ),
        "MOSES Module Researcher": (
            "Reviewing module, course, and assessment information in MOSES.",
            "Relevant MOSES module and assessment information reviewed.",
        ),
        "Degree Regulations Specialist": (
            "Checking binding degree rules, grade weighting, and deadlines.",
            "Binding degree rules, grade weighting, and deadlines checked.",
        ),
        "ISIS Course Info Specialist": (
            "Checking current ISIS course spaces, deadlines, and assignments.",
            "Relevant ISIS course information checked.",
        ),
        "Course Commitment Specialist": (
            "Preparing study-plan changes that require explicit approval.",
            "Approval-dependent study-plan changes reviewed.",
        ),
    }
    if label in copy and status in {"running", "ok", "completed", "done"}:
        return copy[label][0 if active else 1]
    return str(group.get("activity") or _default_activity_for_agent(label))


def _compile_workbench_html(workbench: dict[str, Any], live: bool = False) -> str:
    phases = workbench.get("phases") or []
    groups = {str(group.get("agent_label")): group for group in (workbench.get("groups") or [])}
    events = workbench.get("latest_events") or []
    total_calls = int(workbench.get("total_tool_calls") or 0)
    total_llm_calls = sum(int(group.get("llm_calls") or 0) for group in groups.values())
    total_tokens = sum(int(group.get("total_tokens") or 0) for group in groups.values())
    event_count = int(workbench.get("event_count") or len(events))
    elapsed = _format_elapsed(max((_safe_int(event.get("elapsed_ms")) for event in events), default=0))

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

    flows, has_delegation_sequence = _workbench_execution_sequence(workbench, groups, live=live)
    flow_html = ""
    for idx, item in enumerate(flows):
        active_cls = "active" if item.get("active") else ""
        flow_status = str(item.get("status") or "idle")
        agent = str(item.get("agent") or "")
        connector_html = '<span class="flow-connector">→</span>' if idx > 0 else ""
        invocation = _safe_int(item.get("invocation"))
        invocation_html = (
            f'<span class="flow-invocation">delegation #{invocation}</span>'
            if invocation > 1
            else ""
        )
        flow_html += f"""
        <div class="flow-node-wrap">
          {connector_html}
          <div class="flow-card {active_cls} {html.escape(flow_status)}">
            <span class="flow-step">{idx + 1:02d}</span>
            <span class="flow-agt">{html.escape(agent)}</span>
            {invocation_html}
            <span class="flow-state">{html.escape(flow_status)}</span>
          </div>
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
            "source_system": _source_for_agent_label(label),
            "models": [],
            "total_tokens": 0,
            "stream_chunks": 0,
            "selection": "available",
        })
        status = str(group.get("status") or "idle")
        observer_summary = str(group.get("observer_summary") or "").strip()
        observer_state = str(group.get("observer_state") or "")
        activity = observer_summary or _agent_card_fallback_activity(label, group)
        activity_label_html = (
            '<div class="agent-card-observer-label">'
            f'{"Result summary" if observer_state == "completed" else "Progress summary"}'
            '</div>'
            if observer_summary
            else ""
        )
        llm = int(group.get("llm_calls") or 0)
        tokens = int(group.get("total_tokens") or 0)
        stream_chunks = int(group.get("stream_chunks") or 0)
        calls = group.get("tool_calls") or []
        source = str(group.get("source_system") or "").strip()
        if not source or source.casefold() in {"other", "unknown", "unknown source"}:
            source = _source_for_agent_label(label)
        models = [str(model) for model in (group.get("models") or []) if str(model).strip()]
        model_html = (
            f'<code class="agent-card-model">{html.escape(models[-1])}</code>'
            if models
            else (
                '<span class="agent-card-model pending">model pending</span>'
                if live and status in {"idle", "queued", "running"}
                else '<span class="agent-card-model pending">model not captured</span>'
            )
        )

        theme_class = label.lower().replace(" ", "-")
        pulse_html = '<span class="pulse-indicator"></span>' if status == "running" or (live and status == "idle" and label == "Orchestrator") else ""
        status_label = {
            "not_selected": "excluded by intent",
            "queued": "queued",
            "skipped": "selected · not invoked",
            "eligible": "eligible",
            "not_invoked": "eligible · not invoked",
            "idle": "available",
            "ok": "completed",
        }.get(status, status)

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
            no_tools_label = {
                "not_selected": "No tool calls — outside the intent scope.",
                "queued": "No tool calls — awaiting manager delegation.",
                "skipped": "No tool calls — selected but not invoked.",
                "eligible": "No tool calls — available to the hierarchical manager.",
                "not_invoked": "No tool calls — eligible but not invoked in this run.",
            }.get(status, "No tool calls yet.")
            tools_list_html = f'<div class="no-tools">{html.escape(no_tools_label)}</div>'

        agents_html += f"""
        <div class="agent-card {status} {theme_class}">
          <div class="agent-card-header">
            <div>
              <div class="agent-card-name">{pulse_html}{html.escape(label)}</div>
              <div class="agent-card-source">{html.escape(source)}</div>
            </div>
            <span class="status-badge {status}">{html.escape(status_label)}</span>
          </div>
          {model_html}
          {activity_label_html}
          <div class="agent-card-activity">{html.escape(activity)}</div>
          <div class="agent-card-stats">
            <span>{llm} LLM</span>
            <span>{len(calls)} tools</span>
            <span>{tokens if tokens else '—'} tokens</span>
            <span>{stream_chunks} chunks</span>
          </div>
          <div class="agent-card-tools">
            {tools_list_html}
          </div>
        </div>
        """

    title_label = "Live Agent Coordination Workbench" if live else "Agent Coordination Workbench & Trace"
    pulse_dot = '<span class="live-dot"></span>' if live else ""

    # Compile artifacts inside the workbench container if not live
    artifacts_html = ""
    if not live:
        artifacts = {key: value for key, value in (workbench.get("artifacts") or {}).items() if value}
        if artifacts:
            # Create single-line artifact hint
            artifact_path = next(iter(artifacts.values())) if artifacts else ""
            if artifact_path:
                import os
                artifact_dir = os.path.dirname(str(artifact_path))
                artifacts_html = f"""
                <div class="artifacts-container" style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #cbd5e1; font-size: 0.85rem; color: #475569;">
                  <strong>Trace Artifacts:</strong> <code style="font-size: 0.85rem; background: #f1f5f9; padding: 0.1rem 0.3rem; border-radius: 4px;">{html.escape(str(artifact_dir))}</code>
                </div>
                """

    intent_html = ""
    intent = workbench.get("intent")
    if intent:
        route = intent.get("route") or "Unknown"
        complexity = intent.get("complexity") or "Unknown"
        rationale = intent.get("rationale") or ""
        sources = ", ".join(intent.get("required_sources") or [])
        tool_budget = intent.get("tool_budget")
        write_intent = bool(intent.get("write_intent"))
        sources_html = (
            f'<div class="intent-sources"><strong>Required sources:</strong> {html.escape(sources)}</div>'
            if sources
            else ""
        )
        rationale_html = (
            f'<div class="intent-rationale">Rationale: {html.escape(rationale)}</div>'
            if rationale
            else ""
        )
        intent_html = f"""
        <div class="intent-banner">
          <div class="intent-banner-head">
            <span>Intent route <code class="intent-route">{html.escape(route)}</code></span>
            <span class="intent-complexity">{html.escape(complexity)} · {'write' if write_intent else 'read-only'}{f' · budget {html.escape(str(tool_budget))}' if tool_budget is not None else ''}</span>
          </div>
          {sources_html}
          {rationale_html}
        </div>
        """

    observer_html = ""
    observer_report = workbench.get("observer_report")
    if isinstance(observer_report, dict):
        observer_headline = str(observer_report.get("headline") or "Run in progress")
        observer_detail = str(observer_report.get("detail") or "")
        observer_html = f"""
        <section class="observer-overview">
          <span>Live coordination summary</span>
          <strong>{html.escape(observer_headline)}</strong>
          <p>{html.escape(observer_detail)}</p>
        </section>
        """

    html_content = f"""
    <div class="workbench-container">
      <div class="workbench-header">
        <div class="workbench-title">{pulse_dot}{title_label}</div>
        <div class="workbench-summary">{elapsed} · {total_llm_calls} LLM · {total_calls} tools · {event_count} events · {total_tokens if total_tokens else '—'} tokens</div>
      </div>

      {intent_html}
      {observer_html}

      <!-- Run Phases -->
      <div class="phases-timeline-container">
        <div class="phases-timeline-title">Run Phases</div>
        <div class="phases-timeline">
          {phases_html}
        </div>
      </div>

      <!-- Data Pipeline -->
      <div class="flow-pipeline-container">
        <div class="flow-pipeline-title">{'A2A delegation sequence' if has_delegation_sequence else 'Intent-scoped execution topology'}</div>
        <div class="flow-pipeline">
          {flow_html}
        </div>
      </div>

      <!-- Agents Grid -->
      <div class="agents-grid">
        {agents_html}
      </div>

      {artifacts_html}
    </div>
    """
    return _clean_html(html_content)


def live_workbench_from_events(events: list[dict[str, Any]], completed: bool = False) -> dict[str, Any]:
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
            "stream_chunks": 0,
            "models": [],
            "total_tokens": 0,
            "selection": "available",
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
            if event_name == "llm_stream_chunk":
                group["stream_chunks"] = int(group.get("stream_chunks") or 0) + 1
            else:
                group["events"].append(event)
                group["activity"] = _activity_from_event(event)
                group["status"] = _merge_group_status(str(group.get("status") or "idle"), str(event.get("status") or "ok"))
                group["agent_role"] = event.get("agent_role") or group.get("agent_role")
                if event_name in {"llm_started", "observer_started"}:
                    group["llm_calls"] = int(group.get("llm_calls") or 0) + 1
                model = str(event.get("model") or "").strip()
                if model and model not in group["models"]:
                    group["models"].append(model)
                group["total_tokens"] = int(group.get("total_tokens") or 0) + _usage_total_tokens(event.get("usage"))
                if event.get("source_system"):
                    group["source_system"] = event.get("source_system")
        if event_name not in {"heartbeat", "llm_stream_chunk"}:
            latest_events.append(event)
        if event.get("event") == "tool_start":
            call_id = _safe_int(event.get("call_id"))
            calls_by_id[call_id] = {
                "call_id": call_id,
                "tool_name": event.get("tool_name"),
                "tool_input": event.get("tool_input") or {},
                "agent_role": event.get("agent_role"),
                "agent_label": _event_agent_label(event) or "Orchestrator",
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
                call_id = _safe_int(call.get("call_id"))
                if call_id in calls_by_id:
                    existing_label = calls_by_id[call_id].get("agent_label")
                    if existing_label and existing_label != "Unknown Agent" and not call.get("agent_label"):
                        call["agent_label"] = existing_label
                if not call.get("agent_label") or call.get("agent_label") == "Unknown Agent":
                    call["agent_label"] = _event_agent_label(event) or _event_agent_label(call) or "Orchestrator"
                calls_by_id[call_id] = call

    for call in sorted(calls_by_id.values(), key=lambda item: _safe_int(item.get("call_id"))):
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
                "stream_chunks": 0,
                "models": [],
                "total_tokens": 0,
                "selection": "available",
            },
        )
        group["tool_calls"].append(call)
        group["status"] = _combine_status(str(group.get("status") or "ok"), str(call.get("status") or "ok"))
        if call.get("status") == "running":
            group["activity"] = f"Running {call.get('tool_name') or 'tool'}."
        elif call.get("tool_name"):
            group["activity"] = f"Finished {call.get('tool_name')}."

    agent_dialogue = extract_agent_interactions(events)
    latest_delegation_by_receiver: dict[str, dict[str, Any]] = {}
    for interaction in agent_dialogue:
        receiver = str(interaction.get("receiver") or "")
        if receiver:
            latest_delegation_by_receiver[receiver] = interaction

    has_crew_completed = completed or any(e.get("event") == "crew_completed" for e in events)
    has_crew_failed = any(e.get("event") in {"crew_failed", "ui_error"} for e in events)

    for label, group in groups.items():
        agent_events = group.get("events") or []
        has_task_completed = any(e.get("event") == "task_completed" for e in agent_events)
        has_task_failed = any(e.get("event") == "task_failed" for e in agent_events)

        if label == "Orchestrator":
            if has_crew_failed:
                group["status"] = "error"
            elif has_crew_completed:
                group["status"] = "ok"
            elif any(e.get("status") == "running" for e in agent_events):
                group["status"] = "running"
            else:
                group["status"] = "idle"
        else:
            if has_task_failed:
                group["status"] = "error"
            elif has_task_completed:
                group["status"] = "ok"
            elif has_crew_completed:
                if group.get("tool_calls"):
                    group["status"] = "ok"
                else:
                    group["status"] = "idle"
            elif has_crew_failed:
                has_tool_error = any(c.get("status") == "error" for c in group.get("tool_calls", []))
                if has_tool_error:
                    group["status"] = "error"
                elif group.get("tool_calls"):
                    group["status"] = "ok"
                else:
                    group["status"] = "idle"
            else:
                is_running = (
                    any(e.get("event") in {"task_started", "llm_started", "tool_start"} or e.get("status") == "running" for e in agent_events) or
                    any(c.get("status") == "running" for c in group.get("tool_calls", []))
                )
                if is_running:
                    group["status"] = "running"
                else:
                    group["status"] = "idle"

            # The enclosing A2A response is the user-visible completion
            # boundary. Internal task/tool completion cannot move an agent to
            # completed while its delegation bubble still has no response.
            delegation = latest_delegation_by_receiver.get(label)
            if delegation and not has_crew_completed:
                delegation_status = str(delegation.get("status") or "running")
                if delegation_status == "completed":
                    group["status"] = "ok"
                    group["activity"] = "Response returned to the orchestrator."
                elif delegation_status == "error":
                    group["status"] = "error"
                elif delegation_status == "warning":
                    group["status"] = "warning"
                else:
                    group["status"] = "running"
                    group["activity"] = "Delegated analysis is active; no A2A response has returned yet."

    intent = None
    for event in events:
        if event.get("event") == "intent_classified":
            intent = event.get("intent")
    selected_agents = _selected_agents_for_intent(intent)
    priority_agents = _priority_agents_for_intent(intent)
    route = str((intent or {}).get("route") or "") if isinstance(intent, dict) else ""
    hierarchical_route = route in HIERARCHICAL_ROUTES
    for label, group in groups.items():
        if label == "Orchestrator":
            group["selection"] = "manager"
            continue
        meaningful_events = [
            event
            for event in (group.get("events") or [])
            if event.get("event") not in {"agent_ready"}
        ]
        was_invoked = bool(meaningful_events or group.get("tool_calls") or group.get("llm_calls"))
        if intent is None:
            group["selection"] = "available"
        elif label in selected_agents:
            if was_invoked:
                group["selection"] = "invoked"
            elif hierarchical_route:
                group["selection"] = "priority" if label in priority_agents else "eligible"
                group["status"] = "not_invoked" if has_crew_completed else (
                    "queued" if label in priority_agents else "eligible"
                )
                group["activity"] = (
                    "Eligible for hierarchical delegation but not invoked in the completed run."
                    if has_crew_completed
                    else (
                        "Prioritized by the classified source scope; waiting for manager delegation."
                        if label in priority_agents
                        else "Available to the hierarchical manager if the evolving task requires this specialty."
                    )
                )
            else:
                group["selection"] = "selected"
                group["status"] = "skipped" if has_crew_completed else "queued"
                group["activity"] = (
                    "Selected by the intent router but not invoked before completion."
                    if has_crew_completed
                    else "Selected by the intent router; waiting for delegation."
                )
        elif not was_invoked:
            group["selection"] = "excluded"
            group["status"] = "not_selected"
            group["activity"] = "Excluded from this run by the classified intent and source scope."

    observer_events = events
    if has_crew_completed and not any(event.get("event") == "crew_completed" for event in events):
        observer_events = [*events, {"event": "crew_completed", "status": "ok"}]
    observer_report = _latest_observer_report(observer_events)
    for label, update in _observer_agent_updates_from_report(observer_report).items():
        if label in groups:
            observer_state = update["state"]
            delegation = latest_delegation_by_receiver.get(label)
            if delegation:
                delegation_status = str(delegation.get("status") or "running")
                if delegation_status == "completed":
                    observer_state = "completed"
                elif delegation_status in {"error", "warning"}:
                    observer_state = "blocked"
                else:
                    observer_state = "active"
            groups[label]["observer_summary"] = update["summary"]
            groups[label]["observer_state"] = observer_state
            if label != "Orchestrator" and observer_state in {"active", "completed", "blocked"}:
                groups[label]["selection"] = "invoked"
            if observer_state == "active" and groups[label]["status"] not in {"error", "warning"}:
                groups[label]["status"] = "running"
            elif observer_state == "completed" and groups[label]["status"] not in {"error", "warning"}:
                groups[label]["status"] = "ok"
            elif observer_state == "blocked":
                groups[label]["status"] = "error"
    source_flow = _dynamic_source_flow(events)
    topology = _execution_topology(groups, selected_agents, completed=has_crew_completed)
    return {
        "run_id": run_id,
        "run_dir": None,
        "groups": [groups[label] for label in AGENT_LANES if label in groups],
        "total_tool_calls": len(calls_by_id),
        "source_flow": source_flow,
        "topology": topology,
        "selected_agents": sorted(selected_agents),
        "phases": _trace_phases_from_events(events, calls_by_id),
        "event_count": len(latest_events),
        "latest_events": latest_events[-12:],
        "artifacts": {},
        "intent": intent,
        "observer_report": observer_report,
        "agent_dialogue": agent_dialogue,
    }


def initial_live_trace_events(prompt: str, settings: ChatRuntimeSettings) -> list[dict[str, Any]]:
    temp_status = "enabled" if settings.allow_temp_enrollment else "disabled"
    observer_model = resolve_study_assistant_observer_model(
        observer_model=settings.observer_model,
        manager_model=settings.manager_model,
        specialist_model=settings.specialist_model,
    )
    return [
        {
            "event": "ui_run_started",
            "event_id": 0,
            "elapsed_ms": 0,
            "agent_label": "Orchestrator",
            "phase": "kickoff",
            "status": "running",
            "activity": "Request accepted. Preparing hierarchical crew.",
            "query": prompt[:1200],
        },
        {
            "event": "agent_ready",
            "event_id": 0,
            "elapsed_ms": 0,
            "agent_label": "Runtime Observer",
            "phase": "ready",
            "status": "idle",
            "activity": (
                "Ready to interpret A2A delegation and tool lifecycle evidence."
                if settings.observer_enabled
                else "LLM progress summaries disabled; deterministic lifecycle status remains active."
            ),
            "source_system": "Trace Observer",
            "model": observer_model,
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
            "agent_label": "Grade Optimization Specialist",
            "phase": "ready",
            "status": "idle",
            "activity": "Ready for deterministic grade scenario, sensitivity, and target-grade simulations.",
            "source_system": "Grade Optimization",
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
            "agent_label": "Degree Regulations Specialist",
            "phase": "ready",
            "status": "idle",
            "activity": "Ready for AllgStuPO, StuPO, and Regelstudienplan PDF lookups.",
            "source_system": "Degree Regulations",
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
        {
            "event": "agent_ready",
            "event_id": 0,
            "elapsed_ms": 0,
            "agent_label": "Course Commitment Specialist",
            "phase": "ready",
            "status": "idle",
            "activity": "Ready to create explicit confirmation proposals.",
            "source_system": "Course Commitment",
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
    # Find the last event that was actually marked running
    last_running = next(
        (event for event in reversed(events)
         if event.get("event") != "heartbeat" and event.get("status") == "running"),
        {}
    )
    label = _event_agent_label(last_running) if last_running else "Orchestrator"
    if label not in AGENT_LANES:
        label = "Orchestrator"

    last_overall = next((event for event in reversed(events) if event.get("event") != "heartbeat"), {})
    activity = _heartbeat_activity(last_overall)

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
            candidate = int(event.get("elapsed_ms") or 0)
        except (TypeError, ValueError):
            continue
        if candidate > 0:
            elapsed = candidate
            break
    return elapsed + 1200


def _source_for_agent_label(label: str) -> str:
    if label == "Study Advisor":
        return "Grade Manager"
    if label == "Grade Optimization Specialist":
        return "Grade Optimization"
    if label == "MOSES Module Researcher":
        return "MOSES"
    if label == "Degree Regulations Specialist":
        return "Degree Regulations"
    if label == "ISIS Course Info Specialist":
        return "ISIS"
    if label == "Course Commitment Specialist":
        return "Course Commitment"
    return "CrewAI"


def _usage_total_tokens(usage: Any) -> int:
    data = usage if isinstance(usage, dict) else {}
    for key in ("total_tokens", "total", "tokens"):
        try:
            value = int(data.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            return value
    try:
        return int(data.get("prompt_tokens") or 0) + int(data.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _selected_agents_for_intent(intent: Any) -> set[str]:
    if not isinstance(intent, dict):
        return set()
    route = str(intent.get("route") or "")
    if route in HIERARCHICAL_ROUTES:
        return set(HIERARCHICAL_SPECIALISTS)
    selected: set[str] = set()
    route_agent = ROUTE_AGENT_MAP.get(route)
    if route_agent:
        selected.add(route_agent)
    for source in intent.get("required_sources") or []:
        agent = SOURCE_AGENT_MAP.get(str(source).casefold())
        if agent:
            selected.add(agent)
    if bool(intent.get("write_intent")):
        selected.add("Course Commitment Specialist")
    return selected


def _priority_agents_for_intent(intent: Any) -> set[str]:
    """Return source-prioritized agents without narrowing a hierarchical crew's eligibility."""
    if not isinstance(intent, dict):
        return set()
    priority = {
        agent
        for source in (intent.get("required_sources") or [])
        if (agent := SOURCE_AGENT_MAP.get(str(source).casefold()))
    }
    if bool(intent.get("write_intent")):
        priority.add("Course Commitment Specialist")
    return priority


def _execution_topology(
    groups: dict[str, dict[str, Any]],
    selected_agents: set[str],
    *,
    completed: bool,
) -> list[dict[str, Any]]:
    topology = [
        {
            "agent": "Orchestrator",
            "active": str(groups.get("Orchestrator", {}).get("status") or "") == "running",
            "status": groups.get("Orchestrator", {}).get("status") or "idle",
        }
    ]
    if not selected_agents:
        topology.append({"agent": "Intent Router", "active": not completed, "status": "running" if not completed else "done"})
    else:
        for label in AGENT_LANES:
            if label == "Orchestrator" or label not in selected_agents:
                continue
            status = str(groups.get(label, {}).get("status") or "queued")
            topology.append({"agent": label, "active": status == "running", "status": status})
    topology.append({"agent": "Final Answer", "active": completed, "status": "done" if completed else "queued"})
    return topology


def _default_activity_for_agent(label: str) -> str:
    return {
        "Orchestrator": "Preparing delegation.",
        "Study Advisor": "Waiting for Grade Manager work.",
        "Grade Optimization Specialist": "Waiting for grade optimization simulation work.",
        "MOSES Module Researcher": "Waiting for MOSES work.",
        "Degree Regulations Specialist": "Waiting for regulation PDF work.",
        "ISIS Course Info Specialist": "Waiting for ISIS work.",
        "Course Commitment Specialist": "Waiting for confirmation proposal or execution work.",
    }.get(label, "Waiting for activity.")


def _event_agent_label(event: dict[str, Any]) -> str:
    label = str(event.get("agent_label") or "")
    if label == "Unknown Agent":
        label = ""
    if label in AGENT_LANES:
        return label
    role = str(event.get("agent_role") or "").casefold()
    if "study advisor" in role or "personal study advisor" in role:
        return "Study Advisor"
    if "grade optimization" in role or "grade optimizer" in role:
        return "Grade Optimization Specialist"
    if "moses" in role or "module researcher" in role:
        return "MOSES Module Researcher"
    if "degree regulations" in role or "regulations specialist" in role or "stupo" in role:
        return "Degree Regulations Specialist"
    if "isis" in role or "course information specialist" in role:
        return "ISIS Course Info Specialist"
    if "commitment" in role:
        return "Course Commitment Specialist"
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
    has_routing = has_llm or _has_event(events, "intent_classification_started") or _has_event(events, "intent_classified")
    has_route = _has_event(events, "intent_classified") or _has_event(events, "route_execution_started")
    has_delegation = any("coworker" in str(call.get("tool_name") or "").casefold() for call in calls_by_id.values())
    has_tools = bool(calls_by_id)
    has_completed = _has_event(events, "crew_completed") or _has_event(events, "flow_turn_completed")
    has_failed = _has_event(events, "crew_failed") or _has_event(events, "ui_error")
    return [
        {"label": "1", "title": "Flow ingest", "status": _phase_status(has_started, has_routing or has_tools or has_completed, has_failed)},
        {"label": "2", "title": "Intent routing", "status": _phase_status(has_routing, has_route or has_delegation or has_tools or has_completed, has_failed)},
        {"label": "3", "title": "A2A delegation", "status": _phase_status(has_delegation, has_tools or has_completed, has_failed)},
        {"label": "4", "title": "Agent & tool execution", "status": _phase_status(has_tools, has_completed, has_failed)},
        {"label": "5", "title": "Answer synthesis", "status": "error" if has_failed else ("done" if has_completed else "idle")},
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
        tool_name = str(last_event.get("tool_name") or "tool")
        agent = _event_agent_label(last_event)
        return f"{agent} is still executing {tool_name}; no completion event has arrived yet."
    if event_name == "llm_started":
        agent = _event_agent_label(last_event)
        call_id = last_event.get("call_id")
        reference = f" {call_id}" if call_id else ""
        return f"{agent} LLM call{reference} has not emitted a completion or tool event yet."
    if event_name == "intent_classification_started":
        return "Intent classifier is still computing the route and required source scope."
    if event_name == "route_execution_started":
        return "The selected route is initializing its specialist crew."
    if event_name == "flow_turn_persisting":
        return "Final answer and conversation state are being persisted."
    if event_name in {"ui_run_started", "crew_started"}:
        return "No A2A delegation has been emitted yet; the orchestrator remains the active trace owner."
    return f"Awaiting the next observable trace event after {event_name or 'runtime initialization'}."


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
            if call.get("tool_name") in {"Add Module To Study Plan", "Update Module In Study Plan", "Remove Module From Study Plan"} and "write refused" in str(call.get("output_preview") or "").casefold():
                tool_input = call.get("tool_input") or {}
                module_query = str(tool_input.get("module_query") or "").strip()
                term = str(tool_input.get("target_term") or tool_input.get("term") or tool_input.get("current_term") or "").strip()
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
            semester_reference_context(),
            "Source boundaries: Grade Manager is the source of truth for the student's actual study plan, module status, terms, areas, and active/current course list. MOSES is the source of truth for catalog and module details. ISIS is the source of truth for live course activity, deadlines, materials, and announcements.",
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


def get_profile_messages(profile_slug: str, thread_id: str = "default") -> list[dict[str, Any]]:
    thread = load_chat_thread(profile_slug, thread_id=thread_id)
    messages = [message.model_dump(mode="json") for message in thread.messages]
    store = _chat_store()
    store[profile_slug] = messages
    st.session_state[CHAT_HISTORY_KEY] = store
    return messages


def _append_message(profile_slug: str, message: dict[str, Any], thread_id: str = "default") -> None:
    thread = load_chat_thread(profile_slug, thread_id=thread_id)
    thread.messages.append(ChatMessage.model_validate(message))
    save_chat_thread(thread)
    _set_profile_messages(profile_slug, [item.model_dump(mode="json") for item in thread.messages])


def _update_or_append_assistant_message(
    profile_slug: str,
    assistant_message: dict[str, Any],
    proposals: list[CourseProposal] | None = None,
    thread_id: str = "default",
) -> None:
    thread = load_chat_thread(profile_slug, thread_id=thread_id)
    updated = False
    for msg in reversed(thread.messages):
        if msg.role == "assistant":
            if msg.content.strip() == assistant_message["content"].strip():
                msg.trace_dir = assistant_message.get("trace_dir")
                if not msg.metadata:
                    msg.metadata = {}
                if assistant_message.get("metadata"):
                    msg.metadata.update(assistant_message["metadata"])
                updated = True
                break
    if not updated:
        thread.messages.append(ChatMessage.model_validate(assistant_message))
    if proposals is not None:
        thread.active_proposals = proposals
    save_chat_thread(thread)
    _set_profile_messages(profile_slug, [item.model_dump(mode="json") for item in thread.messages])


def _set_profile_messages(profile_slug: str, messages: list[dict[str, Any]]) -> None:
    store = _chat_store()
    store[profile_slug] = messages
    st.session_state[CHAT_HISTORY_KEY] = store


def _clear_active_course_proposals(profile_slug: str, thread_id: str = "default") -> None:
    thread = load_chat_thread(profile_slug, thread_id=thread_id)
    if not thread.active_proposals:
        return
    thread.active_proposals = []
    save_chat_thread(thread)


def _clear_course_card_state(profile_slug: str, proposals: list[Any]) -> None:
    typed_proposals = [_proposal_model(item) for item in proposals]
    cards = _course_cards_from_proposals([item for item in typed_proposals if item is not None])
    keys: set[str] = set()
    for card in cards:
        keys.add(_course_decision_key(profile_slug, card))
        for action in card["actions"]:
            keys.add(_course_action_toggle_key(profile_slug, card, action.action_id))
    for key in keys:
        st.session_state.pop(key, None)


def _clear_all_course_card_state(profile_slug: str) -> None:
    prefix = f"{PROPOSAL_DECISION_PREFIX}_{profile_slug}_"
    for key in list(st.session_state.keys()):
        if str(key).startswith(prefix):
            st.session_state.pop(key, None)


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
        {"source": "Grade Optimization", "agent": "Grade Optimization Specialist", "target": "Orchestrator", "active": "Grade Optimization" in active_sources},
        {"source": "MOSES", "agent": "MOSES Module Researcher", "target": "Orchestrator", "active": "MOSES" in active_sources},
        {"source": "Degree Regulations", "agent": "Degree Regulations Specialist", "target": "Orchestrator", "active": "Degree Regulations" in active_sources},
        {"source": "ISIS", "agent": "ISIS Course Info Specialist", "target": "Orchestrator", "active": "ISIS" in active_sources},
        {"source": "Course Commitment", "agent": "Course Commitment Specialist", "target": "Student", "active": "Course Commitment" in active_sources},
        {"source": "Orchestrator", "agent": "Final Answer", "target": "Student", "active": active},
    ]


def _dynamic_source_flow(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequence = []
    last_agent = None
    
    # 1. Determine currently active agent
    current_active_agent = None
    for event in reversed(events):
        event_name = event.get("event")
        status = event.get("status")
        agent_lbl = _event_agent_label(event)
        if agent_lbl and agent_lbl not in {"Crew", "System", "User", "Unknown Agent"}:
            if status == "running" or event_name in {"llm_started", "tool_start", "task_started"}:
                current_active_agent = agent_lbl
                break
                
    # 2. Reconstruct chronological sequence of agent invocations
    for event in events:
        agent = _event_agent_label(event)
        if not agent or agent in {"Crew", "System", "User", "Unknown Agent"}:
            continue
        event_name = event.get("event")
        if event_name in {"llm_started", "tool_start", "task_started"}:
            if agent != last_agent:
                sequence.append({
                    "agent": agent,
                    "active": False
                })
                last_agent = agent
                
    # If empty, but kickoff started
    if not sequence:
        for event in events:
            if event.get("event") == "crew_started":
                sequence.append({"agent": "Orchestrator", "active": True})
                break
                
    # Mark the latest occurrence of active agent as active
    if current_active_agent:
        found_active = False
        for item in reversed(sequence):
            if item["agent"] == current_active_agent:
                item["active"] = True
                found_active = True
                break
        if not found_active:
            sequence.append({"agent": current_active_agent, "active": True})
            
    return sequence


def _combine_status(left: str, right: str) -> str:
    order = {"idle": 0, "ok": 1, "running": 2, "warning": 3, "error": 4}
    return right if order.get(right, 0) > order.get(left, 0) else left


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
        [data-testid="stMainBlockContainer"] {
            padding-top: 1.35rem !important;
            padding-bottom: 6.5rem !important;
        }
        .chat-hero {
            display: block;
            border: 0;
            border-bottom: 1px solid var(--border);
            border-radius: 0;
            padding: 0.55rem 0 1rem;
            margin: 0 0 0.8rem;
            background: transparent;
            box-shadow: none;
        }
        .chat-hero-main {
            min-width: 0;
        }
        .chat-hero-kicker {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            color: var(--primary);
            font-size: 0.64rem;
            font-weight: 780;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.26rem;
        }
        .chat-hero-kicker > span {
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 999px;
            background: var(--primary);
            box-shadow: 0 0 0 3px rgba(197, 14, 31, 0.10);
        }
        .chat-hero h1 {
            font-size: clamp(1.55rem, 2.5vw, 2rem);
            line-height: 1.12;
            margin: 0 0 0.38rem;
            letter-spacing: -0.025em;
            color: var(--ink);
        }
        .chat-hero p {
            margin: 0;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.45;
            max-width: 76ch;
        }
        .chat-toolbar-spacer {
            height: 1.35rem;
        }
        .st-key-chat_prompt_starters {
            max-width: 900px;
            margin: 1.6rem auto 0;
        }
        .prompt-starters-heading {
            text-align: center;
            margin-bottom: 0.75rem;
        }
        .prompt-starters-heading h2 {
            color: var(--ink);
            font-size: 1rem;
            line-height: 1.3;
            margin: 0;
        }
        .prompt-starters-heading p {
            color: var(--muted);
            font-size: 0.76rem;
            margin: 0.18rem 0 0;
        }
        .st-key-chat_prompt_starters [class*="st-key-example_btn_"] button {
            justify-content: space-between;
            min-height: 2.4rem;
            border: 0;
            border-bottom: 1px solid var(--border);
            border-radius: 0;
            padding-left: 0.2rem;
            padding-right: 0.2rem;
            color: var(--ink);
            font-size: 0.78rem;
            font-weight: 500;
        }
        .st-key-chat_prompt_starters [class*="st-key-example_btn_"] button:hover {
            border-bottom-color: var(--primary);
            background: transparent;
            color: var(--primary);
        }
        .run-error-card {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            gap: 0.65rem;
            align-items: start;
            border: 1px solid rgba(245, 158, 11, 0.45);
            border-left: 3px solid #f59e0b;
            border-radius: 8px;
            background: rgba(245, 158, 11, 0.08);
            padding: 0.7rem 0.8rem;
        }
        .run-error-mark {
            width: 1.3rem;
            height: 1.3rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: #f59e0b;
            color: #fff;
            font-weight: 800;
            font-size: 0.72rem;
        }
        .run-error-card strong {
            color: var(--ink);
            font-size: 0.82rem;
        }
        .run-error-card p {
            color: var(--muted);
            font-size: 0.73rem;
            line-height: 1.4;
            margin: 0.15rem 0 0 0;
        }
        .empty-state-subtitle {
            color: var(--muted);
            font-size: 0.82rem;
            margin: -0.15rem 0 0.75rem 0;
        }
        .answer-runtime-status {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border: 1px solid var(--border);
            border-left: 3px solid var(--primary);
            border-radius: 9px;
            background: var(--surface-2);
            padding: 0.75rem 0.85rem;
            color: var(--ink);
        }
        .answer-runtime-status strong,
        .answer-runtime-status span {
            display: block;
        }
        .answer-runtime-status strong {
            font-size: 0.82rem;
        }
        .answer-runtime-detail {
            color: var(--muted);
            font-size: 0.74rem;
            margin-top: 0.1rem;
        }
        .answer-runtime-observer-label {
            color: var(--success);
            font-size: 0.58rem;
            font-weight: 780;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.08rem;
        }
        .answer-runtime-spinner {
            width: 0.9rem;
            height: 0.9rem;
            border: 2px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 999px;
            animation: answer-runtime-spin 0.8s linear infinite;
            flex: 0 0 auto;
        }
        @keyframes answer-runtime-spin {
            to { transform: rotate(360deg); }
        }
        .chat-empty-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 1.5rem 0;
        }
        .chat-example {
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 8px;
            padding: 0.85rem;
            color: var(--ink);
            min-height: 4.25rem;
        }
        .course-choice-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface-2);
            padding: 0.75rem 0.9rem;
            margin: 1rem 0 0.7rem 0;
        }
        .course-choice-kicker {
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 750;
        }
        .course-choice-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 760;
            line-height: 1.2;
            margin-top: 0.08rem;
        }
        .course-choice-header span {
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 999px;
            color: var(--muted);
            font-weight: 750;
            font-size: 0.75rem;
            padding: 0.22rem 0.55rem;
            white-space: nowrap;
        }
        .course-choice-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface);
            box-shadow: 0 2px 5px rgba(15, 23, 42, 0.04);
            padding: 0.8rem;
            min-height: 13rem;
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            margin-bottom: 0.45rem;
        }
        .course-choice-card.accept {
            border-color: #86efac;
            box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.18);
        }
        .course-choice-card.reject {
            border-color: #fecaca;
            box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.15);
        }
        .course-choice-card.unsure {
            border-color: #fde68a;
        }
        .course-choice-card-top span {
            display: block;
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.2rem;
        }
        .course-choice-card-top strong {
            display: block;
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 750;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .course-choice-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }
        .course-choice-chip {
            border: 1px solid var(--border);
            background: var(--surface-2);
            color: var(--muted);
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 0.18rem 0.48rem;
        }
        .course-choice-chip.grade-manager {
            border-color: #bfdbfe;
            background: #eff6ff;
            color: #1d4ed8;
        }
        .stApp[data-theme="dark"] .course-choice-chip.grade-manager {
            border-color: rgba(59, 130, 246, 0.4);
            background: rgba(59, 130, 246, 0.1);
            color: #60a5fa;
        }
        .course-choice-chip.isis {
            border-color: #bbf7d0;
            background: #f0fdf4;
            color: #047857;
        }
        .stApp[data-theme="dark"] .course-choice-chip.isis {
            border-color: rgba(34, 197, 94, 0.4);
            background: rgba(34, 197, 94, 0.1);
            color: #4ade80;
        }
        .course-choice-meta {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.4rem;
        }
        .course-choice-meta-item {
            border: 1px solid var(--border);
            background: var(--surface-2);
            border-radius: 6px;
            padding: 0.42rem 0.5rem;
        }
        .course-choice-meta-item span {
            display: block;
            color: var(--muted);
            font-size: 0.64rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.12rem;
        }
        .course-choice-meta-item strong {
            display: block;
            color: var(--ink);
            font-size: 0.78rem;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .course-choice-evidence {
            color: var(--muted);
            font-size: 0.74rem;
            line-height: 1.35;
            border-top: 1px solid var(--border);
            padding-top: 0.48rem;
        }
        @media (max-width: 760px) {
            .course-choice-header {
                align-items: flex-start;
            }
        }
        /* Workbench Container */
        .workbench-container {
            border: 1px solid var(--border);
            border-radius: 12px;
            background-color: var(--surface);
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
            background-color: var(--surface-2);
            padding: 0.85rem 1.25rem;
            border-bottom: 1px solid var(--border);
        }
        .workbench-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--ink);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .workbench-summary {
            font-size: 0.8rem;
            color: var(--muted);
            font-weight: 500;
        }
        .observer-overview {
            border-bottom: 1px solid var(--border);
            background: color-mix(in srgb, var(--surface) 94%, var(--success) 6%);
            padding: 0.85rem 1.25rem;
        }
        .observer-overview span,
        .observer-overview strong,
        .observer-overview p {
            display: block;
        }
        .observer-overview span,
        .agent-card-observer-label {
            color: var(--success);
            font-size: 0.58rem;
            font-weight: 760;
            letter-spacing: 0.075em;
            text-transform: uppercase;
        }
        .observer-overview strong {
            color: var(--ink);
            font-size: 0.86rem;
            margin-top: 0.1rem;
        }
        .observer-overview p {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.45;
            margin: 0.18rem 0 0;
            max-width: 110ch;
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
            border-bottom: 1px solid var(--border);
        }
        .phases-timeline-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            color: var(--muted);
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
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            background-color: var(--surface-2);
            transition: all 0.2s ease-in-out;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .phase-node.idle {
            background-color: var(--surface-2);
            color: var(--muted);
            border-color: var(--border);
        }
        .phase-node.active {
            border-color: var(--accent);
            background-color: rgba(59, 130, 246, 0.15);
            color: var(--ink);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }
        .phase-node.done {
            border-color: var(--success);
            background-color: rgba(22, 163, 74, 0.15);
            color: var(--ink);
        }
        .phase-node.error {
            border-color: var(--danger);
            background-color: rgba(220, 38, 38, 0.15);
            color: var(--ink);
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
        .stApp[data-theme="dark"] .phase-num {
            background-color: rgba(255, 255, 255, 0.1);
        }
        .phase-node.active .phase-num {
            background-color: var(--accent);
            color: #ffffff;
        }
        .phase-node.done .phase-num {
            background-color: var(--success);
            color: #ffffff;
        }
        .phase-node.error .phase-num {
            background-color: var(--danger);
            color: #ffffff;
        }
        .phase-txt {
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Flow pipeline styles */
        .flow-pipeline-container {
            padding: 0.75rem 1.25rem;
            background-color: var(--surface-2);
            border-bottom: 1px solid var(--border);
        }
        .flow-pipeline-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            color: var(--muted);
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
            border: 1px solid var(--border);
            background-color: var(--surface);
            border-radius: 6px;
            padding: 0.35rem 0.6rem;
            font-size: 0.72rem;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-weight: 500;
            color: var(--muted);
        }
        .flow-node-wrap {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            min-width: 0;
        }
        .flow-card.active {
            border-color: var(--success);
            background-color: rgba(22, 163, 74, 0.15);
            color: var(--ink);
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(16, 185, 129, 0.05);
        }
        .flow-src {
            font-weight: 700;
            color: var(--ink);
        }
        .flow-card.active .flow-src {
            color: var(--success);
        }
        .stApp[data-theme="dark"] .flow-card.active .flow-src {
            color: #34d399;
        }
        .flow-connector {
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 750;
            margin: 0 0.1rem;
        }
        .flow-agt {
            color: var(--muted);
        }
        .flow-step {
            color: var(--muted);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.58rem;
            opacity: 0.72;
        }
        .flow-invocation {
            border-radius: 999px;
            background: color-mix(in srgb, var(--accent) 10%, transparent);
            color: var(--accent);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.56rem;
            padding: 0.08rem 0.28rem;
        }
        .flow-card.active .flow-agt {
            color: var(--success);
        }
        .flow-state {
            border-left: 1px solid var(--border);
            color: var(--muted);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.61rem;
            padding-left: 0.35rem;
        }
        .flow-card.queued {
            border-style: dashed;
        }
        .flow-card.done {
            border-color: var(--success);
        }
        .stApp[data-theme="dark"] .flow-card.active .flow-agt {
            color: #34d399;
        }

        /* Agents grid */
        .agents-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1rem;
            padding: 1.25rem;
            background-color: var(--surface);
        }
        .agent-card {
            border: 1px solid var(--border);
            border-radius: 10px;
            background-color: var(--surface);
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            transition: all 0.2s ease-in-out;
            position: relative;
        }
        .agent-card.running {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
        }
        .agent-card.queued {
            border-style: dashed;
            border-color: var(--accent);
        }
        .agent-card.eligible {
            border-color: rgba(99, 102, 241, 0.34);
            background: color-mix(in srgb, var(--surface) 94%, #6366f1 6%);
        }
        .agent-card.not_invoked {
            border-style: dashed;
            opacity: 0.82;
            background: var(--surface-2);
        }
        .agent-card.not_selected {
            border-style: dashed;
            opacity: 0.68;
            background: var(--surface-2);
        }
        .agent-card.skipped {
            border-style: dashed;
            opacity: 0.82;
        }
        .agent-card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .agent-card-name {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--ink);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .agent-card-source {
            font-size: 0.68rem;
            color: var(--muted);
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
            background-color: var(--surface-2);
            color: var(--muted);
            border: 1px solid var(--border);
        }
        .status-badge.queued {
            background-color: rgba(59, 130, 246, 0.10);
            color: #2563eb;
            border: 1px dashed #93c5fd;
        }
        .status-badge.eligible {
            background-color: rgba(99, 102, 241, 0.10);
            color: #4f46e5;
            border: 1px solid rgba(99, 102, 241, 0.35);
        }
        .status-badge.not_selected,
        .status-badge.skipped,
        .status-badge.not_invoked {
            background-color: var(--surface-2);
            color: var(--muted);
            border: 1px dashed var(--border);
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
        .stApp[data-theme="dark"] .status-badge.queued,
        .stApp[data-theme="dark"] .status-badge.running,
        .stApp[data-theme="dark"] .status-badge.eligible {
            background-color: rgba(59, 130, 246, 0.14);
            color: #60a5fa;
            border-color: rgba(96, 165, 250, 0.55);
        }
        .stApp[data-theme="dark"] .status-badge.success,
        .stApp[data-theme="dark"] .status-badge.ok,
        .stApp[data-theme="dark"] .status-badge.done {
            background-color: rgba(16, 185, 129, 0.14);
            color: #34d399;
            border-color: rgba(52, 211, 153, 0.5);
        }
        .stApp[data-theme="dark"] .status-badge.error,
        .stApp[data-theme="dark"] .status-badge.failed {
            background-color: rgba(239, 68, 68, 0.14);
            color: #f87171;
            border-color: rgba(248, 113, 113, 0.5);
        }
        .agent-card-activity {
            font-size: 0.74rem;
            color: var(--ink);
            line-height: 1.4;
            min-height: 3.1rem;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .agent-card-observer-label {
            margin-bottom: -0.45rem;
        }
        .agent-card-model {
            display: block;
            width: fit-content;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: var(--ink);
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.12rem 0.35rem;
            font-size: 0.62rem;
        }
        .agent-card-model.pending {
            color: var(--muted);
            font-style: italic;
        }
        .agent-card-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            font-size: 0.68rem;
            color: var(--muted);
            font-weight: 500;
            border-top: 1px solid var(--border);
            padding-top: 0.5rem;
        }
        .agent-card-stats span {
            background-color: var(--surface-2);
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            border: 1px solid var(--border);
        }
        .agent-card-tools {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }
        .no-tools {
            font-size: 0.68rem;
            color: var(--muted);
            font-style: italic;
        }

        /* Tool items inside agent cards */
        .tool-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.68rem;
            background-color: var(--surface-2);
            border: 1px solid var(--border);
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
            color: var(--ink);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 140px;
        }
        .tool-dur {
            font-family: monospace;
            color: var(--muted);
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
        .agent-card.runtime-observer {
            border-left: 3px solid #64748b;
        }
        .agent-card.study-advisor {
            border-left: 3px solid #10b981;
        }
        .agent-card.grade-optimization-specialist {
            border-left: 3px solid #0ea5e9;
        }
        .agent-card.moses-module-researcher {
            border-left: 3px solid #14b8a6;
        }
        .agent-card.isis-course-info-specialist {
            border-left: 3px solid #f59e0b;
        }
        .agent-card.degree-regulations-specialist {
            border-left: 3px solid #8b5cf6;
        }
        .agent-card.course-commitment-specialist {
            border-left: 3px solid #ec4899;
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

        /* Unified Trace Artifacts Section */
        .artifacts-container {
            padding: 1rem 1.25rem;
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
            background-color: var(--surface-2);
        }
        .section-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.075em;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.65rem;
        }
        .artifacts-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .artifact-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.75rem;
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
        }
        .artifact-label {
            font-weight: 600;
            color: var(--ink);
        }
        .artifact-path {
            font-family: monospace;
            color: var(--muted);
            font-size: 0.7rem;
            word-break: break-all;
            user-select: all;
        }

        /* Unified Trace Tool Logs Section */
        .tool-logs-container {
            padding: 1.25rem;
            background-color: var(--surface);
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        .tool-log-item {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background-color: var(--surface-2);
            transition: border-color 0.2s ease;
        }
        .tool-log-item[open] {
            border-color: var(--border);
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .tool-log-summary {
            padding: 0.65rem 1rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--ink);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            user-select: none;
            background-color: var(--surface-2);
        }
        .tool-log-summary::-webkit-details-marker {
            display: none;
        }
        .tool-log-summary::marker {
            display: none;
        }
        .tool-log-summary::before {
            content: "▶";
            font-size: 0.65rem;
            color: var(--muted);
            transition: transform 0.2s ease;
            display: inline-block;
        }
        .tool-log-item[open] > .tool-log-summary::before {
            transform: rotate(90deg);
        }
        .tool-log-id {
            color: var(--muted);
            font-family: monospace;
            font-weight: bold;
        }
        .tool-log-agent {
            color: #4f46e5;
        }
        .stApp[data-theme="dark"] .tool-log-agent {
            color: #818cf8;
        }
        .tool-log-arrow {
            color: var(--muted);
        }
        .tool-log-name {
            color: var(--ink);
            font-weight: 700;
        }
        .tool-log-status-badge {
            margin-left: auto;
            font-size: 0.62rem;
            font-weight: 700;
            padding: 0.1rem 0.35rem;
            border-radius: 4px;
            text-transform: uppercase;
        }
        .tool-log-status-badge.ok, .tool-log-status-badge.success {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .tool-log-status-badge.running {
            background-color: #dbeafe;
            color: #1e40af;
            border: 1px solid #bfdbfe;
        }
        .tool-log-status-badge.error, .tool-log-status-badge.failed {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .tool-log-details {
            padding: 0.85rem 1rem;
            border-top: 1px solid var(--border);
            background-color: var(--surface);
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        .tool-log-meta {
            font-size: 0.68rem;
            color: var(--muted);
            font-weight: 500;
        }
        .tool-log-section {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }
        .tool-log-section-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--muted);
            letter-spacing: 0.05em;
        }
        .tool-log-code {
            margin: 0;
            padding: 0.65rem;
            background-color: #0f172a;
            color: #e2e8f0;
            border-radius: 6px;
            font-family: Consolas, Monaco, "Lucida Console", monospace;
            font-size: 0.72rem;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            border: 1px solid #1e293b;
        }

        /* Agent dialogue panel */
        .agent-dialogue-panel {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface);
            overflow: hidden;
            margin: 0.7rem 0 1rem 0;
        }
        .agent-dialogue-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 0.9rem;
            border-bottom: 1px solid var(--border);
            background: var(--surface-2);
        }
        .agent-dialogue-title {
            color: var(--ink);
            font-weight: 760;
            font-size: 0.95rem;
            line-height: 1.2;
        }
        .agent-dialogue-count,
        .agent-dialogue-live {
            border: 1px solid var(--border);
            border-radius: 999px;
            color: var(--muted);
            background: var(--surface);
            font-size: 0.68rem;
            font-weight: 760;
            padding: 0.18rem 0.5rem;
            white-space: nowrap;
        }
        .agent-dialogue-live {
            border-color: #86efac;
            color: #047857;
            margin-left: 0.35rem;
        }
        .agent-dialogue-empty {
            padding: 1rem;
            color: var(--muted);
            font-size: 0.86rem;
        }
        .agent-dialogue-list {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            padding: 0.9rem;
        }
        .agent-dialogue-pair {
            display: flex;
            flex-direction: column;
            gap: 0.48rem;
        }
        .agent-dialogue-message {
            display: grid;
            grid-template-columns: 2rem minmax(0, 1fr);
            gap: 0.55rem;
            align-items: start;
            max-width: 96%;
        }
        .agent-dialogue-message.response {
            margin-left: 1.75rem;
        }
        .agent-dialogue-avatar {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--surface);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            line-height: 1;
            flex: 0 0 auto;
        }
        .agent-dialogue-bubble {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface);
            padding: 0.62rem 0.72rem;
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .agent-dialogue-message.request .agent-dialogue-bubble {
            background: var(--surface-2);
        }
        .agent-dialogue-message.response.completed .agent-dialogue-bubble {
            border-color: rgba(34, 197, 94, 0.4);
            background: rgba(34, 197, 94, 0.08);
        }
        .agent-dialogue-message.response.running .agent-dialogue-bubble {
            border-color: rgba(59, 130, 246, 0.4);
            background: rgba(59, 130, 246, 0.08);
        }
        .agent-dialogue-message.response.warning .agent-dialogue-bubble {
            border-color: rgba(245, 158, 11, 0.4);
            background: rgba(245, 158, 11, 0.08);
        }
        .agent-dialogue-message.response.error .agent-dialogue-bubble {
            border-color: rgba(239, 68, 68, 0.4);
            background: rgba(239, 68, 68, 0.08);
        }
        .agent-dialogue-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.35rem;
            color: var(--muted);
            font-size: 0.72rem;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }
        .agent-dialogue-meta strong {
            color: var(--ink);
            font-size: 0.78rem;
        }
        .agent-dialogue-meta span {
            color: var(--muted);
        }
        .agent-dialogue-body {
            color: var(--ink);
            font-size: 0.82rem;
            line-height: 1.46;
        }
        .agent-dialogue-observer-summary {
            border-left: 2px solid var(--success);
            padding-left: 0.6rem;
            margin: 0.12rem 0 0.65rem;
        }
        .agent-dialogue-observer-summary span {
            color: var(--success);
            font-size: 0.6rem;
            font-weight: 760;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .agent-dialogue-observer-summary p {
            color: var(--ink);
            font-size: 0.8rem;
            line-height: 1.42;
            margin: 0.12rem 0 0;
        }
        .agent-dialogue-body table {
            border-collapse: collapse;
            width: 100%;
            margin: 0.5rem 0;
            font-size: 0.76rem;
        }
        .agent-dialogue-body th, 
        .agent-dialogue-body td {
            border: 1px solid var(--border);
            padding: 0.35rem 0.5rem;
            text-align: left;
        }
        .agent-dialogue-body th {
            background-color: var(--surface-2);
            font-weight: 700;
        }
        @media (max-width: 760px) {
            .agent-dialogue-message {
                max-width: 100%;
            }
            .agent-dialogue-message.response {
                margin-left: 0;
            }
        }

        /* Proposals panel styling */
        .pending-write-card {
            border: 1px solid var(--border);
            background-color: var(--surface-2);
            border-radius: 10px;
            padding: 1rem;
            margin: 1.25rem 0 0.85rem 0;
        }
        .pending-write-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }
        .pending-write-subtitle {
            font-size: 0.8rem;
            color: var(--muted);
        }
        .divider {
            height: 1px;
            background-color: var(--border);
            margin: 0.85rem 0;
        }

        /* Align user messages to the right side */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
            text-align: right;
            justify-content: flex-start;
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div {
            text-align: right;
        }

        /* Decrease heading sizes inside chat messages */
        [data-testid="stChatMessage"] h1 {
            font-size: 1.25rem !important;
            margin-top: 0.85rem !important;
            margin-bottom: 0.45rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stChatMessage"] h2 {
            font-size: 1.1rem !important;
            margin-top: 0.75rem !important;
            margin-bottom: 0.35rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stChatMessage"] h3 {
            font-size: 1.0rem !important;
            margin-top: 0.65rem !important;
            margin-bottom: 0.3rem !important;
            font-weight: 700 !important;
        }

        /* Styling for Markdown Output within Tool logs */
        .tool-log-output-markdown {
            margin: 0;
            padding: 0.75rem;
            background-color: var(--surface-2);
            color: var(--ink);
            border-radius: 6px;
            border: 1px solid var(--border);
            font-size: 0.78rem;
            line-height: 1.45;
            overflow-x: auto;
        }
        .tool-log-output-markdown p {
            margin: 0 0 0.5rem 0;
        }
        .tool-log-output-markdown p:last-child {
            margin-bottom: 0;
        }
        .tool-log-output-markdown h1, 
        .tool-log-output-markdown h2, 
        .tool-log-output-markdown h3 {
            font-size: 0.9rem !important;
            font-weight: 700 !important;
            margin: 0.65rem 0 0.3rem 0 !important;
            color: var(--ink) !important;
        }
        .tool-log-output-markdown h1:first-child, 
        .tool-log-output-markdown h2:first-child, 
        .tool-log-output-markdown h3:first-child {
            margin-top: 0 !important;
        }
        .tool-log-output-markdown ul, 
        .tool-log-output-markdown ol {
            margin: 0 0 0.5rem 0;
            padding-left: 1.2rem;
        }
        .tool-log-output-markdown li {
            margin-bottom: 0.15rem;
        }
        .tool-log-output-markdown pre {
            background-color: #0f172a;
            color: #e2e8f0;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.72rem;
            overflow-x: auto;
            margin: 0.5rem 0;
        }
        .tool-log-output-markdown code {
            font-family: monospace;
            background-color: var(--surface);
            padding: 0.1rem 0.25rem;
            border-radius: 3px;
            font-size: 0.72rem;
            color: var(--ink);
        }
        .tool-log-output-markdown pre code {
            background-color: transparent;
            padding: 0;
            color: inherit;
            font-size: inherit;
        }
        .tool-log-output-markdown table {
            border-collapse: collapse;
            width: 100%;
            margin: 0.5rem 0;
            font-size: 0.72rem;
        }
        .tool-log-output-markdown th, 
        .tool-log-output-markdown td {
            border: 1px solid var(--border);
            padding: 0.25rem 0.4rem;
            text-align: left;
        }
        .tool-log-output-markdown th {
            background-color: var(--surface-2);
            font-weight: 700;
        }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 0.75rem !important;
            }
            .chat-hero {
                padding: 0.35rem 0 0.8rem;
            }
            .workbench-header {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.35rem;
                padding: 0.75rem;
            }
            .workbench-summary {
                font-size: 0.7rem;
                line-height: 1.35;
            }
            .phases-timeline-container,
            .flow-pipeline-container,
            .agents-grid {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            .console-container {
                margin-left: 0.75rem;
                margin-right: 0.75rem;
            }
            .phase-node,
            .agent-card {
                min-width: 0;
            }
            .agents-grid {
                grid-template-columns: minmax(0, 1fr);
            }
            .flow-connector {
                display: none;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .live-dot,
            .pulse-indicator,
            .console-status,
            .answer-runtime-spinner {
                animation: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
