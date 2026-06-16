from __future__ import annotations

from pathlib import Path

import streamlit as st

from crew.isis_client import MoodleRestClient
from main import MultiAgentStudyAssistantRunResult
from ui import chat


def setup_function():
    st.session_state.clear()


def _setup_chat_profiles(monkeypatch, tmp_path):
    from core import persistence

    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "profiles"
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(persistence, "PROFILES_FILE", profiles_dir / "profiles.json")
    monkeypatch.setattr(persistence, "_LEGACY_MODULES_FILE", data_dir / "modules.json")
    persistence.save_profiles(
        [
            persistence.ProfileRecord(slug="alice", display_name="Alice", is_primary=True),
            persistence.ProfileRecord(slug="bob", display_name="Bob", is_primary=False),
        ]
    )


def test_live_workbench_from_events_tracks_running_and_finished_calls():
    events = [
        {
            "event": "tool_start",
            "run_id": "run-1",
            "call_id": 1,
            "tool_name": "Get Study Plan Snapshot",
            "tool_input": {},
            "agent_label": "Study Advisor",
            "source_system": "Grade Manager",
            "badges": ["Grade Manager"],
        },
        {
            "event": "tool_finish",
            "run_id": "run-1",
            "tool_call": {
                "call_id": 1,
                "tool_name": "Get Study Plan Snapshot",
                "tool_input": {},
                "agent_label": "Study Advisor",
                "source_system": "Grade Manager",
                "status": "ok",
                "badges": ["Grade Manager"],
                "output_preview": "# Study plan snapshot",
            },
        },
        {
            "event": "tool_start",
            "run_id": "run-1",
            "call_id": 2,
            "tool_name": "Get ISIS Course Assignments",
            "tool_input": {"course_id": 47025},
            "agent_label": "ISIS Course Info Specialist",
            "source_system": "ISIS",
            "badges": ["ISIS"],
        },
    ]

    workbench = chat.live_workbench_from_events(events)

    assert workbench["run_id"] == "run-1"
    assert workbench["total_tool_calls"] == 2
    groups = {group["agent_label"]: group for group in workbench["groups"]}
    assert groups["Study Advisor"]["tool_calls"][0]["status"] == "ok"
    assert groups["ISIS Course Info Specialist"]["tool_calls"][0]["status"] == "running"
    assert [item["agent"] for item in workbench["source_flow"]] == ["Study Advisor", "ISIS Course Info Specialist"]
    assert [item["active"] for item in workbench["source_flow"]] == [False, True]


def test_live_workbench_shows_lifecycle_activity_before_tool_calls():
    settings = chat.ChatRuntimeSettings(
        specialist_model=None,
        manager_model=None,
        temperature=0.2,
        top_p=None,
        trace_enabled=True,
        trace_full=False,
        verbose=False,
        cache=True,
        allow_temp_enrollment=True,
    )
    events = chat.initial_live_trace_events("What courses do I do this semester?", settings)
    events.extend(
        [
            {
                "event": "crew_started",
                "elapsed_ms": 120,
                "agent_label": "Orchestrator",
                "status": "running",
                "activity": "Crew kickoff started.",
            },
            {
                "event": "llm_started",
                "elapsed_ms": 450,
                "agent_label": "Orchestrator",
                "status": "running",
                "activity": "Thinking and selecting the next action.",
            },
        ]
    )

    workbench = chat.live_workbench_from_events(events)

    assert workbench["total_tool_calls"] == 0
    groups = {group["agent_label"]: group for group in workbench["groups"]}
    assert groups["Orchestrator"]["status"] == "running"
    assert groups["Orchestrator"]["llm_calls"] == 1
    assert groups["Study Advisor"]["activity"] == "Ready for Grade Manager reads and confirmed study-plan writes."
    assert [phase["status"] for phase in workbench["phases"][:2]] == ["done", "active"]


def test_pending_write_extraction_uses_refused_tool_inputs():
    result = MultiAgentStudyAssistantRunResult(
        answer="Please confirm the write.",
        tool_summary_lines=[],
        trace_dir=None,
    )
    workbench = {
        "groups": [
            {
                "agent_label": "Study Advisor",
                "tool_calls": [
                    {
                        "tool_name": "Add Module To Study Plan",
                        "tool_input": {"module_query": "40967", "term": "WS 26/27", "area": "Elective"},
                        "output_preview": "Study-plan write refused. The confirmation_token must be exactly ...",
                    }
                ],
            }
        ]
    }

    action = chat.extract_pending_study_plan_write(result, workbench)

    assert action == {
        "module_query": "40967",
        "term": "WS 26/27",
        "area": "Elective",
        "program_key": None,
    }


def test_pending_write_extraction_can_use_answer_text():
    action = chat.extract_pending_study_plan_write_from_text(
        "Soll ich das Modul 40967 fuer WS 26/27 in deinen Plan eintragen?"
    )

    assert action is not None
    assert action["module_query"] == "40967"
    assert action["term"] == "WS 26/27"


def test_pending_write_extraction_ignores_semester_years_without_write_proposal():
    action = chat.extract_pending_study_plan_write_from_text(
        "Deine Kurse im Sommersemester 2026: Natural Language Processing in SS 26."
    )

    assert action is None


def test_trace_has_successful_study_plan_write_detects_badge():
    assert chat.trace_has_successful_study_plan_write(
        {
            "groups": [
                {
                    "tool_calls": [
                        {"tool_name": "Add Module To Study Plan", "badges": ["Grade Manager", "study-plan write"]}
                    ]
                }
            ]
        }
    )


def test_profile_chat_history_and_isis_session_are_scoped(monkeypatch, tmp_path):
    _setup_chat_profiles(monkeypatch, tmp_path)
    chat._append_message("alice", {"role": "user", "content": "Alice question"})
    chat._append_message("bob", {"role": "user", "content": "Bob question"})
    client = MoodleRestClient(wstoken="alice-token")
    st.session_state[chat.ISIS_SESSIONS_KEY] = {
        "alice": {"mode": "session", "client": client, "created_at": "2026-06-16T12:00:00"},
        "bob": {"mode": "env", "client": None, "created_at": "2026-06-16T12:00:00"},
    }

    assert chat.get_profile_messages("alice")[0]["content"] == "Alice question"
    assert chat.get_profile_messages("bob")[0]["content"] == "Bob question"
    assert chat.get_profile_isis_client("alice") is client
    assert chat.get_profile_isis_client("bob") is None


def test_append_heartbeat_event_chooses_correct_running_agent():
    events = [
        {"event": "ui_run_started", "agent_label": "Orchestrator", "status": "running"},
        {"event": "agent_ready", "agent_label": "ISIS Course Info Specialist", "status": "idle"},
    ]
    chat._append_heartbeat_event(events)
    assert events[-1]["event"] == "heartbeat"
    assert events[-1]["agent_label"] == "Orchestrator"

    # Add a running event for ISIS
    events.append({"event": "llm_started", "agent_label": "ISIS Course Info Specialist", "status": "running"})
    # Since the last event is not a heartbeat, _append_heartbeat_event will append a new heartbeat
    chat._append_heartbeat_event(events)
    assert events[-1]["event"] == "heartbeat"
    assert events[-1]["agent_label"] == "ISIS Course Info Specialist"


def test_highlight_json_colors_keys_and_values():
    input_str = '{\n  "module_query": "40967 & ML",\n  "allow_temp": true,\n  "count": 42\n}'
    highlighted = chat._highlight_json(input_str)
    assert 'color: #60a5fa' in highlighted  # key blue
    assert 'color: #10b981' in highlighted  # string green
    assert 'color: #f43f5e' in highlighted  # bool/number pink
    assert '&quot;module_query&quot;' in highlighted
    assert '&amp;' in highlighted  # html escape check


def test_safe_int():
    assert chat._safe_int(42) == 42
    assert chat._safe_int("123") == 123
    assert chat._safe_int("llm-1") == 1
    assert chat._safe_int(None) == 0
    assert chat._safe_int("abc") == 0


def test_live_workbench_agent_status_graceful_tool_failures():
    # 1. Test case: tool call fails (status="error") but task completed successfully (event="task_completed")
    events_completed = [
        {"event": "task_started", "agent_label": "MOSES Module Researcher", "status": "running"},
        {
            "event": "tool_start",
            "call_id": 1,
            "tool_name": "Get Moses Module",
            "agent_label": "MOSES Module Researcher",
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 1,
                "tool_name": "Get Moses Module",
                "agent_label": "MOSES Module Researcher",
                "status": "error",
            },
        },
        {"event": "task_completed", "agent_label": "MOSES Module Researcher", "status": "ok"},
        {"event": "crew_completed", "agent_label": "Orchestrator", "status": "ok"},
    ]
    workbench = chat.live_workbench_from_events(events_completed)
    groups = {g["agent_label"]: g for g in workbench["groups"]}
    assert groups["MOSES Module Researcher"]["status"] == "ok"

    # 2. Test case: tool call fails (status="error") and task failed (event="task_failed")
    events_failed = [
        {"event": "task_started", "agent_label": "MOSES Module Researcher", "status": "running"},
        {
            "event": "tool_start",
            "call_id": 2,
            "tool_name": "Get Moses Module",
            "agent_label": "MOSES Module Researcher",
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 2,
                "tool_name": "Get Moses Module",
                "agent_label": "MOSES Module Researcher",
                "status": "error",
            },
        },
        {"event": "task_failed", "agent_label": "MOSES Module Researcher", "status": "error"},
    ]
    workbench = chat.live_workbench_from_events(events_failed)
    groups = {g["agent_label"]: g for g in workbench["groups"]}
    assert groups["MOSES Module Researcher"]["status"] == "error"

    # 3. Test case: tool call fails (status="error") and crew is still running (no task_completed or task_failed yet)
    events_running = [
        {"event": "task_started", "agent_label": "MOSES Module Researcher", "status": "running"},
        {
            "event": "tool_start",
            "call_id": 3,
            "tool_name": "Get Moses Module",
            "agent_label": "MOSES Module Researcher",
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 3,
                "tool_name": "Get Moses Module",
                "agent_label": "MOSES Module Researcher",
                "status": "error",
            },
        },
    ]
    workbench = chat.live_workbench_from_events(events_running)
    groups = {g["agent_label"]: g for g in workbench["groups"]}
    assert groups["MOSES Module Researcher"]["status"] == "running"

    # 4. Test case: tool call fails (status="error") and crew completed successfully but only via completed=True flag (no events)
    workbench = chat.live_workbench_from_events(events_running, completed=True)
    groups = {g["agent_label"]: g for g in workbench["groups"]}
    assert groups["MOSES Module Researcher"]["status"] == "ok"
