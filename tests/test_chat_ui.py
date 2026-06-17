from __future__ import annotations

from pathlib import Path

import streamlit as st

from crew.isis_client import MoodleRestClient
from crew.tools.proposal_tools import ProposalCourseInput, build_course_proposal
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


def test_agent_interaction_extraction_pairs_coworker_tool_events():
    events = [
        {
            "event": "tool_start",
            "run_id": "run-dialogue",
            "call_id": 9,
            "tool_name": "Delegate work to coworker",
            "tool_input": {
                "coworker": "TU Berlin MOSES Module Researcher",
                "task": "Find Machine Learning modules that fit the student's elective area.",
                "context": "The student already completed ML1.",
            },
            "agent_label": "Orchestrator",
            "status": "running",
            "elapsed_ms": 140,
        },
        {
            "event": "tool_finish",
            "run_id": "run-dialogue",
            "tool_call": {
                "call_id": 9,
                "tool_name": "Delegate work to coworker",
                "tool_input": {
                    "coworker": "TU Berlin MOSES Module Researcher",
                    "task": "Find Machine Learning modules that fit the student's elective area.",
                    "context": "The student already completed ML1.",
                },
                "agent_label": "Orchestrator",
                "status": "ok",
                "duration_ms": 832,
                "output_preview": "Found Reinforcement Learning and Machine Learning 2.",
            },
        },
    ]

    interactions = chat.extract_agent_interactions(events)

    assert len(interactions) == 1
    interaction = interactions[0]
    assert interaction["sender"] == "Orchestrator"
    assert interaction["receiver"] == "MOSES Module Researcher"
    assert interaction["status"] == "completed"
    assert interaction["duration_ms"] == 832
    assert "already completed ML1" in interaction["question"]
    assert "Reinforcement Learning" in interaction["response"]

    workbench = chat.live_workbench_from_events(events, completed=True)
    assert workbench["agent_dialogue"][0]["receiver"] == "MOSES Module Researcher"


def test_current_settings_includes_agent_chat_toggle():
    st.session_state["chat_show_agent_chat_alice"] = True

    settings = chat._current_settings_from_state("alice")

    assert settings.show_agent_chat is True


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
    assert groups["Degree Regulations Specialist"]["activity"] == "Ready for AllgStuPO, StuPO, and Regelstudienplan PDF lookups."
    assert [phase["status"] for phase in workbench["phases"][:2]] == ["done", "active"]


def test_live_workbench_includes_degree_regulations_specialist():
    events = [
        {
            "event": "tool_start",
            "run_id": "run-regulations",
            "call_id": 7,
            "tool_name": "Search Degree Regulation PDFs",
            "tool_input": {"query": "AllgStuPO Wiederholungsprüfung"},
            "agent_role": "TU Berlin Degree Regulations Specialist",
            "source_system": "Degree Regulations",
            "status": "running",
            "badges": ["Degree Regulations"],
        }
    ]

    workbench = chat.live_workbench_from_events(events)
    groups = {group["agent_label"]: group for group in workbench["groups"]}

    assert "Degree Regulations Specialist" in groups
    assert groups["Degree Regulations Specialist"]["status"] == "running"
    assert groups["Degree Regulations Specialist"]["source_system"] == "Degree Regulations"
    assert groups["Degree Regulations Specialist"]["tool_calls"][0]["tool_name"] == "Search Degree Regulation PDFs"
    assert workbench["source_flow"] == [{"agent": "Degree Regulations Specialist", "active": True}]


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


def test_resolve_course_proposals_rebuilds_explicit_proposal_tool_call(monkeypatch, tmp_path):
    _setup_chat_profiles(monkeypatch, tmp_path)
    result = MultiAgentStudyAssistantRunResult(
        answer="The proposal has been prepared.",
        tool_summary_lines=[],
        trace_dir=None,
        proposed_actions=[],
    )
    workbench = {
        "groups": [
            {
                "agent_label": "Course Commitment Specialist",
                "tool_calls": [
                    {
                        "tool_name": "propose_course_actions_for_confirmation",
                        "agent_label": "Course Commitment Specialist",
                        "tool_input": {
                            "proposal_title": "Software Security Lab Enrollment Confirmation",
                            "proposal_summary": "Confirm enrollment and study-plan sync.",
                            "courses": [
                                {
                                    "course_title": "Software Security Lab",
                                    "rationale": "Grade Manager and ISIS are out of sync.",
                                    "evidence": [
                                        "Grade Manager status: In Progress",
                                        "ISIS course ID: 48474",
                                    ],
                                    "module_query": "41240",
                                    "version": 3,
                                    "term": "SS 26",
                                    "area": "Elective",
                                    "isis_course_id": 48474,
                                    "isis_course_query": "Software Security Lab",
                                    "isis_course_url": "https://isis.tu-berlin.de/course/view.php?id=48474",
                                    "isis_term_hint": "SS 26",
                                    "include_grade_manager": True,
                                    "include_isis": True,
                                }
                            ],
                        },
                    }
                ],
            }
        ]
    }

    proposals = chat.resolve_course_proposals(result, workbench)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.title == "Software Security Lab Enrollment Confirmation"
    assert proposal.source_agent == "Course Commitment Specialist"
    assert [action.kind for action in proposal.actions] == ["grade_manager_add", "isis_enroll"]
    assert proposal.actions[0].grade_manager_payload["module_query"] == "41240"
    assert proposal.actions[1].isis_payload["course_id"] == 48474


def test_course_card_decisions_default_to_unsure_and_respect_action_toggles(monkeypatch, tmp_path):
    _setup_chat_profiles(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Security option",
        proposal_summary="One suggested course.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Matches the security focus.",
                module_query="41240",
                term="SS 26",
                area="Elective",
                isis_course_id=48474,
                include_grade_manager=True,
                include_isis=True,
            )
        ],
    )
    card = chat._course_cards_from_proposals([proposal])[0]

    assert len(card["actions"]) == 2
    assert chat._course_card_metadata(card) == [
        ("Term", "SS 26"),
        ("Area", "Elective"),
        ("Module", "41240"),
        ("ISIS ID", "48474"),
    ]

    decisions = chat._collect_course_card_decisions("alice", [proposal])

    assert decisions[0]["decision"] == "unsure"
    assert [action["enabled"] for action in decisions[0]["actions"]] == [True, True]
    assert chat._action_decisions_from_course_card_decisions(decisions) == []

    st.session_state[chat._course_decision_key("alice", card)] = "accept"
    st.session_state[chat._course_action_toggle_key("alice", card, card["actions"][1].action_id)] = False
    decisions = chat._collect_course_card_decisions("alice", [proposal])
    action_decisions = chat._action_decisions_from_course_card_decisions(decisions)

    assert decisions[0]["decision"] == "accept"
    assert [action["enabled"] for action in decisions[0]["actions"]] == [True, False]
    assert len(action_decisions) == 2
    assert action_decisions[0].action_id == card["actions"][0].action_id
    assert action_decisions[0].approved is True
    assert action_decisions[1].action_id == card["actions"][1].action_id
    assert action_decisions[1].approved is False
    assert "disabled" in action_decisions[1].feedback


def test_decision_context_is_compact():
    decisions = [
        {
            "course_title": "Software Security Lab",
            "decision": "reject",
            "actions": [
                {"kind": "grade_manager_add", "enabled": True},
                {"kind": "isis_enroll", "enabled": False},
            ],
        }
    ]

    context = chat._format_course_card_decisions_context(decisions)
    assert "Software Security Lab" in context
    assert '"decision": "reject"' in context
    assert '"enabled": false' in context


def test_clear_active_course_proposals_removes_widgets_and_card_state(monkeypatch, tmp_path):
    _setup_chat_profiles(monkeypatch, tmp_path)
    from crew.chat_persistence import append_turn, load_chat_thread

    proposal = build_course_proposal(
        proposal_title="Security option",
        proposal_summary="One suggested course.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Matches the security focus.",
                module_query="41240",
                term="SS 26",
                isis_course_id=48474,
                include_grade_manager=True,
                include_isis=True,
            )
        ],
    )
    append_turn(
        "alice",
        user_content="Suggest a course.",
        assistant_content="Here is one.",
        proposals=[proposal],
        rolling_summary="Course suggestion.",
    )
    card = chat._course_cards_from_proposals([proposal])[0]
    decision_key = chat._course_decision_key("alice", card)
    toggle_key = chat._course_action_toggle_key("alice", card, card["actions"][0].action_id)
    st.session_state[decision_key] = "accept"
    st.session_state[toggle_key] = False

    chat._clear_active_course_proposals("alice")
    chat._clear_course_card_state("alice", [proposal])

    assert load_chat_thread("alice").active_proposals == []
    assert decision_key not in st.session_state
    assert toggle_key not in st.session_state


def test_clear_all_course_card_state_removes_stale_profile_decisions_only(monkeypatch, tmp_path):
    _setup_chat_profiles(monkeypatch, tmp_path)
    stale_decision = f"{chat.PROPOSAL_DECISION_PREFIX}_alice_moses_40017"
    stale_toggle = f"{chat.PROPOSAL_DECISION_PREFIX}_alice_moses_40017_isis-resolve_enabled"
    other_profile_decision = f"{chat.PROPOSAL_DECISION_PREFIX}_bob_moses_40017"
    unrelated_key = "chat_temperature_alice"
    st.session_state[stale_decision] = "accept"
    st.session_state[stale_toggle] = False
    st.session_state[other_profile_decision] = "accept"
    st.session_state[unrelated_key] = 0.2

    chat._clear_all_course_card_state("alice")

    assert stale_decision not in st.session_state
    assert stale_toggle not in st.session_state
    assert st.session_state[other_profile_decision] == "accept"
    assert st.session_state[unrelated_key] == 0.2


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


def test_course_cards_collapse_same_course_across_proposals(monkeypatch, tmp_path):
    _setup_chat_profiles(monkeypatch, tmp_path)
    proposal1 = build_course_proposal(
        proposal_title="Option A",
        proposal_summary="First suggestion.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Matches the security focus.",
                module_query="41240",
                term="SS 26",
                isis_course_id=48474,
                include_grade_manager=True,
                include_isis=True,
            )
        ],
    )
    proposal2 = build_course_proposal(
        proposal_title="Option B",
        proposal_summary="Second suggestion.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Matches the security focus.",
                module_query="41240",
                term="SS 26",
                isis_course_id=48474,
                include_grade_manager=True,
                include_isis=True,
            )
        ],
    )

    cards = chat._course_cards_from_proposals([proposal1, proposal2])

    assert len(cards) == 1
    card = cards[0]
    key = chat._course_decision_key("alice", card)

    assert [action.kind for action in card["actions"]] == ["grade_manager_add", "isis_enroll"]
    assert "moses_41240" in key
