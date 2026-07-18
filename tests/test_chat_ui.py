from __future__ import annotations

import json
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
    assert workbench["event_count"] == 3
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


def test_saved_agent_interactions_prefer_live_event_journal(tmp_path):
    run_dir = tmp_path / "run-with-live-journal"
    run_dir.mkdir()
    (run_dir / "trace.jsonl").write_text(
        '{"event":"tool_call","tool_name":"Unrelated completed tool"}\n',
        encoding="utf-8",
    )
    lifecycle_events = [
        {
            "event": "tool_start",
            "call_id": 4,
            "tool_name": "delegate_work_to_coworker",
            "tool_input": {
                "coworker": "TU Berlin Degree Regulations Specialist",
                "task": "Inspect the Regelstudienplan.",
                "context": "Find the buffer semester.",
            },
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 4,
                "tool_name": "delegate_work_to_coworker",
                "tool_input": {
                    "coworker": "TU Berlin Degree Regulations Specialist",
                    "task": "Inspect the Regelstudienplan.",
                    "context": "Find the buffer semester.",
                },
                "agent_label": "Orchestrator",
                "status": "ok",
                "output": "Semester 4 contains the 30 LP Masterarbeit.",
            },
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in lifecycle_events) + "\n",
        encoding="utf-8",
    )

    interactions = chat.get_interactions_for_message({"trace_dir": str(run_dir)})

    assert len(interactions) == 1
    assert interactions[0]["receiver"] == "Degree Regulations Specialist"
    assert "30 LP Masterarbeit" in interactions[0]["response"]


def test_current_settings_includes_agent_chat_toggle():
    st.session_state["chat_show_agent_chat_alice"] = True

    settings = chat._current_settings_from_state("alice")

    assert settings.show_agent_chat is True


def test_current_settings_leaves_model_selection_to_env_by_default():
    settings = chat._current_settings_from_state("alice")

    assert settings.specialist_model is None
    assert settings.manager_model is None
    assert settings.observer_model is None
    assert settings.observer_enabled is True


def test_live_status_shows_provider_rate_limit_countdown(monkeypatch):
    monkeypatch.setattr(chat.time, "time", lambda: 1_000.0)

    assert chat._live_run_status(
        [
            {
                "event": "llm_rate_limit_wait",
                "agent_label": "Orchestrator",
                "retry_at_unix": 1_075.0,
            }
        ]
    ) == (
        "Waiting for the API rate limit",
        "Orchestrator will retry in 75 seconds after the provider's requested cooldown.",
    )


def test_intent_scope_keeps_full_roster_and_marks_excluded_agents():
    events = [
        {
            "event": "ui_run_started",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "intent_classified",
            "agent_label": "Orchestrator",
            "status": "ok",
            "intent": {
                "route": "simple_moses",
                "complexity": "simple",
                "required_sources": ["moses"],
                "write_intent": False,
            },
        },
    ]

    workbench = chat.live_workbench_from_events(events)
    groups = {group["agent_label"]: group for group in workbench["groups"]}

    assert set(groups) == set(chat.AGENT_LANES)
    assert groups["MOSES Module Researcher"]["status"] == "queued"
    assert groups["MOSES Module Researcher"]["selection"] == "selected"
    assert groups["Study Advisor"]["status"] == "not_selected"
    assert groups["Study Advisor"]["selection"] == "excluded"
    assert [node["agent"] for node in workbench["topology"]] == [
        "Orchestrator",
        "MOSES Module Researcher",
        "Final Answer",
    ]


def test_deep_dive_keeps_every_hierarchical_specialist_eligible():
    events = [
        {
            "event": "intent_classified",
            "agent_label": "Orchestrator",
            "status": "ok",
            "intent": {
                "route": "deep_dive",
                "complexity": "deep",
                "required_sources": ["grade_manager"],
                "write_intent": False,
            },
        }
    ]

    workbench = chat.live_workbench_from_events(events)
    groups = {group["agent_label"]: group for group in workbench["groups"]}

    assert groups["Study Advisor"]["status"] == "queued"
    assert groups["Study Advisor"]["selection"] == "priority"
    assert groups["MOSES Module Researcher"]["status"] == "eligible"
    assert groups["MOSES Module Researcher"]["selection"] == "eligible"
    assert groups["ISIS Course Info Specialist"]["status"] == "eligible"
    assert groups["ISIS Course Info Specialist"]["selection"] == "eligible"
    assert "Runtime Observer" not in groups
    assert set(workbench["selected_agents"]) == chat.HIERARCHICAL_SPECIALISTS


def test_live_status_prefers_grounded_observer_report():
    events = [
        {
            "event": "observer_progress",
            "agent_label": "Runtime Observer",
            "status": "ok",
            "report": {
                "headline": "MOSES lookup in progress",
                "detail": "The module researcher is comparing relevant catalog entries and their course requirements.",
                "active_agent": "MOSES Module Researcher",
                "evidence": ["tool_start: search_modules"],
            },
        }
    ]

    assert chat._live_run_status(events) == (
        "MOSES lookup in progress",
        "The module researcher is comparing relevant catalog entries and their course requirements.",
    )


def test_live_status_never_exposes_raw_planner_prompt_or_call_metadata():
    raw_prompt = "VERY_RAW_PLANNING_PROMPT " * 400
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie mache ich jetzt am besten mit meinem Master weiter und hole die beste Note heraus?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "llm_started",
            "agent_label": "Orchestrator",
            "status": "running",
            "call_id": "74e9af22-c735-452d-8f31-6b8f7a40650d",
            "task_name": raw_prompt,
            "activity": (
                "Orchestrator LLM call #74e9af22-c735-452d-8f31-6b8f7a40650d "
                f"started for task `{raw_prompt}`; 0 tool schemas exposed."
            ),
        },
    ]

    label, detail = chat._live_run_status(events)
    rendered = f"{label} {detail}"

    assert "Anfrage" in label
    assert "Masterplanung" in detail
    assert "VERY_RAW_PLANNING_PROMPT" not in rendered
    assert "74e9af22" not in rendered
    assert "tool schema" not in rendered


def test_live_workbench_omits_raw_lifecycle_console_and_planner_prompt():
    raw_prompt = "PRIVATE CREWAI PLANNING PROMPT " * 300
    events = [
        {
            "event": "ui_run_started",
            "query": "How should I optimize my Master's plan?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "llm_started",
            "agent_label": "Orchestrator",
            "call_id": "planner-call-id",
            "task_name": raw_prompt,
            "activity": raw_prompt,
            "status": "running",
        },
    ]

    compiled = chat._compile_workbench_html(
        chat.live_workbench_from_events(events),
        live=True,
    )

    assert "Live Log Console" not in compiled
    assert "PRIVATE CREWAI PLANNING PROMPT" not in compiled
    assert "planner-call-id" not in compiled


def test_new_trace_evidence_keeps_rich_observer_detail_while_reconciling_agent_state():
    rich_detail = "The orchestrator is still planning which specialists might be needed."
    events = [
        {
            "event": "ui_run_started",
            "query": "How can I optimize my Master's grade?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "intent_classified",
            "intent": {"route": "deep_dive", "required_sources": ["grade_optimization"]},
            "agent_label": "Orchestrator",
            "status": "ok",
        },
        {
            "event": "observer_progress",
            "agent_label": "Runtime Observer",
            "status": "ok",
            "report": {
                "headline": "Planning specialist sequence",
                "detail": rich_detail,
                "agent_updates": [],
            },
        },
        {
            "event": "tool_start",
            "agent_label": "Orchestrator",
            "tool_name": "Delegate work to coworker",
            "tool_input": {
                "coworker": "TU Berlin Grade Optimization Specialist",
                "task": "Compare thesis outcomes and final-grade scenarios.",
            },
            "status": "running",
        },
    ]

    report = chat._latest_observer_report(events)

    assert report["headline"] == "Planning specialist sequence"
    assert report["detail"] == rich_detail
    assert report["active_agent"] == "Orchestrator"
    updates = {item["agent"]: item for item in report["agent_updates"]}
    assert updates["Grade Optimization Specialist"]["state"] == "active"


def test_completed_delegation_replaces_active_summary_and_historical_running_badge():
    active_summary = (
        "Der Study Advisor wertet die abgerufenen Modul- und Anforderungsdaten aus, "
        "um den aktuellen Leistungspunktestand zu verifizieren."
    )
    delegation = {
        "coworker": "TU Berlin Personal Study Advisor",
        "task": "Prüfe Studienstand, offene Anforderungen und Notenhebel.",
    }
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie optimiere ich meinen Masterabschluss?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "intent_classified",
            "intent": {"route": "deep_dive", "required_sources": ["grade_manager"]},
            "agent_label": "Orchestrator",
            "status": "ok",
        },
        {
            "event": "tool_start",
            "call_id": 5,
            "agent_label": "Orchestrator",
            "tool_name": "ask_question_to_coworker",
            "tool_input": delegation,
            "status": "running",
        },
        {
            "event": "observer_progress",
            "agent_label": "Runtime Observer",
            "status": "ok",
            "report": {
                "headline": "Studienstand wird ausgewertet",
                "detail": active_summary,
                "active_agent": "Study Advisor",
                "agent_updates": [
                    {
                        "agent": "Study Advisor",
                        "state": "active",
                        "summary": active_summary,
                    }
                ],
            },
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 5,
                "agent_label": "Orchestrator",
                "tool_name": "ask_question_to_coworker",
                "tool_input": delegation,
                "status": "ok",
                "output_preview": "57 LP abgeschlossen; zentrale Notenhebel wurden identifiziert.",
            },
        },
    ]

    interactions = chat.extract_agent_interactions(events)
    workbench = chat.live_workbench_from_events(events)
    groups = {group["agent_label"]: group for group in workbench["groups"]}
    dialogue_html = chat._compile_agent_dialogue_html(interactions, live=True)

    assert interactions[0]["status"] == "completed"
    assert interactions[0]["observer_state"] == "completed"
    assert "wertet" not in interactions[0]["observer_summary"]
    assert groups["Study Advisor"]["status"] == "ok"
    assert groups["Study Advisor"]["observer_state"] == "completed"
    assert "wertet" not in groups["Study Advisor"]["observer_summary"]
    assert "Result summary" in dialogue_html
    assert ">working<" not in dialogue_html


def test_internal_task_completion_does_not_precede_visible_a2a_response():
    delegation = {
        "coworker": "TU Berlin Grade Optimization Specialist",
        "task": "Berechne Notenprognose und Sensitivität.",
    }
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie optimiere ich meine Masternote?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "intent_classified",
            "intent": {"route": "deep_dive", "required_sources": ["grade_optimization"]},
            "agent_label": "Orchestrator",
            "status": "ok",
        },
        {
            "event": "tool_start",
            "call_id": 9,
            "agent_label": "Orchestrator",
            "tool_name": "delegate_work_to_coworker",
            "tool_input": delegation,
            "status": "running",
        },
        {
            "event": "task_completed",
            "agent_label": "Grade Optimization Specialist",
            "status": "ok",
            "output_preview": "Interne Simulation abgeschlossen.",
        },
        {
            "event": "observer_progress",
            "agent_label": "Runtime Observer",
            "status": "ok",
            "report": {
                "headline": "Analyse abgeschlossen",
                "detail": "Die gesamte Analyse ist abgeschlossen.",
                "active_agent": "Grade Optimization Specialist",
                "agent_updates": [
                    {
                        "agent": "Grade Optimization Specialist",
                        "state": "completed",
                        "summary": "Hat die Notenprognose abgeschlossen.",
                    }
                ],
            },
        },
    ]

    active_interactions = chat.extract_agent_interactions(events)
    active_workbench = chat.live_workbench_from_events(events)
    active_groups = {group["agent_label"]: group for group in active_workbench["groups"]}

    assert active_interactions[0]["status"] == "running"
    assert active_interactions[0]["response"] == ""
    assert active_interactions[0]["observer_state"] == "active"
    assert active_groups["Grade Optimization Specialist"]["status"] == "running"
    assert active_groups["Grade Optimization Specialist"]["observer_state"] == "active"
    assert "abgeschlossen" not in active_workbench["observer_report"]["detail"].casefold()

    events.append(
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 9,
                "agent_label": "Orchestrator",
                "tool_name": "delegate_work_to_coworker",
                "tool_input": delegation,
                "status": "ok",
                "output_preview": "Forecast 1,4; die Masterarbeit hat den größten Einfluss.",
            },
        }
    )

    completed_interactions = chat.extract_agent_interactions(events)
    completed_workbench = chat.live_workbench_from_events(events)
    completed_groups = {group["agent_label"]: group for group in completed_workbench["groups"]}

    assert completed_interactions[0]["status"] == "completed"
    assert completed_interactions[0]["response"].startswith("Forecast 1,4")
    assert completed_interactions[0]["observer_state"] == "completed"
    assert completed_groups["Grade Optimization Specialist"]["status"] == "ok"
    assert completed_groups["Grade Optimization Specialist"]["observer_state"] == "completed"


def test_observer_agent_updates_enrich_agent_card_and_a2a_bubble():
    summary = "The MOSES researcher is comparing ML1 and Machine Intelligence course details and grading schemes."
    events = [
        {
            "event": "intent_classified",
            "intent": {"route": "deep_dive", "required_sources": ["moses"]},
            "agent_label": "Orchestrator",
            "status": "ok",
        },
        {
            "event": "tool_start",
            "call_id": 8,
            "agent_label": "Orchestrator",
            "tool_name": "Delegate work to coworker",
            "tool_input": {
                "coworker": "TU Berlin MOSES Module Researcher",
                "task": "Compare ML1 and Machine Intelligence.",
            },
            "status": "running",
        },
        {
            "event": "observer_progress",
            "agent_label": "Runtime Observer",
            "status": "ok",
            "report": {
                "headline": "Course comparison underway",
                "detail": "The module and grade specialists are building the requested comparison.",
                "agent_updates": [
                    {
                        "agent": "MOSES Module Researcher",
                        "state": "active",
                        "summary": summary,
                    }
                ],
                "evidence": ["tool_start: raw detail hidden from user summary"],
            },
        },
    ]

    interactions = chat.extract_agent_interactions(events)
    workbench = chat.live_workbench_from_events(events)
    groups = {group["agent_label"]: group for group in workbench["groups"]}
    compiled = chat._compile_workbench_html(workbench, live=True)

    assert interactions[0]["observer_summary"] == summary
    assert groups["MOSES Module Researcher"]["observer_summary"] == summary
    assert workbench["observer_report"]["headline"] == "Course comparison underway"
    assert summary in compiled
    assert "tool_start: raw detail hidden from user summary" not in compiled


def test_completed_trace_reconciles_stale_observer_state_and_only_enriches_latest_a2a_bubble():
    events = [
        {
            "event": "ui_run_started",
            "query": "How does my thesis affect my final grade?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
    ]
    for call_id, task in ((10, "Check the thesis credits."), (11, "Confirm the final-grade formula.")):
        events.extend(
            [
                {
                    "event": "tool_start",
                    "call_id": call_id,
                    "agent_label": "Orchestrator",
                    "tool_name": "Delegate work to coworker",
                    "tool_input": {
                        "coworker": "TU Berlin Degree Regulations Specialist",
                        "task": task,
                    },
                    "status": "running",
                },
                {
                    "event": "tool_finish",
                    "tool_call": {
                        "call_id": call_id,
                        "agent_label": "Orchestrator",
                        "tool_name": "Delegate work to coworker",
                        "tool_input": {
                            "coworker": "TU Berlin Degree Regulations Specialist",
                            "task": task,
                        },
                        "status": "ok",
                        "output_preview": "The requested rule was confirmed.",
                    },
                },
            ]
        )
    events.extend(
        [
            {"event": "crew_completed", "status": "ok"},
            {
                "event": "observer_progress",
                "agent_label": "Runtime Observer",
                "status": "ok",
                "report": {
                    "headline": "Still checking regulations",
                    "detail": "The specialist is currently determining the formal rules.",
                    "active_agent": "Degree Regulations Specialist",
                    "agent_updates": [
                        {
                            "agent": "Degree Regulations Specialist",
                            "state": "active",
                            "summary": "Currently determining the formal thesis rules.",
                        }
                    ],
                    "evidence": [],
                },
            },
        ]
    )

    interactions = chat.extract_agent_interactions(events)
    workbench = chat.live_workbench_from_events(events, completed=True)
    groups = {group["agent_label"]: group for group in workbench["groups"]}

    assert "observer_summary" not in interactions[0]
    assert interactions[1]["observer_state"] == "completed"
    assert "Currently determining" not in interactions[1]["observer_summary"]
    assert groups["Degree Regulations Specialist"]["observer_state"] == "completed"
    assert workbench["observer_report"]["headline"] == "Analysis complete"
    assert "Runtime Observer" not in groups


def test_observer_waits_for_post_intent_activity_then_uses_ten_second_throttle():
    events = [
        {
            "event": "intent_classified",
            "agent_label": "Orchestrator",
            "intent": {"route": "deep_dive"},
            "status": "ok",
        }
    ]

    assert chat.OBSERVER_MIN_INTERVAL_SECONDS == 10.0
    assert chat._observer_trigger_signature(events) is None

    events.append(
        {
            "event": "task_started",
            "agent_label": "Grade Optimization Specialist",
            "task_name": "Compare thesis outcomes",
            "status": "running",
        }
    )

    assert chat._observer_trigger_signature(events) is not None


def test_elapsed_time_skips_events_without_elapsed_timestamp():
    events = [
        {"event": "crew_completed", "elapsed_ms": 398_547},
        {"event": "flow_turn_completed"},
        {"event": "observer_progress", "elapsed_ms": None},
    ]

    assert chat._elapsed_ms_from_events(events) == 399_747


def test_workbench_topology_collapses_repeated_agents_and_omits_observer():
    flows = [
        {"agent": "Orchestrator", "status": "idle"},
        {"agent": "Study Advisor", "status": "idle"},
        {"agent": "Orchestrator", "status": "idle"},
        {"agent": "Runtime Observer", "status": "ok"},
        {"agent": "Study Advisor", "status": "idle"},
        {"agent": "Grade Optimization Specialist", "status": "idle"},
    ]
    groups = {
        "Orchestrator": {"status": "ok"},
        "Study Advisor": {"status": "ok"},
        "Grade Optimization Specialist": {"status": "ok"},
    }

    compact = chat._compact_workbench_topology(flows, groups, live=False)

    assert [item["agent"] for item in compact] == [
        "Orchestrator",
        "Study Advisor",
        "Grade Optimization Specialist",
        "Final Answer",
    ]
    assert all(item["status"] in {"ok", "done"} for item in compact)


def test_workbench_execution_sequence_preserves_real_redelegation_order():
    workbench = {
        "agent_dialogue": [
            {"receiver": "TU Berlin Study Advisor", "status": "completed"},
            {"receiver": "TU Berlin Degree Regulations Specialist", "status": "completed"},
            {"receiver": "TU Berlin Study Advisor", "status": "running"},
        ]
    }
    groups = {"Orchestrator": {"status": "running"}}

    sequence, is_a2a = chat._workbench_execution_sequence(workbench, groups, live=True)

    assert is_a2a is True
    assert [item["agent"] for item in sequence] == [
        "Orchestrator",
        "Study Advisor",
        "Degree Regulations Specialist",
        "Study Advisor",
        "Final Answer",
    ]
    assert sequence[1]["invocation"] == 1
    assert sequence[3]["invocation"] == 2
    assert sequence[3]["status"] == "running"


def test_agent_card_fallback_keeps_raw_tool_identifiers_out_of_summary_copy():
    activity = chat._agent_card_fallback_activity(
        "Grade Optimization Specialist",
        {"status": "ok", "activity": "Finished run_grade_what_if_scenario."},
    )

    assert activity == "Grade scenarios and assessment sensitivity calculated."
    assert "run_grade_what_if_scenario" not in activity


def test_true_stream_preview_resets_planning_text_after_tool_call():
    events = [
        {
            "event": "intent_classified",
            "intent": {"route": "deep_dive", "required_sources": ["moses"]},
        },
        {
            "event": "llm_stream_chunk",
            "agent_label": "Orchestrator",
            "chunk_type": "text",
            "content": "I should ask a specialist before answering.",
        },
        {
            "event": "llm_stream_chunk",
            "agent_label": "Orchestrator",
            "chunk_type": "tool_call",
            "content": "",
        },
        {
            "event": "llm_stream_chunk",
            "agent_label": "MOSES Module Researcher",
            "chunk_type": "text",
            "content": "Internal specialist response that belongs in A2A chat.",
        },
        {
            "event": "llm_stream_chunk",
            "agent_label": "Orchestrator",
            "chunk_type": "text",
            "content": "Here is the grounded final answer from the coordinated agent team.",
        },
    ]

    assert chat._streaming_answer_preview(events) == (
        "Here is the grounded final answer from the coordinated agent team."
    )


def test_answer_animation_reconstructs_final_answer_from_streamed_prefix():
    chunks = list(chat._animated_answer_chunks("A complete answer", "A complete"))

    assert "".join(chunks) == "A complete answer"


def test_message_error_detail_supports_structured_and_legacy_failures():
    structured = {
        "role": "assistant",
        "content": "Run interrupted.",
        "metadata": {"error": {"type": "RuntimeError", "detail": "stream failed"}},
    }
    legacy = {
        "role": "assistant",
        "content": "Could not run the Study Assistant: `legacy stream failed`",
    }

    assert chat._message_error_detail(structured) == "stream failed"
    assert chat._message_error_detail(legacy) == "legacy stream failed"


def test_student_context_uses_source_boundary_wording():
    st.session_state["modules"] = []
    st.session_state["program_view"] = "All"
    st.session_state["relevant_programs"] = []

    context = chat.build_student_context("alice")

    assert "Grade Manager is the source of truth for the student's actual study plan" in context
    assert "MOSES is the source of truth for catalog and module details" in context
    assert "ISIS is the source of truth for live course activity" in context
    assert "source of truth for full module details" not in context


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
    assert groups["Grade Optimization Specialist"]["activity"] == (
        "Ready for deterministic grade scenario, sensitivity, and target-grade simulations."
    )
    assert groups["Degree Regulations Specialist"]["activity"] == "Ready for AllgStuPO, StuPO, and Regelstudienplan PDF lookups."
    assert [phase["status"] for phase in workbench["phases"][:2]] == ["done", "active"]


def test_live_workbench_includes_grade_optimization_specialist():
    events = [
        {
            "event": "tool_start",
            "run_id": "run-grade-optimization",
            "call_id": 8,
            "tool_name": "Run Target Grade Optimizer",
            "tool_input": {"target_grade": 1.7},
            "agent_role": "TU Berlin Grade Optimization Specialist",
            "source_system": "Grade Optimization",
            "status": "running",
            "badges": ["Grade Optimization"],
        }
    ]

    workbench = chat.live_workbench_from_events(events)
    groups = {group["agent_label"]: group for group in workbench["groups"]}

    assert "Grade Optimization Specialist" in groups
    assert groups["Grade Optimization Specialist"]["status"] == "running"
    assert groups["Grade Optimization Specialist"]["source_system"] == "Grade Optimization"
    assert groups["Grade Optimization Specialist"]["tool_calls"][0]["tool_name"] == "Run Target Grade Optimizer"
    assert workbench["source_flow"] == [{"agent": "Grade Optimization Specialist", "active": True}]


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
                                    "verified_isis_course_id": 48474,
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
                verified_isis_course_id=48474,
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
                verified_isis_course_id=48474,
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
                verified_isis_course_id=48474,
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
                verified_isis_course_id=48474,
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
