from __future__ import annotations

from crew import runtime_observer


def test_runtime_observer_snapshot_keeps_intent_a2a_and_tool_evidence():
    events = [
        {
            "event": "intent_classified",
            "intent": {"route": "deep_dive", "required_sources": ["moses"]},
            "agent_label": "Orchestrator",
            "status": "ok",
        },
        {
            "event": "heartbeat",
            "agent_label": "Orchestrator",
            "activity": "Thinking and gathering information.",
        },
        {
            "event": "tool_start",
            "call_id": 4,
            "agent_label": "Orchestrator",
            "tool_name": "Delegate work to coworker",
            "tool_input": {
                "coworker": "TU Berlin MOSES Module Researcher",
                "task": "Resolve the module catalog metadata.",
                "api_token": "must-not-leak",
            },
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 5,
                "agent_label": "MOSES Module Researcher",
                "tool_name": "search_modules",
                "tool_input": {"query": "Machine Learning"},
                "status": "ok",
                "output_preview": "Found 4 matching modules.",
            },
        },
        {
            "event": "llm_stream_chunk",
            "content": "private token stream",
        },
    ]

    snapshot = runtime_observer.runtime_observer_snapshot(events)

    assert snapshot["intent"]["route"] == "deep_dive"
    assert [event["event"] for event in snapshot["recent_changes"]] == [
        "intent_classified",
        "tool_start",
        "tool_finish",
    ]
    assert snapshot["recent_changes"][1]["delegated_to"] == "TU Berlin MOSES Module Researcher"
    assert snapshot["recent_changes"][1]["tool_input"]["api_token"] == "[redacted]"
    assert snapshot["recent_changes"][2]["agent"] == "MOSES Module Researcher"
    assert snapshot["recent_changes"][2]["output_preview"] == "Found 4 matching modules."
    assert snapshot["a2a_delegations"][0]["request"] == "Resolve the module catalog metadata."
    assert "CrewAI runtime" not in {item.get("agent") for item in snapshot["agent_states"]}


def test_runtime_observer_snapshot_carries_prior_report_agent_history_and_study_context():
    events = [
        {"event": "ui_run_started", "query": "How does my thesis affect my final grade?"},
        {
            "event": "observer_progress",
            "report": {
                "headline": "Thesis scenarios underway",
                "detail": "The grade specialist is comparing thesis outcomes.",
                "agent_updates": [],
            },
        },
        {
            "event": "tool_start",
            "agent_label": "Grade Optimization Specialist",
            "tool_name": "run_grade_sensitivity_analysis",
            "tool_input": {"best_grade": 1.0, "worst_grade": 2.0},
        },
    ]

    snapshot = runtime_observer.runtime_observer_snapshot(
        events,
        study_context="Master student; thesis is still missing.",
    )

    assert snapshot["student_request"] == "How does my thesis affect my final grade?"
    assert snapshot["study_context"] == "Master student; thesis is still missing."
    assert snapshot["previous_observer_report"]["headline"] == "Thesis scenarios underway"
    optimizer = next(
        item for item in snapshot["agent_states"] if item["agent"] == "Grade Optimization Specialist"
    )
    assert optimizer["state"] == "active"
    assert snapshot["recent_changes"][-1]["tool_input"] == {"best_grade": 1.0, "worst_grade": 2.0}


def test_generate_runtime_observer_report_uses_lightweight_model_and_schema(monkeypatch):
    captured = {}

    class FakeLlm:
        def call(self, messages, response_model):
            captured["messages"] = messages
            captured["response_model"] = response_model
            return runtime_observer.RuntimeObserverReport(
                headline="Regulation table extracted",
                detail="The regulations specialist has established the planned study sequence and the grade specialist is comparing thesis outcomes.",
                active_agent="Degree Regulations Specialist",
                agent_updates=[
                    runtime_observer.AgentProgressReport(
                        agent="Degree Regulations Specialist",
                        state="completed",
                        summary="The regulations specialist has established where the thesis fits in the study sequence.",
                    )
                ],
                evidence=["tool_finish: extract_regelstudienplan_table"],
            )

    def fake_get_default_llm(**kwargs):
        captured["llm_kwargs"] = kwargs
        return FakeLlm()

    monkeypatch.setattr(
        runtime_observer,
        "resolve_study_assistant_observer_model",
        lambda **kwargs: "meta-llama-3.1-8b-instruct",
    )
    monkeypatch.setattr(runtime_observer, "resolve_study_assistant_observer_timeout", lambda: 120)
    monkeypatch.setattr(runtime_observer, "get_default_llm", fake_get_default_llm)

    result = runtime_observer.generate_runtime_observer_report(
        [
            {
                "event": "task_completed",
                "agent_label": "Degree Regulations Specialist",
                "status": "ok",
            }
        ]
    )

    assert result.model == "meta-llama-3.1-8b-instruct"
    assert result.report.headline == "Regulation table extracted"
    assert captured["llm_kwargs"]["timeout"] == 120
    assert captured["response_model"] is runtime_observer.RuntimeObserverReport
    assert "generic filler" in captured["messages"][0]["content"]
    assert "raw task identifiers" in captured["messages"][0]["content"]
    assert "must never answer the student's request" in captured["messages"][0]["content"]
    assert "study_context is background" in captured["messages"][0]["content"]
    assert "response_ready is authoritative" in captured["messages"][0]["content"]
    assert "run_stage and lifecycle.crew_completed are absolute" in captured["messages"][0]["content"]
    assert "/no_think" in captured["messages"][0]["content"]
    assert result.report.agent_updates[0].state == "completed"


def test_deterministic_report_summarizes_request_without_raw_llm_metadata():
    raw_prompt = "Based on these tasks summary, create a step by step plan. " * 500
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie mache ich jetzt am besten mit meinem Master weiter, um die bestmögliche Note zu erreichen?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "llm_started",
            "agent_label": "Orchestrator",
            "call_id": "38d56fbc-6240-425b-8dfd-584aac0c0",
            "task_name": raw_prompt,
            "activity": (
                "Orchestrator LLM call #38d56fbc-6240-425b-8dfd-584aac0c0 started; "
                "0 tool schemas exposed."
            ),
            "status": "running",
        },
    ]

    report = runtime_observer.deterministic_runtime_observer_report(events)
    rendered = " ".join(
        [report.headline, report.detail, *(item.summary for item in report.agent_updates)]
    )

    assert report.headline == "Anfrage wird eingeordnet"
    assert "Masterplanung" in report.detail
    assert "bestmöglichen Abschlussnote" in report.detail
    assert report.agent_updates[0].agent == "Orchestrator"
    assert "38d56fbc" not in rendered
    assert "tool schema" not in rendered
    assert "Based on these tasks" not in rendered


def test_observer_guardrail_rejects_student_answer_and_recommendation_copy():
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie weit bin ich im Master und was sollte ich für die beste Note tun?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "intent_classified",
            "intent": {
                "route": "deep_dive",
                "required_sources": ["grade_manager", "grade_optimization"],
            },
            "agent_label": "Orchestrator",
            "status": "ok",
        },
        {
            "event": "route_execution_started",
            "agent_label": "Orchestrator",
            "status": "running",
        },
    ]
    bad_report = runtime_observer.RuntimeObserverReport(
        headline=(
            "Aktueller Fortschritt im Masterstudium und strategische Empfehlungen "
            "zur optimalen Notenmaximierung"
        ),
        detail=(
            "Sie sind im Masterstudium und haben bisher 35 Module abgeschlossen. "
            "Um das beste Ergebnis zu erzielen, sollten Sie sich auf hoch gewichtete Module konzentrieren."
        ),
        active_agent="Orchestrator",
        agent_updates=[
            runtime_observer.AgentProgressReport(
                agent="Orchestrator",
                state="active",
                summary="Sie sollten jetzt zuerst die prüfungsrelevanten Module priorisieren.",
            )
        ],
    )

    normalized = runtime_observer.reconcile_runtime_observer_report(events, bad_report)
    rendered = " ".join(
        [normalized.headline, normalized.detail, *(item.summary for item in normalized.agent_updates)]
    )

    assert normalized.headline == "Laufende Analyse"
    assert normalized.detail.startswith("Die Anfrage zur weiteren Masterplanung")
    assert normalized.agent_updates[0].summary.startswith("Koordiniert die Fachanalysen")
    assert "Sie " not in rendered
    assert "sollten" not in rendered
    assert "Empfehlungen" not in rendered
    assert "35 Module" not in rendered


def test_completed_agent_drops_active_wording_but_keeps_grounded_overall_finding():
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie optimiere ich meinen Masterabschluss?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "tool_start",
            "call_id": 3,
            "agent_label": "Orchestrator",
            "tool_name": "ask_question_to_coworker",
            "tool_input": {
                "coworker": "TU Berlin Personal Study Advisor",
                "task": "Prüfe den aktuellen Studienstand.",
            },
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 3,
                "agent_label": "Orchestrator",
                "tool_name": "ask_question_to_coworker",
                "tool_input": {
                    "coworker": "TU Berlin Personal Study Advisor",
                    "task": "Prüfe den aktuellen Studienstand.",
                },
                "status": "ok",
                "output_preview": "57 LP sind abgeschlossen.",
            },
        },
    ]
    report = runtime_observer.RuntimeObserverReport(
        headline="Study Advisor prüft Studienstand",
        detail=(
            "Der Study Advisor wertet die abgerufenen Modul- und Anforderungsdaten aus. "
            "Der Studienstand umfasst bereits 57 abgeschlossene LP."
        ),
        active_agent="Study Advisor",
        agent_updates=[
            runtime_observer.AgentProgressReport(
                agent="Study Advisor",
                state="active",
                summary=(
                    "Der Study Advisor wertet die abgerufenen Modul- und Anforderungsdaten aus, "
                    "um den Leistungspunktestand zu verifizieren."
                ),
            )
        ],
    )

    normalized = runtime_observer.reconcile_runtime_observer_report(events, report)
    advisor = next(item for item in normalized.agent_updates if item.agent == "Study Advisor")

    assert advisor.state == "completed"
    assert "wertet" not in advisor.summary
    assert advisor.summary.startswith("Hat Studienstand")
    assert normalized.active_agent == "Orchestrator"
    assert normalized.headline == "Ergebnisse werden zusammengeführt"
    assert normalized.detail == "Der Studienstand umfasst bereits 57 abgeschlossene LP."


def test_report_normalization_drops_fake_runtime_repetition_and_forces_completed_german_state():
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie beeinflusst die Masterarbeit meine Note und was ist realistisch?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "tool_start",
            "call_id": 7,
            "agent_label": "Orchestrator",
            "tool_name": "Delegate work to coworker",
            "tool_input": {
                "coworker": "TU Berlin Degree Regulations Specialist",
                "task": "Check thesis weighting and final grade rules.",
            },
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 7,
                "agent_label": "Orchestrator",
                "tool_name": "Delegate work to coworker",
                "tool_input": {
                    "coworker": "TU Berlin Degree Regulations Specialist",
                    "task": "Check thesis weighting and final grade rules.",
                },
                "status": "ok",
                "output_preview": "The thesis carries 30 LP.",
            },
        },
        {"event": "crew_completed", "status": "ok"},
    ]
    raw = runtime_observer.RuntimeObserverReport(
        headline="Deep Dive into Grade Optimization",
        detail=(
            "The Orchestrator is currently planning the delegation sequence. "
            "The CrewAI runtime has started a tool execution with input keys. "
            "The Degree Regulations Specialist is currently determining the formal rules."
        ),
        active_agent="Orchestrator",
        agent_updates=[
            runtime_observer.AgentProgressReport(
                agent="Degree Regulations Specialist",
                state="active",
                summary="Determining the formal thesis weighting rules.",
            ),
            runtime_observer.AgentProgressReport(
                agent="CrewAI runtime",
                state="active",
                summary="Executing search_degree_regulation_pd_fs with input keys.",
            ),
        ],
        evidence=["student_request", "tool_finish: regulation search"],
    )

    normalized = runtime_observer.reconcile_runtime_observer_report(events, raw)

    assert normalized.headline == "Analyse abgeschlossen"
    assert normalized.detail.startswith("Die angeforderten Fachanalysen sind abgeschlossen")
    assert normalized.active_agent is None
    assert {item.agent for item in normalized.agent_updates} == {
        "Orchestrator",
        "Degree Regulations Specialist",
    }
    assert {item.state for item in normalized.agent_updates} == {"completed"}
    assert all("CrewAI runtime" not in item.summary for item in normalized.agent_updates)
    assert normalized.evidence[0] == "run_stage=complete"
    assert all("regulation search" not in item for item in normalized.evidence)


def test_a2a_response_is_the_completion_boundary_and_usage_ok_does_not_reopen_agent():
    delegation = {
        "coworker": "TU Berlin Personal Study Advisor",
        "task": "Prüfe Studienstand und offene Anforderungen.",
    }
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie weit bin ich im Master?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "tool_start",
            "call_id": 4,
            "agent_label": "Orchestrator",
            "tool_name": "delegate_work_to_coworker",
            "tool_input": delegation,
            "status": "running",
        },
        {
            "event": "task_completed",
            "agent_label": "Study Advisor",
            "status": "ok",
            "output_preview": "57 LP wurden geprüft.",
        },
    ]

    before_response = runtime_observer.runtime_observer_snapshot(events)
    advisor = next(item for item in before_response["agent_states"] if item["agent"] == "Study Advisor")
    assert advisor["state"] == "active"
    assert advisor["response_ready"] is False
    assert before_response["run_stage"] == "execution"

    events.extend(
        [
            {
                "event": "tool_finish",
                "tool_call": {
                    "call_id": 4,
                    "agent_label": "Orchestrator",
                    "tool_name": "delegate_work_to_coworker",
                    "tool_input": delegation,
                    "status": "ok",
                    "output_preview": "57 LP sind abgeschlossen; 63 LP sind noch offen.",
                },
            },
            {
                "event": "tool_usage_ok",
                "agent_label": "Orchestrator",
                "tool_name": "delegate_work_to_coworker",
                "tool_input": delegation,
                "status": "ok",
            },
            {
                "event": "llm_started",
                "agent_label": "Orchestrator",
                "status": "running",
            },
        ]
    )

    after_response = runtime_observer.runtime_observer_snapshot(events)
    advisor = next(item for item in after_response["agent_states"] if item["agent"] == "Study Advisor")
    assert advisor["state"] == "completed"
    assert advisor["response_ready"] is True
    assert after_response["run_stage"] == "synthesis"
    assert after_response["lifecycle"]["responses_available_for"] == ["Study Advisor"]


def test_global_completion_and_past_tense_are_rejected_while_a2a_response_is_missing():
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie weit bin ich im Master und wie optimiere ich meine Note?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "tool_start",
            "call_id": 2,
            "agent_label": "Orchestrator",
            "tool_name": "delegate_work_to_coworker",
            "tool_input": {
                "coworker": "TU Berlin Grade Optimization Specialist",
                "task": "Berechne Notenprognose und Sensitivität.",
            },
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 3,
                "agent_label": "Grade Optimization Specialist",
                "tool_name": "run_grade_sensitivity_analysis",
                "status": "ok",
                "output_preview": "Die Masterarbeit hat den größten Einfluss.",
            },
        },
    ]
    bad = runtime_observer.RuntimeObserverReport(
        headline="Aktueller Stand der Analyse",
        detail=(
            "Die tiefgehende Analyse des aktuellen Studienfortschritts im Master und "
            "der Möglichkeiten zur Notenoptimierung ist abgeschlossen."
        ),
        active_agent="Grade Optimization Specialist",
        agent_updates=[
            runtime_observer.AgentProgressReport(
                agent="Grade Optimization Specialist",
                state="active",
                summary="Hat eine detaillierte Notenprognose und Sensitivitätsanalyse durchgeführt.",
            )
        ],
        evidence=[
            "MOSES: Alle 120 LP sind abgedeckt.",
            "ISIS: Keine aktiven Deadlines.",
        ],
    )

    normalized = runtime_observer.reconcile_runtime_observer_report(events, bad)
    optimizer = next(item for item in normalized.agent_updates if item.agent == "Grade Optimization Specialist")

    assert normalized.headline == "Aktueller Stand der Analyse"
    assert "abgeschlossen" not in normalized.detail.casefold()
    assert "noch ausstehende a2a-responses" in normalized.detail.casefold()
    assert optimizer.state == "active"
    assert optimizer.summary.startswith("Vergleicht Notenszenarien")
    assert normalized.evidence[0] == "run_stage=execution"
    assert all("MOSES" not in item and "ISIS" not in item for item in normalized.evidence)


def test_synthesis_summary_cannot_claim_whole_run_is_complete_before_crew_completed():
    delegation = {
        "coworker": "TU Berlin Personal Study Advisor",
        "task": "Prüfe den Studienstand.",
    }
    events = [
        {
            "event": "ui_run_started",
            "query": "Wie weit bin ich im Master?",
            "agent_label": "Orchestrator",
            "status": "running",
        },
        {
            "event": "tool_start",
            "call_id": 1,
            "agent_label": "Orchestrator",
            "tool_name": "delegate_work_to_coworker",
            "tool_input": delegation,
            "status": "running",
        },
        {
            "event": "tool_finish",
            "tool_call": {
                "call_id": 1,
                "agent_label": "Orchestrator",
                "tool_name": "delegate_work_to_coworker",
                "tool_input": delegation,
                "status": "ok",
                "output_preview": "57 LP sind abgeschlossen.",
            },
        },
        {
            "event": "llm_started",
            "agent_label": "Orchestrator",
            "status": "running",
        },
    ]
    bad = runtime_observer.RuntimeObserverReport(
        headline="Analyse abgeschlossen",
        detail="Die gesamte Analyse ist abgeschlossen.",
        active_agent="Orchestrator",
        agent_updates=[],
    )

    normalized = runtime_observer.reconcile_runtime_observer_report(events, bad)

    assert normalized.headline == "Ergebnisse werden zusammengeführt"
    assert normalized.detail.startswith("Die Fachagenten haben ihre Responses zurückgeliefert")
    assert "abgeschlossen" not in normalized.detail.casefold()
    assert normalized.active_agent == "Orchestrator"
