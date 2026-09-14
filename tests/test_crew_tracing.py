from __future__ import annotations

import json
from types import SimpleNamespace

from crewai.hooks import get_after_tool_call_hooks, get_before_tool_call_hooks

from crew.tracing import (
    build_trace_workbench,
    capture_tool_traces,
    load_trace_workbench,
    suppress_trace_events,
    suppress_trace_events_from,
    tool_call_summary_from_event,
)


def test_auxiliary_llm_trace_events_can_be_suppressed(tmp_path):
    events = []

    with capture_tool_traces(
        enabled=True,
        query="Observe progress",
        logs_root=tmp_path,
        run_id="suppressed-observer-trace",
        on_event=events.append,
    ) as recorder:
        with suppress_trace_events():
            recorder._emit_event(
                {
                    "event": "llm_started",
                    "agent_label": "Orchestrator",
                    "model": "observer-model",
                    "status": "running",
                }
            )

    assert events == []


def test_auxiliary_llm_source_events_are_suppressed_across_event_bus_threads(tmp_path):
    events = []
    auxiliary_llm = SimpleNamespace()
    suppress_trace_events_from(auxiliary_llm)

    with capture_tool_traces(
        enabled=True,
        query="Observe progress",
        logs_root=tmp_path,
        run_id="suppressed-observer-source",
        on_event=events.append,
    ) as recorder:
        recorder._on_llm_started(
            auxiliary_llm,
            SimpleNamespace(
                agent_role=None,
                call_id="observer-1",
                model="observer-model",
                task_name=None,
                tools=[],
            ),
        )
        recorder._on_llm_completed(
            auxiliary_llm,
            SimpleNamespace(
                agent_role=None,
                call_id="observer-1",
                model="observer-model",
                task_name=None,
                usage={"total_tokens": 8},
                finish_reason="stop",
            ),
        )

    assert events == []
    assert not (tmp_path / "suppressed-observer-source" / "events.jsonl").exists()


def test_capture_tool_traces_registers_unregisters_and_writes_files(tmp_path):
    before_count = len(get_before_tool_call_hooks())
    after_count = len(get_after_tool_call_hooks())

    with capture_tool_traces(
        enabled=True,
        query="Find ML modules",
        model="devstral",
        temperature=0.2,
        logs_root=tmp_path,
        run_id="trace-test",
    ) as recorder:
        assert len(get_before_tool_call_hooks()) == before_count + 1
        assert len(get_after_tool_call_hooks()) == after_count + 1

        context = SimpleNamespace(
            tool_name="Search TU Berlin MOSES Modules",
            tool_input={"query": "Machine Learning", "max_results": 3},
            tool=None,
            agent=SimpleNamespace(role="TU Berlin MOSES Module Researcher"),
            task=SimpleNamespace(name="module_research_task", description="Task description"),
            tool_result="# Search result\n\nMachine Learning 1",
        )
        recorder.before_tool_call(context)
        recorder.after_tool_call(context)
        recorder.write_answer("Final answer", usage_metrics={"total_tokens": 123}, state={"isis_context": {"fallback_search_terms": ["ML"]}})

    assert len(get_before_tool_call_hooks()) == before_count
    assert len(get_after_tool_call_hooks()) == after_count

    run_dir = tmp_path / "trace-test"
    trace_lines = (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 1
    event = json.loads(trace_lines[0])
    assert event["tool_name"] == "Search TU Berlin MOSES Modules"
    assert event["tool_input"] == {"query": "Machine Learning", "max_results": 3}
    assert "Machine Learning 1" in event["output_preview"]

    assert (run_dir / "answer.md").read_text(encoding="utf-8") == "Final answer"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state == {"isis_context": {"fallback_search_terms": ["ML"]}}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["query"] == "Find ML modules"
    assert summary["model"] == "devstral"
    assert summary["tool_call_count"] == 1
    assert summary["state_path"] == str(run_dir / "state.json")
    assert summary["usage_metrics"] == {"total_tokens": 123}

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Prompt" in report
    assert "Find ML modules" in report
    assert "## Observable Execution Trace" in report
    assert "Search TU Berlin MOSES Modules" in report
    assert "## Final Answer" in report


def test_capture_tool_traces_emits_live_events_and_workbench(tmp_path):
    events = []

    with capture_tool_traces(
        enabled=True,
        query="What is due this week?",
        model="devstral",
        logs_root=tmp_path,
        run_id="live-trace-test",
        on_event=events.append,
    ) as recorder:
        context = SimpleNamespace(
            tool_name="Get ISIS Course Assignments",
            tool_input={"course_id": 47025},
            tool=None,
            agent=SimpleNamespace(role="TU Berlin ISIS Course Information Specialist"),
            task=SimpleNamespace(name="study_assistant_task", description="Task description"),
            tool_result=(
                "# ISIS course assignments\n\n"
                "## Access\n"
                "- Initially enrolled: no\n"
                "- Temporary enrollment allowed: yes\n"
                "- Temporarily enrolled by this tool call: yes\n"
                "- Cleanup attempted: yes; succeeded: yes\n"
            ),
        )
        recorder.before_tool_call(context)
        recorder.after_tool_call(context)
        recorder.write_answer("Final answer")

    assert [event["event"] for event in events] == ["tool_start", "tool_finish"]
    assert events[0]["agent_label"] == "ISIS Course Info Specialist"
    assert events[0]["status"] == "running"
    finished = events[1]["tool_call"]
    assert finished["badges"] == ["ISIS", "temporary enrollment", "cleanup succeeded"]
    assert events[1]["workbench"]["groups"][0]["agent_label"] == "ISIS Course Info Specialist"

    summary = json.loads((tmp_path / "live-trace-test" / "summary.json").read_text(encoding="utf-8"))
    assert summary["workbench"]["total_tool_calls"] == 1
    assert summary["workbench"]["groups"][0]["tool_calls"][0]["status"] == "ok"


def test_lifecycle_journal_is_written_before_tool_returns(tmp_path):
    with capture_tool_traces(
        enabled=True,
        query="Inspect the Regelstudienplan",
        logs_root=tmp_path,
        run_id="immediate-lifecycle-journal",
    ) as recorder:
        context = SimpleNamespace(
            tool_name="Extract Regelstudienplan Table",
            tool_input={"program_query": "M.Sc. Computer Science"},
            tool=None,
            agent=SimpleNamespace(role="TU Berlin Degree Regulations Specialist"),
            task=SimpleNamespace(name="degree_regulations_task", description="Extract the plan"),
            tool_result="Page 12: 4. Sem. Masterarbeit, 30 LP",
        )

        recorder.before_tool_call(context)

        journal_path = tmp_path / "immediate-lifecycle-journal" / "events.jsonl"
        started_events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        assert started_events[-1]["event"] == "tool_start"
        assert started_events[-1]["agent_label"] == "Degree Regulations Specialist"
        assert not (tmp_path / "immediate-lifecycle-journal" / "trace.jsonl").exists()

        recorder.after_tool_call(context)

    journal = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in journal[-2:]] == ["tool_start", "tool_finish"]
    assert "Masterarbeit" in journal[-1]["tool_call"]["output"]


def test_public_event_sink_journals_flow_and_agent_output_events(tmp_path):
    with capture_tool_traces(
        enabled=True,
        query="Inspect live events",
        logs_root=tmp_path,
        run_id="unified-event-sink",
    ) as recorder:
        recorder.emit_event(
            {
                "event": "route_execution_started",
                "agent_label": "Orchestrator",
                "status": "running",
            }
        )
        recorder.emit_event(
            {
                "event": "task_completed",
                "agent_label": "Degree Regulations Specialist",
                "status": "ok",
                "output_preview": "The fourth semester contains the 30 LP Masterarbeit.",
            }
        )

    journal_path = tmp_path / "unified-event-sink" / "events.jsonl"
    journal = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in journal] == ["route_execution_started", "task_completed"]
    assert journal[1]["output_preview"].endswith("Masterarbeit.")
    assert [event["event_id"] for event in journal] == [1, 2]


def test_trace_pairing_preserves_fifo_for_same_tool_calls(tmp_path):
    with capture_tool_traces(
        enabled=True,
        query="Resolve several ISIS courses",
        model="devstral",
        logs_root=tmp_path,
        run_id="same-tool-pairing",
    ) as recorder:
        first = SimpleNamespace(
            tool_name="Permanently Enroll In ISIS Course",
            tool_input={"course_query": "Systemprogrammierung"},
            tool=None,
            agent=SimpleNamespace(role="TU Berlin Course Commitment Specialist"),
            task=SimpleNamespace(name="study_assistant_task", description="Task description"),
            tool_result="First result",
        )
        second = SimpleNamespace(
            tool_name="Permanently Enroll In ISIS Course",
            tool_input={"course_query": "Digitale Systeme"},
            tool=None,
            agent=SimpleNamespace(role="TU Berlin Course Commitment Specialist"),
            task=SimpleNamespace(name="study_assistant_task", description="Task description"),
            tool_result="Second result",
        )

        recorder.before_tool_call(first)
        recorder.before_tool_call(second)
        recorder.after_tool_call(first)
        recorder.after_tool_call(second)

    calls = recorder.ordered_tool_calls
    assert [call.tool_input["course_query"] for call in calls] == ["Systemprogrammierung", "Digitale Systeme"]
    assert [call.output_preview for call in calls] == ["First result", "Second result"]


def test_degree_regulations_tools_are_labeled_for_workbench():
    call = tool_call_summary_from_event(
        {
            "call_id": 1,
            "tool_name": "Search Degree Regulation PDFs",
            "tool_input": {"query": "AllgStuPO Wiederholungsprüfung"},
            "output_preview": "# Relevant regulation PDF passages",
            "output_chars": 34,
            "agent_role": "TU Berlin Degree Regulations Specialist",
            "duration_ms": 90,
        }
    )
    workbench = build_trace_workbench([call], run_id="regulations")

    assert call.agent_label == "Degree Regulations Specialist"
    assert call.source_system == "Degree Regulations"
    assert call.badges == ["Degree Regulations"]
    assert [group.agent_label for group in workbench.groups] == ["Degree Regulations Specialist"]
    assert [item["active"] for item in workbench.source_flow] == [False, False, False, True, False, False, True]


def test_commitment_execution_tool_is_labeled_for_workbench():
    call = tool_call_summary_from_event(
        {
            "call_id": 1,
            "tool_name": "Execute Approved Course Commitment Actions",
            "tool_input": {"action_ids": []},
            "output_preview": "# Approved course commitment execution",
            "output_chars": 38,
            "agent_role": "TU Berlin Course Commitment Specialist",
            "duration_ms": 90,
        }
    )

    assert call.agent_label == "Course Commitment Specialist"
    assert call.source_system == "Course Commitment"
    assert call.badges == ["Course Commitment"]


def test_capture_tool_traces_emits_lifecycle_events(tmp_path):
    events = []

    with capture_tool_traces(
        enabled=True,
        query="What courses do I do?",
        model="devstral",
        logs_root=tmp_path,
        run_id="lifecycle-test",
        on_event=events.append,
    ) as recorder:
        task = SimpleNamespace(
            name="study_assistant_task",
            description="Answer the student request",
            agent=SimpleNamespace(role="TU Berlin Personal Study Advisor"),
        )
        recorder._on_crew_started(SimpleNamespace(name="Study Crew"), SimpleNamespace(crew_name="Study Crew"))
        recorder._on_task_started(task, SimpleNamespace(task=task))
        recorder._on_llm_started(
            None,
            SimpleNamespace(
                agent_role="TU Berlin Personal Study Advisor",
                call_id="llm-1",
                model="devstral",
                task_name="PRIVATE PLANNING PROMPT " * 500,
                tools=[{"function": {"name": "Get Study Plan Snapshot"}}],
            ),
        )
        recorder._on_llm_completed(
            None,
            SimpleNamespace(
                agent_role="TU Berlin Personal Study Advisor",
                call_id="llm-1",
                model="devstral",
                task_name="study_assistant_task",
                usage={"total_tokens": 42},
                finish_reason="stop",
            ),
        )
        recorder._on_crew_completed(SimpleNamespace(name="Study Crew"), SimpleNamespace(crew_name="Study Crew", total_tokens=42))

    assert [event["event"] for event in events] == [
        "crew_started",
        "task_started",
        "llm_started",
        "llm_completed",
        "crew_completed",
    ]
    assert events[1]["agent_label"] == "Study Advisor"
    assert events[2]["tool_choices"] == ["Get Study Plan Snapshot"]
    assert events[2]["task_name"].startswith("PRIVATE PLANNING PROMPT")
    assert "PRIVATE PLANNING PROMPT" not in events[2]["activity"]
    assert "llm-1" not in events[2]["activity"]
    assert events[-1]["status"] == "ok"


def test_trace_workbench_groups_calls_by_agent_and_source(tmp_path):
    events = [
        {
            "call_id": 1,
            "tool_name": "Get Study Plan Snapshot",
            "tool_input": {},
            "output_preview": "# Study plan snapshot",
            "output_chars": 21,
            "agent_role": "TU Berlin Personal Study Advisor",
            "duration_ms": 120,
        },
        {
            "call_id": 2,
            "tool_name": "Search TU Berlin MOSES Modules",
            "tool_input": {"query": "ML"},
            "output_preview": "# Moses module search",
            "output_chars": 20,
            "agent_role": "TU Berlin MOSES Module Researcher",
            "duration_ms": 200,
        },
    ]
    calls = [tool_call_summary_from_event(event) for event in events]
    workbench = build_trace_workbench(calls, run_id="grouped", run_dir=tmp_path / "grouped")

    assert [group.agent_label for group in workbench.groups] == ["Study Advisor", "MOSES Module Researcher"]
    assert [item["active"] for item in workbench.source_flow] == [True, False, True, False, False, False, True]

    run_dir = tmp_path / "grouped"
    run_dir.mkdir()
    trace_path = run_dir / "trace.jsonl"
    trace_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    loaded = load_trace_workbench(run_dir)
    assert loaded.total_tool_calls == 2
    assert loaded.artifacts["trace"] == str(trace_path)


def test_trace_workbench_groups_grade_optimization_calls(tmp_path):
    event = {
        "call_id": 3,
        "tool_name": "Run Target Grade Optimizer",
        "tool_input": {"target_grade": 1.7},
        "output_preview": "Target grade 1.7 is feasible.",
        "output_chars": 29,
        "agent_role": "TU Berlin Grade Optimization Specialist",
        "duration_ms": 90,
    }

    call = tool_call_summary_from_event(event)
    workbench = build_trace_workbench([call], run_id="grade-optimization", run_dir=tmp_path / "grade-optimization")

    assert call.agent_label == "Grade Optimization Specialist"
    assert call.source_system == "Grade Optimization"
    assert [group.agent_label for group in workbench.groups] == ["Grade Optimization Specialist"]
    assert workbench.source_flow[1] == {
        "source": "Grade Optimization",
        "agent": "Grade Optimization Specialist",
        "target": "Orchestrator",
        "active": True,
    }


def test_capture_tool_traces_can_be_disabled(tmp_path):
    with capture_tool_traces(
        enabled=False,
        query="Find ML modules",
        model="devstral",
        logs_root=tmp_path,
    ) as recorder:
        recorder.write_answer("No trace")
        assert recorder.compact_summary_lines() == ["Tool tracing disabled."]
        assert recorder.run_dir is None

    assert list(tmp_path.iterdir()) == []


def test_tool_status_distinguishes_response_failures_from_report_content():
    from crew.tracing import status_for_tool_output

    successful = [
        "# Study plan snapshot\nAdditional Courses (missing, error): missing 6 LP",
        "# Degree requirement details\n| missing | error | Seminar |",
        "# Search result\nError correction: failed transmission detection",
        "Error correction and failure detection course",
        '{"status":"success", "data":[{"error":"failed exam"}]}',
        '{"ok":true,"error":null,"results":[]}',
    ]
    failed = [
        "Error executing tool: unknown coworker",
        "Failed to retrieve ISIS grades overview: timeout",
        "Invalid MOSES module search filters: unknown degree",
        "MOSES module details lookup failed for `Machine learning`: timeout",
        "ISIS tool formatting failed after reading course 123: missing data",
        "Study-plan write refused: confirmation missing",
        "Study-plan write failed while saving: disk full",
        '{"status":"failed", "message":"timeout"}',
        '{"ok":false,"error":"timeout"}',
        '{"isError":true,"content":[]}',
    ]
    for output in successful:
        assert status_for_tool_output("Get error reports", output) == "ok", output
    for output in failed:
        assert status_for_tool_output("Search modules", output) == "error", output
    assert status_for_tool_output("ISIS", "# ISIS course access required: course 123") == "warning"
    assert status_for_tool_output("ISIS", "# ISIS course assignments\n- Cleanup attempted: yes; succeeded: no") == "warning"


def test_full_tool_output_drives_status_when_preview_omits_result():
    from crew.tracing import status_for_tool_output

    output = json.dumps({"data": "x" * 5000, "status": "failed"})
    summary = tool_call_summary_from_event({"tool_name": "Lookup", "output": output, "output_preview": output[:30]})
    assert status_for_tool_output("Lookup", output) == "error"
    assert summary.status == "error"


def test_legacy_labels_are_repaired_without_overriding_explicit_failures():
    from crew.tracing import normalize_tool_call_status

    legacy = {"status": "error", "output_preview": "# Study plan snapshot\nRule: error"}
    assert normalize_tool_call_status(legacy)["status"] == "ok"
    assert legacy["status"] == "error"  # Stored evidence remains intact.
    for failure in [
        {"status": "error"},
        {"status": "error", "output": "Error executing tool: timeout"},
        {**legacy, "status_source": "runtime"},
        {**legacy, "error": "transport interrupted"},
    ]:
        assert normalize_tool_call_status(failure)["status"] == "error"


def test_legacy_coworker_answer_containing_domain_error_is_successful():
    from crew.tracing import normalize_tool_call_status
    call = {"status": "error", "tool_name": "ask_question_to_coworker", "output": "Based on your study plan:\nThe Additional Courses rule has an error: exceeded limit."}
    assert normalize_tool_call_status(call)["status"] == "ok"
