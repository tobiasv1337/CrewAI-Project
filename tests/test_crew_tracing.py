from __future__ import annotations

import json
from types import SimpleNamespace

from crewai.hooks import get_after_tool_call_hooks, get_before_tool_call_hooks

from crew.tracing import build_trace_workbench, capture_tool_traces, load_trace_workbench, tool_call_summary_from_event


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
    assert [item["active"] for item in workbench.source_flow] == [True, True, False, True]

    run_dir = tmp_path / "grouped"
    run_dir.mkdir()
    trace_path = run_dir / "trace.jsonl"
    trace_path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    loaded = load_trace_workbench(run_dir)
    assert loaded.total_tool_calls == 2
    assert loaded.artifacts["trace"] == str(trace_path)


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
