from __future__ import annotations

import json
from types import SimpleNamespace

from crewai.hooks import get_after_tool_call_hooks, get_before_tool_call_hooks

from crew.tracing import capture_tool_traces


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
