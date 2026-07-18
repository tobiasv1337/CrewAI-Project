from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import main as cli
from crewai.types.streaming import FlowStreamingOutput, StreamChunk, StreamFrame, StreamSession


def test_flow_stream_is_exhausted_before_result_is_accessed():
    events = []
    streaming = FlowStreamingOutput()
    final_result = object()

    def chunks():
        yield StreamChunk(
            content="A streamed answer",
            task_index=0,
            task_name="Answer task",
            agent_role="TU Berlin Study Advisor",
        )
        streaming._set_result(final_result)

    streaming._sync_iterator = chunks()

    result = cli._consume_flow_streaming_output(streaming, on_trace_event=events.append)

    assert result is final_result
    assert streaming.is_completed is True
    assert [event["event"] for event in events] == [
        "answer_stream_started",
        "llm_stream_chunk",
        "answer_stream_completed",
    ]
    assert events[1]["content"] == "A streamed answer"


def test_current_flow_stream_session_is_exhausted_before_result_is_accessed():
    events = []
    streaming = StreamSession()
    final_result = object()
    timestamp = datetime.now(timezone.utc)

    def frames():
        yield StreamFrame(
            id="flow-frame",
            type="flow_started",
            channel="flow",
            timestamp=timestamp,
            data={"flow_name": "StudyChatFlow"},
        )
        yield StreamFrame(
            id="answer-frame",
            type="llm_stream_chunk",
            channel="llm",
            timestamp=timestamp,
            data={
                "chunk": "A streamed frame answer",
                "agent_role": "TU Berlin Study Advisor",
                "task_name": "Answer task",
            },
        )
        streaming._set_result(final_result)

    streaming._sync_iterator = frames()

    result = cli._consume_flow_streaming_output(streaming, on_trace_event=events.append)

    assert result is final_result
    assert streaming.is_completed is True
    assert streaming.is_exhausted is True
    assert [event["event"] for event in events] == [
        "answer_stream_started",
        "llm_stream_chunk",
        "answer_stream_completed",
    ]
    assert events[1]["content"] == "A streamed frame answer"
    assert events[1]["agent_label"] == "Study Advisor"


def test_ask_study_assistant_dispatches_runner_with_multi_agent_options(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run_study_assistant_query(**kwargs):
        calls.append(kwargs)
        return cli.MultiAgentStudyAssistantRunResult(
            answer="Multi-agent answer",
            tool_summary_lines=[
                "1. Get Study Plan Snapshot args={} -> 42 chars",
                "2. Search TU Berlin MOSES Modules args={} -> 100 chars",
            ],
            trace_dir=tmp_path / "multi-agent-run-1",
        )

    monkeypatch.setattr(cli, "run_study_assistant_query", fake_run_study_assistant_query)

    exit_code = cli.main(
        [
            "ask-study-assistant",
            "What should I take next semester?",
            "--isis-context-json",
            '{"fallback_search_terms": ["Machine Learning 1"]}',
            "--allow-temp-enrollment",
            "--manager-model",
            "qwen-manager",
            "--observer-model",
            "llama-observer",
            "--model",
            "qwen-specialists",
            "--temperature",
            "0.4",
            "--top-p",
            "0.8",
            "--student-context",
            "Program: Computer Science M.Sc.",
            "--trace-full",
            "--verbose",
            "--no-cache",
            "--logs-root",
            str(tmp_path),
            "--run-id",
            "fixed-multi-agent-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [
        {
            "query": "What should I take next semester?",
            "student_context": "Program: Computer Science M.Sc.",
            "isis_context_json": '{"fallback_search_terms": ["Machine Learning 1"]}',
            "allow_temp_enrollment": True,
            "model": "qwen-specialists",
            "manager_model": "qwen-manager",
            "observer_model": "llama-observer",
            "planning_enabled": False,
            "planning_llm_model": None,
            "temperature": 0.4,
            "top_p": 0.8,
            "trace": True,
            "trace_full": True,
            "verbose": True,
            "cache": False,
            "logs_root": tmp_path,
            "run_id": "fixed-multi-agent-run",
        }
    ]
    assert "# Multi-Agent Study Assistant Answer" in captured.out
    assert "Multi-agent answer" in captured.out
    assert "Get Study Plan Snapshot" in captured.out
    assert "Search TU Berlin MOSES Modules" in captured.out
    assert f"Trace directory: {Path(tmp_path) / 'multi-agent-run-1'}" in captured.out
