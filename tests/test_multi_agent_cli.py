from __future__ import annotations

from pathlib import Path

import main as cli


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
