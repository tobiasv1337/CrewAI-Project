from __future__ import annotations

from pathlib import Path

import main as cli


def test_ask_study_advisor_dispatches_runner_with_trace_options(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run_study_advisor_query(**kwargs):
        calls.append(kwargs)
        return cli.StudyAdvisorAgentRunResult(
            answer="Advisor answer",
            tool_summary_lines=["1. Get Study Plan Snapshot args={} -> 42 chars"],
            trace_dir=tmp_path / "advisor-run-1",
        )

    monkeypatch.setattr(cli, "run_study_advisor_query", fake_run_study_advisor_query)

    exit_code = cli.main(
        [
            "ask-study-advisor",
            "What should I take next semester?",
            "--model",
            "gemma-4-31b-instruct",
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
            "fixed-advisor-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [
        {
            "query": "What should I take next semester?",
            "student_context": "Program: Computer Science M.Sc.",
            "model": "gemma-4-31b-instruct",
            "temperature": 0.4,
            "top_p": 0.8,
            "trace": True,
            "trace_full": True,
            "verbose": True,
            "cache": False,
            "logs_root": tmp_path,
            "run_id": "fixed-advisor-run",
        }
    ]
    assert "# Study Advisor Answer" in captured.out
    assert "Advisor answer" in captured.out
    assert "Get Study Plan Snapshot" in captured.out
    assert f"Trace directory: {Path(tmp_path) / 'advisor-run-1'}" in captured.out
