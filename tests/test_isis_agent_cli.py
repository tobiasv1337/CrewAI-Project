from __future__ import annotations

from pathlib import Path

import main as cli


def test_ask_isis_agent_dispatches_runner_with_context_and_temp_flag(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run_isis_agent_query(**kwargs):
        calls.append(kwargs)
        return cli.IsisAgentRunResult(
            answer="ISIS answer",
            tool_summary_lines=["1. List My ISIS Courses args={} -> 42 chars"],
            trace_dir=tmp_path / "isis-run-1",
        )

    monkeypatch.setattr(cli, "run_isis_agent_query", fake_run_isis_agent_query)

    exit_code = cli.main(
        [
            "ask-isis-agent",
            "What is due in ML1?",
            "--isis-context-json",
            '{"fallback_search_terms": ["Machine Learning 1"]}',
            "--allow-temp-enrollment",
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
            "fixed-isis-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [
        {
            "query": "What is due in ML1?",
            "student_context": "Program: Computer Science M.Sc.",
            "isis_context_json": '{"fallback_search_terms": ["Machine Learning 1"]}',
            "allow_temp_enrollment": True,
            "model": "gemma-4-31b-instruct",
            "temperature": 0.4,
            "top_p": 0.8,
            "trace": True,
            "trace_full": True,
            "verbose": True,
            "cache": False,
            "logs_root": tmp_path,
            "run_id": "fixed-isis-run",
        }
    ]
    assert "# ISIS Agent Answer" in captured.out
    assert "ISIS answer" in captured.out
    assert "List My ISIS Courses" in captured.out
    assert f"Trace directory: {Path(tmp_path) / 'isis-run-1'}" in captured.out

