from __future__ import annotations

import csv
import json
from pathlib import Path

import main as moses_cli


def test_ask_moses_agent_dispatches_runner_with_trace_options(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run_moses_agent_query(**kwargs):
        calls.append(kwargs)
        return moses_cli.MosesAgentRunResult(
            answer="Agent answer",
            tool_summary_lines=["1. Search TU Berlin MOSES Modules args={} -> 42 chars"],
            trace_dir=tmp_path / "run-1",
        )

    monkeypatch.setattr(moses_cli, "run_moses_agent_query", fake_run_moses_agent_query)

    exit_code = moses_cli.main(
        [
            "ask-moses-agent",
            "Find ML modules",
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
            "fixed-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [
        {
            "query": "Find ML modules",
            "student_context": "Program: Computer Science M.Sc.",
            "model": "gemma-4-31b-instruct",
            "temperature": 0.4,
            "top_p": 0.8,
            "trace": True,
            "trace_full": True,
            "verbose": True,
            "cache": False,
            "logs_root": tmp_path,
            "run_id": "fixed-run",
        }
    ]
    assert "# Moses Agent Answer" in captured.out
    assert "Agent answer" in captured.out
    assert "Search TU Berlin MOSES Modules" in captured.out
    assert f"Trace directory: {tmp_path / 'run-1'}" in captured.out


def test_eval_moses_agent_writes_summary_csv(monkeypatch, tmp_path, capsys):
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "q1", "language": "en", "query": "Tell me about 40966", "expected_tools": ["details"]}),
                json.dumps({"id": "q2", "language": "de", "query": "Pflichtmodule TI", "expected_tools": ["degree"]}),
            ]
        ),
        encoding="utf-8",
    )

    import crew.tracing

    monkeypatch.setattr(crew.tracing, "make_run_id", lambda: "eval-run")

    def fake_run_moses_agent_query(**kwargs):
        return moses_cli.MosesAgentRunResult(
            answer=f"Answer for {kwargs['query']}",
            tool_summary_lines=["1. Get TU Berlin MOSES Module Details args={} -> 100 chars"],
            trace_dir=Path(kwargs["logs_root"]) / kwargs["run_id"],
        )

    monkeypatch.setattr(moses_cli, "run_moses_agent_query", fake_run_moses_agent_query)

    exit_code = moses_cli.main(
        [
            "eval-moses-agent",
            "--queries",
            str(queries_path),
            "--models",
            "model-a",
            "model-b",
            "--output-dir",
            str(tmp_path / "evals"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    summary_path = tmp_path / "evals" / "eval-run" / "summary.csv"
    assert f"Summary CSV: {summary_path}" in captured.out

    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["model"] == "model-a"
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["tools_called"] == "Get TU Berlin MOSES Module Details"
    assert rows[-1]["model"] == "model-b"
    assert rows[-1]["query_id"] == "q2"
