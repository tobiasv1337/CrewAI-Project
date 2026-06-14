from __future__ import annotations

import json
import os

import pytest

from main import run_moses_agent_query


pytestmark = pytest.mark.live_llm


def _live_enabled() -> bool:
    return os.getenv("RUN_LIVE_CREW_TESTS") == "1" and bool(os.getenv("GWDG_API_KEY"))


@pytest.mark.skipif(not _live_enabled(), reason="Set RUN_LIVE_CREW_TESTS=1 and GWDG_API_KEY to run live CrewAI checks.")
def test_live_degree_aware_german_query_uses_degree_tools():
    result = run_moses_agent_query(
        query="Ich bin im 2. Semester Technische Informatik. Was fuer Pflichtmodule koennte ich dieses Semester machen?",
        model=os.getenv("STUDY_ASSISTANT_MODEL"),
        trace=True,
        verbose=False,
        cache=False,
        run_id="live-degree-aware",
    )

    called_tools = "\n".join(result.tool_summary_lines)
    assert "Degree" in called_tools
    assert result.answer.strip()


@pytest.mark.skipif(not _live_enabled(), reason="Set RUN_LIVE_CREW_TESTS=1 and GWDG_API_KEY to run live CrewAI checks.")
def test_live_broad_ml_query_traces_tool_arguments_and_outputs():
    result = run_moses_agent_query(
        query="Find English Machine Learning modules for Computer Science M.Sc.",
        model=os.getenv("STUDY_ASSISTANT_MODEL"),
        trace=True,
        trace_full=True,
        verbose=False,
        cache=False,
        run_id="live-ml-query",
    )

    assert result.trace_dir is not None
    trace_path = result.trace_dir / "trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert events
    assert all("tool_input" in event for event in events)
    assert all("output" in event for event in events)
    assert result.answer.strip()
