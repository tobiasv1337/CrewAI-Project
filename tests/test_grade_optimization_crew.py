from __future__ import annotations

from crewai import LLM, Process

from crew.grade_optimization_crew import GradeOptimizationCrew
from crew.tools import GRADE_ANALYSIS_TOOLS


def _fake_llm() -> LLM:
    return LLM(model="gpt-4o", provider="openai", api_key="test-key")


def test_grade_optimization_crew_loads_yaml(monkeypatch):
    import crew.grade_optimization_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    study_crew = GradeOptimizationCrew(model="gpt-4o")

    assert "grade_optimization_specialist" in study_crew.agents_config
    assert "grade_optimization_task" in study_crew.tasks_config


def test_grade_optimization_crew_uses_grade_analysis_tools_only(monkeypatch):
    import crew.grade_optimization_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    built_crew = GradeOptimizationCrew(
        model="gpt-4o",
        verbose=True,
        cache=False,
    ).crew()

    assert built_crew.process == Process.sequential
    assert built_crew.cache is False
    assert len(built_crew.agents) == 1
    assert len(built_crew.tasks) == 1
    assert built_crew.tasks[0].agent == built_crew.agents[0]

    agent_tool_names = [tool.name for tool in built_crew.agents[0].tools]
    assert agent_tool_names == [tool.name for tool in GRADE_ANALYSIS_TOOLS]
    assert "Get Grade Scenario Outlook" in agent_tool_names
    assert "Get Degree Grade Contribution Breakdown" in agent_tool_names
    assert "Run Grade Sensitivity Analysis" in agent_tool_names
    assert "Run Grade What-If Scenario" in agent_tool_names
    assert "Run Target Grade Optimizer" in agent_tool_names
    assert "Get Study Plan Snapshot" not in agent_tool_names
    assert "Add Module To Study Plan" not in agent_tool_names
    assert "Search TU Berlin MOSES Modules" not in agent_tool_names
    assert "List My ISIS Courses" not in agent_tool_names
