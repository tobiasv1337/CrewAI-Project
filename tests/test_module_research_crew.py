from __future__ import annotations

from crewai import LLM, Process

from crew.crew import StudyAssistantCrew
from crew.tools import MOSES_MODULE_RESEARCH_TOOLS, MOSES_TOOLS


def _fake_llm() -> LLM:
    return LLM(model="gpt-4o", provider="openai", api_key="test-key")


def test_study_assistant_crew_loads_yaml_keys(monkeypatch):
    import crew.crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    study_crew = StudyAssistantCrew(model="gpt-4o")

    assert "module_researcher" in study_crew.agents_config
    assert "module_research_task" in study_crew.tasks_config


def test_module_research_crew_uses_all_read_only_moses_tools(monkeypatch):
    import crew.crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    built_crew = StudyAssistantCrew(
        model="gpt-4o",
        verbose=True,
        cache=False,
    ).crew()

    assert built_crew.process == Process.sequential
    assert built_crew.cache is False
    assert len(built_crew.agents) == 1
    assert len(built_crew.tasks) == 1
    assert built_crew.tasks[0].agent == built_crew.agents[0]

    expected_names = [tool.name for tool in MOSES_TOOLS]
    agent_tool_names = [tool.name for tool in built_crew.agents[0].tools]
    assert agent_tool_names == expected_names
    assert [tool.name for tool in MOSES_MODULE_RESEARCH_TOOLS] == expected_names
    assert len(agent_tool_names) == 7
    assert not any("Add" in name or "Study Plan" in name for name in agent_tool_names)
