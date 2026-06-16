from __future__ import annotations

from crewai import LLM, Process

from crew.degree_regulations_crew import DegreeRegulationsCrew
from crew.tools import DEGREE_REGULATIONS_TOOLS


def _fake_llm() -> LLM:
    return LLM(model="gpt-4o", provider="openai", api_key="test-key")


def test_degree_regulations_crew_loads_config_and_tools(monkeypatch):
    import crew.degree_regulations_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    built_crew = DegreeRegulationsCrew(model="gpt-4o", verbose=True, cache=False).crew()

    assert built_crew.process == Process.sequential
    assert built_crew.cache is False
    assert len(built_crew.agents) == 1
    assert len(built_crew.tasks) == 1
    agent = built_crew.agents[0]
    assert "Degree Regulations Specialist" in agent.role
    assert [tool.name for tool in agent.tools] == [tool.name for tool in DEGREE_REGULATIONS_TOOLS]
    assert built_crew.tasks[0].agent == agent
