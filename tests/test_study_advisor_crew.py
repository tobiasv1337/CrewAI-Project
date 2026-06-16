from __future__ import annotations

from crewai import LLM, Process

from crew.study_advisor_crew import StudyAdvisorCrew
from crew.tools import GRADE_MANAGER_READ_TOOLS, GRADE_MANAGER_WRITE_TOOLS, STUDY_ADVISOR_TOOLS


def _fake_llm() -> LLM:
    return LLM(model="gpt-4o", provider="openai", api_key="test-key")


def test_study_advisor_crew_loads_separate_yaml(monkeypatch):
    import crew.study_advisor_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    study_crew = StudyAdvisorCrew(model="gpt-4o")

    assert "study_advisor" in study_crew.agents_config
    assert "study_advising_task" in study_crew.tasks_config
    assert "module_researcher" not in study_crew.agents_config
    assert "course_info_specialist" not in study_crew.agents_config


def test_study_advisor_crew_uses_grade_manager_tools_only(monkeypatch):
    import crew.study_advisor_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    built_crew = StudyAdvisorCrew(
        model="gpt-4o",
        verbose=True,
        cache=False,
    ).crew()

    assert built_crew.process == Process.sequential
    assert built_crew.cache is False
    assert len(built_crew.agents) == 1
    assert len(built_crew.tasks) == 1
    assert built_crew.tasks[0].agent == built_crew.agents[0]

    expected_names = [tool.name for tool in STUDY_ADVISOR_TOOLS]
    agent_tool_names = [tool.name for tool in built_crew.agents[0].tools]
    read_names = [tool.name for tool in GRADE_MANAGER_READ_TOOLS]
    write_names = [tool.name for tool in GRADE_MANAGER_WRITE_TOOLS]

    assert agent_tool_names == expected_names
    assert "Add Module To Study Plan" in write_names
    assert "Add Module To Study Plan" not in agent_tool_names
    assert set(read_names).issubset(agent_tool_names)
    assert "Search TU Berlin MOSES Modules" not in agent_tool_names
    assert "Get TU Berlin MOSES Degree Area Modules" not in agent_tool_names
    assert "List My ISIS Courses" not in agent_tool_names
    assert len(agent_tool_names) == len(read_names)
