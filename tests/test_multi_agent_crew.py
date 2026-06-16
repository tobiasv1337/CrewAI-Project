from __future__ import annotations

from crewai import LLM, Process

from crew.multi_agent_crew import MultiAgentStudyAssistantCrew
from crew.tools import COURSE_COMMITMENT_TOOLS, MOSES_MODULE_RESEARCH_TOOLS, STUDY_ADVISOR_TOOLS


def _fake_llm() -> LLM:
    return LLM(model="gpt-4o", provider="openai", api_key="test-key")


def _agent_with_role_fragment(agents, fragment: str):
    return next(agent for agent in agents if fragment in agent.role)


def test_multi_agent_crew_loads_yaml_keys(monkeypatch):
    import crew.multi_agent_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    study_crew = MultiAgentStudyAssistantCrew(model="gpt-4o")

    assert "orchestrator" in study_crew.agents_config
    assert "study_advisor" in study_crew.agents_config
    assert "module_researcher" in study_crew.agents_config
    assert "course_info_specialist" in study_crew.agents_config
    assert "course_commitment_specialist" in study_crew.agents_config
    assert "study_assistant_task" in study_crew.tasks_config


def test_multi_agent_crew_uses_hierarchical_manager_and_specialist_tools(monkeypatch):
    import crew.multi_agent_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    built_crew = MultiAgentStudyAssistantCrew(
        model="gpt-4o",
        manager_model="gpt-4o-manager",
        allow_temp_enrollment=True,
        verbose=True,
        cache=False,
    ).crew()

    assert built_crew.process == Process.hierarchical
    assert built_crew.cache is False
    assert len(built_crew.agents) == 4
    assert len(built_crew.tasks) == 1
    assert built_crew.tasks[0].agent is None
    assert built_crew.manager_agent is not None
    assert "Orchestrator" in built_crew.manager_agent.role
    assert built_crew.manager_agent.tools == []
    assert built_crew.manager_agent not in built_crew.agents

    study_advisor = _agent_with_role_fragment(built_crew.agents, "Personal Study Advisor")
    module_researcher = _agent_with_role_fragment(built_crew.agents, "MOSES Module Researcher")
    course_info = _agent_with_role_fragment(built_crew.agents, "ISIS Course Information Specialist")
    commitment = _agent_with_role_fragment(built_crew.agents, "Course Commitment Specialist")

    assert [tool.name for tool in study_advisor.tools] == [tool.name for tool in STUDY_ADVISOR_TOOLS]
    assert "Add Module To Study Plan" not in [tool.name for tool in study_advisor.tools]
    assert "Search TU Berlin MOSES Modules" not in [tool.name for tool in study_advisor.tools]
    assert "List My ISIS Courses" not in [tool.name for tool in study_advisor.tools]

    assert [tool.name for tool in module_researcher.tools] == [
        tool.name for tool in MOSES_MODULE_RESEARCH_TOOLS
    ]
    assert "Add Module To Study Plan" not in [tool.name for tool in module_researcher.tools]
    assert "List My ISIS Courses" not in [tool.name for tool in module_researcher.tools]

    course_tool_names = [tool.name for tool in course_info.tools]
    assert "List My ISIS Courses" in course_tool_names
    assert "Inspect ISIS Candidate Course With Temporary Access" in course_tool_names
    assert "Permanently Enroll In ISIS Course" not in course_tool_names
    assert "Search TU Berlin MOSES Modules" not in course_tool_names
    assert all(getattr(tool, "allow_temp_enrollment", False) is True for tool in course_info.tools)

    commitment_tool_names = [tool.name for tool in commitment.tools]
    assert commitment_tool_names == [tool.name for tool in COURSE_COMMITMENT_TOOLS]
    assert "Propose Course Actions For Confirmation" in commitment_tool_names
    assert "Add Module To Study Plan" in commitment_tool_names
    assert "Permanently Enroll In ISIS Course" in commitment_tool_names


def test_multi_agent_crew_can_enable_planning_without_default_openai(monkeypatch):
    import crew.multi_agent_crew as crew_module

    calls = []

    def fake_get_default_llm(**kwargs):
        calls.append(kwargs)
        return _fake_llm()

    monkeypatch.setattr(crew_module, "get_default_llm", fake_get_default_llm)

    built_crew = MultiAgentStudyAssistantCrew(
        model="specialist-model",
        manager_model="manager-model",
        planning_enabled=True,
        planning_llm_model="planner-model",
    ).crew()

    assert built_crew.planning is True
    assert built_crew.planning_llm is not None
    assert any(call.get("model") == "planner-model" for call in calls)
