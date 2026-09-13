from importlib import import_module

import pytest

from crew.multi_agent_crew import MultiAgentStudyAssistantCrew


@pytest.fixture
def model_env(monkeypatch):
    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setenv("GWDG_API_BASE", "https://thinking-policy.example.test/v1")
    monkeypatch.setenv("STUDY_ASSISTANT_THINKING_MAX_TOKENS", "16384")


def test_hierarchical_manager_and_planner_think_but_all_specialists_do_not(model_env):
    crew = MultiAgentStudyAssistantCrew(
        model="qwen3.8-27b", manager_model="qwen3.8-27b", planning_enabled=True,
    ).crew()

    for llm in [crew.manager_agent.llm, crew.planning_llm]:
        assert llm.additional_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
        assert llm.max_tokens == 16384
    assert len(crew.agents) == 6
    for agent in crew.agents:
        assert agent.llm.additional_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize(("module", "name"), [
    ("crew.crew", "StudyAssistantCrew"),
    ("crew.study_advisor_crew", "StudyAdvisorCrew"),
    ("crew.grade_optimization_crew", "GradeOptimizationCrew"),
    ("crew.degree_regulations_crew", "DegreeRegulationsCrew"),
    ("crew.isis_crew", "IsisCourseInfoCrew"),
])
def test_standalone_specialist_routes_disable_thinking(model_env, module, name):
    crew = getattr(import_module(module), name)(model="qwen3.8-27b").crew()

    assert len(crew.agents) == 1
    assert crew.agents[0].llm.additional_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
