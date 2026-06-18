from __future__ import annotations

from crewai import LLM, Process

from crew.isis_crew import IsisCourseInfoCrew


def _fake_llm() -> LLM:
    return LLM(model="gpt-4o", provider="openai", api_key="test-key")


def test_isis_course_info_crew_loads_separate_yaml(monkeypatch):
    import crew.isis_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    study_crew = IsisCourseInfoCrew(model="gpt-4o")

    assert "course_info_specialist" in study_crew.agents_config
    assert "isis_course_info_task" in study_crew.tasks_config
    assert "module_researcher" not in study_crew.agents_config
    assert "module_research_task" not in study_crew.tasks_config


def test_isis_task_prompt_defines_grade_manager_scope_precedence(monkeypatch):
    import crew.isis_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    study_crew = IsisCourseInfoCrew(model="gpt-4o")
    task_description = study_crew.tasks_config["isis_course_info_task"]["description"]
    compact_description = " ".join(task_description.split())

    assert "grade_manager_course_scope" in task_description
    assert "Decide from the full student question" in task_description
    assert "Follow this precedence" in task_description
    assert "directly names an ISIS course" in task_description
    assert "explicitly asks for all ISIS/Moodle courses" in task_description
    assert "Answer only for those" in task_description
    assert "do not broaden a normal current-course answer" in compact_description


def test_isis_course_info_crew_uses_read_only_tools_with_temp_flag(monkeypatch):
    import crew.isis_crew as crew_module

    monkeypatch.setattr(crew_module, "get_default_llm", lambda **kwargs: _fake_llm())

    built_crew = IsisCourseInfoCrew(
        model="gpt-4o",
        allow_temp_enrollment=True,
        verbose=True,
        cache=False,
    ).crew()

    assert built_crew.process == Process.sequential
    assert built_crew.cache is False
    assert len(built_crew.agents) == 1
    assert len(built_crew.tasks) == 1
    assert built_crew.tasks[0].agent == built_crew.agents[0]

    tool_names = [tool.name for tool in built_crew.agents[0].tools]
    assert "List My ISIS Courses" in tool_names
    assert "Inspect ISIS Candidate Course With Temporary Access" in tool_names
    assert "Permanently Enroll In ISIS Course" not in tool_names
    assert all(getattr(tool, "allow_temp_enrollment", False) is True for tool in built_crew.agents[0].tools)
