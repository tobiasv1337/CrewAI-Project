from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from crew.config.llm import get_default_llm
from crew.runtime import ensure_crewai_storage_writable
from crew.tools import make_isis_read_only_tools


@CrewBase
class IsisCourseInfoCrew:
    """Phase 3 ISIS/Moodle course-information crew."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/isis_agents.yaml"
    tasks_config = "config/isis_tasks.yaml"

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        allow_temp_enrollment: bool = False,
        verbose: bool = False,
        cache: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.allow_temp_enrollment = allow_temp_enrollment
        self.verbose = verbose
        self.cache = cache

    @agent
    def course_info_specialist(self) -> Agent:
        llm = get_default_llm(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            thinking=False,
        )
        return Agent(
            config=self.agents_config["course_info_specialist"],  # type: ignore[index]
            tools=make_isis_read_only_tools(allow_temp_enrollment=self.allow_temp_enrollment),
            llm=llm,
            verbose=self.verbose,
            cache=self.cache,
            inject_date=True,
            date_format="%Y-%m-%d",
        )

    @task
    def isis_course_info_task(self) -> Task:
        return Task(
            config=self.tasks_config["isis_course_info_task"],  # type: ignore[index]
            markdown=True,
        )

    @crew
    def crew(self) -> Crew:
        ensure_crewai_storage_writable()
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=self.verbose,
            cache=self.cache,
        )
