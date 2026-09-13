from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from crew.config.llm import get_default_llm
from crew.runtime import ensure_crewai_storage_writable
from crew.tools import STUDY_ADVISOR_TOOLS


@CrewBase
class StudyAdvisorCrew:
    """Phase 3B-1 single-agent study-planning crew."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/study_advisor_agents.yaml"
    tasks_config = "config/study_advisor_tasks.yaml"

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        verbose: bool = False,
        cache: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.cache = cache

    @agent
    def study_advisor(self) -> Agent:
        llm = get_default_llm(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            thinking=False,
        )
        return Agent(
            config=self.agents_config["study_advisor"],  # type: ignore[index]
            tools=list(STUDY_ADVISOR_TOOLS),
            llm=llm,
            verbose=self.verbose,
            cache=self.cache,
            inject_date=True,
            date_format="%Y-%m-%d",
        )

    @task
    def study_advising_task(self) -> Task:
        return Task(
            config=self.tasks_config["study_advising_task"],  # type: ignore[index]
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
