from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from crew.config.llm import get_default_llm
from crew.runtime import ensure_crewai_storage_writable
from crew.tools import DEGREE_REGULATIONS_TOOLS


@CrewBase
class DegreeRegulationsCrew:
    """Single-source crew for TU Berlin AllgStuPO/StuPO PDF questions."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/degree_regulations_agents.yaml"
    tasks_config = "config/degree_regulations_tasks.yaml"

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
    def degree_regulations_specialist(self) -> Agent:
        llm = get_default_llm(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return Agent(
            config=self.agents_config["degree_regulations_specialist"],  # type: ignore[index]
            tools=list(DEGREE_REGULATIONS_TOOLS),
            llm=llm,
            verbose=self.verbose,
            cache=self.cache,
            inject_date=True,
            date_format="%Y-%m-%d",
        )

    @task
    def degree_regulations_task(self) -> Task:
        return Task(
            config=self.tasks_config["degree_regulations_task"],  # type: ignore[index]
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
