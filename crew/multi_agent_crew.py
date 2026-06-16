from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from crew.config.llm import get_default_llm
from crew.runtime import ensure_crewai_storage_writable
from crew.tools import COURSE_COMMITMENT_TOOLS, MOSES_MODULE_RESEARCH_TOOLS, STUDY_ADVISOR_TOOLS, make_isis_read_only_tools


@CrewBase
class MultiAgentStudyAssistantCrew:
    """Hierarchical multi-agent entry point for the TU Study Assistant."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/multi_agent_agents.yaml"
    tasks_config = "config/multi_agent_tasks.yaml"

    def __init__(
        self,
        *,
        model: str | None = None,
        manager_model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        allow_temp_enrollment: bool = False,
        verbose: bool = False,
        cache: bool = True,
        planning_enabled: bool = False,
        planning_llm_model: str | None = None,
    ) -> None:
        self.model = model
        self.manager_model = manager_model or model
        self.temperature = temperature
        self.top_p = top_p
        self.allow_temp_enrollment = allow_temp_enrollment
        self.verbose = verbose
        self.cache = cache
        self.planning_enabled = planning_enabled
        self.planning_llm_model = planning_llm_model or self.manager_model

    def _llm(self, *, manager: bool = False):
        return get_default_llm(
            model=self.manager_model if manager else self.model,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def _orchestrator(self) -> Agent:
        return Agent(
            config=self.agents_config["orchestrator"],  # type: ignore[index]
            tools=[],
            llm=self._llm(manager=True),
            verbose=self.verbose,
            cache=self.cache,
            inject_date=True,
            date_format="%Y-%m-%d",
        )

    @agent
    def study_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config["study_advisor"],  # type: ignore[index]
            tools=list(STUDY_ADVISOR_TOOLS),
            llm=self._llm(),
            verbose=self.verbose,
            cache=self.cache,
            inject_date=True,
            date_format="%Y-%m-%d",
        )

    @agent
    def module_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["module_researcher"],  # type: ignore[index]
            tools=list(MOSES_MODULE_RESEARCH_TOOLS),
            llm=self._llm(),
            verbose=self.verbose,
            cache=self.cache,
            inject_date=True,
            date_format="%Y-%m-%d",
        )

    @agent
    def course_info_specialist(self) -> Agent:
        tools = make_isis_read_only_tools(allow_temp_enrollment=self.allow_temp_enrollment)
        return Agent(
            config=self.agents_config["course_info_specialist"],  # type: ignore[index]
            tools=tools,
            llm=self._llm(),
            verbose=self.verbose,
            cache=self.cache,
            inject_date=True,
            date_format="%Y-%m-%d",
        )

    @agent
    def course_commitment_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["course_commitment_specialist"],  # type: ignore[index]
            tools=list(COURSE_COMMITMENT_TOOLS),
            llm=self._llm(),
            verbose=self.verbose,
            cache=self.cache,
            inject_date=True,
            date_format="%Y-%m-%d",
        )

    @task
    def study_assistant_task(self) -> Task:
        return Task(
            config=self.tasks_config["study_assistant_task"],  # type: ignore[index]
            markdown=True,
        )

    @crew
    def crew(self) -> Crew:
        ensure_crewai_storage_writable()
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self._orchestrator(),
            verbose=self.verbose,
            cache=self.cache,
            planning=self.planning_enabled,
            planning_llm=(
                get_default_llm(
                    model=self.planning_llm_model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                if self.planning_enabled
                else None
            ),
        )
