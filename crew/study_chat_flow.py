from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import PrivateAttr

from crew.chat_models import (
    ActionDecision,
    ChatThreadState,
    CourseProposal,
    IntentClassification,
    ProposedAction,
    StudyChatFlowState,
)
from crew.chat_persistence import append_turn, load_chat_thread, reset_chat_thread, save_chat_thread
from crew.config.llm import get_default_llm
from crew.runtime import ensure_crewai_storage_writable
from crew.tools.grademanager_tools import (
    STUDY_PLAN_CONFIRMATION_TOKEN,
    add_module_to_study_plan,
    check_module_against_study_plan,
)
from crew.tools.isis_tools import CONFIRMATION_TOKEN, PermanentlyEnrollInIsisCourseTool
from crew.tools.proposal_tools import current_course_proposals


FlowRunner = Callable[["StudyChatFlow"], str]
Classifier = Callable[[StudyChatFlowState], IntentClassification]


@dataclass(frozen=True)
class StudyChatFlowRuntime:
    model: str | None = None
    manager_model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    allow_temp_enrollment: bool = False
    verbose: bool = False
    cache: bool = True
    planning_enabled: bool = False
    planning_llm_model: str | None = None
    use_llm_classifier: bool = True


class StudyChatFlow(Flow[StudyChatFlowState]):
    """Persistent multi-turn Flow for the Streamlit Study Chat."""

    _runtime: StudyChatFlowRuntime = PrivateAttr(default_factory=StudyChatFlowRuntime)
    _classifier: Classifier | None = PrivateAttr(default=None)
    _runner_overrides: dict[str, FlowRunner] = PrivateAttr(default_factory=dict)
    _on_trace_event: Callable[[dict[str, Any]], None] | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        runtime: StudyChatFlowRuntime | None = None,
        classifier: Classifier | None = None,
        runner_overrides: dict[str, FlowRunner] | None = None,
        on_trace_event: Callable[[dict[str, Any]], None] | None = None,
        **data: Any,
    ) -> None:
        ensure_crewai_storage_writable()
        super().__init__(**data)
        self._runtime = runtime or StudyChatFlowRuntime()
        self._classifier = classifier
        self._runner_overrides = dict(runner_overrides or {})
        self._on_trace_event = on_trace_event

    @start()
    def ingest_turn(self) -> None:
        if self.state.reset_thread:
            thread = reset_chat_thread(self.state.profile_slug, thread_id=self.state.thread_id)
        else:
            thread = load_chat_thread(self.state.profile_slug, thread_id=self.state.thread_id)
        self.state.thread = thread
        self.state.conversation_context = _conversation_context(thread)

    @router(ingest_turn)
    def classify_intent(self) -> str:
        if self.state.approved_actions:
            intent = IntentClassification(
                route="execute_confirmed_actions",
                complexity="scoped",
                required_sources=["grade_manager", "isis"],
                write_intent=True,
                rationale="User submitted explicit UI action decisions.",
            )
            self.state.intent = intent
            self.state.route = "execute_confirmed_actions"
            if self._on_trace_event:
                self._on_trace_event({
                    "event": "intent_classified",
                    "intent": intent.model_dump(mode="json"),
                    "status": "ok",
                })
            return "execute_confirmed_actions"

        classifier = self._classifier or self._classify_with_llm_or_heuristics
        intent = classifier(self.state)
        self.state.intent = intent
        self.state.route = intent.route
        if self._on_trace_event:
            self._on_trace_event({
                "event": "intent_classified",
                "intent": intent.model_dump(mode="json"),
                "status": "ok",
            })
        return intent.route

    @listen("simple_grade_manager")
    def run_simple_grade_manager(self) -> None:
        self.state.answer_markdown = self._run_route("simple_grade_manager")
        self._carry_forward_existing_proposals()

    @listen("simple_moses")
    def run_simple_moses(self) -> None:
        self.state.answer_markdown = self._run_route("simple_moses")
        self._carry_forward_existing_proposals()

    @listen("simple_isis")
    def run_simple_isis(self) -> None:
        self.state.answer_markdown = self._run_route("simple_isis")
        self._carry_forward_existing_proposals()

    @listen("recommendation")
    def run_recommendation_route(self) -> None:
        self.state.answer_markdown = self._run_route("recommendation")
        self._adopt_recorded_or_existing_proposals()

    @listen("deep_dive")
    def run_deep_dive_route(self) -> None:
        self.state.answer_markdown = self._run_route("deep_dive")
        self._adopt_recorded_or_existing_proposals()

    @listen("execute_confirmed_actions")
    def run_confirmed_action_execution(self) -> None:
        self.state.executed_actions = self._execute_action_decisions()
        self.state.proposed_actions = self._remaining_proposals()
        self.state.answer_markdown = _format_execution_answer(
            self.state.executed_actions,
            language=(self.state.intent.language if self.state.intent else "en"),
        )

    @listen(
        or_(
            run_simple_grade_manager,
            run_simple_moses,
            run_simple_isis,
            run_recommendation_route,
            run_deep_dive_route,
            run_confirmed_action_execution,
        )
    )
    def persist_turn(self) -> StudyChatFlowState:
        summary = _summarize_thread(
            self.state.thread,
            self.state.query,
            self.state.answer_markdown,
            self.state.proposed_actions,
        )
        self.state.thread = append_turn(
            self.state.profile_slug,
            user_content=self.state.query,
            assistant_content=self.state.answer_markdown,
            proposals=self.state.proposed_actions,
            rolling_summary=summary,
            thread_id=self.state.thread_id,
        )
        return self.state

    def _run_route(self, route: str) -> str:
        runner = self._runner_overrides.get(route)
        if runner:
            return runner(self)
        if route == "simple_grade_manager":
            from crew.study_advisor_crew import StudyAdvisorCrew

            result = StudyAdvisorCrew(**self._crew_kwargs()).crew().kickoff(
                inputs={
                    "query": self._contextual_query(),
                    "student_context": self.state.student_context or "No student context supplied.",
                }
            )
            return str(getattr(result, "raw", result))
        if route == "simple_moses":
            from crew.crew import StudyAssistantCrew

            result = StudyAssistantCrew(**self._crew_kwargs()).crew().kickoff(
                inputs={
                    "query": self._contextual_query(),
                    "student_context": self.state.student_context or "No student context supplied.",
                }
            )
            return str(getattr(result, "raw", result))
        if route == "simple_isis":
            from crew.isis_crew import IsisCourseInfoCrew

            result = IsisCourseInfoCrew(
                **self._crew_kwargs(),
                allow_temp_enrollment=self._runtime.allow_temp_enrollment,
            ).crew().kickoff(
                inputs={
                    "query": self._contextual_query(),
                    "student_context": self.state.student_context or "No student context supplied.",
                    "isis_context": self.state.isis_context_json or "{}",
                }
            )
            return str(getattr(result, "raw", result))

        from crew.multi_agent_crew import MultiAgentStudyAssistantCrew

        result = MultiAgentStudyAssistantCrew(
            **self._crew_kwargs(),
            manager_model=self._runtime.manager_model,
            allow_temp_enrollment=self._runtime.allow_temp_enrollment,
            planning_enabled=(self._runtime.planning_enabled or route == "deep_dive"),
            planning_llm_model=self._runtime.planning_llm_model,
        ).crew().kickoff(
            inputs={
                "query": self._contextual_query(),
                "student_context": self.state.student_context or "No student context supplied.",
                "isis_context": self.state.isis_context_json or "{}",
            }
        )
        return str(getattr(result, "raw", result))

    def _crew_kwargs(self) -> dict[str, Any]:
        return {
            "model": self._runtime.model,
            "temperature": self._runtime.temperature,
            "top_p": self._runtime.top_p,
            "verbose": self._runtime.verbose,
            "cache": self._runtime.cache,
        }

    def _contextual_query(self) -> str:
        parts = [f"Current user message:\n{self.state.query}"]
        if self.state.conversation_context:
            parts.append(f"Conversation context:\n{self.state.conversation_context}")
        if self.state.thread and self.state.thread.active_proposals:
            proposal_lines = [
                f"- {proposal.title}: {proposal.summary}"
                for proposal in self.state.thread.active_proposals
            ]
            parts.append("Active proposal state:\n" + "\n".join(proposal_lines))
        return "\n\n".join(parts)

    def _classify_with_llm_or_heuristics(self, state: StudyChatFlowState) -> IntentClassification:
        """
        Classify intent using LLM (preferred) or fall back to safe deep_dive.
        
        If GWDG_API_KEY is not set or LLM classification fails, we always use deep_dive
        (the expensive but robust multi-source route) to ensure quality responses.
        Heuristic classification is deprecated in favor of this approach.
        """
        if not self._runtime.use_llm_classifier:
            return classify_intent_heuristically(state.query, state.thread)

        # Check if LLM API is available
        if not os.getenv("GWDG_API_KEY"):
            if self._runtime.verbose:
                print("⚠️  WARNING: GWDG_API_KEY not set. Using fallback deep_dive route for robust handling.")
            return IntentClassification(
                route="deep_dive",
                complexity="deep",
                required_sources=["grade_manager", "moses", "isis"],
                rationale="LLM classifier unavailable (GWDG_API_KEY not set); using safe deep_dive fallback.",
            )
        
        try:
            llm = get_default_llm(
                model=self._runtime.manager_model or self._runtime.model,
                temperature=0.0,
                top_p=self._runtime.top_p,
            )
            result = llm.call(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify a TU Berlin study assistant chat turn into the most efficient route. "
                            "Available routes:\n"
                            "- simple_grade_manager: Single questions about current grades, credits, GPA, degree requirements\n"
                            "- simple_moses: Single questions about module catalog, prerequisites, workload\n"
                            "- simple_isis: Specific questions about deadlines/assignments in a known ISIS course\n"
                            "- recommendation: Semester/course planning, follow-ups to active proposals, course selection\n"
                            "- deep_dive: Broad/multi-source queries, unclear intent, or when combining multiple sources makes sense\n"
                            "Prefer simple routes for single-source efficiency. Use deep_dive for ambiguous or complex requests."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Current message:\n{state.query}\n\n"
                            f"Conversation history:\n{state.conversation_context}\n\n"
                            f"Student profile:\n{state.student_context or '(No student context)'}"
                        ),
                    },
                ],
                response_model=IntentClassification,
            )
            if isinstance(result, IntentClassification):
                return result
        except Exception as e:
            if self._runtime.verbose:
                print(f"⚠️  WARNING: LLM classifier failed ({type(e).__name__}). Using fallback deep_dive route.")
        
        # Fallback to deep_dive on any classification failure
        return IntentClassification(
            route="deep_dive",
            complexity="deep",
            required_sources=["grade_manager", "moses", "isis"],
            rationale="LLM classifier unavailable or failed; using safe deep_dive fallback for robustness.",
        )

    def _adopt_recorded_or_existing_proposals(self) -> None:
        proposals = current_course_proposals()
        if proposals:
            self.state.proposed_actions = proposals
            return
        self._carry_forward_existing_proposals()

    def _carry_forward_existing_proposals(self) -> None:
        thread = self.state.thread
        self.state.proposed_actions = list(thread.active_proposals if thread else [])

    def _execute_action_decisions(self) -> list[ProposedAction]:
        thread = self.state.thread or load_chat_thread(self.state.profile_slug, thread_id=self.state.thread_id)
        action_by_id = {
            action.action_id: action
            for proposal in thread.active_proposals
            for action in proposal.actions
        }
        executed: list[ProposedAction] = []
        for decision in self.state.approved_actions:
            action = action_by_id.get(decision.action_id)
            if action is None:
                executed.append(
                    ProposedAction(
                        action_id=decision.action_id,
                        kind="grade_manager_add",
                        course_title="Unknown action",
                        status="failed",
                        result="No active proposal action matched this decision.",
                    )
                )
                continue
            updated = action.model_copy(deep=True)
            if not decision.approved:
                updated.status = "declined"
                updated.result = decision.feedback or "Declined by user."
            elif updated.kind == "grade_manager_add":
                updated = _execute_grade_manager_action(updated)
            elif updated.kind == "isis_enroll":
                updated = _execute_isis_action(updated)
            action.status = updated.status
            action.result = updated.result
            executed.append(updated)

        thread.proposal_decisions.extend(self.state.approved_actions)
        thread.active_proposals = [
            proposal
            for proposal in thread.active_proposals
            if any(action.status == "proposed" for action in proposal.actions)
        ]
        save_chat_thread(thread)
        self.state.thread = thread
        return executed

    def _remaining_proposals(self) -> list[CourseProposal]:
        thread = self.state.thread
        return list(thread.active_proposals if thread else [])


def classify_intent_heuristically(query: str, thread: ChatThreadState | None = None) -> IntentClassification:
    text = query.casefold()
    has_active_proposal = bool(thread and thread.active_proposals)
    if any(token in text for token in ["approved actions", "confirmation_token", "submit decisions"]):
        return IntentClassification(
            route="execute_confirmed_actions",
            complexity="scoped",
            required_sources=["grade_manager", "isis"],
            write_intent=True,
            rationale="The message appears to contain approval decisions.",
        )
    if has_active_proposal and any(token in text for token in ["replace", "exchange", "more", "weniger", "mehr", "ersetze", "tausch"]):
        return IntentClassification(
            route="recommendation",
            complexity="scoped",
            required_sources=["grade_manager", "moses"],
            rationale="Follow-up modifies an active course proposal.",
        )
    if any(token in text for token in ["next semester", "nächstes semester", "naechstes semester", "recommend", "suggest", "plan", "belegen", "course would"]):
        return IntentClassification(
            route="recommendation",
            language=("de" if _looks_german(text) else "en"),
            complexity="scoped",
            required_sources=["grade_manager", "moses"],
            rationale="Semester/course planning request.",
        )
    if any(token in text for token in ["deadline", "due", "forum", "announcement", "isis", "moodle", "assignment", "next week", "nächste woche"]):
        has_known_course_selector = any(token in text for token in ["course id", "isis id", "course/view.php", "course_id", "id "])
        needs_study_scope = any(
            token in text
            for token in [
                "deadline",
                "due",
                "assignment",
                "next week",
                "nächste woche",
                "current",
                "currently",
                "taking",
                "my courses",
                "meine kurse",
                "to do",
            ]
        )
        route = "simple_isis" if has_known_course_selector and not needs_study_scope else "deep_dive"
        return IntentClassification(
            route=route,
            language=("de" if _looks_german(text) else "en"),
            complexity=("deep" if route == "deep_dive" else "simple"),
            required_sources=(["grade_manager", "isis"] if route == "deep_dive" else ["isis"]),
            rationale="ISIS operational information request.",
        )
    if any(token in text for token in ["missing", "credits", "gpa", "grade", "degree", "regulation", "requirements", "lp", "ects", "abschluss"]):
        return IntentClassification(
            route="simple_grade_manager",
            language=("de" if _looks_german(text) else "en"),
            complexity="simple",
            required_sources=["grade_manager"],
            rationale="Study progress or degree requirement question.",
        )
    if any(token in text for token in ["moses", "module", "catalog", "prerequisite", "exam", "workload", "catalogue", "modul"]):
        return IntentClassification(
            route="simple_moses",
            language=("de" if _looks_german(text) else "en"),
            complexity="simple",
            required_sources=["moses"],
            rationale="MOSES module catalog question.",
        )
    return IntentClassification(
        route="deep_dive",
        language=("de" if _looks_german(text) else "en"),
        complexity="deep",
        required_sources=["grade_manager", "moses", "isis"],
        rationale="Fallback for broad or ambiguous request.",
    )


def _execute_grade_manager_action(action: ProposedAction) -> ProposedAction:
    payload = dict(action.grade_manager_payload or {})
    module_query = str(payload.get("module_query") or action.course_title).strip()
    term = str(payload.get("term") or "").strip()
    if not term:
        return action.model_copy(update={"status": "needs_clarification", "result": "No planned term was provided."})
    result = add_module_to_study_plan(
        module_query=module_query,
        term=term,
        version=payload.get("version"),
        program_key=payload.get("program_key"),
        area=payload.get("area"),
        confirmation_token=STUDY_PLAN_CONFIRMATION_TOKEN,
    )
    verification = check_module_against_study_plan(
        module_query=module_query,
        version=payload.get("version"),
        program_key=payload.get("program_key"),
        term=term,
    )
    status = "executed" if result.startswith("Added `") or result.startswith("Study-plan write refused: this MOSES module is already present") else "failed"
    return action.model_copy(update={"status": status, "result": f"{result}\n\nVerification:\n{verification}"})


def _execute_isis_action(action: ProposedAction) -> ProposedAction:
    payload = {key: value for key, value in dict(action.isis_payload or {}).items() if value not in (None, "")}
    if not any(payload.get(key) for key in ("course_id", "course_query", "course_url")):
        return action.model_copy(update={"status": "needs_clarification", "result": "No ISIS course id, query, or URL was provided."})
    result = PermanentlyEnrollInIsisCourseTool()._run(
        **payload,
        confirmation_token=CONFIRMATION_TOKEN,
    )
    status = "executed" if result.startswith("Permanently enrolled") or result.startswith("Already enrolled") else "failed"
    return action.model_copy(update={"status": status, "result": result})


def _format_execution_answer(actions: list[ProposedAction], *, language: str) -> str:
    if language == "de":
        lines = ["## Bestätigte Aktionen", ""]
        if not actions:
            return "Ich habe keine bestätigten Aktionen erhalten."
        for action in actions:
            lines.append(f"- **{action.course_title}** (`{action.kind}`): {action.status}")
            if action.result:
                lines.append(f"  {action.result.splitlines()[0]}")
        return "\n".join(lines)
    lines = ["## Confirmed Actions", ""]
    if not actions:
        return "I did not receive any confirmed actions to execute."
    for action in actions:
        lines.append(f"- **{action.course_title}** (`{action.kind}`): {action.status}")
        if action.result:
            lines.append(f"  {action.result.splitlines()[0]}")
    return "\n".join(lines)


def _conversation_context(thread: ChatThreadState) -> str:
    parts: list[str] = []
    if thread.rolling_summary:
        parts.append(f"Summary:\n{thread.rolling_summary}")
    recent = thread.recent_messages_text(limit=8)
    if recent:
        parts.append(f"Recent messages:\n{recent}")
    if thread.active_proposals:
        parts.append(
            "Active proposals:\n"
            + "\n".join(
                f"- {proposal.title}: {proposal.summary} ({len(proposal.proposed_actions)} open action(s))"
                for proposal in thread.active_proposals
            )
        )
    return "\n\n".join(parts)


def _summarize_thread(
    thread: ChatThreadState | None,
    query: str,
    answer: str,
    proposals: list[CourseProposal],
) -> str:
    previous = thread.rolling_summary if thread else ""
    bits = []
    if previous:
        bits.append(previous)
    bits.append(f"Latest user request: {query[:300]}")
    bits.append(f"Latest assistant answer: {answer[:500]}")
    if proposals:
        bits.append("Active course proposals: " + ", ".join(proposal.title for proposal in proposals))
    return "\n".join(bits)[-3000:]


def _looks_german(text: str) -> bool:
    return any(token in text for token in [" der ", " die ", " das ", "ich ", "was ", "welche", "nächst", "hinzufügen", "abschluss"])
