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
    UserDecisionInterpretation,
)
from crew.chat_persistence import append_turn, load_chat_thread, reset_chat_thread, save_chat_thread
from crew.config.llm import get_default_llm
from crew.runtime import ensure_crewai_storage_writable
from crew.semester_context import semester_reference_context
from crew.tools.grademanager_tools import (
    STUDY_PLAN_CONFIRMATION_TOKEN,
    add_module_to_study_plan,
    check_module_against_study_plan,
    remove_module_from_study_plan,
    update_module_in_study_plan,
)
from crew.tools.isis_tools import CONFIRMATION_TOKEN, PermanentlyEnrollInIsisCourseTool
from crew.tools.proposal_tools import current_course_proposals
from crew.write_permissions import allow_confirmed_writes


FlowRunner = Callable[["StudyChatFlow"], str]
Classifier = Callable[[StudyChatFlowState], IntentClassification]
DecisionInterpreter = Callable[[StudyChatFlowState], UserDecisionInterpretation]
ACTIVE_PROPOSAL_STATUSES = {"proposed", "approved", "needs_clarification"}


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
    use_llm_decision_interpreter: bool = True


class StudyChatFlow(Flow[StudyChatFlowState]):
    """Persistent multi-turn Flow for the Streamlit Study Chat."""

    _runtime: StudyChatFlowRuntime = PrivateAttr(default_factory=StudyChatFlowRuntime)
    _classifier: Classifier | None = PrivateAttr(default=None)
    _decision_interpreter: DecisionInterpreter | None = PrivateAttr(default=None)
    _runner_overrides: dict[str, FlowRunner] = PrivateAttr(default_factory=dict)
    _on_trace_event: Callable[[dict[str, Any]], None] | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        runtime: StudyChatFlowRuntime | None = None,
        classifier: Classifier | None = None,
        decision_interpreter: DecisionInterpreter | None = None,
        runner_overrides: dict[str, FlowRunner] | None = None,
        on_trace_event: Callable[[dict[str, Any]], None] | None = None,
        **data: Any,
    ) -> None:
        ensure_crewai_storage_writable()
        super().__init__(**data)
        self._runtime = runtime or StudyChatFlowRuntime()
        self._classifier = classifier
        self._decision_interpreter = decision_interpreter
        self._runner_overrides = dict(runner_overrides or {})
        self._on_trace_event = on_trace_event

    @start()
    def ingest_turn(self) -> None:
        if self.state.reset_thread:
            thread = reset_chat_thread(self.state.profile_slug, thread_id=self.state.thread_id)
        else:
            thread = load_chat_thread(self.state.profile_slug, thread_id=self.state.thread_id)
        
        if self.state.ui_decisions:
            decision_by_action = {d.action_id: d for d in self.state.ui_decisions}
            updated_any = False
            for proposal in thread.active_proposals:
                for action in proposal.actions:
                    if action.action_id in decision_by_action:
                        decision = decision_by_action[action.action_id]
                        new_status = "approved" if decision.approved else "declined"
                        if action.status != new_status:
                            action.status = new_status
                            updated_any = True
            if updated_any:
                save_chat_thread(thread)

        self.state.thread = thread
        if thread.isis_context:
            from crew.state import _merge_isis_contexts
            supplied = json.loads(self.state.isis_context_json or "{}")
            merged = _merge_isis_contexts(inferred=thread.isis_context, supplied=supplied)
            if merged:
                self.state.isis_context_json = json.dumps(merged.model_dump(mode="json"), ensure_ascii=False)
                thread.isis_context = merged
                save_chat_thread(thread)
        self.state.conversation_context = _conversation_context(thread)

    @router(ingest_turn)
    def classify_intent(self) -> str:
        if self.state.approved_actions:
            intent = IntentClassification(
                route="recommendation",
                complexity="scoped",
                required_sources=["degree_regulations", "grade_manager", "moses", "isis"],
                write_intent=True,
                rationale="User submitted explicit UI action decisions.",
            )
            self.state.intent = intent
            self.state.route = "recommendation"
            if self._on_trace_event:
                self._on_trace_event(
                    {
                        "event": "intent_classified",
                        "intent": intent.model_dump(mode="json"),
                        "status": "ok",
                    }
                )
            return "recommendation"

        decision = self._interpret_active_proposal_decision()
        if decision is not None:
            self.state.decision_interpretation = decision
            if self._on_trace_event:
                self._on_trace_event({
                    "event": "proposal_decision_interpreted",
                    "decision": decision.model_dump(mode="json"),
                    "status": "ok",
                })

            if decision.discard_active_proposals or decision.intent == "discard_active_proposals":
                self._clear_active_proposals()
                if decision.intent == "discard_active_proposals" and not _has_separate_user_question(self.state.query):
                    intent = IntentClassification(
                        route="discard_active_proposals",
                        complexity="simple",
                        required_sources=[],
                        write_intent=False,
                        rationale=decision.rationale or "User discarded the active course recommendation state.",
                    )
                    self.state.intent = intent
                    self.state.route = intent.route
                    return "discard_active_proposals"

            if decision.needs_user_clarification or decision.intent == "unclear":
                intent = IntentClassification(
                    route="proposal_clarification",
                    complexity="simple",
                    required_sources=[],
                    write_intent=False,
                    rationale=decision.rationale or "The user's decision about active recommendations is unclear.",
                )
                self.state.intent = intent
                self.state.route = intent.route
                return "proposal_clarification"

            if decision.intent in {"apply_selected", "apply_partial_and_revise"}:
                self.state.approved_actions = self._action_decisions_from_interpretation(decision)
                if self.state.approved_actions:
                    intent = IntentClassification(
                        route="recommendation",
                        complexity="scoped",
                        required_sources=["degree_regulations", "grade_manager", "moses", "isis"],
                        write_intent=True,
                        rationale=decision.rationale or "User decision interpreter selected approved UI actions for execution.",
                    )
                    self.state.intent = intent
                    self.state.route = "recommendation"
                    return "recommendation"

            if decision.intent == "revise_only":
                intent = IntentClassification(
                    route="recommendation",
                    complexity="scoped",
                    required_sources=["degree_regulations", "grade_manager", "moses"],
                    write_intent=False,
                    rationale=decision.rationale or "User asked to revise active course recommendations.",
                )
                self.state.intent = intent
                self.state.route = "recommendation"
                return "recommendation"

        classifier = self._classifier or self._classify_with_llm_or_heuristics
        intent = classifier(self.state)
        intent = _normalize_commitment_route(intent, self.state.query)
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

    @listen("simple_degree_regulations")
    def run_simple_degree_regulations(self) -> None:
        self.state.answer_markdown = self._run_route("simple_degree_regulations")
        self._carry_forward_existing_proposals()

    @listen("recommendation")
    def run_recommendation_route(self) -> None:
        self.state.answer_markdown = self._run_route("recommendation")
        self._adopt_recorded_or_existing_proposals()

    @listen("deep_dive")
    def run_deep_dive_route(self) -> None:
        self.state.answer_markdown = self._run_route("deep_dive")
        self._adopt_recorded_or_existing_proposals()

    @listen("discard_active_proposals")
    def run_discard_active_proposals(self) -> None:
        self._clear_active_proposals()
        self.state.proposed_actions = []
        self.state.answer_markdown = _format_discard_answer(self.state.decision_interpretation)

    @listen("proposal_clarification")
    def run_proposal_clarification(self) -> None:
        self.state.proposed_actions = self._remaining_proposals()
        self.state.answer_markdown = _format_proposal_clarification_answer(self.state.decision_interpretation)

    @listen(
        or_(
            run_simple_grade_manager,
            run_simple_moses,
            run_simple_isis,
            run_simple_degree_regulations,
            run_recommendation_route,
            run_deep_dive_route,
            run_discard_active_proposals,
            run_proposal_clarification,
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
        from crew.state import _MOSES_STATE_ARTIFACTS, _moses_state_from_artifacts, _merge_isis_contexts
        artifacts = _MOSES_STATE_ARTIFACTS.get() or []
        if artifacts:
            _, new_isis_context = _moses_state_from_artifacts(answer_markdown="", artifacts=artifacts)
            merged = _merge_isis_contexts(
                inferred=new_isis_context,
                supplied=self.state.thread.isis_context.model_dump(mode="json") if self.state.thread.isis_context else None
            )
            if merged:
                self.state.thread.isis_context = merged
                self.state.isis_context_json = json.dumps(merged.model_dump(mode="json"), ensure_ascii=False)
                save_chat_thread(self.state.thread)
        return self.state

    def _run_route(self, route: str) -> str:
        approved_execution_actions = self._approved_actions_for_execution()
        if approved_execution_actions:
            self.state.executed_actions = self._execute_approved_actions(approved_execution_actions)
            self.state.proposed_actions = self._remaining_proposals()
            if self._execution_only_turn():
                return _format_execution_answer(
                    self.state.executed_actions,
                    language=(self.state.intent.language if self.state.intent else "en"),
                )

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
        if route == "simple_degree_regulations":
            from crew.degree_regulations_crew import DegreeRegulationsCrew

            result = DegreeRegulationsCrew(**self._crew_kwargs()).crew().kickoff(
                inputs={
                    "query": self._contextual_query(),
                    "student_context": self.state.student_context or "No student context supplied.",
                }
            )
            return str(getattr(result, "raw", result))

        from crew.multi_agent_crew import MultiAgentStudyAssistantCrew

        student_context = self.state.student_context or "No student context supplied."
        result = MultiAgentStudyAssistantCrew(
            **self._crew_kwargs(),
            manager_model=self._runtime.manager_model,
            allow_temp_enrollment=self._runtime.allow_temp_enrollment,
            planning_enabled=(self._runtime.planning_enabled or route in {"recommendation", "deep_dive"}),
            planning_llm_model=self._runtime.planning_llm_model,
        ).crew().kickoff(
            inputs={
                "query": self._contextual_query(),
                "student_context": student_context,
                "isis_context": self.state.isis_context_json or "{}",
            }
        )
        return str(getattr(result, "raw", result))

    def _approved_actions_for_execution(self) -> list[ProposedAction]:
        thread = self.state.thread
        if not thread or not self.state.approved_actions:
            return []
        action_by_id = {
            action.action_id: action
            for proposal in thread.active_proposals
            for action in proposal.actions
        }
        execution_actions: list[ProposedAction] = []
        for decision in self.state.approved_actions:
            if not decision.approved:
                continue
            action = action_by_id.get(decision.action_id)
            if action is not None:
                execution_actions.append(action.model_copy(deep=True))
        return execution_actions

    def _execute_approved_actions(self, actions: list[ProposedAction]) -> list[ProposedAction]:
        executed: list[ProposedAction] = []
        with allow_confirmed_writes():
            for action in actions:
                if action.kind == "grade_manager_add":
                    executed.append(_execute_grade_manager_action(action))
                elif action.kind == "grade_manager_update":
                    executed.append(_execute_grade_manager_update_action(action))
                elif action.kind == "grade_manager_remove":
                    executed.append(_execute_grade_manager_remove_action(action))
                elif action.kind in {"isis_enroll", "isis_resolve"}:
                    executed.append(_execute_isis_action(action))
                else:
                    executed.append(
                        action.model_copy(
                            update={
                                "status": "failed",
                                "result": f"Unknown action kind: {action.kind}",
                            }
                        )
                    )
        self._record_executed_actions(executed)
        return executed

    def _record_executed_actions(self, executed: list[ProposedAction]) -> None:
        thread = self.state.thread or load_chat_thread(self.state.profile_slug, thread_id=self.state.thread_id)
        action_by_id = {
            action.action_id: action
            for proposal in thread.active_proposals
            for action in proposal.actions
        }
        executed_by_identity = {
            (_course_identity_for_action(action), action.kind): action
            for action in executed
        }
        for executed_action in executed:
            action = action_by_id.get(executed_action.action_id)
            if action is None:
                continue
            action.status = executed_action.status
            action.result = executed_action.result

        for proposal in thread.active_proposals:
            for action in proposal.actions:
                if action.status not in ACTIVE_PROPOSAL_STATUSES:
                    continue
                identity = _course_identity_for_action(action)
                if action.kind == "isis_resolve" and (identity, "isis_enroll") in executed_by_identity:
                    action.status = "executed"
                    action.result = "Superseded by a resolved ISIS enrollment action for the same course."

        decisions_to_record = self.state.ui_decisions or self.state.approved_actions
        thread.proposal_decisions.extend(decisions_to_record)
        thread.active_proposals = _prune_proposals(thread.active_proposals)
        save_chat_thread(thread)
        self.state.thread = thread

    def _execution_only_turn(self) -> bool:
        decision = self.state.decision_interpretation
        if decision is not None:
            return decision.intent == "apply_selected"
        return bool(self.state.approved_actions)

    def _crew_kwargs(self) -> dict[str, Any]:
        return {
            "model": self._runtime.model,
            "temperature": self._runtime.temperature,
            "top_p": self._runtime.top_p,
            "verbose": self._runtime.verbose,
            "cache": self._runtime.cache,
        }

    def _contextual_query(self) -> str:
        parts = [f"Current user message:\n{self.state.query}", semester_reference_context()]
        if self.state.decision_interpretation:
            parts.append(
                "Interpreted course-recommendation decision:\n"
                + json.dumps(self.state.decision_interpretation.model_dump(mode="json"), ensure_ascii=False, indent=2)
            )
        if self.state.executed_actions:
            executed = [
                {
                    "action_id": action.action_id,
                    "kind": action.kind,
                    "course_title": action.course_title,
                    "status": action.status,
                    "result": action.result,
                }
                for action in self.state.executed_actions
            ]
            parts.append("Actions already executed in this turn:\n" + json.dumps(executed, ensure_ascii=False, indent=2))
        if self.state.conversation_context:
            parts.append(f"Conversation context:\n{self.state.conversation_context}")
        if self.state.thread and self.state.thread.active_proposals:
            proposal_lines = []
            for proposal in self.state.thread.active_proposals:
                action_lines = []
                for action in proposal.actions:
                    action_lines.append(f"  * [{action.status.upper()}] {action.kind}: {action.course_title}")
                actions_str = "\n".join(action_lines)
                proposal_lines.append(f"- {proposal.title}: {proposal.summary}\n{actions_str}")
            parts.append("Active proposal state:\n" + "\n".join(proposal_lines))
        return "\n\n".join(parts)

    def _interpret_active_proposal_decision(self) -> UserDecisionInterpretation | None:
        thread = self.state.thread
        if not thread or not thread.active_proposals:
            return None
        interpreter = self._decision_interpreter or self._interpret_decision_with_llm_or_fallback
        return _sanitize_decision_interpretation(interpreter(self.state), self.state)

    def _interpret_decision_with_llm_or_fallback(self, state: StudyChatFlowState) -> UserDecisionInterpretation:
        if self._runtime.use_llm_decision_interpreter and os.getenv("GWDG_API_KEY"):
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
                                "Interpret one chat turn with active course recommendation cards. "
                                "The UI card state is the authorization gate: only actions explicitly accepted in the UI may be executed. "
                                "Rejected cards must not be executed. Unsure cards must not be executed. "
                                "If an accepted card has an action disabled by toggle, treat that action as intentionally disabled, not as a request for a replacement. "
                                "Infer whether the user wants to apply accepted actions, apply only some and revise others, revise without applying, ask a question, discard the recommendation state, or needs clarification. "
                                "If the user changes to an unrelated topic, set discard_active_proposals=true so the old course cards disappear. "
                                "Do not invent action IDs; choose only from the provided active actions."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Current message:\n{state.query}\n\n"
                                f"Conversation context:\n{state.conversation_context}\n\n"
                                f"Active proposals:\n{_proposal_state_json(state.thread.active_proposals if state.thread else [])}\n\n"
                                f"UI card decisions:\n{_action_decisions_json(state.ui_decisions)}"
                            ),
                        },
                    ],
                    response_model=UserDecisionInterpretation,
                )
                if isinstance(result, UserDecisionInterpretation):
                    return _sanitize_decision_interpretation(result, state)
            except Exception as exc:
                if self._runtime.verbose:
                    print(f"WARNING: LLM decision interpreter failed ({type(exc).__name__}). Using fallback interpreter.")
        return _fallback_decision_interpretation(state)

    def _action_decisions_from_interpretation(self, decision: UserDecisionInterpretation) -> list[ActionDecision]:
        thread = self.state.thread
        if not thread:
            return []
        active_ids = {
            action.action_id
            for proposal in thread.active_proposals
            for action in proposal.actions
            if action.status == "approved"
        }
        requested = set(decision.approved_action_ids)
        if not requested:
            requested = active_ids
        return [
            ActionDecision(action_id=action_id, approved=True, feedback=decision.rationale)
            for action_id in active_ids
            if action_id in requested
        ]

    def _clear_active_proposals(self) -> None:
        thread = self.state.thread or load_chat_thread(self.state.profile_slug, thread_id=self.state.thread_id)
        if thread.active_proposals:
            thread.active_proposals = []
            save_chat_thread(thread)
        self.state.thread = thread
        self.state.proposed_actions = []

    def _classify_with_llm_or_heuristics(self, state: StudyChatFlowState) -> IntentClassification:
        """
        Classify intent using LLM (preferred) or fall back to safe deep_dive.
        
        If GWDG_API_KEY is not set or LLM classification fails, we always use deep_dive
        (the expensive but robust multi-source route) to ensure quality responses.
        """
        if not self._runtime.use_llm_classifier:
            if self._runtime.verbose:
                print("⚠️  WARNING: LLM classifier disabled. Using safe deep_dive route.")
            return IntentClassification(
                route="deep_dive",
                complexity="deep",
                required_sources=["grade_manager", "moses", "isis"],
                rationale="WARNING: LLM classifier disabled; using safe deep_dive fallback.",
            )

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
                            "- simple_isis: Read-only questions about deadlines/assignments/info in a known ISIS course\n"
                            "- simple_degree_regulations: Single questions about AllgStuPO, StuPO, degree rules, exam regulations, free-choice/elective rules, or Regelstudienplan from local PDFs\n"
                            "- recommendation: Semester/course planning, follow-ups to active proposals, course selection, and any request to enroll/register/add/save/write course actions for confirmation\n"
                            "- deep_dive: Broad/multi-source queries, unclear intent, or when combining multiple sources makes sense\n"
                            "Prefer simple routes for read-only single-source efficiency. Never use a simple_* route for write_intent=true."
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
        new_proposals = current_course_proposals()
        existing_proposals = list(self.state.thread.active_proposals if self.state.thread else [])
        if new_proposals:
            self.state.proposed_actions = _merge_proposals(existing_proposals, new_proposals)
            return
        self._carry_forward_existing_proposals()

    def _carry_forward_existing_proposals(self) -> None:
        thread = self.state.thread
        self.state.proposed_actions = _prune_proposals(list(thread.active_proposals if thread else []))

    def _verify_and_update_executed_actions(self) -> list[ProposedAction]:
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
                status, result = _verify_grade_manager_addition(updated)
                updated.status = status
                updated.result = result
            elif updated.kind == "grade_manager_update":
                target_term = str((updated.grade_manager_payload or {}).get("target_term") or "").strip()
                if target_term:
                    status, result = _verify_grade_manager_addition(
                        updated.model_copy(
                            update={
                                "kind": "grade_manager_add",
                                "grade_manager_payload": {
                                    **(updated.grade_manager_payload or {}),
                                    "term": target_term,
                                },
                            }
                        )
                    )
                    updated.status = status
                    updated.result = result
                else:
                    updated.status = "failed"
                    updated.result = "Verification failed: no target_term was provided."
            elif updated.kind == "grade_manager_remove":
                updated.status = "executed" if action.status == "executed" else action.status
                updated.result = action.result or "Remove action processed."
            elif updated.kind in {"isis_enroll", "isis_resolve"}:
                status, result = _verify_isis_commitment(updated)
                updated.status = status
                updated.result = result
            else:
                updated.status = "failed"
                updated.result = f"Unknown action kind: {updated.kind}"
            action.status = updated.status
            action.result = updated.result
            executed.append(updated)

        thread.proposal_decisions.extend(self.state.approved_actions)
        thread.active_proposals = _prune_proposals(thread.active_proposals)
        save_chat_thread(thread)
        self.state.thread = thread
        return executed

    def _remaining_proposals(self) -> list[CourseProposal]:
        thread = self.state.thread
        return _prune_proposals(list(thread.active_proposals if thread else []))



def _verify_grade_manager_addition(action: ProposedAction) -> tuple[str, str]:
    payload = dict(action.grade_manager_payload or {})
    module_query = str(payload.get("module_query") or action.course_title).strip()
    term = str(payload.get("term") or "").strip()
    
    from crew.tools.grademanager_tools import _load_primary_profile_modules, _canonical_term_or_error, _resolve_moses_module
    from core.providers.tu_berlin import moses as moses_provider
    try:
        profile, modules = _load_primary_profile_modules()
        term_label = _canonical_term_or_error(term)
        resolved = _resolve_moses_module(module_query, version=payload.get("version"), term=term_label)
        data = resolved.data
        
        existing = moses_provider.find_existing_module_by_moses_identity_any_program(
            modules,
            number=data.number,
            version=data.version,
        )
        if existing and existing.state.value == "Planned" and existing.term == term_label:
            return "executed", f"Verified: Module '{existing.name}' is planned for {term_label} in study plan."
        else:
            return "failed", f"Failed verification: Module '{data.name}' not found in study plan for {term_label}."
    except Exception as exc:
        return "failed", f"Verification failed with error: {exc}"


def _verify_isis_commitment(action: ProposedAction) -> tuple[str, str]:
    payload = {key: value for key, value in dict(action.isis_payload or {}).items() if value not in (None, "")}
    course_id = str(payload.get("course_id") or "").strip()
    moses_number = str(payload.get("moses_module_number") or "").strip()

    from crew.isis_client import get_default_isis_client
    from crew.isis_models import IsisCourseSelector
    from crew.isis_resolver import IsisCourseResolver

    try:
        client = get_default_isis_client()
        if course_id and moses_number and course_id == moses_number:
            return (
                "needs_clarification",
                (
                    f"ISIS verification blocked: `{course_id}` is also the MOSES module number. "
                    "A verified ISIS course ID or course/view.php URL is required."
                ),
            )
        enrolled_ids = {course.id for course in client.enrolled_course_refs()}
        if course_id:
            course_id_int = int(course_id)
            if course_id_int in enrolled_ids:
                course_ref = client.course_ref_by_id(course_id_int)
                title = course_ref.title if course_ref else action.course_title
                return "executed", f"Verified: Enrolled in ISIS course '{title}' ({course_id_int})."
            return "failed", f"Failed verification: ISIS course ID {course_id_int} not found in enrolled courses."

        selector = IsisCourseSelector(
            course_url=payload.get("course_url"),
            course_query=payload.get("course_query") or action.course_title,
            term_hint=payload.get("term_hint"),
            expected_title=payload.get("expected_title") or action.course_title,
        )
        resolved = IsisCourseResolver(client).resolve(selector)
        if not resolved.is_resolved or resolved.course is None:
            reason = resolved.reason or "ISIS course could not be resolved unambiguously."
            candidates = _format_isis_candidates(resolved.candidates)
            if candidates:
                reason = f"{reason}\n\nCandidate ISIS courses:\n{candidates}"
                return "needs_clarification", reason
            if resolved.status == "not_found":
                return "failed", reason
            return "needs_clarification", reason
        if resolved.course.id in enrolled_ids:
            if action.isis_payload is not None:
                action.isis_payload["course_id"] = resolved.course.id
                action.isis_payload["course_url"] = resolved.course.url
                action.isis_payload["isis_resolution_status"] = "resolved"
            return "executed", f"Verified: Enrolled in ISIS course '{resolved.course.title}' ({resolved.course.id})."
        return (
            "failed",
            (
                f"Resolved ISIS course '{resolved.course.title}' ({resolved.course.id}), "
                "but enrollment was not found after Course Commitment Specialist execution."
            ),
        )
    except Exception as exc:
        return "failed", f"Verification failed with error: {exc}"


def _verify_isis_enrollment(action: ProposedAction) -> tuple[str, str]:
    return _verify_isis_commitment(action)


def _format_isis_candidates(candidates: list[Any]) -> str:
    lines = []
    for candidate in candidates[:8]:
        title = getattr(candidate, "title", None) or getattr(candidate, "fullname", None) or "Unknown ISIS course"
        term = getattr(candidate, "term_hint", None) or "term unknown"
        course_id = getattr(candidate, "id", None)
        lines.append(f"- `{course_id}` {title} ({term})")
    return "\n".join(lines)


def _prune_proposals(proposals: list[CourseProposal]) -> list[CourseProposal]:
    return _merge_proposals(proposals, [])


def _merge_proposals(existing_proposals: list[CourseProposal], new_proposals: list[CourseProposal]) -> list[CourseProposal]:
    action_by_key: dict[tuple[str, str], tuple[CourseProposal, ProposedAction]] = {}

    for proposal in existing_proposals:
        for action in proposal.actions:
            if action.status not in ACTIVE_PROPOSAL_STATUSES:
                continue
            key = (_course_identity_for_action(action), _action_family(action))
            existing = action_by_key.get(key)
            if existing is None or _action_merge_priority(action) > _action_merge_priority(existing[1]):
                action_by_key[key] = (proposal, action)

    for proposal in new_proposals:
        for action in proposal.actions:
            if action.status not in ACTIVE_PROPOSAL_STATUSES:
                continue
            key = (_course_identity_for_action(action), _action_family(action))
            action_by_key[key] = (proposal, action)

    if not action_by_key:
        return []
    template = new_proposals[-1] if new_proposals else next(iter(action_by_key.values()))[0]
    actions = [action for _, action in action_by_key.values()]
    return [template.model_copy(update={"actions": actions}, deep=True)]


def _action_family(action: ProposedAction) -> str:
    if action.kind in {"isis_resolve", "isis_enroll"}:
        return "isis"
    if action.kind in {"grade_manager_add", "grade_manager_update", "grade_manager_remove"}:
        return "grade_manager"
    return action.kind


def _action_merge_priority(action: ProposedAction) -> int:
    priority = {
        "grade_manager_add": 10,
        "grade_manager_update": 20,
        "grade_manager_remove": 30,
        "isis_resolve": 10,
        "isis_enroll": 20,
    }.get(action.kind, 0)
    if action.status == "approved":
        priority += 1
    return priority


def _course_identity_for_action(action: ProposedAction) -> str:
    grade_payload = action.grade_manager_payload or {}
    isis_payload = action.isis_payload or {}
    moses_identity = (
        grade_payload.get("moses_module_number")
        or isis_payload.get("moses_module_number")
        or _numeric_identity(grade_payload.get("module_query"))
    )
    if moses_identity:
        return f"moses:{_identity_text(moses_identity)}"
    title = _identity_text(action.course_title)
    if title:
        return f"title:{title}"
    course_id = isis_payload.get("course_id")
    if course_id:
        return f"isis-id:{_identity_text(course_id)}"
    course_url = isis_payload.get("course_url")
    if course_url:
        return f"isis-url:{_identity_text(course_url)}"
    course_query = isis_payload.get("course_query")
    if course_query:
        return f"isis-query:{_identity_text(course_query)}"
    return f"action:{action.action_id}"


def _identity_text(value: object | None) -> str:
    return " ".join(str(value or "").casefold().strip().split())


def _numeric_identity(value: object | None) -> str | None:
    text = str(value or "").strip()
    return text if text.isdigit() else None


def _proposal_state_json(proposals: list[CourseProposal]) -> str:
    compact = []
    for proposal in proposals:
        compact.append(
            {
                "proposal_id": proposal.proposal_id,
                "title": proposal.title,
                "summary": proposal.summary,
                "actions": [
                    {
                        "action_id": action.action_id,
                        "kind": action.kind,
                        "course_title": action.course_title,
                        "status": action.status,
                        "grade_manager_payload": action.grade_manager_payload,
                        "isis_payload": action.isis_payload,
                    }
                    for action in proposal.actions
                ],
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _action_decisions_json(decisions: list[ActionDecision]) -> str:
    return json.dumps([decision.model_dump(mode="json") for decision in decisions], ensure_ascii=False, indent=2)


def _sanitize_decision_interpretation(
    decision: UserDecisionInterpretation,
    state: StudyChatFlowState,
) -> UserDecisionInterpretation:
    active_ids = {
        action.action_id
        for proposal in (state.thread.active_proposals if state.thread else [])
        for action in proposal.actions
    }
    approved_ui_ids = {decision.action_id for decision in state.ui_decisions if decision.approved}
    rejected_ui_ids = {decision.action_id for decision in state.ui_decisions if not decision.approved}
    approved_ids = [action_id for action_id in decision.approved_action_ids if action_id in active_ids and action_id in approved_ui_ids]
    rejected_ids = [action_id for action_id in decision.rejected_action_ids if action_id in active_ids and action_id in rejected_ui_ids]
    update: dict[str, Any] = {"approved_action_ids": approved_ids, "rejected_action_ids": rejected_ids}
    if approved_ids and rejected_ids and (decision.revision_request or state.query.strip()):
        update.update(
            {
                "intent": "apply_partial_and_revise",
                "revision_request": decision.revision_request or state.query.strip(),
                "needs_user_clarification": False,
                "rationale": (
                    decision.rationale
                    or "Accepted course-card actions can be executed while declined actions are revised."
                ),
            }
        )
    elif approved_ids and decision.needs_user_clarification and decision.intent in {"apply_selected", "unclear"}:
        update.update({"intent": "apply_selected", "needs_user_clarification": False})
    return decision.model_copy(update=update)


def _fallback_decision_interpretation(state: StudyChatFlowState) -> UserDecisionInterpretation:
    approved_ids = [decision.action_id for decision in state.ui_decisions if decision.approved]
    disabled_ids = [
        decision.action_id
        for decision in state.ui_decisions
        if not decision.approved and "disabled" in decision.feedback.casefold()
    ]
    rejected_ids = [
        decision.action_id
        for decision in state.ui_decisions
        if not decision.approved and decision.action_id not in set(disabled_ids)
    ]
    text = state.query.casefold()

    if _looks_like_discard_request(text):
        return UserDecisionInterpretation(
            intent="discard_active_proposals",
            approved_action_ids=[],
            rejected_action_ids=rejected_ids,
            discard_active_proposals=True,
            rationale="Fallback interpreter detected a request to leave the active recommendation flow.",
        )

    if _looks_like_unrelated_new_topic(text) and not approved_ids and not rejected_ids and not disabled_ids:
        return UserDecisionInterpretation(
            intent="ask_question",
            discard_active_proposals=True,
            rationale="Fallback interpreter detected a new unrelated topic.",
        )

    if rejected_ids and not approved_ids:
        return UserDecisionInterpretation(
            intent="revise_only",
            rejected_action_ids=rejected_ids,
            revision_request=state.query,
            rationale="Rejected UI cards should be revised, not executed.",
        )

    if approved_ids and (_looks_like_revision_request(text) or rejected_ids):
        return UserDecisionInterpretation(
            intent="apply_partial_and_revise",
            approved_action_ids=approved_ids,
            rejected_action_ids=[*rejected_ids, *disabled_ids],
            revision_request=state.query,
            rationale="Accepted UI cards can be executed while declined cards are revised.",
        )

    if approved_ids and not _looks_like_information_question(text):
        return UserDecisionInterpretation(
            intent="apply_selected",
            approved_action_ids=approved_ids,
            rejected_action_ids=disabled_ids,
            rationale="Accepted UI cards are treated as the explicit authorization gate.",
        )

    return UserDecisionInterpretation(intent="ask_question", rationale="No execution decision was inferred.")


def _looks_like_discard_request(text: str) -> bool:
    return any(
        token in text
        for token in [
            "cancel",
            "discard",
            "forget this",
            "forget it",
            "start over",
            "new topic",
            "stop recommendation",
            "abbrechen",
            "verwerfen",
            "vergiss",
            "neues thema",
            "nicht weiter",
        ]
    )


def _looks_like_unrelated_new_topic(text: str) -> bool:
    course_terms = [
        "course",
        "module",
        "study",
        "semester",
        "degree",
        "isis",
        "moses",
        "grade",
        "recommend",
        "kurs",
        "modul",
        "studium",
        "semester",
        "empfehl",
        "einschreib",
    ]
    if any(term in text for term in course_terms):
        return False
    unrelated_terms = ["weather", "wetter", "news", "joke", "recipe", "song", "movie", "film"]
    return any(term in text for term in unrelated_terms)


def _looks_like_revision_request(text: str) -> bool:
    return any(
        token in text
        for token in [
            "replace",
            "exchange",
            "swap",
            "different",
            "alternative",
            "revise",
            "change",
            "tausche",
            "ersetze",
            "anders",
            "alternative",
            "wechsel",
            "ändere",
            "aendere",
        ]
    )


def _looks_like_information_question(text: str) -> bool:
    stripped = text.strip()
    if "?" in stripped:
        return True
    return stripped.startswith(("what ", "why ", "how ", "when ", "where ", "wer ", "was ", "wie ", "warum ", "wann ", "wo "))


def _has_separate_user_question(query: str) -> bool:
    text = query.casefold()
    return _looks_like_unrelated_new_topic(text) or _looks_like_information_question(text)


def _normalize_commitment_route(intent: IntentClassification, query: str) -> IntentClassification:
    if intent.route in {"deep_dive"}:
        return intent
    if not intent.write_intent and not _looks_like_course_commitment_request(query.casefold()):
        return intent
    if intent.route == "recommendation":
        return intent

    sources = _dedupe_sources([*intent.required_sources, *_commitment_sources_for_query(query.casefold())])
    return intent.model_copy(
        update={
            "route": "recommendation",
            "complexity": "scoped",
            "required_sources": sources or ["grade_manager", "moses", "isis"],
            "write_intent": True,
            "tool_budget": max(intent.tool_budget, 12),
            "rationale": (
                f"{intent.rationale} Routed through Course Commitment because write/enrollment "
                "requests must not enter read-only simple flows."
            ).strip(),
        }
    )


def _looks_like_course_commitment_request(text: str) -> bool:
    direct_isis_write = [
        "enroll me",
        "enrol me",
        "please enroll",
        "please enrol",
        "self-enroll me",
        "self enrol me",
        "sign me up",
        "register me for",
        "add me to",
        "join the course",
        "join this course",
        "melde mich",
        "meld mich",
        "schreib mich",
        "bitte einschreib",
        "bitte anmelden",
        "trag mich",
    ]
    direct_plan_write = [
        "add ",
        "save ",
        "put ",
        "write ",
        "add this",
        "add it",
        "add that",
        "add the module",
        "add the course",
        "to my study plan",
        "to grade manager",
        "in my plan",
        "eintragen",
        "trag ",
        "füge ",
        "hinzufügen",
    ]
    has_isis_target = any(token in text for token in ["isis", "moodle", "course", "kurs"])
    has_plan_target = any(token in text for token in ["study plan", "grade manager", "module", "modul", "plan"])
    if has_isis_target and any(token in text for token in direct_isis_write):
        return True
    return has_plan_target and any(token in text for token in direct_plan_write)


def _commitment_sources_for_query(text: str) -> list[str]:
    sources: list[str] = []
    if any(token in text for token in ["isis", "moodle", "enroll", "enrol", "register", "einschreib", "anmeld"]):
        sources.append("isis")
    if any(token in text for token in ["study plan", "grade manager", "module", "modul", "plan"]):
        sources.extend(["grade_manager", "moses"])
    return _dedupe_sources(sources)


def _dedupe_sources(sources: list[str]) -> list[str]:
    allowed = {"degree_regulations", "grade_manager", "moses", "isis"}
    result: list[str] = []
    for source in sources:
        if source in allowed and source not in result:
            result.append(source)
    return result


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
        allow_offering_mismatch=bool(payload.get("allow_offering_mismatch", False)),
        confirmation_token=STUDY_PLAN_CONFIRMATION_TOKEN,
    )
    verification = check_module_against_study_plan(
        module_query=module_query,
        version=payload.get("version"),
        program_key=payload.get("program_key"),
        term=term,
    )
    if result.startswith("Added `"):
        status = "executed"
    elif result.startswith("Study-plan write refused: this MOSES module is already present"):
        status, verification_result = _verify_grade_manager_addition(action)
        if status != "executed":
            verification = verification_result
    else:
        status = "failed"
    return action.model_copy(update={"status": status, "result": f"{result}\n\nVerification:\n{verification}"})


def _execute_grade_manager_update_action(action: ProposedAction) -> ProposedAction:
    payload = dict(action.grade_manager_payload or {})
    module_query = str(payload.get("module_query") or action.course_title).strip()
    target_term = str(payload.get("target_term") or payload.get("term") or "").strip()
    if not target_term:
        return action.model_copy(update={"status": "needs_clarification", "result": "No target term was provided."})
    result = update_module_in_study_plan(
        module_query=module_query,
        target_term=target_term,
        version=payload.get("version"),
        program_key=payload.get("program_key"),
        current_term=payload.get("current_term"),
        current_area=payload.get("current_area"),
        current_state=payload.get("current_state") or "Planned",
        target_area=payload.get("target_area") or payload.get("area"),
        target_state=payload.get("target_state"),
        allow_offering_mismatch=bool(payload.get("allow_offering_mismatch", False)),
        allow_non_planned=bool(payload.get("allow_non_planned", False)),
        confirmation_token=STUDY_PLAN_CONFIRMATION_TOKEN,
    )
    verification = check_module_against_study_plan(
        module_query=module_query,
        version=payload.get("version"),
        program_key=payload.get("program_key"),
        term=target_term,
    )
    status = "executed" if result.startswith("Updated `") or result.startswith("No study-plan change needed:") else "failed"
    return action.model_copy(update={"status": status, "result": f"{result}\n\nVerification:\n{verification}"})


def _execute_grade_manager_remove_action(action: ProposedAction) -> ProposedAction:
    payload = dict(action.grade_manager_payload or {})
    module_query = str(payload.get("module_query") or action.course_title).strip()
    result = remove_module_from_study_plan(
        module_query=module_query,
        version=payload.get("version"),
        program_key=payload.get("program_key"),
        current_term=payload.get("current_term") or payload.get("term"),
        current_area=payload.get("current_area") or payload.get("area"),
        current_state=payload.get("current_state") or "Planned",
        allow_non_planned=bool(payload.get("allow_non_planned", False)),
        confirmation_token=STUDY_PLAN_CONFIRMATION_TOKEN,
    )
    status = "executed" if result.startswith("Removed `") else "failed"
    return action.model_copy(update={"status": status, "result": result})


def _execute_isis_action(action: ProposedAction) -> ProposedAction:
    payload = {key: value for key, value in dict(action.isis_payload or {}).items() if value not in (None, "")}
    moses_number = str(payload.get("moses_module_number") or "").strip()
    course_id = str(payload.get("course_id") or "").strip()
    if course_id and moses_number and course_id == moses_number:
        return action.model_copy(
            update={
                "status": "needs_clarification",
                "result": (
                    f"ISIS enrollment blocked: `{course_id}` is also the MOSES module number. "
                    "A verified ISIS course ID or URL is required before enrollment."
                ),
            }
        )
    if not any(payload.get(key) for key in ("course_id", "course_query", "course_url")):
        return action.model_copy(update={"status": "needs_clarification", "result": "No ISIS course id, query, or URL was provided."})
    payload.pop("moses_module_number", None)
    payload.pop("isis_resolution_status", None)
    outcome = PermanentlyEnrollInIsisCourseTool().run_structured(
        **payload,
        confirmation_token=CONFIRMATION_TOKEN,
    )
    status = _isis_action_status_from_outcome(outcome)
    result = outcome.reason or f"ISIS enrollment status: {outcome.status}"
    if outcome.candidates:
        candidate_lines = [
            f"- `{candidate.id}` {candidate.title} ({candidate.term_hint or 'term unknown'})"
            for candidate in outcome.candidates[:8]
        ]
        result = f"{result}\n\nCandidate ISIS courses:\n" + "\n".join(candidate_lines)
    return action.model_copy(update={"status": status, "result": result})


def _isis_action_status_from_outcome(outcome: Any) -> str:
    if getattr(outcome, "ok", False):
        return "executed"
    status = str(getattr(outcome, "status", "") or "")
    candidates = list(getattr(outcome, "candidates", []) or [])
    if status in {"ambiguous_course", "missing_isis_id"}:
        return "needs_clarification"
    if status == "not_found" and candidates:
        return "needs_clarification"
    return "failed"


def _format_discard_answer(decision: UserDecisionInterpretation | None) -> str:
    if decision and decision.rationale:
        return "I cleared the active course recommendations. You can continue with the new topic."
    return "I cleared the active course recommendations."


def _format_proposal_clarification_answer(decision: UserDecisionInterpretation | None) -> str:
    if decision and decision.rationale:
        return f"I need one clarification before changing the selected course actions: {decision.rationale}"
    return "I need one clarification before changing the selected course actions."


def _format_execution_answer(actions: list[ProposedAction], *, language: str) -> str:
    if language == "de":
        lines = ["## Bestätigte Aktionen", ""]
        if not actions:
            return "Ich habe keine bestätigten Aktionen erhalten."
        for course_title, course_actions in _actions_grouped_by_course(actions):
            lines.append(f"- **{course_title}**")
            for action in course_actions:
                lines.append(f"  - {_action_kind_label_de(action.kind)}: `{action.status}`")
                if action.result:
                    lines.append(f"    {action.result.splitlines()[0]}")
        return "\n".join(lines)
    lines = ["## Confirmed Actions", ""]
    if not actions:
        return "I did not receive any confirmed actions to execute."
    for course_title, course_actions in _actions_grouped_by_course(actions):
        lines.append(f"- **{course_title}**")
        for action in course_actions:
            lines.append(f"  - {_action_kind_label_en(action.kind)}: `{action.status}`")
            if action.result:
                lines.append(f"    {action.result.splitlines()[0]}")
    return "\n".join(lines)


def _actions_grouped_by_course(actions: list[ProposedAction]) -> list[tuple[str, list[ProposedAction]]]:
    grouped: dict[str, list[ProposedAction]] = {}
    order: list[str] = []
    for action in actions:
        title = action.course_title or "Course action"
        if title not in grouped:
            grouped[title] = []
            order.append(title)
        grouped[title].append(action)
    return [(title, grouped[title]) for title in order]


def _action_kind_label_en(kind: str) -> str:
    return {
        "grade_manager_add": "Study Manager add",
        "grade_manager_update": "Study Manager update",
        "grade_manager_remove": "Study Manager remove",
        "isis_resolve": "ISIS resolve/enroll",
        "isis_enroll": "ISIS enroll",
    }.get(kind, kind.replace("_", " "))


def _action_kind_label_de(kind: str) -> str:
    return {
        "grade_manager_add": "Study Manager hinzufügen",
        "grade_manager_update": "Study Manager aktualisieren",
        "grade_manager_remove": "Study Manager entfernen",
        "isis_resolve": "ISIS auflösen/einschreiben",
        "isis_enroll": "ISIS einschreiben",
    }.get(kind, kind.replace("_", " "))


def _conversation_context(thread: ChatThreadState) -> str:
    parts: list[str] = []
    if thread.rolling_summary:
        parts.append(f"Summary:\n{thread.rolling_summary}")
    recent = thread.recent_messages_text(limit=8)
    if recent:
        parts.append(f"Recent messages:\n{recent}")
    if thread.active_proposals:
        proposal_lines = []
        for proposal in thread.active_proposals:
            action_lines = []
            for action in proposal.actions:
                action_lines.append(f"  * [{action.status.upper()}] {action.kind}: {action.course_title}")
            actions_str = "\n".join(action_lines)
            proposal_lines.append(f"- {proposal.title}: {proposal.summary}\n{actions_str}")
        parts.append("Active proposals:\n" + "\n".join(proposal_lines))
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
