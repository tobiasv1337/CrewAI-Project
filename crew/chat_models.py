from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from crew.state import IsisLookupContext



MessageRole = Literal["user", "assistant", "system"]
IntentRoute = Literal[
    "simple_grade_manager",
    "simple_grade_optimization",
    "simple_moses",
    "simple_isis",
    "simple_degree_regulations",
    "recommendation",
    "deep_dive",
    "execute_confirmed_actions",
    "execute_then_recommendation",
    "discard_active_proposals",
]
ActionKind = Literal[
    "grade_manager_add",
    "grade_manager_update",
    "grade_manager_remove",
    "isis_resolve",
    "isis_enroll",
]
ActionStatus = Literal["proposed", "approved", "declined", "executed", "failed", "needs_clarification"]
UserDecisionIntent = Literal[
    "apply_selected",
    "apply_partial_and_revise",
    "revise_only",
    "ask_question",
    "discard_active_proposals",
    "unclear",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    created_at: str = Field(default_factory=utc_now_iso)
    trace_dir: str | None = None
    state_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProposedAction(BaseModel):
    action_id: str
    kind: ActionKind
    course_title: str
    evidence: list[str] = Field(default_factory=list)
    grade_manager_payload: dict[str, Any] | None = None
    isis_payload: dict[str, Any] | None = None
    requires_confirmation: bool = True
    status: ActionStatus = "proposed"
    result: str | None = None


class CourseProposal(BaseModel):
    proposal_id: str
    title: str
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)
    actions: list[ProposedAction] = Field(default_factory=list)
    status: ActionStatus = "proposed"
    created_at: str = Field(default_factory=utc_now_iso)
    source_agent: str | None = None

    @property
    def proposed_actions(self) -> list[ProposedAction]:
        return [action for action in self.actions if action.status == "proposed"]


class ActionDecision(BaseModel):
    action_id: str
    approved: bool
    feedback: str = ""


class IntentClassification(BaseModel):
    route: IntentRoute = "deep_dive"
    language: Literal["en", "de"] = "en"
    complexity: Literal["simple", "scoped", "deep"] = "deep"
    follow_up_target: str | None = None
    required_sources: list[Literal["degree_regulations", "grade_manager", "grade_optimization", "moses", "isis"]] = Field(default_factory=list)
    write_intent: bool = False
    tool_budget: int = 12
    rationale: str = ""


class UserDecisionInterpretation(BaseModel):
    intent: UserDecisionIntent = "ask_question"
    approved_action_ids: list[str] = Field(default_factory=list)
    rejected_action_ids: list[str] = Field(default_factory=list)
    revision_request: str | None = None
    discard_active_proposals: bool = False
    needs_user_clarification: bool = False
    rationale: str = ""


class ChatThreadState(BaseModel):
    thread_id: str = "default"
    profile_slug: str = "primary"
    messages: list[ChatMessage] = Field(default_factory=list)
    rolling_summary: str = ""
    active_proposals: list[CourseProposal] = Field(default_factory=list)
    proposal_decisions: list[ActionDecision] = Field(default_factory=list)
    trace_artifacts: list[dict[str, str]] = Field(default_factory=list)
    isis_context: IsisLookupContext = Field(default_factory=IsisLookupContext)
    updated_at: str = Field(default_factory=utc_now_iso)

    def recent_messages_text(self, limit: int = 8) -> str:
        recent = self.messages[-max(limit, 1) :]
        return "\n".join(f"{message.role}: {message.content}" for message in recent)


class StudyChatFlowState(BaseModel):
    query: str = ""
    profile_slug: str = "primary"
    thread_id: str = "default"
    reset_thread: bool = False
    student_context: str = ""
    isis_context_json: str = "{}"
    isis_session_mode: str = "env"
    conversation_context: str = ""
    approved_actions: list[ActionDecision] = Field(default_factory=list)
    ui_decisions: list[ActionDecision] = Field(default_factory=list)
    thread: ChatThreadState | None = None
    intent: IntentClassification | None = None
    decision_interpretation: UserDecisionInterpretation | None = None
    route: IntentRoute = "deep_dive"
    answer_markdown: str = ""
    proposed_actions: list[CourseProposal] = Field(default_factory=list)
    executed_actions: list[ProposedAction] = Field(default_factory=list)
    tool_summary_lines: list[str] = Field(default_factory=list)
    trace_dir: str | None = None
    state_path: str | None = None


class StudyChatTurnResult(BaseModel):
    answer: str
    proposed_actions: list[CourseProposal] = Field(default_factory=list)
    executed_actions: list[ProposedAction] = Field(default_factory=list)
    intent: IntentClassification | None = None
    thread: ChatThreadState | None = None
    trace_dir: str | None = None
    state_path: str | None = None
