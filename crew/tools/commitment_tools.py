from __future__ import annotations

import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from crew.chat_models import ProposedAction
from crew.commitment_scope import current_approved_commitment_scope


class ExecuteApprovedCommitmentsInput(BaseModel):
    action_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Approved UI action IDs to execute. Leave empty to execute every action "
            "in the current approved UI packet."
        ),
    )


class ListApprovedCourseCommitmentActionsTool(BaseTool):
    name: str = "List Approved Course Commitment Actions"
    description: str = (
        "List the exact course commitment action IDs approved by the Streamlit UI. "
        "This is read-only and is the safe first step before executing commitments."
    )

    def _run(self) -> str:
        scope = current_approved_commitment_scope()
        if scope is None:
            return (
                "No approved UI commitment scope is active. Course commitments can only "
                "be executed after the user accepts actions in the UI."
            )
        rows = [
            {
                "action_id": action.action_id,
                "kind": action.kind,
                "course_title": action.course_title,
                "grade_manager_payload": action.grade_manager_payload,
                "isis_payload": action.isis_payload,
            }
            for action in scope.actions
            if action.action_id in scope.approved_action_ids
        ]
        return json.dumps(rows, ensure_ascii=False, indent=2)


class ExecuteApprovedCourseCommitmentsTool(BaseTool):
    name: str = "Execute Approved Course Commitment Actions"
    description: str = (
        "Execute Study Manager and ISIS course commitments that were already accepted "
        "in the Streamlit UI. Guardrails reject calls without an approved UI scope, "
        "unknown action IDs, declined actions, and action-kind mismatches. This tool "
        "does not accept raw confirmation tokens."
    )
    args_schema: Type[BaseModel] = ExecuteApprovedCommitmentsInput

    def _run(self, action_ids: list[str] | None = None) -> str:
        scope = current_approved_commitment_scope()
        if scope is None:
            return (
                "Course commitment execution refused: no approved UI action scope is active. "
                "The user must accept the course actions in the UI first."
            )

        requested_ids = [str(action_id).strip() for action_id in (action_ids or []) if str(action_id).strip()]
        if not requested_ids:
            requested_ids = [action.action_id for action in scope.actions if action.action_id in scope.approved_action_ids]

        action_by_id = scope.actions_by_id
        lines = ["# Approved course commitment execution", ""]
        for action_id in requested_ids:
            if action_id not in scope.approved_action_ids:
                lines.append(f"- `{action_id}`: refused, action was not approved in the UI.")
                continue
            action = action_by_id.get(action_id)
            if action is None:
                lines.append(f"- `{action_id}`: refused, no active UI action exists with this ID.")
                continue
            if action_id in scope.executed_by_id:
                executed = scope.executed_by_id[action_id]
                lines.append(f"- `{action_id}` {executed.course_title} ({executed.kind}): already {executed.status}.")
                continue

            executed = _execute_action(action)
            scope.executed_by_id[action_id] = executed
            lines.append(f"- `{action_id}` {executed.course_title} ({executed.kind}): {executed.status}")
            if executed.result:
                lines.extend(_indented_result_lines(executed))

        if len(lines) == 2:
            lines.append("- No approved actions were available to execute.")
        return "\n".join(lines).rstrip()


def _execute_action(action: ProposedAction) -> ProposedAction:
    from crew import study_chat_flow as flow_module

    if action.kind == "grade_manager_add":
        return flow_module._execute_grade_manager_action(action)
    if action.kind == "grade_manager_update":
        return flow_module._execute_grade_manager_update_action(action)
    if action.kind == "grade_manager_remove":
        return flow_module._execute_grade_manager_remove_action(action)
    if action.kind in {"isis_enroll", "isis_resolve"}:
        return flow_module._execute_isis_action(action)
    return action.model_copy(
        update={
            "status": "failed",
            "result": f"Unknown action kind: {action.kind}",
        }
    )


def _indented_result_lines(action: ProposedAction) -> list[str]:
    result = action.result or ""
    lines = [line for line in result.splitlines() if line.strip()]
    if not lines:
        return []
    include_details = action.status == "needs_clarification" or "Candidate ISIS courses:" in result
    selected = lines[:10] if include_details else lines[:1]
    suffix = ["  ..."] if len(lines) > len(selected) else []
    return [f"  {line}" for line in selected] + suffix


COURSE_COMMITMENT_EXECUTION_TOOLS = [
    ListApprovedCourseCommitmentActionsTool(),
    ExecuteApprovedCourseCommitmentsTool(),
]
