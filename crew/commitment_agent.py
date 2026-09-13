from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from crewai import Agent

from crew.chat_models import ProposedAction
from crew.commitment_scope import approved_commitment_scope
from crew.config.llm import get_default_llm
from crew.tools.commitment_tools import (
    COURSE_COMMITMENT_EXECUTION_TOOLS,
    ExecuteApprovedCourseCommitmentsTool,
)


def run_course_commitment_execution_agent(
    actions: list[ProposedAction],
    *,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    verbose: bool = False,
    cache: bool = True,
    agent_enabled: bool = True,
    on_trace_event: Callable[[dict[str, Any]], None] | None = None,
) -> list[ProposedAction]:
    """Execute approved course commitments through a guarded specialist tool surface."""

    if not actions:
        return []

    with approved_commitment_scope(actions) as scope:
        _emit_trace(
            on_trace_event,
            {
                "event": "task_started",
                "agent_label": "Course Commitment Specialist",
                "agent_role": "TU Berlin Course Commitment Specialist",
                "task_name": "Execute approved course commitments",
                "phase": "task",
                "status": "running",
                "activity": "Executing UI-approved Study Manager and ISIS commitments with guarded tools.",
            },
        )
        if agent_enabled and os.getenv("GWDG_API_KEY"):
            try:
                agent = Agent(
                    role="TU Berlin Course Commitment Specialist",
                    goal="Execute only the exact Study Manager and ISIS course actions accepted in the UI.",
                    backstory=(
                        "You execute confirmed TU Berlin course commitments. The UI approval packet is "
                        "the authorization boundary. You never invent action IDs, never execute declined "
                        "or merely proposed actions, and report unclear ISIS resolution instead of guessing."
                    ),
                    tools=list(COURSE_COMMITMENT_EXECUTION_TOOLS),
                    llm=get_default_llm(model=model, temperature=temperature, top_p=top_p, thinking=False),
                    verbose=verbose,
                    cache=cache,
                    max_iter=8,
                    max_execution_time=480,
                )
                agent.kickoff(_execution_prompt(actions))
            except Exception as exc:
                _emit_trace(
                    on_trace_event,
                    {
                        "event": "task_failed",
                        "agent_label": "Course Commitment Specialist",
                        "agent_role": "TU Berlin Course Commitment Specialist",
                        "task_name": "Execute approved course commitments",
                        "phase": "task",
                        "status": "error",
                        "activity": "Course Commitment agent failed; guarded deterministic fallback will execute remaining approved actions.",
                        "error": str(exc),
                    },
                )

        missing_ids = [
            action.action_id
            for action in actions
            if action.action_id in scope.approved_action_ids and action.action_id not in scope.executed_by_id
        ]
        if missing_ids:
            ExecuteApprovedCourseCommitmentsTool()._run(action_ids=missing_ids)

        executed = scope.ordered_executed_actions()
        _emit_trace(
            on_trace_event,
            {
                "event": "task_completed",
                "agent_label": "Course Commitment Specialist",
                "agent_role": "TU Berlin Course Commitment Specialist",
                "task_name": "Execute approved course commitments",
                "phase": "task",
                "status": "ok",
                "activity": f"Course Commitment execution completed for {len(executed)} action(s).",
            },
        )
        return executed


def _execution_prompt(actions: list[ProposedAction]) -> str:
    compact_actions = [
        {
            "action_id": action.action_id,
            "kind": action.kind,
            "course_title": action.course_title,
            "grade_manager_payload": action.grade_manager_payload,
            "isis_payload": action.isis_payload,
        }
        for action in actions
    ]
    return (
        "Execute the approved UI course commitments.\n\n"
        "Required steps:\n"
        "1. Call `List Approved Course Commitment Actions` to inspect the approved action IDs.\n"
        "2. Call `Execute Approved Course Commitment Actions` once. You may pass the action_ids listed below, "
        "or leave action_ids empty to execute all approved actions.\n"
        "3. In your final response, summarize each action status exactly as returned by the tool. "
        "If an ISIS action needs clarification, preserve the candidate course IDs.\n\n"
        "Approved actions:\n"
        f"{json.dumps(compact_actions, ensure_ascii=False, indent=2)}"
    )


def _emit_trace(on_trace_event: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if on_trace_event is None:
        return
    try:
        on_trace_event(event)
    except Exception:
        return
