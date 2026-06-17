from __future__ import annotations

from crew.chat_models import ProposedAction
from crew.commitment_scope import approved_commitment_scope
from crew.tools.commitment_tools import (
    ExecuteApprovedCourseCommitmentsTool,
    ListApprovedCourseCommitmentActionsTool,
)
from crew.write_permissions import confirmed_writes_enabled


def test_commitment_execution_tool_refuses_without_ui_scope():
    output = ExecuteApprovedCourseCommitmentsTool()._run(action_ids=["some-action"])

    assert "refused" in output
    assert "approved UI action scope" in output


def test_commitment_list_tool_refuses_without_ui_scope():
    output = ListApprovedCourseCommitmentActionsTool()._run()

    assert "No approved UI commitment scope is active" in output


def test_commitment_execution_tool_requires_approved_action_id(monkeypatch):
    action = ProposedAction(
        action_id="grade_manager_add-approved",
        kind="grade_manager_add",
        course_title="Machine Learning 2",
        grade_manager_payload={
            "module_query": "40967",
            "term": "WS 26/27",
        },
    )

    import crew.study_chat_flow as flow_module

    def fake_execute(approved_action):
        assert confirmed_writes_enabled()
        return approved_action.model_copy(update={"status": "executed", "result": "Added."})

    monkeypatch.setattr(flow_module, "_execute_grade_manager_action", fake_execute)

    with approved_commitment_scope([action]) as scope:
        output = ExecuteApprovedCourseCommitmentsTool()._run(
            action_ids=["not-approved", "grade_manager_add-approved"]
        )

        assert "not-approved" in output
        assert "refused, action was not approved in the UI" in output
        assert "grade_manager_add-approved" in scope.executed_by_id
        assert scope.executed_by_id["grade_manager_add-approved"].status == "executed"


def test_commitment_list_tool_exposes_only_ui_action_packet():
    action = ProposedAction(
        action_id="isis-approved",
        kind="isis_resolve",
        course_title="Rechnerorganisation",
        isis_payload={
            "course_query": "Rechnerorganisation",
            "term_hint": "WS 26/27",
            "expected_title": "Rechnerorganisation",
        },
    )

    with approved_commitment_scope([action]):
        output = ListApprovedCourseCommitmentActionsTool()._run()

    assert "isis-approved" in output
    assert "Rechnerorganisation" in output
