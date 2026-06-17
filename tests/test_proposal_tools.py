from __future__ import annotations

import crew.runtime
from core import persistence
from crew.chat_persistence import append_turn, load_chat_thread
from crew.profile_context import use_grade_manager_profile
from crew.tools.proposal_tools import (
    ClearAllCourseProposalsTool,
    DeleteCourseProposalTool,
    ProposalCourseInput,
    ProposeCourseActionsTool,
    build_course_proposal,
    collect_course_proposals,
    current_course_proposals,
    record_course_proposals,
)


def _setup_profile(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "profiles"
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(persistence, "PROFILES_FILE", profiles_dir / "profiles.json")
    monkeypatch.setattr(persistence, "_LEGACY_MODULES_FILE", data_dir / "modules.json")
    persistence.save_profiles(
        [persistence.ProfileRecord(slug="primary", display_name="Primary Test Student", is_primary=True)]
    )


def test_proposal_tool_records_explicit_ui_actions_only_inside_collection_context():
    tool = ProposeCourseActionsTool()

    with collect_course_proposals() as proposals:
        output = tool._run(
            proposal_title="Next semester ML plan",
            proposal_summary="Two ML-focused courses fit the student's open elective credits.",
            courses=[
                {
                    "course_title": "Reinforcement Learning",
                    "rationale": "Matches the ML preference and counts for electives.",
                    "evidence": ["MOSES: 6 LP", "Grade Manager: elective credits open"],
                    "module_query": "40967",
                    "term": "WS 26/27",
                    "area": "Elective",
                    "include_grade_manager": True,
                    "include_isis": True,
                    "isis_course_id": 48000,
                }
            ],
        )

    assert "Prepared UI confirmation proposal" in output
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.title == "Next semester ML plan"
    assert [action.kind for action in proposal.actions] == ["grade_manager_add", "isis_enroll"]
    assert proposal.actions[0].grade_manager_payload["module_query"] == "40967"
    assert proposal.actions[1].isis_payload["course_id"] == 48000
    assert proposal.actions[0].action_id == proposal.actions[0].action_id


def test_proposal_tool_creates_isis_resolution_action_when_id_is_unverified():
    tool = ProposeCourseActionsTool()

    with collect_course_proposals() as proposals:
        tool._run(
            proposal_title="Unresolved ISIS option",
            proposal_summary="ISIS still needs a real Moodle course id.",
            courses=[
                {
                    "course_title": "Systemprogrammierung",
                    "rationale": "Required course.",
                    "evidence": ["MOSES module 40441"],
                    "module_query": "40441",
                    "term": "SS 26",
                    "include_grade_manager": True,
                    "include_isis": True,
                    "isis_course_id": 40441,
                    "isis_course_query": "Systemprogrammierung",
                }
            ],
        )

    assert len(proposals) == 1
    actions = proposals[0].actions
    assert [action.kind for action in actions] == ["grade_manager_add", "isis_resolve"]
    assert actions[1].isis_payload["course_id"] is None
    assert actions[1].isis_payload["moses_module_number"] == "40441"
    assert actions[1].isis_payload["course_query"] == "Systemprogrammierung"


def test_proposal_tool_defaults_to_bundled_study_plan_and_isis_resolution():
    tool = ProposeCourseActionsTool()

    with collect_course_proposals() as proposals:
        tool._run(
            proposal_title="Bundled option",
            proposal_summary="Default course commitment.",
            courses=[
                {
                    "course_title": "Einführung in die Programmierung",
                    "rationale": "Mandatory course.",
                    "module_query": "40017",
                    "term": "WS 26/27",
                    "area": "Mandatory",
                }
            ],
        )

    actions = proposals[0].actions
    assert [action.kind for action in actions] == ["grade_manager_add", "isis_resolve"]
    assert actions[0].grade_manager_payload["module_query"] == "40017"
    assert actions[1].isis_payload["course_query"] == "Einführung in die Programmierung"
    assert actions[1].isis_payload["isis_resolution_status"] == "unresolved"


def test_delete_course_proposal_tool_removes_agent_suggestion_from_thread_and_collector(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    keep = build_course_proposal(
        proposal_title="Keep this",
        proposal_summary="Still useful.",
        courses=[ProposalCourseInput(course_title="Course A", rationale="Keep.", module_query="11111", term="WS 26/27")],
    )
    remove = build_course_proposal(
        proposal_title="Remove this",
        proposal_summary="No longer useful.",
        courses=[ProposalCourseInput(course_title="Course B", rationale="Remove.", module_query="22222", term="WS 26/27")],
    )
    append_turn(
        "primary",
        user_content="Suggest courses.",
        assistant_content="Two suggestions.",
        proposals=[keep, remove],
        rolling_summary="Planning.",
    )

    with use_grade_manager_profile("primary"), collect_course_proposals():
        record_course_proposals([keep, remove])
        output = DeleteCourseProposalTool()._run("Remove this")
        assert "Successfully deleted" in output
        assert [proposal.title for proposal in current_course_proposals()] == ["Keep this"]

    assert [proposal.title for proposal in load_chat_thread("primary").active_proposals] == ["Keep this"]


def test_clear_all_course_proposals_tool_removes_agent_suggestions(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Clear this",
        proposal_summary="Will be removed.",
        courses=[ProposalCourseInput(course_title="Course A", rationale="Clear.", module_query="11111", term="WS 26/27")],
    )
    append_turn(
        "primary",
        user_content="Suggest courses.",
        assistant_content="One suggestion.",
        proposals=[proposal],
        rolling_summary="Planning.",
    )

    with use_grade_manager_profile("primary"), collect_course_proposals() as proposals:
        record_course_proposals([proposal])
        output = ClearAllCourseProposalsTool()._run()
        assert "Successfully cleared" in output
        assert proposals == []

    assert load_chat_thread("primary").active_proposals == []


def test_proposal_tool_does_not_record_without_context():
    tool = ProposeCourseActionsTool()
    output = tool._run(
        proposal_title="No collector",
        proposal_summary="No collector is active.",
        courses=[
            {
                "course_title": "Machine Learning 2",
                "rationale": "Useful course.",
                "module_query": "40968",
                "term": "WS 26/27",
            }
        ],
    )

    assert "Prepared UI confirmation proposal" in output


def test_proposal_tool_propagates_context_to_threads():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    tool = ProposeCourseActionsTool()

    # Test threading.Thread propagation
    with collect_course_proposals() as proposals:
        def target():
            tool._run(
                proposal_title="Thread Proposal",
                proposal_summary="Thread summary.",
                courses=[{"course_title": "Test Course", "rationale": "Test", "term": "SS 26"}],
            )
        t = threading.Thread(target=target)
        t.start()
        t.join()
        assert len(proposals) == 1
        assert proposals[0].title == "Thread Proposal"

    # Test ThreadPoolExecutor propagation
    with collect_course_proposals() as proposals:
        def target_executor():
            tool._run(
                proposal_title="Executor Proposal",
                proposal_summary="Executor summary.",
                courses=[{"course_title": "Test Course", "rationale": "Test", "term": "SS 26"}],
            )
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(target_executor)
            future.result()
        assert len(proposals) == 1
        assert proposals[0].title == "Executor Proposal"
