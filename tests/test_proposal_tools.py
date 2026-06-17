from __future__ import annotations

import crew.runtime
from crew.tools.proposal_tools import ProposeCourseActionsTool, collect_course_proposals


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
