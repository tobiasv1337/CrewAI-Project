from __future__ import annotations

from core import persistence
from crew.chat_models import ActionDecision, StudyChatFlowState
from crew.chat_persistence import append_turn, load_chat_thread
from crew.study_chat_flow import StudyChatFlow, StudyChatFlowRuntime
from crew.tools.grademanager_tools import STUDY_PLAN_CONFIRMATION_TOKEN
from crew.tools.proposal_tools import ProposalCourseInput, ProposeCourseActionsTool, build_course_proposal


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


def _flow(**kwargs):
    runtime = StudyChatFlowRuntime(use_llm_classifier=False)
    return StudyChatFlow(runtime=runtime, **kwargs)


def test_simple_progress_question_routes_only_to_study_advisor(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    calls = []

    def grade_runner(flow):
        calls.append("grade")
        return "You have 60 LP completed."

    def forbidden(flow):
        raise AssertionError("wrong route")

    flow = _flow(
        runner_overrides={
            "simple_grade_manager": grade_runner,
            "simple_moses": forbidden,
            "simple_isis": forbidden,
            "deep_dive": forbidden,
            "recommendation": forbidden,
        }
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="How many credits are still missing in my degree?",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert calls == ["grade"]
    assert flow.state.intent.route == "simple_grade_manager"
    assert load_chat_thread("primary").messages[-1].content == "You have 60 LP completed."


def test_current_week_deadline_question_routes_to_deep_dive_for_grade_manager_scoping(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    calls = []

    def deep_runner(flow):
        calls.append("deep")
        assert "Current user message" in flow._contextual_query()
        return "Scoped deadlines from active Grade Manager courses."

    flow = _flow(runner_overrides={"deep_dive": deep_runner})
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="What deadlines are due next week in my current courses?",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert calls == ["deep"]
    assert flow.state.intent.route == "deep_dive"
    assert flow.state.intent.required_sources == ["grade_manager", "isis"]


def test_recommendation_route_uses_explicit_proposal_tool(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)

    def recommendation_runner(flow):
        ProposeCourseActionsTool()._run(
            proposal_title="ML-heavy next semester",
            proposal_summary="Two ML courses fit the remaining elective room.",
            courses=[
                {
                    "course_title": "Reinforcement Learning",
                    "rationale": "Matches the user's ML preference.",
                    "module_query": "40967",
                    "term": "WS 26/27",
                    "area": "Elective",
                }
            ],
        )
        return "I recommend Reinforcement Learning."

    from crew.tools.proposal_tools import collect_course_proposals

    flow = _flow(runner_overrides={"recommendation": recommendation_runner})
    with collect_course_proposals():
        flow.kickoff(
            inputs=StudyChatFlowState(
                query="What courses would you suggest for next semester?",
                profile_slug="primary",
            ).model_dump(mode="json")
        )

    assert flow.state.intent.route == "recommendation"
    assert flow.state.proposed_actions[0].title == "ML-heavy next semester"
    assert flow.state.proposed_actions[0].actions[0].kind == "grade_manager_add"


def test_follow_up_replacement_keeps_recommendation_route(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Initial plan",
        proposal_summary="Initial courses.",
        courses=[
            ProposalCourseInput(
                course_title="Distributed Systems",
                rationale="Fits elective room.",
                module_query="40001",
                term="WS 26/27",
            )
        ],
    )
    append_turn(
        "primary",
        user_content="Suggest courses.",
        assistant_content="Here is a plan.",
        proposals=[proposal],
        rolling_summary="Planning next semester.",
    )

    flow = _flow(runner_overrides={"recommendation": lambda flow: "I will replace it with an ML course."})
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Replace that course with more ML courses.",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert flow.state.intent.route == "recommendation"


def test_confirmed_grade_manager_action_passes_exact_confirmation_token(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Confirmed plan",
        proposal_summary="One course.",
        courses=[
            ProposalCourseInput(
                course_title="Machine Learning 2",
                rationale="Fits ML preference.",
                module_query="40967",
                term="WS 26/27",
            )
        ],
    )
    append_turn(
        "primary",
        user_content="Suggest courses.",
        assistant_content="Please confirm.",
        proposals=[proposal],
        rolling_summary="Planning next semester.",
    )
    action_id = proposal.actions[0].action_id
    seen = {}

    import crew.study_chat_flow as flow_module

    def fake_add(**kwargs):
        seen.update(kwargs)
        return "Added `Machine Learning 2` (6 LP) to profile `Primary Test Student`."

    monkeypatch.setattr(flow_module, "add_module_to_study_plan", fake_add)
    monkeypatch.setattr(flow_module, "check_module_against_study_plan", lambda **kwargs: "Already in study plan: yes")

    flow = _flow()
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="The user approved the plan.",
            profile_slug="primary",
            approved_actions=[ActionDecision(action_id=action_id, approved=True)],
        ).model_dump(mode="json")
    )

    assert seen["confirmation_token"] == STUDY_PLAN_CONFIRMATION_TOKEN
    assert flow.state.executed_actions[0].status == "executed"
