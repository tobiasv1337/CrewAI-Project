from __future__ import annotations

from core import persistence
from crew.chat_models import ActionDecision, IntentClassification, ProposedAction, StudyChatFlowState, UserDecisionInterpretation
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
    runtime = StudyChatFlowRuntime(use_llm_classifier=False, use_llm_decision_interpreter=False)
    return StudyChatFlow(runtime=runtime, **kwargs)


def _classifier_for(
    route: str,
    *,
    required_sources: list[str] | None = None,
    complexity: str = "simple",
    write_intent: bool = False,
):
    def classify(state):
        return IntentClassification(
            route=route,
            complexity=complexity,
            required_sources=required_sources or [],
            write_intent=write_intent,
            rationale=f"Test classifier for {route}.",
        )

    return classify


def test_simple_progress_question_routes_only_to_study_advisor(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    calls = []

    def grade_runner(flow):
        calls.append("grade")
        return "You have 60 LP completed."

    def forbidden(flow):
        raise AssertionError("wrong route")

    flow = _flow(
        classifier=_classifier_for("simple_grade_manager", required_sources=["grade_manager"]),
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


def test_non_llm_classifier_routes_to_deep_dive_with_warning(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    calls = []

    def deep_runner(flow):
        calls.append("deep")
        contextual_query = flow._contextual_query()
        assert "Current user message" in contextual_query
        assert "Semester reference context:" in contextual_query
        assert "Current semester:" in contextual_query
        assert "Next/upcoming semester:" in contextual_query
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
    assert flow.state.intent.required_sources == ["grade_manager", "moses", "isis"]
    assert "WARNING" in flow.state.intent.rationale


def test_degree_regulation_question_routes_to_simple_regulations(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    calls = []

    def regulation_runner(flow):
        calls.append("regulations")
        return "AllgStuPO answer with PDF citation."

    def forbidden(flow):
        raise AssertionError("wrong route")

    flow = _flow(
        classifier=_classifier_for(
            "simple_degree_regulations",
            required_sources=["degree_regulations"],
        ),
        runner_overrides={
            "simple_degree_regulations": regulation_runner,
            "simple_grade_manager": forbidden,
            "simple_moses": forbidden,
            "simple_isis": forbidden,
            "deep_dive": forbidden,
            "recommendation": forbidden,
        },
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Was sagt die AllgStuPO zu Wiederholungsprüfungen?",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert calls == ["regulations"]
    assert flow.state.intent.route == "simple_degree_regulations"
    assert flow.state.intent.required_sources == ["degree_regulations"]


def test_regelstudienplan_planning_can_use_recommendation_route(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    calls = []

    flow = _flow(
        classifier=_classifier_for(
            "recommendation",
            required_sources=["degree_regulations", "grade_manager", "moses"],
            complexity="scoped",
        ),
        runner_overrides={
            "recommendation": lambda flow: calls.append("recommendation") or "Planned with Regelstudienplan context.",
            "simple_degree_regulations": lambda flow: (_ for _ in ()).throw(AssertionError("wrong route")),
        },
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Plane mein nächstes Semester nach Regelstudienplan.",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert calls == ["recommendation"]
    assert flow.state.intent.route == "recommendation"
    assert "degree_regulations" in flow.state.intent.required_sources


def test_isis_enrollment_classifier_result_never_uses_simple_isis(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    calls = []

    def bad_classifier(state):
        return IntentClassification(
            route="simple_isis",
            complexity="simple",
            required_sources=["isis"],
            write_intent=True,
            rationale="Incorrectly chose the read-only ISIS route.",
        )

    def recommendation_runner(flow):
        calls.append("recommendation")
        return "Prepared an ISIS enrollment confirmation proposal."

    def forbidden(flow):
        raise AssertionError("write-intent ISIS request reached the read-only simple ISIS route")

    flow = _flow(
        classifier=bad_classifier,
        runner_overrides={
            "simple_isis": forbidden,
            "recommendation": recommendation_runner,
        },
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Enroll me in the Software Security Lab course on ISIS. Keep it simple.",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert calls == ["recommendation"]
    assert flow.state.intent.route == "recommendation"
    assert flow.state.intent.write_intent is True
    assert flow.state.intent.required_sources == ["isis"]


def test_non_llm_isis_enrollment_still_uses_deep_dive(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    calls = []

    flow = _flow(
        runner_overrides={
            "simple_isis": lambda flow: (_ for _ in ()).throw(AssertionError("wrong route")),
            "recommendation": lambda flow: (_ for _ in ()).throw(AssertionError("wrong route")),
            "deep_dive": lambda flow: calls.append("deep") or "Handled by full multi-agent crew.",
        }
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Please enroll me in Software Security Lab on ISIS.",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert calls == ["deep"]
    assert flow.state.intent.route == "deep_dive"
    assert flow.state.intent.required_sources == ["grade_manager", "moses", "isis"]
    assert "WARNING" in flow.state.intent.rationale


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

    flow = _flow(
        classifier=_classifier_for(
            "recommendation",
            required_sources=["grade_manager", "moses"],
            complexity="scoped",
        ),
        runner_overrides={"recommendation": recommendation_runner},
    )
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

    flow = _flow(
        classifier=_classifier_for(
            "recommendation",
            required_sources=["grade_manager", "moses"],
            complexity="scoped",
        ),
        runner_overrides={"recommendation": lambda flow: "I will replace it with an ML course."},
    )
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
    from crew.write_permissions import confirmed_writes_enabled

    def fake_add(**kwargs):
        assert confirmed_writes_enabled()
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


def test_natural_language_confirmation_uses_interpreter_not_phrase_list(monkeypatch, tmp_path):
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

    import crew.study_chat_flow as flow_module

    monkeypatch.setattr(
        flow_module,
        "_execute_grade_manager_action",
        lambda action: action.model_copy(update={"status": "executed", "result": "Added."}),
    )

    flow = _flow()
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Klingt gut!",
            profile_slug="primary",
            ui_decisions=[ActionDecision(action_id=proposal.actions[0].action_id, approved=True)],
        ).model_dump(mode="json")
    )

    assert flow.state.route == "execute_confirmed_actions"
    assert flow.state.decision_interpretation.intent == "apply_selected"
    assert flow.state.executed_actions[0].status == "executed"
    assert load_chat_thread("primary").active_proposals == []


def test_chat_confirmation_without_ui_acceptance_does_not_execute(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Displayed plan",
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

    import crew.study_chat_flow as flow_module

    monkeypatch.setattr(
        flow_module,
        "_execute_grade_manager_action",
        lambda action: (_ for _ in ()).throw(AssertionError("write executed without UI approval")),
    )

    flow = _flow(
        classifier=_classifier_for("recommendation", required_sources=["degree_regulations", "grade_manager", "moses"]),
        runner_overrides={"recommendation": lambda flow: "I still need the UI card approval before adding anything."},
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Klingt gut!",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert flow.state.executed_actions == []
    assert load_chat_thread("primary").active_proposals[0].actions[0].status == "proposed"


def test_partial_apply_and_revise_executes_approved_subset_then_runs_recommendation(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Initial plan",
        proposal_summary="Two courses.",
        courses=[
            ProposalCourseInput(
                course_title="Course A",
                rationale="Keep this.",
                module_query="11111",
                term="WS 26/27",
            ),
            ProposalCourseInput(
                course_title="Course B",
                rationale="Replace this.",
                module_query="22222",
                term="WS 26/27",
            ),
        ],
    )
    append_turn(
        "primary",
        user_content="Suggest courses.",
        assistant_content="Please confirm.",
        proposals=[proposal],
        rolling_summary="Planning next semester.",
    )

    calls = []
    import crew.study_chat_flow as flow_module

    monkeypatch.setattr(
        flow_module,
        "_execute_grade_manager_action",
        lambda action: action.model_copy(update={"status": "executed", "result": f"Added {action.course_title}."}),
    )

    flow = _flow(
        runner_overrides={"recommendation": lambda flow: calls.append("recommendation") or "Replacement proposed."}
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Add Course A but exchange Course B.",
            profile_slug="primary",
            ui_decisions=[
                ActionDecision(action_id=proposal.actions[0].action_id, approved=True),
                ActionDecision(action_id=proposal.actions[1].action_id, approved=False),
            ],
        ).model_dump(mode="json")
    )

    assert flow.state.route == "execute_then_recommendation"
    assert calls == ["recommendation"]
    assert [action.course_title for action in flow.state.executed_actions] == ["Course A"]
    assert "Replacement proposed." in flow.state.answer_markdown


def test_unrelated_new_question_discards_active_recommendations_and_routes_new_intent(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Initial plan",
        proposal_summary="One course.",
        courses=[
            ProposalCourseInput(
                course_title="Course A",
                rationale="Suggested.",
                module_query="11111",
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

    calls = []
    flow = _flow(
        classifier=_classifier_for("simple_moses", required_sources=["moses"]),
        runner_overrides={"simple_moses": lambda flow: calls.append("new-intent") or "New topic handled."},
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="What's the weather today?",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert calls == ["new-intent"]
    assert flow.state.decision_interpretation.discard_active_proposals is True
    assert load_chat_thread("primary").active_proposals == []
    assert flow.state.proposed_actions == []


def test_isis_enrollment_blocks_moses_id_used_as_isis_id():
    import crew.study_chat_flow as flow_module

    action = ProposedAction(
        action_id="isis-bad",
        kind="isis_enroll",
        course_title="Systemprogrammierung",
        isis_payload={
            "course_id": 40441,
            "moses_module_number": "40441",
            "course_query": "Systemprogrammierung",
            "expected_title": "Systemprogrammierung",
        },
    )

    result = flow_module._execute_isis_action(action)

    assert result.status == "needs_clarification"
    assert "MOSES module number" in result.result


def test_ui_decisions_update_action_status_and_merge_proposals(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Recommendations",
        proposal_summary="Suggesting courses.",
        courses=[
            ProposalCourseInput(
                course_title="Course A",
                rationale="Fits elective.",
                module_query="11111",
                term="WS 26/27",
            ),
            ProposalCourseInput(
                course_title="Course B",
                rationale="Fits elective.",
                module_query="22222",
                term="WS 26/27",
            ),
        ],
    )
    append_turn(
        "primary",
        user_content="Give me course options.",
        assistant_content="Here you go.",
        proposals=[proposal],
        rolling_summary="Planning.",
    )
    
    ui_decisions = [
        ActionDecision(action_id=proposal.actions[0].action_id, approved=True),
        ActionDecision(action_id=proposal.actions[1].action_id, approved=False),
    ]
    
    flow = _flow(
        classifier=_classifier_for("recommendation", required_sources=["grade_manager"]),
        decision_interpreter=lambda state: UserDecisionInterpretation(
            intent="revise_only",
            rejected_action_ids=[proposal.actions[1].action_id],
            revision_request="Replace Course B.",
            rationale="Test keeps accepted Course A active while revising Course B.",
        ),
        runner_overrides={"recommendation": lambda flow: "Answer"}
    )
    
    new_proposal = build_course_proposal(
        proposal_title="Recommendations",
        proposal_summary="New suggestion.",
        courses=[
            ProposalCourseInput(
                course_title="Course C",
                rationale="Replacement for Course B.",
                module_query="33333",
                term="WS 26/27",
            )
        ]
    )
    
    import crew.study_chat_flow as flow_module
    monkeypatch.setattr(flow_module, "current_course_proposals", lambda: [new_proposal])
    
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Replace Course B.",
            profile_slug="primary",
            ui_decisions=ui_decisions,
        ).model_dump(mode="json")
    )
    
    thread = load_chat_thread("primary")
    assert len(thread.active_proposals) == 1
    actions = thread.active_proposals[0].actions
    
    action_titles = {a.course_title for a in actions}
    assert "Course A" in action_titles
    assert "Course C" in action_titles
    assert "Course B" not in action_titles
    
    action_by_title = {a.course_title: a for a in actions}
    assert action_by_title["Course A"].status == "approved"
    assert action_by_title["Course C"].status == "proposed"


def test_isis_candidates_persisted_to_next_turn(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    from core.models import MosesModuleData, MosesIsisCandidate
    from crew.state import record_moses_module_artifact
    from crew.chat_persistence import load_chat_thread

    # Mock details fetch runner that records artifacts during kickoff
    def moses_runner(flow):
        data = MosesModuleData(
            number="40966",
            version=2,
            title="Machine Learning 1",
            credits=6.0,
            validity="WS 2024/25 onwards",
            responsible_person="Prof. Ada",
            grading_mode="Graded",
            exam_type="Written exam",
            teaching_languages=["English"],
            faculty="Faculty IV",
            institute="Institute of Software Engineering",
            department="Machine Learning",
            semester_count="1 Semester",
            start_semesters=["Winter semester"],
            module_elements=[],
            isis_candidates=[
                MosesIsisCandidate(
                    course_id=47025,
                    course_url="https://isis.tu-berlin.de/course/view.php?id=47025",
                    course_title="[SoSe 2026] Machine Learning 1",
                    term_hint="SoSe 2026",
                    module_title="Machine Learning 1",
                    fallback_search_terms=[],
                    confidence="high",
                    status="resolved",
                )
            ],
            workload_items=[],
            workload_total="180h",
            exam_elements=[],
            normalized_catalogs_by_program={},
        )
        record_moses_module_artifact(data)
        return "I found Machine Learning 1."

    flow = _flow(
        classifier=_classifier_for("simple_moses", required_sources=["moses"]),
        runner_overrides={"simple_moses": moses_runner},
    )

    from crew.state import collect_moses_state_artifacts
    with collect_moses_state_artifacts():
        flow.kickoff(
            inputs=StudyChatFlowState(
                query="Search ML1 on MOSES",
                profile_slug="primary",
            ).model_dump(mode="json")
        )

    # 1. Verify it was persisted to the thread's isis_context on disk
    thread = load_chat_thread("primary")
    assert thread.isis_context is not None
    assert len(thread.isis_context.preferred_course_candidates) == 1
    assert thread.isis_context.preferred_course_candidates[0].course_id == 47025

    # 2. Run next turn and check if isis_context_json is pre-populated
    flow2 = _flow(
        classifier=_classifier_for("simple_isis", required_sources=["isis"]),
        runner_overrides={"simple_isis": lambda f: "Enrollment done"}
    )
    flow2.kickoff(
        inputs=StudyChatFlowState(
            query="Now enroll me please",
            profile_slug="primary",
        ).model_dump(mode="json")
    )
    
    assert flow2.state.isis_context_json != "{}"
    import json
    context_data = json.loads(flow2.state.isis_context_json)
    assert "preferred_course_candidates" in context_data
    assert context_data["preferred_course_candidates"][0]["course_id"] == 47025

