from __future__ import annotations

import json
from types import SimpleNamespace

from core import persistence
from core.models import Module, ModuleState, MosesIsisCandidate, MosesModuleData
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


def _study_module(
    *,
    module_id: str,
    name: str,
    state: ModuleState = ModuleState.PLANNED,
    term: str = "SS 26",
    moses_number: str | None = None,
    moses_version: int | None = None,
    isis_course_id: int | None = None,
    semester_span: int = 1,
) -> Module:
    moses = None
    if moses_number and moses_version is not None:
        candidates = []
        if isis_course_id is not None:
            candidates.append(
                MosesIsisCandidate(
                    course_id=isis_course_id,
                    course_url=f"https://isis.tu-berlin.de/course/view.php?id={isis_course_id}",
                    course_title=f"[SoSe 2026] {name}",
                    term_hint="SoSe 2026" if term == "SS 26" else term,
                    module_title=name,
                    fallback_search_terms=[name],
                    confidence="high",
                    status="resolved",
                )
            )
        moses = MosesModuleData(
            number=moses_number,
            version=moses_version,
            title=name,
            credits=6,
            isis_candidates=candidates,
        )
    return Module(
        id=module_id,
        name=name,
        state=state,
        program_key="TU Berlin - Technische Informatik (B.Sc.)",
        cp=12 if "Analysis" in name else 6,
        area="Mandatory",
        is_graded=True,
        term=term,
        semester_span=semester_span,
        moses_number=moses_number,
        moses_version=moses_version,
        moses=moses,
    )


def _capture_isis_crew_inputs(monkeypatch):
    import crew.isis_crew as isis_crew_module

    captured = {}

    class FakeIsisCourseInfoCrew:
        def __init__(self, **kwargs):
            captured["crew_kwargs"] = kwargs

        def crew(self):
            return self

        def kickoff(self, inputs):
            captured["inputs"] = inputs
            return SimpleNamespace(raw="Scoped ISIS answer.")

    monkeypatch.setattr(isis_crew_module, "IsisCourseInfoCrew", FakeIsisCourseInfoCrew)
    return captured


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


def test_simple_grade_optimization_routes_to_grade_optimization_crew(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    import crew.grade_optimization_crew as grade_optimization_module

    captured = {}

    class FakeGradeOptimizationCrew:
        def __init__(self, **kwargs):
            captured["crew_kwargs"] = kwargs

        def crew(self):
            return self

        def kickoff(self, inputs):
            captured["inputs"] = inputs
            return SimpleNamespace(raw="Target grade optimizer answer.")

    monkeypatch.setattr(grade_optimization_module, "GradeOptimizationCrew", FakeGradeOptimizationCrew)

    flow = _flow(
        classifier=_classifier_for(
            "simple_grade_optimization",
            required_sources=["grade_optimization"],
        ),
        runner_overrides={
            "simple_grade_manager": lambda flow: (_ for _ in ()).throw(AssertionError("wrong route")),
            "simple_moses": lambda flow: (_ for _ in ()).throw(AssertionError("wrong route")),
            "simple_isis": lambda flow: (_ for _ in ()).throw(AssertionError("wrong route")),
            "deep_dive": lambda flow: (_ for _ in ()).throw(AssertionError("wrong route")),
            "recommendation": lambda flow: (_ for _ in ()).throw(AssertionError("wrong route")),
        },
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Can I still reach a 1.7 final grade?",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    assert flow.state.intent.route == "simple_grade_optimization"
    assert captured["inputs"]["student_context"] == "No student context supplied."
    assert "Can I still reach a 1.7 final grade?" in captured["inputs"]["query"]
    assert load_chat_thread("primary").messages[-1].content == "Target grade optimizer answer."


def test_llm_classifier_prompt_includes_grade_optimization_route_and_boundary(monkeypatch):
    import crew.study_chat_flow as flow_module

    captured = {}

    class FakeLlm:
        def call(self, messages, response_model):
            captured["system"] = messages[0]["content"]
            return IntentClassification(
                route="simple_grade_optimization",
                required_sources=["grade_optimization"],
                rationale="single-source grade optimizer question",
            )

    monkeypatch.setenv("GWDG_API_KEY", "test-key")
    monkeypatch.setattr(flow_module, "get_default_llm", lambda **kwargs: FakeLlm())

    flow = StudyChatFlow(
        runtime=StudyChatFlowRuntime(
            use_llm_classifier=True,
            use_llm_decision_interpreter=False,
        )
    )
    result = flow._classify_with_llm_or_heuristics(
        StudyChatFlowState(query="Can I reach a 1.7 final grade?")
    )

    assert result.route == "simple_grade_optimization"
    assert "simple_grade_optimization" in captured["system"]
    assert "target grade optimizer" in captured["system"]
    assert "Do not use simple_grade_optimization" in captured["system"]
    assert "choose modules" in captured["system"]


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


def test_simple_isis_defaults_to_current_grade_manager_scope(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    persistence.save_modules(
        [
            _study_module(
                module_id="analysis",
                name="Analysis I und Lineare Algebra für Ingenieurwissenschaften",
                term="SS 26",
                moses_number="20122",
                moses_version=4,
                isis_course_id=46899,
            ),
            _study_module(
                module_id="future",
                name="Machine Learning and Security",
                term="WS 26/27",
                moses_number="41103",
                moses_version=1,
                isis_course_id=48001,
            ),
            _study_module(
                module_id="completed",
                name="Completed Old Course",
                state=ModuleState.COMPLETED,
                term="SS 26",
                moses_number="11111",
                moses_version=1,
                isis_course_id=47000,
            ),
        ],
        "primary",
    )
    captured = _capture_isis_crew_inputs(monkeypatch)

    flow = _flow(classifier=_classifier_for("simple_isis", required_sources=["isis"]))
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Was steht dieses Semester in meinen aktuellen Kursen an?",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    inputs = captured["inputs"]
    context = json.loads(inputs["isis_context"])
    scope = context["grade_manager_course_scope"]
    assert flow.state.intent.required_sources == ["grade_manager", "isis"]
    assert scope["mode"] == "llm_decides_with_grade_manager_default"
    assert scope["requested_term"] == "SS 26"
    assert [module["name"] for module in scope["scope_modules"]] == [
        "Analysis I und Lineare Algebra für Ingenieurwissenschaften"
    ]
    assert [candidate["course_id"] for candidate in context["preferred_course_candidates"]] == [46899]
    assert "Machine Learning and Security" not in inputs["student_context"]
    assert "Completed Old Course" not in inputs["student_context"]


def test_simple_isis_uses_explicit_semester_for_grade_manager_scope(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    persistence.save_modules(
        [
            _study_module(
                module_id="summer",
                name="Summer Course",
                term="SS 26",
                moses_number="20122",
                moses_version=4,
                isis_course_id=46899,
            ),
            _study_module(
                module_id="winter",
                name="Winter Course",
                term="WS 26/27",
                moses_number="41103",
                moses_version=1,
                isis_course_id=48001,
            ),
        ],
        "primary",
    )
    captured = _capture_isis_crew_inputs(monkeypatch)

    flow = _flow(classifier=_classifier_for("simple_isis", required_sources=["isis"]))
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Welche ISIS-Deadlines gibt es im WS 26/27?",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    context = json.loads(captured["inputs"]["isis_context"])
    scope = context["grade_manager_course_scope"]
    assert scope["requested_term"] == "WS 26/27"
    assert scope["requested_term_source"] == "explicit"
    assert [module["name"] for module in scope["scope_modules"]] == ["Winter Course"]
    assert [candidate["course_id"] for candidate in context["preferred_course_candidates"]] == [48001]


def test_simple_isis_direct_course_requests_leave_decision_to_isis_agent(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    persistence.save_modules(
        [
            _study_module(
                module_id="analysis",
                name="Analysis I und Lineare Algebra für Ingenieurwissenschaften",
                term="SS 26",
                moses_number="20122",
                moses_version=4,
                isis_course_id=46899,
            ),
        ],
        "primary",
    )
    captured = _capture_isis_crew_inputs(monkeypatch)

    flow = _flow(classifier=_classifier_for("simple_isis", required_sources=["isis"]))
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Was steht im ISIS-Kurs Quality and Usability an? ISIS course ID 47235",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    context = json.loads(captured["inputs"]["isis_context"])
    scope = context["grade_manager_course_scope"]
    assert scope["mode"] == "llm_decides_with_grade_manager_default"
    assert "direct_isis_course" in scope["override_policy"]
    assert "outside the Grade Manager scope" in scope["override_policy"]["direct_isis_course"]
    assert [candidate["course_id"] for candidate in context["preferred_course_candidates"]] == [46899]
    assert "ISIS course ID 47235" in captured["inputs"]["query"]


def test_simple_isis_all_isis_courses_request_leaves_broader_scope_decision_to_isis_agent(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    persistence.save_modules(
        [
            _study_module(
                module_id="analysis",
                name="Analysis I und Lineare Algebra für Ingenieurwissenschaften",
                term="SS 26",
                moses_number="20122",
                moses_version=4,
                isis_course_id=46899,
            ),
        ],
        "primary",
    )
    captured = _capture_isis_crew_inputs(monkeypatch)

    flow = _flow(classifier=_classifier_for("simple_isis", required_sources=["isis"]))
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Zeig mir alle aktuellen ISIS-Kurse dieses Semester.",
            profile_slug="primary",
        ).model_dump(mode="json")
    )

    context = json.loads(captured["inputs"]["isis_context"])
    scope = context["grade_manager_course_scope"]
    assert scope["mode"] == "llm_decides_with_grade_manager_default"
    assert scope["requested_term"] == "SS 26"
    assert "all_isis_courses_for_term" in scope["override_policy"]
    assert "all ISIS/Moodle courses" in scope["override_policy"]["all_isis_courses_for_term"]
    assert [candidate["course_id"] for candidate in context["preferred_course_candidates"]] == [46899]
    assert "alle aktuellen ISIS-Kurse" in captured["inputs"]["query"]


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
    assert [action.kind for action in flow.state.proposed_actions[0].actions] == ["grade_manager_add"]


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
                include_isis=False,
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
    from crew.write_permissions import confirmed_writes_enabled, allow_confirmed_writes

    def fake_add(**kwargs):
        assert confirmed_writes_enabled()
        seen.update(kwargs)
        return "Added `Machine Learning 2` (6 LP) to profile `Primary Test Student`."

    monkeypatch.setattr(flow_module, "add_module_to_study_plan", fake_add)
    monkeypatch.setattr(flow_module, "check_module_against_study_plan", lambda **kwargs: "Already in study plan: yes")
    monkeypatch.setattr(flow_module, "_verify_grade_manager_addition", lambda action: ("executed", "Added."))

    def recommendation_runner(flow):
        with allow_confirmed_writes():
            flow_module.add_module_to_study_plan(
                module_query="40967",
                term="WS 26/27",
                confirmation_token=STUDY_PLAN_CONFIRMATION_TOKEN,
            )
        return "Add course."

    flow = _flow(runner_overrides={"recommendation": recommendation_runner})
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="The user approved the plan.",
            profile_slug="primary",
            approved_actions=[ActionDecision(action_id=action_id, approved=True)],
        ).model_dump(mode="json")
    )

    assert seen["confirmation_token"] == STUDY_PLAN_CONFIRMATION_TOKEN
    assert flow.state.route == "recommendation"
    assert flow.state.executed_actions[0].status == "executed"


def test_approved_actions_execute_deterministically_without_crew_manifest(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Confirmed plan",
        proposal_summary="One course.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Fits the student's security focus.",
                module_query="41240",
                term="WS 26/27",
                area="Elective",
                verified_isis_course_id=48474,
                include_grade_manager=True,
                include_isis=True,
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

    import crew.multi_agent_crew as multi_agent_module
    import crew.study_chat_flow as flow_module

    class FakeMultiAgentStudyAssistantCrew:
        def __init__(self, **kwargs):
            raise AssertionError("crew should not run for a pure apply-selected turn")

    monkeypatch.setattr(multi_agent_module, "MultiAgentStudyAssistantCrew", FakeMultiAgentStudyAssistantCrew)
    monkeypatch.setattr(
        flow_module,
        "_execute_grade_manager_action",
        lambda action: action.model_copy(update={"status": "executed", "result": "Study plan updated."}),
    )
    monkeypatch.setattr(
        flow_module,
        "_execute_isis_action",
        lambda action: action.model_copy(update={"status": "executed", "result": "ISIS enrollment updated."}),
    )

    decisions = [ActionDecision(action_id=action.action_id, approved=True) for action in proposal.actions]
    flow = _flow()
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Perfect",
            profile_slug="primary",
            approved_actions=decisions,
        ).model_dump(mode="json")
    )

    assert [action.status for action in flow.state.executed_actions] == ["executed", "executed"]
    assert "Confirmed Actions" in flow.state.answer_markdown
    assert load_chat_thread("primary").active_proposals == []


def test_unavailable_isis_is_omitted_before_confirmation(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    import crew.tools.proposal_tools as proposal_module

    monkeypatch.setattr(
        proposal_module,
        "_resolve_isis_for_proposal",
        lambda payload: proposal_module._IsisPreflightResult(
            status="not_found",
            reason="No matching ISIS course was found by enrolled-course lookup or global ISIS search.",
        ),
    )
    proposal = build_course_proposal(
        proposal_title="Confirmed bundled plan",
        proposal_summary="One course.",
        courses=[
            ProposalCourseInput(
                course_title="Future Mandatory Course",
                rationale="Fits the planned semester.",
                module_query="41111",
                term="WS 26/27",
                area="Mandatory",
                include_grade_manager=True,
                include_isis=True,
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
        lambda action: action.model_copy(update={"status": "executed", "result": "Added to Study Manager."}),
    )
    monkeypatch.setattr(
        flow_module,
        "_execute_isis_action",
        lambda action: action.model_copy(
            update={
                "status": "failed",
                "result": "No matching ISIS course was found by enrolled-course lookup or global ISIS search.",
            }
        ),
    )

    decisions = [ActionDecision(action_id=action.action_id, approved=True) for action in proposal.actions]
    flow = _flow()
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Add them",
            profile_slug="primary",
            approved_actions=decisions,
        ).model_dump(mode="json")
    )

    assert [(action.kind, action.status) for action in flow.state.executed_actions] == [
        ("grade_manager_add", "executed"),
    ]
    assert "Future Mandatory Course" in flow.state.answer_markdown
    assert "Study Manager add" in flow.state.answer_markdown
    assert load_chat_thread("primary").active_proposals == []


def test_bundled_confirmation_clears_accepted_card_when_isis_needs_clarification(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Confirmed bundled plan",
        proposal_summary="One course.",
        courses=[
            ProposalCourseInput(
                course_title="Rechnerorganisation",
                rationale="Mandatory module.",
                module_query="40019",
                term="WS 26/27",
                area="Mandatory",
                verified_isis_course_id=45172,
                include_grade_manager=True,
                include_isis=True,
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
        lambda action: action.model_copy(update={"status": "executed", "result": "Added to Study Manager."}),
    )
    monkeypatch.setattr(
        flow_module,
        "_execute_isis_action",
        lambda action: action.model_copy(
            update={
                "status": "needs_clarification",
                "result": (
                    "No matching ISIS course was found by enrolled-course lookup or global ISIS search.\n\n"
                    "Candidate ISIS courses:\n"
                    "- `45172` [WiSe 2025/26] Rechnerorganisation"
                ),
            }
        ),
    )

    decisions = [ActionDecision(action_id=action.action_id, approved=True) for action in proposal.actions]
    flow = _flow()
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Add it",
            profile_slug="primary",
            approved_actions=decisions,
        ).model_dump(mode="json")
    )

    assert "Candidate ISIS courses" in flow.state.answer_markdown
    assert "`45172`" in flow.state.answer_markdown
    assert load_chat_thread("primary").active_proposals == []


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
                include_isis=False,
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

    flow = _flow(runner_overrides={"recommendation": lambda f: "Add course."})
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Klingt gut!",
            profile_slug="primary",
            ui_decisions=[ActionDecision(action_id=proposal.actions[0].action_id, approved=True)],
        ).model_dump(mode="json")
    )

    assert flow.state.route == "recommendation"
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
                include_isis=False,
            ),
            ProposalCourseInput(
                course_title="Course B",
                rationale="Replace this.",
                module_query="22222",
                term="WS 26/27",
                include_isis=False,
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

    assert flow.state.route == "recommendation"
    assert calls == ["recommendation"]
    assert [action.course_title for action in flow.state.executed_actions] == ["Course A"]
    assert "Replacement proposed." in flow.state.answer_markdown


def test_bundled_partial_apply_removes_declined_course_and_keeps_new_suggestion(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Initial bundled plan",
        proposal_summary="Two bundled course commitments.",
        courses=[
            ProposalCourseInput(
                course_title="Course A",
                rationale="Keep this.",
                module_query="11111",
                term="WS 26/27",
                verified_isis_course_id=10111,
            ),
            ProposalCourseInput(
                course_title="Course B",
                rationale="Remove this.",
                module_query="22222",
                term="WS 26/27",
                verified_isis_course_id=10222,
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

    import crew.study_chat_flow as flow_module

    monkeypatch.setattr(
        flow_module,
        "_execute_grade_manager_action",
        lambda action: action.model_copy(update={"status": "executed", "result": f"Added {action.course_title}."}),
    )
    monkeypatch.setattr(
        flow_module,
        "_execute_isis_action",
        lambda action: action.model_copy(update={"status": "executed", "result": f"Enrolled {action.course_title}."}),
    )

    new_proposal = build_course_proposal(
        proposal_title="Replacement bundled plan",
        proposal_summary="One replacement course.",
        courses=[
            ProposalCourseInput(
                course_title="Course C",
                rationale="Replacement for Course B.",
                module_query="33333",
                term="WS 26/27",
                verified_isis_course_id=10333,
            )
        ],
    )
    monkeypatch.setattr(flow_module, "current_course_proposals", lambda: [new_proposal])

    course_a_ids = [action.action_id for action in proposal.actions if action.course_title == "Course A"]
    course_b_ids = [action.action_id for action in proposal.actions if action.course_title == "Course B"]
    ui_decisions = [
        *[ActionDecision(action_id=action_id, approved=True) for action_id in course_a_ids],
        *[ActionDecision(action_id=action_id, approved=False) for action_id in course_b_ids],
    ]

    flow = _flow(runner_overrides={"recommendation": lambda flow: "Replacement proposed."})
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Add Course A but remove Course B and suggest a replacement.",
            profile_slug="primary",
            ui_decisions=ui_decisions,
        ).model_dump(mode="json")
    )

    assert [action.course_title for action in flow.state.executed_actions] == ["Course A", "Course A"]
    assert sorted(action.kind for action in flow.state.executed_actions) == ["grade_manager_add", "isis_enroll"]

    thread = load_chat_thread("primary")
    active_titles = {action.course_title for proposal in thread.active_proposals for action in proposal.actions}
    assert active_titles == {"Course C"}
    assert [action.kind for action in thread.active_proposals[0].actions] == ["grade_manager_add", "isis_enroll"]


def test_conflicting_clarification_flag_does_not_block_clear_partial_revision(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Semester 1 Pflicht Modules",
        proposal_summary="Initial mandatory modules.",
        courses=[
            ProposalCourseInput(
                course_title="Rechnerorganisation",
                rationale="Accepted module.",
                module_query="40019",
                term="WS 26/27",
                verified_isis_course_id=45119,
            ),
            ProposalCourseInput(
                course_title="Rechnerorganisation Praktikum",
                rationale="Student wants to defer this.",
                module_query="40028",
                term="WS 26/27",
                verified_isis_course_id=45128,
            ),
        ],
    )
    append_turn(
        "primary",
        user_content="Plane mal mein erstes Semester.",
        assistant_content="Please confirm.",
        proposals=[proposal],
        rolling_summary="Planning next semester.",
    )

    accepted_ids = [action.action_id for action in proposal.actions if action.course_title == "Rechnerorganisation"]
    rejected_ids = [action.action_id for action in proposal.actions if action.course_title == "Rechnerorganisation Praktikum"]
    ui_decisions = [
        *[ActionDecision(action_id=action_id, approved=True) for action_id in accepted_ids],
        *[ActionDecision(action_id=action_id, approved=False) for action_id in rejected_ids],
    ]

    import crew.study_chat_flow as flow_module

    monkeypatch.setattr(
        flow_module,
        "_execute_grade_manager_action",
        lambda action: action.model_copy(update={"status": "executed", "result": "Added."}),
    )
    monkeypatch.setattr(
        flow_module,
        "_execute_isis_action",
        lambda action: action.model_copy(update={"status": "executed", "result": "Enrolled."}),
    )

    calls = []
    flow = _flow(
        decision_interpreter=lambda state: UserDecisionInterpretation(
            intent="revise_only",
            approved_action_ids=accepted_ids,
            rejected_action_ids=rejected_ids,
            revision_request="Rechnerorganisation Praktikum später machen und ein anderes Modul im Wintersemester finden.",
            needs_user_clarification=True,
            rationale=(
                "User rejected the Praktikum cards and asks for an alternative module. "
                "Approved modules should be executed, but replacement options are needed."
            ),
        ),
        runner_overrides={"recommendation": lambda flow: calls.append("recommendation") or "I will search alternatives."},
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Ich möchte Rechnerorganisation Praktikum später machen. Kann ich stattdessen ein anderes Modul jetzt im Wintersemester machen?",
            profile_slug="primary",
            ui_decisions=ui_decisions,
        ).model_dump(mode="json")
    )

    assert flow.state.route == "recommendation"
    assert flow.state.decision_interpretation.intent == "apply_partial_and_revise"
    assert flow.state.decision_interpretation.needs_user_clarification is False
    assert calls == ["recommendation"]
    assert sorted(action.kind for action in flow.state.executed_actions) == ["grade_manager_add", "isis_enroll"]


def test_disabled_isis_toggle_executes_study_plan_only_and_clears_disabled_action(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Study-plan-only from toggle",
        proposal_summary="One bundled course.",
        courses=[
            ProposalCourseInput(
                course_title="Course A",
                rationale="Keep only in study plan.",
                module_query="11111",
                term="WS 26/27",
                verified_isis_course_id=10111,
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
    monkeypatch.setattr(
        flow_module,
        "_execute_isis_action",
        lambda action: (_ for _ in ()).throw(AssertionError("disabled ISIS action was executed")),
    )

    grade_action = next(action for action in proposal.actions if action.kind == "grade_manager_add")
    isis_action = next(action for action in proposal.actions if action.kind == "isis_enroll")

    flow = _flow(runner_overrides={"recommendation": lambda flow: "Study plan updated."})
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Perfect",
            profile_slug="primary",
            ui_decisions=[
                ActionDecision(action_id=grade_action.action_id, approved=True),
                ActionDecision(action_id=isis_action.action_id, approved=False, feedback="Course card action disabled by user."),
            ],
        ).model_dump(mode="json")
    )

    assert [(action.kind, action.status) for action in flow.state.executed_actions] == [("grade_manager_add", "executed")]
    assert load_chat_thread("primary").active_proposals == []


def test_approved_grade_manager_remove_executes_through_commitment_guard(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    persistence.save_modules(
        [
            _study_module(
                module_id="analysis",
                name="Analysis I und Lineare Algebra für Ingenieurwissenschaften",
                term="SS 26",
                moses_number="20122",
                moses_version=4,
            ),
        ],
        "primary",
    )
    proposal = build_course_proposal(
        proposal_title="Remove missed current course",
        proposal_summary="Remove a planned module that the student no longer wants to pursue.",
        courses=[
            ProposalCourseInput(
                course_title="Analysis I und Lineare Algebra für Ingenieurwissenschaften",
                rationale="The student asked the Commitment Specialist to remove this planned module.",
                module_query="20122",
                version=4,
                term="SS 26",
                area="Mandatory",
                program_key="TU Berlin - Technische Informatik (B.Sc.)",
                grade_manager_action="remove",
                include_isis=False,
            )
        ],
    )
    append_turn(
        "primary",
        user_content="Entferne Analysis I aus meinem Study Manager.",
        assistant_content="Please confirm the removal card.",
        proposals=[proposal],
        rolling_summary="A Study Manager removal proposal is waiting for UI approval.",
    )

    remove_action = proposal.actions[0]
    assert remove_action.kind == "grade_manager_remove"

    flow = _flow(
        runner_overrides={
            "recommendation": lambda flow: (_ for _ in ()).throw(AssertionError("crew should not run"))
        }
    )
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Ja, entfernen.",
            profile_slug="primary",
            ui_decisions=[ActionDecision(action_id=remove_action.action_id, approved=True)],
        ).model_dump(mode="json")
    )

    assert [(action.course_title, action.kind, action.status) for action in flow.state.executed_actions] == [
        ("Analysis I und Lineare Algebra für Ingenieurwissenschaften", "grade_manager_remove", "executed")
    ]
    assert persistence.load_modules("primary") == []
    assert load_chat_thread("primary").active_proposals == []


def test_approved_grade_manager_update_executes_deterministically(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    update_action = ProposedAction(
        action_id="gm-update-ana2",
        kind="grade_manager_update",
        course_title="Analysis II für Ingenieurwissenschaften",
        grade_manager_payload={
            "module_query": "20130",
            "version": 4,
            "program_key": "TU Berlin - Technische Informatik (B.Sc.)",
            "current_term": "WS 26/27",
            "target_term": "SS 26",
            "target_area": "Mandatory",
        },
    )
    proposal = build_course_proposal(
        proposal_title="placeholder",
        proposal_summary="placeholder",
        courses=[
            ProposalCourseInput(
                course_title="Placeholder",
                rationale="placeholder",
                module_query="99999",
                term="WS 26/27",
                include_isis=False,
            )
        ],
    ).model_copy(update={"proposal_id": "move-ana2", "title": "Move Ana2", "actions": [update_action]}, deep=True)
    append_turn(
        "primary",
        user_content="Ana2 should be in summer.",
        assistant_content="Please confirm.",
        proposals=[proposal],
        rolling_summary="Planning next semester.",
    )

    import crew.study_chat_flow as flow_module

    monkeypatch.setattr(
        flow_module,
        "_execute_grade_manager_update_action",
        lambda action: action.model_copy(update={"status": "executed", "result": "Updated Analysis II."}),
    )

    flow = _flow(runner_overrides={"recommendation": lambda flow: (_ for _ in ()).throw(AssertionError("crew should not run"))})
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Ja",
            profile_slug="primary",
            ui_decisions=[ActionDecision(action_id=update_action.action_id, approved=True)],
        ).model_dump(mode="json")
    )

    assert [(action.kind, action.status) for action in flow.state.executed_actions] == [("grade_manager_update", "executed")]
    assert load_chat_thread("primary").active_proposals == []


def test_analysis_two_update_flow_moves_existing_module_to_summer(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    persistence.save_modules(
        [
            Module(
                id="ana2",
                name="Analysis II für Ingenieurwissenschaften",
                state=ModuleState.PLANNED,
                program_key="TU Berlin - Technische Informatik (B.Sc.)",
                cp=9,
                area="Mandatory",
                is_graded=True,
                term="WS 26/27",
                moses_number="20130",
                moses_version=4,
            )
        ],
        "primary",
    )
    update_action = ProposedAction(
        action_id="gm-update-ana2",
        kind="grade_manager_update",
        course_title="Analysis II für Ingenieurwissenschaften",
        grade_manager_payload={
            "module_query": "20130",
            "version": 4,
            "program_key": "tech_informatik_bsc",
            "current_term": "WS 26/27",
            "target_term": "SS 26",
            "target_area": "Core Module",
        },
    )
    proposal = build_course_proposal(
        proposal_title="placeholder",
        proposal_summary="placeholder",
        courses=[
            ProposalCourseInput(
                course_title="Placeholder",
                rationale="placeholder",
                module_query="99999",
                term="WS 26/27",
                include_isis=False,
            )
        ],
    ).model_copy(update={"proposal_id": "move-ana2", "title": "Move Ana2", "actions": [update_action]}, deep=True)
    append_turn(
        "primary",
        user_content="Ana2 should be in summer.",
        assistant_content="Please confirm.",
        proposals=[proposal],
        rolling_summary="Planning next semester.",
    )

    import crew.study_chat_flow as flow_module

    monkeypatch.setattr(flow_module, "check_module_against_study_plan", lambda **kwargs: "Verified in test profile.")

    flow = _flow()
    flow.kickoff(
        inputs=StudyChatFlowState(
            query="Ja",
            profile_slug="primary",
            ui_decisions=[ActionDecision(action_id=update_action.action_id, approved=True)],
        ).model_dump(mode="json")
    )

    modules = persistence.load_modules("primary")
    assert [(module.name, module.term, module.area, module.program_key) for module in modules] == [
        (
            "Analysis II für Ingenieurwissenschaften",
            "SS 26",
            "Mandatory",
            "TU Berlin - Technische Informatik (B.Sc.)",
        )
    ]
    assert [(action.kind, action.status) for action in flow.state.executed_actions] == [("grade_manager_update", "executed")]
    assert load_chat_thread("primary").active_proposals == []


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


def test_isis_action_status_only_keeps_recoverable_no_match_active():
    from types import SimpleNamespace
    from crew.isis_models import IsisCourseRef
    import crew.study_chat_flow as flow_module

    assert flow_module._isis_action_status_from_outcome(
        SimpleNamespace(ok=False, status="not_found", candidates=[])
    ) == "failed"
    assert flow_module._isis_action_status_from_outcome(
        SimpleNamespace(
            ok=False,
            status="not_found",
            candidates=[IsisCourseRef(id=48001, fullname="[WiSe 2026/27] Course")],
        )
    ) == "needs_clarification"
    assert flow_module._isis_action_status_from_outcome(
        SimpleNamespace(ok=False, status="ambiguous_course", candidates=[])
    ) == "needs_clarification"


def test_isis_resolve_verification_accepts_unambiguous_enrollment(monkeypatch):
    from crew.isis_models import IsisCourseRef
    import crew.isis_client as isis_client_module
    import crew.study_chat_flow as flow_module

    course = IsisCourseRef(
        id=48474,
        fullname="[WiSe 2026/27] Software Security Lab",
        term_hint="WiSe 2026/27",
        enrolled=True,
    )

    class FakeClient:
        def enrolled_course_refs(self):
            return [course]

        def course_ref_by_id(self, course_id):
            return course if course_id == course.id else None

        def search_course_refs(self, query, perpage=20):
            return []

    monkeypatch.setattr(isis_client_module, "get_default_isis_client", lambda: FakeClient())

    action = ProposedAction(
        action_id="isis-resolve",
        kind="isis_resolve",
        course_title="Software Security Lab",
        isis_payload={
            "course_query": "Software Security Lab",
            "term_hint": "WiSe 2026/27",
            "expected_title": "Software Security Lab",
            "isis_resolution_status": "unresolved",
        },
    )

    status, result = flow_module._verify_isis_commitment(action)

    assert status == "executed"
    assert "Software Security Lab" in result
    assert action.isis_payload["course_id"] == 48474


def test_isis_resolve_verification_reports_ambiguous_candidates(monkeypatch):
    from crew.isis_models import IsisCourseRef
    import crew.isis_client as isis_client_module
    import crew.study_chat_flow as flow_module

    candidates = [
        IsisCourseRef(id=48001, fullname="[WiSe 2026/27] Parallel Programming", term_hint="WiSe 2026/27"),
        IsisCourseRef(id=48002, fullname="[WiSe 2026/27] Parallel Programming", term_hint="WiSe 2026/27"),
    ]

    class FakeClient:
        def enrolled_course_refs(self):
            return []

        def course_ref_by_id(self, course_id):
            return None

        def search_course_refs(self, query, perpage=20):
            return candidates

    monkeypatch.setattr(isis_client_module, "get_default_isis_client", lambda: FakeClient())

    action = ProposedAction(
        action_id="isis-ambiguous",
        kind="isis_resolve",
        course_title="Parallel Programming",
        isis_payload={
            "course_query": "Parallel Programming",
            "term_hint": "WiSe 2026/27",
            "expected_title": "Parallel Programming",
            "isis_resolution_status": "unresolved",
        },
    )

    status, result = flow_module._verify_isis_commitment(action)

    assert status == "needs_clarification"
    assert "Candidate ISIS courses" in result
    assert "48001" in result
    assert "48002" in result


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
                include_isis=False,
            ),
            ProposalCourseInput(
                course_title="Course B",
                rationale="Fits elective.",
                module_query="22222",
                term="WS 26/27",
                include_isis=False,
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
                include_isis=False,
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


def test_merge_replaces_stale_proposal_actions_by_course_identity(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    import crew.study_chat_flow as flow_module

    old_proposal = build_course_proposal(
        proposal_title="Old recommendations",
        proposal_summary="Old suggestion.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Initial recommendation.",
                module_query="41240",
                term="WS 26/27",
            )
        ],
    )
    new_proposal = build_course_proposal(
        proposal_title="Updated recommendations",
        proposal_summary="Updated suggestion.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Updated recommendation.",
                module_query="41240",
                term="WS 26/27",
                area="Elective",
            )
        ],
    )

    merged = flow_module._merge_proposals([old_proposal], [new_proposal])

    assert len(merged) == 1
    assert merged[0].title == "Updated recommendations"
    assert [action.kind for action in merged[0].actions] == ["grade_manager_add"]
    assert merged[0].actions[0].grade_manager_payload["area"] == "Elective"


def test_merge_replaces_isis_resolve_with_verified_enroll(monkeypatch, tmp_path):
    _setup_profile(monkeypatch, tmp_path)
    import crew.study_chat_flow as flow_module

    old_proposal = build_course_proposal(
        proposal_title="Old suggestion",
        proposal_summary="Old suggestion.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Initial recommendation.",
                module_query="41240",
                term="WS 26/27",
            )
        ],
    )
    new_proposal = build_course_proposal(
        proposal_title="Resolved ISIS",
        proposal_summary="Resolved suggestion.",
        courses=[
            ProposalCourseInput(
                course_title="Software Security Lab",
                rationale="Resolved ISIS candidate.",
                module_query="41240",
                term="WS 26/27",
                verified_isis_course_id=48474,
                isis_resolution_status="resolved",
            )
        ],
    )

    merged = flow_module._merge_proposals([old_proposal], [new_proposal])

    assert [action.kind for action in merged[0].actions] == ["grade_manager_add", "isis_enroll"]
    assert merged[0].actions[1].isis_payload["course_id"] == 48474


def test_merge_replaces_old_grade_manager_add_with_update_for_same_module():
    import crew.study_chat_flow as flow_module

    old_proposal = build_course_proposal(
        proposal_title="Old Analysis II",
        proposal_summary="Analysis II in winter.",
        courses=[
            ProposalCourseInput(
                course_title="Analysis II für Ingenieurwissenschaften",
                rationale="Old term.",
                module_query="20130",
                version=4,
                term="WS 26/27",
                include_isis=False,
            )
        ],
    )
    update_action = ProposedAction(
        action_id="gm-update-ana2",
        kind="grade_manager_update",
        course_title="Analysis II für Ingenieurwissenschaften",
        grade_manager_payload={
            "module_query": "20130",
            "moses_module_number": "20130",
            "version": 4,
            "current_term": "WS 26/27",
            "target_term": "SS 26",
            "program_key": "TU Berlin - Technische Informatik (B.Sc.)",
        },
    )
    new_proposal = old_proposal.model_copy(
        update={
            "proposal_id": "new-ana2",
            "title": "Move Analysis II",
            "actions": [update_action],
        },
        deep=True,
    )

    merged = flow_module._merge_proposals([old_proposal], [new_proposal])

    assert [action.kind for action in merged[0].actions] == ["grade_manager_update"]
    assert merged[0].actions[0].grade_manager_payload["target_term"] == "SS 26"


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
