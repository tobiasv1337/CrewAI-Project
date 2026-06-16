from __future__ import annotations

from core import persistence
from crew.chat_models import ChatMessage
from crew.chat_persistence import append_turn, clear_chat_thread, load_chat_thread, reset_chat_thread
from crew.tools.proposal_tools import ProposalCourseInput, build_course_proposal


def _setup_profiles(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "profiles"
    monkeypatch.setattr(persistence, "DATA_DIR", data_dir)
    monkeypatch.setattr(persistence, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(persistence, "PROFILES_FILE", profiles_dir / "profiles.json")
    monkeypatch.setattr(persistence, "_LEGACY_MODULES_FILE", data_dir / "modules.json")
    persistence.save_profiles(
        [persistence.ProfileRecord(slug="primary", display_name="Primary Test Student", is_primary=True)]
    )


def test_persistent_chat_thread_roundtrips_messages_and_proposals(monkeypatch, tmp_path):
    _setup_profiles(monkeypatch, tmp_path)
    proposal = build_course_proposal(
        proposal_title="Plan",
        proposal_summary="Recommended courses.",
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
        user_content="What should I take?",
        assistant_content="I suggest ML2.",
        proposals=[proposal],
        rolling_summary="Student is planning next semester.",
    )

    loaded = load_chat_thread("primary")
    assert [message.role for message in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[0].content == "What should I take?"
    assert loaded.rolling_summary == "Student is planning next semester."
    assert loaded.active_proposals[0].actions[0].course_title == "Machine Learning 2"


def test_reset_and_clear_chat_thread(monkeypatch, tmp_path):
    _setup_profiles(monkeypatch, tmp_path)
    thread = reset_chat_thread("primary", thread_id="fresh")
    thread.messages.append(ChatMessage(role="user", content="hello"))
    from crew.chat_persistence import save_chat_thread

    save_chat_thread(thread)
    assert load_chat_thread("primary", thread_id="fresh").messages

    clear_chat_thread("primary")
    assert load_chat_thread("primary").messages == []
