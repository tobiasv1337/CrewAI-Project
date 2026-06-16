from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from core import persistence
from crew.chat_models import ChatMessage, ChatThreadState, CourseProposal, utc_now_iso


CHAT_FILENAME = "study_chat.json"


def chat_path(profile_slug: str) -> Path:
    return persistence.profile_dir(_clean_slug(profile_slug)) / CHAT_FILENAME


def load_chat_thread(profile_slug: str, *, thread_id: str = "default") -> ChatThreadState:
    path = chat_path(profile_slug)
    if not path.exists():
        return ChatThreadState(thread_id=thread_id, profile_slug=_clean_slug(profile_slug))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        thread = ChatThreadState.model_validate(data)
    except Exception as exc:
        import traceback
        print(f"Error loading chat thread for profile '{profile_slug}': {exc}")
        traceback.print_exc()
        return ChatThreadState(thread_id=thread_id, profile_slug=_clean_slug(profile_slug))
    if thread.thread_id != thread_id:
        thread.thread_id = thread_id
    if thread.profile_slug != _clean_slug(profile_slug):
        thread.profile_slug = _clean_slug(profile_slug)
    return thread


def save_chat_thread(thread: ChatThreadState) -> None:
    thread.updated_at = utc_now_iso()
    path = chat_path(thread.profile_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(thread.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if path.exists() and path.read_text(encoding="utf-8") == serialized:
        return
    path.write_text(serialized, encoding="utf-8")


def reset_chat_thread(profile_slug: str, *, thread_id: str | None = None) -> ChatThreadState:
    thread = ChatThreadState(
        thread_id=thread_id or f"thread-{uuid4().hex[:10]}",
        profile_slug=_clean_slug(profile_slug),
    )
    save_chat_thread(thread)
    return thread


def clear_chat_thread(profile_slug: str) -> None:
    path = chat_path(profile_slug)
    if path.exists():
        path.unlink()


def append_chat_message(profile_slug: str, message: ChatMessage, *, thread_id: str = "default") -> ChatThreadState:
    thread = load_chat_thread(profile_slug, thread_id=thread_id)
    thread.messages.append(message)
    save_chat_thread(thread)
    return thread


def replace_active_proposals(
    profile_slug: str,
    proposals: list[CourseProposal],
    *,
    thread_id: str = "default",
) -> ChatThreadState:
    thread = load_chat_thread(profile_slug, thread_id=thread_id)
    thread.active_proposals = proposals
    save_chat_thread(thread)
    return thread


def add_trace_artifact(
    profile_slug: str,
    *,
    trace_dir: str | None,
    state_path: str | None,
    thread_id: str = "default",
) -> ChatThreadState:
    thread = load_chat_thread(profile_slug, thread_id=thread_id)
    artifact = {key: value for key, value in {"trace_dir": trace_dir, "state_path": state_path}.items() if value}
    if artifact:
        thread.trace_artifacts.append(artifact)
        for message in reversed(thread.messages):
            if message.role == "assistant":
                message.trace_dir = trace_dir
                message.state_path = state_path
                break
    save_chat_thread(thread)
    return thread


def append_turn(
    profile_slug: str,
    *,
    user_content: str,
    assistant_content: str,
    proposals: list[CourseProposal],
    rolling_summary: str,
    thread_id: str = "default",
) -> ChatThreadState:
    proposals_list = list(proposals) if proposals is not None else []
    thread = load_chat_thread(profile_slug, thread_id=thread_id)
    if not _last_user_message_matches(thread, user_content):
        thread.messages.append(ChatMessage(role="user", content=user_content))
    thread.messages.append(
        ChatMessage(
            role="assistant",
            content=assistant_content,
            metadata={"proposal_count": len(proposals_list)},
        )
    )
    thread.active_proposals = proposals_list
    thread.rolling_summary = rolling_summary
    save_chat_thread(thread)
    return thread


def _last_user_message_matches(thread: ChatThreadState, content: str) -> bool:
    if not thread.messages:
        return False
    last = thread.messages[-1]
    return last.role == "user" and last.content.strip() == content.strip()


def _clean_slug(profile_slug: str) -> str:
    return str(profile_slug or "primary").strip() or "primary"
