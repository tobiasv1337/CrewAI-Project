"""Chat presentation contracts, with no network calls or writes to live profiles."""
from __future__ import annotations

import json
from html import unescape

from streamlit.testing.v1 import AppTest

from ui import chat


CHAT_FIXTURE_SCRIPT = '''
import streamlit as st
from unittest.mock import patch
from crew.chat_models import ChatMessage, ChatThreadState
from ui import chat

st.session_state["active_profile"] = "chat-ui-test"
st.session_state.setdefault("active_thread_id_chat-ui-test", "default")
full_output = '{"status":"ok","courses":[{"title":"Test course","credits":6}],"note":"FULL_OUTPUT_END"}'
workbench = {
    "run_id": "ui-test-run", "event_count": 4,
    "groups": [{"agent_label":"Study Advisor", "status":"completed", "llm_calls":2,"total_tokens":1240,
        "tool_calls":[
            {"call_id":1,"tool_name":"get_courses","agent_label":"Study Advisor","status":"ok","duration_ms":1240,
             "tool_input":{"query":"security","limit":3},"output_preview":"short preview", "output":full_output,
             "output_chars":len(full_output),"output_truncated":True},
            {"call_id":2,"tool_name":"read_catalog","agent_label":"MOSES Module Researcher","status":"error",
             "status_source":"runtime","error":"Connection timeout","duration_ms":30000,
             "tool_input":{"module_id":101},"output_preview":"Connection timeout"},
            {"call_id":3,"tool_name":"lookup_course","agent_label":"Study Advisor","status":"warning",
             "tool_input":{},"output_preview":'{"status":"partial","note":"One source unavailable"}'},
        ]}],
    "agent_dialogue":[{"sender":"Orchestrator","receiver":"Study Advisor","status":"completed","duration_ms":1240,
        "question":"Compare my course options.","question_full":"Compare my course options. Include degree requirements and workload.",
        "response":"Found a suitable course.",
        "response_full":"## Recommendation\\nA suitable **security course**.\\n\\n| Course | Credits |\\n|---|---|\\n| Test course | 6 |\\n\\n```python\\nif ready:\\n    choose_course()\\n```\\n\\nFULL_DIALOGUE_END",
        "observer_summary":"One eligible course found."}],
    "latest_events":[{"event":"tool_end","call_id":1,"status":"ok"},{"event":"tool_error","call_id":2,"error":"Connection timeout"}],
    "artifacts":{"events_jsonl":"logs/ui-test/events.jsonl","trace_json":"logs/ui-test/trace.json"}
}
messages = [
    ChatMessage(role="user",content="Compare my courses and help me plan next semester.",created_at="2026-09-15T10:00:00+00:00"),
    ChatMessage(role="assistant",content="## Your study options\\nHere is the **complete answer**, with [course information](https://www.tu.berlin).\\n\\n| Course | Credits | Status |\\n|---|---|---|\\n| Test course | 6 | Planned |\\n\\n```python\\nif ready:\\n    choose_course()\\n```",created_at="2026-09-15T10:00:04+00:00",metadata={"workbench":workbench}),
]
thread = ChatThreadState(profile_slug="chat-ui-test",messages=messages)
empty_thread = ChatThreadState(thread_id="empty", profile_slug="chat-ui-test")
threads = {"default":thread,"empty":empty_thread}
def fake_run(*args, **kwargs):
    st.session_state["submitted_query"] = args[1]
    chat._render_chat_message({"role":"user","content":args[1]})
    chat._render_chat_message({"role":"assistant","content":"Isolated response"})
with patch.object(chat, "load_chat_thread", side_effect=lambda slug, thread_id="default":threads.get(thread_id, ChatThreadState(thread_id=thread_id,profile_slug="chat-ui-test"))), \\
     patch.object(chat, "list_chat_threads", return_value=list(threads.values())), \\
     patch.object(chat, "get_profile_messages", side_effect=lambda slug, thread_id="default":[m.model_dump() for m in threads.get(thread_id,empty_thread).messages]), \\
     patch.object(chat, "clear_chat_thread"), \\
     patch.object(chat, "_run_and_render_assistant_turn", side_effect=fake_run):
    chat.render_chat_page()
'''


def test_trace_output_preserves_full_captured_text_and_preview_limits():
    full = '{ "value": 1, "note": "Full output" }\n'
    assert chat._trace_output({"output": full, "output_preview": "cut", "output_truncated": True}) == (full, "json", False)
    assert chat._trace_output({"output_preview": "Only this was captured", "output_truncated": True}) == ("Only this was captured", None, True)
    assert chat._trace_output({"output": "", "output_preview": "stale preview"}) == ("", None, False)
    assert json.loads(chat._trace_output({"output": {"ok": False}})[0]) == {"ok": False}


def test_dialogue_preserves_markdown_code_and_escapes_raw_html():
    full = '## Answer\n**Evidence**\n```python\nif ready:\n    run()\n```\n<script>alert(1)</script>'
    item = {"sender":"Orchestrator","receiver":"Study Advisor","status":"completed",
            "question":"Check options", "response":"Answer…", "response_full":full}
    rendered = chat._compile_agent_dialogue_html([item])
    assert "<h2>Answer</h2>" in rendered
    assert "<strong>Evidence</strong>" in rendered
    assert "\n    run()\n" in unescape(rendered)
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert 'Read full message' in rendered
    assert rendered == chat._compile_agent_dialogue_html([item])


def test_chat_page_preserves_inspectors_settings_and_composer():
    app = AppTest.from_string(CHAT_FIXTURE_SCRIPT, default_timeout=30).run()
    assert not app.exception
    assert [m.name for m in app.chat_message] == ["user", "assistant"]
    assert {"Orchestration & Topology", "Internal Agent Chat (A2A)", "Raw Trace & Tool I/O", "Output", "Input", "Call metadata"} <= {t.label for t in app.tabs}
    assert not any("Agent dialogue in the conversation" == toggle.label for toggle in app.toggle)
    assert full_output_in_codes(app)
    assert any('"query": "security"' in code.value for code in app.code)
    assert any("Connection timeout" == code.value for code in app.code)
    inspector_labels = [e.label for e in app.expander] + [e.label for e in app.get("status")]
    assert "Trace files" in inspector_labels
    assert any("Failed" in label for label in inspector_labels)
    assert any("Success" in label for label in inspector_labels)
    assert "Delete chat" in [b.label for b in app.button]
    assert app.selectbox(key="chat_session_selector_chat-ui-test").value == "default"
    app.chat_input[0].set_value("Compare two courses").run()
    assert not app.exception
    assert app.session_state["submitted_query"] == "Compare two courses"


def full_output_in_codes(app):
    return any("FULL_OUTPUT_END" in code.value for code in app.code)


def test_chat_switches_conversations_and_empty_state_without_live_data():
    app = AppTest.from_string(CHAT_FIXTURE_SCRIPT, default_timeout=30).run()
    app.selectbox(key="chat_session_selector_chat-ui-test").set_value("empty").run()
    assert not app.exception
    assert not app.chat_message
    assert "Plan my next semester" in [b.label for b in app.button]
    assert "Delete chat" in [b.label for b in app.button]
    app.selectbox(key="chat_session_selector_chat-ui-test").set_value("default").run()
    assert not app.exception
    assert full_output_in_codes(app)


def test_new_session_starts_a_stable_unsaved_draft_and_keeps_history():
    script = CHAT_FIXTURE_SCRIPT.replace('st.session_state.setdefault("active_thread_id_chat-ui-test", "default")', '')
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    selector = "chat_session_selector_chat-ui-test"
    draft = app.selectbox(key=selector).value
    assert draft.startswith("thread-")
    assert not app.chat_message
    app.run()
    assert app.selectbox(key=selector).value == draft
    app.selectbox(key=selector).set_value("default").run()
    assert full_output_in_codes(app)
    app.button(key="chat_new_btn_header_chat-ui-test").click().run()
    assert not app.exception
    assert app.selectbox(key=selector).value not in {"default", draft}
    assert not app.chat_message
    app.selectbox(key=selector).set_value("default").run()
    assert full_output_in_codes(app)


def test_delete_action_can_be_cancelled_without_changing_the_conversation():
    app = AppTest.from_string(CHAT_FIXTURE_SCRIPT, default_timeout=30).run()
    app.button(key="chat_del_btn_header_chat-ui-test").click().run()
    assert not app.exception
    next(button for button in app.button if button.label == "Cancel").click().run()
    assert not app.exception
    assert app.selectbox(key="chat_session_selector_chat-ui-test").value == "default"
    assert full_output_in_codes(app)


def test_fresh_drafts_do_not_persist_empty_chat_files(monkeypatch):
    monkeypatch.setattr(chat.st, "session_state", {})
    def unexpected_save(*args, **kwargs):
        raise AssertionError("An empty draft should not be persisted")
    monkeypatch.setattr(chat, "save_chat_thread", unexpected_save)
    first = chat._get_active_thread_id("draft-test")
    assert chat._get_active_thread_id("draft-test") == first
    chat._start_new_chat("draft-test")
    assert chat._get_active_thread_id("draft-test") != first


def test_agent_identities_stay_distinct_from_result_status():
    rendered = chat._compile_agent_dialogue_html([
        {"sender":"Orchestrator","receiver":"Study Advisor","question":"Check","response":"Done","status":"completed"},
        {"sender":"Orchestrator","receiver":"MOSES Module Researcher","question":"Search","response":"Unavailable","status":"error"},
    ])
    for initials in ("OR", "SA", "MO"):
        assert f'>{initials}</div>' in rendered
    assert 'agent-identity study-advisor' in rendered
    assert 'agent-identity moses' in rendered
    assert 'trace-status completed' in rendered
    assert 'trace-status error' in rendered


def test_delete_confirmation_starts_a_fresh_draft_and_keeps_other_chats(monkeypatch):
    deleted = []
    monkeypatch.setattr(chat, "clear_chat_thread", lambda slug, thread_id: deleted.append((slug, thread_id)))
    app = AppTest.from_string(CHAT_FIXTURE_SCRIPT, default_timeout=30).run()
    app.button(key="chat_del_btn_header_chat-ui-test").click().run()
    next(button for button in app.button if button.label == "Delete conversation").click().run()
    assert not app.exception
    assert deleted == [("chat-ui-test", "default")]
    assert app.selectbox(key="chat_session_selector_chat-ui-test").value.startswith("thread-")
    assert not app.chat_message
    assert "Empty conversation" in app.selectbox(key="chat_session_selector_chat-ui-test").options


def test_isis_login_clears_password_and_can_return_to_server_connection(monkeypatch):
    from crew.isis_client import IsisTokenBundle
    captured = []

    def fake_login(credentials):
        captured.append(credentials.username)
        return IsisTokenBundle(wstoken="test-token", cookies={})

    monkeypatch.setattr(chat, "login_via_playwright_sync", fake_login)
    app = AppTest.from_string("from ui import chat; chat._render_isis_account_panel('chat-ui-test')", default_timeout=30).run()
    app.text_input(key="chat_isis_username_chat-ui-test").set_value("test-account")
    app.text_input(key="chat_isis_password_chat-ui-test").set_value("test-password").run()
    app.button(key="chat_isis_login_chat-ui-test").click().run()
    assert not app.exception
    assert not app.error
    assert captured == ["test-account"]
    assert app.text_input(key="chat_isis_password_chat-ui-test").value == ""
    assert "Account connected" in app.success[0].value
    app.button(key="chat_isis_env_chat-ui-test").click().run()
    assert not app.exception
    assert "chat-ui-test" not in app.session_state[chat.ISIS_SESSIONS_KEY]
    assert app.button(key="chat_isis_env_chat-ui-test").disabled


def test_isis_login_failure_keeps_existing_connection(monkeypatch):
    def fail_login(credentials):
        raise ValueError("Test authentication failure")

    monkeypatch.setattr(chat, "login_via_playwright_sync", fail_login)
    app = AppTest.from_string("""
from ui import chat
from crew.isis_client import MoodleRestClient
chat.st.session_state.setdefault(chat.ISIS_SESSIONS_KEY, {'chat-ui-test': {'mode':'session', 'client':MoodleRestClient(wstoken='existing-test-token')}})
chat._render_isis_account_panel('chat-ui-test')
""", default_timeout=30).run()
    app.text_input(key="chat_isis_username_chat-ui-test").set_value("test-account")
    app.text_input(key="chat_isis_password_chat-ui-test").set_value("test-password").run()
    app.button(key="chat_isis_login_chat-ui-test").click().run()
    assert not app.exception
    assert "Test authentication failure" in app.error[0].value
    assert app.session_state[chat.ISIS_SESSIONS_KEY]['chat-ui-test']['client'].wstoken == 'existing-test-token'
