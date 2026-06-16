from __future__ import annotations

import base64
from urllib.parse import quote

from crew.isis_client import extract_mobile_wstoken


def test_extract_mobile_wstoken_accepts_moodlemobile_scheme_location():
    raw = "private-token:::0123456789abcdef0123456789abcdef:::site-id"
    token = quote(base64.b64encode(raw.encode("utf-8")).decode("ascii"))

    assert extract_mobile_wstoken(f"moodlemobile://token={token}") == "0123456789abcdef0123456789abcdef"


def test_isis_client_context_var_isolation():
    import contextvars
    from crew.isis_client import get_default_isis_client, set_default_isis_client, reset_default_isis_client, MoodleRestClient

    client1 = MoodleRestClient(wstoken="token1")
    client2 = MoodleRestClient(wstoken="token2")

    set_default_isis_client(client1)
    assert get_default_isis_client().wstoken == "token1"

    def run_in_context():
        reset_default_isis_client()
        set_default_isis_client(client2)
        assert get_default_isis_client().wstoken == "token2"

    ctx = contextvars.copy_context()
    ctx.run(run_in_context)

    assert get_default_isis_client().wstoken == "token1"
    reset_default_isis_client()


def test_use_default_isis_client_scopes_and_restores_client():
    from crew.isis_client import MoodleRestClient, get_default_isis_client, reset_default_isis_client, set_default_isis_client, use_default_isis_client

    outer = MoodleRestClient(wstoken="outer")
    inner = MoodleRestClient(wstoken="inner")

    set_default_isis_client(outer)
    with use_default_isis_client(inner) as scoped:
        assert scoped is inner
        assert get_default_isis_client().wstoken == "inner"

    assert get_default_isis_client().wstoken == "outer"
    reset_default_isis_client()


def test_self_unenrol_course_fallback_success(monkeypatch):
    from typing import Any
    from crew.isis_client import MoodleRestClient, MoodleApiError

    client = MoodleRestClient(wstoken="test_token")

    # Mock call for enrol_self_unenrol_user to raise MoodleApiError
    def mock_call(wsfunction: str, **params: Any):
        if wsfunction == "enrol_self_unenrol_user":
            raise MoodleApiError("Can't find data record in database.")
        raise RuntimeError(f"Unexpected wsfunction call: {wsfunction}")
    monkeypatch.setattr(client, "call", mock_call)

    # Mock course_enrolment_methods
    monkeypatch.setattr(client, "course_enrolment_methods", lambda cid: [
        {'id': 141002, 'courseid': cid, 'type': 'self', 'name': 'Self enrolment'}
    ])

    # Mock session.get and session.post
    class MockResponse:
        def __init__(self, text="", status_code=200):
            self.text = text
            self.status_code = status_code
        def raise_for_status(self):
            pass

    def mock_get(url, **kwargs):
        if url == client.base_url:
            return MockResponse(text='sesskey":"2kgVySnoKC"')
        return MockResponse()

    post_calls = []
    def mock_post(url, data=None, **kwargs):
        post_calls.append((url, data))
        return MockResponse(status_code=200)

    monkeypatch.setattr(client.session, "get", mock_get)
    monkeypatch.setattr(client.session, "post", mock_post)

    # Mock enrolled_course_refs to return empty list (unenrollment success verification)
    monkeypatch.setattr(client, "enrolled_course_refs", lambda: [])

    # Run the unenrollment
    result = client.self_unenrol_course(46420)

    assert result == {"status": True, "note": "Successfully unenrolled via fallback browser session endpoint."}
    assert len(post_calls) == 1
    assert post_calls[0][0] == "https://isis.tu-berlin.de/enrol/self/unenrolself.php"
    assert post_calls[0][1] == {"enrolid": 141002, "confirm": 1, "sesskey": "2kgVySnoKC"}
