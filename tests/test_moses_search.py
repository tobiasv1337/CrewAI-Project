from __future__ import annotations

from unittest.mock import Mock

import pytest

from core.providers.tu_berlin import moses


class _Response:
    def __init__(self, body: str):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body.encode("utf-8")

    def geturl(self):
        return moses.SEARCH_URL


def test_each_session_loads_its_own_search_form(monkeypatch):
    first_opener = Mock()
    first_opener.open.return_value = _Response("first session's form")
    second_opener = Mock()
    second_opener.open.return_value = _Response("second session's form")
    monkeypatch.setattr(moses, "build_opener", Mock(side_effect=[first_opener, second_opener]))

    first = moses._MosesSession()
    second = moses._MosesSession()

    assert first.get(moses.SEARCH_URL)[0] == "first session's form"
    assert second.get(moses.SEARCH_URL)[0] == "second session's form"
    second_opener.open.assert_called_once()


def test_repeated_post_reaches_server_instead_of_replaying_old_state(monkeypatch):
    opener = Mock()
    opener.open.side_effect = [_Response("old results"), _Response("current results")]
    monkeypatch.setattr(moses, "build_opener", Mock(return_value=opener))
    session = moses._MosesSession()
    payload = {"query": "Digitale Systeme"}

    assert session.post(moses.SEARCH_URL, payload, partial=True) == "old results"
    assert session.post(moses.SEARCH_URL, payload, partial=True) == "current results"
    assert opener.open.call_count == 2


def _search_context():
    return moses._MosesSearchContext(
        form=moses._MosesFormContext(
            form_id="search",
            action_url=moses.SEARCH_URL,
            view_state="view-state",
            client_window="window",
            # A previous result must never be used when a new search fails.
            html='<form id="search"><p>Unrelated previous courses</p></form>',
        ),
        query_input_name="search:query",
        submit_id="search:submit",
        render_id="search",
    )


@pytest.mark.parametrize("degree_search", [False, True])
@pytest.mark.parametrize(
    "partial",
    [
        '<partial-response><redirect url="/expired"/></partial-response>',
        "<partial-response><error><error-name>jakarta.faces.application.ViewExpiredException</error-name></error></partial-response>",
        "<html>Service unavailable</html>",
    ],
)
def test_failed_search_does_not_return_previous_courses(monkeypatch, degree_search, partial):
    session = Mock()
    session.get.return_value = ("search page", moses.SEARCH_URL)
    session.post.return_value = partial
    monkeypatch.setattr(moses, "_extract_search_context", Mock(return_value=_search_context()))
    monkeypatch.setattr(moses, "_extract_degree_program_search_context", Mock(return_value=_search_context()))
    parse_courses = Mock()
    parse_degrees = Mock()
    monkeypatch.setattr(moses, "_parse_search_results", parse_courses)
    monkeypatch.setattr(moses, "_parse_degree_program_search_results", parse_degrees)

    with pytest.raises(ValueError, match="MOSES did not return search results"):
        if degree_search:
            moses._search_degree_programs(session, "Informatik", degree_type=None, provider=None, max_results=20)
        else:
            moses._search_courses(session, "Digitale Systeme", max_results=20, filters=None)

    parse_courses.assert_not_called()
    parse_degrees.assert_not_called()


def test_valid_empty_search_still_returns_no_matches(monkeypatch):
    session = Mock()
    session.get.return_value = ("search page", moses.SEARCH_URL)
    session.post.return_value = """
        <partial-response><changes><update id="search"><![CDATA[
          <table><tbody id="search:ergebnisliste_data">
            <tr><td colspan="10">No records found.</td></tr>
          </tbody></table>
        ]]></update></changes></partial-response>
    """
    monkeypatch.setattr(moses, "_extract_search_context", Mock(return_value=_search_context()))

    assert moses._search_courses(session, "Unknown course", max_results=20, filters=None) == []
