from threading import Event, Lock
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.catalog_descriptions import CatalogDescription, CatalogDescriptions, fetch_catalog_description
from core.providers.tu_berlin import moses
from ui.course_search import _description_excerpt


def course(number="1", version=1, **kwargs):
    return CatalogDescription(**kwargs)


def test_background_requests_are_bounded_and_new_searches_do_not_wait_for_a_large_queue():
    release = Event()
    started = Event()
    calls = []
    lock = Lock()

    def fetch(number, version, term):
        with lock:
            calls.append(number)
            if len(calls) == 2:
                started.set()
        assert release.wait(5)
        return course(number, version)

    store = CatalogDescriptions(fetch, workers=2)
    keys = [(str(i), 1, "") for i in range(20)]
    try:
        results = store.request(keys)
        assert started.wait(2)
        assert set(calls) == {"0", "1"}
        assert results[keys[-1]].state == "queued"
        # Repeated redraws do not duplicate requests or queue the other 18.
        store.request(keys)
        assert len(calls) == 2
        release.set()
        assert store.resolve(keys[0]) == course()
        assert store.resolve(keys[1]) == course()
        new = ("new-search", 1, "")
        store.request([new])
        assert store.resolve(new) == course()
        assert len(calls) == 3 and set(calls) == {"0", "1", "new-search"}
    finally:
        release.set()
        store.close()


def test_cache_is_version_and_semester_specific_and_expires():
    now = [0]
    fetch = Mock(side_effect=lambda number, version, term: course(number, version))
    store = CatalogDescriptions(fetch, capacity=2, ttl=10, clock=lambda: now[0])
    try:
        key = ("1", 1, "WS 26/27")
        assert store.resolve(key) == course()
        assert store.resolve(key) == course()
        assert fetch.call_count == 1
        assert store.resolve(("1", 2, "WS 26/27")) == course()
        store.resolve(("1", 1, "SS 27"))
        assert fetch.call_count == 3
        store.resolve(key)  # Oldest entry was evicted.
        assert fetch.call_count == 4
        now[0] = 11
        store.resolve(key)
        assert fetch.call_count == 5
    finally:
        store.close()


def test_errors_are_visible_without_repeated_background_requests_and_preview_can_retry():
    fetch = Mock(side_effect=TimeoutError("MOSES unavailable"))
    store = CatalogDescriptions(fetch)
    key = ("1", 1, "")
    try:
        with pytest.raises(TimeoutError):
            store.resolve(key)
        assert store.request([key])[key].state == "error"
        assert fetch.call_count == 1
        fetch.side_effect = None
        fetch.return_value = course()
        assert store.resolve(key) == course()
        assert store.request([key])[key].state == "ready"
        assert fetch.call_count == 2
    finally:
        store.close()


def test_excerpt_prefers_learning_outcomes_and_handles_missing_descriptions():
    assert _description_excerpt(course(learning_outcomes="Learn secure design", contents="Other content")) == ("Learning outcomes", "Learn secure design")
    assert _description_excerpt(course(learning_outcomes="Keine Angabe", contents="Course topics")) == ("Course content", "Course topics")
    assert _description_excerpt(course())[1] == "No course description provided by MOSES."
    label, excerpt = _description_excerpt(course(learning_outcomes="Explore secure systems. " * 100))
    assert label == "Learning outcomes"
    assert len(excerpt) <= 360


@pytest.mark.parametrize("language", ["de", "en"])
def test_snippets_use_one_detail_page_without_full_scraping(monkeypatch, language):
    markup = (Path(__file__).parent / "fixtures/moses/detail_page.html").read_text()
    if language == "en":
        markup = markup.replace("Lernergebnisse", "Learning outcomes").replace("Lehrinhalte", "Contents")
    fetch = Mock(return_value=markup)
    full_fetch = Mock(side_effect=AssertionError("A card must not expand degree catalogs or ISIS links"))
    monkeypatch.setattr(moses, "fetch_html", fetch)
    monkeypatch.setattr(moses, "fetch_course_details", full_fetch)
    result = fetch_catalog_description("40388", 8, "WS 26/27")
    assert result.learning_outcomes == "Students gain research presentation and review experience."
    assert result.contents == "Recent topics in computer security across multiple subfields."
    fetch.assert_called_once_with(moses._canonical_detail_url("40388", 8), timeout=10)
    full_fetch.assert_not_called()
