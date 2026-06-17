from __future__ import annotations

import asyncio
import base64
from contextvars import ContextVar
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
import html
import os
import re
import threading
import time
from typing import Any, Iterator
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests

from crew.isis_models import ISIS_BASE_URL, IsisCourseRef, clean_text, normalize_optional_text, parse_isis_course_id_from_url


ISIS_API_PATH = "/webservice/rest/server.php"
DEFAULT_REQUEST_DELAY_SECONDS = 1.2
DEFAULT_TIMEOUT_SECONDS = 30
MOBILE_SERVICE = "moodle_mobile_app"


class IsisError(RuntimeError):
    """Base exception for ISIS client errors."""


class IsisAuthenticationError(IsisError):
    """Raised when ISIS credentials or token extraction fail."""


class MoodleApiError(IsisError):
    """Raised when Moodle returns a webservice exception."""

    def __init__(
        self,
        message: str,
        *,
        errorcode: str | None = None,
        exception: str | None = None,
        debuginfo: str | None = None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.errorcode = errorcode
        self.exception = exception
        self.debuginfo = debuginfo
        self.response = response

    @property
    def is_access_error(self) -> bool:
        text = " ".join(
            clean_text(value).lower()
            for value in (self.errorcode, self.exception, self.debuginfo, str(self))
            if value
        )
        markers = (
            "access",
            "not enrolled",
            "notenrolled",
            "permission",
            "nopermissions",
            "requirelogin",
            "kein recht",
            "keine rechte",
            "not allowed",
            "cannot view",
            "coursehidden",
            "invalidrecord",
        )
        return any(marker in text for marker in markers)


@dataclass(frozen=True)
class IsisCredentials:
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "IsisCredentials":
        load_dotenv()
        username = os.getenv("ISIS_USERNAME") or os.getenv("TUB_USERNAME") or os.getenv("TU_BERLIN_USERNAME")
        password = os.getenv("ISIS_PASSWORD") or os.getenv("TUB_PASSWORD") or os.getenv("TU_BERLIN_PASSWORD")
        if not username or not password:
            raise IsisAuthenticationError("ISIS_USERNAME and ISIS_PASSWORD must be set in .env or the environment.")
        return cls(username=username, password=password)


@dataclass(frozen=True)
class IsisTokenBundle:
    wstoken: str
    cookies: dict[str, str]


def strip_html(value: object | None) -> str:
    raw = "" if value is None else str(value)
    if not raw:
        return ""
    if "<" not in raw and raw.strip().lower().startswith(("http://", "https://")):
        return clean_text(raw)
    soup = BeautifulSoup(html.unescape(raw), "html.parser")
    return clean_text(soup.get_text(" ", strip=True))


def is_coursemanager_lvvid_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(html.unescape(url))
    return parsed.path.rstrip("/") == "/local/coursemanager/search.php" and bool(parse_qs(parsed.query).get("lvvid"))


def extract_mobile_wstoken(location_header: str) -> str:
    match = re.search(r"(?:^|[?&/])token=([^&]+)", location_header or "")
    if not match:
        raise IsisAuthenticationError("Moodle mobile launch response did not contain a token parameter.")
    token_b64 = unquote(match.group(1))
    try:
        raw = base64.b64decode(token_b64).decode("utf-8")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise IsisAuthenticationError("Moodle mobile launch token could not be base64-decoded.") from exc
    parts = raw.split(":::")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    if parts and parts[0]:
        return parts[0]
    raise IsisAuthenticationError("Moodle mobile launch token did not contain a usable wstoken.")


async def login_via_playwright(credentials: IsisCredentials, *, headless: bool = True, base_url: str = ISIS_BASE_URL) -> IsisTokenBundle:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        ctx = await browser.new_context()
        page = await ctx.new_page()
        try:
            await page.goto(f"{base_url}/login/index.php")
            await page.wait_for_load_state("domcontentloaded")

            with suppress(Exception):
                button = page.get_by_text("Fortsetzen", exact=True)
                if await button.is_visible(timeout=3000):
                    await button.click()
                    await page.wait_for_load_state("domcontentloaded")

            await page.get_by_text("TU-Login", exact=True).click()
            await page.wait_for_selector('[name="j_username"]', timeout=20000)
            await page.fill('[name="j_username"]', credentials.username)
            await page.fill('[name="j_password"]', credentials.password)
            await page.click('[name="_eventId_proceed"]')
            await page.wait_for_url(f"{base_url}/my/**", timeout=30000)

            cookies = {
                item["name"]: item["value"]
                for item in await ctx.cookies()
                if "isis.tu-berlin.de" in item.get("domain", "")
            }
        finally:
            await browser.close()

    session = requests.Session()
    for key, value in cookies.items():
        session.cookies.set(key, value, domain="isis.tu-berlin.de")
    response = session.get(
        f"{base_url}/admin/tool/mobile/launch.php",
        params={"service": MOBILE_SERVICE, "passport": "x"},
        allow_redirects=False,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    location = response.headers.get("Location", "")
    wstoken = extract_mobile_wstoken(location)
    return IsisTokenBundle(wstoken=wstoken, cookies=cookies)


def login_via_playwright_sync(credentials: IsisCredentials, *, headless: bool = True, base_url: str = ISIS_BASE_URL) -> IsisTokenBundle:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise IsisAuthenticationError("Cannot run Playwright login synchronously while an event loop is already running.")
    return asyncio.run(login_via_playwright(credentials, headless=headless, base_url=base_url))


class MoodleRestClient:
    """Small Moodle REST client for ISIS mobile webservice functions."""

    def __init__(
        self,
        *,
        wstoken: str,
        cookies: dict[str, str] | None = None,
        base_url: str = ISIS_BASE_URL,
        session: requests.Session | None = None,
        request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}{ISIS_API_PATH}"
        self.wstoken = wstoken
        self.session = session or requests.Session()
        self.request_delay_seconds = request_delay_seconds
        self.timeout = timeout
        self._last_request_at = 0.0
        self._site_info: dict[str, Any] | None = None
        for key, value in (cookies or {}).items():
            self.session.cookies.set(key, value, domain="isis.tu-berlin.de")

    @classmethod
    def from_env(cls, *, request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS) -> "MoodleRestClient":
        load_dotenv()
        token = os.getenv("ISIS_WSTOKEN")
        cookies: dict[str, str] = {}
        if token:
            return cls(wstoken=token, cookies=cookies, request_delay_seconds=request_delay_seconds)
        credentials = IsisCredentials.from_env()
        bundle = login_via_playwright_sync(credentials)
        return cls(wstoken=bundle.wstoken, cookies=bundle.cookies, request_delay_seconds=request_delay_seconds)

    def call(self, wsfunction: str, **params: Any) -> Any:
        self._respect_delay()
        payload = {
            "wstoken": self.wstoken,
            "wsfunction": wsfunction,
            "moodlewsrestformat": "json",
        }
        payload.update(_flatten_params(params))
        response = self.session.get(self.api_url, params=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and ("exception" in data or "errorcode" in data):
            message = clean_text(data.get("message") or data.get("debuginfo") or data.get("errorcode") or "Moodle API error")
            raise MoodleApiError(
                message,
                errorcode=normalize_optional_text(data.get("errorcode")),
                exception=normalize_optional_text(data.get("exception")),
                debuginfo=normalize_optional_text(data.get("debuginfo")),
                response=data,
            )
        return data

    def _respect_delay(self) -> None:
        if self.request_delay_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.request_delay_seconds:
            time.sleep(self.request_delay_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def site_info(self) -> dict[str, Any]:
        if self._site_info is None:
            self._site_info = self.call("core_webservice_get_site_info")
        return self._site_info

    def user_id(self) -> int:
        return int(self.site_info()["userid"])

    def session_time_remaining(self) -> Any:
        return self.call("core_session_time_remaining")

    def enrolled_courses(self) -> list[dict[str, Any]]:
        return list(self.call("core_enrol_get_users_courses", userid=self.user_id()) or [])

    def enrolled_course_refs(self) -> list[IsisCourseRef]:
        return [course_ref_from_moodle(course, enrolled=True, base_url=self.base_url) for course in self.enrolled_courses()]

    def search_courses(self, query: str, *, perpage: int = 10, page: int = 0) -> list[dict[str, Any]]:
        result = self.call(
            "core_course_search_courses",
            criterianame="search",
            criteriavalue=query,
            perpage=perpage,
            page=page,
        )
        if isinstance(result, dict):
            return list(result.get("courses") or [])
        return []

    def search_course_refs(self, query: str, *, perpage: int = 10, page: int = 0) -> list[IsisCourseRef]:
        return [course_ref_from_moodle(course, enrolled=False, base_url=self.base_url) for course in self.search_courses(query, perpage=perpage, page=page)]

    def get_courses_by_field(self, *, field: str, value: str) -> list[dict[str, Any]]:
        result = self.call("core_course_get_courses_by_field", field=field, value=value)
        if isinstance(result, dict):
            return list(result.get("courses") or [])
        return []

    def course_ref_by_id(self, course_id: int) -> IsisCourseRef | None:
        for course in self.enrolled_course_refs():
            if course.id == course_id:
                return course
        try:
            courses = self.get_courses_by_field(field="id", value=str(course_id))
        except MoodleApiError:
            courses = []
        if courses:
            return course_ref_from_moodle(courses[0], enrolled=False, base_url=self.base_url)
        return IsisCourseRef(id=course_id, url=f"{self.base_url}/course/view.php?id={course_id}")

    def course_contents(self, course_id: int) -> list[dict[str, Any]]:
        return list(self.call("core_course_get_contents", courseid=course_id) or [])

    def course_updates_since(self, course_id: int, since: int) -> Any:
        return self.call("core_course_get_updates_since", courseid=course_id, since=since)

    def course_enrolment_methods(self, course_id: int) -> list[dict[str, Any]]:
        data = self.call("core_enrol_get_course_enrolment_methods", courseid=course_id)
        return list(data or [])

    def self_enrol_course(self, course_id: int, *, password: str | None = None, instance_id: int | None = None) -> Any:
        params: dict[str, Any] = {"courseid": course_id}
        if password:
            params["password"] = password
        if instance_id is not None:
            params["instanceid"] = instance_id
        return self.call("enrol_self_enrol_user", **params)

    def self_unenrol_course(self, course_id: int) -> Any:
        try:
            return self.call("enrol_self_unenrol_user", courseid=course_id)
        except MoodleApiError as exc:
            # If the web service function is not registered/enabled on the server, Moodle returns 
            # 'Can't find data record in database.'. In this case, fall back to browser-style session unenrollment.
            if "Can't find data record in database" not in str(exc):
                raise

            methods = self.course_enrolment_methods(course_id)
            # Find any active self or manual enrollment method
            target_method = None
            for method in methods:
                if method.get("type") in {"self", "manual"}:
                    target_method = method
                    break

            if not target_method:
                raise ValueError(
                    f"No self or manual enrollment method was found for course ID {course_id} to perform fallback unenrollment."
                ) from exc

            # Fetch the ISIS homepage to extract the user's session key (sesskey)
            r_home = self.session.get(self.base_url)
            r_home.raise_for_status()
            sesskey_match = re.search(r'sesskey\":\"([a-zA-Z0-9]+)\"', r_home.text)
            if not sesskey_match:
                sesskey_match = re.search(r'sesskey=([a-zA-Z0-9]+)', r_home.text)

            if not sesskey_match:
                raise ValueError("Could not find Moodle session key (sesskey) in the web page source.") from exc

            sesskey = sesskey_match.group(1)
            enrol_id = target_method["id"]
            method_type = target_method["type"]

            # Perform the unenrollment request
            post_url = f"{self.base_url}/enrol/{method_type}/unenrolself.php"
            payload = {"enrolid": enrol_id, "confirm": 1, "sesskey": sesskey}
            r_unenroll = self.session.post(post_url, data=payload, timeout=self.timeout)
            r_unenroll.raise_for_status()

            # Verify unenrollment succeeded by checking if the course is still returned in enrolled courses
            enrolled = {c.id for c in self.enrolled_course_refs()}
            if course_id in enrolled:
                raise RuntimeError(
                    f"Fallback unenrollment submitted successfully but user is still enrolled in course {course_id}."
                ) from exc

            return {"status": True, "note": "Successfully unenrolled via fallback browser session endpoint."}

    def assignments(self, course_id: int) -> dict[str, Any]:
        return dict(self.call("mod_assign_get_assignments", **{"courseids[0]": course_id}) or {})

    def assignment_submission_status(self, assignment_id: int) -> dict[str, Any]:
        return dict(self.call("mod_assign_get_submission_status", assignid=assignment_id) or {})

    def assignment_grades(self, assignment_id: int) -> dict[str, Any]:
        return dict(self.call("mod_assign_get_grades", assignmentids=[assignment_id]) or {})

    def forums(self, course_id: int) -> list[dict[str, Any]]:
        return list(self.call("mod_forum_get_forums_by_courses", **{"courseids[0]": course_id}) or [])

    def forum_discussions(self, forum_id: int, *, perpage: int = 10, page: int = 0) -> dict[str, Any]:
        return dict(self.call("mod_forum_get_forum_discussions", forumid=forum_id, perpage=perpage, page=page) or {})

    def forum_discussion_posts(self, discussion_id: int) -> dict[str, Any]:
        return dict(self.call("mod_forum_get_discussion_posts", discussionid=discussion_id) or {})

    def grade_items(self, course_id: int) -> dict[str, Any]:
        return dict(self.call("gradereport_user_get_grade_items", courseid=course_id, userid=self.user_id()) or {})

    def overview_course_grades(self) -> dict[str, Any]:
        return dict(self.call("gradereport_overview_get_course_grades", userid=self.user_id()) or {})

    def calendar_events(self, course_id: int, *, days_ahead: int = 180, days_past: int = 0) -> dict[str, Any]:
        now = int(time.time())
        return dict(
            self.call(
                "core_calendar_get_calendar_events",
                **{
                    "events[courseids][0]": course_id,
                    "options[timestart]": now - max(days_past, 0) * 86400,
                    "options[timeend]": now + max(days_ahead, 1) * 86400,
                },
            )
            or {}
        )

    def action_events_by_course(self, course_id: int, *, days_ahead: int = 180, days_past: int = 0) -> dict[str, Any]:
        now = int(time.time())
        return dict(
            self.call(
                "core_calendar_get_action_events_by_course",
                courseid=course_id,
                timesortfrom=now - max(days_past, 0) * 86400,
                timesortto=now + max(days_ahead, 1) * 86400,
            )
            or {}
        )

    def course_activity_collection(self, wsfunction: str, course_id: int, key: str) -> list[dict[str, Any]]:
        result = self.call(wsfunction, **{"courseids[0]": course_id})
        if isinstance(result, dict):
            return list(result.get(key) or [])
        return []

    def resources(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_resource_get_resources_by_courses", course_id, "resources")

    def folders(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_folder_get_folders_by_courses", course_id, "folders")

    def pages(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_page_get_pages_by_courses", course_id, "pages")

    def urls(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_url_get_urls_by_courses", course_id, "urls")

    def books(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_book_get_books_by_courses", course_id, "books")

    def quizzes(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_quiz_get_quizzes_by_courses", course_id, "quizzes")

    def lessons(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_lesson_get_lessons_by_courses", course_id, "lessons")

    def feedbacks(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_feedback_get_feedbacks_by_courses", course_id, "feedbacks")

    def choices(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_choice_get_choices_by_courses", course_id, "choices")

    def workshops(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_workshop_get_workshops_by_courses", course_id, "workshops")

    def glossaries(self, course_id: int) -> list[dict[str, Any]]:
        return self.course_activity_collection("mod_glossary_get_glossaries_by_courses", course_id, "glossaries")


def course_ref_from_moodle(data: dict[str, Any], *, enrolled: bool, base_url: str = ISIS_BASE_URL) -> IsisCourseRef:
    course_id = int(data.get("id") or data.get("courseid"))
    fullname = normalize_optional_text(data.get("fullname") or data.get("displayname") or data.get("name"))
    shortname = normalize_optional_text(data.get("shortname"))
    displayname = normalize_optional_text(data.get("displayname"))
    term_hint = _extract_term_hint(fullname, shortname, displayname)
    return IsisCourseRef(
        id=course_id,
        fullname=fullname,
        shortname=shortname,
        displayname=displayname,
        url=f"{base_url.rstrip('/')}/course/view.php?id={course_id}",
        visible=_bool_or_none(data.get("visible")),
        enrolled=enrolled,
        term_hint=term_hint,
        summary=strip_html(data.get("summary")),
        categoryid=_int_or_none(data.get("categoryid")),
        startdate=_int_or_none(data.get("startdate")),
        enddate=_int_or_none(data.get("enddate")),
    )


def safe_call(call: Callable[[], Any]) -> tuple[Any | None, str | None]:
    try:
        return call(), None
    except MoodleApiError as exc:
        return None, f"{exc.errorcode or exc.exception or 'moodle_error'}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive guard for live endpoints
        return None, str(exc)


def unix_date(value: object | None) -> str | None:
    timestamp = _int_or_none(value)
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp).isoformat(sep=" ", timespec="minutes")


def days_ago_timestamp(days: int) -> int:
    return int((datetime.now() - timedelta(days=max(days, 0))).timestamp())


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if "[" in key:
            flattened[key] = value
            continue
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                flattened[f"{key}[{index}]"] = item
            continue
        if isinstance(value, dict):
            for inner_key, item in value.items():
                flattened[f"{key}[{inner_key}]"] = item
            continue
        flattened[key] = value
    return flattened


def _extract_term_hint(*values: str | None) -> str | None:
    text = " ".join(value or "" for value in values)
    patterns = (
        r"\b(?:SoSe|SS)\s*20?\d{2}\b",
        r"\b(?:WiSe|WS)\s*20?\d{2}(?:/\d{2})?\b",
        r"\[(SoSe\s*20?\d{2}|SS\s*20?\d{2}|WiSe\s*20?\d{2}/?\d{0,2}|WS\s*20?\d{2}/?\d{0,2})\]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return clean_text(match.group(1) if match.groups() else match.group(0))
    return None


def _bool_or_none(value: object | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    text = clean_text(value).lower()
    if text in {"1", "true", "yes", "ja"}:
        return True
    if text in {"0", "false", "no", "nein"}:
        return False
    return None


def _int_or_none(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    with suppress(TypeError, ValueError):
        return int(value)
    return None


_CLIENT_VAR: ContextVar[MoodleRestClient | None] = ContextVar("isis_client", default=None)
_ENV_CLIENT: MoodleRestClient | None = None
_ENV_CLIENT_LOCK = threading.RLock()


def get_default_isis_client() -> MoodleRestClient:
    global _ENV_CLIENT
    client = _CLIENT_VAR.get()
    if client is None:
        with _ENV_CLIENT_LOCK:
            if _ENV_CLIENT is None:
                _ENV_CLIENT = MoodleRestClient.from_env()
            client = _ENV_CLIENT
        _CLIENT_VAR.set(client)
    return client


def current_scoped_isis_client() -> MoodleRestClient | None:
    """Return the explicitly scoped ISIS client without falling back to env login."""
    return _CLIENT_VAR.get()


def set_default_isis_client(client: MoodleRestClient) -> None:
    _CLIENT_VAR.set(client)


def reset_default_isis_client() -> None:
    global _ENV_CLIENT
    _CLIENT_VAR.set(None)
    with _ENV_CLIENT_LOCK:
        _ENV_CLIENT = None


@contextmanager
def use_default_isis_client(client: MoodleRestClient | None) -> Iterator[MoodleRestClient | None]:
    """Temporarily scope ISIS tools to a UI/session-provided client.

    Passing ``None`` intentionally clears the scoped client for the duration of
    the context, so callers can fall back to the normal environment-based
    behavior outside the context.
    """
    token = _CLIENT_VAR.set(client)
    try:
        yield client
    finally:
        _CLIENT_VAR.reset(token)
