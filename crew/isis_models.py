from __future__ import annotations

from datetime import datetime
import html
from contextlib import suppress
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


ISIS_BASE_URL = "https://isis.tu-berlin.de"


def parse_isis_course_id_from_url(url: str | None) -> int | None:
    if not url:
        return None
    parsed = urlparse(html.unescape(url))
    if parsed.netloc and not parsed.netloc.endswith("isis.tu-berlin.de"):
        return None
    if parsed.path.rstrip("/") not in {"/course/view.php", "/enrol/index.php"}:
        return None
    values = parse_qs(parsed.query).get("id")
    if not values:
        return None
    with suppress(TypeError, ValueError):
        return int(values[0])
    return None


def clean_text(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def normalize_optional_text(value: object | None) -> str | None:
    text = clean_text(value)
    return text or None


def normalize_text_list(values: object | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list | tuple | set):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


class IsisCourseRef(BaseModel):
    """Stable public course identity used by tools and traces."""

    id: int
    fullname: str | None = None
    shortname: str | None = None
    displayname: str | None = None
    url: str | None = None
    visible: bool | None = None
    enrolled: bool = False
    term_hint: str | None = None
    summary: str | None = None
    categoryid: int | None = None
    startdate: int | None = None
    enddate: int | None = None

    @field_validator("fullname", "shortname", "displayname", "url", "term_hint", "summary", mode="before")
    @classmethod
    def _normalize_text(cls, value: object | None) -> str | None:
        return normalize_optional_text(value)

    @property
    def title(self) -> str:
        return self.fullname or self.displayname or self.shortname or f"ISIS course {self.id}"


class IsisCourseSelector(BaseModel):
    """A user- or flow-provided course locator for all course-read tools."""

    course_id: int | None = Field(default=None, description="Real ISIS/Moodle course id, not a MOSES module number or lvvid.")
    course_url: str | None = Field(default=None, description="ISIS course URL like https://isis.tu-berlin.de/course/view.php?id=47025.")
    course_query: str | None = Field(default=None, description="Course title/search text when no ISIS course id or URL is known.")
    term_hint: str | None = Field(default=None, description="Optional term hint, e.g. SoSe 2026 or WiSe 2025/26.")
    expected_title: str | None = Field(default=None, description="Optional title hint used to verify the resolved course.")

    @field_validator("course_url", "course_query", "term_hint", "expected_title", mode="before")
    @classmethod
    def _normalize_text(cls, value: object | None) -> str | None:
        return normalize_optional_text(value)

    @field_validator("course_id")
    @classmethod
    def _validate_course_id(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("ISIS course_id must be a positive integer.")
        if value < 1000:
            raise ValueError(
                "This looks like a MOSES/coursemanager lookup id such as lvvid, not an ISIS course_id. "
                "Use an ISIS course URL/id or course_query instead."
            )
        return value

    @model_validator(mode="after")
    def _require_one_locator(self) -> "IsisCourseSelector":
        locators = [
            self.course_id is not None,
            bool(self.course_url),
            bool(self.course_query),
        ]
        if sum(locators) == 0:
            raise ValueError("Provide at least one course locator: course_id, course_url, or course_query.")

        if self.course_url and "lvvid=" in self.course_url.lower():
            raise ValueError("A MOSES coursemanager lvvid URL is not an ISIS course URL. Use a resolved course/view.php?id=... URL.")
        if self.course_query and "lvvid" in self.course_query.lower():
            raise ValueError("Do not pass lvvid values to ISIS tools. Use course_query text or a resolved ISIS course_id.")

        # Normalize/prioritize locators: course_id > course_url > course_query
        if self.course_id is not None:
            if self.course_url:
                url_id = parse_isis_course_id_from_url(self.course_url)
                if url_id is not None and url_id != self.course_id:
                    raise ValueError(
                        f"Conflicting course locators: course_id={self.course_id} "
                        f"does not match the ID in course_url ({url_id})."
                    )
            self.course_url = None
            self.course_query = None
        elif self.course_url:
            url_id = parse_isis_course_id_from_url(self.course_url)
            if url_id is not None:
                if url_id < 1000:
                    raise ValueError(
                        "This looks like a MOSES/coursemanager lookup id such as lvvid, not an ISIS course_id. "
                        "Use an ISIS course URL/id or course_query instead."
                    )
                self.course_id = url_id
                self.course_url = None
                self.course_query = None
            else:
                self.course_query = None

        if self.course_query and self.course_query.strip().isdigit():
            raise ValueError(
                "Numeric course_query is ambiguous and may be a MOSES module number. "
                "Use course_id only for verified ISIS course IDs, or provide a course title."
            )
        return self


class IsisResolvedCourse(BaseModel):
    status: Literal["resolved", "ambiguous", "not_found", "invalid"] = "not_found"
    course: IsisCourseRef | None = None
    candidates: list[IsisCourseRef] = Field(default_factory=list)
    reason: str = ""
    search_terms: list[str] = Field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved" and self.course is not None


class IsisAccessReport(BaseModel):
    course_id: int
    operation: str
    initially_enrolled: bool = False
    temporary_enrollment_allowed: bool = False
    temporary_enrolled: bool = False
    cleanup_attempted: bool = False
    cleanup_succeeded: bool | None = None
    cleanup_error: str | None = None
    access_note: str | None = None


class IsisReadResult(BaseModel):
    course: IsisCourseRef
    access: IsisAccessReport
    data: Any = None


class IsisDateHit(BaseModel):
    source: str
    title: str | None = None
    text: str
    date_text: str | None = None
    time_text: str | None = None
    url: str | None = None


class IsisTraceEvent(BaseModel):
    event: str
    course_id: int | None = None
    course_title: str | None = None
    search_terms: list[str] = Field(default_factory=list)
    candidates: list[IsisCourseRef] = Field(default_factory=list)
    note: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
