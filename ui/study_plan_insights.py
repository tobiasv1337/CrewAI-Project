from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from core.models import Module, ModuleState
from core.module_filters import is_possible_course
from core.terms import term_sort_key

_ADMIN_TOPIC_VALUES = {
    "auto",
    "manual",
    "elective",
    "free choice",
    "additional courses",
    "possible candidates",
    "mandatory",
    "compulsory",
    "completed",
    "in progress",
    "planned",
    "possible candidate",
    "english",
    "german",
    "deutsch",
    "englisch",
    "wintersemester",
    "sommersemester",
    "winter- und sommersemester",
    "winter and summer semester",
    "ws & ss",
    "ws only",
    "ss only",
    "tutor",
    "problem sets",
    "vl",
    "iv",
    "pj",
    "sem",
    "tut",
    "lab",
}

_TOPIC_ALIASES: dict[str, tuple[str, ...]] = {
    "3d printing": ("Graphics & Interactive Systems",),
    "agentic systems": ("AI & Machine Learning",),
    "ai": ("AI & Machine Learning",),
    "algorithm optimization": ("Algorithms & Theory",),
    "algorithms": ("Algorithms & Theory",),
    "assembly": ("Systems Programming & Operating Systems", "Embedded Systems & Hardware"),
    "automata": ("Algorithms & Theory",),
    "autonomous driving": ("Robotics & Autonomous Systems",),
    "c": ("Systems Programming & Operating Systems",),
    "circuits": ("Electronics & Signals",),
    "cognition": ("Psychology & Human Factors",),
    "compiler": ("Programming Languages & Compilers",),
    "computer architecture": ("Embedded Systems & Hardware",),
    "computer graphics": ("Graphics & Interactive Systems",),
    "computer vision": ("AI & Machine Learning", "Graphics & Interactive Systems"),
    "data science": ("Data & Information Systems", "AI & Machine Learning"),
    "data structures": ("Algorithms & Theory",),
    "digital design": ("Embedded Systems & Hardware", "Electronics & Signals"),
    "distributed systems": ("Networks & Distributed Systems",),
    "ee fundamentals": ("Electronics & Signals",),
    "embedded": ("Embedded Systems & Hardware",),
    "embedded systems": ("Embedded Systems & Hardware",),
    "formula student": ("Robotics & Autonomous Systems",),
    "fpga": ("Embedded Systems & Hardware",),
    "frequency domain": ("Electronics & Signals",),
    "functional programming": ("Programming Languages & Compilers",),
    "game development": ("Graphics & Interactive Systems",),
    "graphics programming": ("Graphics & Interactive Systems",),
    "hardware-software-codesign": ("Embedded Systems & Hardware",),
    "hci": ("HCI & Usability",),
    "hdl": ("Embedded Systems & Hardware",),
    "human factors": ("Psychology & Human Factors", "HCI & Usability"),
    "instrumentation": ("Electronics & Signals",),
    "audio and speech": ("Audio & Speech",),
    "audio und sprache": ("Audio & Speech",),
    "bild und video": ("Graphics & Interactive Systems",),
    "interactive systems": ("HCI & Usability", "Graphics & Interactive Systems"),
    "intelligent systems": ("AI & Machine Learning",),
    "information science": ("Data & Information Systems",),
    "informationswissenschaft": ("Data & Information Systems",),
    "it security": ("IT Security",),
    "linear passive circuits": ("Electronics & Signals",),
    "llm": ("AI & Machine Learning",),
    "logic": ("Algorithms & Theory", "Embedded Systems & Hardware"),
    "logical programming": ("Programming Languages & Compilers",),
    "mathematics": ("Mathematics",),
    "mcp": ("AI & Machine Learning",),
    "media communication and effects": ("Media Communication",),
    "media economics": ("Media Economics",),
    "media systems and networks": ("Networks & Distributed Systems",),
    "medienkommunikation und -wirkung": ("Media Communication",),
    "medienwirtschaft": ("Media Economics",),
    "mediensysteme und netze": ("Networks & Distributed Systems",),
    "measurement": ("Electronics & Signals",),
    "measurement basics": ("Electronics & Signals",),
    "mips": ("Embedded Systems & Hardware", "Systems Programming & Operating Systems"),
    "mosfets": ("Electronics & Signals",),
    "multi-agent systems": ("AI & Machine Learning",),
    "mensch-maschine-interaktion": ("HCI & Usability",),
    "network analysis": ("Electronics & Signals",),
    "networking": ("Networks & Distributed Systems",),
    "opamps": ("Electronics & Signals",),
    "operating systems": ("Systems Programming & Operating Systems",),
    "orchestration": ("AI & Machine Learning", "Software Engineering"),
    "programming": ("Software Engineering",),
    "programming languages": ("Programming Languages & Compilers",),
    "psychology": ("Psychology & Human Factors",),
    "react": ("Software Engineering",),
    "risc-v": ("Embedded Systems & Hardware", "Systems Programming & Operating Systems"),
    "robotics": ("Robotics & Autonomous Systems",),
    "ros": ("Robotics & Autonomous Systems",),
    "semiconductors": ("Electronics & Signals",),
    "signals": ("Electronics & Signals",),
    "signals & systems": ("Electronics & Signals",),
    "software engineering": ("Software Engineering",),
    "systems programming": ("Systems Programming & Operating Systems",),
    "testing": ("Software Engineering",),
    "theoretical informatics": ("Algorithms & Theory",),
    "time domain": ("Electronics & Signals",),
    "transients": ("Electronics & Signals",),
    "transistors": ("Electronics & Signals",),
    "usability": ("HCI & Usability",),
    "web": ("Software Engineering",),
}

_TITLE_TOPIC_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("AI & Machine Learning", "Artificial Intelligence", ("artificial intelligence", "kunstliche intelligenz", "künstliche intelligenz", "ki ")),
    ("AI & Machine Learning", "Machine Learning", ("machine intelligence", "machine learning", "maschinelles lernen", "adversarial machine learning", "learning and data")),
    ("AI & Machine Learning", "Natural Language Processing", ("natural language processing",)),
    ("AI & Machine Learning", "Computer Vision", ("computer vision",)),
    ("IT Security", "Security", ("security", "sicherheit", "vulnerability")),
    ("Robotics & Autonomous Systems", "Robotics", ("robotic", "robotics")),
    ("Robotics & Autonomous Systems", "Motion Planning", ("motion planning",)),
    ("Robotics & Autonomous Systems", "Autonomous Systems", ("autonomous systems", "autonomous driving", "uas", "avionics")),
    ("Embedded Systems & Hardware", "Embedded Systems", ("embedded", "rechnerorganisation", "digitale systeme")),
    ("Embedded Systems & Hardware", "Hardware", ("hardware", "fpga", "hdl", "risc-v", "mips", "avionics")),
    ("Electronics & Signals", "Electrical Engineering", ("elektrotechnik", "elektrische", "schaltung", "halbleiter")),
    ("Electronics & Signals", "Measurement & Signals", ("messtechnik", "measurement", "signals", "avionics")),
    ("Programming Languages & Compilers", "Compilers", ("compiler", "compilers")),
    ("Programming Languages & Compilers", "Programming Languages", ("programming languages", "programmiersprachen", "functional programming", "logical programming")),
    ("Systems Programming & Operating Systems", "Operating Systems", ("operating system", "betriebssystem")),
    ("Systems Programming & Operating Systems", "Systems Programming", ("systemprogrammierung", "systems programming")),
    ("Networks & Distributed Systems", "Networking", ("network protocol", "network security", "internet and network", "rechnernetze")),
    ("Networks & Distributed Systems", "Distributed Systems", ("web-service", "verteilte systeme", "distributed systems")),
    ("Data & Information Systems", "Data Science", ("data science",)),
    ("Data & Information Systems", "Data Management", ("data management", "information retrieval", "data integration", "information systems")),
    ("HCI & Usability", "Usability", ("usability", "quality & usability")),
    ("HCI & Usability", "Human-Computer Interaction", ("human-computer",)),
    ("Graphics & Interactive Systems", "Computer Graphics", ("computer graphics", "graphics", "game programming")),
    ("Graphics & Interactive Systems", "Computer Vision", ("computer vision",)),
    ("Software Engineering", "Software Engineering", ("software engineering", "softwaretechnik", "web-service engineering", "amos project")),
    ("Algorithms & Theory", "Algorithms", ("algorithm", "algorithmen")),
    ("Algorithms & Theory", "Theory", ("automata", "automaten", "theoretical informatics")),
    ("Mathematics", "Mathematics", ("mathematics", "mathematik")),
    ("Psychology & Human Factors", "Psychology", ("psychologie", "psychology")),
    ("Psychology & Human Factors", "Human Factors", ("human factors", "cognition")),
)


def _clean(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _format_grade(value: object) -> str:
    if isinstance(value, (int, float)) and float(value) > 0:
        return f"{float(value):.2f}"
    return "-"


def _format_credit_value(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return f"{rounded:d} LP"
    return f"{float(value):.1f}".rstrip("0").rstrip(".") + " LP"


def _grade_average(modules: Iterable[Module], *, weighted: bool, include_estimates: bool) -> str:
    values: list[tuple[float, float]] = []
    for module in modules:
        if is_possible_course(module) or not module.is_graded or module.cp <= 0:
            continue
        grade = module.grade
        if grade is None and include_estimates:
            grade = module.estimated_grade
        if grade is None:
            continue
        values.append((float(grade), float(module.cp)))
    if not values:
        return "-"
    if not weighted:
        return _format_grade(mean(grade for grade, _credits in values))
    total_cp = sum(credits for _grade, credits in values)
    if total_cp <= 0:
        return "-"
    return _format_grade(sum(grade * credits for grade, credits in values) / total_cp)


def _is_academic_topic(value: str) -> bool:
    text = _clean(value)
    if not text:
        return False
    normalized = text.casefold()
    compact = "".join(normalized.split())
    return normalized not in _ADMIN_TOPIC_VALUES and compact not in _ADMIN_TOPIC_VALUES


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def _unique_pairs(values: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for topic, subtopic in values:
        topic_text = _clean(topic)
        subtopic_text = _clean(subtopic)
        key = (topic_text.casefold(), subtopic_text.casefold())
        if topic_text and subtopic_text and key not in seen:
            result.append((topic_text, subtopic_text))
            seen.add(key)
    return result


def _inferred_topic_pairs(module: Module) -> list[tuple[str, str]]:
    normalized = f" {module.name} ".casefold()
    return _unique_pairs(
        (topic, subtopic)
        for topic, subtopic, keywords in _TITLE_TOPIC_RULES
        if any(keyword in normalized for keyword in keywords)
    )


def _inferred_topics(module: Module) -> list[str]:
    return _unique(topic for topic, _subtopic in _inferred_topic_pairs(module))


def clean_topic_values(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        key = text.casefold()
        if _is_academic_topic(text) and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def module_keywords(module: Module) -> list[str]:
    values: list[str] = []
    values.extend(module.catalogs)
    values.extend(module.tags)
    return clean_topic_values(values)


def module_topic_pairs(module: Module) -> list[tuple[str, str]]:
    explicit_keywords = module_keywords(module)
    pairs: list[tuple[str, str]] = []
    for keyword in explicit_keywords:
        for topic in _TOPIC_ALIASES.get(keyword.casefold(), (keyword,)):
            pairs.append((topic, keyword))
    return _unique_pairs(pairs)


def _module_topic_pairs_with_inference(module: Module) -> list[tuple[str, str]]:
    pairs = module_topic_pairs(module)
    explicit_topics = {topic.casefold() for topic, _subtopic in pairs}
    pairs.extend(
        (topic, subtopic)
        for topic, subtopic in _inferred_topic_pairs(module)
        if topic.casefold() not in explicit_topics
    )
    return _unique_pairs(pairs)


def module_topics(module: Module) -> list[str]:
    return _unique(topic for topic, _subtopic in _module_topic_pairs_with_inference(module))


def _topic_rows(modules: list[Module]) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, object]] = defaultdict(
        lambda: {"Topic": "", "Courses": 0, "Credits": 0.0, "Completed Credits": 0.0, "Open Credits": 0.0, "Candidate Credits": 0.0}
    )
    for module in modules:
        topics = module_topics(module)
        if not topics:
            topics = ["Unassigned"]
        for topic in topics:
            row = buckets[topic]
            row["Topic"] = topic
            row["Courses"] = int(row["Courses"]) + 1
            row["Credits"] = float(row["Credits"]) + float(module.cp)
            if module.state == ModuleState.COMPLETED and not is_possible_course(module):
                row["Completed Credits"] = float(row["Completed Credits"]) + float(module.cp)
            if module.state != ModuleState.COMPLETED and not is_possible_course(module):
                row["Open Credits"] = float(row.get("Open Credits") or 0.0) + float(module.cp)
            if is_possible_course(module):
                row["Candidate Credits"] = float(row["Candidate Credits"]) + float(module.cp)
    return sorted(
        buckets.values(),
        key=lambda row: (-float(row["Credits"]), str(row["Topic"]).lower()),
    )


def _subtopic_rows(modules: list[Module]) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "Topic": "",
            "Subtopic": "",
            "Courses": 0,
            "Credits": 0.0,
            "Completed Credits": 0.0,
            "Open Credits": 0.0,
            "Candidate Credits": 0.0,
        }
    )
    for module in modules:
        pairs = module_topic_pairs(module)
        if not pairs:
            continue
        for topic, subtopic in pairs:
            row = buckets[f"{topic.casefold()}\0{subtopic.casefold()}"]
            row["Topic"] = topic
            row["Subtopic"] = subtopic
            row["Courses"] = int(row["Courses"]) + 1
            row["Credits"] = float(row["Credits"]) + float(module.cp)
            if module.state == ModuleState.COMPLETED and not is_possible_course(module):
                row["Completed Credits"] = float(row["Completed Credits"]) + float(module.cp)
            if module.state != ModuleState.COMPLETED and not is_possible_course(module):
                row["Open Credits"] = float(row["Open Credits"]) + float(module.cp)
            if is_possible_course(module):
                row["Candidate Credits"] = float(row["Candidate Credits"]) + float(module.cp)
    return sorted(
        buckets.values(),
        key=lambda row: (-float(row["Credits"]), str(row["Topic"]).lower(), str(row["Subtopic"]).lower()),
    )


def _term_load_rows(modules: list[Module]) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, object]] = defaultdict(
        lambda: {"Term": "", "Credits": 0.0, "Courses": 0, "Open Credits": 0.0}
    )
    for module in modules:
        if is_possible_course(module) and not module.term:
            continue
        term = module.term or "Unknown"
        row = buckets[term]
        row["Term"] = term
        row["Credits"] = float(row["Credits"]) + float(module.cp)
        row["Courses"] = int(row["Courses"]) + 1
        if module.state != ModuleState.COMPLETED:
            row["Open Credits"] = float(row["Open Credits"]) + float(module.cp)
    return sorted(buckets.values(), key=lambda row: term_sort_key(str(row["Term"]), newest_first=True))


def _risk_rows(modules: list[Module], degree_summaries: list[dict[str, object]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    completed_missing_grade = [
        module for module in modules if module.state == ModuleState.COMPLETED and module.is_graded and module.grade is None
    ]
    open_missing_estimate = [
        module
        for module in modules
        if module.state != ModuleState.COMPLETED
        and not is_possible_course(module)
        and module.is_graded
        and module.estimated_grade is None
    ]
    candidates = [module for module in modules if is_possible_course(module)]
    overloaded_terms = [row for row in _term_load_rows(modules) if float(row["Credits"]) > 30.0]
    rule_errors = sum(int(summary.get("error_count") or 0) for summary in degree_summaries)
    rule_warnings = sum(int(summary.get("warning_count") or 0) for summary in degree_summaries)

    if completed_missing_grade:
        rows.append(
            {
                "Severity": "Warning",
                "Topic": "Completed grades",
                "Note": f"{len(completed_missing_grade)} completed graded course(s) have no final grade.",
            }
        )
    if open_missing_estimate:
        rows.append(
            {
                "Severity": "Warning",
                "Topic": "Forecast quality",
                "Note": f"{len(open_missing_estimate)} open graded course(s) have no estimated grade.",
            }
        )
    if overloaded_terms:
        labels = ", ".join(str(row["Term"]) for row in overloaded_terms[:3])
        rows.append(
            {
                "Severity": "Info",
                "Topic": "Semester load",
                "Note": f"{len(overloaded_terms)} term(s) exceed 30 LP: {labels}.",
            }
        )
    if candidates:
        rows.append(
            {
                "Severity": "Info",
                "Topic": "Possible Candidates",
                "Note": f"{len(candidates)} candidate course(s), {_format_credit_value(sum(module.cp for module in candidates))}.",
            }
        )
    if rule_errors or rule_warnings:
        rows.append(
            {
                "Severity": "Issue",
                "Topic": "Degree checks",
                "Note": f"{rule_errors} error(s), {rule_warnings} warning(s) across exported degree checks.",
            }
        )
    return rows


def _llm_course_rows(modules: list[Module]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for module in sorted(modules, key=lambda module: term_sort_key(module.term or "Unknown") + (module.name.lower(),)):
        rows.append(
            {
                "Course": module.name,
                "Program": module.program_key or "-",
                "Semester": module.term or "Unknown",
                "Status": module.state.value,
                "Credits": module.cp,
                "Grade": module.grade,
                "Estimated Grade": module.estimated_grade,
                "Area": module.area,
                "Topic Clusters": ", ".join(module_topics(module)),
                "Keywords": ", ".join(module_keywords(module)),
                "Source": module.source.value,
                "Institution": module.institution or "TU Berlin",
                "Course URL": module.url or "",
                "GitHub URL": module.github_url or "",
            }
        )
    return rows


def build_study_plan_insights(
    modules: list[Module],
    *,
    degree_summaries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    degree_summaries = degree_summaries or []
    non_candidate = [module for module in modules if not is_possible_course(module)]
    open_modules = [module for module in non_candidate if module.state != ModuleState.COMPLETED]
    topic_rows = _topic_rows(modules)
    unofficial_grade_rows = [
        {
            "Metric": "Completed weighted average",
            "Value": _grade_average(non_candidate, weighted=True, include_estimates=False),
            "Scope": "Visible completed graded courses; no official discard",
        },
        {
            "Metric": "Completed unweighted average",
            "Value": _grade_average(non_candidate, weighted=False, include_estimates=False),
            "Scope": "Visible completed graded courses; every course counts equally",
        },
        {
            "Metric": "Forecast weighted average",
            "Value": _grade_average(non_candidate, weighted=True, include_estimates=True),
            "Scope": "Visible graded courses with final or estimated grades; no official discard",
        },
        {
            "Metric": "Forecast unweighted average",
            "Value": _grade_average(non_candidate, weighted=False, include_estimates=True),
            "Scope": "Visible graded courses with final or estimated grades; every course counts equally",
        },
        {
            "Metric": "Open graded credits without estimate",
            "Value": _format_credit_value(
                sum(
                    module.cp
                    for module in open_modules
                    if module.is_graded and module.estimated_grade is None
                )
            ),
            "Scope": "Forecast data quality",
        },
    ]

    return {
        "unofficial_grade_rows": unofficial_grade_rows,
        "topic_rows": topic_rows,
        "subtopic_rows": _subtopic_rows(modules),
        "risk_rows": _risk_rows(modules, degree_summaries),
        "llm_course_rows": _llm_course_rows(modules),
    }
