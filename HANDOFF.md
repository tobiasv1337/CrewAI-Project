# TU Study Assistant — CrewAI Project Handoff & Implementation Plan

> **This document is the complete handoff for building the TU Study Assistant in a new repository.**
> It contains everything a coding agent or developer needs to know — from architecture to API details to implementation phases.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [What to Copy into the New Repo](#2-what-to-copy-into-the-new-repo)
3. [MVP Implementation Phases](#3-mvp-implementation-phases)
4. [Moses Scraper — Complete Documentation](#4-moses-scraper--complete-documentation)
5. [ISIS (Moodle) API — Complete Extraction Guide](#5-isis-moodle-api--complete-extraction-guide)
6. [Complete Tool Catalog](#6-complete-tool-catalog)
7. [Final Agent Design](#7-final-agent-design)
8. [Architecture & Target File Structure](#8-architecture--target-file-structure)
9. [Grade Manager — Data Model & Integration](#9-grade-manager--data-model--integration)
10. [LLM Configuration](#10-llm-configuration)
11. [UI Plan (Streamlit Chat)](#11-ui-plan-streamlit-chat)
12. [Key Decisions & Rationale](#12-key-decisions--rationale)

---

## 1. Project Overview

### What is this?
A **CrewAI multi-agent system** for TU Berlin students that combines three data sources into an intelligent study assistant:

| Data Source | What it provides | Tech |
|-------------|-----------------|------|
| **Moses** (module catalog) | Module search, descriptions, LP, exam format, prerequisites, catalog assignments | HTML scraping (JSF) via urllib |
| **ISIS** (Moodle) | Enrolled courses, assignments, deadlines, forum, grades, announcements | Playwright login + Moodle REST API |
| **Grade Manager** (existing app) | Student's completed modules, GPA, remaining requirements, study plan | Pydantic models + JSON persistence |

### Seminar Context
- **Course**: TU Berlin seminar on "Multi-Agent LLM Frameworks"
- **Framework**: CrewAI (mandatory)
- **Requirements**:
  - Must have **3-4+ agents** collaborating (not just 1 agent with tools)
  - Must be a **functional demo** (not a concept)
  - Must demonstrate **multi-agent orchestration** value
- **Timeline**: ~2-3 weeks

### The End Vision
```
Student: "Was soll ich nächstes Semester belegen? Ich interessiere mich für ML."

System (4 agents collaborating):
  → Study Advisor checks student's progress: 78/120 LP, missing 12 LP in Elective
  → Study Advisor checks what student already took: ML1, Deep Learning ✓
  → Module Researcher searches Moses for ML modules, filters already-completed ones
  → Module Researcher checks catalog fit for remaining modules
  → Course Info Agent checks ISIS for schedule/deadline conflicts
  → Orchestrator combines everything:

  "Du hast 78 von 120 LP im M.Sc. Informatik. Dir fehlen noch 12 LP im
   Wahlbereich. Du hast bereits ML1 und Deep Learning abgeschlossen.

   Passende Module für WS 26/27:
   1. Machine Learning 2 (6 LP) — Wahlbereich ✅
   2. Reinforcement Learning (6 LP) — Wahlbereich ✅

   Soll ich eines davon in deinen Studienplan eintragen?"

Student: "Ja, trag bitte RL ein für WS 26/27."

  → Study Advisor calls add_planned_module() → Module appears in Grade Manager
```

---

## 2. What to Copy into the New Repo

### Recommendation: Copy (Almost) Everything

Copy the entire TU Notenmanager **except** `.venv/`, `tests/`, `__pycache__/`, `.git/`, and `.agents/`. This way the new repo contains the complete target picture — including the UI where the chat will eventually be integrated.

### Exact Copy List

```
COPY these:
├── app.py                        # Streamlit main app (the integration target)
├── requirements.txt              # Base dependencies
├── README.md                     # Keep for reference
├── .gitignore                    # Adapt for new repo
├── .streamlit/
│   └── config.toml               # Streamlit config
├── assets/
│   ├── style.css                 # 54KB custom CSS
│   └── tu-berlin-logo.svg        # TU Berlin logo
├── core/                         # THE COMPLETE CORE — all of it
│   ├── __init__.py
│   ├── models.py                 # Pydantic models (shared by everything)
│   ├── terms.py                  # Semester parsing
│   ├── registry.py               # Degree program registry
│   ├── interfaces.py             # DegreeStrategy protocol
│   ├── manager.py                # DegreeManager facade
│   ├── persistence.py            # load/save modules, profiles
│   ├── analytics.py              # Progress stats, area distribution
│   ├── calculations.py           # GPA calculation with Streichliste
│   ├── calculation_variants.py   # Calculation variant helpers
│   ├── grade_targets.py          # What-if grade analysis
│   ├── module_filters.py         # Module filtering helpers
│   ├── module_ids.py             # ID generation
│   ├── module_origin.py          # Source/institution inference
│   ├── projections.py            # Semester projections
│   ├── rules.py                  # Degree validation rules
│   ├── scenarios.py              # Grade scenarios enum
│   ├── impl/                     # Degree program implementations
│   │   └── tu_berlin/
│   │       ├── computer_science_master.py
│   │       ├── media_informatics_master.py
│   │       ├── medientechnik_bachelor.py
│   │       └── technische_informatik_bachelor.py
│   └── providers/
│       └── tu_berlin/
│           └── moses.py          # THE MOSES SCRAPER (2587 lines)
├── ui/                           # Complete UI (integration target)
│   ├── dashboard.py
│   ├── details.py
│   ├── manual_add.py
│   ├── modules.py
│   ├── moses.py
│   ├── program_labels.py
│   ├── registrations.py
│   ├── settings.py
│   ├── study_plan_export.py
│   ├── study_plan_export_ui.py
│   ├── study_plan_insights.py
│   ├── term_controls.py
│   ├── timeline.py
│   └── components/
│       └── timeline_board/
└── data/                         # Student data (gitignored, but keep structure)
    └── profiles/
        └── (user's modules.json lives here)

DO NOT COPY:
├── .venv/                        # Rebuild in new repo
├── .git/                         # New repo, new git history
├── __pycache__/                  # Auto-generated
├── .agents/                      # Agent config, not needed
├── tests/                        # Rebuild for new project
└── skills-lock.json              # Agent tooling config
```


### After Copying: Add New Directories

```
NEW directories to create in the new repo:
├── crew/                         # CrewAI package (ALL NEW CODE GOES HERE)
│   ├── __init__.py
│   ├── config.py
│   ├── isis_client.py
│   ├── crew.py
│   ├── agents.yaml
│   ├── tasks.yaml
│   └── tools/
│       ├── __init__.py
│       ├── moses_tools.py
│       ├── isis_tools.py
│       └── grademanager_tools.py
├── pages/                        # Streamlit multipage (Phase 4+)
│   └── 3_💬_Study_Chat.py
├── tests/                        # New test suite
│   ├── test_moses_tools.py
│   ├── test_isis_client.py
│   └── test_crew.py
└── main.py                       # CLI entry point (for testing without Streamlit)
```

### Updated .env (needed)
```bash
# ISIS/Moodle credentials
ISIS_USERNAME=your_tub_username
ISIS_PASSWORD=your_tub_password

# LLM Configuration (using GWDG Open-Weight Models from the start)
GWDG_API_KEY=your_gwdg_api_token
GWDG_API_BASE=https://chat-ai.hpc.gwdg.de/v1
STUDY_ASSISTANT_MODEL=devstral-2-123b-instruct-2512
```

### Updated requirements.txt
```
# Existing
streamlit>=1.56.0
pydantic
pandas
plotly
matplotlib
beautifulsoup4

# New for CrewAI
crewai
crewai-tools
playwright
requests
python-dotenv
```

---

## 3. MVP Implementation Phases

### Phase 1: Moses Tools — No LLM (1-2 days) 🏗️

**Goal**: Get Moses working as clean, testable tool functions.

**Steps**:
1. Set up new repo with copied files
2. `uv pip install -r requirements.txt`
3. Create `crew/tools/moses_tools.py` — 3 functions wrapped as plain Python (test without @tool first)
4. Create `tests/test_moses_tools.py` — verify search, details, catalogs
5. Run tests, fix any import issues from the copy

**Test script** (`main.py`):
```python
from crew.tools.moses_tools import search_modules, get_module_details, get_module_catalogs

# Test 1: Search
print(search_modules("Machine Learning"))

# Test 2: Details
print(get_module_details("40966", 2))

# Test 3: Catalogs
print(get_module_catalogs("40966", 2))
```

**Deliverable**: 3 Moses tool functions working and tested.

---

### Phase 2: First CrewAI Agent (1-2 days) 🤖

**Goal**: Single agent using Moses tools via CrewAI.

**Steps**:
1. Add `@tool` decorators to Moses functions
2. Create `crew/crew.py` with one agent (Module Researcher)
3. Create a simple crew with one task
4. Test: `crew.kickoff(inputs={"query": "Find Machine Learning modules"})`
5. Iterate on prompts until the agent reliably uses the tools

**Deliverable**: Working single-agent CrewAI system.

---

### Phase 3A: ISIS Client (2-3 days) 🔐

**Goal**: ISIS API client that authenticates and retrieves course data.

**Steps**:
1. `uv pip install playwright && uv run playwright install chromium`
2. Build `crew/isis_client.py` (see [Section 5](#5-isis-moodle-api--complete-extraction-guide))
3. Create `crew/tools/isis_tools.py` with @tool wrappers
4. Test: login → enrolled courses → assignments → grades
5. Add 2nd agent (Course Info Specialist)

**Deliverable**: Working ISIS tools with real Moodle data.

---

### Phase 3B: Multi-Agent System (2-3 days) 🤝

**Goal**: 3-4 agents collaborating via CrewAI hierarchical process.

**Steps**:
1. Add Study Advisor agent (Moses + Grade Manager read tools)
2. Add Orchestrator agent (manager in hierarchical mode)
3. Configure `Process.hierarchical` crew
4. Test multi-agent scenarios:
   - "What ML modules exist?" → Module Researcher only
   - "What's due this week?" → Course Info only
   - "Plan my next semester" → Study Advisor + Module Researcher

**Deliverable**: Multi-agent crew handling diverse queries.

---

### Phase 4: Grade Manager Integration (2-3 days) 📊

**Goal**: Agents can read student progress AND write to study plan.

**Steps**:
1. Create `crew/tools/grademanager_tools.py`
2. Read tools: `get_student_progress`, `get_completed_modules`, `get_missing_requirements`
3. Write tool: `add_planned_module` (adds to `modules.json` as "Planned")
4. Connect to existing `persistence.py` + `manager.py`
5. Test: "Add RL to my plan for WS 26/27" → appears in Grade Manager

**Deliverable**: Full bidirectional integration.

---

### Phase 5: Streamlit Chat UI (2-3 days) ✨

**Goal**: Beautiful chat interface in the existing Streamlit app.

**Steps**:
1. Create `pages/3_💬_Study_Chat.py`
2. Chat input/output with `st.chat_input()` / `st.chat_message()`
3. ISIS login widget in sidebar
4. Connect chat to `crew.kickoff()`
5. Session state for conversation history

**Deliverable**: Integrated chat page in the Grade Manager app.

---

### Phase 6: Polish & Demo (2 days) 🎬

**Goal**: Demo-ready, error-handled, presentation-worthy.

**Steps**:
1. Error handling for all tools (network failures, auth expiry)
2. Output formatting (markdown, tables, emojis)
3. Demo scenarios scripted and tested
4. README for the seminar submission
5. Testing and Optimization of GWDG Open-Weight Models

---

## 4. Moses Scraper — Complete Documentation

### Overview

File: `core/providers/tu_berlin/moses.py` (2587 lines)
Website: [Moses Module Transfer System](https://moseskonto.tu-berlin.de/moses/modultransfersystem/bolognamodule/suchen.html)

### Should We Rewrite Moses? → **No, keep as-is.**

| Option | Verdict | Reason |
|--------|---------|--------|
| Keep as-is + wrap as @tool | ✅ **Recommended** | Works perfectly, 2587 lines of battle-tested code, no auth needed |
| Rewrite with Playwright | ❌ | Overkill — Moses doesn't need JS rendering, current urllib approach works fine |
| Rewrite with requests | ❌ | Would break JSF form handling (needs cookie jar, ViewState tracking) |

The Moses scraper is **fully functional and stable**. Just wrap the 3 key functions as `@tool` and move on to the harder problems (ISIS, multi-agent).

### How Moses Works Internally

Moses uses **JSF (JavaServer Faces)** — a stateful Java web framework. This means:
1. Every page load sets a `ViewState` and `ClientWindow` token
2. Searches are **partial AJAX POST requests** (not simple GET requests)
3. The scraper maintains a session with cookies across requests

```
search_courses("Machine Learning")
  → GET moseskonto.tu-berlin.de/.../suchen.html?sprache=en
  → Parse HTML: extract form_id, ViewState, ClientWindow, input names
  → POST (Faces-Request: partial/ajax) with search query
  → Parse XML response → extract updated results HTML
  → Parse results table → list[MosesSearchResult]

fetch_course_details("40966", 2)
  → GET .../beschreibung/anzeigen.html?nummer=40966&version=2
  → Parse full detail page HTML
  → Extract: title, LP, exam type, prerequisites, learning outcomes,
     workload, module elements, degree usages, catalog assignments
  → Return MosesModuleData
```

### Key Public Functions

```python
# SEARCH — returns list of matching modules
search_courses(query: str, max_results: int = 20) -> list[MosesSearchResult]

# DETAILS — full module information by number + version
fetch_course_details(number: str, version: int) -> MosesModuleData

# DETAILS BY URL — same but from a Moses URL
fetch_course_details_from_url(url: str) -> MosesModuleData

# DESCRIPTION — human-readable markdown summary
build_moses_description(data: MosesModuleData) -> Optional[str]

# AREA SUGGESTION — which study area this module fits
suggest_area_for_module(program_key: str, data: MosesModuleData) -> Optional[str]

# URL PARSING — extract number+version from a Moses URL
parse_number_version_from_url(url: str) -> Optional[tuple[str, int]]

# MODULE CREATION — create a Grade Manager Module from Moses data
create_module_from_moses_data(data, program_key, area, state, module_id, term) -> Module
```

### Key Data Models (from `core/models.py`)

```python
class MosesSearchResult:
    number: str          # "40966"
    version: int         # 2
    title: str           # "Machine Learning 1"
    detail_url: str      # Full URL to detail page
    languages: list[str] # ["English"]
    credits: float       # 6.0
    grading_mode: str    # "Prüfungsäquivalente Studienleistungen"
    responsible_person: str
    department: str

class MosesModuleData:
    number: str
    version: int
    title: str
    credits: float
    validity: str                    # "WS 2024/25 onwards"
    responsible_person: str
    grading_mode: str
    exam_type: str
    teaching_languages: list[str]
    faculty: str
    learning_outcomes: str           # RICH TEXT
    contents: str                    # RICH TEXT
    teaching_and_learning_methods: str
    prerequisites: str
    exam_description: str
    offered_in: ModuleOffering       # WS_ONLY, SS_ONLY, or BOTH
    module_elements: list[MosesModuleElement]   # VL, UE, TUT, etc.
    workload_items: list[MosesWorkloadItem]
    exam_elements: list[MosesExamElement]
    degree_usages: list[MosesDegreeUsage]
    normalized_catalogs_by_program: dict[str, list[str]]  # KEY for degree planning
```

### Important: Degree Program Keys

The Moses scraper and Grade Manager use these exact program keys:
```python
PROGRAM_REGISTRY = {
    "TU Berlin - Computer Science (M.Sc.)":        "...:TUBerlinComputerScienceMaster",
    "TU Berlin - Medieninformatik (M.Sc.)":         "...:TUBerlinMediaInformaticsMaster",
    "TU Berlin - Medientechnik (B.Sc.)":            "...:TUBerlinMedientechnikBachelor",
    "TU Berlin - Technische Informatik (B.Sc.)":    "...:TUBerlinTechnischeInformatikBachelor",
}
```

These keys are used in `normalized_catalogs_by_program`, in `Module.program_key`, and in all Grade Manager operations.

---

## 5. ISIS (Moodle) API — Complete Extraction Guide

### Authentication Flow (TESTED & VERIFIED ✅)

ISIS uses **Shibboleth SSO**. Standard Moodle `login/token.php` does **NOT work**. We use **Playwright** to perform the browser login, then extract a `wstoken` for REST API access.

#### Step-by-Step Login Flow

```
Step 1: Navigate to https://isis.tu-berlin.de/login/index.php
        → ISIS login page

Step 2: Accept cookie consent ("Fortsetzen" button)
        → Only on first visit

Step 3: Click "TU-Login" button
        → Redirects to shibboleth.tubit.tu-berlin.de

Step 4: Fill Shibboleth login form:
        - name="j_username" → TUB-Kontoname
        - name="j_password" → Password
        - Submit: name="_eventId_proceed"

Step 5: Wait for SAML redirect chain:
        → POST /Shibboleth.sso/SAML2/POST-SimpleSign (302)
        → /auth/shibboleth/index.php (303)
        → /my/ (200) — Dashboard = LOGIN SUCCESS

Step 6: Extract cookies from browser:
        - MoodleSession (session cookie)
        - _shibsession_... (Shibboleth, long hex name)

Step 7: Extract wstoken via mobile launch endpoint:
        GET /admin/tool/mobile/launch.php?service=moodle_mobile_app&passport=x
        ⚠️ MUST use allow_redirects=False!
        → 303 Location: moodlemobile://token=<BASE64>
        → Decode: base64 → "privatetoken:::wstoken:::siteid"
        → wstoken is the middle part (32 hex chars)
```

> **CRITICAL**: The mobile launch endpoint redirects to `moodlemobile://` URL scheme. Using `allow_redirects=True` with `requests` will **CRASH** with `InvalidSchema`. Always use `allow_redirects=False` and read the `Location` header.

#### Complete Working Login Code

```python
import asyncio, base64, re
import requests
from playwright.async_api import async_playwright

ISIS_BASE = "https://isis.tu-berlin.de"
ISIS_API  = f"{ISIS_BASE}/webservice/rest/server.php"

async def isis_login(username: str, password: str) -> tuple[str, dict]:
    """
    Login to ISIS via Playwright (Shibboleth SSO).
    Returns: (wstoken, cookies_dict)
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx  = await browser.new_context()
        page = await ctx.new_page()

        # Navigate to ISIS login
        await page.goto(f"{ISIS_BASE}/login/index.php")
        await page.wait_for_load_state("domcontentloaded")

        # Accept cookie consent if visible
        try:
            btn = page.get_by_text("Fortsetzen", exact=True)
            if await btn.is_visible(timeout=3000):
                await btn.click()
                await page.wait_for_load_state("domcontentloaded")
        except:
            pass

        # Click "TU-Login" → Shibboleth
        await page.get_by_text("TU-Login", exact=True).click()

        # Fill Shibboleth credentials
        await page.wait_for_selector('[name="j_username"]', timeout=20000)
        await page.fill('[name="j_username"]', username)
        await page.fill('[name="j_password"]', password)
        await page.click('[name="_eventId_proceed"]')

        # Wait for redirect to ISIS dashboard
        await page.wait_for_url(f"{ISIS_BASE}/my/**", timeout=20000)

        # Extract cookies
        cookies = {c["name"]: c["value"]
                   for c in await ctx.cookies()
                   if "isis" in c["domain"]}
        await browser.close()

    # Extract wstoken via mobile launch
    session = requests.Session()
    for k, v in cookies.items():
        session.cookies.set(k, v, domain="isis.tu-berlin.de")

    resp = session.get(
        f"{ISIS_BASE}/admin/tool/mobile/launch.php",
        params={"service": "moodle_mobile_app", "passport": "x"},
        allow_redirects=False,  # CRITICAL: moodlemobile:// would crash requests
    )
    location = resp.headers.get("Location", "")
    token_b64 = re.search(r"token=([A-Za-z0-9+/=]+)", location).group(1)
    raw = base64.b64decode(token_b64).decode("utf-8")
    parts = raw.split(":::")
    wstoken = parts[1] if len(parts) >= 2 else parts[0]

    return wstoken, cookies


def isis_api(wstoken: str, function: str, **params) -> dict:
    """Call any Moodle REST API function."""
    r = requests.get(ISIS_API, params={
        "wstoken": wstoken,
        "wsfunction": function,
        "moodlewsrestformat": "json",
        **params,
    })
    return r.json()
```

### All Tested & Working API Endpoints

#### 1. Get User Info (site_info)
```python
info = isis_api(wstoken, "core_webservice_get_site_info")
user_id = info["userid"]  # Needed for other calls
# Returns: sitename, username, firstname, lastname, fullname, userid, lang
```

#### 2. Get Enrolled Courses
```python
courses = isis_api(wstoken, "core_enrol_get_users_courses", userid=user_id)
# Returns: list of dicts with {id, fullname, shortname, visible, ...}
# TESTED: 50+ courses returned
# Example: {"id": 45924, "fullname": "[WiSe 2025/26] Machine Learning 1", ...}
```

#### 3. Search All Courses
```python
result = isis_api(wstoken, "core_course_search_courses",
                  criterianame="search", criteriavalue="Machine Learning", perpage=10)
# Returns: {"total": 399, "courses": [...]}
# Searches ALL ISIS courses, not just enrolled ones
```

#### 4. Get Course Contents (sections + modules)
```python
contents = isis_api(wstoken, "core_course_get_contents", courseid=COURSE_ID)
# Returns: list of sections, each with "modules" list
# Each module: {modname: "forum"/"folder"/"assign"/..., name: "...", url: "..."}
# Example section: "Organisatorisches" with modules [Announcements, Unterlagen, ...]
```

#### 5. Get Assignments (homework + deadlines)
```python
assignments = isis_api(wstoken, "mod_assign_get_assignments",
                       **{"courseids[0]": COURSE_ID})
# Returns: {"courses": [{"assignments": [{name, intro, duedate, grade}, ...]}]}
# duedate is Unix timestamp (convert with datetime.fromtimestamp())
# grade is max points
# TESTED: "Abgabe Projektergebnisse" Due: 2024-03-01, max grade: 60
```

#### 6. Get Forums + Discussions
```python
# Get all forums in a course
forums = isis_api(wstoken, "mod_forum_get_forums_by_courses",
                  **{"courseids[0]": COURSE_ID})
# Returns: [{id, name, type, intro}, ...]
# type: "news" = announcements, "general" = regular forum

# Get discussions in a specific forum
discussions = isis_api(wstoken, "mod_forum_get_forum_discussions",
                       forumid=FORUM_ID, perpage=5)
# Returns: {"discussions": [{name, userfullname, timemodified, message}, ...]}
# TESTED: "Noten und Board Rückgabe" by Nicolai Stawinoga (2024-08-14)
```

#### 7. Get Grades
```python
grades = isis_api(wstoken, "gradereport_user_get_grade_items",
                  courseid=COURSE_ID, userid=user_id)
# Returns: {"usergrades": [{"gradeitems": [{itemname, gradeformatted, grademax}, ...]}]}
# TESTED: "Abgabe Projektergebnisse" Grade: - / 60
# Note: itemname can be None for "Course total" — handle with: name = item.get("itemname") or "Course total"
```

#### 8. Get Calendar Events
```python
import time
now = int(time.time())
events = isis_api(wstoken, "core_calendar_get_calendar_events",
                  **{"events[courseids][0]": COURSE_ID,
                     "options[timestart]": now,
                     "options[timeend]": now + 86400*180})
# Returns: {"events": [{name, timestart, eventtype}, ...]}
# Note: Array parameters use bracket notation: events[courseids][0]
```

#### 9. Course Details (enrolled courses only!)
```python
detail = isis_api(wstoken, "core_course_get_courses",
                  **{"options[ids][0]": COURSE_ID})
# Returns: [{summary, startdate, enddate, categoryid}, ...]
# ⚠️ FAILS for courses you're NOT enrolled in: "kein Recht"
```

### What Does NOT Work

| Method | Issue |
|--------|-------|
| `login/token.php` (standard Moodle) | ❌ "Ungültige Anmeldung" — Shibboleth blocks it |
| REST API with session cookie only (no wstoken) | ❌ "Ungültiges Token" |
| `core_course_get_courses` for non-enrolled courses | ❌ Permission denied |

### Rate Limiting
Add **1.2-2 second delays** between API calls. No strict limits observed, but be respectful — it's a university server.

---

## 6. Complete Tool Catalog

### @tool vs MCP → **Use @tool (CrewAI native)**

| | @tool (CrewAI) | MCP (FastMCP) |
|---|---|---|
| **Complexity** | Low — plain Python function | High — separate process, stdio, JSON-RPC |
| **Debugging** | Easy — same process | Hard — cross-process |
| **Dev speed** | Fast | Slow |
| **Multi-agent value** | Same | Same — MCP is a tool protocol, not agent orchestration |
| **Seminar grade** | Comes from agent collaboration | Not from tool protocol |

**Decision**: Use `@tool`. The seminar evaluates multi-agent orchestration, not tool protocols. If time permits, wrapping the same functions as MCP servers is a 1-day stretch goal (just add `@mcp.tool()` to the same functions).

### Moses Tools (3 tools — wrap existing `moses.py`)

#### `search_modules`
```python
@tool("Search TU Berlin Modules")
def search_modules(query: str) -> str:
    """Search the TU Berlin module catalog (Moses) by keyword.
    Returns module numbers, titles, credits, and exam formats.
    Use this to find modules about a topic like 'Machine Learning' or 'Robotics'.
    Args:
        query: Search keyword(s) for the module catalog.
    """
    from core.providers.tu_berlin.moses import search_courses
    results = search_courses(query, max_results=10)
    if not results:
        return f"No modules found for '{query}'."
    lines = [f"Found {len(results)} module(s) for '{query}':\n"]
    for r in results:
        lines.append(
            f"- [{r.number} v{r.version}] {r.title}\n"
            f"  Credits: {r.credits} LP | Exam: {r.grading_mode or '?'}\n"
            f"  Responsible: {r.responsible_person or '?'} | Languages: {', '.join(r.languages)}"
        )
    return "\n".join(lines)
```

#### `get_module_details`
```python
@tool("Get Module Details")
def get_module_details(module_number: str, version: int) -> str:
    """Get complete details for a specific TU Berlin module from Moses.
    Includes learning outcomes, contents, prerequisites, exam format, workload.
    Args:
        module_number: The Moses module number (e.g., '40966').
        version: The module version (e.g., 2).
    """
    from core.providers.tu_berlin.moses import fetch_course_details, build_moses_description
    data = fetch_course_details(module_number, version)
    desc = build_moses_description(data) or ""
    header = (
        f"# {data.title} ({data.number} v{data.version})\n"
        f"Credits: {data.credits} LP | Exam: {data.exam_type or data.grading_mode}\n"
        f"Offered: {data.offered_in.value} | Languages: {', '.join(data.teaching_languages)}\n"
        f"Responsible: {data.responsible_person}\n\n"
    )
    return header + desc
```

#### `get_module_catalogs`
```python
@tool("Check Module Catalog Fit")
def get_module_catalogs(module_number: str, version: int) -> str:
    """Check which degree programs and catalog areas a module counts for.
    Important for knowing if a module fits into a specific study plan.
    Args:
        module_number: The Moses module number (e.g., '40966').
        version: The module version (e.g., 2).
    """
    from core.providers.tu_berlin.moses import fetch_course_details
    data = fetch_course_details(module_number, version)
    lines = [f"Catalog assignments for {data.title} ({data.number} v{data.version}):\n"]
    for program, catalogs in data.normalized_catalogs_by_program.items():
        lines.append(f"  {program}:")
        for cat in catalogs:
            lines.append(f"    - {cat}")
    if not data.normalized_catalogs_by_program:
        lines.append("  No catalog assignments found.")
    return "\n".join(lines)
```

### ISIS Tools (6 tools — new `isis_client.py`)

#### `search_isis_courses`
```python
@tool("Search ISIS Courses")
def search_isis_courses(query: str) -> str:
    """Search for courses on ISIS (TU Berlin Moodle) by keyword.
    Returns course IDs and names. Use course IDs for other ISIS tools.
    Args:
        query: Search term (e.g., 'Machine Learning', 'Seminar AI Agents').
    """
```

#### `get_enrolled_courses`
```python
@tool("Get My Enrolled ISIS Courses")
def get_enrolled_courses() -> str:
    """List all ISIS courses the student is currently enrolled in.
    Returns course IDs, names, and shortnames.
    No arguments needed.
    """
```

#### `get_course_assignments`
```python
@tool("Get Course Assignments and Deadlines")
def get_course_assignments(course_id: int) -> str:
    """Get all assignments/homework for an ISIS course with deadlines.
    Args:
        course_id: The ISIS course ID (from search or enrolled courses).
    """
```

#### `get_course_announcements`
```python
@tool("Get Course Announcements")
def get_course_announcements(course_id: int) -> str:
    """Get recent announcements from an ISIS course (news forum).
    Args:
        course_id: The ISIS course ID.
    """
```

#### `get_course_grades`
```python
@tool("Get My Course Grades")
def get_course_grades(course_id: int) -> str:
    """Get the student's current grades for an ISIS course.
    Args:
        course_id: The ISIS course ID.
    """
```

#### `get_course_forum`
```python
@tool("Get Course Forum Discussions")
def get_course_forum(course_id: int) -> str:
    """Get recent forum discussions from an ISIS course.
    Args:
        course_id: The ISIS course ID.
    """
```

### Grade Manager Tools (5 tools — wrap existing `persistence.py` + `manager.py`)

These tools give agents access to the student's actual academic data.

#### `get_student_progress` (READ)
```python
@tool("Get Student Study Progress")
def get_student_progress(program_key: str = "") -> str:
    """Get the student's study progress: completed LP, current GPA,
    LP breakdown by area, and overall completion percentage.
    Args:
        program_key: Optional. Degree program to check (e.g., 'TU Berlin - Computer Science (M.Sc.)').
                     If empty, uses the primary program.
    """
    from core.persistence import load_modules, load_profiles
    from core.manager import DegreeManager
    from core.registry import create_program, list_programs
    from core.analytics import progress_stats, area_distribution

    profiles = load_profiles()
    primary = next(p for p in profiles if p.is_primary)
    modules = load_modules(primary.slug)

    if not program_key:
        program_key = list_programs()[0]

    mgr = DegreeManager(create_program(program_key))
    deg_modules = mgr.filter_degree_modules(modules)
    result = mgr.calculate(modules)
    stats = progress_stats(deg_modules)
    validations = mgr.validate(modules)

    lines = [
        f"Study Progress for {program_key}:",
        f"  GPA: {result.final_grade:.2f}",
        f"  Completed: {stats['completed_cp']:.0f} / {mgr.get_total_cp_required():.0f} LP ({stats['percent']:.0f}%)",
        f"\nArea breakdown:",
    ]
    for row in area_distribution(deg_modules):
        lines.append(f"  {row['Area']}: {row['Credits']:.0f} LP")

    failed = [v for v in validations if not v.satisfied]
    if failed:
        lines.append(f"\nMissing requirements ({len(failed)}):")
        for v in failed:
            lines.append(f"  ❌ {v.rule_name}: {v.message}")
    else:
        lines.append("\n✅ All degree requirements satisfied!")

    return "\n".join(lines)
```

#### `get_completed_modules` (READ)
```python
@tool("Get Completed Modules")
def get_completed_modules(program_key: str = "") -> str:
    """Get all modules the student has completed, with grades, areas, and terms.
    Useful for understanding what the student has already taken.
    Args:
        program_key: Optional. Filter by degree program.
    """
```

#### `get_planned_modules` (READ)
```python
@tool("Get Planned Modules")
def get_planned_modules(program_key: str = "") -> str:
    """Get all modules the student plans to take or is currently taking.
    Args:
        program_key: Optional. Filter by degree program.
    """
```

#### `get_missing_requirements` (READ)
```python
@tool("Get Missing Degree Requirements")
def get_missing_requirements(program_key: str = "") -> str:
    """Check which degree requirements are not yet satisfied.
    Returns specific rules that still need to be fulfilled (e.g., 'need 6 more LP in Elective').
    Args:
        program_key: Optional. Degree program to check.
    """
```

#### `add_planned_module` (WRITE — creates a new Module in modules.json)
```python
@tool("Add Module to Study Plan")
def add_planned_module(module_number: str, version: int, area: str, term: str) -> str:
    """Add a Moses module to the student's study plan as 'Planned'.
    This creates a new entry in the Grade Manager.
    ONLY call this after the student explicitly confirms they want to add it.
    Args:
        module_number: Moses module number (e.g., '40966').
        version: Moses module version (e.g., 2).
        area: Study area (e.g., 'Elective', 'Mandatory', 'Free Choice').
        term: Semester label (e.g., 'WS 26/27' or 'SS 27').
    """
    from core.providers.tu_berlin.moses import (
        fetch_course_details, create_module_from_moses_data, suggest_area_for_module,
    )
    from core.persistence import load_modules, save_modules, load_profiles
    from core.module_ids import generate_module_id  # or uuid4
    import uuid

    # Fetch Moses data
    data = fetch_course_details(module_number, version)

    # Determine program key (use primary profile's most common program)
    profiles = load_profiles()
    primary = next(p for p in profiles if p.is_primary)
    modules = load_modules(primary.slug)

    program_key = "TU Berlin - Computer Science (M.Sc.)"  # or detect from existing modules

    # Create the module
    new_module = create_module_from_moses_data(
        data,
        program_key=program_key,
        area=area,
        state="Planned",
        module_id=str(uuid.uuid4()),
        term=term,
    )

    # Add and save
    modules.append(new_module)
    save_modules(modules, primary.slug)

    return f"✅ Added '{data.title}' ({data.credits} LP) as Planned for {term} in {area}."
```

#### `check_module_fits_degree` (READ — combines Moses + Grade Manager)
```python
@tool("Check if Module Fits Degree")
def check_module_fits_degree(module_number: str, version: int) -> str:
    """Check if a specific module fits into the student's degree program.
    Shows which catalog area it would count for and whether the student already has it.
    Args:
        module_number: Moses module number.
        version: Moses module version.
    """
    from core.providers.tu_berlin.moses import (
        fetch_course_details, suggest_area_for_module,
        find_existing_module_by_moses_identity_any_program,
    )
    from core.persistence import load_modules, load_profiles

    data = fetch_course_details(module_number, version)

    profiles = load_profiles()
    primary = next(p for p in profiles if p.is_primary)
    modules = load_modules(primary.slug)

    # Check if already in study plan
    existing = find_existing_module_by_moses_identity_any_program(
        modules, number=module_number, version=version
    )

    lines = [f"Module: {data.title} ({data.number} v{data.version}, {data.credits} LP)\n"]

    if existing:
        lines.append(f"⚠️ Already in study plan as '{existing.state.value}' in {existing.area}")
        if existing.grade:
            lines.append(f"   Grade: {existing.grade}")
    else:
        lines.append("✅ Not yet in study plan.")

    # Check catalog fit for each registered program
    for program_key, catalogs in data.normalized_catalogs_by_program.items():
        area = suggest_area_for_module(program_key, data)
        lines.append(f"\n{program_key}:")
        lines.append(f"  Suggested area: {area}")
        lines.append(f"  Catalogs: {', '.join(catalogs) if catalogs else 'none'}")

    return "\n".join(lines)
```

---

## 7. Final Agent Design

### Agent 1: Module Researcher 🔍

| Property | Value |
|----------|-------|
| **Role** | TU Berlin Module Researcher |
| **Goal** | Find and analyze modules from the TU Berlin module catalog (Moses) that match the student's interests and degree requirements |
| **Backstory** | Expert on the TU Berlin module catalog. Knows how to search for modules, analyze their content, prerequisites, and exam formats, and determine which degree programs they count for. |
| **Tools** | `search_modules`, `get_module_details`, `get_module_catalogs`, `check_module_fits_degree` |
| **When used** | "What ML modules exist?", "Tell me about module 40966", "Does this module fit my Elective area?" |

### Agent 2: Course Info Specialist 📚

| Property | Value |
|----------|-------|
| **Role** | ISIS Course Information Specialist |
| **Goal** | Retrieve practical information from ISIS (Moodle): assignments, deadlines, announcements, forum, grades |
| **Backstory** | ISIS/Moodle expert who can find any practical information about active courses — from deadlines to announcements and grades. |
| **Tools** | `search_isis_courses`, `get_enrolled_courses`, `get_course_assignments`, `get_course_announcements`, `get_course_grades`, `get_course_forum` |
| **When used** | "What's due this week?", "Any new announcements in ML1?", "Show my grades" |

### Agent 3: Study Advisor 🎓

| Property | Value |
|----------|-------|
| **Role** | Personal Study Advisor |
| **Goal** | Give personalized academic advice based on the student's completed modules, grades, remaining requirements, and available courses. Help plan future semesters. |
| **Backstory** | Knowledgeable study advisor who understands TU Berlin's degree programs, credit requirements, and study regulations. Combines academic history with available offerings. |
| **Tools** | `get_student_progress`, `get_completed_modules`, `get_planned_modules`, `get_missing_requirements`, `check_module_fits_degree`, `add_planned_module`, `search_modules` |
| **When used** | "Plan my next semester", "How many LP do I still need?", "Add RL to my plan" |
| **Key capability** | This is the ONLY agent that can WRITE to the Grade Manager (via `add_planned_module`) |

### Agent 4: Orchestrator 🧭

| Property | Value |
|----------|-------|
| **Role** | Student Assistant Orchestrator |
| **Goal** | Understand the student's question and delegate to the right specialist(s). Combine results from multiple agents into a coherent response. |
| **Backstory** | Main point of contact for TU Berlin students. Understands whether a question needs module search, ISIS info, or study planning — and delegates. |
| **Tools** | None — acts as the `manager_agent` in hierarchical process |
| **When used** | Always — entry point for all user queries |

### CrewAI Configuration

```python
from crewai import Agent, Crew, Process, Task

crew = Crew(
    agents=[module_researcher, course_info_specialist, study_advisor],
    tasks=[...],
    process=Process.hierarchical,
    manager_agent=orchestrator,
    verbose=True,
)

# User query comes in:
result = crew.kickoff(inputs={"query": user_message})
```

### Agent Collaboration Example

```
User: "Welche ML Module kann ich noch belegen?"

Orchestrator:
  → Creates task for Study Advisor: "Check student's current modules"
  → Creates task for Module Researcher: "Search for ML modules"

Study Advisor:
  → get_completed_modules() → ["ML1", "Deep Learning", "GPU Computing"]
  → get_missing_requirements() → "12 LP missing in Elective"

Module Researcher:
  → search_modules("Machine Learning") → 10 results
  → For each: check_module_fits_degree() → filter already completed
  → Result: ML2, RL, NLP fit in Elective ✅

Orchestrator combines:
  → "Du hast ML1, DL, GPU Computing abgeschlossen.
     Dir fehlen 12 LP im Wahlbereich.
     Passende ML Module: ML2 (6 LP), RL (6 LP), NLP (6 LP)."
```

---

## 8. Architecture & Target File Structure

```
tu-study-assistant/
├── README.md                         # Project README for seminar submission
├── HANDOFF.md                        # THIS FILE — the planning document
├── requirements.txt                  # All dependencies
├── .env                              # ISIS + LLM credentials (gitignored)
├── .gitignore
├── main.py                           # CLI entry point for testing
│
├── core/                             # COPIED from TU Notenmanager (mostly unchanged)
│   ├── models.py                     # Shared Pydantic models
│   ├── terms.py                      # Semester parsing
│   ├── registry.py                   # Degree program registry
│   ├── interfaces.py                 # DegreeStrategy protocol
│   ├── manager.py                    # DegreeManager facade
│   ├── persistence.py                # Module load/save
│   ├── analytics.py                  # Progress statistics
│   ├── calculations.py               # GPA calculation
│   ├── rules.py                      # Degree validation
│   ├── impl/tu_berlin/               # 4 degree program implementations
│   └── providers/tu_berlin/moses.py  # Moses scraper (2587 lines)
│
├── crew/                             # NEW: All CrewAI code
│   ├── __init__.py
│   ├── config.py                     # LLM configuration
│   ├── isis_client.py                # ISIS Playwright + REST API
│   ├── crew.py                       # @CrewBase crew definition
│   ├── agents.yaml                   # 4 agent configs
│   ├── tasks.yaml                    # Task templates
│   └── tools/
│       ├── __init__.py
│       ├── moses_tools.py            # 3 tools: search, details, catalogs
│       ├── isis_tools.py             # 6 tools: courses, assignments, grades...
│       └── grademanager_tools.py     # 6 tools: progress, modules, add_planned...
│
├── pages/                            # Streamlit multipage app
│   └── 3_💬_Study_Chat.py            # Chat UI (Phase 5)
│
├── ui/                               # COPIED from TU Notenmanager (unchanged)
│   ├── dashboard.py
│   ├── modules.py
│   ├── timeline.py
│   └── ...
│
├── assets/                           # COPIED (unchanged)
│   ├── style.css
│   └── tu-berlin-logo.svg
│
├── .streamlit/config.toml            # COPIED (unchanged)
├── app.py                            # COPIED — Streamlit main app
├── data/                             # Student data (gitignored)
│   └── profiles/primary/modules.json
│
└── tests/
    ├── test_moses_tools.py
    ├── test_isis_client.py
    └── test_crew.py
```

---

## 9. Grade Manager — Data Model & Integration

### Student Data Format (`data/profiles/primary/modules.json`)

```json
[
    {
        "id": "uuid-here",
        "name": "Machine Learning 1",
        "state": "Completed",
        "program_key": "TU Berlin - Computer Science (M.Sc.)",
        "cp": 6.0,
        "grade": 1.3,
        "area": "Elective",
        "is_graded": true,
        "term": "WS 25/26",
        "offered_in": "WS only",
        "source": "MOSES",
        "institution": "TU Berlin",
        "catalogs": ["Machine Learning"],
        "moses_number": "40966",
        "moses_version": 2,
        "moses": { /* full MosesModuleData object */ }
    },
    {
        "id": "another-uuid",
        "name": "Reinforcement Learning",
        "state": "Planned",
        "program_key": "TU Berlin - Computer Science (M.Sc.)",
        "cp": 6.0,
        "grade": null,
        "area": "Elective",
        "term": "WS 26/27",
        ...
    }
]
```

### Module States
```python
class ModuleState(str, Enum):
    COMPLETED = "Completed"          # Grade assigned, LP counted
    IN_PROGRESS = "In Progress"      # Currently taking
    PLANNED = "Planned"              # In study plan for a future semester
    POSSIBLE_CANDIDATE = "Possible Candidate"  # Maybe will take, not committed
```

### How to Read/Write Student Data

```python
from core.persistence import load_modules, save_modules, load_profiles

# Load
profiles = load_profiles()
primary = next(p for p in profiles if p.is_primary)
modules = load_modules(primary.slug)

# Filter by state
completed = [m for m in modules if m.state == ModuleState.COMPLETED]
planned = [m for m in modules if m.state == ModuleState.PLANNED]

# Check if a module is already in the plan
from core.providers.tu_berlin.moses import find_existing_module_by_moses_identity_any_program
existing = find_existing_module_by_moses_identity_any_program(
    modules, number="40966", version=2
)

# Add a new planned module
from core.providers.tu_berlin.moses import create_module_from_moses_data, fetch_course_details
import uuid
data = fetch_course_details("40966", 2)
new_mod = create_module_from_moses_data(
    data,
    program_key="TU Berlin - Computer Science (M.Sc.)",
    area="Elective",
    state=ModuleState.PLANNED,
    module_id=str(uuid.uuid4()),
    term="WS 26/27",
)
modules.append(new_mod)
save_modules(modules, primary.slug)
```

### GPA Calculation & Validation

```python
from core.manager import DegreeManager
from core.registry import create_program

mgr = DegreeManager(create_program("TU Berlin - Computer Science (M.Sc.)"))

# GPA
result = mgr.calculate(modules)
print(f"GPA: {result.final_grade:.2f}")
print(f"Total LP: {result.total_cp}")
print(f"Discarded (Streichliste): {len(result.discarded_modules)} modules, {result.discarded_cp} LP")

# Validation (missing requirements)
for v in mgr.validate(modules):
    status = "✅" if v.satisfied else "❌"
    print(f"{status} {v.rule_name}: {v.message}")
```

---

## 10. LLM Configuration

The project is designed to use **GWDG-hosted open-weight models** from the start for development, testing, and production. This avoids dependence on proprietary services like OpenAI and ensures compliance with academic infrastructure.

### GWDG Models Overview & Recommendations

Based on the models provided by GWDG, the following models are recommended for testing and evaluation in this multi-agent system:

| Organization | Model Name / Identifier | Open | Context | Strengths / Use Cases | Recommended Settings |
|--------------|-------------------------|------|---------|-----------------------|----------------------|
| 🇫🇷 **Mistral** | `devstral-2-123b-instruct-2512` | Yes | 256K | **Primary Recommendation**. Specifically optimized for coding, tool calling, and agentic workflows. Excellent at multi-step tasks. | `default` |
| 🇺🇸 **Google** | `gemma-4-31b-instruct` | Yes | 256K | High speed, large context window. Excellent general-purpose assistant. Good alternative for agents with simpler tools. | `default` |
| 🇨🇳 **DeepSeek** | `deepseek-r1-distill-llama-70b` | Yes | 32K | Strong reasoning capabilities. Useful for complex planning tasks (e.g. Study Advisor synthesizing study plans), but has a smaller context window. | `temp=0.7`, `top_p=0.8` |
| 🇸🇬 **Z.ai** | `glm-4.7` | Yes | 200K | Strong general performance. | `temp=1.0`, `top_p=0.95` |
| 🇨🇭 **Swiss AI** | `apertus-70b-instruct-2509` | Yes | 65K | Fully open-source and multilingual. | `temp=0.8`, `top_p=0.9` |

### Recommended Evaluation Strategy
1. **Primary Agent LLM**: Set `devstral-2-123b-instruct-2512` as the default LLM for the orchestrator and code-heavy/tool-heavy agents (Module Researcher, Course Info Specialist).
2. **Reasoning Agent LLM**: Test `deepseek-r1-distill-llama-70b` for the Study Advisor agent to see if its chain-of-thought planning improves semester recommendation quality.
3. **Fallback / Fast Agent LLM**: Use `gemma-4-31b-instruct` for fast, lightweight sub-agent tasks.

### CrewAI Configuration

Configure your CrewAI LLM objects to point to the GWDG API endpoints using the standard OpenAI compatibility mode.

**Environment Setup (`.env`):**
```bash
GWDG_API_KEY=your_gwdg_api_token
GWDG_API_BASE=https://chat-ai.hpc.gwdg.de/v1
STUDY_ASSISTANT_MODEL=devstral-2-123b-instruct-2512
```

**Python Implementation (`crew/config.py`):**
```python
import os
from crewai import LLM

def get_default_llm(temperature=0.2):
    return LLM(
        model=f"openai/{os.getenv('STUDY_ASSISTANT_MODEL', 'devstral-2-123b-instruct-2512')}",
        base_url=os.getenv('GWDG_API_BASE', 'https://chat-ai.hpc.gwdg.de/v1'),
        api_key=os.getenv('GWDG_API_KEY'),
        temperature=temperature,
    )

# Specialized model definition
def get_reasoning_llm():
    return LLM(
        model="openai/deepseek-r1-distill-llama-70b",
        base_url=os.getenv('GWDG_API_BASE', 'https://chat-ai.hpc.gwdg.de/v1'),
        api_key=os.getenv('GWDG_API_KEY'),
        temperature=0.7,
    )
```

> **Warning**: Open-weight models sometimes struggle with complex tool-use chains. If tool-calling fails frequently, refine agent descriptions/system prompts or try other GWDG models (e.g. Qwen or Llama variants if available).


---

## 11. UI Plan (Streamlit Chat)

### Phase 5: New Chat Page

```python
# pages/3_💬_Study_Chat.py

import streamlit as st

st.set_page_config(page_title="TU Study Chat", page_icon="💬", layout="wide")
st.title("💬 TU Study Assistant")
st.caption("Powered by CrewAI — Multi-Agent System for TU Berlin Students")

# Sidebar: ISIS login
with st.sidebar:
    st.header("🔐 ISIS Login")
    isis_user = st.text_input("TUB-Kontoname", key="isis_user")
    isis_pass = st.text_input("Passwort", type="password", key="isis_pass")
    if st.button("🔑 Login to ISIS"):
        with st.spinner("Logging in via Shibboleth..."):
            # Store credentials in session_state ONLY (never on disk)
            st.session_state.isis_credentials = (isis_user, isis_pass)
            st.success("Logged in!")

    st.divider()
    st.header("📊 Quick Stats")
    # Show student progress from Grade Manager
    # ...

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Was möchtest du wissen?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents arbeiten zusammen..."):
            from crew.crew import StudyAssistantCrew
            crew = StudyAssistantCrew()
            result = crew.kickoff(inputs={"query": prompt})
        st.markdown(result.raw)
    st.session_state.messages.append({"role": "assistant", "content": result.raw})
```

---

## 12. Key Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **@tool vs MCP** | @tool | Seminar evaluates agent collaboration, not tool protocols. Faster dev, easier debug. MCP is a stretch goal. |
| **Moses: keep vs rewrite** | Keep as-is | 2587 lines of working code. No JS needed (urllib works fine). Just wrap as @tool. |
| **Separate repo vs extend Grade Manager** | New repo, copy all files | Clean start for seminar submission, but contains full Grade Manager for integration. |
| **Streamlit vs React** | Streamlit | Same tech stack as Grade Manager. `st.chat_input()` built-in. No separate frontend build. |
| **Proprietary LLMs vs GWDG Open-Weight** | GWDG Open-Weight from the start | GWDG models (like Devstral 2 and Gemma 4) are hosted academically, free of cost, and support large context windows. We test different models (Devstral, Gemma, DeepSeek R1 Distill) to find the best fit for each agent. |
| **ISIS: Playwright vs pure scraping** | Playwright for login, REST API for data | Shibboleth requires real browser. But after login, Moodle REST API gives clean JSON — no HTML scraping needed. |
| **4 agents vs fewer** | 4 agents | Seminar requires demonstrating multi-agent value. Fewer agents = less impressive. |
| **Hierarchical vs sequential process** | Hierarchical | Orchestrator can dynamically decide which agents to involve based on the query. More flexible, more impressive for demo. |
