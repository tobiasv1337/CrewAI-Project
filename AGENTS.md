# AGENTS.md — Developer Agent Instructions (Codex/GPT 5.5)

---

## 1. Project Objective

The primary goal is to **build, test, and demonstrate the capabilities of the CrewAI multi-agent framework** using the **TU Study Assistant** (Moses scraper, ISIS Moodle REST API, and student Grade Manager) as the core use case.

---

## 2. Core Resources & Documentation

To successfully implement this project, you **must** read and refer to the following local resources in this repository:

### A. CrewAI Framework Skills
Before writing any CrewAI configurations or agent code, inspect the custom skills located in the `.agents/skills/` directory. These contain official guidelines and templates for:
- [getting-started](file:///Users/tobias/Documents/Uni/Seminar%20LLM%20Agentic%20Systems/CrewAI-Project/.agents/skills/getting-started/SKILL.md): Project structure setup, `@CrewBase`, and flow creation.
- [design-agent](file:///Users/tobias/Documents/Uni/Seminar%20LLM%20Agentic%20Systems/CrewAI-Project/.agents/skills/design-agent/SKILL.md): Defining agent roles, goals, backstories, and LLM parameters.
- [design-task](file:///Users/tobias/Documents/Uni/Seminar%20LLM%20Agentic%20Systems/CrewAI-Project/.agents/skills/design-task/SKILL.md): Configuring task descriptions, outputs, and validation.
- [ask-docs](file:///Users/tobias/Documents/Uni/Seminar%20LLM%20Agentic%20Systems/CrewAI-Project/.agents/skills/ask-docs/SKILL.md): Official API details and advanced options.

### B. Implementation Blueprint
- [HANDOFF.md](file:///Users/tobias/Documents/Uni/Seminar%20LLM%20Agentic%20Systems/CrewAI-Project/HANDOFF.md): Contains the detailed background, scraper mechanics, Shibboleth login flow, data model structures, and the 6-phase implementation roadmap.

---

## 3. Creative Autonomy & Suggestions

The implementation plan outlined in `HANDOFF.md` is a **general guide**. 

As you write code, test agent behaviors, and connect tools:
- **Proactively identify opportunities for improvement** (e.g., more robust error handling for Playwright logins, cleaner markdown output formatting, optimized token utilization, or improved prompt structuring).
- **Propose better architectural or agentic design ideas** to the user in the chat before implementing them, so they can be incorporated into the project.
- Focus on making the multi-agent collaboration (such as hierarchical task delegation) feel seamless and intelligent.

---

## 4. Key Component Architecture

### A. Moses Module Scraper
- **File**: [moses.py](file:///Users/tobias/Documents/Uni/Seminar%20LLM%20Agentic%20Systems/CrewAI-Project/core/providers/tu_berlin/moses.py)
- **Rule**: **Do NOT rewrite this scraper.** It is fully functional. Simply import its functions and wrap them with CrewAI's `@tool` decorator in `crew/tools/moses_tools.py`.

### B. ISIS (Moodle) REST API Client
- **File**: `crew/isis_client.py` (and wrapped in `crew/tools/isis_tools.py`)
- **Rule**: Standard Moodle token requests do not work because of Shibboleth SSO. Use **Playwright** to log in through the Shibboleth interface, extract the session cookies, and then fetch the `wstoken` via Moodle's mobile launch endpoint (`allow_redirects=False` to prevent schema crash). Then use standard JSON REST requests.

### C. Grade Manager Integration
- **Files**: [persistence.py](file:///Users/tobias/Documents/Uni/Seminar%20LLM%20Agentic%20Systems/CrewAI-Project/core/persistence.py) and [manager.py](file:///Users/tobias/Documents/Uni/Seminar%20LLM%20Agentic%20Systems/CrewAI-Project/core/manager.py)
- **Rule**: Allow the agents to read student history/grades and write planned modules to `data/profiles/primary/modules.json`. Ensure the `Study Advisor` agent is the only one equipped with write permissions (`add_planned_module` tool).

### D. CrewAI Orchestrator (Manager Agent)
- **Process**: Use `Process.hierarchical` in your Crew definition.
- **Orchestrator Role**: Acts as the manager agent, receiving student requests, determining which specialist agents are needed, delegating tasks, and combining the answers into a polished German or English response.

---

## 5. Environment & Commands

The project environment is managed using **uv** and is configured in `pyproject.toml`.

- **Run tests**:
  ```bash
  uv run pytest
  ```
- **Run the Streamlit application**:
  ```bash
  uv run streamlit run app.py
  ```
- **API Credentials**: Store your credentials in a `.env` file at the root. Do NOT commit the `.env` file.

---

## 6. Git Conventions

- **Smaller, Incremental Commits**: Always make small, focused git commits as you implement each part of the roadmap (e.g., commit after creating a single tool, commit after writing tests, commit after defining agents). Do not group multiple distinct features or phases into a single commit.

