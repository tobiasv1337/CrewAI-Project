from __future__ import annotations

import os

import pytest

from crew.degree_regulations_crew import DegreeRegulationsCrew
from crew.degree_regulations_rag import KNOWLEDGE_ROOT, discover_regulation_pdfs


pytestmark = pytest.mark.live_llm


def _live_enabled() -> bool:
    return os.getenv("RUN_LIVE_CREW_TESTS") == "1" and bool(os.getenv("GWDG_API_KEY"))


@pytest.mark.skipif(
    not _live_enabled() or not discover_regulation_pdfs(KNOWLEDGE_ROOT),
    reason="Set RUN_LIVE_CREW_TESTS=1, GWDG_API_KEY, and add PDFs under knowledge/degree_regulations.",
)
def test_live_degree_regulations_crew_answers_from_local_pdfs():
    result = DegreeRegulationsCrew(
        model=os.getenv("STUDY_ASSISTANT_MODEL"),
        verbose=False,
        cache=False,
    ).crew().kickoff(
        inputs={
            "query": "Welche Informationen enthalten die lokalen StuPO PDFs zum Regelstudienplan?",
            "student_context": "Live smoke test without personal progress context.",
            "language": "German",
        }
    )

    assert str(getattr(result, "raw", result)).strip()
