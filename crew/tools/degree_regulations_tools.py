from __future__ import annotations

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from crew.degree_regulations_rag import (
    DEFAULT_RESULTS_LIMIT,
    DEFAULT_SCORE_THRESHOLD,
    extract_regelstudienplan,
    format_pdf_manifest,
    search_regulation_pdfs,
)


class DegreeRegulationsBaseInput(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def normalize_none_like_strings(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = "".join(ch for ch in value.casefold() if ch.isalnum())
            if normalized in {"", "none", "null", "nil", "na", "notlisted", "notavailable"}:
                return None
        return value


class SearchDegreeRegulationsInput(DegreeRegulationsBaseInput):
    query: str = Field(..., description="Question or search phrase about TU Berlin degree regulations.")
    limit: int = Field(default=DEFAULT_RESULTS_LIMIT, description="Maximum retrieved passages to return.")
    score_threshold: float = Field(
        default=DEFAULT_SCORE_THRESHOLD,
        description="Minimum similarity score between 0 and 1. Lower values return more passages.",
    )
    force_refresh: bool = Field(default=False, description="Force rebuilding the PDF index before searching.")


class ExtractRegelstudienplanInput(DegreeRegulationsBaseInput):
    program_query: str | None = Field(
        default=None,
        description="Optional program name hint used to prioritize matching StuPO PDFs.",
    )
    max_pages: int = Field(default=12, description="Maximum matching pages to return.")


class EmptyInput(BaseModel):
    pass


class SearchDegreeRegulationPdfsTool(BaseTool):
    name: str = "Search Degree Regulation PDFs"
    description: str = (
        "Search the local AllgStuPO and program StuPO PDF knowledge base using CrewAI-native RAG. "
        "Use for degree rules, examination rules, allowed elective/free-choice rules, and source-backed regulation answers."
    )
    args_schema: Type[BaseModel] = SearchDegreeRegulationsInput

    def _run(
        self,
        query: str,
        limit: int = DEFAULT_RESULTS_LIMIT,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        force_refresh: bool = False,
    ) -> str:
        return search_regulation_pdfs(
            query,
            limit=limit,
            score_threshold=score_threshold,
            force_refresh=force_refresh,
        )


class ListDegreeRegulationPdfsTool(BaseTool):
    name: str = "List Degree Regulation PDFs"
    description: str = (
        "List the AllgStuPO and program StuPO PDFs currently available in the local regulation knowledge folder."
    )
    args_schema: Type[BaseModel] = EmptyInput

    def _run(self) -> str:
        return format_pdf_manifest()


class ExtractRegelstudienplanTableTool(BaseTool):
    name: str = "Extract Regelstudienplan Table"
    description: str = (
        "Extract likely Regelstudienplan or Studienverlaufsplan tables/text from local StuPO PDFs with source/page citations."
    )
    args_schema: Type[BaseModel] = ExtractRegelstudienplanInput

    def _run(self, program_query: str | None = None, max_pages: int = 12) -> str:
        return extract_regelstudienplan(program_query=program_query, max_pages=max_pages)


DEGREE_REGULATIONS_TOOLS = [
    SearchDegreeRegulationPdfsTool(),
    ListDegreeRegulationPdfsTool(),
    ExtractRegelstudienplanTableTool(),
]


__all__ = [
    "DEGREE_REGULATIONS_TOOLS",
    "SearchDegreeRegulationPdfsTool",
    "ListDegreeRegulationPdfsTool",
    "ExtractRegelstudienplanTableTool",
]
