from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from crew.degree_regulations_rag import (
    build_pdf_manifest,
    extract_regelstudienplan,
    index_regulation_pdfs,
    load_pdf_chunks,
    manifest_hash,
    search_regulation_pdfs,
)


class FakeRagClient:
    def __init__(self) -> None:
        self.collections: dict[str, list[dict[str, Any]]] = {}
        self.deleted_count = 0
        self.add_count = 0

    def get_or_create_collection(self, *, collection_name: str, **_: Any) -> object:
        self.collections.setdefault(collection_name, [])
        return object()

    def delete_collection(self, *, collection_name: str, **_: Any) -> None:
        self.deleted_count += 1
        self.collections[collection_name] = []

    def add_documents(self, *, collection_name: str, documents: list[dict[str, Any]], **_: Any) -> None:
        self.add_count += 1
        self.collections.setdefault(collection_name, [])
        by_id = {doc.get("doc_id"): doc for doc in self.collections[collection_name]}
        for doc in documents:
            by_id[doc.get("doc_id")] = doc
        self.collections[collection_name] = list(by_id.values())

    def search(
        self,
        *,
        collection_name: str,
        query: str,
        limit: int,
        score_threshold: float,
        **_: Any,
    ) -> list[dict[str, Any]]:
        tokens = [token.casefold() for token in query.split() if len(token) >= 4]
        hits = []
        for doc in self.collections.get(collection_name, []):
            content = str(doc["content"])
            if any(token in content.casefold() for token in tokens):
                hits.append(
                    {
                        "content": content,
                        "metadata": doc.get("metadata", {}),
                        "score": max(score_threshold, 0.91),
                    }
                )
        return hits[:limit]


def test_pdf_manifest_and_chunks_include_source_page_prefix(tmp_path: Path) -> None:
    root = tmp_path / "knowledge" / "degree_regulations"
    _write_text_pdf(
        root / "allgstupo.pdf",
        ["AllgStuPO § 50 Wiederholungsprüfung erlaubt mehrere Prüfungsversuche."],
    )

    entries = build_pdf_manifest(root)
    assert len(entries) == 1
    assert entries[0].relative_path == "allgstupo.pdf"
    assert manifest_hash(entries) == manifest_hash(entries)

    chunks = load_pdf_chunks(root, entries)
    assert chunks
    assert "Source: allgstupo.pdf" in chunks[0].content
    assert "Page: 1" in chunks[0].content
    assert "Wiederholungsprüfung" in chunks[0].content


def test_empty_folder_reports_drop_in_location(tmp_path: Path) -> None:
    result = search_regulation_pdfs(
        "Wiederholungsprüfung",
        root=tmp_path / "knowledge" / "degree_regulations",
        storage_dir=tmp_path / "storage",
        client=FakeRagClient(),
    )

    assert "No degree regulation PDFs are available yet" in result
    assert "degree_regulations" in result


def test_search_indexes_generated_pdf_with_fake_crewai_rag_client(tmp_path: Path) -> None:
    root = tmp_path / "knowledge" / "degree_regulations"
    _write_text_pdf(
        root / "allgstupo.pdf",
        ["AllgStuPO § 50: Die Wiederholungsprüfung ist in der Ordnung geregelt."],
    )
    client = FakeRagClient()

    result = search_regulation_pdfs(
        "Was sagt die AllgStuPO zur Wiederholungsprüfung?",
        root=root,
        storage_dir=tmp_path / "storage",
        collection_name="test_degree_regulations",
        force_refresh=True,
        client=client,
    )

    assert "Relevant regulation PDF passages" in result
    assert "allgstupo.pdf, page 1" in result
    assert "Wiederholungsprüfung" in result
    assert client.add_count == 1


def test_index_refreshes_when_pdf_manifest_changes(tmp_path: Path) -> None:
    root = tmp_path / "knowledge" / "degree_regulations"
    storage = tmp_path / "storage"
    client = FakeRagClient()
    collection = "test_refresh"
    _write_text_pdf(root / "allgstupo.pdf", ["AllgStuPO initial text."])

    first = index_regulation_pdfs(root=root, storage_dir=storage, collection_name=collection, client=client)
    second = index_regulation_pdfs(root=root, storage_dir=storage, collection_name=collection, client=client)
    _write_text_pdf(root / "informatik/stupo.pdf", ["StuPO Informatik new text."])
    third = index_regulation_pdfs(root=root, storage_dir=storage, collection_name=collection, client=client)

    assert "refreshed" in first
    assert "up to date" in second
    assert "refreshed" in third
    assert client.add_count == 2


def test_extract_regelstudienplan_text_from_pdf(tmp_path: Path) -> None:
    root = tmp_path / "knowledge" / "degree_regulations"
    _write_text_pdf(
        root / "informatik/stupo.pdf",
        [
            "Studienordnung Informatik",
            "Regelstudienplan Informatik\nSemester 1: Mathematik 1, Programmierung 1\nSemester 2: Algorithmen und Datenstrukturen",
        ],
    )

    result = extract_regelstudienplan(program_query="Informatik", root=root)

    assert "informatik/stupo.pdf, page 2" in result
    assert "Regelstudienplan Informatik" in result
    assert "Semester 1" in result


def _write_text_pdf(path: Path, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for text in pages:
            fig = Figure(figsize=(8.27, 11.69))
            ax = fig.subplots()
            ax.axis("off")
            ax.text(0.08, 0.92, text, va="top", ha="left", wrap=True, fontsize=12)
            pdf.savefig(fig)
