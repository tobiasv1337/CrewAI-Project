from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chromadb.config import Settings
from crewai.rag.chromadb.config import ChromaDBConfig
from crewai.rag.config.types import RagConfigType
from crewai.rag.core.base_client import BaseClient
from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.factory import create_client
from crewai.rag.types import BaseRecord, SearchResult

from crew.runtime import ensure_crewai_storage_writable


KNOWLEDGE_ROOT = Path("knowledge") / "degree_regulations"
DEFAULT_COLLECTION_NAME = "degree_regulations"
DEFAULT_CHUNK_SIZE = 3200
DEFAULT_CHUNK_OVERLAP = 350
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_RESULTS_LIMIT = 6
REGELSTUDIENPLAN_TERMS = (
    "regelstudienplan",
    "studienverlaufsplan",
    "modulplan",
    "studienplan",
)


@dataclass(frozen=True)
class PdfManifestEntry:
    path: Path
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PdfChunk:
    doc_id: str
    content: str
    metadata: dict[str, str | int | float | bool]


def discover_regulation_pdfs(root: Path | str = KNOWLEDGE_ROOT) -> list[Path]:
    """Return regulation PDFs from the drop-in knowledge folder."""
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*.pdf") if path.is_file())


def build_pdf_manifest(root: Path | str = KNOWLEDGE_ROOT) -> list[PdfManifestEntry]:
    """Build a deterministic content manifest for all regulation PDFs."""
    base = Path(root)
    entries: list[PdfManifestEntry] = []
    for path in discover_regulation_pdfs(base):
        content = path.read_bytes()
        entries.append(
            PdfManifestEntry(
                path=path,
                relative_path=_relative_pdf_path(path, base),
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    return entries


def manifest_hash(entries: list[PdfManifestEntry]) -> str:
    payload = [
        {
            "path": entry.relative_path,
            "size": entry.size,
            "sha256": entry.sha256,
        }
        for entry in entries
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def index_regulation_pdfs(
    *,
    root: Path | str = KNOWLEDGE_ROOT,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    storage_dir: Path | str | None = None,
    force_refresh: bool = False,
    client: BaseClient | None = None,
    embedder: dict[str, Any] | None = None,
) -> str:
    """Ensure the CrewAI RAG collection is synced with the PDF folder."""
    base = Path(root)
    entries = build_pdf_manifest(base)
    current_hash = manifest_hash(entries)
    storage = _storage_dir(storage_dir)
    marker = storage / f"{collection_name}_manifest.json"
    previous_hash = _read_manifest_hash(marker)

    if not entries:
        if force_refresh or previous_hash != current_hash:
            _delete_collection(_client(client, embedder, storage), collection_name)
            _write_manifest(marker, current_hash, entries, chunk_count=0)
        return _format_index_summary(entries, current_hash, 0, refreshed=False)

    if not force_refresh and previous_hash == current_hash:
        return _format_index_summary(entries, current_hash, None, refreshed=False)

    rag_client = _client(client, embedder, storage)
    _delete_collection(rag_client, collection_name)
    rag_client.get_or_create_collection(collection_name=collection_name)

    chunks = load_pdf_chunks(base, entries)
    if chunks:
        records: list[BaseRecord] = [
            {
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        rag_client.add_documents(collection_name=collection_name, documents=records)

    _write_manifest(marker, current_hash, entries, chunk_count=len(chunks))
    return _format_index_summary(entries, current_hash, len(chunks), refreshed=True)


def search_regulation_pdfs(
    query: str,
    *,
    root: Path | str = KNOWLEDGE_ROOT,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    storage_dir: Path | str | None = None,
    limit: int = DEFAULT_RESULTS_LIMIT,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    force_refresh: bool = False,
    client: BaseClient | None = None,
    embedder: dict[str, Any] | None = None,
) -> str:
    """Search the indexed regulation PDF collection and return cited chunks."""
    clean_query = str(query or "").strip()
    if not clean_query:
        return "No regulation search query was provided."

    entries = build_pdf_manifest(root)
    if not entries:
        return _empty_folder_message(root)

    storage = _storage_dir(storage_dir)
    index_regulation_pdfs(
        root=root,
        collection_name=collection_name,
        storage_dir=storage,
        force_refresh=force_refresh,
        client=client,
        embedder=embedder,
    )
    results = _client(client, embedder, storage).search(
        collection_name=collection_name,
        query=clean_query,
        limit=max(1, min(int(limit or DEFAULT_RESULTS_LIMIT), 20)),
        score_threshold=max(0.0, min(float(score_threshold), 1.0)),
    )
    if not results:
        return (
            "No relevant regulation PDF passages were found.\n\n"
            f"Indexed documents:\n{format_pdf_manifest(root)}"
        )
    return format_search_results(results, query=clean_query)


def format_pdf_manifest(root: Path | str = KNOWLEDGE_ROOT) -> str:
    entries = build_pdf_manifest(root)
    if not entries:
        return _empty_folder_message(root)

    lines = ["# Indexed degree regulation PDFs", ""]
    for entry in entries:
        hints = infer_document_hints(entry.relative_path)
        lines.append(
            f"- `{entry.relative_path}` ({entry.size} bytes, sha256 `{entry.sha256[:12]}`)"
            f" — {hints}"
        )
    return "\n".join(lines)


def extract_regelstudienplan(
    *,
    program_query: str | None = None,
    root: Path | str = KNOWLEDGE_ROOT,
    max_pages: int = 12,
) -> str:
    """Extract likely Regelstudienplan text/tables from regulation PDFs."""
    entries = _filter_entries_by_program(build_pdf_manifest(root), program_query)
    if not entries:
        return _empty_folder_message(root)

    import pdfplumber

    lines = ["# Regelstudienplan extraction", ""]
    hit_count = 0
    for entry in entries:
        with pdfplumber.open(entry.path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                text = _normalize_whitespace(page.extract_text() or "")
                if not _looks_like_regelstudienplan(text):
                    continue
                hit_count += 1
                lines.extend(
                    [
                        f"## {entry.relative_path}, page {page_index}",
                        "",
                    ]
                )
                table_lines = _extract_page_tables(page)
                if table_lines:
                    lines.extend(table_lines)
                elif text:
                    lines.append(_truncate(text, 2400))
                else:
                    lines.append("This page matched the keywords but no extractable text/table was found.")
                lines.append("")
                if hit_count >= max(1, int(max_pages or 1)):
                    return "\n".join(lines).rstrip()
    if hit_count == 0:
        query_note = f" matching `{program_query}`" if program_query else ""
        return (
            f"No Regelstudienplan-like pages were found in regulation PDFs{query_note}.\n\n"
            "Searched for: "
            + ", ".join(f"`{term}`" for term in REGELSTUDIENPLAN_TERMS)
        )
    return "\n".join(lines).rstrip()


def load_pdf_chunks(root: Path, entries: list[PdfManifestEntry]) -> list[PdfChunk]:
    import pdfplumber

    chunks: list[PdfChunk] = []
    for entry in entries:
        with pdfplumber.open(entry.path) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                text = _normalize_whitespace(page.extract_text() or "")
                if not text:
                    continue
                page_prefix = (
                    f"Source: {entry.relative_path}\n"
                    f"Page: {page_index}\n"
                    f"Document type: {infer_document_hints(entry.relative_path)}\n\n"
                )
                for chunk_index, chunk_text in enumerate(_chunk_text(text), start=1):
                    content = page_prefix + chunk_text
                    stable_id = _stable_chunk_id(entry.relative_path, page_index, chunk_index, content)
                    chunks.append(
                        PdfChunk(
                            doc_id=stable_id,
                            content=content,
                            metadata={
                                "source": entry.relative_path,
                                "page": page_index,
                                "chunk_index": chunk_index,
                                "sha256": entry.sha256,
                            },
                        )
                    )
    return chunks


def format_search_results(results: list[SearchResult], *, query: str) -> str:
    lines = [
        "# Relevant regulation PDF passages",
        "",
        f"Query: `{query}`",
        "",
    ]
    for index, result in enumerate(results, start=1):
        content = _normalize_whitespace(str(result.get("content") or ""))
        metadata = result.get("metadata") or {}
        source = metadata.get("source") if isinstance(metadata, dict) else None
        page = metadata.get("page") if isinstance(metadata, dict) else None
        score = result.get("score")
        citation = f"{source or 'unknown source'}"
        if page:
            citation += f", page {page}"
        score_text = f" (score {float(score):.2f})" if isinstance(score, (int, float)) else ""
        lines.extend(
            [
                f"## Passage {index}: {citation}{score_text}",
                "",
                _truncate(content, 1800),
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def infer_document_hints(relative_path: str) -> str:
    text = relative_path.replace("_", " ").replace("-", " ").casefold()
    hints: list[str] = []
    if "allgstupo" in text or "allg stupo" in text or "allgemeine" in text:
        hints.append("general AllgStuPO")
    if "stupo" in text or "studienordnung" in text or "pruefungsordnung" in text or "prüfungsordnung" in text:
        hints.append("program-specific StuPO candidate")
    program = re.sub(r"\.pdf$", "", relative_path, flags=re.IGNORECASE)
    program = re.sub(r"[/_\\-]+", " ", program).strip()
    if program:
        hints.append(f"path hint: {program}")
    return "; ".join(hints) if hints else "unknown regulation type"


def _client(
    client: BaseClient | None,
    embedder: dict[str, Any] | None,
    storage_dir: Path | str | None,
) -> BaseClient:
    if client is not None:
        return client
    config = _rag_config(embedder or resolve_embedder_config(), _storage_dir(storage_dir))
    return create_client(config)


def _rag_config(embedder: dict[str, Any], storage_dir: Path) -> RagConfigType:
    chroma_dir = storage_dir / "chromadb"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        persist_directory=str(chroma_dir),
        allow_reset=True,
        is_persistent=True,
        anonymized_telemetry=False,
    )
    return ChromaDBConfig(settings=settings, embedding_function=build_embedder(embedder))


def resolve_embedder_config() -> dict[str, Any]:
    provider = os.getenv("DEGREE_REGULATIONS_EMBEDDER_PROVIDER", "onnx").strip() or "onnx"
    if provider == "openai":
        config: dict[str, Any] = {}
        if api_key := os.getenv("DEGREE_REGULATIONS_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"):
            config["api_key"] = api_key
        if model := os.getenv("DEGREE_REGULATIONS_OPENAI_MODEL"):
            config["model_name"] = model
        if api_base := os.getenv("DEGREE_REGULATIONS_OPENAI_API_BASE") or os.getenv("OPENAI_API_BASE"):
            config["api_base"] = api_base
        if dimensions := os.getenv("DEGREE_REGULATIONS_OPENAI_DIMENSIONS"):
            config["dimensions"] = int(dimensions)
        return {"provider": "openai", "config": config}
    if provider == "ollama":
        config = {}
        if model_name := os.getenv("DEGREE_REGULATIONS_OLLAMA_MODEL"):
            config["model_name"] = model_name
        if url := os.getenv("DEGREE_REGULATIONS_OLLAMA_URL"):
            config["url"] = url
        return {"provider": "ollama", "config": config}
    return {"provider": "onnx", "config": {}}


def _storage_dir(storage_dir: Path | str | None) -> Path:
    if storage_dir is not None:
        path = Path(storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return ensure_crewai_storage_writable() / "degree_regulations_rag"


def _delete_collection(client: BaseClient, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name=collection_name)  # type: ignore[attr-defined]
    except Exception:
        pass


def _write_manifest(
    marker: Path,
    current_hash: str,
    entries: list[PdfManifestEntry],
    *,
    chunk_count: int,
) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_hash": current_hash,
        "chunk_count": chunk_count,
        "documents": [
            {
                "path": entry.relative_path,
                "size": entry.size,
                "sha256": entry.sha256,
            }
            for entry in entries
        ],
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_manifest_hash(marker: Path) -> str | None:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("manifest_hash")
    return str(value) if value else None


def _format_index_summary(
    entries: list[PdfManifestEntry],
    current_hash: str,
    chunk_count: int | None,
    *,
    refreshed: bool,
) -> str:
    status = "refreshed" if refreshed else "up to date"
    lines = [
        f"Degree regulation RAG index is {status}.",
        f"- Manifest: `{current_hash[:12]}`",
        f"- PDFs: {len(entries)}",
    ]
    if chunk_count is not None:
        lines.append(f"- Chunks: {chunk_count}")
    if entries:
        lines.append("- Documents:")
        lines.extend(f"  - `{entry.relative_path}`" for entry in entries)
    return "\n".join(lines)


def _empty_folder_message(root: Path | str) -> str:
    return (
        "No degree regulation PDFs are available yet.\n\n"
        f"Add AllgStuPO and program StuPO PDFs under `{Path(root)}` and retry."
    )


def _relative_pdf_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _chunk_text(text: str, *, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def _stable_chunk_id(relative_path: str, page_index: int, chunk_index: int, content: str) -> str:
    material = f"{relative_path}:{page_index}:{chunk_index}:{content}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 15)].rstrip() + "\n...[truncated]"


def _looks_like_regelstudienplan(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in REGELSTUDIENPLAN_TERMS)


def _extract_page_tables(page: Any) -> list[str]:
    try:
        tables = page.extract_tables() or []
    except Exception:
        tables = []
    lines: list[str] = []
    for table in tables:
        normalized = [
            [str(cell or "").strip().replace("\n", " ") for cell in row]
            for row in table
            if any(str(cell or "").strip() for cell in row)
        ]
        if not normalized:
            continue
        width = max(len(row) for row in normalized)
        padded = [row + [""] * (width - len(row)) for row in normalized]
        header = padded[0]
        lines.append("| " + " | ".join(_escape_md(cell or f"Column {idx}") for idx, cell in enumerate(header, start=1)) + " |")
        lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
        for row in padded[1:]:
            lines.append("| " + " | ".join(_escape_md(cell) for cell in row) + " |")
        lines.append("")
    return lines


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")


def _filter_entries_by_program(
    entries: list[PdfManifestEntry],
    program_query: str | None,
) -> list[PdfManifestEntry]:
    query = str(program_query or "").strip().casefold()
    if not query:
        return entries
    tokens = [token for token in re.split(r"\W+", query) if len(token) >= 3]
    if not tokens:
        return entries
    matched = [
        entry
        for entry in entries
        if any(token in entry.relative_path.casefold() for token in tokens)
    ]
    return matched or entries
