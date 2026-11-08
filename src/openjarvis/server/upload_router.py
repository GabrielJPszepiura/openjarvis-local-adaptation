"""Upload / Paste router for ingesting documents into the knowledge store."""

from __future__ import annotations

import io
import logging
import uuid
import zipfile
from pathlib import PurePath
from typing import List, Optional
from xml.etree import ElementTree

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.core.config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/connectors/upload", tags=["upload"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}
_MAX_CHAT_FILES = 4
_MAX_FILE_BYTES = 10 * 1024 * 1024
_MAX_EXTRACTED_CHARS = 80_000


def _chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Split *text* into ~max_chars pieces at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    # Guard against very large paragraphs that exceed max_chars
    final: List[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            # Find last space within limit
            split_at = chunk.rfind(" ", 0, max_chars)
            if split_at == -1:
                split_at = max_chars
            final.append(chunk[:split_at].strip())
            chunk = chunk[split_at:].strip()
        if chunk:
            final.append(chunk)
    return final


def _extract_text_from_pdf(data: bytes) -> str:
    """Extract text from a PDF using pdfplumber or PyPDF2."""
    # Try pdfplumber first
    try:
        import pdfplumber  # type: ignore[import-untyped]

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(pages)
    except ImportError:
        pass

    # Fall back to PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore[import-untyped]

        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF parsing requires pdfplumber or PyPDF2. "
                "Install one with: pip install pdfplumber"
            ),
        )


def _extract_text_from_docx(data: bytes) -> str:
    """Extract text from a .docx file using its standard XML container."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise HTTPException(status_code=400, detail="Invalid DOCX file") from exc

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise HTTPException(status_code=400, detail="Invalid DOCX file") from exc
    paragraphs: List[str] = []
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    for paragraph in root.iter(f"{{{namespace}}}p"):
        text = "".join(
            node.text or "" for node in paragraph.iter(f"{{{namespace}}}t")
        ).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _extract_text_from_xlsx(data: bytes) -> str:
    """Extract worksheet values from an XLSX file without executing formulas."""
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            shared: List[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in shared_root.iter(f"{{{namespace}}}si"):
                    shared.append(
                        "".join(node.text or "" for node in item.iter(f"{{{namespace}}}t"))
                    )

            sheets: List[str] = []
            sheet_paths = sorted(
                name
                for name in archive.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
            for sheet_index, sheet_path in enumerate(sheet_paths, start=1):
                root = ElementTree.fromstring(archive.read(sheet_path))
                rows: List[str] = []
                for row in root.iter(f"{{{namespace}}}row"):
                    values: List[str] = []
                    for cell in row.findall(f"{{{namespace}}}c"):
                        cell_type = cell.attrib.get("t")
                        value_node = cell.find(f"{{{namespace}}}v")
                        inline_node = cell.find(f"{{{namespace}}}is")
                        value = ""
                        if cell_type == "inlineStr" and inline_node is not None:
                            value = "".join(
                                node.text or ""
                                for node in inline_node.iter(f"{{{namespace}}}t")
                            )
                        elif value_node is not None and value_node.text is not None:
                            value = value_node.text
                            if cell_type == "s":
                                try:
                                    value = shared[int(value)]
                                except (ValueError, IndexError):
                                    pass
                        values.append(value)
                    if any(value for value in values):
                        rows.append("\t".join(values))
                sheets.append(f"Worksheet {sheet_index}\n" + "\n".join(rows))
    except (zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise HTTPException(status_code=400, detail="Invalid XLSX file") from exc
    return "\n\n".join(sheets)


def _extension(filename: str) -> str:
    return PurePath(filename).suffix.lower()


def _extract_document(filename: str, data: bytes) -> str:
    """Extract safe plain text from one supported document."""
    ext = _extension(filename)
    if ext not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed}",
        )
    if ext in (".txt", ".md", ".csv"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")
    if ext == ".pdf":
        return _extract_text_from_pdf(data)
    if ext == ".docx":
        return _extract_text_from_docx(data)
    if ext == ".xlsx":
        return _extract_text_from_xlsx(data)
    return ""


def _get_store() -> KnowledgeStore:
    """Return a KnowledgeStore pointing at the default knowledge DB."""
    db_path = DEFAULT_CONFIG_DIR / "knowledge.db"
    return KnowledgeStore(db_path=db_path)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PasteRequest(BaseModel):
    title: str = ""
    content: str


class IngestResponse(BaseModel):
    chunks_added: int
    source: str = "upload"


class ExtractedAttachment(BaseModel):
    name: str
    size: int
    type: str
    text: str
    characters: int
    truncated: bool = False


class ExtractAttachmentsResponse(BaseModel):
    attachments: List[ExtractedAttachment]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest_paste(body: PasteRequest) -> IngestResponse:
    """Ingest pasted text into the knowledge store."""
    text = body.content.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Content is empty")

    store = _get_store()
    doc_id = str(uuid.uuid4())
    chunks = _chunk_text(text)

    for idx, chunk in enumerate(chunks):
        store.store(
            chunk,
            source="upload",
            doc_type="paste",
            doc_id=doc_id,
            title=body.title or "Pasted text",
            chunk_index=idx,
        )

    logger.info("Ingested %d chunks from pasted text (doc_id=%s)", len(chunks), doc_id)
    return IngestResponse(chunks_added=len(chunks))


@router.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(...),
    title: Optional[str] = Form(None),
) -> IngestResponse:
    """Ingest uploaded files into the knowledge store."""
    store = _get_store()
    total_chunks = 0

    for upload in files:
        filename = upload.filename or "untitled"
        ext = _extension(filename)
        data = await upload.read()
        text = _extract_document(filename, data)

        text = text.strip()
        if not text:
            continue

        doc_id = str(uuid.uuid4())
        doc_title = title or filename
        chunks = _chunk_text(text)

        for idx, chunk in enumerate(chunks):
            store.store(
                chunk,
                source="upload",
                doc_type=ext.lstrip("."),
                doc_id=doc_id,
                title=doc_title,
                chunk_index=idx,
            )

        total_chunks += len(chunks)
        logger.info(
            "Ingested %d chunks from file %s (doc_id=%s)",
            len(chunks),
            filename,
            doc_id,
        )

    return IngestResponse(chunks_added=total_chunks)


@router.post("/extract", response_model=ExtractAttachmentsResponse)
async def extract_chat_attachments(
    files: List[UploadFile] = File(...),
) -> ExtractAttachmentsResponse:
    """Extract temporary chat context without writing to the knowledge store."""
    if not files or len(files) > _MAX_CHAT_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Attach between 1 and {_MAX_CHAT_FILES} files at a time",
        )

    attachments: List[ExtractedAttachment] = []
    per_file_chars = _MAX_EXTRACTED_CHARS // len(files)
    for upload in files:
        filename = upload.filename or "untitled"
        data = upload.file.read(_MAX_FILE_BYTES + 1)
        if len(data) > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than 10 MB: {filename}",
            )

        extracted = _extract_document(filename, data).strip()
        if not extracted:
            raise HTTPException(
                status_code=422,
                detail=f"No readable text was found in {filename}",
            )
        original_chars = len(extracted)
        text = extracted[:per_file_chars]
        truncated = len(text) < original_chars
        attachments.append(
            ExtractedAttachment(
                name=filename,
                size=len(data),
                type=_extension(filename).lstrip("."),
                text=text,
                characters=original_chars,
                truncated=truncated,
            )
        )

    return ExtractAttachmentsResponse(attachments=attachments)
