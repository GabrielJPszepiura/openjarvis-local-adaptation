"""Document extraction used by temporary chat attachments."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import HTTPException

from openjarvis.server.upload_router import (
    _extract_document,
    extract_chat_attachments,
)


def _archive(entries: dict[str, str]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return stream.getvalue()


def test_extracts_text_and_docx_without_persisting() -> None:
    document = _archive({
        "word/document.xml": """
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body><w:p><w:r><w:t>Hello Word</w:t></w:r></w:p></w:body>
        </w:document>
        """,
    })

    assert _extract_document("notes.txt", "olá".encode()) == "olá"
    assert _extract_document("report.docx", document) == "Hello Word"


def test_extracts_xlsx_shared_and_numeric_cells() -> None:
    workbook = _archive({
        "xl/sharedStrings.xml": """
        <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <si><t>Revenue</t></si>
        </sst>
        """,
        "xl/worksheets/sheet1.xml": """
        <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
          <sheetData><row><c t="s"><v>0</v></c><c><v>42</v></c></row></sheetData>
        </worksheet>
        """,
    })

    text = _extract_document("numbers.xlsx", workbook)
    assert "Revenue\t42" in text


def test_rejects_unsupported_documents() -> None:
    with pytest.raises(HTTPException) as error:
        _extract_document("malware.exe", b"no")
    assert error.value.status_code == 400
