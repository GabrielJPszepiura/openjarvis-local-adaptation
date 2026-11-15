"""Document extraction used by temporary chat attachments."""
# Modified by Gabriel_JP for the OpenJarvis local adaptation (2026-11-15).

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


def test_extracts_verified_financial_calculations_from_xlsx() -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Dados"
    worksheet.append(["Mês", "Receitas", "Despesas", "Saldo acumulado"])
    worksheet.append(["Janeiro", 12500, 4800, "=B2-C2"])
    worksheet.append(["Fevereiro", 9300, 3200, "=D2+B3-C3"])
    buffer = io.BytesIO()
    workbook.save(buffer)

    text = _extract_document("controle.xlsx", buffer.getvalue())

    assert "Planilha: Dados" in text
    assert "<verified_spreadsheet_calculations sheet=\"Dados\">" in text
    assert "Total de receitas: R$ 21.800,00" in text
    assert "Total de despesas: R$ 8.000,00" in text
    assert "Resultado líquido: R$ 13.800,00" in text
    assert "Fevereiro: receitas R$ 9.300,00; despesas R$ 3.200,00; resultado R$ 6.100,00" in text


def test_rejects_unsupported_documents() -> None:
    with pytest.raises(HTTPException) as error:
        _extract_document("malware.exe", b"no")
    assert error.value.status_code == 400
