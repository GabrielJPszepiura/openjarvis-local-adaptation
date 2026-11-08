from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.tools.document_create import (
    CreateDocumentTool,
    CreateSpreadsheetTool,
)


def test_create_word_document(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    result = CreateDocumentTool([str(tmp_path)]).execute(
        filename="relatorio",
        format="docx",
        title="Relatório de teste",
        content="# Resumo\nConteúdo em português.\n- Primeiro item",
    )

    assert result.success is True
    output = tmp_path / "relatorio.docx"
    assert output.exists()
    document = docx.Document(output)
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Relatório de teste" in text
    assert "Conteúdo em português." in text


def test_create_pdf_document(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    result = CreateDocumentTool([str(tmp_path)]).execute(
        filename="resumo.pdf",
        format="pdf",
        title="Resumo",
        content="Documento criado localmente.",
    )

    assert result.success is True
    output = tmp_path / "resumo.pdf"
    assert output.read_bytes().startswith(b"%PDF")


def test_create_spreadsheet(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    result = CreateSpreadsheetTool([str(tmp_path)]).execute(
        filename="vendas",
        sheet_name="Produtos",
        columns=["Produto", "Quantidade"],
        rows=[["Café", 3], ["Leite", 2]],
    )

    assert result.success is True
    workbook = openpyxl.load_workbook(tmp_path / "vendas.xlsx")
    sheet = workbook["Produtos"]
    assert sheet["A2"].value == "Café"
    assert sheet["B3"].value == 2


def test_numbers_duplicate_document_without_overwriting(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    path = tmp_path / "existente.docx"
    path.write_bytes(b"original")
    result = CreateDocumentTool([str(tmp_path)]).execute(
        filename=path.name,
        format="docx",
        content="Novo conteúdo",
    )

    assert result.success is True
    assert path.read_bytes() == b"original"
    numbered = tmp_path / "existente (1).docx"
    assert numbered.exists()
    assert result.metadata["filename"] == numbered.name
    document = docx.Document(numbered)
    assert "Novo conteúdo" in "\n".join(
        paragraph.text for paragraph in document.paragraphs
    )


def test_increments_duplicate_document_number(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    (tmp_path / "relatorio.docx").write_bytes(b"original")
    (tmp_path / "relatorio (1).docx").write_bytes(b"first copy")

    result = CreateDocumentTool([str(tmp_path)]).execute(
        filename="relatorio.docx",
        format="docx",
        content="Terceira versão",
    )

    assert result.success is True
    assert result.metadata["filename"] == "relatorio (2).docx"
    assert (tmp_path / "relatorio (2).docx").exists()


def test_strips_path_traversal_from_filename(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    result = CreateSpreadsheetTool([str(tmp_path)]).execute(
        filename="../fora.xlsx",
        columns=["Nome"],
        rows=[["Seguro"]],
    )

    assert result.success is True
    assert (tmp_path / "fora.xlsx").exists()
    assert not (tmp_path.parent / "fora.xlsx").exists()
