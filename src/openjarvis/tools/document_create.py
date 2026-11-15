"""Local document creation tools for DOCX, PDF, and XLSX files."""
# Modified by Gabriel_JP for the OpenJarvis local adaptation (2026-11-15).

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List, Optional

from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.security.file_policy import is_sensitive_file
from openjarvis.tools._paths import resolve_allowed_dirs
from openjarvis.tools._stubs import BaseTool, ToolSpec

_DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "OpenJarvis Files"
_MAX_TEXT_CHARS = 500_000
_MAX_ROWS = 10_000
_MAX_COLUMNS = 200


def generated_output_roots(allowed_dirs: Optional[List[str]] = None) -> List[Path]:
    """Return the directories from which generated artifacts may be served."""
    configured = [
        Path(directory).expanduser().resolve()
        for directory in resolve_allowed_dirs(allowed_dirs)
    ]
    return configured or [_DEFAULT_OUTPUT_DIR.resolve()]


def _safe_filename(filename: str, extension: str) -> str:
    raw = Path(str(filename).strip()).name
    if not raw or raw in {".", ".."}:
        raise ValueError("A valid filename is required.")
    return raw if raw.lower().endswith(extension) else raw + extension


def _semantic_filename(filename: str, extension: str, *, title: str, content: str) -> str:
    """Replace only generic timestamp names with a compact subject-based name."""
    raw = _safe_filename(filename, extension)
    stem = Path(raw).stem.casefold()
    if not re.fullmatch(r"(?:resumo|relatorio)_\d{8}_\d{6}", stem):
        return raw

    heading = next(
        (
            match.group(1)
            for line in content.splitlines()
            if (match := re.match(r"^#\s+(.+)$", line.strip()))
        ),
        "",
    )
    subject = heading or title
    normalized = unicodedata.normalize("NFKD", subject)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    words = re.findall(r"[a-z0-9]+", normalized.casefold())
    ignored = {
        "analise", "detalhada", "arquivo", "word", "relatorio", "resumo",
        "sobre", "do", "da", "de", "em", "no", "na", "o", "a",
    }
    topic = "_".join(word for word in words if word not in ignored)[:80].strip("_")
    prefix = "relatorio" if stem.startswith("relatorio") else "resumo"
    return f"{prefix}_{topic}{extension}" if topic else raw


def _is_within(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in roots)


class _LocalOutputTool(BaseTool):
    """Validate output paths shared by all local artifact tools."""

    def __init__(self, allowed_dirs: Optional[List[str]] = None) -> None:
        # Unlike the legacy file_write tool, an empty global allow-list does
        # not mean unrestricted access here. Generated files remain confined
        # to one predictable user folder.
        self._output_roots = generated_output_roots(allowed_dirs)

    def _output_path(
        self, filename: str, extension: str, *, overwrite: bool
    ) -> Path:
        path = (self._output_roots[0] / _safe_filename(filename, extension)).resolve()
        if not _is_within(path, self._output_roots):
            raise ValueError("The output path is outside the allowed directories.")
        if is_sensitive_file(path):
            raise ValueError("The requested output is considered a sensitive file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            # Keep every generated artifact instead of failing or silently
            # replacing an older result. This mirrors the familiar browser
            # convention: report.docx, report (1).docx, report (2).docx.
            stem = path.stem
            suffix = path.suffix
            counter = 1
            while True:
                candidate = path.with_name(f"{stem} ({counter}){suffix}").resolve()
                if not _is_within(candidate, self._output_roots):
                    raise ValueError(
                        "The output path is outside the allowed directories."
                    )
                if is_sensitive_file(candidate):
                    raise ValueError(
                        "The requested output is considered a sensitive file."
                    )
                if not candidate.exists():
                    path = candidate
                    break
                counter += 1
        return path


def _content_blocks(content: str):
    """Parse lightweight Markdown into semantic blocks."""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            yield "blank", ""
        elif match := re.match(r"^(#{1,3})\s+(.+)$", line):
            yield f"heading{len(match.group(1))}", match.group(2)
        elif re.match(r"^[-*]\s+", line):
            yield "bullet", re.sub(r"^[-*]\s+", "", line)
        elif re.match(r"^\d+[.)]\s+", line):
            yield "number", re.sub(r"^\d+[.)]\s+", "", line)
        else:
            yield "paragraph", line


def _markdown_table_cells(line: str) -> list[str]:
    """Split a simple Markdown-table row while preserving ordinary text."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_table_separator(line: str) -> bool:
    cells = _markdown_table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


@ToolRegistry.register("create_document")
class CreateDocumentTool(_LocalOutputTool):
    """Create a formatted Word or PDF file on the local machine."""

    tool_id = "create_document"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Create a local Word (.docx) or PDF (.pdf) document. Use this "
                "when the user asks to generate, save, export, or prepare a document. "
                "The content may contain simple Markdown headings and lists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Output filename, for example relatorio.docx.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["docx", "pdf"],
                        "description": "Document format to create.",
                    },
                    "title": {"type": "string", "description": "Optional title."},
                    "content": {
                        "type": "string",
                        "description": (
                            "Document body in plain text or simple Markdown."
                        ),
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "Replace an existing file. When false, a numbered copy "
                            "is created automatically. Default: false."
                        ),
                    },
                },
                "required": ["filename", "format", "content"],
            },
            category="filesystem",
            timeout_seconds=45.0,
            required_capabilities=["file:write"],
        )

    def execute(self, **params: Any) -> ToolResult:
        fmt = str(params.get("format", "")).lower().strip()
        content = str(params.get("content", ""))
        if fmt not in {"docx", "pdf"}:
            return self._failure("Format must be 'docx' or 'pdf'.")
        if not content.strip():
            return self._failure("Document content cannot be empty.")
        if len(content) > _MAX_TEXT_CHARS:
            return self._failure(
                f"Document content is too large (maximum {_MAX_TEXT_CHARS} characters)."
            )
        try:
            title = str(params.get("title", "")).strip()
            filename = _semantic_filename(
                str(params.get("filename", "")),
                f".{fmt}",
                title=title,
                content=content,
            )
            path = self._output_path(
                filename,
                f".{fmt}",
                overwrite=bool(params.get("overwrite", False)),
            )
            if fmt == "docx":
                self._create_docx(path, title, content)
            else:
                self._create_pdf(path, title, content)
        except (ValueError, FileExistsError, OSError) as exc:
            return self._failure(str(exc))
        except ImportError as exc:
            package = "python-docx" if fmt == "docx" else "reportlab"
            return self._failure(
                f"{package} is required to create {fmt.upper()} files: {exc}"
            )
        except Exception as exc:  # pragma: no cover - third-party boundary
            return self._failure(f"Could not create document: {exc}")

        return ToolResult(
            tool_name=self.tool_id,
            content=f"Document created successfully: {path}",
            success=True,
            metadata={
                "path": str(path),
                "filename": path.name,
                "format": fmt,
                "size_bytes": path.stat().st_size,
            },
        )

    def _failure(self, message: str) -> ToolResult:
        return ToolResult(tool_name=self.tool_id, content=message, success=False)

    @staticmethod
    def _create_docx(path: Path, title: str, content: str) -> None:
        from docx import Document
        from docx.shared import Pt

        document = Document()
        document.styles["Normal"].font.name = "Arial"
        document.styles["Normal"].font.size = Pt(11)
        if title:
            document.add_heading(title, level=0)
        for kind, text in _content_blocks(content):
            if kind == "blank":
                document.add_paragraph()
            elif kind.startswith("heading"):
                document.add_heading(text, level=int(kind[-1]))
            elif kind == "bullet":
                document.add_paragraph(text, style="List Bullet")
            elif kind == "number":
                document.add_paragraph(text, style="List Number")
            else:
                document.add_paragraph(text)
        document.save(path)

    @staticmethod
    def _create_pdf(path: Path, title: str, content: str) -> None:
        from xml.sax.saxutils import escape

        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "OpenJarvisTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            spaceAfter=10 * mm,
        )
        document = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            rightMargin=20 * mm,
            leftMargin=20 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            title=title or path.stem,
        )
        story = []
        if title:
            story.append(Paragraph(escape(title), title_style))
        style_map = {
            "heading1": styles["Heading1"],
            "heading2": styles["Heading2"],
            "heading3": styles["Heading3"],
            "paragraph": styles["BodyText"],
            "bullet": styles["BodyText"],
            "number": styles["BodyText"],
        }
        body_style = ParagraphStyle(
            "OpenJarvisBody",
            parent=styles["BodyText"],
            leading=15,
            spaceAfter=0,
        )

        def inline_markup(text: str) -> str:
            """Escape user content, then restore a small safe Markdown subset."""
            escaped = escape(text)
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
            escaped = re.sub(r"`([^`]+)`", r"<font name=\"Courier\">\1</font>", escaped)
            return escaped

        lines = content.splitlines()
        number = 0
        index = 0
        while index < len(lines):
            raw_line = lines[index]
            text = raw_line.strip()
            # Render a Markdown table as an actual PDF table. It must have a
            # header followed by the familiar --- separator to avoid treating
            # normal prose containing a pipe character as a table.
            if (
                "|" in text
                and index + 1 < len(lines)
                and _is_markdown_table_separator(lines[index + 1])
            ):
                header = _markdown_table_cells(text)
                table_rows = [header]
                cursor = index + 2
                while cursor < len(lines) and "|" in lines[cursor]:
                    row = _markdown_table_cells(lines[cursor])
                    if len(row) != len(header):
                        break
                    table_rows.append(row)
                    cursor += 1
                if len(table_rows) > 1:
                    available_width = A4[0] - 40 * mm
                    column_width = available_width / len(header)
                    pdf_rows = [
                        [Paragraph(inline_markup(cell), body_style) for cell in row]
                        for row in table_rows
                    ]
                    table = Table(pdf_rows, colWidths=[column_width] * len(header), repeatRows=1)
                    table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0891B2")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]))
                    story.extend([table, Spacer(1, 4 * mm)])
                    index = cursor
                    number = 0
                    continue

            if re.fullmatch(r"-{3,}", text):
                story.extend([HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")), Spacer(1, 4 * mm)])
                index += 1
                continue

            blocks = list(_content_blocks(raw_line))
            kind, text = blocks[0] if blocks else ("blank", "")
            if kind == "blank":
                story.append(Spacer(1, 4 * mm))
                index += 1
                continue
            prefix = ""
            if kind == "bullet":
                prefix = "• "
            elif kind == "number":
                number += 1
                prefix = f"{number}. "
            else:
                number = 0
            style = body_style if kind in {"paragraph", "bullet", "number"} else style_map[kind]
            story.append(Paragraph(inline_markup(prefix + text), style))
            story.append(Spacer(1, 2 * mm))
            index += 1
        document.build(story)


@ToolRegistry.register("create_spreadsheet")
class CreateSpreadsheetTool(_LocalOutputTool):
    """Create a formatted Excel workbook on the local machine."""

    tool_id = "create_spreadsheet"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description=(
                "Create a local Excel .xlsx spreadsheet from columns and rows. "
                "Use this when the user asks for an Excel file, workbook, "
                "table, or sheet."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Output filename, for example vendas.xlsx.",
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Worksheet name. Default: Dados.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional worksheet title displayed above the table.",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column headers.",
                    },
                    "rows": {
                        "type": "array",
                        "items": {"type": "array", "items": {}},
                        "description": "Rows of cell values in column order.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "Replace an existing file. When false, a numbered copy "
                            "is created automatically. Default: false."
                        ),
                    },
                    "include_summary": {
                        "type": "boolean",
                        "description": (
                            "Create a calculated Resumo sheet when the columns "
                            "describe financial receipts, expenses and balance."
                        ),
                    },
                },
                "required": ["filename", "columns", "rows"],
            },
            category="filesystem",
            timeout_seconds=45.0,
            required_capabilities=["file:write"],
        )

    def execute(self, **params: Any) -> ToolResult:
        columns = params.get("columns")
        rows = params.get("rows")
        if not isinstance(columns, list) or not columns:
            return self._failure("At least one spreadsheet column is required.")
        if not isinstance(rows, list) or any(not isinstance(row, list) for row in rows):
            return self._failure("Spreadsheet rows must be an array of arrays.")
        if len(columns) > _MAX_COLUMNS or len(rows) > _MAX_ROWS:
            return self._failure("The spreadsheet exceeds the supported size.")
        if any(len(row) > len(columns) for row in rows):
            return self._failure(
                "A row contains more values than the declared columns."
            )
        try:
            path = self._output_path(
                str(params.get("filename", "")),
                ".xlsx",
                overwrite=bool(params.get("overwrite", False)),
            )
            self._create_xlsx(
                path,
                str(params.get("sheet_name", "Dados")),
                [str(column) for column in columns],
                rows,
                title=str(params.get("title", "")).strip(),
                include_summary=bool(params.get("include_summary", False)),
            )
        except (ValueError, FileExistsError, OSError) as exc:
            return self._failure(str(exc))
        except ImportError as exc:
            return self._failure(f"openpyxl is required to create Excel files: {exc}")
        except Exception as exc:  # pragma: no cover - third-party boundary
            return self._failure(f"Could not create spreadsheet: {exc}")

        return ToolResult(
            tool_name=self.tool_id,
            content=f"Spreadsheet created successfully: {path}",
            success=True,
            metadata={
                "path": str(path),
                "filename": path.name,
                "format": "xlsx",
                "rows": len(rows),
                "columns": len(columns),
                "size_bytes": path.stat().st_size,
            },
        )

    def _failure(self, message: str) -> ToolResult:
        return ToolResult(tool_name=self.tool_id, content=message, success=False)

    @staticmethod
    def _create_xlsx(
        path: Path,
        sheet_name: str,
        columns: List[str],
        rows: List[List[Any]],
        *,
        title: str = "",
        include_summary: bool = False,
    ) -> None:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.worksheet.table import Table, TableStyleInfo
        from openpyxl.utils import get_column_letter

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = (sheet_name.strip() or "Dados")[:31]
        header_row = 2 if title else 1
        data_start_row = header_row + 1
        if title:
            sheet.merge_cells(
                start_row=1, start_column=1, end_row=1, end_column=len(columns)
            )
            title_cell = sheet.cell(row=1, column=1, value=title)
            title_cell.font = Font(color="FFFFFF", bold=True, size=14)
            title_cell.alignment = Alignment(horizontal="left", vertical="center")
            title_cell.fill = PatternFill("solid", fgColor="0F172A")
            sheet.row_dimensions[1].height = 26

        for index, column in enumerate(columns, start=1):
            sheet.cell(row=header_row, column=index, value=column)

        normalized_columns = [
            unicodedata.normalize("NFKD", str(column)).encode("ascii", "ignore")
            .decode("ascii")
            .casefold()
            for column in columns
        ]
        income_index = next(
            (index + 1 for index, value in enumerate(normalized_columns) if "receita" in value),
            None,
        )
        expense_index = next(
            (index + 1 for index, value in enumerate(normalized_columns) if "despesa" in value),
            None,
        )
        balance_index = next(
            (index + 1 for index, value in enumerate(normalized_columns) if "saldo" in value),
            None,
        )
        currency_indexes = {
            index + 1
            for index, value in enumerate(normalized_columns)
            if any(token in value for token in ("receita", "despesa", "saldo", "valor", "total"))
        }

        for row_index, row in enumerate(rows, start=data_start_row):
            for column_index, value in enumerate(row, start=1):
                sheet.cell(
                    row=row_index,
                    column=column_index,
                    value=CreateSpreadsheetTool._safe_cell_value(
                        value, currency=column_index in currency_indexes
                    ),
                )

        if income_index and expense_index and balance_index:
            income_column = get_column_letter(income_index)
            expense_column = get_column_letter(expense_index)
            balance_column = get_column_letter(balance_index)
            for row_index in range(data_start_row, data_start_row + len(rows)):
                preceding = (
                    f"{balance_column}{row_index - 1}+"
                    if row_index > data_start_row
                    else ""
                )
                sheet.cell(
                    row=row_index,
                    column=balance_index,
                    value=(
                        f"={preceding}{income_column}{row_index}-{expense_column}{row_index}"
                    ),
                )

        header_fill = PatternFill("solid", fgColor="0891B2")
        for cell in sheet[header_row]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
        thin_border = Border(bottom=Side(style="thin", color="CBD5E1"))
        for row in sheet.iter_rows(min_row=data_start_row, max_row=sheet.max_row):
            for cell in row:
                cell.border = thin_border
                cell.alignment = Alignment(vertical="center")
                if cell.column in currency_indexes:
                    cell.number_format = 'R$ #,##0.00'

        sheet.freeze_panes = f"A{data_start_row}"
        sheet.auto_filter.ref = (
            f"A{header_row}:{get_column_letter(len(columns))}{sheet.max_row}"
        )
        sheet.sheet_view.showGridLines = False
        for index, column in enumerate(columns, start=1):
            values = [column, *(row[index - 1] for row in rows if len(row) >= index)]
            width = min(max(len(str(value)) for value in values) + 3, 32)
            sheet.column_dimensions[get_column_letter(index)].width = max(width, 12)

        if rows:
            table = Table(
                displayName="DadosTabela",
                ref=f"A{header_row}:{get_column_letter(len(columns))}{sheet.max_row}",
            )
            table.tableStyleInfo = TableStyleInfo(
                name="TableStyleMedium2",
                showFirstColumn=False,
                showLastColumn=False,
                showRowStripes=True,
                showColumnStripes=False,
            )
            sheet.add_table(table)

        if income_index and expense_index and balance_index:
            CreateSpreadsheetTool._add_financial_summary(
                workbook,
                data_sheet=sheet,
                data_start_row=data_start_row,
                data_end_row=sheet.max_row,
                income_index=income_index,
                expense_index=expense_index,
                balance_index=balance_index,
                title=title,
            )
        workbook.save(path)

    @staticmethod
    def _safe_cell_value(value: Any, *, currency: bool) -> Any:
        """Keep untrusted text literal while storing supplied monetary values as numbers."""
        if currency and isinstance(value, str):
            raw = value.strip().replace("R$", "").replace(" ", "")
            if re.fullmatch(r"-?[\d.]+,[\d]{1,2}", raw):
                return float(raw.replace(".", "").replace(",", "."))
            if re.fullmatch(r"-?[\d,]+(?:\.\d{1,2})?", raw):
                return float(raw.replace(",", ""))
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    @staticmethod
    def _add_financial_summary(
        workbook: Any,
        *,
        data_sheet: Any,
        data_start_row: int,
        data_end_row: int,
        income_index: int,
        expense_index: int,
        balance_index: int,
        title: str,
    ) -> None:
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        summary = workbook.create_sheet("Resumo", 0)
        summary.merge_cells("A1:B1")
        summary["A1"] = title or "Resumo financeiro"
        summary["A1"].font = Font(color="FFFFFF", bold=True, size=14)
        summary["A1"].fill = PatternFill("solid", fgColor="0F172A")
        summary["A1"].alignment = Alignment(horizontal="left", vertical="center")
        summary.row_dimensions[1].height = 26
        summary["A3"] = "Indicador"
        summary["B3"] = "Valor"
        for cell in summary[3]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = PatternFill("solid", fgColor="0891B2")
            cell.alignment = Alignment(horizontal="center")

        sheet_name = data_sheet.title.replace("'", "''")
        income_column = get_column_letter(income_index)
        expense_column = get_column_letter(expense_index)
        balance_column = get_column_letter(balance_index)
        summary_rows = [
            ("Total de receitas", f"=SUM('{sheet_name}'!{income_column}{data_start_row}:{income_column}{data_end_row})"),
            ("Total de despesas", f"=SUM('{sheet_name}'!{expense_column}{data_start_row}:{expense_column}{data_end_row})"),
            ("Resultado líquido", "=B4-B5"),
            ("Saldo acumulado", f"='{sheet_name}'!{balance_column}{data_end_row}"),
        ]
        for row_index, (label, formula) in enumerate(summary_rows, start=4):
            summary.cell(row=row_index, column=1, value=label)
            value_cell = summary.cell(row=row_index, column=2, value=formula)
            value_cell.number_format = 'R$ #,##0.00'
            if row_index in {6, 7}:
                value_cell.font = Font(bold=True)
        summary.column_dimensions["A"].width = 24
        summary.column_dimensions["B"].width = 18
        summary.sheet_view.showGridLines = False


__all__ = ["CreateDocumentTool", "CreateSpreadsheetTool"]
