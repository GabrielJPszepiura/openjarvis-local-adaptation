"""Local document creation tools for DOCX, PDF, and XLSX files."""

from __future__ import annotations

import re
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
            path = self._output_path(
                str(params.get("filename", "")),
                f".{fmt}",
                overwrite=bool(params.get("overwrite", False)),
            )
            title = str(params.get("title", "")).strip()
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
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

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
        number = 0
        for kind, text in _content_blocks(content):
            if kind == "blank":
                story.append(Spacer(1, 4 * mm))
                continue
            prefix = ""
            if kind == "bullet":
                prefix = "• "
            elif kind == "number":
                number += 1
                prefix = f"{number}. "
            else:
                number = 0
            story.append(Paragraph(escape(prefix + text), style_map[kind]))
            story.append(Spacer(1, 2 * mm))
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
        path: Path, sheet_name: str, columns: List[str], rows: List[List[Any]]
    ) -> None:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = (sheet_name.strip() or "Dados")[:31]
        sheet.append(columns)
        for row in rows:
            # Store formula-looking model/user text as literal text to prevent
            # spreadsheet formula injection when the workbook is opened.
            safe_row = [
                "'" + value
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                else value
                for value in row
            ]
            sheet.append(safe_row)

        header_fill = PatternFill("solid", fgColor="0891B2")
        for cell in sheet[1]:
            cell.font = Font(color="FFFFFF", bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for index, column in enumerate(columns, start=1):
            values = [column, *(row[index - 1] for row in rows if len(row) >= index)]
            width = min(max(len(str(value)) for value in values) + 2, 60)
            sheet.column_dimensions[get_column_letter(index)].width = max(width, 10)
        workbook.save(path)


__all__ = ["CreateDocumentTool", "CreateSpreadsheetTool"]
