"""Authenticated downloads for files created by OpenJarvis tools."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from openjarvis.security.file_policy import is_sensitive_file
from openjarvis.tools.document_create import generated_output_roots

router = APIRouter(prefix="/v1/generated-files", tags=["generated-files"])

_MEDIA_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@router.get("/{filename}")
def download_generated_file(filename: str) -> FileResponse:
    """Download one generated artifact without accepting arbitrary paths."""
    safe_name = Path(filename).name
    if safe_name != filename or Path(safe_name).suffix.lower() not in _MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Invalid generated filename")

    for root in generated_output_roots():
        candidate = (root / safe_name).resolve()
        if candidate.parent != root.resolve():
            continue
        if candidate.is_file() and not is_sensitive_file(candidate):
            return FileResponse(
                candidate,
                media_type=_MEDIA_TYPES[candidate.suffix.lower()],
                filename=candidate.name,
                content_disposition_type="attachment",
            )
    raise HTTPException(status_code=404, detail="Generated file not found")


__all__ = ["router"]
