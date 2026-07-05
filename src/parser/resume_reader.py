"""Unified entry point for reading any supported resume format."""
from __future__ import annotations

from pathlib import Path

from src.parser.docx_parser import extract_text_from_docx, extract_text_from_txt
from src.parser.pdf_parser import extract_text_from_pdf
from src.parser.text_cleaner import clean_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DISPATCH = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
}


class UnsupportedFormatError(Exception):
    pass


def read_resume(file_path: str | Path, min_text_length: int = 50) -> dict:
    """Read + clean a resume file, returning a small result envelope so
    callers can distinguish "parsed fine" from "parsed but suspicious"
    without ever raising on a low-quality (e.g. scanned) file.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext not in _DISPATCH:
        raise UnsupportedFormatError(
            f"Unsupported resume format '{ext}'. Supported: {list(_DISPATCH)}"
        )

    raw_text = _DISPATCH[ext](file_path)
    cleaned = clean_text(raw_text)

    quality_flag = "ok" if len(cleaned) >= min_text_length else "low_text_extracted"
    if quality_flag != "ok":
        logger.warning(
            "%s: only %d chars extracted (possible scanned/image resume).",
            file_path.name, len(cleaned),
        )

    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "format": ext.lstrip("."),
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "char_count": len(cleaned),
        "quality_flag": quality_flag,
    }
