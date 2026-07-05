"""PDF resume text extraction (PyMuPDF / fitz)."""
from __future__ import annotations

from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF cannot be read or yields no usable text."""


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract raw text from a PDF resume.

    Uses PyMuPDF for speed and robustness against malformed PDFs. Falls back
    to a clear, actionable error instead of crashing the pipeline so a single
    bad resume never takes down a batch run.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise PDFParseError(f"File not found: {file_path}")

    try:
        import fitz # PyMuPDF
    except ImportError as exc:
        raise PDFParseError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        ) from exc

    text_chunks: list[str] = []
    try:
        with fitz.open(file_path) as doc:
            if doc.is_encrypted:
                # Try an empty-password unlock (common for "protected" exports).
                if not doc.authenticate(""):
                    raise PDFParseError(f"PDF is password-protected: {file_path.name}")
            for page in doc:
                text_chunks.append(page.get_text("text"))
    except PDFParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - we want ANY parser failure caught
        raise PDFParseError(f"Failed to parse PDF '{file_path.name}': {exc}") from exc

    raw_text = "\n".join(text_chunks)
    if not raw_text.strip():
        logger.warning("No extractable text found in %s (likely a scanned/image PDF).", file_path.name)
    return raw_text
