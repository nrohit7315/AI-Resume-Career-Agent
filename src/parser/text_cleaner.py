"""Normalizes raw extracted text before feature extraction / similarity scoring."""
import re
import unicodedata

_BULLET_CHARS = "•▪◦‣∙●○·–—"
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(raw_text: str) -> str:
    """Normalize unicode, strip bullets/control chars, collapse whitespace."""
    if not raw_text:
        return ""

    text = unicodedata.normalize("NFKC", raw_text)

    # Strip common bullet glyphs used in resumes (kept as newlines so list
    # structure doesn't collapse into one giant sentence).
    for ch in _BULLET_CHARS:
        text = text.replace(ch, "\n")

    # Remove non-printable / control characters but keep newlines & tabs.
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")

    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def to_plain_sentence_blob(cleaned_text: str) -> str:
    """Flatten cleaned, multi-line resume text into a single blob for TF-IDF /
    embedding similarity scoring (line breaks -> spaces)."""
    return re.sub(r"\s+", " ", cleaned_text).strip()
