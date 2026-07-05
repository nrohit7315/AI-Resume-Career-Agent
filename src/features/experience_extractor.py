"""Extracts work-experience signals: total years, seniority level.

Uses regex heuristics over date ranges + explicit "X years" phrases. This is
intentionally rule-based (not ML) because resume date formatting is highly
irregular and a small labeled dataset would overfit; see PROJECT_NOTES.md.

IMPORTANT: date-range inference is scoped to the resume's EXPERIENCE /
WORK HISTORY section only (see `_slice_experience_section`). Without this,
education date ranges ("2018 - 2021") get misread as work-experience
duration, which silently inflates every candidate's years.
"""
from __future__ import annotations

import re
from datetime import datetime

_YEAR_PHRASE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b(?:\s*of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_MONTH}\.?\s*\d{{4}}|\d{{4}})\s*(?:-|–|—|to)\s*"
    rf"(?P<end>{_MONTH}\.?\s*\d{{4}}|\d{{4}}|present|current|ongoing)",
    re.IGNORECASE,
)

SENIORITY_KEYWORDS = {
    "intern": ["intern", "internship", "trainee"],
    "entry": ["junior", "associate", "entry level", "graduate"],
    "mid": ["engineer", "developer", "analyst", "specialist"],
    "senior": ["senior", "lead", "principal", "staff"],
    "management": ["manager", "director", "head of", "vp", "chief", "cto", "ceo"],
}

# Section headers that bound the "experience" region of a resume.
_EXPERIENCE_HEADERS = [
    "experience", "work experience", "professional experience",
    "employment history", "work history",
]
_SECTION_STOP_HEADERS = [
    "education", "skills", "projects", "certifications", "certification",
    "achievements", "publications", "awards", "summary", "objective",
    "interests", "references",
]


def _slice_experience_section(text: str) -> str:
    """Return only the text between an EXPERIENCE-like header and the next
    section header. If no clear header is found, fall back to the full text
    (safer than returning nothing) but callers should treat that as lower
    confidence -- reflected via `source` in the returned dict.
    """
    lines = text.split("\n")
    start_idx = None
    for i, line in enumerate(lines):
        clean = line.strip().lower().rstrip(":")
        if clean in _EXPERIENCE_HEADERS:
            start_idx = i + 1
            break

    if start_idx is None:
        return text  # fallback: whole document

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        clean = lines[j].strip().lower().rstrip(":")
        if clean in _SECTION_STOP_HEADERS:
            end_idx = j
            break

    return "\n".join(lines[start_idx:end_idx])


def _parse_year(token: str) -> int | None:
    token = token.strip().lower()
    if token in {"present", "current", "ongoing"}:
        return datetime.now().year
    match = re.search(r"\d{4}", token)
    return int(match.group()) if match else None


def _years_from_date_ranges(text: str) -> float:
    total_months = 0
    seen_ranges = set()
    for m in _DATE_RANGE_RE.finditer(text):
        start_year = _parse_year(m.group("start"))
        end_year = _parse_year(m.group("end"))
        if start_year and end_year and end_year >= start_year:
            key = (start_year, end_year, m.start())
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            years = end_year - start_year
            total_months += max(years, 0) * 12 + 6  # +6mo avg mid-year offset
    return round(total_months / 12, 1)


def _years_from_explicit_phrase(text: str) -> float:
    matches = [float(m.group(1)) for m in _YEAR_PHRASE_RE.finditer(text)]
    return max(matches) if matches else 0.0


def _detect_seniority(text: str) -> str:
    text_lower = text.lower()
    # Priority order: management > senior > mid > entry > intern
    for level in ["management", "senior", "mid", "entry", "intern"]:
        for kw in SENIORITY_KEYWORDS[level]:
            if kw in text_lower:
                return level
    return "unspecified"


def extract_experience(text: str) -> dict:
    if not text:
        return {"total_years": 0.0, "seniority": "unspecified", "source": "none"}

    # Explicit "X years of experience" phrases can appear anywhere (often in
    # a summary at the top), so scan the full document for those.
    explicit_years = _years_from_explicit_phrase(text)

    # Date-range inference is scoped to the EXPERIENCE section to avoid
    # picking up education years.
    experience_section = _slice_experience_section(text)
    inferred_years = _years_from_date_ranges(experience_section)

    if explicit_years > 0:
        total_years = explicit_years
        source = "explicit_phrase"
    elif inferred_years > 0:
        total_years = inferred_years
        source = "date_range_inference_scoped"
    else:
        total_years = 0.0
        source = "none"

    return {
        "total_years": total_years,
        "seniority": _detect_seniority(text),
        "source": source,
    }
