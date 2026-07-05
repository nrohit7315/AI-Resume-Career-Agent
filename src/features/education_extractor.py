"""Extracts education signals: highest degree level + field of study.

NOTE: degree keyword matching uses word-boundary regex patterns rather than
naive substring checks. Short abbreviations like "B.E." / "M.E." are
otherwise dangerous substring-wise (e.g. the plain-English phrase "time
management" contains the raw substring "me ", which would false-positive
match a bare "me" keyword for a Master's degree). The dotted forms
("B.E.", "M.E.") are required specifically for those two to avoid
colliding with ordinary English words ("be", "me").
"""
from __future__ import annotations

import re

_DIPLOMA_PATTERNS = [r"\bdiploma\b", r"\bpolytechnic\b"]
_BACHELOR_PATTERNS = [
    r"\bb\.e\.?\b", r"\bb\.?\s?tech\b", r"\bbachelor(?:'s)?\b",
    r"\bb\.?\s?sc\.?\b", r"\bb\.?\s?a\.?\b", r"\bbca\b", r"\bb\.?\s?com\b",
]
_MASTER_PATTERNS = [
    r"\bm\.e\.?\b", r"\bm\.?\s?tech\b", r"\bmaster(?:'s)?\b",
    r"\bm\.?\s?sc\.?\b", r"\bmba\b", r"\bmca\b",
]
_PHD_PATTERNS = [r"\bph\.?\s?d\.?\b", r"\bdoctorate\b", r"\bdoctoral\b"]

# Ordered lowest -> highest so we can take the max when multiple are found.
DEGREE_LEVELS = [
    ("diploma", [re.compile(p, re.IGNORECASE) for p in _DIPLOMA_PATTERNS]),
    ("bachelor", [re.compile(p, re.IGNORECASE) for p in _BACHELOR_PATTERNS]),
    ("master", [re.compile(p, re.IGNORECASE) for p in _MASTER_PATTERNS]),
    ("phd", [re.compile(p, re.IGNORECASE) for p in _PHD_PATTERNS]),
]

FIELD_KEYWORDS = [
    "artificial intelligence", "machine learning", "computer science",
    "information technology", "data science", "electronics", "electrical",
    "mechanical", "civil engineering", "business administration",
    "software engineering",
]


def extract_education(text: str) -> dict:
    if not text:
        return {"highest_degree": "unspecified", "fields": [], "degree_level_rank": 0}

    found_level = "unspecified"
    found_rank = 0
    for rank, (level, patterns) in enumerate(DEGREE_LEVELS, start=1):
        if any(p.search(text) for p in patterns):
            found_level = level
            found_rank = rank  # keep updating -> ends on highest match

    text_lower = text.lower()
    fields = sorted({f for f in FIELD_KEYWORDS if f in text_lower})

    return {
        "highest_degree": found_level,
        "degree_level_rank": found_rank,   # 0-4, used for numeric comparisons
        "fields": fields,
    }
