"""Extracts contact info: email, phone, LinkedIn/GitHub links."""
from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/\S+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/\S+", re.IGNORECASE)


def extract_contact(text: str) -> dict:
    if not text:
        return {"email": None, "phone": None, "linkedin": None, "github": None}

    email_match = _EMAIL_RE.search(text)
    phone_match = _PHONE_RE.search(text)
    linkedin_match = _LINKEDIN_RE.search(text)
    github_match = _GITHUB_RE.search(text)

    return {
        "email": email_match.group().strip() if email_match else None,
        "phone": re.sub(r"[\s-]", "", phone_match.group()).strip() if phone_match else None,
        "linkedin": linkedin_match.group().rstrip(".,") if linkedin_match else None,
        "github": github_match.group().rstrip(".,") if github_match else None,
    }
