"""Turns extracted profile features into normalized [0, 1] sub-scores that
feed the ranking model. Kept separate from scorer.py so each component is
independently testable.
"""
from __future__ import annotations

import re


def skills_match_score(candidate_skills: list[str], jd_skills: list[str]) -> dict:
    """Coverage of JD-required skills found in the candidate's resume.

    Uses coverage (JD-skills-found / JD-skills-total) rather than plain
    Jaccard similarity, because a candidate having *extra* skills the JD
    didn't ask for shouldn't lower their score.
    """
    jd_set = set(jd_skills)
    cand_set = set(candidate_skills)

    if not jd_set:
        return {"score": 0.0, "matched": [], "missing": [], "extra": sorted(cand_set)}

    matched = sorted(jd_set & cand_set)
    missing = sorted(jd_set - cand_set)
    extra = sorted(cand_set - jd_set)

    score = round(len(matched) / len(jd_set), 4)
    return {"score": score, "matched": matched, "missing": missing, "extra": extra}


_REQUIRED_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b(?:\s*of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)


def extract_required_years(jd_text: str) -> float | None:
    matches = [float(m.group(1)) for m in _REQUIRED_YEARS_RE.finditer(jd_text)]
    return max(matches) if matches else None


def experience_match_score(candidate_years: float, required_years: float | None) -> float:
    """If the JD states a required years-of-experience, score how close the
    candidate is (meeting/exceeding = full score, under = linear penalty).
    If the JD doesn't specify, fall back to a saturating curve so more
    experience still helps but with diminishing returns.
    """
    if required_years is None or required_years <= 0:
        # Saturating curve: 0y -> 0.2, 3y -> ~0.7, 6y+ -> ~0.95+
        return round(min(0.95, 0.2 + candidate_years * 0.15), 4)

    if candidate_years >= required_years:
        return 1.0

    return round(max(0.0, candidate_years / required_years), 4)


_DEGREE_RANK_MIN_FOR_JD = {
    "diploma": 1, "bachelor": 2, "master": 3, "phd": 4, "unspecified": 0,
}


def education_match_score(candidate_rank: int, jd_text: str) -> float:
    """If the JD names a minimum degree level, score against it; otherwise
    reward any completed higher education on a gentle scale.
    """
    jd_lower = jd_text.lower()
    required_rank = 0
    for level, rank in _DEGREE_RANK_MIN_FOR_JD.items():
        if level != "unspecified" and level in jd_lower:
            required_rank = max(required_rank, rank)

    if required_rank == 0:
        # No explicit requirement -> gentle reward curve capped at 1.0
        return round(min(1.0, candidate_rank / 3), 4)

    if candidate_rank >= required_rank:
        return 1.0
    if candidate_rank == 0:
        return 0.0
    return round(candidate_rank / required_rank, 4)
