"""Extracts a normalized skill set from resume/JD text using a curated taxonomy.

Approach: deterministic keyword + alias matching over a domain taxonomy
(config/skills_taxonomy.json) rather than a black-box NER model. This is a
deliberate tradeoff — see PROJECT_NOTES.md, section "Why keyword-taxonomy
matching over spaCy NER":
  - Zero external model downloads -> zero network/runtime failure surface.
  - Fully deterministic & explainable (auditable per skill hit).
  - Easy to extend (just edit the JSON taxonomy).
  - Tradeoff: won't catch skills phrased in ways not in the taxonomy/aliases.
"""
from __future__ import annotations

import re
from functools import lru_cache

from src.utils.io import load_skills_taxonomy


def _build_skill_index(taxonomy: dict) -> dict[str, str]:
    """Map every known surface form (skill + its aliases) -> canonical skill name."""
    index: dict[str, str] = {}
    for category, skills in taxonomy.items():
        if category == "aliases":
            continue
        for skill in skills:
            index[skill.lower()] = skill.lower()

    for alias, canonical in taxonomy.get("aliases", {}).items():
        index[alias.lower()] = canonical.lower()

    return index


@lru_cache(maxsize=1)
def _get_index_and_category_map() -> tuple[dict[str, str], dict[str, str]]:
    taxonomy = load_skills_taxonomy()
    index = _build_skill_index(taxonomy)

    category_map: dict[str, str] = {}
    for category, skills in taxonomy.items():
        if category == "aliases":
            continue
        for skill in skills:
            category_map[skill.lower()] = category

    return index, category_map


def _compile_pattern(surface_forms: list[str]) -> re.Pattern:
    # Longest-first so multi-word phrases ("machine learning") match before
    # any accidental single-word overlap.
    escaped = sorted((re.escape(s) for s in surface_forms), key=len, reverse=True)
    pattern = r"(?<![a-zA-Z0-9])(" + "|".join(escaped) + r")(?![a-zA-Z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


def extract_skills(text: str) -> dict:
    """Return canonical skills found in `text`, grouped by taxonomy category."""
    if not text:
        return {"skills": [], "by_category": {}, "raw_hits": {}}

    index, category_map = _get_index_and_category_map()
    pattern = _compile_pattern(list(index.keys()))

    text_lower = text.lower()
    raw_hits: dict[str, int] = {}
    for match in pattern.finditer(text_lower):
        surface = match.group(1).lower()
        raw_hits[surface] = raw_hits.get(surface, 0) + 1

    canonical_hits: dict[str, int] = {}
    for surface, count in raw_hits.items():
        canonical = index[surface]
        canonical_hits[canonical] = canonical_hits.get(canonical, 0) + count

    by_category: dict[str, list[str]] = {}
    for skill in canonical_hits:
        category = category_map.get(skill, "other")
        by_category.setdefault(category, []).append(skill)

    for cat in by_category:
        by_category[cat] = sorted(by_category[cat])

    return {
        "skills": sorted(canonical_hits.keys()),
        "skill_frequency": canonical_hits,
        "by_category": by_category,
        "raw_hits": raw_hits,
    }
