"""Builds one structured candidate profile from parsed resume text."""
from __future__ import annotations

from src.features.contact_extractor import extract_contact
from src.features.education_extractor import extract_education
from src.features.experience_extractor import extract_experience
from src.features.skills_extractor import extract_skills


def build_candidate_profile(parsed_resume: dict) -> dict:
    """`parsed_resume` is the dict returned by parser.resume_reader.read_resume()."""
    text = parsed_resume["cleaned_text"]

    return {
        "file_name": parsed_resume["file_name"],
        "quality_flag": parsed_resume["quality_flag"],
        "contact": extract_contact(text),
        "skills": extract_skills(text),
        "experience": extract_experience(text),
        "education": extract_education(text),
        "char_count": parsed_resume["char_count"],
    }


def build_jd_profile(jd_text: str) -> dict:
    """Job descriptions get the same skill/education extraction so we can
    diff them against candidate profiles."""
    return {
        "skills": extract_skills(jd_text),
        "education": extract_education(jd_text),
    }
