"""Unit tests for feature extraction modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.education_extractor import extract_education
from src.features.experience_extractor import extract_experience
from src.features.skills_extractor import extract_skills
from src.features.contact_extractor import extract_contact


def test_skills_extractor_basic():
    text = "Experienced in Python, scikit-learn, and Machine Learning. Also know SQL."
    result = extract_skills(text)
    assert "python" in result["skills"]
    assert "scikit-learn" in result["skills"]
    assert "machine learning" in result["skills"]
    assert "sql" in result["skills"]


def test_skills_extractor_alias_resolution():
    text = "Strong background in NLP and ML, familiar with k8s and postgres."
    result = extract_skills(text)
    assert "natural language processing" in result["skills"]
    assert "machine learning" in result["skills"]
    assert "kubernetes" in result["skills"]
    assert "postgresql" in result["skills"]


def test_education_no_false_positive_on_time_management():
    """Regression test: 'time management' must NOT be detected as a Master's degree."""
    text = "Skills: Content Writing, Communication, Excel, Presentation, Time Management, Teamwork"
    result = extract_education(text)
    assert result["highest_degree"] == "unspecified"


def test_education_detects_bachelor():
    text = "B.A. in English Literature, Delhi University, 2018 - 2021"
    result = extract_education(text)
    assert result["highest_degree"] == "bachelor"


def test_education_detects_masters_with_dots():
    text = "M.E. in Computer Science, XYZ University"
    result = extract_education(text)
    assert result["highest_degree"] == "master"


def test_education_picks_highest_degree():
    text = "B.Tech in IT (2016-2020). Later completed M.Tech in Artificial Intelligence (2021-2023)."
    result = extract_education(text)
    assert result["highest_degree"] == "master"
    assert result["degree_level_rank"] == 3


def test_experience_ignores_education_date_range():
    """Regression test: education date ranges must not be counted as work experience."""
    text = (
        "EDUCATION\n"
        "B.Tech in Computer Science, VIT Vellore, 2015 - 2019\n\n"
        "EXPERIENCE\n"
        "ML Engineer, Acme Corp (2023 - Present)\n"
        "- Built ML pipelines.\n"
    )
    result = extract_experience(text)
    # Should be based only on the 2023-present range, not 2015-2019 too.
    assert result["total_years"] < 6.0


def test_experience_explicit_phrase_takes_precedence():
    text = "Data Scientist with 5 years of experience in machine learning."
    result = extract_experience(text)
    assert result["total_years"] == 5.0
    assert result["source"] == "explicit_phrase"


def test_experience_empty_text():
    result = extract_experience("")
    assert result["total_years"] == 0.0
    assert result["source"] == "none"


def test_contact_extractor():
    text = "Contact: jane.doe@example.com | +91-9876543210\nlinkedin.com/in/janedoe"
    result = extract_contact(text)
    assert result["email"] == "jane.doe@example.com"
    assert result["linkedin"] is not None


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
