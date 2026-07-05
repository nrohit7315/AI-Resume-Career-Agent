"""Unit tests for similarity + ranking modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.similarity.tfidf_matcher import tfidf_similarity
from src.ranking.feature_scores import (
    education_match_score,
    experience_match_score,
    extract_required_years,
    skills_match_score,
)
from src.ranking.scorer import RankingModel, build_synthetic_training_set, heuristic_score
from src.ranking.ranker import rank_candidates


def test_tfidf_similarity_identical_text_is_high():
    text = "Python machine learning engineer with NLP experience."
    score = tfidf_similarity(text, text)
    assert score > 0.99


def test_tfidf_similarity_unrelated_text_is_low():
    a = "Python machine learning NLP scikit-learn TensorFlow."
    b = "Gardening flowers soil watering plants seasons."
    score = tfidf_similarity(a, b)
    assert score < 0.2


def test_tfidf_similarity_empty_text_returns_zero():
    assert tfidf_similarity("", "something") == 0.0
    assert tfidf_similarity("something", "") == 0.0


def test_skills_match_score_full_coverage():
    result = skills_match_score(["python", "sql", "excel"], ["python", "sql"])
    assert result["score"] == 1.0
    assert result["missing"] == []


def test_skills_match_score_partial_coverage():
    result = skills_match_score(["python"], ["python", "sql", "excel"])
    assert abs(result["score"] - (1 / 3)) < 1e-3  # score is rounded to 4 decimals
    assert "sql" in result["missing"]
    assert "excel" in result["missing"]


def test_skills_match_score_no_jd_skills():
    result = skills_match_score(["python"], [])
    assert result["score"] == 0.0


def test_extract_required_years():
    jd = "We require 3+ years of experience in software engineering."
    assert extract_required_years(jd) == 3.0


def test_experience_match_meets_requirement():
    assert experience_match_score(5.0, 3.0) == 1.0


def test_experience_match_below_requirement():
    score = experience_match_score(1.5, 3.0)
    assert 0.0 < score < 1.0


def test_education_match_meets_requirement():
    jd = "Requires a Master's degree in Computer Science."
    score = education_match_score(candidate_rank=3, jd_text=jd)  # 3 = master
    assert score == 1.0


def test_education_match_below_requirement():
    jd = "Requires a Master's degree in Computer Science."
    score = education_match_score(candidate_rank=2, jd_text=jd)  # 2 = bachelor
    assert 0.0 < score < 1.0


def test_heuristic_score_weighted_sum():
    weights = {"skills_match": 0.4, "semantic_similarity": 0.25, "experience_match": 0.2, "education_match": 0.15}
    subscores = {"skills_match": 1.0, "semantic_similarity": 1.0, "experience_match": 1.0, "education_match": 1.0}
    assert heuristic_score(subscores, weights) == 1.0


def test_ranking_model_fits_and_predicts_in_range():
    weights = {"skills_match": 0.4, "semantic_similarity": 0.25, "experience_match": 0.2, "education_match": 0.15}
    X, y_cont, y_bin = build_synthetic_training_set(weights, n_samples=200, random_state=1)
    model = RankingModel(model_type="gradient_boosting", random_state=1)
    model.fit(X, y_cont, y_bin)

    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    proba = model.predict_shortlist_probability(X[:10])
    assert proba is not None
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_rank_candidates_sorts_descending_by_ml_score():
    candidates = [
        {"file_name": "a.pdf", "ml_score": 0.5, "subscores": {"skills_match": 0.5}, "experience": {"total_years": 2}},
        {"file_name": "b.pdf", "ml_score": 0.9, "subscores": {"skills_match": 0.8}, "experience": {"total_years": 3}},
        {"file_name": "c.pdf", "ml_score": 0.7, "subscores": {"skills_match": 0.6}, "experience": {"total_years": 1}},
    ]
    ranked = rank_candidates(candidates)
    assert [c["file_name"] for c in ranked] == ["b.pdf", "c.pdf", "a.pdf"]
    assert ranked[0]["rank"] == 1


def test_rank_candidates_tiebreak_by_skills_then_experience():
    candidates = [
        {"file_name": "a.pdf", "ml_score": 0.8, "subscores": {"skills_match": 0.5}, "experience": {"total_years": 5}},
        {"file_name": "b.pdf", "ml_score": 0.8, "subscores": {"skills_match": 0.9}, "experience": {"total_years": 1}},
    ]
    ranked = rank_candidates(candidates)
    assert ranked[0]["file_name"] == "b.pdf"  # higher skills_match wins the tie


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
