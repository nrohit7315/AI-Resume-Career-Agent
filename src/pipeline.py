"""End-to-end pipeline: resumes + JD -> ranked leaderboard.

This is the single orchestration point reused by the CLI (main.py), the
REST API (api.py), and the Streamlit UI (ui/app.py) so all three surfaces
are guaranteed to produce identical results from identical inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.profile_builder import build_candidate_profile, build_jd_profile
from src.parser.resume_reader import read_resume
from src.ranking.feature_scores import (
    education_match_score,
    experience_match_score,
    extract_required_years,
    skills_match_score,
)
from src.ranking.ranker import rank_candidates
from src.ranking.scorer import (
    RankingModel,
    build_synthetic_training_set,
    heuristic_score,
    subscores_to_vector,
)
from src.similarity.hybrid_scorer import hybrid_similarity
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResumeRankingPipeline:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.weights = self.config["scoring_weights"]
        self.embeddings_enabled = self.config["similarity"]["embeddings"]["enabled"]
        self.min_text_length = self.config["parsing"]["min_text_length"]

        self.model = RankingModel(
            model_type=self.config["ranking"]["model_type"],
            random_state=self.config["ranking"]["random_state"],
        )
        self._train_model()

    def _train_model(self) -> None:
        X, y_cont, y_bin = build_synthetic_training_set(
            weights=self.weights,
            random_state=self.config["ranking"]["random_state"],
        )
        self.model.fit(X, y_cont, y_bin)
        logger.info("Ranking model (%s) trained on %d synthetic samples.", self.model.model_type, len(X))

    def score_one(self, resume_path: str | Path, jd_text: str) -> dict:
        """Full pipeline for a single resume against a single JD."""
        parsed = read_resume(resume_path, min_text_length=self.min_text_length)
        profile = build_candidate_profile(parsed)
        jd_profile = build_jd_profile(jd_text)

        sim = hybrid_similarity(
            parsed["cleaned_text"], jd_text,
            embeddings_enabled=self.embeddings_enabled,
        )

        skills_result = skills_match_score(profile["skills"]["skills"], jd_profile["skills"]["skills"])
        required_years = extract_required_years(jd_text)
        exp_score = experience_match_score(profile["experience"]["total_years"], required_years)
        edu_score = education_match_score(profile["education"]["degree_level_rank"], jd_text)

        subscores = {
            "skills_match": skills_result["score"],
            "semantic_similarity": sim["hybrid_score"],
            "experience_match": exp_score,
            "education_match": edu_score,
        }

        h_score = heuristic_score(subscores, self.weights)
        vector = np.array([subscores_to_vector(subscores)])
        ml_score = float(round(self.model.predict(vector)[0], 4))
        shortlist_proba = self.model.predict_shortlist_probability(vector)
        shortlist_probability = float(round(shortlist_proba[0], 4)) if shortlist_proba is not None else None

        return {
            "file_name": profile["file_name"],
            "quality_flag": profile["quality_flag"],
            "contact": profile["contact"],
            "skills": profile["skills"]["skills"],
            "skills_by_category": profile["skills"]["by_category"],
            "experience": profile["experience"],
            "education": profile["education"],
            "similarity": sim,
            "skills_match_detail": skills_result,
            "subscores": subscores,
            "heuristic_score": h_score,
            "ml_score": ml_score,
            "shortlist_probability": shortlist_probability,
            "required_years_detected": required_years,
        }

    def rank_folder(self, resumes_dir: str | Path, jd_text: str) -> tuple[list[dict], list[dict]]:
        from src.utils.io import list_resume_files

        results = []
        errors = []
        for resume_path in list_resume_files(resumes_dir):
            try:
                results.append(self.score_one(resume_path, jd_text))
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
                logger.error("Skipping '%s' due to error: %s", resume_path.name, exc)
                errors.append({"file_name": resume_path.name, "error": str(exc)})

        ranked = rank_candidates(results)
        return ranked, errors
