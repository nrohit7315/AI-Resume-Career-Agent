"""Combines TF-IDF and semantic embedding similarity into one hybrid score.

If embeddings are unavailable, the hybrid score degrades gracefully to
TF-IDF alone (weight redistributed to 1.0) rather than erroring.
"""
from __future__ import annotations

from src.similarity.embedding_matcher import embedding_similarity
from src.similarity.tfidf_matcher import tfidf_similarity


def hybrid_similarity(
    resume_text: str,
    jd_text: str,
    embeddings_enabled: bool = True,
    tfidf_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> dict:
    tfidf_score = tfidf_similarity(resume_text, jd_text)

    embed_score = None
    if embeddings_enabled:
        embed_score = embedding_similarity(resume_text, jd_text)

    if embed_score is None:
        hybrid_score = tfidf_score
        mode = "tfidf_only"
    else:
        hybrid_score = round(tfidf_weight * tfidf_score + embedding_weight * embed_score, 4)
        mode = "hybrid_tfidf_embedding"

    return {
        "tfidf_score": tfidf_score,
        "embedding_score": embed_score,
        "hybrid_score": hybrid_score,
        "mode": mode,
    }
