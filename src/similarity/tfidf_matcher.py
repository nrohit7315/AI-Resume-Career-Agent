"""TF-IDF + cosine similarity between a resume and a job description.

Always available (no external model downloads), so this is the guaranteed
baseline signal even when the optional embedding matcher is unavailable.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(
    resume_text: str,
    jd_text: str,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str = "english",
) -> float:
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    except ValueError:
        # e.g. text is entirely stop-words / too short to build a vocabulary.
        return 0.0

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(round(score, 4))
