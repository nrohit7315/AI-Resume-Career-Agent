"""Optional Sentence-Transformer embedding similarity.

Design note (see PROJECT_NOTES.md for the full tradeoff writeup): this
matcher requires downloading a pretrained model (~80MB) from the Hugging
Face Hub on first use. In locked-down / offline environments that download
can fail. Rather than let that crash the pipeline, this module:

  1. Lazily imports sentence-transformers only when called.
  2. Caches the loaded model at module scope (loaded once per process).
  3. Catches *any* failure (import error, network error, OOM) and returns
     None so the hybrid_scorer can transparently fall back to TF-IDF-only
     scoring with zero user-facing error.

This keeps the "never error mid-pipeline" requirement true even when the
network layer is unavailable.
"""
from __future__ import annotations

from functools import lru_cache

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_NAME_DEFAULT = "all-MiniLM-L6-v2"
_embedding_unavailable_logged = False


@lru_cache(maxsize=1)
def _load_model(model_name: str = _MODEL_NAME_DEFAULT):
    from sentence_transformers import SentenceTransformer  # local import by design
    return SentenceTransformer(model_name)


def embedding_similarity(resume_text: str, jd_text: str, model_name: str = _MODEL_NAME_DEFAULT) -> float | None:
    """Returns cosine similarity in [0, 1], or None if embeddings are unavailable."""
    global _embedding_unavailable_logged

    if not resume_text.strip() or not jd_text.strip():
        return None

    try:
        model = _load_model(model_name)
        from sentence_transformers import util

        embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return float(round(max(0.0, min(1.0, score)), 4))
    except Exception as exc:  # noqa: BLE001 - any failure => graceful fallback
        if not _embedding_unavailable_logged:
            logger.info(
                "Semantic embedding model unavailable (%s: %s). "
                "Falling back to TF-IDF-only similarity for this run.",
                type(exc).__name__, exc,
            )
            _embedding_unavailable_logged = True
        return None
