"""Turns per-candidate scores into a final sorted leaderboard with tie-breaks."""
from __future__ import annotations


def rank_candidates(candidate_results: list[dict]) -> list[dict]:
    """Sort by ml_score desc; tie-break by skills_match, then experience_years.

    Ties are common at the extremes (e.g. two candidates both hitting every
    JD skill) so a deterministic tie-break avoids ranking looking arbitrary.
    """
    def sort_key(c: dict):
        return (
            -c["ml_score"],
            -c["subscores"]["skills_match"],
            -c["experience"]["total_years"],
        )

    ranked = sorted(candidate_results, key=sort_key)
    for idx, candidate in enumerate(ranked, start=1):
        candidate["rank"] = idx
    return ranked
