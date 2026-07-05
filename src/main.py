"""CLI entry point.

Usage:
    python -m src.main --jd data/job_descriptions/sample_jd.txt --resumes-dir data/resumes
    python -m src.main --jd data/job_descriptions/sample_jd.txt --resume data/resumes/sample.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.pipeline import ResumeRankingPipeline
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _print_leaderboard(ranked: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'RANK':<5}{'CANDIDATE':<28}{'ML SCORE':<11}{'SKILLS':<9}{'EXP (yrs)':<10}{'DEGREE'}")
    print("-" * 78)
    for c in ranked:
        print(
            f"{c['rank']:<5}{c['file_name'][:26]:<28}{c['ml_score']:<11}"
            f"{c['subscores']['skills_match']:<9}{c['experience']['total_years']:<10}"
            f"{c['education']['highest_degree']}"
        )
    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume Screening & Ranking System")
    parser.add_argument("--jd", required=True, help="Path to job description (.txt)")
    parser.add_argument("--resumes-dir", default="data/resumes", help="Folder of resumes to rank")
    parser.add_argument("--resume", default=None, help="Score a single resume file instead of a folder")
    parser.add_argument("--out", default="data/processed/ranking_results.json", help="Output JSON path")
    args = parser.parse_args()

    jd_path = Path(args.jd)
    if not jd_path.exists():
        logger.error("Job description file not found: %s", jd_path)
        return 1
    jd_text = jd_path.read_text(encoding="utf-8")

    pipeline = ResumeRankingPipeline()

    if args.resume:
        result = pipeline.score_one(args.resume, jd_text)
        print(json.dumps(result, indent=2))
        save_json(result, args.out)
        logger.info("Saved single-resume result to %s", args.out)
        return 0

    ranked, errors = pipeline.rank_folder(args.resumes_dir, jd_text)
    if not ranked:
        logger.warning("No resumes were successfully processed in %s", args.resumes_dir)

    _print_leaderboard(ranked)

    output = {"jd_source": str(jd_path), "results": ranked, "errors": errors}
    save_json(output, args.out)
    logger.info("Saved ranking results to %s (%d ranked, %d errors)", args.out, len(ranked), len(errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
