"""FastAPI REST server for the resume screening & ranking pipeline.

Run with:
    uvicorn src.api:app --reload --port 8000
Docs at:
    http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import ResumeRankingPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="HireAI Resume Ranker API",
    description="Parses resumes, extracts features, and ranks candidates against a job description.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: ResumeRankingPipeline | None = None


def get_pipeline() -> ResumeRankingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ResumeRankingPipeline()
    return _pipeline


class HealthResponse(BaseModel):
    status: str
    model_type: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline = get_pipeline()
    return HealthResponse(status="ok", model_type=pipeline.model.model_type)


@app.post("/rank")
async def rank_resumes(
    job_description: str = Form(..., description="Raw job description text"),
    files: list[UploadFile] = File(..., description="One or more resume files (.pdf, .docx, .txt)"),
):
    """Upload one or more resumes + a job description; get back a ranked leaderboard."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one resume file is required.")

    pipeline = get_pipeline()
    results = []
    errors = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in {".pdf", ".docx", ".txt"}:
                errors.append({"file_name": upload.filename, "error": f"Unsupported format '{suffix}'"})
                continue

            tmp_path = Path(tmp_dir) / upload.filename
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(upload.file, f)

            try:
                results.append(pipeline.score_one(tmp_path, job_description))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed scoring %s: %s", upload.filename, exc)
                errors.append({"file_name": upload.filename, "error": str(exc)})

    from src.ranking.ranker import rank_candidates
    ranked = rank_candidates(results)

    return {"results": ranked, "errors": errors, "total_submitted": len(files), "total_ranked": len(ranked)}


@app.get("/")
def root():
    return {
        "service": "HireAI Resume Ranker API",
        "docs": "/docs",
        "endpoints": {"health": "GET /health", "rank": "POST /rank (multipart: job_description, files[])"},
    }
