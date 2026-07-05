"""Small IO helpers shared across the pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_skills_taxonomy(taxonomy_path: str | Path = "config/skills_taxonomy.json") -> dict:
    path = Path(taxonomy_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(in_path: str | Path) -> Any:
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_resume_files(resumes_dir: str | Path) -> list[Path]:
    resumes_dir = Path(resumes_dir)
    if not resumes_dir.is_absolute():
        resumes_dir = PROJECT_ROOT / resumes_dir
    exts = {".pdf", ".docx", ".txt"}
    return sorted([p for p in resumes_dir.glob("*") if p.suffix.lower() in exts])


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
