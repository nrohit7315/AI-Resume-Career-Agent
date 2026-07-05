# 🎬 HireAI — Resume Screening & Ranking System

An end-to-end **NLP + Machine Learning** system that parses resumes, extracts
structured candidate features, scores each candidate against a job
description on multiple dimensions, and produces a ranked shortlist —
built for the Rooman Technologies **AI Product Engineer** project
assessment.

> Assessment brief: *Resume Screening and Ranking System Using NLP and ML*
> — parse resumes, extract skills/experience/education, compute an
> NLP-based similarity score against a job description, and rank
> candidates using a machine learning model.

---

## ✨ What it does

1. **Parses** resumes in PDF, DOCX, or TXT format.
2. **Extracts structured features**: skills (taxonomy + alias matching),
   years of experience & seniority, highest education level, and contact
   info.
3. **Scores semantic fit** against the job description using a hybrid
   **TF-IDF + sentence-embedding** similarity pipeline (embeddings are
   optional and the system degrades gracefully to TF-IDF-only if the
   embedding model isn't available — see [Notes](#notes--tradeoffs)).
4. **Ranks candidates** with a trained **scikit-learn Gradient Boosting**
   model that combines skills / semantic / experience / education
   sub-scores, alongside a Logistic Regression classifier that estimates
   shortlist probability.
5. **Serves results** three ways: a CLI, a REST API (FastAPI), and an
   interactive Streamlit UI ("Screening Room").

---

## 🖥️ Screening Room UI

A themed Streamlit front-end — resumes are treated as "reels," the ranked
output as tonight's "Callback List." Upload a JD and a batch of resumes,
get a live-scored, expandable leaderboard per candidate (matched vs.
missing skills, sub-score breakdown, contact info).

```
streamlit run ui/app.py
```

---

## 🏗️ Architecture

```
                     ┌──────────────────┐
   resumes (.pdf/    │   src/parser/    │   cleaned text
   .docx/.txt)  ───▶  │  (PyMuPDF,       │ ───────────────┐
                     │  python-docx)     │                 │
                     └──────────────────┘                 ▼
                                                ┌───────────────────┐
   job description ──────────────────────────▶ │  src/features/     │
   (raw text)                                   │  skills / exp /    │
                                                 │  education /       │
                                                 │  contact           │
                                                 └─────────┬──────────┘
                                                            │
                              ┌─────────────────────────────┴─────────────┐
                              ▼                                           ▼
                 ┌─────────────────────────┐             ┌──────────────────────────┐
                 │  src/similarity/         │             │  src/ranking/             │
                 │  TF-IDF + embeddings      │            │  feature_scores.py        │
                 │  (hybrid_scorer.py)       │            │  (skills/exp/edu match)   │
                 └─────────────┬─────────────┘             └─────────────┬─────────────┘
                                └───────────────────┬───────────────────┘
                                                     ▼
                                     ┌───────────────────────────────┐
                                     │  src/ranking/scorer.py          │
                                     │  GradientBoostingRegressor +    │
                                     │  LogisticRegression classifier  │
                                     └───────────────┬─────────────────┘
                                                      ▼
                                     ┌───────────────────────────────┐
                                     │  src/ranking/ranker.py          │
                                     │  sorted + tie-broken leaderboard│
                                     └───────────────┬─────────────────┘
                                                      ▼
                         ┌─────────────┬──────────────────┬─────────────┐
                         ▼             ▼                  ▼
                  CLI (main.py)   REST API (api.py)   Streamlit UI (ui/app.py)
```

All three surfaces (CLI / API / UI) call the same `ResumeRankingPipeline`
in `src/pipeline.py`, so results are guaranteed identical across them.

---

## 📁 Project structure

```
resume-ranker/
├── config/
│   ├── config.yaml              # scoring weights, model config, paths
│   └── skills_taxonomy.json     # curated skills + alias dictionary
├── data/
│   ├── resumes/                 # sample resumes (pdf/docx/txt)
│   ├── job_descriptions/        # sample JD
│   └── processed/               # pipeline output (generated)
├── src/
│   ├── parser/                  # pdf_parser, docx_parser, resume_reader, text_cleaner
│   ├── features/                # skills / experience / education / contact extractors
│   ├── similarity/               # tfidf_matcher, embedding_matcher, hybrid_scorer
│   ├── ranking/                  # feature_scores, scorer (ML model), ranker
│   ├── utils/                    # logger, io helpers
│   ├── pipeline.py               # single orchestration point
│   ├── main.py                   # CLI entry point
│   └── api.py                    # FastAPI REST server
├── ui/
│   └── app.py                    # Streamlit "Screening Room" UI
├── tests/
│   ├── test_features.py
│   └── test_ranking.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting started

### Prerequisites
- Python 3.10+
- pip

### Install
```bash
git clone <this-repo-url>
cd resume-ranker
pip install -r requirements.txt
```

> `sentence-transformers` (semantic embeddings) is included in
> `requirements.txt` but optional at runtime — if it's not installed or a
> model can't be downloaded, the pipeline automatically falls back to
> TF-IDF-only similarity with no errors. See [Notes](#notes--tradeoffs).

### Run — CLI
```bash
python -m src.main --jd data/job_descriptions/sample_jd.txt --resumes-dir data/resumes
```
Ranked results are printed to the console and saved to
`data/processed/ranking_results.json`.

### Run — REST API
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive docs: `http://localhost:8000/docs`
`POST /rank` accepts multipart form data: `job_description` (text) +
`files` (one or more resume uploads).

### Run — Streamlit UI
```bash
streamlit run ui/app.py
```

### Run tests
```bash
python tests/test_features.py
python tests/test_ranking.py
```

---

## 🧠 Scoring methodology

Each candidate is scored on four sub-dimensions, combined by a trained
ML model (see `src/ranking/scorer.py`):

| Sub-score | Weight | How it's computed |
|---|---|---|
| Skills match | 40% | Coverage of JD-required skills found in the resume (taxonomy + alias matching) |
| Semantic similarity | 25% | Hybrid TF-IDF + sentence-embedding cosine similarity between resume and JD |
| Experience match | 20% | Candidate years vs. JD-stated requirement (or a saturating curve if unstated) |
| Education match | 15% | Candidate's highest degree vs. JD-stated minimum (or a gentle reward curve if unstated) |

Both a **heuristic weighted-sum score** and the **ML model's predicted
score** are reported side-by-side in every result, so the model's behavior
stays auditable rather than a black box.

---

## 🛠️ Tech stack

`Python` · `scikit-learn` (GradientBoostingRegressor, LogisticRegression)
· `PyMuPDF` · `python-docx` · `TF-IDF` (scikit-learn) ·
`sentence-transformers` (optional) · `FastAPI` · `Streamlit` · `pandas` /
`numpy`

---

## ⚖️ Notes & tradeoffs

- **Skill extraction** uses a curated taxonomy + alias dictionary
  (`config/skills_taxonomy.json`) rather than a black-box NER model — fully
  deterministic, explainable, and requires no model downloads. Tradeoff:
  it only recognizes skills phrased in ways present in the taxonomy.
- **Semantic similarity** is hybrid TF-IDF + embeddings when
  `sentence-transformers` is available, and gracefully degrades to
  TF-IDF-only otherwise — the pipeline never errors due to a missing
  optional model.
- **Experience extraction** is scoped to the resume's Experience section
  specifically, to avoid misreading education date ranges as work
  experience.
- **The ranking model** is trained via weak-supervision distillation from
  the heuristic weighted-sum scorer (no real hiring-outcome dataset was
  available for this assessment) — documented in-code in
  `src/ranking/scorer.py`.

---

## 🚧 Out of scope (per assessment brief)

Front-end/UI, external job-board integrations, and interview
scheduling/communication features were explicitly out of scope for
grading. The Streamlit UI included here is a bonus layer on top of a
fully self-contained backend pipeline, not a substitute for the graded
deliverables above.

---

## 📄 License

Built for academic/assessment purposes.
