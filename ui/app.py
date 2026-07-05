"""HireAI — Screening Room
Streamlit front-end for the Resume Screening & Ranking pipeline.

Design concept: the assessment's own vocabulary ("Resume Screening") is
treated literally as a cinema screening room — resumes are "reels" up for
review, the ranked shortlist is the "Callback List", and per-candidate
detail opens as a "Reel Report". Palette: near-black stage background,
gold marquee accents for the top score, emerald for matched skills,
crimson for gaps/missing skills — the same family used across this
person's prior CineAgent-style tools.

Run with:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import ResumeRankingPipeline  # noqa: E402
from src.ranking.ranker import rank_candidates  # noqa: E402

st.set_page_config(
    page_title="HireAI — Screening Room",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# THEME / TOKENS
# ----------------------------------------------------------------------------
# Background:  #0B0C0E (stage black)   Surface: #16181C   Surface-raised: #1E2126
# Gold:        #D4AF37 (marquee gold)  Red: #B32C33 (velvet crimson)
# Green:       #2E8B57 (emerald cue)   Text: #F2EEE3   Muted: #8B8D93
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --stage-black: #0B0C0E;
  --surface: #16181C;
  --surface-raised: #1E2126;
  --gold: #D4AF37;
  --gold-soft: #E8CD6E;
  --crimson: #B32C33;
  --crimson-soft: #E0575F;
  --emerald: #2E8B57;
  --emerald-soft: #57C486;
  --text: #F2EEE3;
  --muted: #8B8D93;
  --hairline: #2A2D33;
}

.stApp {
  background:
    radial-gradient(ellipse 1200px 500px at 50% -10%, rgba(212,175,55,0.08), transparent),
    var(--stage-black);
  color: var(--text);
  font-family: 'IBM Plex Sans', sans-serif;
}

section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--hairline);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, .marquee-title {
  font-family: 'Bebas Neue', sans-serif;
  letter-spacing: 0.04em;
}

.marquee {
  border: 1px solid var(--hairline);
  border-top: 3px solid var(--gold);
  border-bottom: 3px solid var(--gold);
  background: linear-gradient(180deg, rgba(212,175,55,0.07), transparent 60%);
  padding: 28px 32px 22px 32px;
  margin-bottom: 28px;
  position: relative;
}
.marquee-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  color: var(--gold);
  font-size: 12px;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.marquee-title {
  font-size: 46px;
  color: var(--text);
  line-height: 1;
  margin: 0;
}
.marquee-title span { color: var(--gold); }
.marquee-sub {
  color: var(--muted);
  font-size: 14.5px;
  margin-top: 8px;
  max-width: 720px;
}

.reel-card {
  background: var(--surface);
  border: 1px solid var(--hairline);
  border-left: 4px solid var(--hairline);
  border-radius: 2px;
  padding: 20px 22px;
  margin-bottom: 16px;
  position: relative;
  transition: border-left-color 0.2s ease;
}
.reel-card.rank-1 { border-left-color: var(--gold); }
.reel-card.rank-2 { border-left-color: var(--gold-soft); opacity: 0.96; }
.reel-card.rank-3 { border-left-color: var(--gold-soft); opacity: 0.92; }
.reel-card.rank-other { border-left-color: var(--hairline); }

.rank-badge {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 28px;
  color: var(--gold);
  display: inline-block;
  min-width: 46px;
}
.rank-badge.top { color: var(--gold); text-shadow: 0 0 14px rgba(212,175,55,0.5); }

.candidate-name {
  font-size: 18px;
  font-weight: 600;
  color: var(--text);
}
.candidate-meta {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  color: var(--muted);
  margin-top: 2px;
}

.score-track {
  width: 100%;
  height: 8px;
  background: var(--surface-raised);
  border-radius: 4px;
  overflow: hidden;
  margin-top: 8px;
}
.score-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--crimson) 0%, var(--gold) 55%, var(--emerald) 100%);
}
.score-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 22px;
  font-weight: 600;
  color: var(--gold);
}

.chip {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px;
  padding: 3px 9px;
  border-radius: 3px;
  margin: 3px 5px 3px 0;
  border: 1px solid transparent;
}
.chip-match { background: rgba(46,139,87,0.14); color: var(--emerald-soft); border-color: rgba(46,139,87,0.4); }
.chip-missing { background: rgba(179,44,51,0.13); color: var(--crimson-soft); border-color: rgba(179,44,51,0.4); }

.section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 14px 0 6px 0;
}

.callback-tag {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  padding: 3px 10px;
  border-radius: 3px;
  text-transform: uppercase;
}
.callback-yes { background: rgba(46,139,87,0.18); color: var(--emerald-soft); border: 1px solid rgba(46,139,87,0.5); }
.callback-no { background: rgba(179,44,51,0.13); color: var(--crimson-soft); border: 1px solid rgba(179,44,51,0.4); }

hr.filmstrip {
  border: none;
  height: 6px;
  margin: 26px 0;
  background-image: repeating-linear-gradient(90deg, var(--hairline) 0 10px, transparent 10px 20px);
}

.stButton > button {
  background: var(--gold) !important;
  color: var(--stage-black) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em;
  border: none !important;
  border-radius: 3px !important;
}
.stButton > button:hover { background: var(--gold-soft) !important; }

[data-testid="stFileUploaderDropzone"] {
  background: var(--surface-raised) !important;
  border: 1px dashed var(--hairline) !important;
}

.footer-credit {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  text-align: center;
  margin-top: 40px;
  letter-spacing: 0.06em;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> ResumeRankingPipeline:
    return ResumeRankingPipeline()


def render_header():
    st.markdown(
        """
        <div class="marquee">
          <div class="marquee-eyebrow">Now Screening &nbsp;·&nbsp; AI Product Engineer Assessment</div>
          <div class="marquee-title">HIRE<span>AI</span> &nbsp;SCREENING ROOM</div>
          <div class="marquee-sub">
            Upload a job description and a stack of resumes. Every reel gets parsed,
            scored against the role on skills / experience / education / semantic fit,
            and ranked into tonight's Callback List by a trained ML scoring model.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_candidate_card(c: dict, threshold: float):
    rank = c["rank"]
    rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
    badge_class = "top" if rank == 1 else ""
    score_pct = round(c["ml_score"] * 100, 1)
    callback = c["ml_score"] >= threshold
    callback_html = (
        '<span class="callback-tag callback-yes">Callback</span>'
        if callback else
        '<span class="callback-tag callback-no">No callback</span>'
    )

    matched = c["skills_match_detail"]["matched"]
    missing = c["skills_match_detail"]["missing"]
    matched_chips = "".join(f'<span class="chip chip-match">✓ {s}</span>' for s in matched[:14]) or (
        '<span class="candidate-meta">No overlapping skills detected</span>'
    )
    missing_chips = "".join(f'<span class="chip chip-missing">✕ {s}</span>' for s in missing[:14])

    st.markdown(
        f"""
        <div class="reel-card {rank_class}">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="display:flex; gap:16px; align-items:center;">
              <div class="rank-badge {badge_class}">#{rank:02d}</div>
              <div>
                <div class="candidate-name">{c['file_name']}</div>
                <div class="candidate-meta">
                  {c['education']['highest_degree'].upper()} &nbsp;·&nbsp;
                  {c['experience']['total_years']} yrs &nbsp;·&nbsp;
                  {c['experience']['seniority'].upper()} &nbsp;·&nbsp;
                  {c['quality_flag']}
                </div>
              </div>
            </div>
            <div style="text-align:right;">
              <div class="score-label">{score_pct}</div>
              {callback_html}
            </div>
          </div>
          <div class="score-track"><div class="score-fill" style="width:{score_pct}%;"></div></div>
          <div class="section-label">Matched skills</div>
          <div>{matched_chips}</div>
          {f'<div class="section-label">Gaps vs. job description</div><div>{missing_chips}</div>' if missing else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"Full reel report — {c['file_name']}"):
        col1, col2, col3 = st.columns(3)
        col1.metric("ML score", c["ml_score"])
        col2.metric("Heuristic score", c["heuristic_score"])
        col3.metric(
            "Shortlist probability",
            c["shortlist_probability"] if c["shortlist_probability"] is not None else "n/a",
        )

        st.markdown("**Sub-scores**")
        sc = c["subscores"]
        st.write(
            f"- Skills match: `{sc['skills_match']}`\n"
            f"- Semantic similarity: `{sc['semantic_similarity']}` "
            f"({c['similarity']['mode']})\n"
            f"- Experience match: `{sc['experience_match']}`\n"
            f"- Education match: `{sc['education_match']}`"
        )

        st.markdown("**Contact**")
        contact = c["contact"]
        st.write(
            f"- Email: {contact['email'] or '—'}\n"
            f"- Phone: {contact['phone'] or '—'}\n"
            f"- LinkedIn: {contact['linkedin'] or '—'}\n"
            f"- GitHub: {contact['github'] or '—'}"
        )


def main():
    render_header()

    with st.sidebar:
        st.markdown("### 🎬 Casting Brief")
        jd_input_mode = st.radio("Job description input", ["Paste text", "Upload .txt"], horizontal=False)
        jd_text = ""
        if jd_input_mode == "Paste text":
            jd_text = st.text_area("Job description", height=220, placeholder="Paste the role's job description here…")
        else:
            jd_file = st.file_uploader("Upload job description (.txt)", type=["txt"])
            if jd_file is not None:
                jd_text = jd_file.read().decode("utf-8", errors="ignore")

        st.markdown("---")
        st.markdown("### 🎞️ Submit Reels")
        resume_files = st.file_uploader(
            "Upload resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        st.markdown("---")
        threshold = st.slider("Callback threshold (ML score)", 0.0, 1.0, 0.55, 0.05)
        run_clicked = st.button("▶ Run Screening", use_container_width=True)

    if not run_clicked:
        st.info("Fill in the casting brief and submit reels in the sidebar, then hit **Run Screening**.")
        return

    if not jd_text.strip():
        st.error("A job description is required before screening can begin.")
        return
    if not resume_files:
        st.error("Upload at least one resume to screen.")
        return

    pipeline = get_pipeline()
    results, errors = [], []

    with st.spinner("Rolling the reels… parsing, scoring, ranking…"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for uploaded in resume_files:
                suffix = Path(uploaded.name).suffix.lower()
                if suffix not in {".pdf", ".docx", ".txt"}:
                    errors.append({"file_name": uploaded.name, "error": f"Unsupported format '{suffix}'"})
                    continue
                tmp_path = Path(tmp_dir) / uploaded.name
                tmp_path.write_bytes(uploaded.read())
                try:
                    results.append(pipeline.score_one(tmp_path, jd_text))
                except Exception as exc:  # noqa: BLE001
                    errors.append({"file_name": uploaded.name, "error": str(exc)})

        ranked = rank_candidates(results)

    st.markdown('<hr class="filmstrip"/>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="marquee-title" style="font-size:26px;">TONIGHT\'S CALLBACK LIST</div>',
        unsafe_allow_html=True,
    )
    callback_count = sum(1 for c in ranked if c["ml_score"] >= threshold)
    st.markdown(
        f'<div class="candidate-meta" style="margin-bottom:18px;">'
        f'{len(ranked)} reel(s) screened &nbsp;·&nbsp; {callback_count} above callback threshold '
        f'&nbsp;·&nbsp; {len(errors)} error(s)</div>',
        unsafe_allow_html=True,
    )

    for c in ranked:
        render_candidate_card(c, threshold)

    if errors:
        st.markdown('<div class="section-label">Skipped reels</div>', unsafe_allow_html=True)
        for e in errors:
            st.warning(f"{e['file_name']}: {e['error']}")

    st.markdown(
        '<div class="footer-credit">HIREAI SCREENING ROOM &nbsp;·&nbsp; '
        'built for the Rooman AI Product Engineer assessment</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
