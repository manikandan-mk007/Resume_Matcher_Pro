import io
import os
import re
import tempfile
from collections import Counter
from typing import Dict, List, Tuple

import docx2txt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Resume Matcher Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.25rem;
    }

    .sub-text {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #111827;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #111827;
    }

    .candidate-box {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        background: #ffffff;
        margin-bottom: 1rem;
    }

    .skill-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        border-radius: 999px;
        background: #f3f4f6;
        color: #374151;
        font-size: 0.82rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
    }

    div.stButton > button {
        border-radius: 10px;
        height: 2.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


model = load_model()


SKILL_KEYWORDS = [
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "react", "angular", "vue", "html", "css", "bootstrap", "tailwind",
    "node", "express", "django", "flask", "fastapi",
    "sql", "mysql", "postgresql", "mongodb", "sqlite",
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "tensorflow", "pytorch", "machine learning", "deep learning",
    "nlp", "data science", "data analysis",
    "git", "github", "docker", "kubernetes", "aws", "azure",
    "rest api", "api", "oop", "linux", "streamlit"
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s@.+#/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return " ".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        text = docx2txt.process(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return text or ""


def extract_resume_text(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".pdf"):
        return clean_text(extract_text_from_pdf(file_bytes))
    if filename.endswith(".docx"):
        return clean_text(extract_text_from_docx(file_bytes))
    return ""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += max(chunk_size - overlap, 1)

    return chunks


def average_embeddings(chunks: List[str], embedder: SentenceTransformer) -> np.ndarray:
    if not chunks:
        dim = embedder.get_sentence_embedding_dimension()
        return np.zeros(dim)

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return np.mean(embeddings, axis=0)


def normalize_score(score: float) -> float:
    score_0_1 = (score + 1) / 2
    return round(score_0_1 * 100, 2)


def extract_email(text: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else "Not found"


def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s\-]{8,15}\d)", text)
    return match.group(0) if match else "Not found"


def classify_resume(text: str) -> str:
    text_lower = text.lower()

    categories = {
        "Data Science": [
            "machine learning", "data science", "python", "pandas", "numpy",
            "tensorflow", "pytorch", "deep learning", "statistics", "scikit-learn"
        ],
        "Web Development": [
            "html", "css", "javascript", "react", "node", "django",
            "flask", "frontend", "backend", "full stack"
        ],
        "Mobile Development": [
            "android", "ios", "flutter", "react native", "kotlin", "swift"
        ],
        "Cloud / DevOps": [
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
            "devops", "linux", "terraform", "ci/cd"
        ],
        "Testing / QA": [
            "testing", "selenium", "qa", "test cases", "automation testing", "manual testing"
        ]
    }

    scores = {}
    for category, keywords in categories.items():
        scores[category] = sum(1 for keyword in keywords if keyword in text_lower)

    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "General / Other"


def extract_skills(text: str) -> List[str]:
    text_lower = text.lower()
    found_skills = []

    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return sorted(set(found_skills))


def explain_match(resume_skills: List[str], jd_skills: List[str], score: float) -> str:
    common_skills = sorted(set(resume_skills).intersection(set(jd_skills)))

    if common_skills:
        top_common = ", ".join(common_skills[:8])
        return (
            f"This resume matched because it shares relevant skills such as "
            f"{top_common}. The overall similarity score is {score}%."
        )

    return (
        f"This resume has a similarity score of {score}%, but strong direct keyword overlap "
        f"was not detected from the current skill list."
    )


def process_resumes(uploaded_files, job_description: str) -> Tuple[pd.DataFrame, List[Dict], Counter]:
    job_description = clean_text(job_description)
    jd_chunks = chunk_text(job_description)
    jd_embedding = average_embeddings(jd_chunks, model)
    jd_skills = extract_skills(job_description)

    results = []
    full_data = []
    all_skill_counter = Counter()

    for file in uploaded_files:
        resume_text = extract_resume_text(file)

        if not resume_text:
            continue

        resume_chunks = chunk_text(resume_text)
        resume_embedding = average_embeddings(resume_chunks, model)

        similarity = cosine_similarity(
            jd_embedding.reshape(1, -1),
            resume_embedding.reshape(1, -1)
        )[0][0]

        score = normalize_score(float(similarity))
        category = classify_resume(resume_text)
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        resume_skills = extract_skills(resume_text)
        explanation = explain_match(resume_skills, jd_skills, score)

        all_skill_counter.update(resume_skills)

        results.append({
            "Resume Name": file.name,
            "Category": category,
            "Match Score (%)": score,
            "Email": email,
            "Phone": phone,
            "Skills": ", ".join(resume_skills[:8]) if resume_skills else "None"
        })

        full_data.append({
            "filename": file.name,
            "category": category,
            "score": score,
            "email": email,
            "phone": phone,
            "skills": resume_skills,
            "explanation": explanation,
            "text": resume_text
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

    full_data = sorted(full_data, key=lambda x: x["score"], reverse=True)
    return df, full_data, all_skill_counter


def plot_top_skills(skill_counter: Counter, top_n: int = 8):
    if not skill_counter:
        return None

    top_items = skill_counter.most_common(top_n)
    skills = [x[0].upper() for x in top_items]
    counts = [x[1] for x in top_items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(skills[::-1], counts[::-1])
    ax.set_title("Top Skills", fontsize=13, pad=12)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center")

    plt.tight_layout()
    return fig


def plot_resume_scores(full_data: List[Dict]):
    if not full_data:
        return None

    names = [
        item["filename"].replace(".pdf", "").replace(".docx", "")
        for item in full_data[:10]
    ]
    scores = [item["score"] for item in full_data[:10]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, scores)
    ax.set_title("Resume Match Scores", fontsize=13, pad=12)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=25, ha="right")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 1,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    return fig


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_skill_pills(skills: List[str]):
    if not skills:
        st.write("No skills detected")
        return

    pills = "".join(
        [f'<span class="skill-pill">{skill}</span>' for skill in skills[:12]]
    )
    st.markdown(pills, unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Settings")
    st.caption("Model: all-MiniLM-L6-v2")
    st.caption("Formats: PDF, DOCX")
    st.caption("Method: Embeddings + cosine similarity")


st.markdown('<div class="main-title">Resume Matcher Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Semantic resume ranking, skill detection, and recruiter-friendly candidate insights.</div>',
    unsafe_allow_html=True
)


left_col, right_col = st.columns([1.25, 1])

with left_col:
    st.markdown('<div class="section-title">Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Enter job description",
        height=260,
        label_visibility="collapsed",
        placeholder=(
            "We are looking for a Python Full Stack Developer with experience in "
            "Django, React, REST API, MySQL, Git, and deployment."
        )
    )

with right_col:
    st.markdown('<div class="section-title">Resume Upload</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

run_button = st.button("Match Resumes", use_container_width=True)


if run_button:
    if not job_description.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("Analyzing resumes and generating rankings..."):
            result_df, full_data, skill_counter = process_resumes(uploaded_files, job_description)

        if result_df.empty:
            st.error("No readable text found in the uploaded resumes.")
        else:
            st.success("Resume matching completed successfully.")

            total_resumes = len(full_data)
            top_score = f"{full_data[0]['score']}%" if full_data else "0%"
            total_skills = len(skill_counter.keys())

            st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)

            with m1:
                metric_card("Total Resumes", str(total_resumes))
            with m2:
                metric_card("Top Match Score", top_score)
            with m3:
                metric_card("Unique Skills Found", str(total_skills))

            st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="resume_matching_results.csv",
                mime="text/csv"
            )

            st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                skill_fig = plot_top_skills(skill_counter)
                if skill_fig:
                    st.pyplot(skill_fig, use_container_width=True)

            with col2:
                score_fig = plot_resume_scores(full_data)
                if score_fig:
                    st.pyplot(score_fig, use_container_width=True)

            st.markdown('<div class="section-title">Top Candidates</div>', unsafe_allow_html=True)

            top_n = min(5, len(full_data))
            for i in range(top_n):
                item = full_data[i]

                st.markdown(
                    f"""
                    <div class="candidate-box">
                        <div style="font-size:1.05rem; font-weight:600; color:#111827;">
                            {i + 1}. {item['filename']}
                        </div>
                        <div style="color:#6b7280; font-size:0.9rem; margin-top:0.2rem;">
                            {item['category']} | Score: {item['score']}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns([1, 1])

                with c1:
                    st.write(f"**Email:** {item['email']}")
                    st.write(f"**Phone:** {item['phone']}")
                    st.write("**Skills:**")
                    render_skill_pills(item["skills"])

                with c2:
                    st.write("**Match Explanation:**")
                    st.write(item["explanation"])

                with st.expander("Resume Preview"):
                    st.text_area(
                        "Resume Preview",
                        item["text"][:2500],
                        height=220,
                        key=f"preview_{i}",
                        label_visibility="collapsed"
                    )