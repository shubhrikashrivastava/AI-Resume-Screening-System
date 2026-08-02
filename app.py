"""
AI Resume Screening System
---------------------------
Upload real resumes (PDF / DOCX / TXT), paste a job description, and get
semantically-ranked candidates with an ATS-style match score and a skills
gap breakdown.

Run with:  streamlit run app.py
"""

import io
import re

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- OPTIONAL / GRACEFUL IMPORTS ----------------
# PDF and DOCX text extraction
try:
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    import docx  # python-docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# Semantic similarity (preferred). Falls back to TF-IDF if the model/
# package isn't installed, so the app still runs without a big download.
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide",
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp{background:#050816;color:white;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
.title{font-size:48px;font-weight:bold;color:#00F5FF;text-align:center;text-shadow:0px 0px 20px cyan;}
.subtitle{text-align:center;color:#ff00ff;font-size:18px;margin-bottom:25px;}
.card{background:#10172A;padding:20px;border-radius:15px;border:2px solid cyan;box-shadow:0px 0px 15px cyan;}
.metric{background:#111827;padding:20px;border-radius:15px;text-align:center;border:2px solid #00F5FF;box-shadow:0px 0px 20px cyan;}
.metric h2{color:#00F5FF;}
.stButton>button{background:linear-gradient(90deg,#00F5FF,#FF00FF);color:black;font-size:18px;font-weight:bold;border:none;border-radius:10px;padding:10px 25px;}
.stButton>button:hover{transform:scale(1.05);}
textarea{background:#111827 !important;color:white !important;}
.skill-tag{display:inline-block;background:#0f766e;color:white;padding:3px 10px;border-radius:12px;margin:2px;font-size:13px;}
.skill-tag-missing{display:inline-block;background:#7f1d1d;color:white;padding:3px 10px;border-radius:12px;margin:2px;font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="title">🤖 AI Resume Screening System</p>', unsafe_allow_html=True)
engine_label = "Sentence-Embeddings" if HAS_EMBEDDINGS else "TF-IDF"
st.markdown(
    f'<p class="subtitle">Powered by NLP • {engine_label} • Cosine Similarity</p>',
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚡ Dashboard")
st.sidebar.write("### Features")
st.sidebar.write("✅ Upload real resumes (PDF/DOCX/TXT)")
st.sidebar.write("✅ Semantic ATS Match Score")
st.sidebar.write("✅ Dynamic Skill Gap Analysis")
st.sidebar.write("✅ Explainable, keyword-highlighted results")

if not HAS_EMBEDDINGS:
    st.sidebar.warning(
        "sentence-transformers isn't installed, so matching is falling back "
        "to TF-IDF. Run `pip install sentence-transformers` for stronger, "
        "meaning-based matching."
    )

# ---------------- STOPWORDS (no NLTK download needed) ----------------
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no
nor not now of off on once only or other our ours ourselves out over own same
she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what
when where which while who whom why will with you your yours yourself
yourselves
""".split())

# ---------------- SKILLS TAXONOMY ----------------
# A broader, extensible list. In production, swap this for a maintained
# taxonomy (e.g. ESCO, LinkedIn Skills API) or an NER model.
SKILLS_TAXONOMY = [
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql",
    "machine learning", "deep learning", "nlp", "computer vision",
    "data science", "data analysis", "data engineering", "statistics",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "power bi", "tableau", "excel", "looker",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "flask", "django", "fastapi", "react", "angular", "vue", "node.js",
    "html", "css", "rest api", "graphql", "git", "linux",
    "agile", "scrum", "project management", "communication", "leadership",
    "spark", "hadoop", "airflow", "etl", "kafka",
]


def extract_skills(text: str) -> set:
    text_lower = text.lower()
    return {skill for skill in SKILLS_TAXONOMY if skill in text_lower}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


# ---------------- FILE TEXT EXTRACTION ----------------
def extract_text_from_pdf(file) -> str:
    if not HAS_PDF:
        return ""
    text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(file) -> str:
    if not HAS_DOCX:
        return ""
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    buffer = io.BytesIO(data)
    if name.endswith(".pdf"):
        return extract_text_from_pdf(buffer)
    if name.endswith(".docx"):
        return extract_text_from_docx(buffer)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    return ""


# ---------------- SAMPLE DATASET (optional fallback) ----------------
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv("Resume/Resume.csv")
        return df[["ID", "Category", "Resume_str"]].rename(
            columns={"ID": "Candidate", "Resume_str": "Resume Text"}
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["Candidate", "Category", "Resume Text"])


# ---------------- EMBEDDING MODEL ----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def score_with_embeddings(resume_texts, jd_text):
    model = load_embedding_model()
    resume_embeddings = model.encode(resume_texts, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    scores = st_util.cos_sim(jd_embedding, resume_embeddings).cpu().numpy().flatten()
    return scores


def score_with_tfidf(resume_texts, jd_text):
    cleaned = [clean_text(t) for t in resume_texts] + [clean_text(jd_text)]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(cleaned)
    scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    return scores


# ---------------- SOURCE SELECTION ----------------
st.header("📥 Candidate Resumes")
source = st.radio(
    "Choose a resume source",
    ["Upload resumes", "Use sample dataset"],
    horizontal=True,
)

candidates = []  # list of dicts: Candidate, Category, Resume Text

if source == "Upload resumes":
    if not (HAS_PDF or HAS_DOCX):
        st.info(
            "For PDF/DOCX support, install `pdfplumber` and `python-docx` "
            "(see requirements.txt). Plain .txt files work regardless."
        )
    uploaded_files = st.file_uploader(
        "Upload one or more resumes (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            text = extract_text(f)
            if text.strip():
                candidates.append({
                    "Candidate": f.name,
                    "Category": "Uploaded",
                    "Resume Text": text,
                })
            else:
                st.warning(f"Couldn't extract text from **{f.name}** — skipping.")
else:
    sample_df = load_sample_data()
    if sample_df.empty:
        st.error("Sample dataset not found at Resume/Resume.csv.")
    else:
        n = st.slider("Number of sample resumes to screen", 10, min(500, len(sample_df)), 50)
        candidates = sample_df.sample(n, random_state=42).to_dict("records")

st.write("---")

# ---------------- DATA OVERVIEW ----------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric'><h2>{len(candidates)}</h2><p>Resumes Loaded</p></div>", unsafe_allow_html=True)
with c2:
    n_categories = len({c["Category"] for c in candidates}) if candidates else 0
    st.markdown(f"<div class='metric'><h2>{n_categories}</h2><p>Categories</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric'><h2>{engine_label}</h2><p>Matching Engine</p></div>", unsafe_allow_html=True)

st.write("---")

# ---------------- JOB DESCRIPTION ----------------
st.header("📝 Enter Job Description")
job_description = st.text_area(
    "Paste the Job Description",
    height=200,
    placeholder="Example:\nLooking for a Data Scientist with Python, SQL, Machine Learning, NLP and Deep Learning...",
)

screen = st.button("🚀 Screen Resumes")

if screen:
    if job_description.strip() == "":
        st.warning("Please enter a job description.")
        st.stop()
    if not candidates:
        st.warning("Please upload at least one resume or load the sample dataset.")
        st.stop()

    resume_texts = [c["Resume Text"] for c in candidates]
    jd_skills = extract_skills(job_description)

    with st.spinner(f"Scoring {len(candidates)} resumes with {engine_label}..."):
        if HAS_EMBEDDINGS:
            scores = score_with_embeddings(resume_texts, job_description)
        else:
            scores = score_with_tfidf(resume_texts, job_description)

    results = pd.DataFrame(candidates)
    results["ATS Score"] = (scores * 100).round(2)

    matched_list, missing_list = [], []
    for text in resume_texts:
        resume_skills = extract_skills(text)
        matched_list.append(", ".join(sorted(jd_skills & resume_skills)) or "None")
        missing_list.append(", ".join(sorted(jd_skills - resume_skills)) or "None")
    results["Matched Skills"] = matched_list
    results["Missing Skills"] = missing_list

    ranked = results.sort_values("ATS Score", ascending=False).reset_index(drop=True)

    st.success("Resume screening completed successfully!")
    st.subheader("🏆 Ranked Candidates")

    if jd_skills:
        st.caption("Skills detected in the job description: " + ", ".join(sorted(jd_skills)))
    else:
        st.caption("No skills from the built-in taxonomy were detected in the job description — "
                   "ranking is based on overall text similarity only.")

    st.dataframe(
        ranked[["Candidate", "Category", "ATS Score", "Matched Skills", "Missing Skills"]].head(20),
        use_container_width=True,
    )

    st.bar_chart(ranked.head(10).set_index("Candidate")["ATS Score"])

    # Per-candidate detail view
    with st.expander("🔍 Inspect a candidate"):
        pick = st.selectbox("Choose a candidate", ranked["Candidate"].head(20))
        row = ranked[ranked["Candidate"] == pick].iloc[0]
        st.markdown(f"**ATS Score:** {row['ATS Score']}%")
        matched_html = "".join(f"<span class='skill-tag'>{s}</span>" for s in row["Matched Skills"].split(", ") if s != "None")
        missing_html = "".join(f"<span class='skill-tag-missing'>{s}</span>" for s in row["Missing Skills"].split(", ") if s != "None")
        st.markdown("Matched: " + (matched_html or "—"), unsafe_allow_html=True)
        st.markdown("Missing: " + (missing_html or "—"), unsafe_allow_html=True)

    st.download_button(
        "📥 Download Ranking",
        ranked.to_csv(index=False),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("📈 Business Insights")
    st.write("""
- Candidates are ranked by semantic similarity to the job description, not just keyword overlap.
- Matched/Missing skills show exactly why a candidate scored the way they did.
- Recruiters can inspect any candidate's detail before shortlisting.
- Results are for triage only — always have a human review before rejecting a candidate.
""")