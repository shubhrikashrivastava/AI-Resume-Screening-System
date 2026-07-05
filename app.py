import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
background:#050816;
color:white;
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.title{
font-size:52px;
font-weight:bold;
color:#00F5FF;
text-align:center;
text-shadow:0px 0px 20px cyan;
}

.subtitle{
text-align:center;
color:#ff00ff;
font-size:20px;
margin-bottom:30px;
}

.card{
background:#10172A;
padding:20px;
border-radius:15px;
border:2px solid cyan;
box-shadow:0px 0px 15px cyan;
}

.metric{
background:#111827;
padding:20px;
border-radius:15px;
text-align:center;
border:2px solid #00F5FF;
box-shadow:0px 0px 20px cyan;
}

.metric h2{
color:#00F5FF;
}

.stButton>button{
background:linear-gradient(90deg,#00F5FF,#FF00FF);
color:black;
font-size:18px;
font-weight:bold;
border:none;
border-radius:10px;
padding:10px 25px;
}

.stButton>button:hover{
transform:scale(1.05);
}

textarea{
background:#111827 !important;
color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown('<p class="title">🤖 AI Resume Screening System</p>', unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">Powered by NLP • TF-IDF • Cosine Similarity</p>',
unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚡ Dashboard")

st.sidebar.success("Future Interns - Task 3")

st.sidebar.write("### Features")

st.sidebar.write("✅ Resume Ranking")
st.sidebar.write("✅ ATS Match Score")
st.sidebar.write("✅ Skill Gap Analysis")
st.sidebar.write("✅ AI Candidate Screening")

# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("Resume/Resume.csv")
    return df

df = load_data()

# ---------------- TEXT CLEANING ----------------

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

# ---------------- DATA OVERVIEW ----------------

st.write("")

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class='metric'>
    <h2>{len(df)}</h2>
    <p>Total Resumes</p>
    </div>
    """,unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class='metric'>
    <h2>{df['Category'].nunique()}</h2>
    <p>Job Categories</p>
    </div>
    """,unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class='metric'>
    <h2>AI</h2>
    <p>Resume Screening</p>
    </div>
    """,unsafe_allow_html=True)

st.write("---")

# ---------------- JOB DESCRIPTION ----------------

st.header("📝 Enter Job Description")

job_description = st.text_area(
"Paste the Job Description",
height=220,
placeholder="Example:\nLooking for a Data Scientist with Python, SQL, Machine Learning, NLP and Deep Learning..."
)

screen = st.button("🚀 Screen Resumes")