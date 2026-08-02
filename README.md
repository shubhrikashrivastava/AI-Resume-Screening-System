# 🤖 AI Resume Screening System

An AI-powered Resume Screening System developed using **Python, Streamlit, Natural Language Processing (NLP), and Machine Learning**. The application compares resumes with a job description, calculates an ATS-style match score, ranks candidates, and highlights missing skills to assist recruiters in shortlisting the best applicants.

---

## 📌 Features

- 📄 Resume text preprocessing
- 🤖 AI-powered candidate screening
- 📊 TF-IDF feature extraction
- 🎯 Cosine Similarity based ATS Match Score
- 🏆 Candidate ranking
- 🔍 Missing skill analysis
- 📈 Interactive visualizations
- 🌌 Dark Neon Streamlit UI
- 💾 Download ranked candidates as CSV

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib

---

## 📂 Dataset

**Dataset:** Resume Dataset

Source: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Dataset contains:
- Resume Text
- Resume Category
- Resume HTML
- Resume ID

---

## 📁 Project Structure

```
FUTURE_ML_03/
│
├── app.py
├── requirements.txt
├── README.md
│
└── Resume/
    └── Resume.csv
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/FUTURE_ML_03.git
```

Move to the project folder:

```bash
cd FUTURE_ML_03
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📊 How It Works

1. Load the resume dataset.
2. Clean and preprocess resume text.
3. Enter a job description.
4. Convert text into TF-IDF vectors.
5. Calculate Cosine Similarity between resumes and the job description.
6. Rank candidates based on their ATS Match Score.
7. Display the top matching candidates along with missing skills and insights.

---

## 💼 Business Use Cases

- Automates resume screening
- Reduces manual hiring effort
- Helps recruiters identify top candidates
- Improves recruitment efficiency
- Supports faster hiring decisions

---

## 👩‍💻 Author

**Shubhrika Shrivastava**


