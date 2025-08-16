
# 🧠 Resume Analyzer — Job Match Scoring (NLP)

An advanced Streamlit app that analyzes a **resume PDF** against a **Job Description (JD)** using NLP.
It computes a **match score**, highlights **matched/missing keywords**, and shows **visualizations** (word cloud, frequency bars).

## ✨ Features
- Upload resume (PDF)
- Paste job description text
- NLP preprocessing (stopwords removal, lemmatization)
- TF-IDF + Cosine Similarity for match score
- Keyword overlap (matched & missing JD terms)
- Visualizations: Word Cloud, Top Terms, Matched vs Missing counts
- Actionable suggestions

## 🧰 Tech Stack
- Python, Streamlit
- scikit-learn (TF-IDF, cosine)
- NLTK (preprocessing)
- PyPDF2 (PDF text extraction)
- Matplotlib, WordCloud (visuals)

## 🚀 Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## 📝 Notes
- The app downloads required **NLTK** data on first run (stopwords, wordnet, etc.).
- PDF extraction quality depends on your PDF. Exporting your resume as "text-based PDF" (not a scanned image) improves results.
- You can extend the project with:
  - DOCX parsing (python-docx)
  - Skill ontologies (e.g., predefined skill lists)
  - Section detection (Experience, Skills, Projects)
  - Model-based keyword extraction (KeyBERT, spaCy noun chunks)
  - Weighting of keywords by importance

## 🧪 How it works (Simplified)
1. Extract text from the resume PDF
2. Clean and lemmatize tokens
3. Compute TF-IDF vectors for resume & JD
4. Cosine similarity → match score (%)
5. Extract top keywords from each and compare overlaps

## 📁 Project Structure
```
resume-analyzer/
├── app.py
├── utils.py
├── requirements.txt
└── README.md
```

## 📌 Resume bullet ideas (to add in your CV)
- Built a Resume Analyzer web app using Streamlit enabling recruiters to compare resumes with JDs.
- Implemented TF‑IDF + cosine similarity to compute a job match score.
- Added keyword analysis, word‑cloud visualization, and frequency charts using Matplotlib.
- Used NLTK for preprocessing: tokenization, stopword removal, and lemmatization.
- Packaged app with clean modular code and documentation.

---

**Author:** Yours truly @ekanikaa
```
```
