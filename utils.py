
import re
import string
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# PDF
from PyPDF2 import PdfReader


def _ensure_nltk():
    """
    Ensure required NLTK resources are available.
    """
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


_ensure_nltk()
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def extract_text_from_pdf(file) -> str:
    """
    Extracts text from an uploaded PDF file-like object using PyPDF2.
    """
    try:
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            content = page.extract_text() or ""
            texts.append(content)
        return "\n".join(texts)
    except Exception as e:
        return f""


def clean_text(text: str) -> str:
    """
    Basic cleaning: lowercase, remove punctuation/numbers, lemmatize, remove stopwords.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = nltk.word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t.isalpha() and t not in STOPWORDS]
    return " ".join(tokens)


def get_keywords(text: str, top_k: int = 50) -> List[str]:
    """
    Very simple keyword extraction using TF-IDF on the single document:
    Returns top_k unigrams/bigrams by TF-IDF score.
    """
    if not text:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform([text])
    # single document -> tfidf scores are just feature weights
    scores = X.toarray().ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    idx = np.argsort(scores)[::-1]
    top = terms[idx][:top_k]
    return top.tolist()


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute cosine similarity between TF-IDF vectors of resume and job description.
    """
    if not resume_text or not jd_text:
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform([resume_text, jd_text])
    sim = cosine_similarity(X[0:1], X[1:2])[0][0]
    return float(sim)


def compare_keywords(resume_kw: List[str], jd_kw: List[str]) -> Tuple[List[str], List[str]]:
    """
    Return matched and missing JD keywords relative to resume keywords.
    """
    rset = set(resume_kw)
    jset = set(jd_kw)
    matched = sorted(list(rset.intersection(jset)))
    missing = sorted(list(jset.difference(rset)))
    return matched, missing


def top_n_terms(text: str, n: int = 15) -> List[Tuple[str, int]]:
    """
    Returns the top-n most frequent lemmatized tokens from text.
    """
    if not text:
        return []
    tokens = clean_text(text).split()
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:n]
