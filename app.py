
import io
import streamlit as st
from typing import List, Tuple
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from utils import (
    extract_text_from_pdf,
    clean_text,
    get_keywords,
    compute_match_score,
    compare_keywords,
    top_n_terms,
)


st.set_page_config(page_title="Resume Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 Resume Analyzer — Job Match Scoring (NLP)")

with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        **How to use:**
        1. Upload your resume (PDF).
        2. Paste the Job Description (JD).
        3. Click **Analyze**.
        4. View your **Match Score**, **Matched/Missing Skills**, and **Visualizations**.
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit, scikit-learn, NLTK, Matplotlib, WordCloud")

col1, col2 = st.columns([1, 1])

with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

with col2:
    jd_text_input = st.text_area("📝 Paste Job Description (JD) here", height=250, placeholder="Paste the JD...")

analyze = st.button("🔎 Analyze")

if analyze:
    if not resume_file or not jd_text_input.strip():
        st.error("Please upload a PDF resume and paste a job description.")
    else:
        # Extract and clean
        raw_resume = extract_text_from_pdf(resume_file)
        if not raw_resume.strip():
            st.error("Couldn't extract text from the PDF. Please try another file.")
        else:
            cleaned_resume = clean_text(raw_resume)
            cleaned_jd = clean_text(jd_text_input)

            # Compute score
            score = compute_match_score(cleaned_resume, cleaned_jd)
            pct = round(score * 100, 2)

            st.markdown(f"## 📊 Job Match Score: **{pct}%**")

            # Keywords
            resume_kw = get_keywords(cleaned_resume, top_k=80)
            jd_kw = get_keywords(cleaned_jd, top_k=80)
            matched, missing = compare_keywords(resume_kw, jd_kw)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("✅ Matched Keywords")
                if matched:
                    st.write(", ".join(matched))
                else:
                    st.write("_No overlaps found. Consider adding relevant skills from the JD if you have them._")

            with c2:
                st.subheader("❗ Missing Keywords (from JD)")
                if missing:
                    st.write(", ".join(missing))
                else:
                    st.write("_Great! Your resume covers most JD keywords._")

            st.markdown("---")

            # Visualizations
            st.subheader("📈 Visualizations")

            # Word Cloud of Resume
            st.markdown("**Resume Word Cloud**")
            wc = WordCloud(width=900, height=500, background_color="white").generate(cleaned_resume or "resume")
            fig_wc, ax_wc = plt.subplots(figsize=(9, 4.5))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            # Bar Chart: Top Resume Terms
            st.markdown("**Top Resume Terms (Frequency)**")
            top_terms = top_n_terms(raw_resume, n=15)
            if top_terms:
                terms, counts = zip(*top_terms)
                fig_bar, ax_bar = plt.subplots(figsize=(9, 4.5))
                ax_bar.bar(terms, counts)
                ax_bar.set_xlabel("Terms")
                ax_bar.set_ylabel("Frequency")
                ax_bar.set_title("Most Frequent Terms in Resume")
                plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")
                st.pyplot(fig_bar)
            else:
                st.info("Not enough text to display term frequency.")

            # Bar Chart: Matched vs Missing count
            st.markdown("**Matched vs Missing Keywords**")
            fig_mm, ax_mm = plt.subplots(figsize=(6, 4))
            ax_mm.bar(["Matched", "Missing"], [len(matched), len(missing)])
            ax_mm.set_ylabel("Count")
            ax_mm.set_title("JD Keywords Coverage")
            st.pyplot(fig_mm)

            # Tips section
            st.markdown("---")
            st.subheader("💡 Suggestions")
            tips = []
            if pct < 60:
                tips.append("Tailor your resume summary to include 3–5 key JD keywords.")
            if missing:
                tips.append("Integrate missing skills naturally into your experience bullets (only if you truly have them).")
            if "python" in missing:
                tips.append("If you use Python, add specific libraries (e.g., Pandas, NumPy, scikit-learn) and measurable outcomes.")
            if "sql" in missing:
                tips.append("Mention SQL queries, joins, window functions, and performance tuning if applicable.")
            if not tips:
                tips.append("Your resume aligns well with the JD. Consider adding quantifiable metrics to strengthen impact.")
            st.write("\n\n".join([f"- {t}" for t in tips]))
