import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os
import nltk

# Load environment variables
load_dotenv()

EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

nltk.download('stopwords')

st.set_page_config(page_title="🔥 AI Resume Ranker", layout="wide", page_icon=":briefcase:")
st.title("🔥 AI Resume Ranker for HR")
st.markdown(
    """
Upload multiple resumes (PDF), enter job description, set score threshold, get ranked results.
Preview resumes, download, export to Excel, and optionally send emails to top candidates.
"""
)

if not os.path.exists("resumes"):
    os.makedirs("resumes")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
email_toggle = st.sidebar.checkbox("Enable Auto Email to Candidates", value=False)
score_threshold = st.sidebar.slider("Score Threshold (%)", 0, 100, 50)

uploaded_files = st.file_uploader("📁 Upload Resumes (PDF)", type=['pdf'], accept_multiple_files=True)
job_description = st.text_area("🧾 Paste Job Description")

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None

def send_email(recipient, score):
    msg = EmailMessage()
    msg['Subject'] = "🎉 Congratulations! You have been shortlisted"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipient
    msg.set_content(
        f"Hello,\n\nYou have been shortlisted for the job with a score of {score:.2f}%.\nWe will contact you soon.\n\nBest Regards,\nHR Team"
    )
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.warning(f"⚠️ Email to {recipient} failed: {e}")
        return False

def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        st.warning(f"Error reading PDF {file.name}: {e}")
    return text

if st.button("🚀 Rank Resumes"):
    if not uploaded_files or not job_description.strip():
        st.error("Please upload resumes and enter job description!")
    else:
        results = []
        with st.spinner("⏳ Processing Resumes..."):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                if not text.strip():
                    st.warning(f"⚠️ Could not extract text from {file.name}")
                    continue

                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform([text, job_description])
                score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

                email = extract_email(text) or "N/A"

                results.append({'file': file.name, 'score': score, 'email': email, 'text': text})

        filtered = [r for r in results if r['score'] >= score_threshold]
        filtered.sort(key=lambda x: x['score'], reverse=True)

        if filtered:
            df = pd.DataFrame(filtered)
            df_display = df[['file', 'score', 'email']].copy()
            df_display.rename(columns={
                'file': 'Resume File',
                'score': 'Score (%)',
                'email': 'Email'
            }, inplace=True)

            st.success(f"Found {len(filtered)} candidates above {score_threshold}% score")
            st.dataframe(df_display.style.background_gradient(cmap='Blues'))

            for r in filtered:
                with st.expander(f"📄 {r['file']} — Score: {r['score']:.2f}%"):
                    preview_text = r['text']
                    st.text_area("Resume Preview", preview_text, height=300)

            excel_file = "ranked_candidates.xlsx"
            df.to_excel(excel_file, index=False)
            with open(excel_file, "rb") as f:
                st.download_button("📥 Download Excel Report", f, file_name=excel_file)

            if email_toggle:
                st.info("Sending emails to shortlisted candidates...")
                for r in filtered:
                    if r['email'] != "N/A":
                        sent = send_email(r['email'], r['score'])
                        if sent:
                            st.success(f"Email sent to {r['file']} ({r['email']})")
                        else:
                            st.error(f"Failed to send email to {r['file']} ({r['email']})")
        else:
            st.warning("No candidates found above the score threshold.")
