import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from youtube_transcript_api import YouTubeTranscriptApi
from newspaper import Article
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from moviepy.editor import AudioFileClip
from speech_recognition import Recognizer, AudioFile
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document
import os

# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

# Initialize data storage
if "knowledge_data.csv" not in os.listdir():
    pd.DataFrame(columns=["Source", "Content"]).to_csv("knowledge_data.csv", index=False)

# Load existing data
data = pd.read_csv("knowledge_data.csv")

# Helper functions
def extract_youtube_content(url):
    """Extract transcript content from a YouTube video."""
    video_id = urlparse(url).query.split("v=")[-1]
    transcript = ""
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([entry["text"] for entry in transcript_data])
    except Exception as e:
        transcript = f"Transcript not available: {e}"
    return transcript

def extract_article_content(url):
    """Extract text content from an article."""
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def extract_generic_content(url):
    """Scrape visible text from a generic webpage."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    return text

def extract_text_file_content(file):
    """Extract content from a text file."""
    return file.read().decode("utf-8")

def extract_pdf_content(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()
    return content

def extract_word_content(file):
    """Extract text from a Word document."""
    doc = Document(file)
    content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return content

def extract_excel_content(file):
    """Extract content from an Excel file."""
    df = pd.read_excel(file)
    return df.to_string()

def transcribe_audio(file_path):
    """Transcribe audio file content."""
    recognizer = Recognizer()
    with AudioFile(file_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

def extract_audio_from_video(file_path):
    """Extract audio from video and transcribe it."""
    audio_path = "temp_audio.wav"
    video = AudioFileClip(file_path)
    video.audio.write_audiofile(audio_path)
    transcript = transcribe_audio(audio_path)
    os.remove(audio_path)  # Clean up temporary file
    return transcript

# Streamlit App
st.title("Personal Knowledge Management Tool")
st.markdown("Organize and connect content from online sources or uploaded files.")

# Input options
st.header("Add Content")
input_type = st.radio(
    "Choose input method:",
    ("Enter URL", "Upload File", "Enter Text")
)

content = ""
source = ""

if input_type == "Enter URL":
    url = st.text_input("Enter the URL (video, article, or other)")
    if st.button("Add Content from URL"):
        if url:
            parsed_url = urlparse(url)
            source = url
            if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
                content = extract_youtube_content(url)
            elif "http" in parsed_url.scheme:
                try:
                    content = extract_article_content(url)
                except:
                    content = extract_generic_content(url)
            st.success("Content added successfully!")
        else:
            st.error("Please provide a valid URL.")

elif input_type == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["txt", "pdf", "docx", "xlsx", "mp3", "wav", "mp4"]
    )
    if uploaded_file:
        file_type = uploaded_file.type
        source = uploaded_file.name
        if "text/plain" in file_type:
            content = extract_text_file_content(uploaded_file)
        elif "pdf" in file_type:
            content = extract_pdf_content(uploaded_file)
        elif "wordprocessingml" in file_type:
            content = extract_word_content(uploaded_file)
        elif "spreadsheetml" in file_type:
            content = extract_excel_content(uploaded_file)
        elif "audio" in file_type or "video" in file_type:
            with open("temp_file", "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            if "audio" in file_type:
                content = transcribe_audio("temp_file")
            elif "video" in file_type:
                content = extract_audio_from_video("temp_file")
            os.remove("temp_file")
        st.success("Content added successfully!")

elif input_type == "Enter Text":
    content = st.text_area("Enter text")
    if st.button("Add Content from Text"):
        source = "User Input"
        st.success("Content added successfully!")

# Save content if available
if content:
    new_entry = {"Source": source, "Content": content}
    new_row = pd.DataFrame([new_entry])
    data = pd.concat([data, new_row], ignore_index=True)
    data.to_csv("knowledge_data.csv", index=False)

# Display saved content
st.header("Saved Content")
if not data.empty:
    st.write(data)
else:
    st.info("No entries found. Add some content to get started!")
