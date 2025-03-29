import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from youtube_transcript_api import YouTubeTranscriptApi
from newspaper import Article
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
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

def extract_key_phrases(content, top_n=10):
    """Extract key phrases using KeyBERT."""
    keywords = keybert_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]

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
        type=["txt", "pdf", "docx"]
    )
    if uploaded_file:
        file_type = uploaded_file.type
        source = uploaded_file.name
        if "text/plain" in file_type:
            content = uploaded_file.read().decode("utf-8")
        elif "pdf" in file_type:
            content = extract_pdf_content(uploaded_file)
        elif "wordprocessingml" in file_type:
            content = extract_word_content(uploaded_file)
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

# Generate and display knowledge graph
st.header("Content Connections")
if st.button("Generate Connections"):
    G = nx.Graph()

    # Extract key phrases from each content
    key_phrases = []
    phrase_to_source = {}
    for index, row in data.iterrows():
        phrases = extract_key_phrases(row["Content"], top_n=10)
        key_phrases.extend(phrases)
        phrase_to_source.update({phrase: row["Source"] for phrase in phrases})

    # Create embeddings for key phrases
    embeddings = model.encode(key_phrases)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Add nodes and filtered edges
    for i, phrase_i in enumerate(key_phrases):
        G.add_node(phrase_i, label=phrase_i, color="green")
        similarity_scores = [
            (key_phrases[j], similarity_matrix[i][j])
            for j in range(len(key_phrases))
            if i != j and phrase_to_source[phrase_i] != phrase_to_source[key_phrases[j]]
        ]
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]
        for phrase_j, score in sorted_scores:
            if score > 0.4:
                G.add_edge(phrase_i, phrase_j)

    # Visualize graph
    if len(G.nodes) > 0:
        net = Network(height="700px", width="100%")
        net.from_nx(G)
        net.write_html("knowledge_graph.html")

        try:
            with open("knowledge_graph.html", "r") as f:
                st.components.v1.html(f.read(), height=700)
        except FileNotFoundError:
            st.error("Unable to render the graph. The HTML file was not created.")
    else:
        st.error("The graph is empty. Add more content to create connections.")
