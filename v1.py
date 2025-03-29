"""
There are too many nodes, too many connections and each node has very long sentences, sometimes even to the point that
they are meaningless.
"""

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
import spacy
import os

# Load spaCy model for sentence tokenization
nlp = spacy.load("en_core_web_sm")

# NLP model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

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

def sent_tokenize_spacy(text):
    """Tokenize text into sentences using spaCy."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Streamlit App
st.title("Personal Knowledge Management Tool")
st.markdown("Organize and connect content from online sources.")

# Input form
st.header("Add Content by URL")
with st.form("content_entry"):
    url = st.text_input("Enter the URL (video, article, or other)")
    submit = st.form_submit_button("Add Content")

if submit:
    if url:
        parsed_url = urlparse(url)
        content = ""

        if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
            content = extract_youtube_content(url)
        elif "http" in parsed_url.scheme:
            try:
                content = extract_article_content(url)
            except:
                content = extract_generic_content(url)

        # Add content to data
        new_entry = {"Source": url, "Content": content}
        new_row = pd.DataFrame([new_entry])
        data = pd.concat([data, new_row], ignore_index=True)
        data.to_csv("knowledge_data.csv", index=False)
        st.success("Content added successfully!")
    else:
        st.error("Please provide a valid URL.")

# Display entries
st.header("Saved Content")
if not data.empty:
    st.write(data)
else:
    st.info("No entries found. Add some content to get started!")

# Generate and display knowledge graph
st.header("Content Connections")
if st.button("Generate Connections"):
    G = nx.Graph()

    # Process content into sentences
    content_sentences = []
    sentence_to_source = {}
    for index, row in data.iterrows():
        sentences = sent_tokenize_spacy(row["Content"])
        content_sentences.extend(sentences)
        sentence_to_source.update({sentence: row["Source"] for sentence in sentences})

    # Create embeddings for sentences
    embeddings = model.encode(content_sentences)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Add nodes and edges across different sources
    for i, sentence_i in enumerate(content_sentences):
        G.add_node(sentence_i, label=sentence_i[:50] + "...", color="green")  # Use first 50 characters as label
        for j, sentence_j in enumerate(content_sentences):
            if i != j and sentence_to_source[sentence_i] != sentence_to_source[sentence_j]:  # Different sources
                if similarity_matrix[i][j] > 0.4:  # Lower threshold for better connections
                    G.add_edge(sentence_i, sentence_j)

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

# Instructions
st.markdown("---")
st.markdown("**Instructions:** Add content links and generate a graph to visualize relationships.")
