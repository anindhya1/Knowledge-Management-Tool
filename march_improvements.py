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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import PyPDF2
from docx import Document
import os

# Load local LLM (Mistral-7B-Instruct)
@st.cache_resource
def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_local_model()

# NLP models
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

# Initialize data storage
if "knowledge_data.csv" not in os.listdir():
    pd.DataFrame(columns=["Source", "Content"]).to_csv("knowledge_data.csv", index=False)

# Load existing data
data = pd.read_csv("knowledge_data.csv")

# Custom CSS for theming
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff;
            color: #333333;
        }
        .stSidebar {
            background-color: rgba(0, 128, 128, 0.2);
        }
        .stButton>button {
            background-color: #008080;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #005757;
            transform: scale(1.05);
            transition: all 0.2s ease-in-out;
        }
        .stSidebar .stButton {
            margin: 10px 0;
            width: 90%;
        }
        .stSidebar .stButton>button {
            background-color: #008080;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stSidebar .stButton>button:hover {
            background-color: #005757;
            transform: scale(1.05);
            transition: all 0.2s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Add CSS to the app
add_custom_css()

# --- Extraction Helpers ---
def extract_youtube_transcript(url):
    video_id = url.split("v=")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([segment['text'] for segment in transcript])

def extract_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def extract_file_text(uploaded_file):
    ext = uploaded_file.name.split('.')[-1]
    if ext == "txt":
        return uploaded_file.read().decode("utf-8")
    elif ext == "pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif ext == "docx":
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

# --- Knowledge Graph & Insights ---
def extract_key_phrases(content, top_n=10):
    keywords = keybert_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]

def generate_knowledge_graph(data):
    G = nx.Graph()
    key_phrases = []
    phrase_to_source = {}

    for _, row in data.iterrows():
        phrases = extract_key_phrases(row["Content"], top_n=10)
        key_phrases.extend(phrases)
        phrase_to_source.update({phrase: row["Source"] for phrase in phrases})

    embeddings = model_embed.encode(key_phrases)
    similarity_matrix = cosine_similarity(embeddings)
    threshold = 0.5

    for i, phrase_i in enumerate(key_phrases):
        G.add_node(phrase_i, label=phrase_i, color="green")
        similarity_scores = [
            (key_phrases[j], similarity_matrix[i][j])
            for j in range(len(key_phrases))
            if i != j and phrase_to_source[phrase_i] != phrase_to_source[key_phrases[j]]
        ]
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]
        for phrase_j, score in sorted_scores:
            if score > threshold:
                G.add_edge(phrase_i, phrase_j, weight=float(score))

    return G

def generate_local_response(prompt, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_insights(G, data):
    insights = []
    degree_centrality = nx.degree_centrality(G)
    avg_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    threshold = avg_centrality * 1.2
    crux_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]

    for crux_node in crux_nodes[:5]:
        neighbors = list(G.neighbors(crux_node))
        neighbor_summary = ", ".join(neighbors[:3]) if neighbors else "No direct connections"
        crux_context = data.loc[data["Content"].str.contains(crux_node, na=False, case=False), "Content"].head(1).values
        context_summary = crux_context[0][:1000] if crux_context.size > 0 else "No specific context available."
        prompt = (
            f"You are an AI tasked with generating insightful observations.\n\n"
            f"Focus concept: {crux_node}\n"
            f"Related themes: {neighbor_summary}\n"
            f"Context: {context_summary}\n\n"
            f"Provide a concise and meaningful insight based on the above."
        )
        try:
            response = generate_local_response(prompt)
            insights.append(response.strip())
        except Exception as e:
            insights.append(f"Error generating insight for '{crux_node}': {e}")

    if not insights or all("Error generating insight" in insight for insight in insights):
        insights = ["No meaningful insights could be generated from the data."]

    return "\n\n".join(insights)

# --- Streamlit UI ---
if "section" not in st.session_state:
    st.session_state.section = "Add Content"

st.sidebar.title("")
if st.sidebar.button("Add Content"):
    st.session_state.section = "Add Content"
if st.sidebar.button("Saved Content"):
    st.session_state.section = "Saved Content"
if st.sidebar.button("Generate Connections"):
    st.session_state.section = "Generate Connections"

section = st.session_state.section

if section == "Add Content":
    st.title("Add New Content")
    input_type = st.radio("Choose input method:", ["Enter URL", "Upload File", "Enter Text"])
    content = ""
    source = ""

    if input_type == "Enter URL":
        url = st.text_input("Enter the URL (video, article, or other)")
        if st.button("Add Content from URL"):
            try:
                source = url
                if "youtube.com" in url or "youtu.be" in url:
                    content = extract_youtube_transcript(url)
                else:
                    content = extract_article_text(url)
                st.success("Content extracted and added!")
            except Exception as e:
                st.error(f"Failed to extract content: {e}")

    elif input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
        if uploaded_file:
            try:
                source = uploaded_file.name
                content = extract_file_text(uploaded_file)
                st.success("File uploaded and processed successfully!")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    elif input_type == "Enter Text":
        content = st.text_area("Enter text")
        if st.button("Add Content from Text"):
            source = "User Input"
            st.success("Content added successfully!")

    if content:
        new_entry = {"Source": source, "Content": content}
        new_row = pd.DataFrame([new_entry])
        data = pd.concat([data, new_row], ignore_index=True)
        data.to_csv("knowledge_data.csv", index=False)

elif section == "Saved Content":
    st.title("Saved Content")
    if not data.empty:
        st.dataframe(data)
    else:
        st.info("No content added yet!")

elif section == "Generate Connections":
    st.title("Generate Connections")
    if not data.empty:
        with st.spinner("Generating knowledge graph..."):
            G = generate_knowledge_graph(data)
            net = Network(height="700px", width="100%", notebook=False)
            net.from_nx(G)
            net.show_buttons(filter_=['physics'])
            net.write_html("knowledge_graph.html")
            try:
                with open("knowledge_graph.html", "r") as f:
                    st.components.v1.html(f.read(), height=700)
            except FileNotFoundError:
                st.error("Unable to render the graph.")

        st.header("Graph Insights")
        insights = generate_insights(G, data)
        st.subheader("AI-Generated Insights:")
        st.write(insights)
    else:
        st.warning("No data available to generate connections!")

st.sidebar.markdown("---")
st.sidebar.markdown("Built using Streamlit.")