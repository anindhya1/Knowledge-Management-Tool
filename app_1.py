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
from transformers import pipeline
import PyPDF2
from docx import Document
import os

# Initialize a local text-generation pipeline
text_generator = pipeline("text-generation", model="gpt2")

# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
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

# Context-based graph generator

def generate_context_graph(data, threshold=0.6):
    G = nx.Graph()

    contexts = data["Content"].fillna("").tolist()
    sources = data["Source"].fillna("Unknown").tolist()

    embeddings = model.encode(contexts)
    sim_matrix = cosine_similarity(embeddings)

    for i, text in enumerate(contexts):
        G.add_node(i,
                   label=f"Context {i+1}",
                   title=f"<b>Source:</b> {sources[i]}<br><b>Preview:</b> {text[:200]}...",
                   group="Context",
                   color="teal")

    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            if sim_matrix[i][j] > threshold:
                G.add_edge(i, j, weight=float(sim_matrix[i][j]))

    return G

# Generate insights from context graph

def generate_insights(G, data):
    insights = []
    max_new_tokens = 200

    for node in list(G.nodes)[:5]:
        neighbors = list(G.neighbors(node))
        neighbor_indices = ", ".join([f"Context {n+1}" for n in neighbors])

        context_text = data.iloc[node]["Content"]
        source_text = data.iloc[node]["Source"]

        prompt = (
            f"You are an AI tasked with generating insightful observations from a user’s knowledge graph.\n\n"
            f"Focus context (Context {node+1}):\n{context_text[:300]}...\n\n"
            f"Related to: {neighbor_indices if neighbor_indices else 'No direct context links'}\n\n"
            f"Source: {source_text}\n\n"
            f"Give a meaningful insight or interpretation."
        )

        try:
            response = text_generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)[0]["generated_text"]
            insights.append(response.strip())
        except Exception as e:
            insights.append(f"Error generating insight: {e}")

    return "\n\n".join(insights)

# Streamlit App UI
if "section" not in st.session_state:
    st.session_state.section = "Add Content"

# Navigation bar
st.sidebar.title("")
if st.sidebar.button("Add Content"):
    st.session_state.section = "Add Content"
if st.sidebar.button("Saved Content"):
    st.session_state.section = "Saved Content"
if st.sidebar.button("Generate Connections"):
    st.session_state.section = "Generate Connections"

section = st.session_state.section

# Add Content Section
if section == "Add Content":
    st.title("Add New Content")
    input_type = st.radio("Choose input method:", ["Enter URL", "Upload File", "Enter Text"])

    content = ""
    source = ""

    if input_type == "Enter URL":
        url = st.text_input("Enter the URL (video, article, or other)")
        if st.button("Add Content from URL"):
            source = url
            content = f"Extracted content from {url}"
            st.success("Content added successfully!")

    elif input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
        if uploaded_file:
            source = uploaded_file.name
            content = f"Content from file: {uploaded_file.name}"
            st.success("File uploaded and processed successfully!")

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

# Saved Content Section
elif section == "Saved Content":
    st.title("Saved Content")

    if not data.empty:
        for idx, row in data.iterrows():
            with st.container():
                cols = st.columns([5, 5, 1])
                cols[0].markdown(f"**Title:** {row.get('Title', 'No Title')}")
                cols[1].markdown(f"**Source:** {row.get('Source', 'Unknown')}")
                delete_button_key = f"delete_{idx}"

                if cols[2].button("🗑️", key=delete_button_key):
                    st.session_state[f"confirm_delete_{idx}"] = True

            # Confirm deletion section
            if st.session_state.get(f"confirm_delete_{idx}", False):
                st.warning(f"Are you sure you want to delete '{row.get('Title', 'this item')}'?")
                confirm_cols = st.columns([1, 1])
                if confirm_cols[0].button("Yes", key=f"confirm_yes_{idx}"):
                    data = data.drop(index=idx).reset_index(drop=True)
                    data.to_csv("knowledge_data.csv", index=False)
                    st.success(f"Deleted '{row.get('Title', 'item')}'")
                    st.rerun()
                if confirm_cols[1].button("No", key=f"confirm_no_{idx}"):
                    st.session_state[f"confirm_delete_{idx}"] = False
    else:
        st.info("No content added yet!")


# Generate Connections Section
elif section == "Generate Connections":
    st.title("Generate Connections")
    if not data.empty:
        with st.spinner("Generating context graph..."):
            G = generate_context_graph(data)
            net = Network(height="700px", width="100%")
            net.from_nx(G)
            net.write_html("context_graph.html")
            try:
                with open("context_graph.html", "r") as f:
                    st.components.v1.html(f.read(), height=700)
            except FileNotFoundError:
                st.error("Unable to render the graph.")

        st.header("Graph Insights")
        insights = generate_insights(G, data)
        st.subheader("AI-Generated Insights:")
        st.write(insights)
    else:
        st.warning("No data available to generate connections!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built using Streamlit.")
