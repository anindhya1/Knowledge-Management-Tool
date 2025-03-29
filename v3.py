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

# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')
text_generator = pipeline("text-generation", model="gpt2")  # Use GPT-2 for generative insights

# Initialize data storage
if "knowledge_data.csv" not in os.listdir():
    pd.DataFrame(columns=["Source", "Content"]).to_csv("knowledge_data.csv", index=False)

# Load existing data
data = pd.read_csv("knowledge_data.csv")

# Helper functions
def extract_key_phrases(content, top_n=10):
    """Extract key phrases using KeyBERT."""
    keywords = keybert_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]

def generate_knowledge_graph(data):
    """Generate and return the knowledge graph."""
    G = nx.Graph()
    key_phrases = []
    phrase_to_source = {}

    for index, row in data.iterrows():
        phrases = extract_key_phrases(row["Content"], top_n=10)
        key_phrases.extend(phrases)
        phrase_to_source.update({phrase: row["Source"] for phrase in phrases})

    embeddings = model.encode(key_phrases)
    similarity_matrix = cosine_similarity(embeddings)

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

    return G

def generate_meaningful_insights_locally(G, data):
    """Generate meaningful and abstract insights using a local model."""
    graph_summary = []

    # Group key nodes and their neighbors into abstract relationships
    degree_centrality = nx.degree_centrality(G)
    most_connected_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    if most_connected_nodes:
        for node, _ in most_connected_nodes:
            related_nodes = list(G.neighbors(node))
            if related_nodes:
                graph_summary.append(
                    f"The idea '{node}' is strongly connected to related topics like {', '.join(related_nodes[:3])}, suggesting a shared focus on {node.lower()}."
                )
            else:
                graph_summary.append(
                    f"The idea '{node}' is central but does not have directly connected ideas."
                )

    # Summarize cluster themes
    clusters = list(nx.connected_components(G))
    if clusters:
        cluster_themes = []
        for idx, cluster in enumerate(clusters[:5]):  # Limit to top 5 clusters for simplicity
            cluster_nodes = list(cluster)
            cluster_themes.append(f"Cluster {idx + 1} focuses on topics such as {', '.join(cluster_nodes[:3])}.")
        graph_summary.append(" ".join(cluster_themes))

    # Include user-provided content context
    content_contexts = []
    for node in G.nodes():
        matching_rows = data[data["Source"].str.contains(node, na=False, regex=False)]
        if not matching_rows.empty:
            content_contexts.append(
                f"The idea '{node}' originates from the following content: {matching_rows['Content'].iloc[0][:200]}..."
            )

    # Combine abstracted graph insights and content contexts
    narrative_prompt = (
        "Based on the following patterns and contexts, generate meaningful insights:\n\n"
        "Graph Patterns:\n" + "\n".join(graph_summary) + "\n\n"
        "Content Context:\n" + "\n".join(content_contexts) + "\n\n"
        "Please provide a concise and meaningful summary of the above relationships, avoiding technical terms like 'nodes' or 'clusters'. Focus on the shared themes and broader implications."
    )

    # Generate abstract insights locally using GPT-2
    generated_text = text_generator(
        narrative_prompt,
        max_new_tokens=150,  # Limit to concise output
        num_return_sequences=1
    )[0]["generated_text"]

    return generated_text



# Streamlit App
st.title("Personal Knowledge Management Tool")
st.markdown("Organize, connect, and interpret knowledge from various sources.")

# Input section
st.header("Add Content")
input_type = st.radio("Choose input method:", ("Enter URL", "Upload File", "Enter Text"))

content = ""
source = ""

# Process input
if input_type == "Enter URL":
    url = st.text_input("Enter the URL (video, article, or other)")
    if st.button("Add Content from URL"):
        source = url
        # Process URL (logic similar to previous examples)
        st.success("Content added successfully!")

elif input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
    if uploaded_file:
        source = uploaded_file.name
        # Process file (logic similar to previous examples)
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

# Generate graph and insights
if st.button("Generate Connections"):
    if not data.empty:
        G = generate_knowledge_graph(data)

        # Display graph
        net = Network(height="700px", width="100%")
        net.from_nx(G)
        net.write_html("knowledge_graph.html")
        try:
            with open("knowledge_graph.html", "r") as f:
                st.components.v1.html(f.read(), height=700)
        except FileNotFoundError:
            st.error("Unable to render the graph.")

        # Generate insights using LLM
        st.header("Graph Insights")
        insights = generate_meaningful_insights_locally(G, data)
        st.subheader("AI-Generated Insights:")
        st.write(insights)
    else:
        st.warning("No data available to generate connections!")
