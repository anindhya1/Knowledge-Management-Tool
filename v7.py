"""
pull insights based on those nodes that have the most edges and the nodes that are connected to such nodes.
Nodes with most edges probably act like a centroid.

along with the fact that we are using nodes that have the most edges, how about generating insights from the nodes
that are connected to the nodes of different sources as well?
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
from keybert import KeyBERT
from transformers import pipeline
import PyPDF2
from docx import Document
import os
import anthropic
from transformers import pipeline

# Initialize a local text-generation pipeline
text_generator = pipeline("text-generation", model="gpt2")



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
            if score > 0.3:
                G.add_edge(phrase_i, phrase_j)

    return G


def generate_meaningful_insights_locally(G, data):
    """Generate meaningful insights centered on nodes with the most neighbors and diverse-source connections."""
    # Step 1: Identify nodes with the most neighbors (high degree centrality)
    degree_centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    # Step 2: Filter nodes with neighbors from different sources
    insights_data = []
    sources = data.set_index("Source")["Content"].to_dict()  # Map source to content for quick lookup

    for node, _ in sorted_nodes:
        neighbors = list(G.neighbors(node))

        # Identify neighbors from diverse sources
        node_source = sources.get(node, None)
        diverse_neighbors = [
            neighbor for neighbor in neighbors if sources.get(neighbor, None) != node_source
        ]

        # Fallback to all neighbors if no diverse neighbors found
        if not diverse_neighbors:
            diverse_neighbors = neighbors

        # Extract context for the node
        source_context = data.loc[data["Content"].str.contains(node, na=False, case=False), "Content"].head(1).values
        node_context = source_context[0] if len(source_context) > 0 else f"Insights related to '{node}'."

        # Extract context for neighbors
        neighbor_contexts = [
            data.loc[data["Content"].str.contains(neighbor, na=False, case=False), "Content"].head(1).values
            for neighbor in diverse_neighbors[:3]  # Limit to 3 diverse neighbors
        ]
        diverse_context = " ".join(
            [ctx[0] if len(ctx) > 0 else f"Related topic: {neighbor}" for ctx, neighbor in zip(neighbor_contexts, diverse_neighbors)]
        )

        insights_data.append({"context": node_context, "diverse_context": diverse_context})

        # Limit to top 3 nodes for simplicity
        if len(insights_data) >= 3:
            break

    # If no nodes meet criteria, provide fallback
    if not insights_data:
        insights_data = [{"context": "No meaningful nodes identified.", "diverse_context": "No relevant connections found."}]

    # Step 3: Build the LLM prompt
    insights_prompts = []
    for item in insights_data:
        context = item["context"]
        diverse_context = item["diverse_context"]
        insights_prompts.append(f"Context: {context}. Related Topics: {diverse_context}.")

    prompt = "\n\n".join(insights_prompts)
    if len(prompt) > 500:  # Truncate if too long
        prompt = prompt[:500]

    # Step 4: Generate insights using the text generator
    try:
        generated_text = text_generator(
            prompt,
            max_new_tokens=300,  # Generate up to 300 tokens
            num_return_sequences=1,
        )[0]["generated_text"]
    except Exception as e:
        generated_text = f"Error during insight generation: {e}"

    # Step 5: Format the insights
    insights = post_process_paragraphs(generated_text, insights_data)
    return insights


def post_process_paragraphs(text, insights_data):
    """Post-process the generated text into clean paragraphs with contextual headings."""
    lines = text.split("\n")
    paragraphs = []
    for idx, item in enumerate(insights_data):
        heading = f"### {item['context']}"
        paragraph = lines[idx].strip() if idx < len(lines) else "No additional insights generated."
        paragraphs.append(f"{heading}\n\n{paragraph}")
    return "\n\n".join(paragraphs)





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