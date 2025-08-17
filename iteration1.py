"""
This the first iteration of the project 'ALCMEAON'. Here we are using KEYBERT to break text into key phrases. A certain
number of top n phrases are then chosen, to maintain high degree of relevance. The phrases are then converted into
embeddings and using cosine similarities, under a certain threshold value, connection are made between the these phrases.
This is then represented as a knowledge graph. The key phrases are treated as nodes, and those nodes that are connected
are represented as edges.
Now, having created a wide number of node-edge pairs, there would be some nodes that will have more connections than
the rest. These nodes are treated as central-nodes, which we call crux nodes here. These central nodes are given higher
weightage. This is done by calculating 'node centrality'.
These central nodes are then, along with the older content are thrown into the LLM (in this case Mistral), packaged into
a prompt. This helps the LLM know where to lay focus, considering what is being asked of it.

- @Author: Anindhya Kushagra
- Spervised by Thomas J. Borrelli
- Rochester Institute of Technology
"""

import pandas as pd
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi
from newspaper import Article
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline
import PyPDF2
from docx import Document
import networkx as nx
import os
import requests
import json
import streamlit as st


# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

# Initialize data storage
if "knowledge_data.csv" not in os.listdir():
    pd.DataFrame(columns=["Source", "Content"]).to_csv("knowledge_data.csv", index=False)

# Load existing data
data = pd.read_csv("knowledge_data.csv")


def add_custom_css():
    """
    To maintain a decent UX, frontend is directly being integrated here.
    """
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

# Define the generate_knowledge_graph function
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

    # Thershold factor. Increase to increase noise. Decrease for vice-versa.
    threshold = 0.5

    for i, phrase_i in enumerate(key_phrases):
        G.add_node(phrase_i, label=phrase_i, color="#008080")
        similarity_scores = [
            (key_phrases[j], similarity_matrix[i][j])
            for j in range(len(key_phrases))
            if i != j and phrase_to_source[phrase_i] != phrase_to_source[key_phrases[j]]
        ]
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]  # Top 3 connections
        for phrase_j, score in sorted_scores:
            if score > threshold:
                G.add_edge(phrase_i, phrase_j)

    return G


# Helper function to extract key phrases
def extract_key_phrases(content, top_n=50):
    """Extract top-n key phrases using KeyBERT."""
    keywords = keybert_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]


def generate_insight_mistral(prompt):
    """Mistral seemed like the best choice for a local-logical reasoning model."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": True,
            "num_predict": 400,  # Use this instead of max_tokens
            "temperature": 0.7
        },
        stream=True
    )

    output = ""
    try:
        for line in response.iter_lines():
            if line:
                line_data = json.loads(line.decode("utf-8"))
                if "response" in line_data:
                    output += line_data["response"]
    except Exception as e:
        output = f"Error: {e}"
    return output.strip()


# Generating insights from the graph and csv data
def generate_insights(G, data):
    """Generate concise, meaningful, and insightful paragraphs using LLM with transcript context."""
    insights = []
    max_new_tokens = 200  # Adjusted to ensure detailed LLM responses

    # Identify key nodes based on degree centrality
    degree_centrality = nx.degree_centrality(G)
    avg_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    threshold = avg_centrality * 1.2  # Threshold for selecting key nodes
    crux_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]

    for crux_node in crux_nodes[:5]:  # Limit to top 5 key nodes
        # Gather neighbors and summarize their relationships
        neighbors = list(G.neighbors(crux_node))
        neighbor_summary = ", ".join(neighbors[:3]) if neighbors else "No direct connections"

        # Extract detailed context for the crux node from the transcripts or content
        crux_context = data.loc[data["Content"].str.contains(crux_node, na=False, case=False), "Content"].head(1).values
        context_summary = (crux_context[0] if crux_context.size > 0 else "No specific context available.")

        # Construct the LLM prompt
        prompt = (
            "You are my personal assistant examining knowledge relationships. Analyze these high-confidence" 
            "triplets and provide concise strategic insights about the underlying patterns, themes, or implications.\n\n"
            f"Main Topic: \"{crux_node}\"\n"
            f"Related Themes: {neighbor_summary}\n\n"
            "Context:\n"
            f"\"\"\"\n{context_summary}\n\"\"\"\n\n"
            "Based on the above, write a concise, original insight that draws a meaningful connection between the topic "
            "and its themes. "
            "Avoid stating the obvious, and highlight a non-trivial pattern or implication.\n\n"
            "Keep the insight within 3–5 sentences."
        )

        print("Length of prompt: ", len(prompt))

        try:
            # Generate insight using LLM
            print("=== PROMPT ===")
            print(prompt)
            print("==============")

            response = generate_insight_mistral(prompt)
            clean_response = response.strip()  # Remove any unnecessary formatting or text
            insights.append(clean_response)
        except Exception as e:
            insights.append(f"Error generating insight for '{crux_node}': {e}")

    # Handle cases with no valid insights
    if not insights or all("Error generating insight" in insight for insight in insights):
        insights = ["No meaningful insights could be generated from the data."]

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

# Retrieve current section from session state
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
            content = f"Extracted content from {url}"  # Replace with real extraction logic
            st.success("Content added successfully!")

    elif input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
        if uploaded_file:
            source = uploaded_file.name
            content = f"Content from file: {uploaded_file.name}"  # Replace with real file processing logic
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
        st.dataframe(data)  # Show saved data in an interactive table
    else:
        st.info("No content added yet!")

# Generate Connections Section
elif section == "Generate Connections":
    st.title("Generate Connections")
    if not data.empty:
        with st.spinner("Generating knowledge graph..."):
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

        # Generate insights
        st.header("Graph Insights")
        insights = generate_insights(G, data)
        st.subheader("AI-Generated Insights:")
        st.write(insights)
    else:
        st.warning("No data available to generate connections!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built using Streamlit.")