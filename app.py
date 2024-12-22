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

# Define the generate_knowledge_graph function
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

    # Increased similarity threshold to reduce noise
    threshold = 0.5

    for i, phrase_i in enumerate(key_phrases):
        G.add_node(phrase_i, label=phrase_i, color="green")
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
def extract_key_phrases(content, top_n=10):
    """Extract key phrases using KeyBERT."""
    keywords = keybert_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]

# Generate insights from the graph and data
# Generate insights from the graph and data
def generate_insights(G, data):
    """Generate concise and meaningful insights using Hugging Face LLM."""
    insights = []

    # Identify key nodes based on degree centrality
    degree_centrality = nx.degree_centrality(G)
    avg_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    threshold = avg_centrality * 1.2  # Adjust threshold for meaningful crux nodes
    crux_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]

    for crux_node in crux_nodes[:5]:  # Limit to top 5 nodes
        # Get neighbors and summarize their relationships
        neighbors = list(G.neighbors(crux_node))
        neighbor_summary = ", ".join(neighbors[:3])  # Summarize up to 3 neighbors for brevity

        # Extract context for the crux node
        crux_context = data.loc[data["Content"].str.contains(crux_node, na=False, case=False), "Content"].head(1).values
        if crux_context.size > 0:  # Explicit check for non-empty array
            if len(crux_context[0]) > 300:
                context_summary = crux_context[0][:300] + "..."
            else:
                context_summary = crux_context[0]
        else:
            context_summary = "No specific context available."

        # Construct prompt for LLM
        prompt = (
            f"Concept: {crux_node}\n"
            f"Connected Themes: {neighbor_summary}\n"
            f"Context: {context_summary}\n\n"
            f"Generate a concise and meaningful insight based on this information."
        )

        # Generate insight using the LLM
        try:
            llm_response = text_generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
            insights.append(llm_response.strip())  # Append only the insight
        except Exception as e:
            insights.append(f"Error generating insight for {crux_node}: {e}")

    # Fallback if no insights are generated
    if not insights:
        insights = ["No significant insights could be generated from the data."]

    return "\n\n".join(insights)









# Streamlit App UI
# Initialize session state for navigation
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
