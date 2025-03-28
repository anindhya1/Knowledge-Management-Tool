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
import matplotlib.pyplot as plt
import networkx as nx

# Initialize a local text-generation pipeline
text_generator = pipeline("text-generation", model="gpt2")

# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

# Initialize data storage
if "knowledge_data.csv" not in os.listdir():
    pd.DataFrame(columns=["Source", "Content", "Type"]).to_csv("knowledge_data.csv", index=False)

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

# Extract transcripts from YouTube videos
def extract_youtube_transcript(video_url):
    try:
        video_id = urlparse(video_url).query.split('=')[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error extracting transcript: {e}"

# Extract text from articles
def extract_article_text(article_url):
    try:
        article = Article(article_url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting article text: {e}"

# Extract text from PDFs
def extract_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return " ".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        return f"Error extracting PDF text: {e}"

# Extract text from Word documents
def extract_docx_text(docx_file):
    try:
        doc = Document(docx_file)
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        return f"Error extracting DOCX text: {e}"

# Define the generate_knowledge_graph function
# Helper function to extract meaningful key phrases
def extract_key_phrases(content, top_n=10):
    """Extract meaningful key phrases using KeyBERT and validate them."""
    # Extract keywords using KeyBERT
    raw_keywords = keybert_model.extract_keywords(
        content,
        keyphrase_ngram_range=(2, 3),  # Focus on 2-3 word phrases
        stop_words="english",
        top_n=top_n
    )

    # Validate extracted phrases by checking semantic coherence
    meaningful_phrases = []
    for phrase, score in raw_keywords:
        if len(phrase.split()) > 1:  # Only consider multi-word phrases
            # Ensure the phrase has meaningful embeddings
            phrase_embedding = model.encode([phrase])
            content_embedding = model.encode([content])
            similarity = cosine_similarity(phrase_embedding, content_embedding)[0][0]
            if similarity > 0.4:  # Semantic relevance threshold
                meaningful_phrases.append(phrase)

    return meaningful_phrases if meaningful_phrases else ["No meaningful phrases found"]


# Define the generate_knowledge_graph function
def generate_knowledge_graph(data):
    """Generate and return the knowledge graph with validated node tags."""
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
    threshold = 0.3

    for i, phrase_i in enumerate(key_phrases):
        # Assign phrase as a node, ensure it's capitalized for readability
        G.add_node(phrase_i, label=phrase_i.capitalize(), color="blue")
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


# Define the visualize_graph function
def visualize_graph(G):
    """Visualize the graph with normalized edge thickness for readability."""
    plt.figure(figsize=(12, 8))

    # Position nodes using a layout algorithm
    pos = nx.spring_layout(G, seed=42)  # You can try other layouts like `nx.kamada_kawai_layout`

    # Extract edge weights
    edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]

    # Normalize edge thickness
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 1
    normalized_weights = [
        1 + 4 * ((weight - min_weight) / (max_weight - min_weight))  # Scale between 1 and 5
        if max_weight > min_weight else 1
        for weight in edge_weights
    ]

    # Draw nodes and edges
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        width=normalized_weights,  # Apply normalized weights to edge thickness
        node_size=500,
        font_size=10,
        font_color="black",
    )

    # Add a title and show the plot
    plt.title("Knowledge Graph", fontsize=16)
    plt.show()

# Generate insights from the graph and data
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
        # Gather neighbors and context
        neighbors = list(G.neighbors(crux_node))
        neighbor_summary = ", ".join(neighbors[:3]) if neighbors else "No direct connections"

        crux_context = data.loc[data["Content"].str.contains(crux_node, na=False, case=False), "Content"].head(1).values
        context_summary = (
            crux_context[0] if crux_context.size > 0 else "No specific context available."
        )

        # Construct the LLM prompt
        prompt = (
            f"You are an AI tasked with generating insightful observations.\n\n"
            f"Focus concept: {crux_node}\n"
            f"Related themes: {neighbor_summary}\n"
            f"Context: {context_summary}\n\n"
            f"Provide a concise and meaningful insight based on the above."
        )

        try:
            # Generate insight using LLM
            response = text_generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)[0]["generated_text"]
            clean_response = response.strip()  # Remove any unnecessary formatting or text
            insights.append(clean_response)
        except Exception as e:
            insights.append(f"Error generating insight for '{crux_node}': {e}")

    # Handle cases with no valid insights
    if not insights or all("Error generating insight" in insight for insight in insights):
        insights = ["No meaningful insights could be generated from the data."]

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
            if "youtube.com" in url:
                content = extract_youtube_transcript(url)
            else:
                content = extract_article_text(url)
            st.success("Content added successfully!")

    elif input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
        if uploaded_file:
            source = uploaded_file.name
            if uploaded_file.type == "application/pdf":
                content = extract_pdf_text(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content = extract_docx_text(uploaded_file)
            else:
                content = uploaded_file.read().decode("utf-8")
            st.success("File uploaded and processed successfully!")

    elif input_type == "Enter Text":
        content = st.text_area("Enter text")
        if st.button("Add Content from Text"):
            source = "User Input"
            st.success("Content added successfully!")

    if content:
        new_entry = {"Source": source, "Content": content, "Type": input_type}
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

            # Visualize the graph
            visualize_graph(G)

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
