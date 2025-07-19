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
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
from docx import Document
import os
import requests
import json
import torch
from nltk.tokenize import sent_tokenize
import nltk
import networkx as nx

# Initialize a local text-generation pipeline
# text_generator = pipeline("text-generation", model="gpt2")

# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
# keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load REBEL model
@st.cache_resource
def load_rebel_model():
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    return tokenizer, model

tokenizer, model = load_rebel_model()

# Parse REBEL output
def parse_rebel_output(text):
    triples = []
    split_text = text.split("<triplet>")
    for chunk in split_text:
        if "<subj>" in chunk and "<rel>" in chunk and "<obj>" in chunk:
            try:
                subj = chunk.split("<subj>")[1].split("<rel>")[0].strip()
                rel = chunk.split("<rel>")[1].split("<obj>")[0].strip()
                obj = chunk.split("<obj>")[1].strip()
                triples.append((subj, rel, obj))
            except:
                continue
    return triples

# Extract triples from text
def extract_triples(text):
    inputs = tokenizer([text], return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return parse_rebel_output(decoded)


def generate_knowledge_graph(data):

    G = nx.MultiDiGraph()
    alias_map = {}

    for index, row in data.iterrows():
        content = row["Content"]
        sentences = sent_tokenize(content)

        for sentence in sentences:
            if len(sentence.split()) < 5:
                continue  # Skip short/unstructured sentences

            triples = extract_triples(sentence)

            print(f"Sentence: {sentence}")
            print(f"Triples: {triples}")

            for subj, rel, obj in triples:
                subj_key = subj.lower().strip()
                obj_key = obj.lower().strip()

                # Basic alias mapping
                for key in [subj_key, obj_key]:
                    if key not in alias_map:
                        alias_map[key] = {key}

                canonical_subj = max(alias_map[subj_key], key=len)
                canonical_obj = max(alias_map[obj_key], key=len)

                G.add_node(canonical_subj, label=canonical_subj)
                G.add_node(canonical_obj, label=canonical_obj)
                G.add_edge(canonical_subj, canonical_obj, label=rel)

    return G


# Helper function to extract key phrases
# top_n is a variable that dictates how many relevant key phrases should it return
# def extract_key_phrases(content, top_n=50):
#     """Extract key phrases using KeyBERT."""
#     keywords = keybert_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
#     return [kw[0] for kw in keywords]


def generate_insight_mistral(prompt):
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




# Generate insights from the graph and data
def generate_insights(G, data):
    """Generate concise, meaningful, and insightful paragraphs using LLM with transcript context."""
    insights = []
    max_new_tokens = 200  # Adjusted to ensure detailed LLM responses

    if G.number_of_nodes() == 0:
        return "No data connections found in the knowledge graph."

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
        context_summary = (
            crux_context[0] if crux_context.size > 0 else "No specific context available."
        )

        # Construct the LLM prompt
        prompt = (
            "You are a research analyst generating strategic insights from cross-domain knowledge.\n\n"
            f"Main Topic: \"{crux_node}\"\n"
            f"Related Themes: {neighbor_summary}\n\n"
            "Context:\n"
            f"\"\"\"\n{context_summary}\n\"\"\"\n\n"
            "Based on the above, write a concise, original insight that draws a meaningful connection between the topic and its themes. "
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
            for u, v, data_edge in G.edges(data=True):
                net.add_node(u, label=u)
                net.add_node(v, label=v)
                net.add_edge(u, v, title=data_edge.get("label", ""), label=data_edge.get("label", ""))

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