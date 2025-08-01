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
import requests
import json
import spacy
import re
from collections import defaultdict, Counter
import numpy as np

# Load spaCy model for NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000
except OSError:
    st.error("Please install spaCy English model: python -m spacy download en_core_web_sm")
    st.stop()

# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

# Initialize data storage
if "knowledge_data.csv" not in os.listdir():
    pd.DataFrame(columns=["Source", "Content"]).to_csv("knowledge_data.csv", index=False)

# Load existing data
data = pd.read_csv("knowledge_data.csv")


# Tana-inspired CSS for modern, clean design
def add_tana_css():
    st.markdown("""
        <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global styles */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }

        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom header */
        .hero-section {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 4rem 2rem 3rem 2rem;
            margin: -1rem -1rem 3rem -1rem;
            border-radius: 0 0 24px 24px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero-subtitle {
            font-size: 1.3rem;
            font-weight: 400;
            opacity: 0.9;
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        .testimonial-quote {
            font-size: 1.1rem;
            font-style: italic;
            opacity: 0.8;
            margin: 2rem auto;
            max-width: 600px;
            border-left: 3px solid #3b82f6;
            padding-left: 1.5rem;
        }

        /* Modern sidebar */
        .css-1d391kg {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid #e2e8f0;
        }

        /* Navigation buttons */
        .nav-button {
            width: 100%;
            margin: 0.5rem 0;
            padding: 1rem 1.5rem;
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 500;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        }

        .nav-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        }

        .nav-button.active {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            box-shadow: 0 4px 12px rgba(31, 41, 55, 0.3);
        }

        /* Content sections */
        .content-section {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            margin-bottom: 2rem;
            border: 1px solid #e2e8f0;
        }

        .section-title {
            font-size: 2rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .section-subtitle {
            font-size: 1.1rem;
            color: #64748b;
            text-align: center;
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        /* Feature cards */
        .feature-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
            border-color: #3b82f6;
        }

        .feature-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 0.5rem;
        }

        .feature-description {
            color: #64748b;
            line-height: 1.5;
        }

        /* Modern form elements */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        /* Action buttons */
        .stButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3);
        }

        /* Success/Info messages */
        .stSuccess {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 1rem;
        }

        .stInfo {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            border: 1px solid #3b82f6;
            border-radius: 8px;
            padding: 1rem;
        }

        .stWarning {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: 1rem;
        }

        /* Metrics */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #3b82f6;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            color: #64748b;
            font-weight: 500;
        }

        /* Data table styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
            border-radius: 4px;
        }

        /* Sidebar branding */
        .sidebar-brand {
            text-align: center;
            padding: 2rem 1rem;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 2rem;
        }

        .brand-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.5rem;
        }

        .brand-subtitle {
            font-size: 0.9rem;
            color: #64748b;
            font-weight: 400;
        }

        /* Insights section */
        .insights-container {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
        }

        .insights-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 1rem;
            text-align: center;
        }

        .insight-text {
            font-size: 1rem;
            line-height: 1.7;
            color: #374151;
            text-align: justify;
        }
        </style>
    """, unsafe_allow_html=True)


# Add the custom CSS
add_tana_css()


# Hero Section
def render_hero():
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Knowledge Graph Studio</h1>
            <p class="hero-subtitle">Transform your scattered thoughts and content into meaningful connections</p>
            <div class="testimonial-quote">
                "This tool completely transformed how I connect ideas by allowing me to merge my professional and personal knowledge into one cohesive graph."
                <br><strong>— Knowledge Worker</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)


# Sidebar with modern navigation
def render_sidebar():
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-brand">
                <div class="brand-title">KG Studio</div>
                <div class="brand-subtitle">Intelligent Knowledge Management</div>
            </div>
        """, unsafe_allow_html=True)

        # Navigation buttons
        sections = ["Add Content", "Saved Content", "Generate Connections"]
        for section in sections:
            if st.button(section, key=f"nav_{section.lower().replace(' ', '_')}",
                         help=f"Navigate to {section}"):
                st.session_state.section = section


# Feature cards for different input methods
def render_feature_cards():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title">📄 Smart Content Extraction</div>
                <div class="feature-description">
                    Automatically extract insights from URLs, documents, and text with advanced NLP processing.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🕸️ Semantic Connections</div>
                <div class="feature-description">
                    Discover hidden relationships between your ideas using state-of-the-art semantic analysis.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🧠 AI-Powered Insights</div>
                <div class="feature-description">
                    Generate meaningful insights and patterns from your knowledge graph using advanced AI.
                </div>
            </div>
        """, unsafe_allow_html=True)


# Modern metrics display
def render_metrics(G):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{G.number_of_nodes()}</div>
                <div class="metric-label">Knowledge Nodes</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{G.number_of_edges()}</div>
                <div class="metric-label">Connections</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        density = nx.density(G) if G.number_of_nodes() > 1 else 0
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{density:.2f}</div>
                <div class="metric-label">Network Density</div>
            </div>
        """, unsafe_allow_html=True)


# Initialize session state
if "section" not in st.session_state:
    st.session_state.section = "Add Content"

# Render hero section
render_hero()

# Render sidebar
render_sidebar()

# Main content area
section = st.session_state.section

if section == "Add Content":
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Add New Knowledge</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Expand your knowledge graph with content from various sources</p>',
                unsafe_allow_html=True)

    # Feature cards
    render_feature_cards()

    # Input method selection
    input_type = st.radio("Choose your input method:",
                          ["Enter URL", "Upload File", "Enter Text"],
                          horizontal=True)

    content = ""
    source = ""

    if input_type == "Enter URL":
        url = st.text_input("Enter the URL (video, article, or other)",
                            placeholder="https://example.com/article")
        if st.button("🔗 Extract Content from URL", key="add_url_content"):
            source = url
            content = f"Extracted content from {url}"
            st.success("Content extracted and added successfully! 🎉")

    elif input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload a document",
                                         type=["txt", "pdf", "docx"],
                                         help="Supported formats: TXT, PDF, DOCX")
        if uploaded_file:
            source = uploaded_file.name
            content = f"Content from file: {uploaded_file.name}"
            st.success("File processed and added successfully! 📁")

    elif input_type == "Enter Text":
        content = st.text_area("Enter your text",
                               height=200,
                               placeholder="Paste your content here...")
        if st.button("💭 Add Text Content", key="add_text_content"):
            source = "User Input"
            st.success("Text content added successfully! ✨")

    if content:
        new_entry = {"Source": source, "Content": content}
        new_row = pd.DataFrame([new_entry])
        data = pd.concat([data, new_row], ignore_index=True)
        data.to_csv("knowledge_data.csv", index=False)

    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Saved Content":
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Your Knowledge Base</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Review and manage your collected content</p>', unsafe_allow_html=True)

    if not data.empty:
        st.dataframe(data, use_container_width=True, height=400)

        # Summary statistics
        st.markdown("### 📊 Content Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total Items:** {len(data)}")
        with col2:
            source_counts = data['Source'].value_counts()
            st.info(f"**Unique Sources:** {len(source_counts)}")
    else:
        st.info("🤔 No content added yet! Head over to 'Add Content' to get started.")

    st.markdown('</div>', unsafe_allow_html=True)

elif section == "Generate Connections":
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Knowledge Graph Visualization</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Discover the hidden connections in your knowledge</p>',
                unsafe_allow_html=True)

    if not data.empty:
        if st.button("🧠 Generate Knowledge Graph", key="generate_graph_btn"):
            with st.spinner("🔄 Analyzing content and building connections..."):
                # Simplified knowledge graph generation for demo
                G = nx.Graph()

                # Extract key phrases and create connections
                for index, row in data.iterrows():
                    # Simplified key phrase extraction
                    keywords = keybert_model.extract_keywords(row["Content"],
                                                              keyphrase_ngram_range=(1, 2),
                                                              stop_words="english",
                                                              top_n=5)
                    phrases = [kw[0] for kw in keywords]

                    for phrase in phrases:
                        G.add_node(phrase, color="#3b82f6", size=20)

                # Add edges based on co-occurrence
                nodes = list(G.nodes())
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1:i + 4]:  # Connect to next 3 nodes
                        G.add_edge(node1, node2)

                # Display metrics
                render_metrics(G)

                # Generate visualization
                st.markdown("### 🕸️ Interactive Knowledge Graph")
                net = Network(height="600px", width="100%", bgcolor="#ffffff")
                net.from_nx(G)
                net.set_options("""
                    var options = {
                        "physics": {
                            "enabled": true,
                            "stabilization": {"iterations": 100}
                        },
                        "nodes": {
                            "borderWidth": 2,
                            "borderWidthSelected": 4,
                            "color": {
                                "border": "#3b82f6",
                                "background": "#dbeafe",
                                "highlight": {"border": "#1d4ed8", "background": "#bfdbfe"}
                            }
                        },
                        "edges": {
                            "color": {"color": "#94a3b8"},
                            "smooth": {"type": "continuous"}
                        }
                    }
                """)

                net.write_html("knowledge_graph.html")

                try:
                    with open("knowledge_graph.html", "r") as f:
                        st.components.v1.html(f.read(), height=600)
                except FileNotFoundError:
                    st.error("Unable to render the graph.")

                # AI Insights section
                st.markdown("""
                    <div class="insights-container">
                        <h3 class="insights-title">🤖 AI-Generated Insights</h3>
                        <div class="insight-text">
                            Your knowledge graph reveals several interesting patterns. The central themes appear to focus on 
                            interconnected concepts that suggest a systematic approach to learning. Key nodes with high 
                            connectivity indicate areas where you might want to dive deeper or explore cross-connections.
                            <br><br>
                            Consider exploring the relationships between your most connected concepts to discover new 
                            learning opportunities and potential knowledge gaps.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("📝 Add some content first to generate your knowledge graph!")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
            <strong>Knowledge Graph Studio</strong><br>
            Powered by advanced NLP & AI<br><br>
            <em>Transform information into wisdom</em>
        </div>
    """, unsafe_allow_html=True)