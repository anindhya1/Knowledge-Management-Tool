"""
This the fourth iteration of the project 'ALCMEAON'. Here we combine the best of what we have had in the previous
iterations. It involves the leveraging spacy's dependency parser, a separate prompting technique to generate similarities
in the content that has been extracted. These methods are based on the same grammar rules of the REBEL model, however,
it removes the constraints the REBEL model.
Now, having created a wide number of node-edge pairs, there would be some nodes that will have more connections than
the rest. These nodes are treated as central-nodes, which we call crux nodes here. These central nodes are given higher
weightage. This is done by calculating 'node centrality'.
These central nodes are then, along with the older content are thrown into the LLM (in this case Mistral), packaged into
a prompt. This helps the LLM know where to lay focus, considering what is being asked of it.

- @Author: Anindhya Kushagra
- Spervised by Thomas J. Borrelli
- Rochester Institute of Technology

Run with: streamlit run frontend/app.py  (from the project root)
"""

import sys
import os

# Ensure the project root is on sys.path so backend imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import PyPDF2
from docx import Document
from pyvis.network import Network
from urllib.parse import urlparse

from backend.data_manager import load_data, save_data
from backend.content_extractor import extract_video_id, get_youtube_transcript, extract_article_content
from backend.graph_builder import generate_semantic_knowledge_graph
from backend.insight_generator import generate_semantic_insights


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


# ── App setup ──────────────────────────────────────────────────────────────────

add_custom_css()

if "section" not in st.session_state:
    st.session_state.section = "Add Content"

# Navigation
if st.sidebar.button("Add Content", key="nav_add_content"):
    st.session_state.section = "Add Content"
if st.sidebar.button("Saved Content", key="nav_saved_content"):
    st.session_state.section = "Saved Content"
if st.sidebar.button("Generate Connections", key="nav_generate_graph"):
    st.session_state.section = "Generate Semantic Graph"

section = st.session_state.section
data = load_data()


# ── Add Content ────────────────────────────────────────────────────────────────

if section == "Add Content":
    st.title("Add New Content")
    input_type = st.radio("Choose input method:", ["Enter URL", "Upload File", "Enter Text"])

    if input_type == "Enter URL":
        with st.form("url_form"):
            url = st.text_input("Enter the URL (video, article, or other)")
            title = st.text_input("Enter a short title for this content")
            submit_url = st.form_submit_button("Add Content from URL")

        if submit_url and url.strip() and title.strip():
            with st.spinner("Extracting content..."):
                try:
                    parsed_url = urlparse(url)
                    content = ""

                    if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
                        video_id = extract_video_id(url)
                        if video_id:
                            content = get_youtube_transcript(video_id, title)
                            print("Content after calling a function:", content)
                        else:
                            content = f"YouTube Video - {title}. URL: {url}. Error: Could not extract valid video ID."
                    else:
                        content = extract_article_content(url)

                    if content.strip():
                        new_entry = {"Title": title.strip(), "Source": url.strip(), "Content": content.strip()}
                        if save_data(new_entry):
                            st.success("Content extracted and saved successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to save content")
                    else:
                        st.warning("Could not extract meaningful content from the URL")

                except Exception as e:
                    st.error(f"Error processing URL: {e}")
        elif submit_url:
            st.warning("Please enter both URL and title")

    elif input_type == "Upload File":
        with st.form("file_form"):
            uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])
            title = st.text_input("Enter a short title for this file")
            submit_file = st.form_submit_button("Add Content from File")

        if submit_file and uploaded_file and title.strip():
            with st.spinner("Processing file..."):
                try:
                    source = uploaded_file.name
                    content = ""

                    if uploaded_file.name.endswith(".pdf"):
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        content = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                    elif uploaded_file.name.endswith(".docx"):
                        doc = Document(uploaded_file)
                        content = "\n".join([p.text for p in doc.paragraphs])
                    else:
                        content = uploaded_file.getvalue().decode("utf-8")

                    if content.strip():
                        new_entry = {"Title": title.strip(), "Source": source.strip(), "Content": content.strip()}
                        if save_data(new_entry):
                            st.success("File processed and content saved successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to save content")
                    else:
                        st.warning("File appears to be empty or unreadable")
                except Exception as e:
                    st.error(f"Failed to process file: {e}")
        elif submit_file:
            st.warning("Please enter a title and upload a file")

    elif input_type == "Enter Text":
        with st.form("text_form"):
            title = st.text_input("Enter a short title for this text")
            content = st.text_area("Enter text", height=200)
            submit_text = st.form_submit_button("Add Content from Text")
        if submit_text and title.strip() and content.strip():
            max_chars = 50000
            source = "User Input"
            if len(content) > max_chars:
                st.warning(f"Text is too long ({len(content)} characters). Truncating to {max_chars} characters to"
                           f" prevent processing issues.")
                content = content[:max_chars] + "... [Text truncated due to length]"
            new_entry = {"Title": title.strip(), "Source": source.strip(), "Content": content.strip()}
            if save_data(new_entry):
                st.success("Content added and saved successfully!")
                st.rerun()
            else:
                st.error("Failed to save content")
        elif submit_text:
            st.warning("Please enter both title and content")


# ── Saved Content ──────────────────────────────────────────────────────────────

elif section == "Saved Content":
    st.title("Saved Content")
    if not data.empty:
        display_cols = ["Title", "Source", "Content"] if "Title" in data.columns else data.columns
        st.dataframe(data[display_cols], use_container_width=True)
    else:
        st.info("No content added yet!")


# ── Generate Semantic Graph ────────────────────────────────────────────────────

elif section == "Generate Semantic Graph":
    st.title("Generate Semantic Knowledge Graph")

    if not data.empty:
        extraction_method = st.selectbox(
            "Choose semantic extraction method:", ["spacy", "llm", "combined"],
            help="SpaCy uses dependency parsing, LLM uses Mistral AI, Combined uses both methods"
        )

        with st.expander("Advanced Options"):
            min_confidence = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.3, 0.1)

        if st.button("Generate Semantic Graph", key="generate_semantic_btn"):
            with st.spinner("Extracting semantic relationships..."):
                G = generate_semantic_knowledge_graph(data, method=extraction_method)

                if G.nodes():
                    st.subheader("Interactive Semantic Graph")
                    net = Network(height="800px", width="100%", directed=True,
                                  bgcolor="#ffffff", font_color="#333333")

                    for node_id, node_data in G.nodes(data=True):
                        net.add_node(
                            node_id,
                            label=node_data.get('label', node_id),
                            color=node_data.get('color', '#95A5A6'),
                            size=node_data.get('size', 20),
                            title=f"Type: {node_data.get('entity_type', 'OTHER')}\nCentrality: {node_data.get('centrality', 0):.2f}",
                            font={'size': 14, 'color': '#333333'}
                        )

                    for source, target, edge_data in G.edges(data=True):
                        confidence = edge_data.get('confidence', 0.5)
                        if confidence >= min_confidence:
                            net.add_edge(
                                source, target,
                                label=edge_data.get('label', ''),
                                color=edge_data.get('color', '#95A5A6'),
                                width=max(1, confidence * 5),
                                title=f"Relation: {edge_data.get('relation_type', 'other')}\nConfidence: {confidence:.2f}",
                                arrows={'to': {'enabled': True, 'scaleFactor': 1.2}}
                            )

                    # Graph physics and control system
                    net.set_options("""
                    var options = {
                        "physics": {
                            "enabled": true,
                            "stabilization": {"iterations": 150},
                            "barnesHut": {
                                "gravitationalConstant": -8000,
                                "centralGravity": 0.3,
                                "springLength": 95,
                                "springConstant": 0.04,
                                "damping": 0.09,
                                "avoidOverlap": 0.1
                            }
                        },
                        "nodes": {
                            "borderWidth": 2,
                            "borderWidthSelected": 4
                        },
                        "edges": {
                            "smooth": {
                                "type": "continuous",
                                "forceDirection": "none"
                            }
                        },
                        "interaction": {
                            "hover": true,
                            "tooltipDelay": 200
                        }
                    }
                    """)

                    net.write_html("semantic_graph.html")

                    try:
                        with open("semantic_graph.html", "r") as f:
                            st.components.v1.html(f.read(), height=800)
                    except FileNotFoundError:
                        st.error("Unable to render the graph.")

                    # Display basic statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Entities", G.number_of_nodes())
                    with col2:
                        st.metric("Total Relations", G.number_of_edges())
                    with col3:
                        total_triplets = len(G.graph.get('triplets', []))
                        st.metric("Total Triplets", total_triplets)

                    # Generate and display insights using Mistral
                    st.header("Knowledge Graph Insights")
                    insights = generate_semantic_insights(G, data)
                    st.write(insights)

                else:
                    st.warning("No meaningful semantic relationships could be extracted from the content.")
                    st.info("Try adjusting the confidence threshold or adding more detailed content.")
    else:
        st.warning("No data available to generate semantic graph!")
        st.info("Please add some content first using the 'Add Content' section.")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.sidebar.markdown("Built using Streamlit.")
