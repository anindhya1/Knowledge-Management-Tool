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
import subprocess
import sys


# Load spaCy model for NLP processing
@st.cache_resource
def load_spacy_model():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        # Increase the maximum text length limit
        nlp.max_length = 2000000  # 2 million characters
        return nlp
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        st.info("Some NLP features will be limited.")
        return None

# Load the model
nlp = load_spacy_model()

# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

# Initialize data storage
if "knowledge_data.csv" not in os.listdir():
    pd.DataFrame(columns=["Source", "Content"]).to_csv("knowledge_data.csv", index=False)

# Load existing data
data = pd.read_csv("knowledge_data.csv")

# Semantic categories for entities and relations
ENTITY_TYPES = {
    'PERSON': {'color': '#FF6B6B', 'size': 25},
    'ORG': {'color': '#4ECDC4', 'size': 25},
    'GPE': {'color': '#45B7D1', 'size': 25},
    'PRODUCT': {'color': '#96CEB4', 'size': 25},
    'EVENT': {'color': '#FFEAA7', 'size': 25},
    'CONCEPT': {'color': '#DDA0DD', 'size': 20},
    'OTHER': {'color': '#95A5A6', 'size': 15}
}

RELATION_CATEGORIES = {
    'causal': ['cause', 'lead', 'result', 'create', 'generate', 'produce', 'trigger'],
    'temporal': ['follow', 'precede', 'happen', 'occur', 'begin', 'end', 'start'],
    'spatial': ['locate', 'contain', 'include', 'place', 'position', 'exist'],
    'attribution': ['have', 'own', 'possess', 'belong', 'associate', 'relate'],
    'action': ['do', 'make', 'perform', 'execute', 'implement', 'build'],
    'comparison': ['compare', 'contrast', 'differ', 'similar', 'equal', 'exceed']
}


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


def classify_entity_type(entity_text, doc=None):
    """Classify entity into semantic types using spaCy NER and heuristics."""
    if doc:
        # Use spaCy NER for classification
        for ent in doc.ents:
            if entity_text.lower() in ent.text.lower():
                if ent.label_ in ['PERSON']:
                    return 'PERSON'
                elif ent.label_ in ['ORG']:
                    return 'ORG'
                elif ent.label_ in ['GPE', 'LOC']:
                    return 'GPE'
                elif ent.label_ in ['PRODUCT', 'WORK_OF_ART']:
                    return 'PRODUCT'
                elif ent.label_ in ['EVENT']:
                    return 'EVENT'

    # Heuristic classification
    entity_lower = entity_text.lower()

    # Person indicators
    if any(title in entity_lower for title in ['mr.', 'mrs.', 'dr.', 'prof.', 'ceo', 'president']):
        return 'PERSON'

    # Organization indicators
    if any(org in entity_lower for org in ['company', 'corporation', 'inc', 'ltd', 'university', 'school']):
        return 'ORG'

    # Location indicators
    if any(loc in entity_lower for loc in ['city', 'country', 'state', 'region', 'area']):
        return 'GPE'

    # Product indicators
    if any(prod in entity_lower for prod in ['system', 'tool', 'software', 'application', 'platform']):
        return 'PRODUCT'

    # Event indicators
    if any(event in entity_lower for event in ['meeting', 'conference', 'workshop', 'event', 'session']):
        return 'EVENT'

    # Abstract concept indicators
    if any(concept in entity_lower for concept in ['concept', 'idea', 'theory', 'principle', 'method', 'approach']):
        return 'CONCEPT'

    return 'OTHER'


def classify_relation_type(predicate):
    """Classify relationship into semantic categories."""
    predicate_lower = predicate.lower()

    for category, keywords in RELATION_CATEGORIES.items():
        if any(keyword in predicate_lower for keyword in keywords):
            return category

    return 'other'


def extract_semantic_triplets_spacy(text, max_triplets=50):
    """Extract semantically rich triplets using spaCy with entity typing."""
    max_chunk_size = 100000

    if len(text) > max_chunk_size:
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        st.info(f"Processing large text in {len(chunks)} chunks...")
    else:
        chunks = [text]

    all_triplets = []
    triplets_per_chunk = max_triplets // len(chunks) if len(chunks) > 1 else max_triplets

    for i, chunk in enumerate(chunks):
        try:
            doc = nlp(chunk)
            chunk_triplets = []

            # Extract entities first for better classification
            entities = {ent.text: ent.label_ for ent in doc.ents}

            for sent in doc.sents:
                # Look for more sophisticated patterns
                for token in sent:
                    if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                        predicate = token.lemma_

                        # Find subject with entity information
                        subject = None
                        subject_type = 'OTHER'
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                                subject = get_enhanced_noun_phrase(child)
                                subject_type = classify_entity_type(subject, doc)
                                break

                        # Find object with entity information
                        obj = None
                        obj_type = 'OTHER'
                        for child in token.children:
                            if child.dep_ in ["dobj", "pobj", "attr", "acomp", "xcomp"]:
                                obj = get_enhanced_noun_phrase(child)
                                obj_type = classify_entity_type(obj, doc)
                                break

                        # Enhanced filtering for meaningful triplets
                        if subject and obj and len(subject) > 2 and len(obj) > 2:
                            # Skip trivial relationships
                            if not is_trivial_triplet(subject, predicate, obj):
                                relation_type = classify_relation_type(predicate)

                                triplet = {
                                    'subject': subject,
                                    'predicate': predicate,
                                    'object': obj,
                                    'subject_type': subject_type,
                                    'object_type': obj_type,
                                    'relation_type': relation_type,
                                    'confidence': calculate_triplet_confidence(subject, predicate, obj, entities)
                                }
                                chunk_triplets.append(triplet)

                                if len(chunk_triplets) >= triplets_per_chunk:
                                    break

                if len(chunk_triplets) >= triplets_per_chunk:
                    break

            all_triplets.extend(chunk_triplets)

        except Exception as e:
            st.warning(f"Error processing chunk {i + 1}: {str(e)}")
            continue

    # Sort by confidence and return top triplets
    all_triplets.sort(key=lambda x: x['confidence'], reverse=True)
    return all_triplets[:max_triplets]


def get_enhanced_noun_phrase(token):
    """Extract enhanced noun phrase with better coverage."""
    phrase_tokens = [token]

    # Add modifiers
    for child in token.children:
        if child.dep_ in ["det", "amod", "compound", "prep", "pobj", "nummod", "nmod"]:
            phrase_tokens.append(child)
            # Get prepositional object children
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ in ["pobj", "pcomp"]:
                        phrase_tokens.append(grandchild)

    # Add conjunctions
    for child in token.children:
        if child.dep_ == "conj":
            phrase_tokens.append(child)

    # Sort by position and clean
    phrase_tokens.sort(key=lambda x: x.i)
    phrase = " ".join([t.text for t in phrase_tokens if not t.is_stop or t.pos_ in ["NOUN", "PROPN"]])

    return re.sub(r'\s+', ' ', phrase).strip()


def is_trivial_triplet(subject, predicate, object_text):
    """Filter out trivial or meaningless triplets."""
    trivial_patterns = [
        # Very short or single character elements
        len(subject) < 3 or len(object_text) < 3,
        # Very common but meaningless verbs
        predicate in ['be', 'have', 'do', 'get', 'go', 'say'],
        # Pronouns
        subject.lower() in ['it', 'he', 'she', 'they', 'this', 'that'],
        object_text.lower() in ['it', 'he', 'she', 'they', 'this', 'that'],
        # Numbers or dates only
        subject.isdigit() or object_text.isdigit(),
        # Very generic terms
        subject.lower() in ['thing', 'stuff', 'item', 'way'] or object_text.lower() in ['thing', 'stuff', 'item', 'way']
    ]

    return any(trivial_patterns)


def calculate_triplet_confidence(subject, predicate, obj, entities):
    """Calculate confidence score for triplet quality."""
    confidence = 0.5  # Base confidence

    # Boost for named entities
    if subject in entities:
        confidence += 0.2
    if obj in entities:
        confidence += 0.2

    # Boost for specific verbs
    specific_verbs = ['create', 'develop', 'implement', 'design', 'analyze', 'establish']
    if predicate in specific_verbs:
        confidence += 0.15

    # Boost for longer, more descriptive phrases
    if len(subject.split()) > 1:
        confidence += 0.05
    if len(obj.split()) > 1:
        confidence += 0.05

    return min(confidence, 1.0)


def extract_semantic_triplets_llm(text, max_triplets=30):
    """Extract semantically typed triplets using LLM."""
    max_chunk_size = 3000
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    all_triplets = []
    triplets_per_chunk = max(1, max_triplets // min(len(chunks), 5))

    for i, chunk in enumerate(chunks[:5]):
        prompt = f"""Extract meaningful subject-predicate-object triplets from the text below. 
For each triplet, also classify the subject and object as: PERSON, ORGANIZATION, LOCATION, CONCEPT, PRODUCT, EVENT, or OTHER.

Format: Subject | Subject_Type | Predicate | Object | Object_Type

Focus on:
- Causal relationships (X causes Y, X leads to Y)
- Actions and their outcomes
- Associations between entities
- Temporal relationships
- Hierarchical relationships

Avoid generic verbs like "is", "has", "are". Prefer specific, meaningful relationships.
Limit to {triplets_per_chunk} triplets.

Text: {chunk}

Semantic Triplets:"""

        try:
            response = generate_insight_mistral(prompt)
            lines = response.split('\n')
            chunk_triplets = []

            for line in lines:
                if '|' in line and not line.strip().startswith('#'):
                    parts = [part.strip() for part in line.split('|')]
                    if len(parts) == 5 and all(len(part) > 1 for part in parts):
                        triplet = {
                            'subject': parts[0],
                            'predicate': parts[2],
                            'object': parts[3],
                            'subject_type': parts[1].upper(),
                            'object_type': parts[4].upper(),
                            'relation_type': classify_relation_type(parts[2]),
                            'confidence': 0.8  # Higher confidence for LLM-extracted triplets
                        }
                        chunk_triplets.append(triplet)
                        if len(chunk_triplets) >= triplets_per_chunk:
                            break

            all_triplets.extend(chunk_triplets)

        except Exception as e:
            st.warning(f"Error extracting semantic triplets with LLM for chunk {i + 1}: {e}")
            continue

    return all_triplets[:max_triplets]


def generate_semantic_knowledge_graph(data, method="spacy"):
    """Generate semantically enhanced knowledge graph."""
    G = nx.DiGraph()
    all_triplets = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_rows = len(data)

    for index, row in data.iterrows():
        status_text.text(f"Processing content {index + 1} of {total_rows}...")
        progress_bar.progress((index + 1) / total_rows)

        try:
            if method == "spacy":
                triplets = extract_semantic_triplets_spacy(row["Content"])
            elif method == "llm":
                triplets = extract_semantic_triplets_llm(row["Content"])
            else:
                # Combine both methods
                spacy_triplets = extract_semantic_triplets_spacy(row["Content"], max_triplets=25)
                llm_triplets = extract_semantic_triplets_llm(row["Content"], max_triplets=25)
                triplets = spacy_triplets + llm_triplets

            # Add source information
            for triplet in triplets:
                triplet['source'] = row["Source"]
                all_triplets.append(triplet)

        except Exception as e:
            st.warning(f"Error processing content from {row['Source']}: {str(e)}")
            continue

    progress_bar.empty()
    status_text.empty()

    # Filter and deduplicate triplets
    filtered_triplets = filter_and_deduplicate_triplets(all_triplets)
    st.success(f"Extracted {len(filtered_triplets)} high-quality semantic triplets")

    # Build enhanced graph
    entity_centrality = defaultdict(int)
    relation_stats = defaultdict(int)

    for triplet in filtered_triplets:
        subject = triplet['subject']
        obj = triplet['object']
        predicate = triplet['predicate']

        # Add nodes with semantic information
        subject_type = triplet.get('subject_type', 'OTHER')
        obj_type = triplet.get('object_type', 'OTHER')

        if not G.has_node(subject):
            node_props = ENTITY_TYPES.get(subject_type, ENTITY_TYPES['OTHER']).copy()
            node_props.update({
                'label': subject,
                'entity_type': subject_type,
                'centrality': 0
            })
            G.add_node(subject, **node_props)

        if not G.has_node(obj):
            node_props = ENTITY_TYPES.get(obj_type, ENTITY_TYPES['OTHER']).copy()
            node_props.update({
                'label': obj,
                'entity_type': obj_type,
                'centrality': 0
            })
            G.add_node(obj, **node_props)

        # Update centrality
        entity_centrality[subject] += 1
        entity_centrality[obj] += 1

        # Add edge with semantic information
        relation_type = triplet.get('relation_type', 'other')
        confidence = triplet.get('confidence', 0.5)

        edge_props = {
            'label': predicate,
            'relation_type': relation_type,
            'confidence': confidence,
            'weight': confidence,
            'color': get_relation_color(relation_type),
            'source': triplet['source']
        }

        if G.has_edge(subject, obj):
            # Combine multiple relations
            existing_label = G[subject][obj]['label']
            G[subject][obj]['label'] = f"{existing_label}; {predicate}"
            G[subject][obj]['weight'] += confidence
        else:
            G.add_edge(subject, obj, **edge_props)

        relation_stats[relation_type] += 1

    # Update node sizes based on centrality
    max_centrality = max(entity_centrality.values()) if entity_centrality else 1
    for node in G.nodes():
        centrality_score = entity_centrality[node] / max_centrality
        G.nodes[node]['size'] = 15 + (centrality_score * 25)  # Size between 15-40
        G.nodes[node]['centrality'] = centrality_score

    # Store metadata for insights
    G.graph['triplets'] = filtered_triplets
    G.graph['relation_stats'] = dict(relation_stats)
    G.graph['entity_types'] = {node: data['entity_type'] for node, data in G.nodes(data=True)}

    return G


def filter_and_deduplicate_triplets(triplets):
    """Filter and deduplicate triplets for quality."""
    # Sort by confidence
    triplets.sort(key=lambda x: x.get('confidence', 0.5), reverse=True)

    seen_triplets = set()
    filtered = []

    for triplet in triplets:
        # Create a normalized key for deduplication
        key = (
            triplet['subject'].lower().strip(),
            triplet['predicate'].lower().strip(),
            triplet['object'].lower().strip()
        )

        if key not in seen_triplets:
            seen_triplets.add(key)
            filtered.append(triplet)

    return filtered


def get_relation_color(relation_type):
    """Get color for relation type."""
    colors = {
        'causal': '#E74C3C',  # Red for causation
        'temporal': '#3498DB',  # Blue for time
        'spatial': '#2ECC71',  # Green for space
        'attribution': '#F39C12',  # Orange for ownership
        'action': '#9B59B6',  # Purple for actions
        'comparison': '#1ABC9C',  # Teal for comparison
        'other': '#95A5A6'  # Gray for other
    }
    return colors.get(relation_type, '#95A5A6')


def generate_semantic_insights(G, data):
    """Generate enhanced insights from semantic knowledge graph using Mistral."""
    insights = []

    # Get stored data
    relation_stats = G.graph.get('relation_stats', {})
    all_triplets = G.graph.get('triplets', [])

    # Basic statistics
    if relation_stats:
        top_relations = sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)[:3]
        relation_summary = ", ".join([f"{r[0]} ({r[1]} instances)" for r in top_relations])
        insights.append(f"Dominant Relationship Types: {relation_summary}")

    # High-centrality entities
    if G.nodes():
        degree_centrality = nx.degree_centrality(G)
        top_entities = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        entity_list = ", ".join([f"'{e[0]}'" for e in top_entities])
        insights.append(f"Most Connected Entities: {entity_list}")

    # Generate LLM insights using Mistral through Ollama
    if all_triplets:
        high_conf_triplets = [t for t in all_triplets if t.get('confidence', 0) > 0.7][:10]
        if high_conf_triplets:
            triplet_text = "\n".join([
                f"- {t['subject']} → {t['predicate']} → {t['object']}"
                for t in high_conf_triplets
            ])
            print(triplet_text)
            prompt = f"""You are my personal assistant examining knowledge relationships. Analyze these high-confidence 
            triplets and provide concise strategic insights about the underlying patterns, themes, or implications.

Knowledge Relationships:
{triplet_text}

Focus on:
1. What are some of the deeper insights you gather from the various contexts I added.
2. Is there an underlying semantic logic to why I'm consuming this content? If there is, what are some actions I can 
take to boost those motives?
3. Further are there any deeper links, comparisons or relations you can make here which I otherwise might have missed out on?
If so, please explain what they are and why. Try to make interlinked connections between the contexts you pull from the 
Subjects, Predicates and Objects present in triplet_text.

Provide your analysis in 2-3 clear, actionable paragraphs."""

            try:
                llm_insight = generate_insight_mistral(prompt)
                insights.append(f"Strategic Analysis:\n{llm_insight}")
            except Exception as e:
                insights.append(f"Strategic Analysis: Could not generate LLM insights - {str(e)}")

    return "\n\n".join(insights) if insights else "No semantic insights could be generated."


def generate_insight_mistral(prompt):
    """Generate insights using Mistral through Ollama API."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,  # Set to False for simpler handling
                "options": {
                    "num_predict": 400,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated").strip()
        else:
            return f"Error: HTTP {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Please ensure Ollama is running and Mistral model is available."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model may be processing a large request."
    except Exception as e:
        return f"Error: {str(e)}"


# Streamlit App UI
if "section" not in st.session_state:
    st.session_state.section = "Add Content"

# Navigation bar
if st.sidebar.button("Add Content", key="nav_add_content"):
    st.session_state.section = "Add Content"
if st.sidebar.button("Saved Content", key="nav_saved_content"):
    st.session_state.section = "Saved Content"
if st.sidebar.button("Generate Connections", key="nav_generate_graph"):
    st.session_state.section = "Generate Semantic Graph"

section = st.session_state.section

# Add Content Section
if section == "Add Content":
    st.title("Add New Content")
    input_type = st.radio("Choose input method:", ["Enter URL", "Upload File", "Enter Text"])

    content = ""
    source = ""

    if input_type == "Enter URL":
        url = st.text_input("Enter the URL (video, article, or other)")
        if st.button("Add Content from URL", key="add_url_content"):
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
        content = st.text_area("Enter text", height=200)
        if st.button("Add Content from Text", key="add_text_content"):
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
        st.dataframe(data, use_container_width=True)
    else:
        st.info("No content added yet!")

# Generate Semantic Graph Section
elif section == "Generate Semantic Graph":
    st.title("Generate Semantic Knowledge Graph")

    if not data.empty:
        # Method selection
        extraction_method = st.selectbox(
            "Choose semantic extraction method:",
            ["spacy", "llm", "combined"],
            help="SpaCy uses dependency parsing, LLM uses Mistral AI, Combined uses both methods"
        )

        # Advanced options
        with st.expander("Advanced Options"):
            min_confidence = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.3, 0.1)

        if st.button("Generate Semantic Graph", key="generate_semantic_btn"):
            with st.spinner("Extracting semantic relationships..."):
                G = generate_semantic_knowledge_graph(data, method=extraction_method)

                if G.nodes():
                    # Create enhanced visualization
                    st.subheader("Interactive Semantic Graph")
                    net = Network(height="800px", width="100%", directed=True,
                                  bgcolor="#ffffff", font_color="#333333")

                    # Add nodes with semantic styling
                    for node_id, node_data in G.nodes(data=True):
                        net.add_node(
                            node_id,
                            label=node_data.get('label', node_id),
                            color=node_data.get('color', '#95A5A6'),
                            size=node_data.get('size', 20),
                            title=f"Type: {node_data.get('entity_type', 'OTHER')}\nCentrality: {node_data.get('centrality', 0):.2f}",
                            font={'size': 14, 'color': '#333333'}
                        )

                    # Add edges with semantic information
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

                    # Enhanced physics and layout
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

                    # # Show sample semantic triplets
                    # if st.checkbox("Show Sample Semantic Triplets", key="show_semantic_triplets"):
                    #     sample_triplets = G.graph.get('triplets', [])[:20]
                    #     st.subheader("Sample Extracted Semantic Triplets")
                    #
                    #     for i, triplet in enumerate(sample_triplets, 1):
                    #         confidence = triplet.get('confidence', 0.5)
                    #         relation_type = triplet.get('relation_type', 'other')
                    #         subject_type = triplet.get('subject_type', 'OTHER')
                    #         object_type = triplet.get('object_type', 'OTHER')
                    #
                    #         confidence_indicator = "High" if confidence > 0.7 else "Med" if confidence > 0.5 else "Low"
                    #
                    #         st.write(
                    #             f"{i}. [{confidence_indicator}] **{triplet['subject']}** ({subject_type}) "
                    #             f"→ *{triplet['predicate']}* ({relation_type}) → "
                    #             f"**{triplet['object']}** ({object_type}) "
                    #             f"[{confidence:.2f}]"
                    #         )

                    # # Export options
                    # if st.checkbox("Export Graph Data", key="export_data"):
                    #     st.subheader("Export Options")
                    #
                    #     col1, col2 = st.columns(2)
                    #
                    #     with col1:
                    #         # Export triplets as CSV
                    #         triplets_df = pd.DataFrame(G.graph.get('triplets', []))
                    #         if not triplets_df.empty:
                    #             csv_data = triplets_df.to_csv(index=False)
                    #             st.download_button(
                    #                 label="Download Triplets as CSV",
                    #                 data=csv_data,
                    #                 file_name="semantic_triplets.csv",
                    #                 mime="text/csv"
                    #             )
                    #
                    #     with col2:
                    #         # Export graph as JSON
                    #         import json
                    #         from networkx.readwrite import json_graph
                    #
                    #         graph_data = json_graph.node_link_data(G)
                    #         json_data = json.dumps(graph_data, indent=2)
                    #         st.download_button(
                    #             label="Download Graph as JSON",
                    #             data=json_data,
                    #             file_name="semantic_graph.json",
                    #             mime="application/json"
                    #         )
                else:
                    st.warning("No meaningful semantic relationships could be extracted from the content.")
                    st.info("Try adjusting the confidence threshold or adding more detailed content.")
    else:
        st.warning("No data available to generate semantic graph!")
        st.info("Please add some content first using the 'Add Content' section.")

# Footer
st.sidebar.markdown("Built using Streamlit.")