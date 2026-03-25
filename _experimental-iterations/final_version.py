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
"""

import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from youtube_transcript_api import YouTubeTranscriptApi
from newspaper import Article
from urllib.parse import urlparse, parse_qs
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
nlp = spacy.load("en_core_web_sm")
# Increase the maximum text length limit
nlp.max_length = 2000000  # 2 million characters

# NLP models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

# Initialize data storage
if "knowledge_data.csv" not in os.listdir():
    pd.DataFrame(columns=["Title", "Source", "Content"]).to_csv("knowledge_data.csv", index=False)

# Load existing data
@st.cache_data
def load_data():
    data = pd.read_csv("knowledge_data.csv")
    # Ensure all required columns exist
    required_cols = ["Title", "Source", "Content"]
    for col in required_cols:
        if col not in data.columns:
            data[col] = ""
    return data


def save_data(new_entry):
    """Save new entry to CSV and clear cache"""
    # Load current data
    current_data = pd.read_csv("knowledge_data.csv")

    # Add new entry
    new_row = pd.DataFrame([new_entry])
    updated_data = pd.concat([current_data, new_row], ignore_index=True)

    # Save to CSV
    updated_data.to_csv("knowledge_data.csv", index=False)

    # Clear the cache so data reloads
    load_data.clear()

    return True


# Categories for entities and relations (Creating manually, for now)
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


def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""

    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    elif "youtube.com/watch" in url:
        from urllib.parse import parse_qs
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get('v', [None])[0]
    elif "youtube.com/embed/" in url:
        video_id = url.split("embed/")[-1].split("?")[0]
    else:
        video_id_match = re.search(r'[?&]v=([^&]+)', url)
        video_id = video_id_match.group(1) if video_id_match else None

    if video_id:
        video_id = re.sub(r'[^a-zA-Z0-9_-].*', '', video_id)

    return video_id if video_id and len(video_id) == 11 else None


def get_youtube_transcript(video_id, title):
    """Extract transcript from YouTube video - simplified approach"""
    st.info(f"Extracting transcript for video ID: {video_id}")

    # Simple direct approach - this works for most videos
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id)

    # Combine all transcript text
    transcript_text = " ".join([entry['text'] for entry in transcript_data])

    # Clean up the text
    transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()

    # Format for CSV
    content = f"YouTube Video: {title}\n\nTranscript:\n{transcript_text}"
    content = content.replace('"', "'")  # Escape quotes for CSV

    st.success(f"Successfully extracted transcript ({len(transcript_text)} characters)")
    return content


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

    filtered_triplets = filter_and_deduplicate_triplets(all_triplets)
    st.success(f"Extracted {len(filtered_triplets)} high-quality semantic triplets")

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

        entity_centrality[subject] += 1
        entity_centrality[obj] += 1

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
    triplets.sort(key=lambda x: x.get('confidence', 0.5), reverse=True)

    seen_triplets = set()
    filtered = []

    for triplet in triplets:
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

    relation_stats = G.graph.get('relation_stats', {})
    all_triplets = G.graph.get('triplets', [])

    if relation_stats:
        top_relations = sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)[:3]
        relation_summary = ", ".join([f"{r[0]} ({r[1]} instances)" for r in top_relations])
        insights.append(f"Dominant Relationship Types: {relation_summary}")

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

            llm_insight = generate_insight_mistral(prompt)
            insights.append(f"Strategic Analysis:\n{llm_insight}")


    return "\n\n".join(insights) if insights else "No semantic insights could be generated."


def generate_insight_mistral(prompt):
    """Generate insights using Mistral through Ollama API."""
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
        timeout=1200
    )

    if response.status_code == 200:
        result = response.json()
        return result.get("response", "No response generated").strip()
    else:
        return f"Error: HTTP {response.status_code}"


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
data = load_data()


# Add Content Section
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
                        # Handle non-YouTube links
                        article = Article(url)
                        article.download()
                        article.parse()
                        content = article.text

                    # Save the extracted content to CSV
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
            if len(content) > max_chars:
                st.warning(f"Text is too long ({len(content)} characters). Truncating to {max_chars} characters to"
                           f" prevent processing issues.")
                content = content[:max_chars] + "... [Text truncated due to length]"
                source = "User Input"
            new_entry = {"Title": title.strip(), "Source": source.strip(), "Content": content.strip()}
            if save_data(new_entry):
                st.success("Content added and saved successfully!")
                st.rerun()
            else:
                st.error("Failed to save content")
        elif submit_text:
            st.warning("Please enter both title and content")

# Saved Content Section
elif section == "Saved Content":
    st.title("Saved Content")
    if not data.empty:
        # Ensure all three columns are shown
        display_cols = ["Title", "Source", "Content"] if "Title" in data.columns else data.columns
        st.dataframe(data[display_cols], use_container_width=True)
    else:
        st.info("No content added yet!")


# Generate Semantic Graph Section
elif section == "Generate Semantic Graph":
    st.title("Generate Semantic Knowledge Graph")

    # Choose the method for generating graph
    if not data.empty:
        extraction_method = st.selectbox(
            "Choose semantic extraction method:", ["spacy", "llm", "combined"],
            help="SpaCy uses dependency parsing, LLM uses Mistral AI, Combined uses both methods"
        )

        # Advanced options
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

# Footer
st.sidebar.markdown("Built using Streamlit.")