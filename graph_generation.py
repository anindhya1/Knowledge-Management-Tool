import networkx as nx
from pyvis.network import Network
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# NLP Models
model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')

def generate_knowledge_graph(data):
    """Generate a knowledge graph from the data."""
    G = nx.Graph()
    key_phrases = []
    phrase_to_source = {}

    for _, row in data.iterrows():
        phrases = keybert_model.extract_keywords(row["Content"], keyphrase_ngram_range=(1, 2), top_n=10)
        key_phrases.extend([kw[0] for kw in phrases])
        phrase_to_source.update({kw[0]: row["Source"] for kw in phrases})

    embeddings = model.encode(key_phrases)
    similarity_matrix = cosine_similarity(embeddings)

    for i, phrase_i in enumerate(key_phrases):
        G.add_node(phrase_i, label=phrase_i, color="green")
        for j, phrase_j in enumerate(key_phrases):
            if i != j and phrase_to_source[phrase_i] != phrase_to_source[phrase_j] and similarity_matrix[i][j] > 0.4:
                G.add_edge(phrase_i, phrase_j)

    return G

def display_graph(G):
    """Display the graph using PyVis."""
    net = Network(height="700px", width="100%")
    net.from_nx(G)
    net.show("knowledge_graph.html")
    st.components.v1.html(open("knowledge_graph.html", "r").read(), height=700)
