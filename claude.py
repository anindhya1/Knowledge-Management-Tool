# Import statements
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="Knowledge Graph")


class KnowledgeGraph:
    def __init__(self, encoder_model, keybert_model, spacy_model):
        self.encoder = encoder_model
        self.keybert = keybert_model
        self.spacy = spacy_model

    def extract_entities_and_concepts(self, text):
        """Extract meaningful entities and concepts from text."""
        doc = self.spacy(text)

        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc if ent.label_ in ['ORG', 'PERSON', 'GPE', 'CONCEPT']]

        # Extract key phrases using KeyBERT
        key_phrases = self.keybert.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_maxsum=True,
            nr_candidates=20,
            top_n=10
        )

        return entities, [phrase[0] for phrase in key_phrases]

    def build_graph(self, data):
        """Build knowledge graph with meaningful connections."""
        G = nx.Graph()
        all_concepts = []
        concept_to_source = {}
        concept_to_context = {}

        # First pass: Extract concepts and their context
        for idx, row in data.iterrows():
            # Check if the content is string, if not convert it
            content = str(row["Content"]) if not isinstance(row["Content"], str) else row["Content"]
            entities, concepts = self.extract_entities_and_concepts(content)

            # Store concepts with their context
            for concept in concepts:
                all_concepts.append(concept)
                concept_to_source[concept] = row["Source"]

                # Extract relevant context around the concept
                context = self._extract_context(content, concept)
                concept_to_context[concept] = context

        if not all_concepts:
            st.warning("No concepts were extracted from the data. Please check your content.")
            return None

        # Calculate semantic similarities
        embeddings = self.encoder.encode(all_concepts)
        similarity_matrix = cosine_similarity(embeddings)

        # Build graph with meaningful connections
        threshold = 0.6  # Higher threshold for more meaningful connections
        for i, concept_i in enumerate(all_concepts):
            G.add_node(
                concept_i,
                label=concept_i,
                context=concept_to_context[concept_i],
                source=concept_to_source[concept_i]
            )

            # Find meaningful connections
            for j, concept_j in enumerate(all_concepts):
                if i != j and similarity_matrix[i][j] > threshold:
                    # Only connect concepts from different sources
                    if concept_to_source[concept_i] != concept_to_source[concept_j]:
                        G.add_edge(
                            concept_i,
                            concept_j,
                            weight=similarity_matrix[i][j]
                        )

        return G

    def _extract_context(self, text, concept, window_size=100):
        """Extract relevant context around a concept."""
        concept_pos = text.lower().find(concept.lower())
        if concept_pos == -1:
            return ""

        start = max(0, concept_pos - window_size)
        end = min(len(text), concept_pos + len(concept) + window_size)
        return text[start:end].strip()

    def generate_insights(self, G):
        """Generate meaningful insights from the graph."""
        if G is None or G.number_of_nodes() == 0:
            return []

        insights = []

        # Find important concepts using centrality measures
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)

        # Combine centrality measures
        important_concepts = {
            node: 0.7 * degree_cent[node] + 0.3 * betweenness_cent[node]
            for node in G.nodes()
        }

        # Find connected components (concept clusters)
        components = list(nx.connected_components(G))

        for component in components:
            if len(component) < 2:
                continue

            # Find the most central concept in the component
            central_concept = max(
                component,
                key=lambda x: important_concepts[x]
            )

            # Get related concepts and their contexts
            related = [
                (neighbor, G.nodes[neighbor]['context'])
                for neighbor in G.neighbors(central_concept)
                if neighbor in component
            ]

            # Generate insight based on the concept cluster
            insight = {
                'central_concept': central_concept,
                'central_context': G.nodes[central_concept]['context'],
                'related_concepts': related,
                'sources': list(set(
                    G.nodes[n]['source'] for n in component
                ))
            }

            insights.append(insight)

        return insights


def render_knowledge_graph(kg, data):
    """Render the Knowledge Graph page"""
    st.title("Knowledge Graph")

    if data.empty:
        st.warning("No content available in the CSV file.")
        return

    with st.spinner("Generating knowledge graph..."):
        G = kg.build_graph(data)

        if G is None or G.number_of_nodes() == 0:
            st.warning("Could not generate graph from the data.")
            return

        # Create interactive visualization
        net = Network(height="600px", width="100%", bgcolor="#ffffff")

        # Add nodes and edges with custom styling
        for node in G.nodes():
            net.add_node(
                node,
                label=node,
                title=G.nodes[node]['context'][:100] + "...",
                color="#6B9080"
            )

        for edge in G.edges():
            net.add_edge(
                edge[0],
                edge[1],
                weight=G.edges[edge]['weight'] * 5,
                color="#A4C3B2"
            )

        # Save and display
        net.write_html("knowledge_graph.html")
        with open("knowledge_graph.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600)


def render_insights(kg, data):
    """Render the Insights page"""
    st.title("Generated Insights")

    if data.empty:
        st.warning("No content available in the CSV file.")
        return

    with st.spinner("Generating insights..."):
        G = kg.build_graph(data)

        if G is None or G.number_of_nodes() == 0:
            st.warning("Could not generate insights from the data.")
            return

        insights = kg.generate_insights(G)

        if not insights:
            st.warning("No meaningful insights could be generated from the data.")
            return

        for i, insight in enumerate(insights, 1):
            with st.expander(f"Insight {i}: {insight['central_concept']}", expanded=True):
                st.write("**Central Theme:**", insight['central_concept'])
                st.write("**Context:**", insight['central_context'])

                st.write("**Related Concepts:**")
                for concept, context in insight['related_concepts']:
                    st.markdown(f"- **{concept}**")
                    st.markdown(f"  *Context:* {context}")

                st.write("**Sources:**", ", ".join(insight['sources']))


def main():
    try:
        # Load the existing CSV file
        data = pd.read_csv("knowledge_data.csv")
        st.sidebar.success("Successfully loaded knowledge_data.csv")
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return

    # Initialize models
    with st.spinner("Loading models..."):
        encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
        keybert_model = KeyBERT(model='all-MiniLM-L6-v2')
        spacy_model = spacy.load('en_core_web_sm')

    # Initialize Knowledge Graph
    kg = KnowledgeGraph(encoder_model, keybert_model, spacy_model)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Knowledge Graph", "Insights"])

    # Show data preview
    if st.sidebar.checkbox("Show Data Preview"):
        st.sidebar.dataframe(data.head())

    # Render appropriate page
    if page == "Knowledge Graph":
        render_knowledge_graph(kg, data)
    else:
        render_insights(kg, data)


if __name__ == "__main__":
    main()