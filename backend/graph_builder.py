from collections import defaultdict

import networkx as nx
import streamlit as st

from backend.config import ENTITY_TYPES
from backend.nlp_processor import (
    extract_semantic_triplets_spacy,
    extract_semantic_triplets_llm,
    filter_and_deduplicate_triplets,
)


def get_relation_color(relation_type):
    """Get color for relation type."""
    colors = {
        'causal': '#E74C3C',      # Red for causation
        'temporal': '#3498DB',    # Blue for time
        'spatial': '#2ECC71',     # Green for space
        'attribution': '#F39C12', # Orange for ownership
        'action': '#9B59B6',      # Purple for actions
        'comparison': '#1ABC9C',  # Teal for comparison
        'other': '#95A5A6'        # Gray for other
    }
    return colors.get(relation_type, '#95A5A6')


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
