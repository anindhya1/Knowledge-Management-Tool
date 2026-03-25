import networkx as nx

from backend.llm_client import generate_insight_mistral


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
