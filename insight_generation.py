import networkx as nx

def generate_insights(G, data):
    """Generate insights from the graph."""
    insights = []
    degree_centrality = nx.degree_centrality(G)
    avg_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    threshold = avg_centrality * 1.5

    crux_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]

    for crux_node in crux_nodes[:3]:
        neighbors = list(G.neighbors(crux_node))
        related_topics = ", ".join(neighbors[:3])
        insights.append(f"The concept '{crux_node}' connects to themes such as {related_topics}.")

    if not insights:
        return "No significant patterns or insights could be generated."

    return "\n\n".join(insights)
