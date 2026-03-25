CSV_FILE = "knowledge_data.csv"

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
