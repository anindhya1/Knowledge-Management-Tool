import re

import spacy
import streamlit as st
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from backend.config import ENTITY_TYPES, RELATION_CATEGORIES
from backend.llm_client import generate_insight_mistral

# Load models at module level
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000  # 2 million characters

model = SentenceTransformer('all-MiniLM-L6-v2')
keybert_model = KeyBERT(model='all-MiniLM-L6-v2')


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
