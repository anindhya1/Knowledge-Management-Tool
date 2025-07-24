# Personal Knowledge Management Tool

## Overview  
The Personal Knowledge Management (PKM) Tool is a streamlined platform for organizing, visualizing, and extracting meaningful insights from diverse knowledge sources. It enables users to input content from URLs, text files, and other formats to build an interconnected knowledge graph and generate AI-driven insights, making it easier to understand and analyze complex information.

This enhanced version supports semantic aliasing of key phrases, relation triplet extraction via REBEL, and locally hosted LLMs (e.g., Mistral via Ollama) for context-aware insight generation.

## Features
- **Add Content**: Upload content via URLs, files, or text input.
- **Key Phrase Extraction**: Extracts concepts using KeyBERT and sentence embeddings.
- **Semantic Aliasing**: Clusters semantically similar phrases using DBSCAN and assigns canonical aliases.
- **Relation Extraction**: Uses REBEL model to extract (subject, relation, object) triplets from context sentences.
- **Knowledge Graph Generation**: Builds interactive knowledge graphs combining semantic and relational edges.
- **AI-Powered Insights**: Uses local models like Mistral (via Ollama) to generate strategic insights based on graph structure and content.
- **Interactive UI**: User-friendly interface with a sleek design inspired by modern web aesthetics.

## Technology Stack
- **Frontend**: [Streamlit](https://streamlit.io/) for building an intuitive UI.
- **Backend**:
  - [Sentence Transformers](https://www.sbert.net/) for semantic embeddings.
  - [KeyBERT](https://github.com/MaartenGr/KeyBERT) for keyword extraction.
  - [REBEL (Babelscape)](https://huggingface.co/Babelscape/rebel-large) for relation extraction.
  - [spaCy](https://spacy.io/) and [nltk](https://www.nltk.org/) for lightweight NLP.
  - [Mistral](https://ollama.com/library/mistral) served locally using [Ollama](https://ollama.com/) for insight generation.
  - [NetworkX](https://networkx.org/) and [PyVis](https://pyvis.readthedocs.io/) for generating and visualizing knowledge graphs.
- **Storage**: CSV-based local storage for content management.

## Installation
### Prerequisites
- Python 3.8+
- `pip` package manager
- [Ollama](https://ollama.com/) installed locally (for LLM support)

### Steps
1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/pkm-tool.git
    cd pkm-tool
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    python -m nltk.downloader punkt
    python -m spacy download en_core_web_sm
    ```

4. Start the Mistral model locally:

    ```bash
    ollama run mistral
    ```

5. Run the application:

    ```bash
    streamlit run app.py
    ```

## Usage
### Add Content
1. Select "Add Content" from the sidebar.
2. Choose an input method:
   - **URL**: Enter a link to a YouTube video or an article.
   - **File Upload**: Upload a `.txt`, `.pdf`, or `.docx` file.
   - **Text Input**: Enter custom text.
3. Click the corresponding button to process and save the content.

### Saved Content
View a list of all added content in the "Saved Content" section.

### Generate Connections
1. Navigate to the "Generate Connections" section.
2. Click the button to create a knowledge graph.
3. View the graph, explore semantic connections and REBEL triplets, and review AI-generated insights.

## File Structure
```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── knowledge_data.csv   # Local storage for added content
├── README.md            # Project documentation
└── assets/              # Static assets (e.g., images, icons)
```


## Future Enhancements
- Allow toggling between graph modes: keyphrase similarity, alias-only, and REBEL-only.
- Improve REBEL triplet decoding and filtering.
- Support real-time collaboration on graph editing.
- Advanced filtering and semantic search for insights.
- Export options for graphs and insights (PDF, JSON, CSV).
- Optional support for external APIs (e.g., OpenAI, Cohere).

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.

2. Create a new branch for your feature/bugfix:

    ```bash
    git checkout -b feature-name
    ```

3. Commit your changes:

    ```bash
    git commit -m "Add feature description"
    ```

4. Push the branch:

    ```bash
    git push origin feature-name
    ```

5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [Sentence Transformers](https://www.sbert.net/)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- [REBEL by Babelscape](https://huggingface.co/Babelscape/rebel-large)
- [PyVis](https://pyvis.readthedocs.io/)
- [Ollama](https://ollama.com/)

---

## Author

Developed by [Anindhya Kushagra](https://github.com/anindhyakushagra)
