# Personal Knowledge Management Tool

## Overview
The Personal Knowledge Management (PKM) Tool is a streamlined platform for organizing, visualizing, and extracting meaningful insights from diverse knowledge sources. It enables users to input content from URLs, text files, and other formats to build an interconnected knowledge graph and generate AI-driven insights, making it easier to understand and analyze complex information.

## Features
- **Add Content**: Upload content via URLs, files, or text input.
- **Knowledge Graph**: Automatically generate a visual representation of relationships between concepts extracted from the input data.
- **AI-Powered Insights**: Use advanced machine learning models to generate concise and meaningful insights based on interconnected themes.
- **Interactive UI**: User-friendly interface with a sleek design inspired by modern web aesthetics.

## Technology Stack
- **Frontend**: [Streamlit](https://streamlit.io/) for building an intuitive UI.
- **Backend**: 
  - [Sentence Transformers](https://www.sbert.net/) for semantic embeddings.
  - [KeyBERT](https://github.com/MaartenGr/KeyBERT) for keyword extraction.
  - [Hugging Face Transformers](https://huggingface.co/) for GPT-2-based insight generation.
  - [NetworkX](https://networkx.org/) and [PyVis](https://pyvis.readthedocs.io/) for generating and visualizing knowledge graphs.
- **Storage**: CSV-based local storage for content management.

## Installation
### Prerequisites
- Python 3.8+
- `pip` package manager

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
    ```
4. Run the application:
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
View a list of all added content along with their titles in the "Saved Content" section.

### Generate Connections
1. Navigate to the "Generate Connections" section.
2. Click the button to create a knowledge graph.
3. View the graph and generated AI-driven insights.

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
- Support for real-time collaboration on knowledge graphs.
- Advanced filtering and search functionality for insights.
- Export options for graphs and insights in various formats (PDF, JSON, etc.).
- Integration with external APIs for enhanced content analysis.

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
- [PyVis](https://pyvis.readthedocs.io/)

---

Built with ❤️ using cutting-edge machine learning and NLP technologies.
