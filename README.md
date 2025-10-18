# 📚 Smart Document Q&A

An intelligent document question-answering system built with **Retrieval-Augmented Generation (RAG)** technology. Upload your PDF or text documents and ask questions - get accurate, context-aware answers powered by AI, all running **100% locally** on your machine.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 📄 **Multi-format Support**: Upload PDF or TXT documents
- 🤖 **AI-Powered Answers**: Get intelligent responses using Mistral-7B-Instruct
- 🔍 **Semantic Search**: Advanced vector-based document retrieval with FAISS
- 💬 **Interactive Chat**: Natural conversation interface
- 📊 **Source Citations**: See which parts of your document support each answer
- 🔒 **Privacy First**: All processing happens locally - no data leaves your machine
- ⚡ **Fast & Efficient**: Optimized chunking and embedding generation

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Mistral-7B-Instruct (via Ollama) |
| **Embeddings** | all-MiniLM-L6-v2 (SentenceTransformers) |
| **Vector Database** | FAISS |
| **Framework** | LangChain |
| **UI** | Streamlit |
| **Document Processing** | PyPDF, LangChain Text Splitters |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-document-qa.git
cd smart-document-qa
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install and setup Ollama**

Download from [ollama.ai](https://ollama.ai/download) and install, then pull the model:
```bash
ollama pull mistral:7b-instruct
```

### Running the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload Document**: Click "Choose a PDF or TXT file" in the sidebar
2. **Process**: Click "Process Document" button
3. **Ask Questions**: Type your question in the chat input
4. **View Sources**: Expand the sources section to see relevant document excerpts
5. **New Document**: Click "Clear Document" to load a different file

## 🏗️ Project Structure
```
smart-document-qa/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration parameters
│
├── src/
│   ├── __init__.py
│   ├── document_processor.py      # PDF/text loading and chunking
│   ├── embeddings.py               # Embedding generation
│   ├── vector_store.py             # FAISS vector database operations
│   ├── llm_handler.py              # LLM interaction (Ollama)
│   ├── retriever.py                # Document retrieval logic
│   └── rag_chain.py                # Complete RAG pipeline
│
├── utils/
│   ├── __init__.py
│   └── prompt_templates.py         # Prompt templates for Q&A
│
└── data/
    ├── uploads/                    # Temporary uploaded files
    └── vector_db/                  # Stored FAISS indices
```

## ⚙️ Configuration

Edit `config/settings.py` to customize:

- **Chunk Size**: Adjust document splitting (default: 800 tokens)
- **Top-K Results**: Number of relevant chunks to retrieve (default: 4)
- **LLM Model**: Change the Ollama model
- **Temperature**: Control response creativity (default: 0.1)
- **Max File Size**: Maximum upload size (default: 50MB)

## 🧪 Testing Individual Components

Each component can be tested independently:
```bash
# Test document processor
python src/document_processor.py

# Test embeddings
python src/embeddings.py

# Test vector store
python src/vector_store.py

# Test LLM handler
python src/llm_handler.py

# Test retriever
python src/retriever.py

# Test complete RAG chain
python src/rag_chain.py
```

## 🎯 How It Works

1. **Document Processing**: Documents are split into overlapping chunks for better context preservation
2. **Embedding Generation**: Each chunk is converted to a 384-dimensional vector using SentenceTransformers
3. **Vector Storage**: Embeddings are stored in FAISS for efficient similarity search
4. **Query Processing**: User questions are embedded and matched against document chunks
5. **Answer Generation**: Retrieved context is sent to Mistral-7B-Instruct to generate accurate answers

## 📊 Performance

- **Document Processing**: ~1-2 seconds per page
- **Embedding Generation**: ~0.1 seconds per chunk
- **Query Response**: ~2-5 seconds depending on document size
- **Memory Usage**: ~2-4GB RAM for typical documents

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anthropic Claude](https://www.anthropic.com/) for development assistance
- [Ollama](https://ollama.ai/) for local LLM inference
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io/) for the UI framework

## 📧 Contact

Chinmay Mendse - mendsechinmay@gmail.com
---

⭐ If you find this project useful, please consider giving it a star!
