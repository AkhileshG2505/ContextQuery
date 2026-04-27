# ContextQuery: Document-Aware Conversational AI

A document-aware conversational chatbot built with Python, enabling users to interactively query multiple PDF files. This application utilizes **Retrieval-Augmented Generation (RAG)** to provide highly accurate, context-based responses without hallucination.

## 🚀 Features
* **Multi-PDF Processing:** Upload and parse multiple PDF documents simultaneously.
* **Semantic Search:** Utilizes LangChain and ChromaDB for efficient text chunking and vector-based retrieval.
* **Contextual AI Responses:** Integrated with the Google Gemini API to synthesize accurate answers grounded purely in the uploaded documents.
* **Interactive UI:** A clean, responsive frontend built entirely in Python using Streamlit.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Core Logic:** Python, LangChain
* **Vector Database:** ChromaDB
* **LLM:** Google Gemini API
* **Document Parsing:** PyPDF

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/doc-query-rag.git](https://github.com/yourusername/doc-query-rag.git)
   cd doc-query-rag