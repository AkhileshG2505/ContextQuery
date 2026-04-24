"""
Simple PDF Question Answering RAG app.

Pipeline:
1. User uploads a PDF in the Streamlit UI.
2. PyPDFLoader reads the PDF into LangChain Documents.
3. RecursiveCharacterTextSplitter splits the text into manageable chunks.
4. HuggingFace sentence-transformers/all-MiniLM-L6-v2 generates embeddings.
5. FAISS stores the embeddings for similarity search.
6. The most relevant chunks for the user's question are retrieved.
7. The retrieved context is sent to a Groq LLM along with the question.
8. The LLM's answer is displayed in the Streamlit UI.
"""

# Standard library imports
import os
import tempfile

# Load environment variables from .env (e.g. GROQ_API_KEY)
from dotenv import load_dotenv

# Streamlit for the web UI
import streamlit as st

# LangChain v1 compatible imports
from langchain_community.document_loaders import PyPDFLoader  # PDF loader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text into chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Embedding model wrapper
from langchain_community.vectorstores import FAISS  # In-memory vector store
from langchain_groq import ChatGroq  # Groq chat model
from langchain_core.prompts import ChatPromptTemplate  # Prompt template
from langchain_core.output_parsers import StrOutputParser  # Parses LLM output to string
from langchain_core.runnables import RunnablePassthrough  # Passes input through a chain step


# Load .env so GROQ_API_KEY (and any other env vars) become available
load_dotenv()


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(page_title="PDF Q&A (RAG)", page_icon=":books:")
st.title("PDF Question Answering")
st.caption("Upload a PDF, ask a question, and get an answer powered by RAG + Groq.")


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """
    Build the embeddings model once and cache it.
    Uses sentence-transformers/all-MiniLM-L6-v2 — a small, fast embedding model.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(uploaded_file):
    """
    Take an uploaded PDF, read it, split it into chunks, embed it,
    and return a FAISS vector store ready for similarity search.
    """
    # Step 1: Save the uploaded PDF to a temporary file so PyPDFLoader can read it from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Step 2: Read the PDF with PyPDFLoader (one Document per page)
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Clean up the temporary file once loaded
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    # Step 3: Split the text into ~1000-char chunks with overlap so context is not lost between chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # Step 4: Generate embeddings for each chunk using the cached embeddings model
    embeddings = get_embeddings()

    # Step 5: Store the embeddings in a FAISS vector store for fast similarity search
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks)


def format_docs(docs):
    """Join retrieved Document chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vector_store):
    """
    Build a simple RAG chain:
    retriever -> prompt -> Groq LLM -> string output.
    """
    # Step 6: Create a retriever that returns the top-k most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Step 7a: Initialize the Groq chat model (uses GROQ_API_KEY from the environment)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Step 7b: Define a simple prompt that grounds the answer in the retrieved context
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant answering questions about a PDF. "
                "Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't know.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ]
    )

    # Step 7c: Compose the chain using LangChain Runnables
    # - "context" is filled by retrieving relevant docs and formatting them
    # - "question" is passed straight through
    # - The prompt is sent to the Groq LLM
    # - StrOutputParser returns a plain string answer
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# -----------------------------
# Streamlit UI flow
# -----------------------------

# Warn the user early if the Groq API key is missing
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to your environment (.env) and reload.")

# Use Streamlit session state to keep the vector store and chain between reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Step 1 (UI): Let the user upload one PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# When a new PDF is uploaded, (re)build the vector store and chain
if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
    with st.spinner("Reading PDF, creating embeddings, and indexing..."):
        vector_store, num_chunks = build_vector_store(uploaded_file)
        st.session_state.vector_store = vector_store
        st.session_state.chain = build_qa_chain(vector_store)
        st.session_state.pdf_name = uploaded_file.name
    st.success(f"Indexed '{uploaded_file.name}' into {num_chunks} chunks.")

# Step 8 (UI): Ask a question and show the answer
if st.session_state.chain is not None:
    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)
        st.subheader("Answer")
        st.write(answer)
else:
    st.info("Upload a PDF to get started.")
