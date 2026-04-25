"""
Simple PDF Question Answering RAG app with chat history.

Pipeline:
1. User uploads a PDF in the Streamlit UI.
2. PyPDFLoader reads the PDF into LangChain Documents.
3. RecursiveCharacterTextSplitter splits the text into manageable chunks.
4. HuggingFace sentence-transformers/all-MiniLM-L6-v2 generates embeddings.
5. FAISS stores the embeddings for similarity search.
6. The most relevant chunks for the user's question are retrieved.
7. The retrieved context + previous chat turns are sent to a Groq LLM.
8. The LLM's answer is shown and added to the on-screen chat history.
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Prompt template
from langchain_core.output_parsers import StrOutputParser  # Parses LLM output to string
from langchain_core.messages import HumanMessage, AIMessage  # Chat message types


# Load .env so GROQ_API_KEY (and any other env vars) become available
load_dotenv()


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(page_title="PDF Q&A (RAG)", page_icon=":books:")
st.title("PDF Question Answering")
st.caption("Upload a PDF, ask questions, and follow up — answers stay on screen as a chat.")


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


def build_components(vector_store):
    """
    Build the retriever, prompt, LLM, and parser used per question.
    Returns them separately so we can plug in chat history at call time.
    """
    # Step 6: Create a retriever that returns the top-k most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Step 7a: Initialize the Groq chat model (uses GROQ_API_KEY from the environment)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Step 7b: Prompt with a slot for the prior chat history (for follow-up questions)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant answering questions about a PDF. "
                "Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't know. "
                "Use the prior chat history to understand follow-up questions.",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ]
    )

    parser = StrOutputParser()
    return retriever, prompt, llm, parser


def answer_question(question: str):
    """Run the RAG pipeline for one question, using the stored chat history.
    Returns (answer_text, list_of_source_dicts)."""
    retriever = st.session_state.retriever
    prompt = st.session_state.prompt
    llm = st.session_state.llm
    parser = st.session_state.parser

    # Retrieve relevant chunks for THIS question
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build a lightweight, serializable list of sources for display
    sources = []
    for d in docs:
        # PyPDFLoader sets metadata["page"] as a 0-based page index
        page = d.metadata.get("page")
        page_label = (page + 1) if isinstance(page, int) else "?"
        sources.append({"page": page_label, "text": d.page_content})

    # Convert stored UI messages into LangChain message objects for the prompt
    history_messages = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            history_messages.append(HumanMessage(content=m["content"]))
        else:
            history_messages.append(AIMessage(content=m["content"]))

    # Build the chain on the fly and invoke it
    chain = prompt | llm | parser
    answer = chain.invoke(
        {"context": context, "question": question, "history": history_messages}
    )
    return answer, sources


def render_sources(sources):
    """Render the retrieved chunks under an answer as an expandable section."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] Page {s['page']}**")
            st.write(s["text"])
            if i < len(sources):
                st.divider()


# -----------------------------
# Streamlit UI flow
# -----------------------------

# Warn the user early if the Groq API key is missing
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to your environment (.env) and reload.")

# Session state: vector store, chain components, chat history, and current PDF name
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "prompt" not in st.session_state:
    st.session_state.prompt = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "parser" not in st.session_state:
    st.session_state.parser = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}

# Sidebar: PDF upload + clear-chat control
with st.sidebar:
    st.header("PDF")
    # Step 1 (UI): Let the user upload one PDF
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    # Button to wipe the chat history (keeps the indexed PDF)
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# When a new PDF is uploaded, (re)build the vector store and reset chat history
if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
    with st.spinner("Reading PDF, creating embeddings, and indexing..."):
        vector_store, num_chunks = build_vector_store(uploaded_file)
        retriever, prompt, llm, parser = build_components(vector_store)
        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever
        st.session_state.prompt = prompt
        st.session_state.llm = llm
        st.session_state.parser = parser
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.messages = []  # Fresh PDF -> fresh conversation
    st.success(f"Indexed '{uploaded_file.name}' into {num_chunks} chunks.")

# Render the running chat history (every previous turn stays on screen)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        # Show sources for assistant messages that have them attached
        if m["role"] == "assistant" and m.get("sources"):
            render_sources(m["sources"])

# Step 8 (UI): Chat input — only shown once a PDF is indexed
if st.session_state.retriever is not None:
    user_input = st.chat_input("Ask a question about the PDF...")
    if user_input:
        # Show the user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and show the assistant's answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # NOTE: history passed to the LLM is everything BEFORE this new user turn
                history_before = st.session_state.messages[:-1]
                # Temporarily swap so answer_question sees only prior turns as history
                full = st.session_state.messages
                st.session_state.messages = history_before
                try:
                    answer, sources = answer_question(user_input)
                finally:
                    st.session_state.messages = full
            st.markdown(answer)
            render_sources(sources)

        # Persist the assistant's reply (with its sources) in the chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
else:
    st.info("Upload a PDF in the sidebar to start chatting.")
