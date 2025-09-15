# app_streamlit_rag.py

import os
import streamlit as st
import pdfplumber
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# -----------------------------
# CONFIG
# -----------------------------
KB_DIR = "kb_pdfs"
os.makedirs(KB_DIR, exist_ok=True)

# Load HuggingFace embeddings (for vector search)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Extractive QA (fast + precise)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Generative fallback (summarization if extractive fails)
gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# -----------------------------
# HELPERS
# -----------------------------
def load_pdfs_to_docs(pdf_dir: str):
    """Extract text from PDFs and return as LangChain Documents."""
    docs = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": fname, "page": i+1}))
    return docs

def build_vectorstore(docs):
    """Split documents and build FAISS vector index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(split_docs, embedding_model)

def answer_with_hybrid(query, retriever):
    """First try extractive QA; if low score, fall back to Flan-T5 summarization."""
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "Sorry, I couldn't find an answer in the knowledge base.", []

    # Build context
    context = " ".join([doc.page_content for doc in docs[:3]])

    # Extractive QA
    result = qa_pipeline({"question": query, "context": context})

    if result.get("score", 0) > 0.3:  # confident extractive answer
        return result["answer"], docs[:3]
    else:  # fallback to Flan-T5 summarization
        prompt = f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {query}"
        gen_result = gen_pipeline(prompt, max_length=150, clean_up_tokenization_spaces=True)
        return gen_result[0]["generated_text"], docs[:3]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Patent FAQ Chatbot", page_icon="📘", layout="centered")
st.title("📘 Patent & BIS Knowledge-Base Chatbot")

# Sidebar
with st.sidebar:
    st.header("Knowledge Base")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            save_path = os.path.join(KB_DIR, up.name)
            with open(save_path, "wb") as f:
                f.write(up.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s).")

    # Show current KB files in use
    st.markdown("**Current KB files:**")
    kb_files = [f for f in os.listdir(KB_DIR) if f.lower().endswith(".pdf")]
    if kb_files:
        for f in kb_files:
            st.write("-", f)
    else:
        st.info("No PDFs found in KB.")

    if st.button("🔄 Reload KB"):
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]
        st.success("Knowledge base reloaded. Please ask your question.")


# Build vector store (only once unless reload triggered)
if "vectorstore" not in st.session_state:
    docs = load_pdfs_to_docs(KB_DIR)
    if docs:
        st.session_state.vectorstore = build_vectorstore(docs)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    else:
        st.session_state.vectorstore = None
        st.session_state.retriever = None

# Chat UI
st.subheader("💬 Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question:", key="chat_input")

if query and st.session_state.retriever:
    answer, sources = answer_with_hybrid(query, st.session_state.retriever)
    st.session_state.chat_history.append({"q": query, "a": answer, "sources": sources})

# Display chat
for entry in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {entry['q']}")
    st.markdown(f"**Bot:** {entry['a']}")
    if entry["sources"]:
        with st.expander("Source(s)", expanded=False):
            for s in entry["sources"]:
                st.caption(f"{s.metadata['source']}, page {s.metadata['page']}")
    st.markdown("---")

if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
