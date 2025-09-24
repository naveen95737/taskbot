# app_streamlit_final.py

import os
import re
import streamlit as st
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from rank_bm25 import BM25Okapi

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "knowledge_base"
os.makedirs(DATA_DIR, exist_ok=True)

SOURCE_URL = "http://www.ipindia.gov.in/ (FREQUENTLY ASKED QUESTIONS - PATENTS)"

bm25_index = None
kb_texts = []


# -------------------------------
# Load KB
# -------------------------------
def load_knowledge_base():
    """Load PDFs from DATA_DIR, build FAISS index, and return QA chain."""
    global bm25_index, kb_texts

    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())

    if not docs:
        return None

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # ensures Streamlit Cloud compatibility
    )

    # FAISS vector store
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Lightweight LLM for rephrasing (not inventing!)
    gen_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
        model_kwargs={
            "max_length": 300,
            "no_repeat_ngram_size": 3,
            "num_beams": 4,
            "early_stopping": True
        }
    )
    llm = HuggingFacePipeline(pipeline=gen_pipeline)

    # BM25 keyword search
    kb_texts = [doc.page_content for doc in split_docs]
    tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in kb_texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )


# -------------------------------
# Related Question Suggestions
# -------------------------------
def suggest_related_questions(query, top_n=3):
    global bm25_index, kb_texts
    if not bm25_index:
        return []
    tokenized_query = re.findall(r"\w+", query.lower())
    scores = bm25_index.get_scores(tokenized_query)
    ranked = sorted(zip(scores, kb_texts), key=lambda x: x[0], reverse=True)
    return [t for _, t in ranked[:top_n]]


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Patent FAQ Chatbot", page_icon="📘", layout="wide")
st.title("📘 Chat with the Knowledge Base")

# Sidebar – KB Management
st.sidebar.header("📂 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload new PDF(s) to KB", type=["pdf"], accept_multiple_files=True
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s).")

if st.sidebar.button("🔄 Reload KB"):
    st.session_state.qa_chain = load_knowledge_base()
    st.sidebar.success("Knowledge Base reloaded!")

# Show current KB files
st.sidebar.subheader("📑 Current KB Files")
kb_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
if kb_files:
    for f in kb_files:
        st.sidebar.write(f"- {f}")
else:
    st.sidebar.info("No PDFs uploaded yet.")


# -------------------------------
# Initialize QA chain
# -------------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_knowledge_base()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------
# Chat UI
# -------------------------------
st.subheader("💬 Ask a question about patents in India:")
query = st.text_input("Type your question here:")


def clean_answer(text):
    """Remove duplicate/unfinished sentences."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    seen, cleaned = set(), []
    for s in sentences:
        if s not in seen:
            cleaned.append(s)
            seen.add(s)
    return ". ".join(cleaned)


if query:
    if st.session_state.qa_chain:
        result = st.session_state.qa_chain.invoke({"query": query})
        raw_answer = result["result"].strip()

        # Handle missing answer
        if not raw_answer or raw_answer.lower().startswith("i don’t"):
            answer = "I don’t have an exact answer in the knowledge base."
        else:
            answer = clean_answer(raw_answer)

        # Collect sources
        sources = []
        for doc in result.get("source_documents", []):
            sources.append(f"{doc.metadata.get('source', 'Unknown Source')}")

        # Save Q/A turn
        entry = {"q": query, "a": answer, "sources": list(set(sources))}
        entry["related"] = suggest_related_questions(query, top_n=3)
        st.session_state.chat_history.append(entry)
    else:
        st.warning("Please upload and reload the knowledge base first.")


# -------------------------------
# Display Chat History
# -------------------------------
for entry in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {entry['q']}")
    st.markdown(f"**Bot:** {entry['a']}")
    if entry.get("sources"):
        with st.expander("📖 Source(s)"):
            for _ in entry["sources"]:
                st.caption(f"Source URL: {SOURCE_URL}")
    if entry.get("related"):
        with st.expander("💡 Related Questions"):
            for rq in entry["related"]:
                st.write(f"- {rq}")
    st.markdown("---")


# -------------------------------
# Clear Chat
# -------------------------------
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
