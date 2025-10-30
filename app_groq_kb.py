"""
app_groq_kb.py
Streamlit chatbot: KB-only answers (PDFs) + FAISS retrieval + BM25 suggestions + Groq LLM
Set GROQ_API_KEY in Streamlit secrets or environment variables before deploying.
"""

import os
import re
import io
import hashlib
import time
import json
import pathlib
from typing import List, Tuple, Dict

import streamlit as st
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# NOTE: we import ChatGroq from langchain_groq if available.
# If your deployment requires a different Groq client, swap call_groq() implementation.
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # we will check later and show instructions if missing

# -------------------------
# Config
# -------------------------
DATA_DIR = "knowledge_base"
os.makedirs(DATA_DIR, exist_ok=True)

# A fallback source URL to display with sources (customize if needed)
SOURCE_URL = "http://www.ipindia.gov.in/ (FREQUENTLY ASKED QUESTIONS - PATENTS)"

# Retriever artifacts (kept in session for speed)
if "kb_index" not in st.session_state:
    st.session_state.kb_index = None
if "kb_texts" not in st.session_state:
    st.session_state.kb_texts = []
if "kb_metadatas" not in st.session_state:
    st.session_state.kb_metadatas = []
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "embed_model" not in st.session_state:
    st.session_state.embed_model = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Helper: PDF -> chunks
# -------------------------
def extract_pdf_text_by_page(path: str) -> List[Tuple[int, str]]:
    """Return list of (page_number, text) for the PDF."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Simple chunking by characters preserving word boundaries."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        # backtrack to last space if not end
        if end < L:
            while end > start and text[end] != " ":
                end -= 1
            if end == start:
                end = min(start + chunk_size, L)  # forced
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)  # overlap
    return chunks

# -------------------------
# Build KB (embeddings, FAISS, BM25)
# -------------------------
def build_kb_from_folder(data_dir: str):
    """Load all PDFs in DATA_DIR -> produce embeddings, faiss index, bm25 index, and metadata list."""
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    all_texts = []
    metadatas = []  # list of dicts: {source, page}
    if not files:
        return [], None, None, None

    for fname in sorted(files):
        fpath = os.path.join(data_dir, fname)
        pages = extract_pdf_text_by_page(fpath)
        for page_no, raw_text in pages:
            # chunk page text into smaller chunks
            page_chunks = chunk_text(raw_text, chunk_size=900, overlap=200)
            for chunk in page_chunks:
                all_texts.append(chunk)
                metadatas.append({"source": fname, "page": page_no})
    # create BM25
    tokenized = [re.findall(r"\w+", t.lower()) for t in all_texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None

    # embeddings (sentence-transformers)
    model = st.session_state.get("embed_model")
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.embed_model = model

    if all_texts:
        embeddings = model.encode(all_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # inner-product since embeddings normalized -> cosine similarity
        index.add(np.asarray(embeddings))
    else:
        index = None

    return all_texts, metadatas, index, bm25

# -------------------------
# Utilities: search & suggestions
# -------------------------
def semantic_search(query: str, k: int = 4):
    if st.session_state.faiss_index is None or st.session_state.kb_texts is None:
        return []
    model = st.session_state.embed_model
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = st.session_state.faiss_index.search(np.asarray(q_emb), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(st.session_state.kb_texts):
            continue
        results.append((float(score), st.session_state.kb_texts[idx], st.session_state.kb_metadatas[idx]))
    return results

def keyword_suggestions(query: str, top_n: int = 3):
    bm25 = st.session_state.get("bm25")
    corpus = st.session_state.get("kb_texts", [])
    if bm25 is None or not corpus:
        return []
    tokenized_query = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    suggestions = []
    for idx, score in ranked[:min(len(ranked), top_n*3)]:
        text = corpus[idx]
        # pick a sentence-like piece for suggestion
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s_clean = s.strip()
            if 40 < len(s_clean) < 160 and any(tok in s_clean.lower() for tok in tokenized_query):
                if not s_clean.endswith("?"):
                    s_clean += "?"
                suggestions.append(s_clean)
                break
        if len(suggestions) >= top_n:
            break
    return suggestions

# -------------------------
# LLM: call Groq
# -------------------------
def init_groq_client():
    if ChatGroq is None:
        return None
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        return None
    # initialize ChatGroq (langchain_groq wrapper) - usage may vary depending on package version
    try:
        client = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.0)
    except TypeError:
        # alternative signature (older/newer)
        client = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.0)
    return client

def call_groq_chat(client, prompt: str, max_tokens: int = 512) -> str:
    """
    Call Groq client and return assistant text.
    This function uses ChatGroq.invoke() if available, else tries .chat or .complete depending on installed SDK.
    """
    if client is None:
        raise RuntimeError("Groq client not initialized. Set GROQ_API_KEY and install langchain_groq.")
    # try several call styles to maximize compatibility
    try:
        # langchain_groq.ChatGroq: .invoke or .generate or .chat may exist
        if hasattr(client, "invoke"):
            out = client.invoke(prompt)
            if isinstance(out, str):
                return out
            # langchain-like response
            if hasattr(out, "content"):
                return out.content
            if isinstance(out, dict) and "content" in out:
                return out["content"]
        if hasattr(client, "chat"):
            res = client.chat([{"role": "user", "content": prompt}])
            if isinstance(res, dict) and "content" in res:
                return res["content"]
            if hasattr(res, "content"):
                return res.content
        if hasattr(client, "generate"):
            res = client.generate(prompt)
            if isinstance(res, dict) and "text" in res:
                return res["text"]
    except Exception as e:
        st.error(f"Error calling Groq: {e}")
        raise
    # last-resort: stringfy
    return str(out)

# -------------------------
# High-level answer pipeline
# -------------------------
def compose_context(sem_results: List[Tuple[float, str, Dict]], k=3) -> str:
    selected = sem_results[:k]
    parts = []
    for score, text, meta in selected:
        src = meta.get("source", "Uploaded PDF")
        page = meta.get("page", "N/A")
        parts.append(f"[Source: {src} | Page: {page}]\n{text}")
    return "\n\n---\n\n".join(parts)

def generate_answer_groq(client, question: str, context: str) -> str:
    """
    Provide a strict KB-only answer instruction. If the KB doesn't contain the answer, instruct model to reply:
    "I don’t have an exact answer in the knowledge base."
    """
    # Safety: limit context size
    if len(context) > 32000:
        context = context[-32000:]
    prompt = (
        "You are an assistant that answers only using the provided context. "
        "Do NOT invent facts. If the answer is not contained explicitly in the context, respond with: "
        "\"I don't have an exact answer in the knowledge base.\" "
        "Always keep answers concise and mention source filenames and page numbers at the end.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    response = call_groq_chat(client, prompt)
    # Ensure response doesn't hallucinate sources — user is responsible to inspect source section shown separately
    return response.strip()

# -------------------------
# UI: Sidebar - upload & reload
# -------------------------
st.set_page_config(page_title="Patent FAQ Chatbot (Groq)", layout="wide", page_icon="📘")
st.title("📘 Patent FAQ Chatbot — Groq (KB-only answers)")

with st.sidebar:
    st.header("Knowledge Base")
    st.markdown("Upload PDFs to add to the knowledge base (or push files into `knowledge_base/` on the repo).")
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            path = os.path.join(DATA_DIR, uf.name)
            with open(path, "wb") as f:
                f.write(uf.read())
        st.success(f"Saved {len(uploaded)} file(s). Please click **Reload KB** to rebuild indexes.")
    if st.button("🔄 Reload KB"):
        with st.spinner("Building knowledge base (embeddings + index)..."):
            texts, metas, index, bm25 = build_kb_from_folder(DATA_DIR)
            st.session_state.kb_texts = texts
            st.session_state.kb_metadatas = metas
            st.session_state.faiss_index = index
            st.session_state.bm25 = bm25
        st.success("Knowledge base reloaded.")

    st.markdown("---")
    st.header("Groq Settings")
    groq_ok = ChatGroq is not None and (st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY"))
    if not groq_ok:
        st.warning("Groq client NOT available or GROQ_API_KEY missing. Set GROQ_API_KEY in Streamlit secrets.")
    else:
        st.write("Groq available. Using configured API key.")
    st.markdown("")

# initialize if not loaded yet
if st.session_state.get("kb_texts") is None or st.session_state.get("faiss_index") is None:
    # automatic build if folder contains PDFs
    with st.spinner("Checking knowledge base..."):
        texts, metas, index, bm25 = build_kb_from_folder(DATA_DIR)
        st.session_state.kb_texts = texts
        st.session_state.kb_metadatas = metas
        st.session_state.faiss_index = index
        st.session_state.bm25 = bm25

# initialize Groq client (lazily)
if "groq_client" not in st.session_state:
    st.session_state.groq_client = init_groq_client()

# -------------------------
# Chat UI
# -------------------------
st.subheader("Ask questions (about uploaded PDFs) — I'll answer only from the documents.")

query = st.text_input("Your question:", key="user_query")
col1, col2 = st.columns([3, 1])

with col2:
    if st.button("Suggest related FAQs"):
        if query:
            s = keyword_suggestions(query, top_n=4)
            if s:
                st.success("Suggested related questions:")
                for q in s:
                    st.markdown(f"- {q}")
            else:
                st.info("No close matches found for suggestions.")
        else:
            st.info("Type a short query first to get suggestions.")

if query:
    # quick checks
    if not st.session_state.faiss_index or not st.session_state.kb_texts:
        st.warning("Knowledge base empty — upload PDFs and click Reload KB (sidebar) or push PDFs to `knowledge_base/`.")
    elif st.session_state.groq_client is None:
        st.error("Groq client not initialized. Set GROQ_API_KEY in Streamlit secrets and include langchain_groq in requirements.")
    else:
        with st.spinner("Retrieving relevant passages..."):
            sem = semantic_search(query, k=6)  # returns list of (score, text, meta)
            # optional threshold to consider "no result"
            top_scores = [s for s, _, _ in sem]
            max_score = max(top_scores) if top_scores else 0.0
            # also get BM25 suggestions
            suggestions = keyword_suggestions(query, top_n=4)

            # Decide if KB likely contains answer (heuristic)
            if max_score < 0.20 and (not suggestions):
                answer_text = "I don't have an exact answer in the knowledge base."
                used_sources = []
            else:
                context = compose_context(sem, k=4)
                answer_text = generate_answer_groq(st.session_state.groq_client, query, context)
                # collect sources shown in context
                used_sources = []
                for _, _, meta in sem[:4]:
                    used_sources.append(f"{meta.get('source','Uploaded PDF')} (Page {meta.get('page','N/A')})")

        # Save to chat history
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st.session_state.chat_history.append({"q": query, "a": answer_text, "sources": used_sources, "time": timestamp})

# Display history (most recent first)
if st.session_state.chat_history:
    st.markdown("### Conversation")
    for turn in reversed(st.session_state.chat_history[-12:]):
        st.markdown(f"**You:** {turn['q']}")
        st.markdown(f"**Bot:** {turn['a']}")
        if turn["sources"]:
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.caption(f"{s} — Source URL: {SOURCE_URL}")
        st.markdown("---")

st.caption("Strictly answers from uploaded PDFs. If the KB doesn't contain an answer, the bot will say so and show related FAQ suggestions.")
