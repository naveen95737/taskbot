"""
app_groq_kb.py
Streamlit chatbot: KB-only answers (PDFs) + FAISS retrieval + BM25 suggestions + Groq LLM
Now with Conversation Memory, History Awareness, and Clean Answer Handling.
"""

import os
import re
import time
import pdfplumber
import numpy as np
import streamlit as st
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# Optional Groq client from LangChain
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # Handle gracefully

# -------------------------
# Configuration
# -------------------------
DATA_DIR = "knowledge_base"
os.makedirs(DATA_DIR, exist_ok=True)
SOURCE_URL = "http://www.ipindia.gov.in/ (FREQUENTLY ASKED QUESTIONS - PATENTS)"

# Initialize session variables
for key, default in {
    "kb_texts": [],
    "kb_metadatas": [],
    "bm25": None,
    "embed_model": None,
    "faiss_index": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# PDF and Text Processing
# -------------------------
def extract_pdf_text_by_page(path: str) -> List[Tuple[int, str]]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            while end > start and text[end] != " ":
                end -= 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

# -------------------------
# Knowledge Base Builder
# -------------------------
def build_kb_from_folder(data_dir: str):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    all_texts, metadatas = [], []
    if not files:
        return [], [], None, None

    for fname in sorted(files):
        fpath = os.path.join(data_dir, fname)
        for page_no, raw_text in extract_pdf_text_by_page(fpath):
            for chunk in chunk_text(raw_text):
                all_texts.append(chunk)
                metadatas.append({"source": fname, "page": page_no})

    # BM25
    tokenized = [re.findall(r"\w+", t.lower()) for t in all_texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None

    # Embeddings + FAISS
    model = st.session_state.get("embed_model")
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.embed_model = model

    if all_texts:
        embeddings = model.encode(all_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(np.asarray(embeddings))
    else:
        index = None

    return all_texts, metadatas, index, bm25

# -------------------------
# Search & Suggestion Utils
# -------------------------
def semantic_search(query: str, k: int = 4):
    if st.session_state.faiss_index is None:
        return []
    model = st.session_state.embed_model
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = st.session_state.faiss_index.search(np.asarray(q_emb), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(st.session_state.kb_texts):
            results.append((float(score), st.session_state.kb_texts[idx], st.session_state.kb_metadatas[idx]))
    return results

def keyword_suggestions(query: str, top_n: int = 3):
    bm25, corpus = st.session_state.bm25, st.session_state.kb_texts
    if bm25 is None or not corpus:
        return []
    tokens = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    suggestions = []
    for idx, _ in ranked[:top_n * 3]:
        text = corpus[idx]
        for s in re.split(r"(?<=[.!?])\s+", text):
            s_clean = s.strip()
            if 40 < len(s_clean) < 160 and any(tok in s_clean.lower() for tok in tokens):
                if not s_clean.endswith("?"):
                    s_clean += "?"
                suggestions.append(s_clean)
                break
        if len(suggestions) >= top_n:
            break
    return suggestions

# -------------------------
# Clean Answer
# -------------------------
def clean_answer(text):
    if not text.strip():
        return "I don't have a specific answer in the knowledge base."
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    seen, cleaned = set(), []
    for s in sentences:
        if s not in seen:
            cleaned.append(s)
            seen.add(s)
    return ". ".join(cleaned) + "." if cleaned else text.strip()[:300]

# -------------------------
# Groq Client
# -------------------------
def init_groq_client():
    if ChatGroq is None:
        return None
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        return None
    try:
        return ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0.0)
    except TypeError:
        return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.0)

def call_groq_chat(client, prompt: str) -> str:
    if client is None:
        raise RuntimeError("Groq client not initialized.")
    try:
        if hasattr(client, "invoke"):
            out = client.invoke(prompt)
            return out if isinstance(out, str) else getattr(out, "content", str(out))
        elif hasattr(client, "chat"):
            res = client.chat([{"role": "user", "content": prompt}])
            return res.get("content", "") if isinstance(res, dict) else getattr(res, "content", "")
    except Exception as e:
        st.error(f"Error calling Groq: {e}")
    return ""

# -------------------------
# Conversation Memory Helpers
# -------------------------
def summarize_chat_history(chat_history: List[Dict], max_turns: int = 5) -> str:
    if not chat_history:
        return ""
    summary = []
    for turn in chat_history[-max_turns:]:
        summary.append(f"User: {turn['q']}\nAssistant: {turn['a']}")
    return "\n".join(summary)

def detect_memory_query(user_query: str) -> bool:
    keywords = [
        "previous question", "earlier question", "last question",
        "what were we talking", "previous topic", "remind me what we discussed",
        "earlier chat", "past conversation", "what did i ask", "summary of chat"
    ]
    text = user_query.lower()
    return any(kw in text for kw in keywords)

def answer_from_memory(user_query: str, chat_history: List[Dict]) -> str:
    if not chat_history:
        return "We haven’t talked about anything yet."
    last_turn = chat_history[-1]
    if "last question" in user_query.lower() or "previous question" in user_query.lower():
        return f"Your previous question was: '{last_turn['q']}'"
    elif "previous answer" in user_query.lower():
        return f"My previous answer was: '{last_turn['a']}'"
    elif "talking about" in user_query.lower() or "discussing" in user_query.lower():
        return f"We were talking about: '{last_turn['q']}' — {last_turn['a'][:200]}..."
    elif "summary" in user_query.lower():
        return summarize_chat_history(chat_history)
    else:
        return summarize_chat_history(chat_history)

# -------------------------
# Answer Composition
# -------------------------
def compose_context(sem_results: List[Tuple[float, str, Dict]], k=3) -> str:
    parts = []
    for score, text, meta in sem_results[:k]:
        src = meta.get("source", "Uploaded PDF")
        page = meta.get("page", "N/A")
        parts.append(f"[Source: {src} | Page: {page}]\n{text}")
    return "\n\n---\n\n".join(parts)

def generate_answer_groq(client, question: str, context: str) -> str:
    if len(context) > 32000:
        context = context[-32000:]
    prompt = (
        "You are an assistant that answers ONLY using the provided context. "
        "If the answer is not in the context, respond: 'I don't have an exact answer in the knowledge base.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return clean_answer(call_groq_chat(client, prompt).strip())

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Patent FAQ Chatbot (Groq)", layout="wide", page_icon="📘")
st.title("📘 Patent FAQ Chatbot")

# Sidebar
with st.sidebar:
    st.header("📂 Knowledge Base Management")

    kb_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if kb_files:
        st.subheader("📑 Current KB Files")
        for f in kb_files:
            st.write(f"📄 {f}")
    else:
        st.warning("⚠️ No PDFs in knowledge_base folder.")

    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            path = os.path.join(DATA_DIR, uf.name)
            with open(path, "wb") as f:
                f.write(uf.read())
        st.success(f"✅ Saved {len(uploaded)} file(s). KB will be rebuilt now.")
        texts, metas, index, bm25 = build_kb_from_folder(DATA_DIR)
        st.session_state.kb_texts, st.session_state.kb_metadatas = texts, metas
        st.session_state.faiss_index, st.session_state.bm25 = index, bm25
        st.info("🔄 Knowledge base rebuilt successfully!")

    if st.button("🔁 Rebuild KB manually"):
        with st.spinner("Rebuilding KB..."):
            texts, metas, index, bm25 = build_kb_from_folder(DATA_DIR)
            st.session_state.kb_texts, st.session_state.kb_metadatas = texts, metas
            st.session_state.faiss_index, st.session_state.bm25 = index, bm25
        st.success("✅ KB rebuilt!")

    st.markdown("---")
    st.header("Groq Settings")
    if not (ChatGroq and (st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY"))):
        st.warning("⚠️ Missing Groq API key. Add it in Streamlit Secrets.")
    else:
        st.info("✅ Groq client configured successfully.")

# Init KB and Groq
if not st.session_state.kb_texts:
    with st.spinner("Initializing KB..."):
        texts, metas, index, bm25 = build_kb_from_folder(DATA_DIR)
        st.session_state.kb_texts, st.session_state.kb_metadatas = texts, metas
        st.session_state.faiss_index, st.session_state.bm25 = index, bm25

if "groq_client" not in st.session_state:
    st.session_state.groq_client = init_groq_client()

# Chat Interface
st.subheader("💬 Ask questions based on uploaded documents")
query = st.text_input("Your question:")

if query:
    if not st.session_state.faiss_index:
        st.warning("Upload PDFs and rebuild KB first.")
    elif st.session_state.groq_client is None:
        st.error("Groq client not initialized.")
    else:
        if detect_memory_query(query):
            answer_text = answer_from_memory(query, st.session_state.chat_history)
            used_sources = []
        else:
            with st.spinner("Retrieving relevant passages..."):
                sem = semantic_search(query, k=6)
                top_scores = [s for s, _, _ in sem]
                max_score = max(top_scores) if top_scores else 0.0
                suggestions = keyword_suggestions(query, top_n=4)

                if max_score < 0.20 and not suggestions:
                    answer_text = "I don't have an exact answer in the knowledge base."
                    used_sources = []
                else:
                    chat_context = summarize_chat_history(st.session_state.chat_history, max_turns=4)
                    context = compose_context(sem, k=4)
                    full_context = f"Previous conversation:\n{chat_context}\n\nKnowledge Base Context:\n{context}"
                    answer_text = generate_answer_groq(st.session_state.groq_client, query, full_context)
                    used_sources = [f"{m.get('source')} (Page {m.get('page')})" for _, _, m in sem[:4]]

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({"q": query, "a": answer_text, "sources": used_sources, "time": timestamp})

        st.markdown(f"**You:** {query}")
        st.markdown(f"**Bot:** {answer_text}")

        if used_sources:
            with st.expander("📖 Sources"):
                for s in used_sources:
                    st.caption(f"{s} — {SOURCE_URL}")

        suggestions = keyword_suggestions(query, top_n=3)
        if suggestions:
            st.info("💡 Suggested Related Questions:")
            for s in suggestions:
                st.markdown(f"- {s}")

# Display Chat History
if st.session_state.chat_history:
    st.markdown("### 📝 Conversation History")
    for turn in reversed(st.session_state.chat_history[-12:]):
        st.markdown(f"**You:** {turn['q']}")
        st.markdown(f"**Bot:** {turn['a']}")
        if turn["sources"]:
            with st.expander("📚 Sources"):
                for s in turn["sources"]:
                    st.caption(f"{s} — {SOURCE_URL}")
        st.markdown("---")

st.caption("Patent FAQ Chatbot • Powered by Groq & LangChain ")

