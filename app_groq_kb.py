"""
app_groq_kb_fixed.py
Streamlit chatbot: KB-only answers (PDFs) + FAISS retrieval + BM25 suggestions + Groq LLM
Merged + fixes: lower threshold, BM25 fallback, improved suggestions, answer cleaning.
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

# Semantic-match threshold (tunable)
SEMANTIC_SCORE_THRESHOLD = 0.07  # lowered from 0.20 to reduce false negatives

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
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        if end < L:
            # backtrack to last space to avoid breaking words
            while end > start and text[end] != " ":
                end -= 1
            if end == start:
                end = min(start + chunk_size, L)
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
        # use normalized embeddings (so inner product ~ cosine similarity)
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
    """Return list of (score, text, metadata)."""
    if st.session_state.faiss_index is None or not st.session_state.kb_texts:
        return []
    model = st.session_state.embed_model
    if model is None:
        return []
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # Ensure right shape
    q_arr = np.asarray(q_emb).astype(np.float32)
    try:
        D, I = st.session_state.faiss_index.search(q_arr, k)
    except Exception:
        # some FAISS builds require 2D float32 input explicitly
        D, I = st.session_state.faiss_index.search(np.ascontiguousarray(q_arr, dtype=np.float32), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(st.session_state.kb_texts):
            results.append((float(score), st.session_state.kb_texts[idx], st.session_state.kb_metadatas[idx]))
    return results

def bm25_top_chunks(query: str, top_n: int = 3):
    """Return top (text, metadata) using BM25 ranking (useful as fallback)."""
    bm25 = st.session_state.get("bm25")
    corpus = st.session_state.get("kb_texts", [])
    metas = st.session_state.get("kb_metadatas", [])
    if bm25 is None or not corpus:
        return []
    tokenized_query = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokenized_query)
    ranked_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for idx in ranked_idxs[:top_n]:
        results.append((float(scores[idx]), corpus[idx], metas[idx]))
    return results

def keyword_suggestions(query: str, top_n: int = 3):
    """
    Return short, useful suggestion strings based on BM25 top chunks.
    We try to pick FAQ-looking sentences containing query tokens.
    """
    bm25 = st.session_state.get("bm25")
    corpus = st.session_state.get("kb_texts", [])
    if bm25 is None or not corpus:
        return []
    tokens = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    suggestions = []
    for idx, _ in ranked[:min(len(ranked), top_n * 6)]:
        text = corpus[idx]
        # choose sentences and prefer those that look like answers or Q/A lines
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 40 or len(s_clean) > 300:
                continue
            # prefer sentences containing one of tokens
            if any(tok in s_clean.lower() for tok in tokens):
                # if the sentence is a statement, convert into a suggestion question only if it's short and clear
                if s_clean.endswith("?"):
                    suggestion = s_clean
                else:
                    # make a suggestion that is short and question-like
                    # create a question from the main phrase (best-effort)
                    suggestion = s_clean
                    if not suggestion.endswith("?"):
                        suggestion = suggestion.rstrip(". ")
                        if len(suggestion) < 200:
                            suggestion = suggestion + "?"
                suggestions.append(suggestion)
                break
        if len(suggestions) >= top_n:
            break
    # unique preserve order
    seen = set(); uniq = []
    for s in suggestions:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq[:top_n]

# -------------------------
# Clean Answer
# -------------------------
def clean_answer(text):
    if not text or not text.strip():
        return "I don't have a specific answer in the knowledge base."
    # preserve sentences > minimal length, remove duplicates
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    seen, cleaned = set(), []
    for s in sentences:
        if s not in seen:
            cleaned.append(s)
            seen.add(s)
    if not cleaned:
        return text.strip()[:400]
    # rejoin ensuring punctuation
    joined = " ".join(cleaned)
    # ensure ends with punctuation
    if joined and joined[-1] not in ".!?":
        joined += "."
    return joined

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
    q_lower = user_query.lower()
    if "last question" in q_lower or "previous question" in q_lower or "what did i ask" in q_lower:
        return f"Your previous question was: '{last_turn['q']}'"
    elif "previous answer" in q_lower or "last answer" in q_lower:
        return f"My previous answer was: '{last_turn['a']}'"
    elif "talking about" in q_lower or "discussing" in q_lower or "what were we" in q_lower:
        preview = last_turn['a'][:300]
        return f"We were talking about: '{last_turn['q']}'. Briefly: {preview}"
    elif "summary" in q_lower:
        return summarize_chat_history(chat_history, max_turns=8)
    else:
        return summarize_chat_history(chat_history, max_turns=4)

# -------------------------
# Answer Composition
# -------------------------
def compose_context(sem_results: List[Tuple[float, str, Dict]], k=3) -> str:
    parts = []
    for score, text, meta in sem_results[:k]:
        src = meta.get("source", "Uploaded PDF")
        page = meta.get("page", "N/A")
        parts.append(f"[Source: {src} | Page: {page} | Score: {score:.3f}]\n{text}")
    return "\n\n---\n\n".join(parts)

def generate_answer_groq(client, question: str, context: str) -> str:
    # safe guard on length
    if len(context) > 32000:
        context = context[-32000:]
    prompt = (
        "You are an assistant that answers ONLY using the provided context. "
        "If the answer is not in the context, respond exactly: 'I don't have an exact answer in the knowledge base.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    raw = call_groq_chat(client, prompt).strip()
    return clean_answer(raw)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Patent FAQ Chatbot (Groq)", layout="wide", page_icon="📘")
st.title("📘 Patent FAQ Chatbot)")

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

# Init KB and Groq on start if empty
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
        # memory queries handled first
        if detect_memory_query(query):
            answer_text = answer_from_memory(query, st.session_state.chat_history)
            used_sources = []
        else:
            with st.spinner("Retrieving relevant passages..."):
                sem = semantic_search(query, k=6)
                top_scores = [s for s, _, _ in sem] if sem else []
                max_score = max(top_scores) if top_scores else 0.0

                # BM25 suggestions & top chunks
                suggestions = keyword_suggestions(query, top_n=4)
                bm25_chunks = bm25_top_chunks(query, top_n=3)

                # Decide strategy:
                # - if high semantic match -> use semantic chunks
                # - if low semantic but BM25 has good matches -> use BM25 chunks as fallback
                if max_score >= SEMANTIC_SCORE_THRESHOLD:
                    used_context_results = sem
                elif bm25_chunks:
                    # convert bm25 chunks to same tuple structure with score
                    used_context_results = bm25_chunks
                else:
                    used_context_results = []

                if not used_context_results:
                    answer_text = "I don't have an exact answer in the knowledge base."
                    used_sources = []
                else:
                    # compose readable context
                    context = compose_context(used_context_results, k=4)
                    chat_context = summarize_chat_history(st.session_state.chat_history, max_turns=4)
                    full_context = f"Previous conversation:\n{chat_context}\n\nKnowledge Base Context:\n{context}"
                    answer_text = generate_answer_groq(st.session_state.groq_client, query, full_context)
                    used_sources = []
                    for _, _, meta in used_context_results[:4]:
                        # meta might be a tuple (score,text,meta) OR (score,text,meta) from bm25_top_chunks
                        if isinstance(meta, dict):
                            used_sources.append(f"{meta.get('source')} (Page {meta.get('page')})")
                        else:
                            # If structure differs, try to protectively extract
                            try:
                                used_sources.append(str(meta))
                            except Exception:
                                pass

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({"q": query, "a": answer_text, "sources": used_sources, "time": timestamp})

        # display current Q/A
        st.markdown(f"**You:** {query}")
        st.markdown(f"**Bot:** {answer_text}")

        # show sources if any
        if used_sources:
            with st.expander("📖 Sources"):
                for s in used_sources:
                    st.caption(f"{s} — {SOURCE_URL}")

        # show suggestions (top 3)
        suggestion_list = keyword_suggestions(query, top_n=3)
        if suggestion_list:
            st.info("💡 Suggested Related Questions:")
            for s in suggestion_list:
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

st.caption("Patent FAQ Chatbot • Powered by Groq & LangChain • Context + KB Memory + Related Suggestions")

