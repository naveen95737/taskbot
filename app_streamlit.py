# app_streamlit.py
import os
import re
import io
import json
import hashlib
import time
from typing import List, Dict, Any

import streamlit as st
import pdfplumber
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# Ensure punkt tokenizer available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
# -----------------------------
# CONFIG
# -----------------------------
KB_DIR = "kb_pdfs"                # where PDFs are stored
CACHE_DIR = "kb_cache"            # where embeddings/index metadata saved
os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

PDF_EXT = ".pdf"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE_SENTENCES = 2          # small = higher chance Q/A in one chunk
TOP_K_CHUNKS = 3
EXTRACT_SENTENCES = 3
MIN_SCORE_THRESHOLD = 0.40        # tuneable: how confident we must be to "answer"
FINGERPRINT_FILE = os.path.join(CACHE_DIR, "fingerprint.txt")
INDEX_FILE = os.path.join(CACHE_DIR, "faiss.index")
EMB_FILE = os.path.join(CACHE_DIR, "embeddings.npy")
META_FILE = os.path.join(CACHE_DIR, "corpus_meta.json")

# default mapping of filename key -> friendly Source URL (edit as needed)
DEFAULT_SOURCE_URLS = {
    # If filenames contain 'PATENT' or 'ipindia' we'll map to IP India by default:
    "patent": "http://www.ipindia.gov.in/ (FREQUENTLY_ASKED_QUESTIONS_-PATENTS)",
    "bis": "http://crsbis.in/ or BIS website (FINAL_FAQs_June_2018.pdf)"
}

# -----------------------------
# HELPERS: PDF load / chunk
# -----------------------------
def load_pdf_pages(path: str) -> List[Dict[str,Any]]:
    """Extract text by page. Returns list of {'page':n, 'text':str}."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages):
            text = p.extract_text() or ""
            text = re.sub(r'\r\n', '\n', text)
            text = re.sub(r'\n{2,}', '\n\n', text)
            text = re.sub(r'[ \t]+\n', '\n', text)
            pages.append({"page": i+1, "text": text})
    return pages

def chunk_text_by_sentences(text: str, sentences_per_chunk: int) -> List[str]:
    sents = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sents), sentences_per_chunk):
        chunk = " ".join(sents[i:i+sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def filename_to_source_url(filename: str) -> str:
    fn = filename.lower()
    for k,v in DEFAULT_SOURCE_URLS.items():
        if k in fn:
            return v
    return f"{filename} (uploaded PDF)"

# fingerprint used to detect KB changes
def compute_fingerprint(filepaths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(filepaths):
        with open(p, "rb") as f:
            data = f.read()
            h.update(hashlib.sha256(data).digest())
    return h.hexdigest()

# -----------------------------
# BUILD / LOAD corpus + index
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def build_or_load_index(force_rebuild: bool=False) -> Dict[str,Any]:
    """
    Build the corpus, embeddings and FAISS index if needed.
    Returns dict with keys: embedder, index, emb_matrix, corpus (list meta), texts, bm25
    """
    # 1) list PDFs
    pdf_files = [os.path.join(KB_DIR, f) for f in os.listdir(KB_DIR) if f.lower().endswith(PDF_EXT)]
    if not pdf_files:
        return {"index": None, "emb_matrix": None, "corpus": [], "texts": [], "bm25": None, "embedder": get_embedder()}

    fingerprint = compute_fingerprint(pdf_files)
    cached_fp = None
    if os.path.exists(FINGERPRINT_FILE):
        with open(FINGERPRINT_FILE, "r") as fh:
            cached_fp = fh.read().strip()

    need_rebuild = force_rebuild or (cached_fp != fingerprint) or (not os.path.exists(INDEX_FILE)) or (not os.path.exists(EMB_FILE)) or (not os.path.exists(META_FILE))
    embedder = get_embedder()

    if need_rebuild:
        st.info("Building KB index — this may take some time (model download on first run).")
        corpus = []
        for p in pdf_files:
            fname = os.path.basename(p)
            pages = load_pdf_pages(p)
            for pg in pages:
                # chunk page text into small chunks of sentences
                chunks = chunk_text_by_sentences(pg["text"], CHUNK_SIZE_SENTENCES)
                for i, c in enumerate(chunks):
                    corpus.append({
                        "id": f"{fname}__p{pg['page']}__c{i}",
                        "filename": fname,
                        "page": pg["page"],
                        "text": c
                    })
        texts = [c["text"] for c in corpus]
        if len(texts) == 0:
            st.warning("No textual content found in uploaded PDFs.")
            return {"index": None, "emb_matrix": None, "corpus": corpus, "texts": [], "bm25": None, "embedder": embedder}

        # embeddings
        emb_matrix = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        # normalize for cosine sim
        emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

        # faiss index
        dim = emb_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_matrix)

        # persist
        faiss.write_index(index, INDEX_FILE)
        np.save(EMB_FILE, emb_matrix)
        with open(META_FILE, "w", encoding="utf-8") as fh:
            json.dump(corpus, fh, ensure_ascii=False, indent=2)
        with open(FINGERPRINT_FILE, "w") as fh:
            fh.write(fingerprint)

        # BM25
        tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
        bm25 = BM25Okapi(tokenized)

        return {"embedder": embedder, "index": index, "emb_matrix": emb_matrix, "corpus": corpus, "texts": texts, "bm25": bm25}
    else:
        # load cached
        st.info("Loading cached KB index (fast).")
        index = faiss.read_index(INDEX_FILE)
        emb_matrix = np.load(EMB_FILE)
        with open(META_FILE, "r", encoding="utf-8") as fh:
            corpus = json.load(fh)
        texts = [c["text"] for c in corpus]
        tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
        bm25 = BM25Okapi(tokenized)
        return {"embedder": embedder, "index": index, "emb_matrix": emb_matrix, "corpus": corpus, "texts": texts, "bm25": bm25}

# -----------------------------
# RETRIEVE + EXTRACT answer
# -----------------------------
def retrieve_top_chunks(query: str, resources: Dict[str,Any], top_k:int=TOP_K_CHUNKS) -> List[Dict[str,Any]]:
    embedder = resources["embedder"]
    index = resources["index"]
    corpus = resources["corpus"]
    if index is None or len(corpus) == 0:
        return []

    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    # fetch a larger set and rerank heuristically
    fetch_k = max(top_k*4, 10)
    D, I = index.search(q_emb, fetch_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        item = corpus[idx]
        # tiny boost if chunk starts like a question and matches query start
        boost = 0.0
        q_start = query.strip().lower().split()[0] if query.strip() else ""
        if item["text"].lower().startswith(q_start):
            boost += 0.03
        results.append({"id": item["id"], "filename": item["filename"], "page": item["page"], "text": item["text"], "score": float(score+boost)})
    results = sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]
    return results

def extractive_answer(query: str, chunks: List[Dict[str,Any]], resources: Dict[str,Any], max_sentences:int=EXTRACT_SENTENCES) -> tuple[str, float]:
    if not chunks:
        return "", 0.0
    embedder = resources["embedder"]
    # collect candidate sentences from chunks
    candidate_sents = []
    for c in chunks:
        sents = sent_tokenize(c["text"])
        for s in sents:
            candidate_sents.append({"sent": s.strip(), "filename": c["filename"], "page": c["page"]})
    s_texts = [s["sent"] for s in candidate_sents]
    if not s_texts:
        return chunks[0]["text"], chunks[0]["score"]

    s_embs = embedder.encode(s_texts, convert_to_numpy=True)
    s_embs = s_embs / np.linalg.norm(s_embs, axis=1, keepdims=True)
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sims = (s_embs @ q_emb.T).squeeze()

    # take top candidate sentences by similarity, keep natural order
    top_idx = np.argsort(-sims)[: max(12, max_sentences*4)]
    top_idx_sorted = sorted(top_idx.tolist())
    chosen = []
    chosen_set = set()
    for i in top_idx_sorted:
        if len(chosen) >= max_sentences:
            break
        if i not in chosen_set:
            chosen.append((float(sims[i]), s_texts[i], candidate_sents[i]["filename"], candidate_sents[i]["page"]))
            chosen_set.add(i)

    final_score = max([c["score"] for c in chunks]) if chunks else 0.0
    answer_lines = [line for sim, line, fn, pg in chosen]
    answer = " ".join(answer_lines).strip()
    return answer, final_score

def suggest_related_questions(query: str, resources: Dict[str,Any], n:int=6) -> List[str]:
    bm25 = resources.get("bm25")
    texts = resources.get("texts", [])
    if not bm25:
        return []
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    suggestions = []
    for i in top_n:
        preview = sent_tokenize(texts[i])[0]
        suggestions.append(preview if len(preview) < 220 else preview[:220]+"...")
    return suggestions

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="KB Chatbot (PDF)", page_icon="📘", layout="centered")

st.title("📘 Knowledge-Base Chatbot (PDF FAQs)")
st.markdown(
    """
    Ask questions and get answers **only from the uploaded PDF knowledge base**.
    - Upload new PDFs in the sidebar and click **Update KB**.
    - Answers are extractive (sentences from PDFs). Source file + page shown for each answer.
    """
)

# Sidebar: upload + KB management
with st.sidebar:
    st.header("Knowledge Base")
    st.markdown("Upload one or more PDF files (FAQs, manuals). They will be stored in the `kb_pdfs/` folder.")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            save_path = os.path.join(KB_DIR, up.name)
            # write file bytes to KB_DIR
            with open(save_path, "wb") as f:
                f.write(up.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to {KB_DIR}. Click 'Update KB' to rebuild index.")
    if st.button("Update KB (rebuild index)"):
        # build index and cache
        resources = build_or_load_index(force_rebuild=True)
        st.success("KB rebuilt.")
        st.experimental_rerun()

    st.markdown("**Current KB files:**")
    pdf_list = [f for f in os.listdir(KB_DIR) if f.lower().endswith(PDF_EXT)]
    if pdf_list:
        for f in pdf_list:
            st.write("-", f)
    else:
        st.info("No PDFs in KB. Upload on the top of this sidebar or place PDFs into the 'kb_pdfs' folder.")

    st.markdown("---")
    st.markdown("**Cache / Index**")
    if os.path.exists(INDEX_FILE):
        st.write("Index status: ✅ cached")
        if st.button("Force rebuild index"):
            resources = build_or_load_index(force_rebuild=True)
            st.success("Rebuilt index.")
            st.experimental_rerun()
    else:
        st.write("Index status: ❌ not yet built")

# Build/load index (lazy)
with st.spinner("Preparing knowledge base (this may take a moment on first run)..."):
    resources = build_or_load_index(force_rebuild=False)

# Input area
query = st.text_area("Ask a question:", height=120, placeholder="e.g. When can an applicant withdraw a patent application in India?")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Ask"):
        if not query.strip():
            st.warning("Type a question first.")
        else:
            # perform retrieval & extraction
            with st.spinner("Retrieving answer from KB..."):
                chunks = retrieve_top_chunks(query, resources, TOP_K_CHUNKS)
                if not chunks:
                    suggestions = suggest_related_questions(query, resources)
                    st.error("No direct answer found in the knowledge base.")
                    if suggestions:
                        st.info("Related or suggested questions:")
                        for s in suggestions:
                            if st.button(s, key=f"sugg_{hash(s)}"):
                                st.experimental_set_query_params(q=s)
                                st.experimental_rerun()
                else:
                    answer, score = extractive_answer(query, chunks, resources, EXTRACT_SENTENCES)
                    if score < MIN_SCORE_THRESHOLD:
                        # low confidence -> refuse + give suggestions
                        st.warning("This specific question might not be covered exactly. Here are related suggestions:")
                        for s in suggest_related_questions(query, resources):
                            if st.button(s, key=f"s_{hash(s)}"):
                                st.experimental_set_query_params(q=s)
                                st.experimental_rerun()
                    else:
                        # show answer
                        st.success("Chatbot Response (extractive):")
                        st.write(answer)
                        # show source(s)
                        # gather sources from chosen top chunks/sentences
                        shown_sources = {}
                        for c in chunks:
                            fname = c["filename"]
                            page = c["page"]
                            src = filename_to_source_url(fname)
                            shown_sources[f"{fname}::p{page}"] = {"file": fname, "page": page, "source": src}
                        st.markdown("**Source(s):**")
                        for k,v in shown_sources.items():
                            st.markdown(f"- **{v['file']}** — page {v['page']} — {v['source']}")

                        # record conversation in session state
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.insert(0, {"q": query, "a": answer, "sources": list(shown_sources.values()), "score": float(score)})
    # quick suggest button
    if st.button("Get suggestions"):
        if not query.strip():
            st.warning("Type a few keywords to get suggestions.")
        else:
            suggestions = suggest_related_questions(query, resources)
            if suggestions:
                st.info("Suggested questions (click to ask):")
                for s in suggestions:
                    if st.button(s, key=f"ss_{hash(s)}"):
                        st.experimental_set_query_params(q=s)
                        st.experimental_rerun()
            else:
                st.info("No suggestions found.")

with col2:
    st.markdown("### Conversation history")
    if "history" not in st.session_state or len(st.session_state.history) == 0:
        st.info("No history yet. Ask a question to start.")
    else:
        for i, e in enumerate(st.session_state.history):
            with st.expander(f"Q: {e['q']}", expanded=(i==0)):
                st.write(e["a"])
                st.markdown("**Sources:**")
                for s in e["sources"]:
                    st.markdown(f"- {s['file']} — page {s['page']} — {s['source']}")
                st.markdown(f"Confidence score: {e['score']:.3f}")
                if st.button("Ask this again", key=f"reask_{i}"):
                    st.experimental_set_query_params(q=e['q'])
                    st.experimental_rerun()

st.markdown("---")
st.markdown("### Developer / Notes")
st.markdown(
    "- This app **only** returns answers extracted from the uploaded PDF files (no external LLM generation).\n"
    "- To update the KB: upload PDFs in the sidebar and click **Update KB (rebuild index)**.\n"
    "- The first run downloads the embedding model and builds the index — this may take 30–120 seconds depending on your machine.\n"
)

