
import os
import re
import hashlib
import logging
import traceback
from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "knowledge_base"
os.makedirs(DATA_DIR, exist_ok=True)

SOURCE_URL = "http://www.ipindia.gov.in/ (FREQUENTLY ASKED QUESTIONS - PATENTS)"

st.set_page_config(page_title="Patent FAQ Chatbot", page_icon="📘", layout="wide")
st.title("📘 Patent FAQ Chatbot (India)")

# -------------------------------
# Simple Logging / Diagnostics
# -------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "streamlit_app.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def log_exception(tag: str, exc: Exception):
    logging.error("%s: %s", tag, str(exc))
    logging.error(traceback.format_exc())
    st.session_state.last_error = f"{tag}: {str(exc)}\n\n{traceback.format_exc()}"


# Ensure session keys exist
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# -------------------------------
# Load Knowledge Base
# -------------------------------
def load_knowledge_base():
    """Load PDFs, create FAISS index, return retriever + BM25 index."""
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(DATA_DIR, file))
                docs.extend(loader.load())
            except Exception as e:
                logging.warning("Failed to load %s: %s", file, str(e))

    if not docs:
        return None, None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # BM25 for keyword-based suggestions
    kb_texts = [doc.page_content for doc in split_docs]
    tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in kb_texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    return vectorstore.as_retriever(search_kwargs={"k": 5}), bm25_index, kb_texts


# -------------------------------
# Handle Meta-History Questions
# -------------------------------
def handle_history_question(query: str, chat_history: list):
    """Detect and respond to questions about conversation history."""
    query_lower = query.lower().strip()
    if any(x in query_lower for x in ["previous question", "last question"]):
        if not chat_history:
            return "No previous question yet.", True
        last_q, _ = chat_history[-1]
        return f"Your previous question was: '{last_q}'", True
    return None, False


# -------------------------------
# Related Question Suggestions
# -------------------------------
def suggest_related_questions(query, bm25_index, kb_texts, top_n=3):
    if not bm25_index or not kb_texts:
        return []
    keywords = re.findall(r"\w+", query.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 2]
    if not keywords:
        return []
    scores = bm25_index.get_scores(keywords)
    ranked = sorted(zip(scores, kb_texts), key=lambda x: x[0], reverse=True)
    suggestions = []
    for _, text in ranked[:top_n * 3]:
        sentences = re.split(r'[.!?]+', text)
        for s in sentences:
            s_clean = s.strip()
            if 40 < len(s_clean) < 150 and any(kw in s_clean.lower() for kw in keywords):
                if not s_clean.endswith("?"):
                    s_clean += "?"
                suggestions.append(s_clean)
                break
        if len(suggestions) >= top_n:
            break
    return suggestions[:top_n]


# -------------------------------
# Clean Answer
# -------------------------------
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


# -------------------------------
# Sidebar: KB Management
# -------------------------------
st.sidebar.header("📂 Knowledge Base Management")

# Show current KB files
kb_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
if kb_files:
    st.sidebar.subheader("📑 Current KB Files")
    for f in kb_files:
        st.sidebar.write(f"📄 {f}")
else:
    st.sidebar.warning("⚠️ No PDFs found in knowledge_base folder.")

# File uploader for new PDFs
uploaded_files = st.sidebar.file_uploader(
    "Upload additional PDF(s)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for uf in uploaded_files:
        save_path = os.path.join(DATA_DIR, uf.name)
        try:
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
            st.sidebar.success(f"✅ Added: {uf.name}")
        except Exception as e:
            log_exception("File save failed", e)
            st.sidebar.error(f"Failed to save {uf.name}. See logs.")

    # Rebuild KB (safely) and store BM25/texts even if LLM missing
    try:
        retriever, bm25_index, kb_texts = load_knowledge_base()
        st.session_state.bm25_index = bm25_index
        st.session_state.kb_texts = kb_texts
        if retriever:
            # only create QA chain if LLM + prompt exist
            if st.session_state.get("llm") and st.session_state.get("prompt"):
                try:
                    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=st.session_state.llm,
                        retriever=retriever,
                        return_source_documents=True,
                        combine_docs_chain_kwargs={"prompt": st.session_state.prompt}
                    )
                    st.sidebar.success("🔄 Knowledge Base reloaded with new files and LLM attached!")
                except Exception as e:
                    log_exception("QA chain init after upload failed", e)
                    st.sidebar.error("Error initializing QA chain after upload. Check logs (Show last error).")
            else:
                st.sidebar.info(
                    "Knowledge base rebuilt. LLM not initialized — set GROQ_API_KEY or click 'Initialize LLM' below."
                )
        else:
            st.sidebar.warning("No retriever could be built from uploaded files.")
    except Exception as e:
        log_exception("KB reload after upload failed", e)
        st.sidebar.error("Failed to rebuild knowledge base. See logs (Show last error).")

st.sidebar.markdown("---")

# -------------------------------
# Initialize Session
# -------------------------------
if "qa_chain" not in st.session_state:
    try:
        retriever, bm25_index, kb_texts = load_knowledge_base()
    except Exception as e:
        log_exception("Initial KB load failed", e)
        retriever, bm25_index, kb_texts = None, None, None

    if retriever:
        groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not groq_api_key:
            st.error(
                "⚠️ GROQ_API_KEY is missing.\n\n"
                "How to fix:\n"
                "- Add GROQ_API_KEY in Streamlit app Secrets (App settings → Secrets) OR\n"
                "- Set environment variable GROQ_API_KEY on the host/container.\n\n"
                "After adding the key, either reload the app or click 'Initialize LLM' in the sidebar."
            )
            # still expose BM25/texts so suggestions work without LLM
            st.session_state.bm25_index = bm25_index
            st.session_state.kb_texts = kb_texts
            st.session_state.qa_chain = None
        else:
            try:
                llm = ChatGroq(
                    api_key=groq_api_key,
                    model="openai/gpt-oss-20b",
                    temperature=0,
                    max_tokens=512
                )
                prompt_template = """Use the following context to answer the user’s question.
If you don’t know, just say so. Do not add extra info.

{context}

Question: {question}
Answer:"""
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm,
                    retriever=retriever,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": PROMPT}
                )
                st.session_state.llm = llm
                st.session_state.prompt = PROMPT
                st.session_state.qa_chain = qa_chain
                st.session_state.bm25_index = bm25_index
                st.session_state.kb_texts = kb_texts
                st.success("✅ LLM initialized and knowledge base attached.")
            except Exception as e:
                log_exception("LLM / QA chain initialization failed", e)
                st.error("Failed to initialize LLM or QA chain. Click 'Show last error' in the sidebar for details.")
    else:
        st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Sidebar Controls: Initialize LLM / Show last error
# -------------------------------
if st.sidebar.button("Initialize LLM"):
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not groq_api_key:
            st.sidebar.error("GROQ_API_KEY not found. Add it to Streamlit Secrets or env vars, then try again.")
        else:
            retriever = None
            # use existing retriever in session if available, otherwise rebuild
            if st.session_state.get("bm25_index") and st.session_state.get("kb_texts"):
                try:
                    retriever, bm25_index, kb_texts = load_knowledge_base()
                except Exception as e:
                    log_exception("Load KB during Initialize LLM failed", e)
            if not retriever:
                st.sidebar.warning("No documents found to build KB. Upload PDFs first.")
            else:
                try:
                    llm = ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b", temperature=0, max_tokens=512)
                    PROMPT = st.session_state.get("prompt") or PromptTemplate(
                        template="""Use the following context to answer the user’s question.
If you don’t know, just say so. Do not add extra info.

{context}

Question: {question}
Answer:""", input_variables=["context", "question"]
                    )
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm, retriever=retriever, return_source_documents=True, combine_docs_chain_kwargs={"prompt": PROMPT}
                    )
                    st.session_state.llm = llm
                    st.session_state.prompt = PROMPT
                    st.session_state.qa_chain = qa_chain
                    st.session_state.bm25_index = bm25_index
                    st.session_state.kb_texts = kb_texts
                    st.sidebar.success("✅ LLM initialized and QA chain created.")
                except Exception as e:
                    log_exception("Initialize LLM button failed", e)
                    st.sidebar.error("Failed to initialize LLM. See logs (Show last error).")
    except Exception as e:
        log_exception("Initialize LLM outer failure", e)
        st.sidebar.error("Unexpected error while initializing LLM. See logs (Show last error).")

if st.sidebar.button("Show last error"):
    last = st.session_state.get("last_error")
    if last:
        st.sidebar.text_area("Last captured exception", value=last, height=300)
        try:
            with open(LOG_FILE, "rb") as lf:
                st.sidebar.download_button("Download full log", lf, file_name=os.path.basename(LOG_FILE))
        except Exception as e:
            log_exception("Failed to open log file", e)
            st.sidebar.error("Unable to read log file.")
    else:
        st.sidebar.info("No errors captured in this session.")

st.sidebar.markdown("---")

# -------------------------------
# Chat Input
# -------------------------------
query = st.text_input("💬 Ask a question about patents in India:")

if query and st.session_state.qa_chain:
    meta_response, is_meta = handle_history_question(query, st.session_state.chat_history)
    if is_meta:
        answer = meta_response
        sources = []
        related = []
    else:
        try:
            # Use the chain to get an answer and source documents
            result = st.session_state.qa_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
            raw_answer = result.get("answer", "")
            answer = clean_answer(raw_answer)
            sources = []
            for doc in result.get("source_documents", []):
                source_file = os.path.basename(doc.metadata.get("source", ""))
                page = doc.metadata.get("page", "N/A")
                sources.append(f"{source_file} (Page {page})")
            related = suggest_related_questions(query, st.session_state.bm25_index, st.session_state.kb_texts)
        except Exception as e:
            log_exception("Runtime QA invocation failed", e)
            answer = "Sorry — an error occurred while generating an answer. Click 'Show last error' in the sidebar for details."
            sources = []
            related = []

    st.session_state.chat_history.append((query, answer))
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Bot:** {answer}")

    if sources:
        with st.expander("📖 Sources"):
            for s in sources:
                st.caption(f"{s} — Source URL: {SOURCE_URL}")

    if related:
        st.info("💡 Suggested Related Questions:")
        for rq in related:
            st.markdown(f"- {rq}")

# -------------------------------
# Display Chat History
# -------------------------------
if st.session_state.chat_history:
    st.subheader("📝 Conversation History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown(" ")

st.markdown("---")
st.caption("Patent FAQ Chatbot • Powered by Groq & LangChain • Strictly based on provided KB documents")
