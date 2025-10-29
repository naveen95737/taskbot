import os
import re
import hashlib
import streamlit as st

# LangChain + Ecosystem Imports (for LangChain 1.0+)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Keyword Search
from rank_bm25 import BM25Okapi


# -------------------------------
# Config
# -------------------------------
DATA_DIR = "knowledge_base"
os.makedirs(DATA_DIR, exist_ok=True)

SOURCE_URL = "http://www.ipindia.gov.in/ (FREQUENTLY ASKED QUESTIONS - PATENTS)"

# -------------------------------
# Streamlit Page
# -------------------------------
st.set_page_config(page_title="Patent FAQ Chatbot", page_icon="📘", layout="wide")
st.title("📘 Patent FAQ Chatbot (India)")

# Sidebar for KB Management
st.sidebar.header("📂 Knowledge Base Management")
uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for uf in uploaded_files:
        save_path = os.path.join(DATA_DIR, uf.name)
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
    st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s).")

# Show current KB files
kb_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
if kb_files:
    st.sidebar.subheader("📑 Current KB Files")
    for f in kb_files:
        st.sidebar.write(f"📄 {f}")
else:
    st.sidebar.warning("⚠️ No PDFs uploaded yet.")

# -------------------------------
# Load Knowledge Base
# -------------------------------
def load_knowledge_base():
    """Load PDFs, create FAISS index, return retriever + BM25 index."""
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())

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
# Handle Meta-History Questions - FIXED
# -------------------------------
def handle_history_question(query: str, chat_history: list) -> tuple[str | None, bool]:
    """Return only the immediate previous question/answer instead of full history."""
    query_lower = query.lower().strip()

    if "previous" in query_lower or "last" in query_lower:
        if not chat_history:
            return "No previous question available yet.", True
        
        last_q, last_a = chat_history[-1]  # Just the last turn

        if "answer" in query_lower or "reply" in query_lower:
            return f"Your previous question was: '{last_q}'\n\nAnd my answer was: {last_a}", True
        else:
            return f"Your previous question was: '{last_q}'", True

    return None, False


# -------------------------------
# Related Question Suggestions - FIXED for Gibberish
# -------------------------------
def suggest_related_questions(query, bm25_index, kb_texts, top_n=3):
    if not bm25_index or not kb_texts:
        return []
    
    keywords = re.findall(r"\w+", query.lower())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'type', 'text', 'updated', 'revision', 'june', 'part', 'frequently', 'asked', 'questions', '[type', 'text]'}
    keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 2]
    
    if not keywords:
        return []
    
    scores = bm25_index.get_scores(keywords)
    ranked = sorted(zip(scores, kb_texts), key=lambda x: x[0], reverse=True)
    
    suggestions = []
    for _, text in ranked[:top_n * 3]:  # More candidates to filter gibberish
        # Split into sentences and filter meaningful ones
        sentences = re.split(r'[.!?]+', text)
        for s in sentences:
            s_clean = re.sub(r'[^a-zA-Z0-9\s\?\!]', '', s.strip())  # Clean artifacts
            if (len(s_clean) > 40 and len(s_clean) < 150 and 
                any(kw in s_clean.lower() for kw in keywords) and 
                not any(artifact in s_clean.lower() for artifact in ['type text', 'updated revision', 'june 2018', 'part - generic']) and
                s_clean not in suggestions):
                # Format as question if not already
                if not s_clean.endswith('?'):
                    s_clean += '?'
                suggestions.append(s_clean)
                break  # One per chunk
        
        if len(suggestions) >= top_n:
            break
    
    return suggestions[:top_n] if suggestions else ["What is the patent application procedure?", "How to file a patent in India?", "What are patent fees?"]  # Fallback

# -------------------------------
# Clean Answer
# -------------------------------
def clean_answer(text):
    if not text.strip():
        return "I don't have a specific answer in the knowledge base."
    # Remove gibberish/artifacts
    text = re.sub(r'\[type text\]|\(revision \d+\)|updated: june \d+|part - generic issues', '', text)
    sentences = [s.strip() for s in text.split(".") if s.strip() and len(s.strip()) > 10]
    seen, cleaned = set(), []
    for s in sentences:
        if s not in seen:
            cleaned.append(s)
            seen.add(s)
    return ". ".join(cleaned) + "." if cleaned else text.strip()[:300]

# -------------------------------
# Initialize Session
# -------------------------------
if "qa_chain" not in st.session_state:
    retriever, bm25_index, kb_texts = load_knowledge_base()
    if retriever:
        groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not groq_api_key:
            st.error("⚠️ Please set your GROQ_API_KEY in Streamlit secrets or environment variables.")
        else:
            llm = ChatGroq(
                    api_key=groq_api_key,
                    model="openai/gpt-oss-20b",   # ✅ supported model
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

            st.session_state.qa_chain = qa_chain
            st.session_state.bm25_index = bm25_index
            st.session_state.kb_texts = kb_texts
    else:
        st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Chat Input
# -------------------------------
query = st.text_input("💬 Ask a question about patents in India:")

if query and st.session_state.qa_chain:
    # Check for meta-history question first - FIXED
    meta_response, is_meta = handle_history_question(query, st.session_state.chat_history)
    if is_meta:
        answer = meta_response
        sources = []
        related = []
    else:
        # Use full history for context in chain
        result = st.session_state.qa_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
        raw_answer = result["answer"]
        answer = clean_answer(raw_answer)

        sources = []
        for doc in result.get("source_documents", []):
            source_file = os.path.basename(doc.metadata.get("source", ""))
            page = doc.metadata.get("page", "N/A")
            sources.append(f"{source_file} (Page {page})")

        related = suggest_related_questions(query, st.session_state.bm25_index, st.session_state.kb_texts)

    st.session_state.chat_history.append((query, answer))

    # Display response
    st.markdown(f"**You:** {query}")
    st.markdown(f"**Bot:** {answer}")

    if sources:
        with st.expander("📖 Sources"):
            for s in sources:
                st.caption(f"{s} — Source URL: {SOURCE_URL}")

    if related:
        st.info("💡 Suggested Related Questions:")
        for rq in related:
            st.markdown(f"- {rq}")  # Non-clickable text

# -------------------------------
# Display Chat History - LIFO (Newest First)
# -------------------------------
if st.session_state.chat_history:
    st.subheader("📝 Conversation History")
    for q, a in reversed(st.session_state.chat_history):  # LIFO: Newest first
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown(" ")  # Minimal spacing

st.markdown("---")

st.caption("Patent FAQ Chatbot • Powered by Groq & LangChain • Strictly based on provided KB documents")

