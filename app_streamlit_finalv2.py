import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate  # Kept for potential future use; not central now
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    st.error("Missing 'rank_bm25' package. Install via 'pip install rank_bm25'.")
    BM25Okapi = None


# -------------------------------
# Config
# -------------------------------
DATA_DIR = "knowledge_base"
os.makedirs(DATA_DIR, exist_ok=True)

SOURCE_URL = "http://www.ipindia.gov.in/ (FREQUENTLY ASKED QUESTIONS - PATENTS)"

st.set_page_config(page_title="Patent FAQ Chatbot", page_icon="📘", layout="wide")
st.title("📘 Patent FAQ Chatbot (India)")

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
# Create LCEL QA Chain
# -------------------------------
def create_convo_qa_chain(llm, retriever):
    """Create the LCEL conversational retrieval chain."""
    # Condense question prompt (rephrase using history)
    condense_question_system_template = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # History-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    # QA prompt (adapted from original: context + question, with history)
    system_prompt = (
        "Use the following pieces of retrieved context to answer the user’s question. "
        "If you don’t know, just say so. Do not add extra info. "
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Stuff documents chain
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full retrieval chain
    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return convo_qa_chain

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
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        st.sidebar.success(f"✅ Added: {uf.name}")
    # Reload KB instantly
    retriever, bm25_index, kb_texts = load_knowledge_base()
    if retriever and "llm" in st.session_state:
        try:
            st.session_state.qa_chain = create_convo_qa_chain(st.session_state.llm, retriever)
            st.session_state.bm25_index = bm25_index
            st.session_state.kb_texts = kb_texts
            st.sidebar.success("🔄 Knowledge Base reloaded with new files!")
        except Exception as e:
            st.sidebar.error(f"❌ Reload failed: {str(e)}")

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
            try:
                llm = ChatGroq(
                        api_key=groq_api_key,
                        model="openai/gpt-oss-20b",
                        temperature=0,
                        max_tokens=512
                        )
                qa_chain = create_convo_qa_chain(llm, retriever)
                st.session_state.llm = llm
                st.session_state.qa_chain = qa_chain
                st.session_state.bm25_index = bm25_index
                st.session_state.kb_texts = kb_texts
            except Exception as e:
                st.error(f"❌ Failed to initialize QA chain: {str(e)}. Check logs for details.")
    else:
        st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        # Convert chat_history to messages
        chat_history_msgs = []
        for human, ai in st.session_state.chat_history:
            chat_history_msgs.extend([HumanMessage(content=human), AIMessage(content=ai)])
        
        result = st.session_state.qa_chain.invoke(
            {
                "input": query,
                "chat_history": chat_history_msgs,
            }
        )
        raw_answer = result["answer"]
        answer = clean_answer(raw_answer)
        sources = []
        for doc in result.get("context", []):
            source_file = os.path.basename(doc.metadata.get("source", ""))
            page = doc.metadata.get("page", "N/A")
            sources.append(f"{source_file} (Page {page})")
        related = suggest_related_questions(query, st.session_state.bm25_index, st.session_state.kb_texts)

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
st.caption("Patent FAQ Chatbot • Powered by Groq & LangChain (LCEL) • Strictly based on provided KB documents")
