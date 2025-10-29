import os
import re
import hashlib
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi


# ---------------------- STREAMLIT CONFIG ----------------------
st.set_page_config(page_title="Patent FAQ Chatbot", layout="wide")
st.title("📘 Patent FAQ Chatbot")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------- FUNCTION: LOAD PDF ----------------------
def load_and_split_pdf(pdf_path):
    """Load and split a PDF into text chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


# ---------------------- FUNCTION: CREATE VECTOR DB ----------------------
def create_vector_db(documents):
    """Create FAISS vector database from documents."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents, embedding=embeddings)
    return vector_db


# ---------------------- FUNCTION: CREATE BM25 INDEX ----------------------
def create_bm25_index(documents):
    """Create a BM25 index for keyword-based retrieval."""
    corpus = [doc.page_content for doc in documents]
    tokenized_corpus = [re.findall(r"\w+", text.lower()) for text in corpus]
    return BM25Okapi(tokenized_corpus), corpus


# ---------------------- FUNCTION: GET RETRIEVER ----------------------
def get_retriever(vector_db, bm25_index, corpus, query):
    """Combine semantic (FAISS) and keyword (BM25) retrieval."""
    tokenized_query = re.findall(r"\w+", query.lower())
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:3]
    bm25_docs = [corpus[i] for i in top_bm25_idx]

    faiss_docs = [d.page_content for d in vector_db.similarity_search(query, k=3)]
    all_docs = list(set(faiss_docs + bm25_docs))
    return all_docs


# ---------------------- FUNCTION: CREATE LLM ----------------------
def create_llm():
    """Initialize Groq LLM (or any other supported model)."""
    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables or Streamlit secrets.")
        st.stop()
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, groq_api_key=groq_api_key)


# ---------------------- MAIN WORKFLOW ----------------------
uploaded_pdf = st.file_uploader("📂 Upload a Patent FAQ PDF", type=["pdf"])

if uploaded_pdf:
    with st.spinner("Processing document..."):
        temp_path = f"temp_{hashlib.md5(uploaded_pdf.name.encode()).hexdigest()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.read())

        documents = load_and_split_pdf(temp_path)
        vector_db = create_vector_db(documents)
        bm25_index, corpus = create_bm25_index(documents)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        llm = create_llm()
        qa_chain = create_retrieval_chain(retriever, llm)

        st.success("✅ Document processed successfully!")

    # Chat UI
    user_input = st.text_input("💬 Ask a question about the document:")

    if user_input:
        with st.spinner("Thinking..."):
            relevant_docs = get_retriever(vector_db, bm25_index, corpus, user_input)
            context = "\n\n".join(relevant_docs[:3])
            prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {user_input}"
            response = llm.invoke(prompt)

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response.content))

        # Display chat
        for speaker, text in st.session_state.chat_history[-6:]:
            if speaker == "You":
                st.markdown(f"🧑‍💬 **You:** {text}")
            else:
                st.markdown(f"🤖 **Bot:** {text}")
