import os
import re
import hashlib
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from rank_bm25 import BM25Okapi


# ---------------------- STREAMLIT CONFIG ----------------------
st.set_page_config(page_title="Patent FAQ Chatbot", layout="wide")
st.title("📘 Patent FAQ Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------- LOAD PDF ----------------------
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


# ---------------------- BUILD VECTORSTORE ----------------------
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


# ---------------------- BM25 SETUP ----------------------
def create_bm25_index(docs):
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [re.findall(r"\w+", text.lower()) for text in corpus]
    return BM25Okapi(tokenized_corpus), corpus


# ---------------------- LLM ----------------------
def load_llm():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not api_key:
        st.error("❌ Missing GROQ_API_KEY in Streamlit secrets or environment.")
        st.stop()
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, groq_api_key=api_key)


# ---------------------- MAIN ----------------------
uploaded_file = st.file_uploader("📂 Upload your Patent FAQ PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        path = f"temp_{hashlib.md5(uploaded_file.name.encode()).hexdigest()}.pdf"
        with open(path, "wb") as f:
            f.write(uploaded_file.read())

        docs = load_and_split_pdf(path)
        vector_db = build_vectorstore(docs)
        bm25_index, corpus = create_bm25_index(docs)
        llm = load_llm()

        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        st.success("✅ Knowledge base ready!")

    query = st.text_input("💬 Ask a question about patents:")

    if query:
        with st.spinner("Searching knowledge base..."):
            result = qa_chain.invoke({"query": query})
            answer = result["result"].strip()
            sources = [doc.metadata.get("source", "Uploaded PDF") for doc in result["source_documents"]]

        st.session_state.chat_history.append((query, answer, sources))

    for q, a, src in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        with st.expander("📖 Sources"):
            for s in src:
                st.caption(s)
        st.markdown("---")
