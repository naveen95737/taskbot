# 📘 Patent FAQ Chatbot

**A conversational AI chatbot that answers Indian patent-related queries using PDF knowledge base documents.**

---

## Overview
The **Patent FAQ Chatbot** is designed to provide accurate, knowledge-base-driven responses about Indian patent procedures.  
It strictly uses the provided PDFs and does not generate answers outside the knowledge base. Users can upload PDFs, ask questions, and receive answers with sources and related FAQ suggestions.

---

## Features

- **KB-only Answers:** Responses strictly come from uploaded PDFs; no external generation.  
- **Upload & Reload PDFs:** Easily manage and update knowledge base documents.  
- **Related Questions:** Keyword-based suggestions with clickable buttons for instant retrieval.  
- **Conversation History:** Maintains previous Q&A interactions for continuity.  
- **Source Attribution:** Each answer shows the official source URL.  
- **Answer Cleanup:** Removes duplicates and incomplete sentences for readability.  

---

## Tech Stack

- **Python** – Core programming language  
- **Streamlit** – Interactive front-end  
- **LangChain** – Document retrieval orchestration  
- **FAISS** – Vector store for semantic search  
- **HuggingFace Embeddings** – Sentence embeddings  
- **BM25Okapi** – Keyword-based related question retrieval  
- **Transformers (FLAN-T5)** – Lightweight LLM for coherent answer formatting  

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/username/patent-faq-chatbot.git
   cd patent-faq-chatbot
