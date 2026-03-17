import os
import tempfile
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_pdf,
    chunk_documents,
    build_embeddings_model,
    build_vectorstore,
    answer_question,
)

# ── Setup ─────────────────────────────────────────────────────────────────────

load_dotenv()


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = None
    if "collection" not in st.session_state:
        st.session_state.collection = None
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = None
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "pdf_pages" not in st.session_state:
        st.session_state.pdf_pages = None


def reset_pdf_state():
    st.session_state.collection = None
    st.session_state.chroma_client = None
    st.session_state.pdf_name = None
    st.session_state.pdf_pages = None


def index_pdf(path: str, file_name: str):
    with st.spinner("Reading and indexing PDF..."):
        pages, num_pages = load_pdf(path)
        chunks = chunk_documents(pages)

        # Load embeddings model only once
        if st.session_state.model is None:
            with st.spinner("Loading embedding model (first time only)..."):
                st.session_state.model = build_embeddings_model()

        collection, client = build_vectorstore(chunks, st.session_state.model)
        st.session_state.collection = collection
        st.session_state.chroma_client = client
        st.session_state.pdf_name = file_name
        st.session_state.pdf_pages = num_pages


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("📄 PDF Info")
        if st.session_state.pdf_name:
            st.success("PDF loaded!")
            st.markdown(f"**File:** {st.session_state.pdf_name}")
            st.markdown(f"**Pages:** {st.session_state.pdf_pages}")
        else:
            st.info("No PDF uploaded yet.")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.title("📄 Chat with PDF")
    st.caption("Ask anything about your PDF — powered by Groq LLaMA3 + local embeddings.")

    init_session_state()
    render_sidebar()

    # Check API key
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file:\n\nGROQ_API_KEY=your_key_here")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            reset_pdf_state()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                index_pdf(tmp_path, uploaded_file.name)
                st.success(f"✅ Ready! Ask questions about **{uploaded_file.name}**")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        st.info("👆 Upload a PDF above to get started.")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if st.session_state.collection is None:
        if prompt := st.chat_input("Upload a PDF first..."):
            st.warning("Please upload a PDF before asking questions.")
        return

    if prompt := st.chat_input("Ask a question about your PDF..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(
                    question=prompt,
                    collection=st.session_state.collection,
                    model=st.session_state.model,
                    groq_api_key=groq_api_key,
                )
                st.markdown(answer)

                # Show sources
                if sources:
                    pages = sorted(set(s["page"] for s in sources))
                    pages_str = ", ".join(str(p) for p in pages)
                    st.markdown(f"_📖 Sources: page(s) {pages_str}_")

        full_response = answer + f"\n\n_📖 Sources: page(s) {', '.join(str(s['page']) for s in sources)}_"
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()