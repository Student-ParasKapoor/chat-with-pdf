# 📄 Chat with PDF

Ask questions about any PDF using AI.

**Live Demo:** [chat-with-pdf-by-paras.streamlit.app](https://chat-with-pdf-by-paras.streamlit.app/)

## Stack
- Groq LLaMA 3 (LLM)
- ChromaDB (vector store)
- Sentence Transformers (embeddings)
- Streamlit (UI)

## Run Locally
1. Clone the repo
2. Create `.env` with `GROQ_API_KEY=your_key`
3. `pip install -r requirements.txt`
4. `streamlit run app.py`