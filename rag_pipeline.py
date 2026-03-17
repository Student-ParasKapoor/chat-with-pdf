import os
from typing import List, Tuple, Dict
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq


# ── PDF Loading ───────────────────────────────────────────────────────────────

def load_pdf(path: str) -> Tuple[List[Dict], int]:
    """Load PDF pages as text dicts. Tries normal text extraction first."""
    doc = fitz.open(path)
    num_pages = doc.page_count
    file_name = os.path.basename(path)
    pages = []

    for i in range(num_pages):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "text": text,
                "page": i + 1,
                "source": file_name,
            })

    doc.close()
    return pages, num_pages


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(pages: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Split page texts into smaller overlapping chunks."""
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                    "source": page["source"],
                })
            start += chunk_size - overlap
    return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────

def build_embeddings_model() -> SentenceTransformer:
    """Load sentence-transformers model locally (free, no API needed)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Vector Store ──────────────────────────────────────────────────────────────

def build_vectorstore(chunks: List[Dict], model: SentenceTransformer):
    """Embed chunks and store in in-memory ChromaDB."""

    if not chunks:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned/image-based PDF. "
            "Please try a PDF with selectable text."
        )

    client = chromadb.Client()

    try:
        client.delete_collection("pdf_chunks")
    except Exception:
        pass

    collection = client.create_collection("pdf_chunks")

    texts = [c["text"] for c in chunks]

    # Encode and explicitly convert to plain Python list
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings_list = [e.tolist() for e in embeddings]

    if not embeddings_list:
        raise ValueError("Embedding generation failed — got empty embeddings.")

    collection.add(
        documents=texts,
        embeddings=embeddings_list,
        metadatas=[{"page": c["page"], "source": c["source"]} for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    return collection, client


# ── Answer Question ───────────────────────────────────────────────────────────

def answer_question(
    question: str,
    collection,
    model: SentenceTransformer,
    groq_api_key: str,
) -> Tuple[str, List[Dict]]:
    """Retrieve relevant chunks and generate answer using Groq LLM."""

    # 1. Embed question
    question_embedding = model.encode([question])[0].tolist()

    # 2. Retrieve top 3 chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(3, collection.count()),
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    # 3. Build context
    context = "\n\n".join(docs)

    # 4. Call Groq
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question "
                    "using ONLY the context provided. If the answer is not in "
                    "the context, say 'I could not find this in the document'.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content
    return answer, metadatas