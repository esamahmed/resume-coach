"""
app/rag/vector_store.py

FAISS vector store for RAG-powered conversational Q&A.

The Q&A agent uses this to answer questions like:
  "Why didn't you flag my Python skills?"
  "Which of my bullets should I rewrite first?"

Rather than re-reading the full resume+JD on every turn, the Q&A
agent retrieves the 3-5 most relevant chunks — much cheaper and more
precise than full-context repetition.

Pattern adapted from the AI Incident Investigation project (Bedrock + FAISS).
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import get_settings

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


def _build_embeddings():
    settings = get_settings()
    provider = settings.embedding_provider
    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(
            model_id=settings.bedrock_embedding_model_id,
            region_name=settings.bedrock_region,
        )
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model_openai,
        )
    # default: ollama
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model_ollama,
    )


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple character-level chunker with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 20]


def build_session_index(
    resume_text: str,
    jd_text: str,
    gap_analysis: dict | None = None,
    coaching_report: dict | None = None,
) -> "FAISS":
    """
    Build an in-memory FAISS index for a single coaching session.

    Indexes:
      - Resume chunks
      - JD chunks
      - Gap analysis findings (if available)
      - Coaching recommendations (if available)

    The Q&A agent queries this so it can ground its answers in
    the actual session content rather than hallucinating.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    settings = get_settings()
    embeddings = _build_embeddings()

    documents: list[Document] = []

    # Resume chunks
    for i, chunk in enumerate(_chunk_text(
        resume_text,
        settings.embedding_chunk_size,
        settings.embedding_chunk_overlap,
    )):
        documents.append(Document(
            page_content=chunk,
            metadata={"source": "resume", "chunk_id": i},
        ))

    # JD chunks
    for i, chunk in enumerate(_chunk_text(
        jd_text,
        settings.embedding_chunk_size,
        settings.embedding_chunk_overlap,
    )):
        documents.append(Document(
            page_content=chunk,
            metadata={"source": "jd", "chunk_id": i},
        ))

    # Gap findings (serialised to text for embedding)
    if gap_analysis:
        for section, content in gap_analysis.items():
            text = f"Gap analysis — {section}: {content}"
            documents.append(Document(
                page_content=text,
                metadata={"source": "gap_analysis", "section": section},
            ))

    # Coaching recommendations
    if coaching_report:
        recs = coaching_report.get("recommendations", [])
        for i, rec in enumerate(recs):
            documents.append(Document(
                page_content=str(rec),
                metadata={"source": "coaching_report", "rec_id": i},
            ))

    logger.info("Building FAISS index: %d documents", len(documents))
    index = FAISS.from_documents(documents, embeddings)
    return index


def retrieve(index: "FAISS", query: str, k: int | None = None) -> list[str]:
    """Retrieve top-k relevant chunks for a query."""
    settings = get_settings()
    top_k = k or settings.rag_top_k
    docs = index.similarity_search(query, k=top_k)
    return [d.page_content for d in docs]
