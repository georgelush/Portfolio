import logging
import os
from typing import Callable, Coroutine, Any, Optional

import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_collection = None
_embedder: Optional[SentenceTransformer] = None

_KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "portfolio_data.txt")

_RAG_SYSTEM = """You are a helpful assistant for George Rusu's AI portfolio.
Answer the visitor's question using ONLY the context below.
Be concise, warm, and direct. If the answer isn't in the context, say so honestly.

Context:
{context}"""


def _chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def init_rag() -> None:
    global _collection, _embedder

    logger.info("Loading sentence-transformer model…")
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Reading knowledge base…")
    with open(_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = _chunk_text(text)
    embeddings = _embedder.encode(chunks, show_progress_bar=False).tolist()

    client = chromadb.EphemeralClient()
    _collection = client.create_collection("portfolio")
    _collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"c{i}" for i in range(len(chunks))],
    )
    logger.info(f"RAG ready — {len(chunks)} chunks indexed.")


async def _to_english(query: str) -> str:
    """Translate the query to English so the English-only embedder retrieves correctly."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    resp = await llm.ainvoke([
        SystemMessage(content="Translate the message to English. Reply with ONLY the English translation. If it is already English, return it unchanged."),
        HumanMessage(content=query),
    ])
    return resp.content.strip()


async def run_rag(
    query: str,
    on_progress: Optional[Callable[[str], Coroutine[Any, Any, None]]] = None,
    top_k: int = 4,
) -> str:
    if _collection is None or _embedder is None:
        return "Knowledge base not initialised. Please restart the service."

    if on_progress:
        await on_progress("Searching for relevant information…")

    search_query = await _to_english(query)
    q_vec = _embedder.encode([search_query], show_progress_bar=False)[0].tolist()
    results = _collection.query(query_embeddings=[q_vec], n_results=top_k)
    docs = results["documents"][0]
    context = "\n\n---\n\n".join(docs)

    if on_progress:
        await on_progress("Found what I need — writing your answer…")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    response = await llm.ainvoke([
        SystemMessage(content=_RAG_SYSTEM.format(context=context)),
        HumanMessage(content=query),
    ])
    return response.content
