"""ChromaDB research history for ScholarMind.

The user's OpenAI API key is passed per operation. It is not read or cached at
module import time.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

try:
    import chromadb
except Exception:
    chromadb = None

COLLECTION_NAME = "research_history"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_STORAGE_PATH = os.getenv("CHROMA_STORAGE_PATH", "chroma_storage")


class ChromaUnavailableError(RuntimeError):
    """Raised when ChromaDB cannot be used."""


def _validate_api_key(api_key: str) -> str:
    cleaned = (api_key or "").strip()
    if not cleaned:
        raise ValueError("OpenAI API anahtarı boş olamaz.")
    return cleaned


def _embedding(text: str, api_key: str) -> list[float]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        raise ValueError("Embedding üretmek için boş metin verilemez.")

    client = OpenAI(api_key=_validate_api_key(api_key))
    response = client.embeddings.create(
        input=[cleaned],
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


@lru_cache(maxsize=1)
def _get_collection():
    """Create the persistent Chroma client lazily."""
    if chromadb is None:
        raise ChromaUnavailableError(
            "ChromaDB kurulamadı veya bu ortamda desteklenmiyor."
        )

    storage_path = Path(CHROMA_STORAGE_PATH)
    storage_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(storage_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_memory(
    id: str,
    content: str,
    api_key: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Insert or update one research-history document."""
    document_id = (id or "").strip()
    cleaned_content = " ".join((content or "").split())

    if not document_id:
        raise ValueError("Chroma kayıt ID'si boş olamaz.")
    if not cleaned_content:
        raise ValueError("Chroma'ya boş içerik kaydedilemez.")

    collection = _get_collection()
    vector = _embedding(cleaned_content, api_key)

    # upsert avoids duplicate-ID failures when the same search is run again.
    collection.upsert(
        ids=[document_id],
        documents=[cleaned_content],
        embeddings=[vector],
        metadatas=[metadata or {}],
    )
    return True


def search_memory(
    query: str,
    api_key: str,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Search saved research history semantically."""
    cleaned_query = " ".join((query or "").split())
    if not cleaned_query:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    vector = _embedding(cleaned_query, api_key)
    return collection.query(
        query_embeddings=[vector],
        n_results=min(max(1, int(top_k)), count),
        include=["documents", "metadatas", "distances"],
    )
