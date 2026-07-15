"""Local FAISS similarity engine for ScholarMind.

This module keeps an in-process index for the current Streamlit session/process.
The OpenAI API key is passed per request; it is never stored globally.
"""

from __future__ import annotations

import os
import textwrap
from typing import List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

DIMENSION = int(os.getenv("OPENAI_EMBEDDING_DIMENSION", "1536"))
MAX_CHUNK_SIZE = 1000
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

faiss_index = faiss.IndexFlatL2(DIMENSION)
stored_chunks: List[Tuple[str, str]] = []


def _validate_api_key(api_key: str) -> str:
    cleaned = (api_key or "").strip()
    if not cleaned:
        raise ValueError("OpenAI API anahtarı boş olamaz.")
    return cleaned


def get_embedding(text: str, api_key: str) -> np.ndarray:
    """Convert text to a normalized OpenAI embedding vector."""
    cleaned = " ".join((text or "").replace("\n", " ").split())
    if not cleaned:
        raise ValueError("Embedding üretmek için boş metin verilemez.")

    client = OpenAI(api_key=_validate_api_key(api_key))
    response = client.embeddings.create(
        input=[cleaned],
        model=EMBEDDING_MODEL,
    )
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)

    if vector.shape != (DIMENSION,):
        raise ValueError(
            f"Embedding boyutu beklenenden farklı: {vector.shape[0]} != {DIMENSION}"
        )

    # Cosine similarity can be implemented through L2 distance after normalization.
    faiss.normalize_L2(vector.reshape(1, -1))
    return vector


def chunk_text(text: str, max_length: int = MAX_CHUNK_SIZE) -> List[str]:
    """Split text into non-empty character-bounded chunks."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    return textwrap.wrap(
        cleaned,
        width=max_length,
        break_long_words=False,
        break_on_hyphens=False,
    )


def reset_index() -> None:
    """Clear the temporary FAISS index before a new search result set."""
    global faiss_index, stored_chunks
    faiss_index = faiss.IndexFlatL2(DIMENSION)
    stored_chunks = []


def add_text_to_index(
    text: str,
    source_id: str = "unknown",
    api_key: str = "",
) -> int:
    """Chunk, embed and add text to the temporary FAISS index."""
    chunks = chunk_text(text)
    if not chunks:
        return 0

    source = (source_id or "unknown").strip()
    vectors = [get_embedding(chunk, api_key=api_key) for chunk in chunks]
    matrix = np.vstack(vectors).astype(np.float32)

    faiss_index.add(matrix)
    stored_chunks.extend((chunk, source) for chunk in chunks)
    return len(chunks)


def search_similar(
    text: str,
    top_k: int = 3,
    api_key: str = "",
    exclude_source_id: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return distinct similar sources, optionally excluding the current paper."""
    if faiss_index.ntotal == 0 or not stored_chunks:
        return []

    requested = max(1, int(top_k))
    # Ask for extra neighbours because some may belong to the current source
    # or be duplicate chunks from the same paper.
    candidate_count = min(
        faiss_index.ntotal,
        max(requested * 5, requested),
    )

    query_vector = get_embedding(text, api_key=api_key).reshape(1, -1)
    _, indices = faiss_index.search(query_vector, candidate_count)

    excluded = (exclude_source_id or "").strip()
    seen_sources = set()
    matches: List[Tuple[str, str]] = []

    for raw_index in indices[0]:
        index = int(raw_index)
        if index < 0 or index >= len(stored_chunks):
            continue

        chunk, source = stored_chunks[index]

        if excluded and source == excluded:
            continue
        if source in seen_sources:
            continue

        seen_sources.add(source)
        matches.append((chunk, source))

        if len(matches) >= requested:
            break

    return matches


def suggest_topics_based_on_text(
    text: str,
    api_key: str,
    model: str = "gpt-4o",
) -> List[str]:
    """Suggest five academic research topics based on the supplied text."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []

    client = OpenAI(api_key=_validate_api_key(api_key))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen akademik araştırma konuları öneren bir uzmansın. "
                    "Önerileri numaralı ve birbirinden farklı biçimde üret."
                ),
            },
            {
                "role": "user",
                "content": f"Bu metne göre 5 akademik araştırma konusu öner:\n\n{cleaned}",
            },
        ],
    )

    content = response.choices[0].message.content.strip()
    return [line.strip() for line in content.splitlines() if line.strip()]
