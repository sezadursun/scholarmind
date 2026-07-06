"""Milvus / Zilliz Cloud memory engine for ScholarMind.

Place this file at: app/milvus_engine.py

Required environment variables / Streamlit secrets:
- MILVUS_URI
- MILVUS_USER
- MILVUS_PASSWORD

OpenAI API key is intentionally passed from the UI per request.
"""

from __future__ import annotations

import os
import re
import uuid
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from openai import OpenAI
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "research_memory")
DIMENSION = int(os.getenv("MILVUS_EMBEDDING_DIMENSION", "1536"))
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

MAX_CHUNK_WORDS = 500
MAX_CHUNK_CHARS = 7500


def _get_secret(name: str) -> str:
    """Read secret from environment first, then Streamlit secrets if available."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st  # imported lazily so this module can be tested outside Streamlit

        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    raise RuntimeError(
        f"{name} bulunamadı. Streamlit Secrets veya environment variable olarak tanımlayın."
    )


def _escape_milvus_string(value: str) -> str:
    """Escape double quotes and backslashes for simple Milvus expr strings."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _safe_doc_id(doc_id: str) -> str:
    """Normalize doc IDs so they are query-safe and readable."""
    cleaned = re.sub(r"[^\w\-.()\[\] ]+", "_", doc_id, flags=re.UNICODE).strip()
    return cleaned[:180] or "document"


@lru_cache(maxsize=1)
def connect_milvus() -> bool:
    """Open a reusable connection to Milvus / Zilliz Cloud."""
    uri = _get_secret("MILVUS_URI")
    user = _get_secret("MILVUS_USER")
    password = _get_secret("MILVUS_PASSWORD")

    connections.connect(
        alias="default",
        uri=uri,
        user=user,
        password=password,
        secure=True,
    )
    return True


def create_collection() -> Collection:
    """Create collection and index if missing, then return the collection."""
    connect_milvus()

    if COLLECTION_NAME in utility.list_collections():
        collection = Collection(name=COLLECTION_NAME)
        if not collection.has_index():
            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                },
            )
        return collection

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=64,
        ),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]

    schema = CollectionSchema(fields, description="ScholarMind chunked document memory")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        },
    )
    return collection


def get_embedding(text: str, api_key: str) -> np.ndarray:
    """Create an OpenAI embedding vector."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        raise ValueError("Embedding üretmek için boş metin verilemez.")

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[cleaned], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)


def chunk_text(text: str, chunk_size: int = MAX_CHUNK_WORDS) -> List[str]:
    """Split text into chunks that fit Milvus VARCHAR limits."""
    words = (text or "").split()
    chunks: List[str] = []

    for start in range(0, len(words), chunk_size):
        candidate = " ".join(words[start : start + chunk_size]).strip()
        if not candidate:
            continue

        while len(candidate) > MAX_CHUNK_CHARS:
            chunks.append(candidate[:MAX_CHUNK_CHARS])
            candidate = candidate[MAX_CHUNK_CHARS:].strip()

        if candidate:
            chunks.append(candidate)

    return chunks


def add_to_milvus(user_id: str, doc_id: str, text: str, api_key: str) -> bool:
    """Chunk text, embed chunks, and persist them in Milvus."""
    if not user_id or not user_id.strip():
        raise ValueError("user_id boş olamaz.")
    if not doc_id or not doc_id.strip():
        raise ValueError("doc_id boş olamaz.")

    collection = create_collection()
    chunks = chunk_text(text)
    if not chunks:
        return False

    safe_doc_id = _safe_doc_id(doc_id)
    rows = []

    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk, api_key)
        rows.append(
            [
                str(uuid.uuid4()),
                user_id.strip(),
                f"{safe_doc_id}_chunk_{i}",
                chunk,
                vector.tolist(),
            ]
        )

    columns = list(map(list, zip(*rows)))
    collection.insert(columns)
    collection.flush()
    return True


def search_milvus(
    query: str,
    user_id: str,
    api_key: str,
    top_k: int = 5,
) -> List[Tuple[str, str, float]]:
    """Search user-specific memory in Milvus."""
    if not query or not query.strip():
        return []
    if not user_id or not user_id.strip():
        raise ValueError("user_id boş olamaz.")

    collection = create_collection()
    collection.load()

    query_vector = get_embedding(query, api_key)
    safe_user_id = _escape_milvus_string(user_id.strip())

    results = collection.search(
        data=[query_vector.tolist()],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        expr=f'user_id == "{safe_user_id}"',
        output_fields=["doc_id", "chunk"],
    )

    return [
        (
            hit.entity.get("doc_id"),
            hit.entity.get("chunk"),
            float(hit.distance),
        )
        for hit in results[0]
    ]


def list_titles(user_id: str, session_user_id: str) -> List[str]:
    """List unique document names for the active user only."""
    if user_id != session_user_id:
        raise PermissionError("Başka bir kullanıcının kayıtlı içeriklerini görüntüleyemezsiniz.")

    collection = create_collection()
    collection.load()

    safe_user_id = _escape_milvus_string(user_id.strip())
    results = collection.query(
        expr=f'user_id == "{safe_user_id}"',
        output_fields=["doc_id"],
    )

    doc_ids = [row.get("doc_id", "") for row in results]
    titles = sorted({doc_id.split("_chunk_")[0] for doc_id in doc_ids if doc_id})
    return titles


def delete_doc_id(user_id: str, doc_id: str) -> bool:
    """Delete all chunks that belong to one user document."""
    collection = create_collection()
    collection.load()

    safe_user_id = _escape_milvus_string(user_id.strip())
    safe_doc_id = _safe_doc_id(doc_id)

    rows = collection.query(
        expr=f'user_id == "{safe_user_id}"',
        output_fields=["id", "doc_id"],
    )

    ids_to_delete = [
        row["id"]
        for row in rows
        if row.get("doc_id", "").startswith(f"{safe_doc_id}_chunk_")
    ]

    if not ids_to_delete:
        return True

    quoted_ids = ",".join([f'"{_escape_milvus_string(item_id)}"' for item_id in ids_to_delete])
    collection.delete(expr=f"id in [{quoted_ids}]")
    collection.flush()
    return True


def clear_user_memory(user_id: str) -> bool:
    """Delete every memory chunk for the given user."""
    collection = create_collection()
    collection.load()

    safe_user_id = _escape_milvus_string(user_id.strip())
    collection.delete(expr=f'user_id == "{safe_user_id}"')
    collection.flush()
    return True
