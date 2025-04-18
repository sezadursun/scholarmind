# Manages history using ChromaDB
# 🗃️ chroma.py – ChromaDB ile araştırma geçmişini kaydetme ve geri çağırma

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Chroma istemcisi başlatılıyor
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="chroma_storage"  # kalıcı saklama klasörü
))

# Embedding fonksiyonu: OpenAI kullanılıyor
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

# Koleksiyon oluşturuluyor (veya varsa çağrılıyor)
collection = chroma_client.get_or_create_collection(
    name="research_history",
    embedding_function=openai_ef
)

def add_to_memory(id: str, content: str, metadata: dict = None):
    """
    Yeni bir içerik hafızaya eklenir.
    ID benzersiz olmalıdır (örn: paper_123).
    Metadata örn: {"source": "PDF", "author": "Smith"}
    """
    collection.add(
        documents=[content],
        ids=[id],
        metadatas=[metadata or {}]
    )

def search_memory(query: str, top_k=3):
    """
    Hafızadaki içerikler arasında semantik olarak en benzer olanları bulur.
    """
    results = collection.query(query_texts=[query], n_results=top_k)
    return results
