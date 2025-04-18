# Manages history using ChromaDB
# 🧳️ chroma.py – ChromaDB ile araştırma geçmişini kaydetme ve geri çağırma

import os

chroma_supported = True

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except Exception as e:
    chroma_supported = False
    chromadb = None
    embedding_functions = None

# Chroma istemcisi başlatılıyor (yalnızca destekleniyorsa)
if chroma_supported:
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="chroma_storage"  # kalıcı saklama klasörü
    ))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-ada-002"
    )

    collection = chroma_client.get_or_create_collection(
        name="research_history",
        embedding_function=openai_ef
    )
else:
    chroma_client = None
    collection = None

def add_to_memory(id: str, content: str, metadata: dict = None):
    """
    Yeni bir içeriği hafızaya ekler.
    """
    if not chroma_supported or collection is None:
        print("⛘️ Chroma desteklenmiyor. Hafızaya ekleme yapılamaz.")
        return

    try:
        collection.add(
            documents=[content],
            ids=[id],
            metadatas=[metadata or {}]
        )
    except Exception as e:
        print(f"❌ Chroma ekleme hatası: {e}")

def search_memory(query: str, top_k=3):
    """
    Hafızadaki içerikler arasında semantik olarak en benzer olanları bulur.
    """
    if not chroma_supported or collection is None:
        print("⛘️ Chroma desteklenmiyor. Hafızadan arama yapılamaz.")
        return {"documents": [[]], "metadatas": [[]]}

    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        return results
    except Exception as e:
        print(f"❌ Chroma sorgusu başarısız oldu: {e}")
        return {"documents": [[]], "metadatas": [[]]}
