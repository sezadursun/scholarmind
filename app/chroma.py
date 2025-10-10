# 🧳️ chroma.py – ChromaDB ile araştırma geçmişini kaydetme ve geri çağırma

import os
from dotenv import load_dotenv

load_dotenv()

chroma_supported = True

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except Exception as e:
    print(f"❌ ChromaDB modülü yüklenemedi: {e}")
    chroma_supported = False
    chromadb = None
    embedding_functions = None

# ✅ Chroma istemcisi başlatılıyor (yalnızca destekleniyorsa)
if chroma_supported:
    try:
        chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="chroma_storage"  # 🗃️ kalıcı saklama klasörü
        ))

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("❌ OPENAI_API_KEY environment variable not found.")

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"
        )

        collection = chroma_client.get_or_create_collection(
            name="research_history",
            embedding_function=openai_ef
        )
    except Exception as e:
        print(f"❌ Chroma istemcisi oluşturulamadı: {e}")
        chroma_supported = False
        chroma_client = None
        collection = None
else:
    chroma_client = None
    collection = None

# ✅ Hafızaya içerik ekleme
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
        print(f"✅ '{id}' hafızaya eklendi.")
    except Exception as e:
        print(f"❌ Chroma ekleme hatası: {e}")

# ✅ Hafızadan benzer içerik arama
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
