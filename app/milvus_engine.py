from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from openai import OpenAI
import numpy as np
import uuid
from typing import List

# 🔌 Milvus bağlantısı (Zilliz Cloud)
connections.connect(
    alias="default",
    uri="https://in03-3ebddd479cd8e47.serverless.gcp-us-west1.cloud.zilliz.com",
    user="db_3ebddd479cd8e47",
    password="Os2*%t,c<3QOcpq6",
    secure=True
)

# 📊 Koleksiyon bilgileri
COLLECTION_NAME = "research_memory"
DIMENSION = 1536  # text-embedding-ada-002
EMBEDDING_MODEL = "text-embedding-ada-002"

# 📄 Koleksiyon oluşturma
def create_collection():
    if COLLECTION_NAME in utility.list_collections():
        collection = Collection(name=COLLECTION_NAME)
        if not collection.has_index():
            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            )
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields, description="Chunked document memory")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create()

    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
    )

# 🧠 Embedding oluştur
def get_embedding(text: str, api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)

# 🧩 Chunk metni
def chunk_text(text: str, chunk_size: int = 500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# 💾 Milvus'a veri ekle
def add_to_milvus(user_id: str, doc_id: str, text: str, api_key: str):
    create_collection()
    collection = Collection(name=COLLECTION_NAME)

    chunks = chunk_text(text)
    records = []
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk, api_key)
        record_id = str(uuid.uuid4())
        records.append([
            record_id,
            user_id,
            f"{doc_id}_chunk_{i}",
            chunk,
            vector.tolist()
        ])

    records = list(map(list, zip(*records)))  # Transpose
    collection.insert(records)
    collection.flush()

# 🔍 Arama yap
def search_milvus(query: str, user_id: str, api_key: str, top_k: int = 5):
    create_collection()
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    query_vector = get_embedding(query, api_key)
    results = collection.search(
        data=[query_vector.tolist()],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        expr=f"user_id == '{user_id}'",
        output_fields=["doc_id", "chunk"]
    )

    hits = results[0]
    return [(hit.entity.get("doc_id"), hit.entity.get("chunk"), hit.distance) for hit in hits]

# 📋 Kullanıcının başlıklarını listeler
def list_titles(user_id: str, session_user_id: str) -> List[str]:
    """Sadece kullanıcı kendi verilerini görebilir. Chunk başlıklarını gruplayarak döndürür."""
    if user_id != session_user_id:
        raise PermissionError("Başka bir kullanıcının kayıtlı içeriklerini görüntüleyemezsiniz.")
    
    create_collection()
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    
    results = collection.query(
        expr=f"user_id == '{user_id}'",
        output_fields=["doc_id"]
    )
    
    all_doc_ids = [res["doc_id"] for res in results]
    grouped = list(set([doc_id.split("_chunk_")[0] for doc_id in all_doc_ids]))
    return grouped
