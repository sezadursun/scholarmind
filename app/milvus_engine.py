from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from openai import OpenAI
import numpy as np
import uuid

# 🔌 Zilliz Cloud bağlantısı
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
        return
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields, description="Research memory by user")
    Collection(name=COLLECTION_NAME, schema=schema).create()

# 🧠 Embedding oluştur
def get_embedding(text: str, api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)

# 💾 Milvus'a veri ekle
def add_to_milvus(user_id: str, title: str, text: str, api_key: str):
    create_collection()
    collection = Collection(name=COLLECTION_NAME)
    vector = get_embedding(text, api_key)
    record_id = str(uuid.uuid4())
    data = [[record_id], [user_id], [title], [vector.tolist()]]
    collection.insert(data)
    collection.flush()

# 🔍 Arama yap
def search_milvus(query: str, user_id: str, api_key: str, top_k: int = 3):
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
        output_fields=["title"]
    )

    hits = results[0]
    return [(hit.entity.get("title"), hit.distance) for hit in hits]
