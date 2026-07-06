from pathlib import Path

code = '''import os
import uuid
from typing import List

import numpy as np
from openai import OpenAI
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


# 🔌 Milvus / Zilliz Cloud bağlantı bilgileri
# Güvenlik için bunları Streamlit Secrets veya environment variable olarak tut.
MILVUS_URI = os.getenv(
    "MILVUS_URI",
    "https://in03-3ebddd479cd8e47.serverless.gcp-us-west1.cloud.zilliz.com",
)
MILVUS_USER = os.getenv("MILVUS_USER", "db_3ebddd479cd8e47")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Os2*%t,c<3QOcpq6")

COLLECTION_NAME = "research_memory"
DIMENSION = 1536
EMBEDDING_MODEL = "text-embedding-ada-002"


def connect_milvus():
    """Milvus bağlantısını güvenli şekilde başlatır."""
    try:
        connections.connect(
            alias="default",
            uri=MILVUS_URI,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD,
            secure=True,
        )
    except Exception:
        # Bağlantı zaten açıksa veya yeniden bağlanma sorunu varsa sessiz geç.
        pass


def create_collection():
    """Collection yoksa oluşturur, index yoksa ekler."""
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
        return

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=64,
        ),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]

    schema = CollectionSchema(fields, description="Chunked document memory")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        },
    )


def get_embedding(text: str, api_key: str) -> np.ndarray:
    """OpenAI embedding üretir."""
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Metni kelime bazlı parçalara böler."""
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
        if " ".join(words[i : i + chunk_size]).strip()
    ]


def add_to_milvus(user_id: str, doc_id: str, text: str, api_key: str) -> bool:
    """Metni chunk'lara böler ve Milvus'a kaydeder."""
    create_collection()
    collection = Collection(name=COLLECTION_NAME)

    chunks = chunk_text(text)

    if not chunks:
        return False

    rows = []
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk, api_key)
        rows.append(
            [
                str(uuid.uuid4()),
                user_id,
                f"{doc_id}_chunk_{i}",
                chunk,
                vector.tolist(),
            ]
        )

    columns = list(map(list, zip(*rows)))
    collection.insert(columns)
    collection.flush()
    return True


def search_milvus(query: str, user_id: str, api_key: str, top_k: int = 5):
    """Kullanıcının hafızasında vektör araması yapar."""
    create_collection()
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    query_vector = get_embedding(query, api_key)

    results = collection.search(
        data=[query_vector.tolist()],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        expr=f'user_id == "{user_id}"',
        output_fields=["doc_id", "chunk"],
    )

    hits = results[0]
    return [
        (
            hit.entity.get("doc_id"),
            hit.entity.get("chunk"),
            hit.distance,
        )
        for hit in hits
    ]


def list_titles(user_id: str, session_user_id: str) -> List[str]:
    """Sadece kullanıcı kendi kayıtlı başlıklarını görebilir."""
    if user_id != session_user_id:
        raise PermissionError(
            "Başka bir kullanıcının kayıtlı içeriklerini görüntüleyemezsiniz."
        )

    create_collection()
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    results = collection.query(
        expr=f'user_id == "{user_id}"',
        output_fields=["doc_id"],
    )

    all_doc_ids = [res["doc_id"] for res in results]
    grouped = sorted(set(doc_id.split("_chunk_")[0] for doc_id in all_doc_ids))
    return grouped


def delete_doc_id(user_id: str, doc_id: str) -> bool:
    """Belirli kullanıcıya ait tek dokümanın tüm chunk'larını siler."""
    create_collection()
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    try:
        results = collection.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["id", "doc_id"],
        )

        ids_to_delete = [
            row["id"]
            for row in results
            if row.get("doc_id", "").startswith(f"{doc_id}_chunk_")
        ]

        if not ids_to_delete:
            return True

        quoted_ids = ",".join([f'"{item_id}"' for item_id in ids_to_delete])
        collection.delete(expr=f"id in [{quoted_ids}]")
        collection.flush()
        return True

    except Exception as e:
        print(f"Silme işlemi sırasında hata: {str(e)}")
        return False


def clear_user_memory(user_id: str) -> bool:
    """Belirli kullanıcıya ait tüm Milvus hafızasını siler."""
    create_collection()
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    try:
        collection.delete(expr=f'user_id == "{user_id}"')
        collection.flush()
        return True
    except Exception as e:
        print(f"Hafıza temizleme sırasında hata: {str(e)}")
        return False
'''

out_path = Path("/mnt/data/milvus_engine.py")
out_path.write_text(code, encoding="utf-8")
out_path.as_posix()
