# Uses FAISS for similarity search among paper embeddings
import faiss
import numpy as np
import textwrap
from openai import OpenAI

# Embedding ayarları
DIMENSION = 1536  # text-embedding-ada-002 boyutu
MAX_CHUNK_SIZE = 1000  # karakter bazlı güvenli limit
EMBEDDING_MODEL = "text-embedding-ada-002"

# FAISS index + metin listesi
faiss_index = faiss.IndexFlatL2(DIMENSION)
stored_chunks = []  # [(text, source_info)]

def get_embedding(text: str, api_key: str) -> np.ndarray:
    """Metni OpenAI ile vektöre dönüştürür."""
    client = OpenAI(api_key=api_key)
    cleaned = text.replace("\n", " ").strip()
    response = client.embeddings.create(
        input=[cleaned],
        model=EMBEDDING_MODEL
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def chunk_text(text: str, max_length: int = MAX_CHUNK_SIZE) -> list:
    """Uzun metinleri parçalayarak döndürür (token değil karakter)."""
    return textwrap.wrap(text, max_length)

def add_text_to_index(text: str, source_id: str = "unknown", api_key: str = ""):
    """Metni parçalara ayır, vektörleştir, FAISS’e ekle."""
    chunks = chunk_text(text)
    for chunk in chunks:
        vec = get_embedding(chunk, api_key=api_key)
        faiss_index.add(np.array([vec]))
        stored_chunks.append((chunk, source_id))

def search_similar(text: str, top_k=3, api_key: str = ""):
    """Verilen metne benzer içerikleri önerir."""
    vec = get_embedding(text, api_key=api_key)
    D, I = faiss_index.search(np.array([vec]), top_k)
    return [stored_chunks[i] for i in I[0]]

def suggest_topics_based_on_text(text, api_key: str, model="gpt-4o"):
    client = OpenAI(api_key=api_key)
    prompt = f"Bu metne göre 5 akademik araştırma konusu öner: {text}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Sen akademik bir konu uzmanısın."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip().split("\n")


    prompt = f"""
Aşağıdaki akademik içeriklere benzer 3 araştırma konusu öner:
Metin:
{combined}

3 öneri:
"""

    client = OpenAI(api_key=api_key)
    response = clienot.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Sen araştırma konularında öneriler sunan bir asistansın."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
