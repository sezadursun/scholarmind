# rag_qa_engine.py

from app.model_selector import get_model
from typing import List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 🧠 Embedder (MiniLM, hızlı ve hafif)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # MiniLM vektör boyutu
faiss_index = faiss.IndexFlatL2(dimension)

# Hafızada da saklıyoruz
stored_chunks: List[str] = []

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """PDF'ten alınan tam metni belirli uzunlukta parçalara ayırır."""
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embed_text(texts: List[str]) -> np.ndarray:
    """Parçaları MiniLM ile embed eder (batch olarak)."""
    vectors = embedding_model.encode(texts)
    return np.array(vectors).astype("float32")

def build_index_from_text(full_text: str) -> None:
    """Chunk'ları embed edip FAISS index'e ekler."""
    global stored_chunks
    chunks = chunk_text(full_text)
    vectors = embed_text(chunks)
    faiss_index.add(vectors)
    stored_chunks = chunks  # global hafızada tut

def get_similar_chunks(question: str, top_k: int = 3) -> List[str]:
    """Soruya en benzeyen chunk'ları getir."""
    query_vec = embed_text([question])
    _, indices = faiss_index.search(query_vec, top_k)
    return [stored_chunks[i] for i in indices[0]]

def answer_with_context(question: str, api_key: str, top_k: int = 3) -> str:
    """En benzer chunk'lara dayanarak, uygun LLM ile soruyu yanıtla."""
    top_chunks = get_similar_chunks(question, top_k=top_k)
    context = "\n\n".join(top_chunks)

    prompt = f"""
Aşağıdaki parçalar, bir akademik makaleye aittir:

{context}

Soru: {question}

Lütfen sadece yukarıdaki içeriklere dayanarak soruyu yanıtla. Tahminde bulunma. Cevabın açık, akademik ve net olsun.
"""

    try:
        client = OpenAI(api_key=api_key)
        model = get_model("qa")  # 🔁 GPT-4o yerine dinamik seçim
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Sen bir akademik asistan GPT'sin. Sadece içerikteki bilgiye dayanarak cevap ver."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Yanıtlama sırasında hata oluştu: {str(e)}"

