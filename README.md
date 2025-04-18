# Academic Agent

This project allows academic research via Semantic Scholar and GPT summarization.

---

# 📚 Academic Research Assistant

AI-powered assistant for researchers to:
- 🔍 Search academic papers by topic
- 📝 Summarize abstracts and full papers using GPT-4
- 📄 Upload PDFs and ask direct questions about content
- 🔁 Store and recall past research memory using **Milvus** or optional **Chroma**
- 🧠 Ask questions with **RAG (retrieval-based)** or 🌱 **AAG (analogy-based)** generation modes

---

## 🚀 Features

| Feature               | Description                                                 |
|----------------------|-------------------------------------------------------------|
| 🔍 Semantic Search    | Search academic papers with Semantic Scholar API            |
| 🧠 RAG Mode           | Retrieval-Augmented Generation for precise answers          |
| 🌱 AAG Mode           | Analogy-based explanation and creative responses            |
| 📄 PDF Q&A            | Ask questions about any uploaded academic PDF               |
| 🔁 Milvus Memory      | Persistent, user-specific research recall engine            |
| 📎 Streamlit UI       | Responsive web app with tab-based interface                 |

---

## 🖥️ Tech Stack

- [x] **Python 3.10** – Tüm uygulamanın temel çalışma ortamı
- [x] **Streamlit** – UI/UX arayüzü, sekmeli yapı ve kullanıcı etkileşimi
- [x] **OpenAI GPT-4 API**
  - 🔹 `chat/completions` ile özetleme ve Q&A
  - 🔹 `embeddings` ile vektörleştirme
- [x] **Semantic Scholar API** – Literatür taraması ve makale başlıkları
- [x] **arXiv API** – Preprint makale arama ve özetleme
- [x] **ChromaDB** – Hafıza yönetimi için geçici vektör veritabanı
- [x] **FAISS (Facebook AI Similarity Search)** – Lokal benzerlik önerileri
- [x] **Milvus (Zilliz Cloud)** – Kalıcı kullanıcı hafızası (RAG + AAG için)
- [x] **PyMuPDF / PyPDF2** – PDF parsing & içerik çıkarımı
- [x] **Inter Font + ScholarMind Theme** – Akademik minimalist UI tasarımı

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/academic-research-assistant.git
cd academic-research-assistant
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
