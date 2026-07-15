"""ScholarMind Streamlit UI.

Place this file at: ui/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# Make imports stable both locally and on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

from scholarmind_ui_theme import apply_scholarmind_theme
from app.arxiv import search_arxiv
from app.chroma import add_to_memory as add_to_chroma_memory, search_memory
from app.faiss_engine import (
    add_text_to_index,
    reset_index,
    search_similar,
    suggest_topics_based_on_text,
)
from app.milvus_engine import add_to_milvus, list_titles
from app.paper_search import PaperSearchError, search_papers
from app.prompts import SYSTEM_MESSAGE
from app.rag_milvus import streamlit_memory_qa_tab
from app.rag_qa_engine import answer_with_context, build_index_from_text
from app.summarize import summarize_paper

apply_scholarmind_theme()


def get_openai_api_key() -> str:
    """Read OpenAI API key from sidebar, Streamlit secrets, or environment."""
    st.sidebar.markdown("## 🔐 OpenAI API Key")
    typed_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    if typed_key:
        return typed_key

    secret_key = st.secrets.get("OPENAI_API_KEY", None)
    if secret_key:
        return str(secret_key)

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    return ""


def extract_pdf_text(uploaded_file) -> str:
    """Extract readable text from a Streamlit-uploaded PDF."""
    pdf_reader = PdfReader(uploaded_file)
    pages = []
    for page in pdf_reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def render_model_test(client: OpenAI) -> None:
    with st.sidebar.expander("🤖 GPT-4o Erişim Testi"):
        if st.button("GPT-4o Erişimini Test Et"):
            try:
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Sadece çalıştığını kanıtla"}],
                    max_tokens=5,
                )
                st.success("✅ GPT-4o modeline erişiminiz var!")
            except Exception as e:
                message = str(e)
                if "Incorrect API key" in message or "401" in message:
                    st.error("❌ API anahtarınız geçersiz olabilir.")
                elif "model" in message and "not found" in message:
                    st.error("🚫 GPT-4o modeline erişiminiz yok.")
                else:
                    st.error(f"⚠️ Bilinmeyen hata: {message}")


api_key = get_openai_api_key()
if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

client = OpenAI(api_key=api_key)
render_model_test(client)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: nowrap;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F0F0;
        color: #333;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        border: 1px solid #d0d0d0;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4B3F72 !important;
        color: white !important;
        border: 2px solid #4B3F72;
        box-shadow: inset 0 -4px 0 #F44336;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E0E0E0;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(":brain: ScholarMind")
st.caption("Bilge araştırma hafızanız. Arayın, özetleyin, hatırlayın.")

TAB_LABELS = [
    "🔍 Ara",
    "⏪ Geçmiş",
    "🥚 ArXiv",
    "📖 Soru Sor",
    "🧠 Hafızadan Sor",
    "🧾 PDF ➕ Hafıza",
    "📂 Başlıkları Gör",
]

tab_search, tab_history, tab_arxiv, tab_pdf_qa, tab_memory_qa, tab_pdf_memory, tab_titles = st.tabs(TAB_LABELS)


with tab_search:
    st.subheader("🔍 Akademik Makale Ara")
    query = st.text_input("🔍 Konu:", "transformer visual recognition")
    year = st.slider("📅 Minimum Yayın Yılı", 2000, 2026, 2020)
    limit = st.selectbox("📄 Kaç makale getirilsin?", [3, 5, 7, 10], index=1)

    if st.button("Ara ve Özetle"):
        with st.spinner("Makaleler aranıyor..."):
            try:
                semantic_scholar_api_key = None
                try:
                    semantic_scholar_api_key = st.secrets.get(
                        "SEMANTIC_SCHOLAR_API_KEY"
                    )
                except Exception:
                    pass

                papers = search_papers(
                    query=query,
                    year=year,
                    limit=limit,
                    api_key=semantic_scholar_api_key,
                )
            except PaperSearchError as e:
                st.error(f"📱 Semantic Scholar API hatası: {e}")
                if e.status_code == 429:
                    st.info(
                        "Bu hata gerçek bir 'sonuç bulunamadı' durumu değildir. "
                        "Semantic Scholar ortak istek limiti dolmuş olabilir."
                    )
                st.stop()
            except Exception as e:
                st.error(f"📱 Makale arama hatası: {str(e)}")
                st.stop()

        if not papers:
            st.warning("❗ Aradığınız konuda sonuç bulunamadı.")
        else:
            st.success(f"{len(papers)} makale bulundu.")

        # First summarize every result. Then build one temporary FAISS index
        # containing the whole result set. This prevents each paper from being
        # recommended as its own "similar paper".
        enriched_papers = []

        for idx, paper in enumerate(papers, 1):
            title = paper.get("title", "Başlık yok")
            abstract = paper.get("abstract", "")

            with st.spinner(f"{idx}/{len(papers)} kısa özet hazırlanıyor..."):
                try:
                    short_summary = summarize_paper(
                        {"title": title, "abstract": abstract},
                        api_key,
                    )
                except Exception as e:
                    short_summary = ""
                    st.warning(f"⚠️ '{title}' özetlenemedi: {str(e)}")

            enriched = dict(paper)
            enriched["short_summary"] = short_summary
            enriched["combined_text"] = (
                f"{title} - {short_summary or abstract}"
            ).strip()
            enriched_papers.append(enriched)

        # FAISS is temporary and search-specific.
        reset_index()
        faiss_ready = True

        for paper in enriched_papers:
            try:
                add_text_to_index(
                    paper["combined_text"],
                    source_id=paper.get("paper_id") or paper["title"],
                    api_key=api_key,
                )
            except Exception as e:
                faiss_ready = False
                st.warning(
                    "⚠️ FAISS geçici indeksine ekleme başarısız oldu: "
                    f"{paper['title']} — {str(e)}"
                )

        # Chroma is persistent history. Its failure must not block FAISS or UI.
        for idx, paper in enumerate(enriched_papers, 1):
            try:
                stable_id = (
                    paper.get("paper_id")
                    or f"semantic-scholar-{idx}-{abs(hash(paper['title']))}"
                )
                add_to_chroma_memory(
                    id=str(stable_id),
                    content=paper["combined_text"],
                    api_key=api_key,
                    metadata={
                        "source": "SemanticScholar",
                        "title": paper["title"],
                        "paper_id": paper.get("paper_id") or "",
                    },
                )
            except Exception as e:
                st.warning(
                    "⚠️ Chroma geçmişine kayıt başarısız oldu: "
                    f"{paper['title']} — {str(e)}"
                )

        for idx, paper in enumerate(enriched_papers, 1):
            title = paper.get("title", "Başlık yok")
            abstract = paper.get("abstract", "")
            short_summary = paper.get("short_summary", "")
            combined_text = paper.get("combined_text", title)
            paper_year = paper.get("year", "Yıl bilgisi yok")
            citation_count = paper.get("citationCount", 0)
            url = paper.get("url", "#")
            source_id = paper.get("paper_id") or title

            st.markdown(f"## {idx}. {title}")

            authors_data = paper.get("authors", [])
            if isinstance(authors_data, list):
                authors = ", ".join(
                    a.get("name", "Bilinmeyen Yazar")
                    for a in authors_data
                    if isinstance(a, dict)
                )
            elif isinstance(authors_data, str):
                authors = authors_data
            else:
                authors = "Yazar bilgisi yok"

            st.markdown(
                f"**Yazarlar:** {authors or 'Yazar bilgisi yok'}  \n"
                f"**Yıl:** {paper_year}  \n"
                f"**Alıntı:** {citation_count}"
            )

            if url and url != "#":
                st.markdown(f"🔗 [Orijinal Makale]({url})")

            if short_summary:
                st.success(f"**Kısa Özet:** {short_summary}")
            else:
                st.info("Bu makale için kısa özet üretilemedi.")

            with st.expander("📜 Detaylı Özet"):
                detailed_prompt = f"""
Makale başlığı: {title}

Özeti: {abstract}

Bu makaleyi aşağıdaki başlıklar altında detaylıca analiz et:

1. Problem tanımı
2. Kullanılan yöntem ve veri
3. Sonuçlar ve katkılar
4. Bu çalışmanın önem düzeyi

Hepsini sade ve akademik bir dille açıkla (6-10 cümle arası).
"""
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "user", "content": detailed_prompt},
                        ],
                    )
                    st.info(response.choices[0].message.content.strip())
                except Exception as e:
                    st.error(f"GPT-4o hata: {str(e)}")

            st.markdown("### 🔍 Benzer Makaleler")
            if not faiss_ready:
                st.info("Geçici benzerlik indeksi tam olarak oluşturulamadı.")
            else:
                try:
                    similar = search_similar(
                        combined_text,
                        top_k=min(3, max(0, len(enriched_papers) - 1)),
                        api_key=api_key,
                        exclude_source_id=source_id,
                    )

                    if not similar:
                        st.info(
                            "Bu sonuç kümesinde mevcut makale dışında "
                            "benzer bir makale bulunamadı."
                        )

                    for sim_idx, (chunk, similar_source_id) in enumerate(similar, 1):
                        similar_paper = next(
                            (
                                item
                                for item in enriched_papers
                                if (item.get("paper_id") or item["title"])
                                == similar_source_id
                            ),
                            None,
                        )
                        similar_title = (
                            similar_paper["title"]
                            if similar_paper
                            else similar_source_id
                        )
                        st.markdown(f"**{sim_idx}. {similar_title}**")
                        st.write(f"_{chunk[:300]}..._")
                except Exception as e:
                    st.warning(f"⚠️ Benzer makale arama hatası: {str(e)}")

            st.markdown("### 💡 Yeni Araştırma Konu Önerileri")
            try:
                topics = suggest_topics_based_on_text(
                    combined_text,
                    api_key=api_key,
                )
                st.success("\n".join(topics) if isinstance(topics, list) else topics)
            except Exception as e:
                st.warning(f"⚠️ Konu önerisi hatası: {str(e)}")

            st.markdown("---")


with tab_history:
    st.subheader("🔁 Daha Önce Eklediğiniz Araştırmalar")
    search_term = st.text_input("📅 Geçmişte aradığınız bir konuyu yazın:")

    if search_term:
        with st.spinner("Geçmiş taranıyor..."):
            try:
                results = search_memory(search_term, api_key=api_key)
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]

                if not documents:
                    st.info("Bu arama için geçmişte kayıt bulunamadı.")

                for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
                    st.markdown(f"**{i}. {meta.get('source', 'Kaynak Yok')}**")
                    st.info((doc or "")[:500] + "...")
            except Exception as e:
                st.error(f"Geçmiş arama hatası: {str(e)}")


with tab_arxiv:
    st.subheader("🧪 ArXiv Preprint Arama")
    arxiv_query = st.text_input("🔍 ArXiv'te aramak istediğiniz konu:", "self-supervised learning")
    max_results = st.slider("Kaç makale getirilsin?", 1, 10, 5)

    if st.button("ArXiv'te Ara"):
        with st.spinner("arXiv API'den sonuçlar getiriliyor..."):
            try:
                arxiv_papers = search_arxiv(arxiv_query, max_results=max_results)
            except Exception as e:
                st.error(f"ArXiv API hatası: {str(e)}")
                arxiv_papers = []

        if not arxiv_papers:
            st.warning("Hiçbir preprint bulunamadı.")

        for i, paper in enumerate(arxiv_papers, 1):
            st.markdown(f"### {i}. {paper.get('title', 'Başlık yok')}")
            st.markdown(f"**Yazarlar:** {paper.get('authors', 'Yazar bilgisi yok')}")
            st.markdown(f"**Yayın Tarihi:** {paper.get('published', 'Tarih yok')}")
            st.write(f"**Özet:** {paper.get('summary', '')[:500]}...")

            link = paper.get("link", "#")
            if link and link != "#":
                st.markdown(f"[🔗 ArXiv Linki]({link})")

            st.markdown("---")


with tab_pdf_qa:
    st.subheader("📖 Yüklediğiniz makaleye soru sorun")
    uploaded_file = st.file_uploader("📌 PDF yükleyin", type=["pdf"], key="pdf_qa_file")
    question = st.text_input("❓ Bu makaleyle ilgili ne öğrenmek istiyorsunuz?", "")

    if uploaded_file and question and st.button("🧠 Soruyu Yanıtla"):
        with st.spinner("PDF okunuyor ve analiz ediliyor..."):
            try:
                full_text = extract_pdf_text(uploaded_file)

                if len(full_text) < 100:
                    st.warning("Bu PDF'den yeterince metin çıkarılamadı.")
                else:
                    build_index_from_text(full_text)
                    with st.spinner("Yanıt oluşturuluyor..."):
                        answer = answer_with_context(question, api_key)
                        st.success("✅ Yanıt:")
                        st.write(answer)
            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")


with tab_memory_qa:
    streamlit_memory_qa_tab(api_key)


with tab_pdf_memory:
    st.subheader("📌 PDF'yi Milvus Hafızasına Ekle")
    user_id = st.text_input("👤 Kullanıcı ID:", value="demo-user", key="milvus_user_id")
    uploaded_file = st.file_uploader("📎 PDF yükleyin", type=["pdf"], key="milvus_pdf")

    if uploaded_file and user_id and st.button("💾 Hafızaya Kaydet"):
        with st.spinner("PDF okunuyor ve Milvus'a kaydediliyor..."):
            try:
                full_text = extract_pdf_text(uploaded_file)

                if len(full_text.strip()) < 100:
                    st.warning("Bu PDF'den yeterince metin çıkarılamadı.")
                else:
                    base_doc_id = uploaded_file.name.rsplit(".", 1)[0]
                    saved = add_to_milvus(
                        user_id=user_id,
                        doc_id=base_doc_id,
                        text=full_text,
                        api_key=api_key,
                    )
                    if saved:
                        st.success("✅ PDF içeriği Milvus hafızasına başarıyla eklendi!")
                    else:
                        st.warning("PDF okundu ama kaydedilecek anlamlı metin bulunamadı.")
            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")


with tab_titles:
    st.subheader("📚 Kayıtlı Başlıklarınızı Görüntüleyin")
    current_user_id = st.text_input(
        "👤 Kullanıcı ID (başlıkları görmek için):",
        value="demo-user",
        key="titles_user_id",
    )

    if st.button("📂 Başlıkları Göster"):
        try:
            titles = list_titles(user_id=current_user_id, session_user_id=current_user_id)

            if titles:
                st.success(f"✅ {len(titles)} başlık bulundu:")
                for title in titles:
                    st.markdown(f"- 📄 **{title}**")
            else:
                st.info("🔍 Henüz eklenmiş bir başlık bulunamadı.")
        except PermissionError as e:
            st.error(f"🚫 Yetkisiz erişim: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Bir hata oluştu: {str(e)}")
