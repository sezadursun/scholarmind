import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from scholarmind_ui_theme import apply_scholarmind_theme
apply_scholarmind_theme()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from app.paper_search import search_papers
from app.summarize import summarize_paper, summarize_fulltext
from app.prompts import SYSTEM_MESSAGE
from app.faiss_engine import add_text_to_index, search_similar, suggest_topics_based_on_text
from app.chroma import add_to_memory as add_to_chroma_memory, search_memory
from app.arxiv import search_arxiv
from app.rag_qa_engine import build_index_from_text, answer_with_context
from app.rag_milvus import streamlit_memory_qa_tab
from app.milvus_engine import add_to_milvus, list_titles
from PyPDF2 import PdfReader

# Sidebar: API Key
st.sidebar.markdown("## \U0001f512 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# Sidebar: GPT-4o Test
with st.sidebar.expander("\U0001f916 GPT-4o Erişimi Testi"):
    if st.button("GPT-4o Erişimini Test Et"):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Sadece çalıştığını kanıtla"}],
                max_tokens=5
            )
            st.success("\u2705 GPT-4o modeline erişiminiz var!")
        except Exception as e:
            if "Incorrect API key" in str(e) or "401" in str(e):
                st.error("\u274c API anahtarınız geçersiz olabilir.")
            elif "model" in str(e) and "not found" in str(e):
                st.error("\ud83d\udeab GPT-4o modeline erişiminiz yok.")
            else:
                st.error(f"\u26a0\ufe0f Bilinmeyen hata: {str(e)}")

# ScholarMind Ana Sayfa
st.title(":brain: ScholarMind")
st.caption("Bilge araştırma hafızanız. Arayın, özetleyin, hatırlayın.")

# Ana sekmeler
tab = st.radio("\U0001f4c4 Menü", [
    "🔍 Ara", "⏪ Geçmiş", "🥚 ArXiv", "📖 Soru Sor", "🧠 Hafızadan Sor", "🧾 PDF ➕ Hafıza", "📂 Başlıkları Gör"
])

# 1. Ara Sekmesi
if tab == "🔍 Ara":
    query = st.text_input("🔍 Konu:", "transformer visual recognition")
    year = st.slider("📅 Minimum Yayın Yılı", 2000, 2024, 2020)
    limit = st.selectbox("📄 Kaç makale getirilsin?", [3, 5, 7, 10], index=1)
    if st.button("Ara ve Özetle"):
        with st.spinner("Makaleler aranıyor..."):
            try:
                papers = search_papers(query=query, year=year, limit=limit)
            except Exception as e:
                st.error(f"\U0001f4f1 Semantic Scholar API hatası: {str(e)}")
                st.stop()

        if not papers:
            st.warning("❗ Aradığınız konuda sonuç bulunamadı.")
        else:
            st.success(f"{len(papers)} makale bulundu.")
            for idx, paper in enumerate(papers, 1):
                st.markdown(f"## {idx}. {paper['title']}")
                authors = ", ".join([a['name'] for a in paper['authors']])
                st.markdown(f"**Yazarlar:** {authors}  \n**Yıl:** {paper['year']}  \n**Alıntı:** {paper['citationCount']}")
                st.markdown(f"\U0001f517 [Orijinal Makale]({paper['url']})")

                with st.spinner("Kısa özet hazırlanıyor..."):
                    try:
                        short_summary = summarize_paper({
                            "title": paper["title"],
                            "abstract": paper["abstract"]
                        }, api_key)
                        st.success(f"**Kısa Özet:** {short_summary}")
                    except Exception as e:
                        st.error(f"\u26a0\ufe0f Özetleme hatası: {str(e)}")

                combined_text = f"{paper['title']} - {short_summary}"
                add_text_to_index(combined_text, source_id=paper['title'], api_key=api_key)
                add_to_chroma_memory(
                    id=f"{paper['title']}_{idx}",
                    content=combined_text,
                    metadata={"source": "SemanticScholar", "title": paper['title']}
                )

                st.markdown("### \U0001f4dc Benzer Makaleler")
                similar = search_similar(combined_text, top_k=3, api_key=api_key)
                for sim_idx, (chunk, src) in enumerate(similar, 1):
                    st.markdown(f"**{sim_idx}. ({src})**")
                    st.write(f"_{chunk[:300]}..._")

                st.markdown("### \U0001f4a1 Yeni Araştırma Konu Önerileri")
                topics = suggest_topics_based_on_text(combined_text, api_key=api_key)
                st.success(topics)

# 2. Geçmiş
elif tab == "⏪ Geçmiş":
    st.subheader("\u23ea Daha Önce Eklediğiniz Araştırmalar")
    search_term = st.text_input("\ud83d\udcc5 Geçmişte aradığınız bir konuyu yazın:")
    if search_term:
        with st.spinner("Geçmiş taranıyor..."):
            results = search_memory(search_term)
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                st.markdown(f"**{i+1}. {meta.get('source', 'Kaynak Yok')}**")
                st.info(doc[:500] + "...")

# 3. ArXiv
elif tab == "🥚 ArXiv":
    st.subheader("\ud83e\udd5a ArXiv Preprint Arama")
    arxiv_query = st.text_input("\U0001f50d ArXiv'te aramak istediğiniz konu:", "self-supervised learning")
    max_results = st.slider("Kaç makale getirilsin?", 1, 10, 5)

    if st.button("ArXiv'te Ara"):
        with st.spinner("arXiv API'den sonuçlar getiriliyor..."):
            arxiv_papers = search_arxiv(arxiv_query, max_results=max_results)

        if not arxiv_papers:
            st.warning("Hiçbir preprint bulunamadı.")
        else:
            for i, paper in enumerate(arxiv_papers, 1):
                st.markdown(f"### {i}. {paper['title']}")
                st.markdown(f"**Yazarlar:** {paper['authors']}")
                st.markdown(f"**Yayın Tarihi:** {paper['published']}")
                st.write(f"**Özet:** {paper['summary'][:500]}...")
                st.markdown(f"[\U0001f517 ArXiv Linki]({paper['link']})")
                st.markdown("---")

# 4. Soru Sor (RAG)
elif tab == "📖 Soru Sor":
    st.subheader("\ud83d\udcd6 Yüklediğiniz makaleye soru sorun")

    uploaded_file = st.file_uploader("\ud83d\udccc PDF yükleyin", type=["pdf"])
    question = st.text_input("❓ Bu makaleyle ilgili ne öğrenmek istiyorsunuz?", "")

    if uploaded_file and question and st.button("\U0001f9e0 Soruyu Yanıtla"):
        with st.spinner("PDF okunuyor ve analiz ediliyor..."):
            try:
                pdf_reader = PdfReader(uploaded_file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() or ""

                if len(full_text) < 100:
                    st.warning("Bu PDF'den yeterince metin çıkarılamadı.")
                else:
                    build_index_from_text(full_text)
                    with st.spinner("Yanıt oluşturuluyor..."):
                        answer = answer_with_context(question, api_key)
                        st.success("\u2705 Yanıt:")
                        st.write(answer)
            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")

# 5. Hafızadan Soru Sor
elif tab == "🧠 Hafızadan Sor":
    streamlit_memory_qa_tab(api_key)

# 6. PDF + Hafıza
elif tab == "🧾 PDF ➕ Hafıza":
    st.subheader("\ud83d\udccc PDF'yi Milvus Hafızasına Ekle")

    user_id = st.text_input("\ud83d\udc64 Kullanıcı ID:", value="demo-user")
    uploaded_file = st.file_uploader("\ud83d\udd8a\ufe0f PDF yükleyin", type=["pdf"], key="milvus_pdf")

    if uploaded_file and user_id and st.button("\ud83d\udbe1 Hafızaya Kaydet"):
        with st.spinner("PDF okunuyor ve embedding Milvus'a kaydediliyor..."):
            try:
                pdf_reader = PdfReader(uploaded_file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() or ""

                if len(full_text.strip()) < 100:
                    st.warning("Bu PDF'den yeterince metin çıkarılamadı.")
                else:
                    words = full_text.split()
                    chunk_size = 500
                    base_doc_id = uploaded_file.name.replace(".pdf", "")

                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i+chunk_size])
                        chunked_doc_id = f"{base_doc_id}_chunk_{i//chunk_size + 1}"
                        add_to_milvus(user_id=user_id, doc_id=chunked_doc_id, text=chunk, api_key=api_key)

                    st.success("\u2705 PDF içeriği parçalara ayrıldı ve Milvus'a başarıyla eklendi!")
            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")

# 7. Başlıkları Gör
elif tab == "📂 Başlıkları Gör":
    st.subheader("\ud83d\udcda Kayıtlı Başlıklarınızı Görüntüleyin")

    current_user_id = st.text_input("\ud83d\udc64 Kullanıcı ID (başlıkları görmek için):", value="demo-user")

    if st.button("\ud83d\udcc2 Başlıkları Göster"):
        try:
            titles = list_titles(user_id=current_user_id, session_user_id=current_user_id)
            if titles:
                st.success(f"\u2705 {len(titles)} başlık bulundu:")
                for title in titles:
                    st.markdown(f"- \ud83d\udcc4 **{title}**")
            else:
                st.info("\ud83d\udd0d Henüz eklenmiş bir başlık bulunamadı.")
        except PermissionError as e:
            st.error(f"\ud83d\udeab Yetkisiz erişim: {str(e)}")
        except Exception as e:
            st.error(f"\u26a0\ufe0f Bir hata oluştu: {str(e)}")
