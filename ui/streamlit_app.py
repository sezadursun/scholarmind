import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from scholarmind_ui_theme import apply_scholarmind_theme
apply_scholarmind_theme()

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from openai._exceptions import AuthenticationError, InvalidRequestError

from app.paper_search import search_papers
from app.summarize import summarize_paper, summarize_fulltext
from app.prompts import SYSTEM_MESSAGE
from app.faiss_engine import add_text_to_index, search_similar, suggest_topics_based_on_text
from app.chroma import add_to_memory as add_to_chroma_memory, search_memory
from app.arxiv import search_arxiv
from app.rag_qa_engine import build_index_from_text, answer_with_context
from app.rag_milvus import streamlit_memory_qa_tab
from app.milvus_engine import add_to_milvus
from PyPDF2 import PdfReader

# 🔐 API Key
st.sidebar.markdown("## 🔐 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# ✅ MODEL TESTİ
with st.sidebar.expander("🤖 GPT-4 Erişim Testi"):
    if st.button("GPT-4 Erişimini Test Et"):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Sadece çalıştığını kanıtla"}],
                max_tokens=5
            )
            st.success("✅ GPT-4 modeline erişiminiz var!")
        except AuthenticationError:
            st.error("❌ API anahtarınız geçersiz olabilir.")
        except InvalidRequestError as e:
            if "model" in str(e) and "not found" in str(e):
                st.error("🚫 GPT-4 modeline erişiminiz yok.")
            else:
                st.error(f"⚠️ Hata: {str(e)}")
        except Exception as e:
            st.error(f"❗ Beklenmeyen hata: {str(e)}")

# 🧠 ScholarMind
st.title(":brain: ScholarMind")
st.caption("Bilge araştırma hafızanız. Arayın, özetleyin, hatırlayın.")

TAB_LABELS = [
    "🔍 Makale Ara", "📌 PDF Yükle", "🔁 Geçmiş Araştırmalarım",
    "🥚 ArXiv Preprint Arama", "📖 Makaleye Soru Sor",
    "🧠 Hafızaya Dayalı Soru", "📌 PDF'yi Hafızaya Ekle"
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(TAB_LABELS)

# 🔍 Makale Arama
with tab1:
    query = st.text_input("🔍 Konu:", "transformer visual recognition")
    year = st.slider("📅 Minimum Yayın Yılı", 2000, 2024, 2020)
    limit = st.selectbox("📄 Kaç makale getirilsin?", [3, 5, 7, 10], index=1)
    run = st.button("Ara ve Özetle")

    if run:
        with st.spinner("Makaleler aranıyor..."):
            try:
                papers = search_papers(query=query, year=year, limit=limit)
            except Exception as e:
                st.error(f"📱 Semantic Scholar API hatası: {str(e)}")
                st.stop()

        if not papers:
            st.warning("❗ Aradığınız konuda sonuç bulunamadı.")
        else:
            st.success(f"{len(papers)} makale bulundu.")
            for idx, paper in enumerate(papers, 1):
                st.markdown(f"## {idx}. {paper['title']}")
                authors = ", ".join([a['name'] for a in paper['authors']])
                st.markdown(f"**Yazarlar:** {authors}  \n**Yıl:** {paper['year']}  \n**Alıntı:** {paper['citationCount']}")
                st.markdown(f"🔗 [Orijinal Makale]({paper['url']})")

                with st.spinner("Kısa özet hazırlanıyor..."):
                    try:
                        short_summary = summarize_paper({
                            "title": paper["title"],
                            "abstract": paper["abstract"]
                        }, api_key)
                        st.success(f"**Kısa Özet:** {short_summary}")
                    except Exception as e:
                        st.error(f"⚠️ Özetleme hatası: {str(e)}")

                combined_text = f"{paper['title']} - {short_summary}"
                add_text_to_index(combined_text, source_id=paper['title'], api_key=api_key)
                add_to_chroma_memory(
                    id=f"{paper['title']}_{idx}",
                    content=combined_text,
                    metadata={"source": "SemanticScholar", "title": paper['title']}
                )

                with st.expander("📜 Detaylı Özet (isteğe bağlı açılır)"):
                    with st.spinner("Detaylı özet hazırlanıyor..."):
                        detailed_prompt = f"""
Makale başlığı: {paper['title']}

Özeti: {paper['abstract']}

Bu makaleyi aşağıdaki başlıklar altında detaylıca analiz et:

1. Problem tanımı  
2. Kullanılan yöntem ve veri  
3. Sonuçlar ve katkılar  
4. Bu çalışmanın önem düzeyi

Hepsini sade ve akademik bir dille açıkla (6-10 cümle arası).
"""
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": SYSTEM_MESSAGE},
                                    {"role": "user", "content": detailed_prompt}
                                ]
                            )
                            st.info(response.choices[0].message.content.strip())
                        except Exception as e:
                            st.error(f"GPT-4 hata: {str(e)}")

                st.markdown("### 🔍 Benzer Makaleler")
                similar = search_similar(combined_text, top_k=3, api_key=api_key)
                for sim_idx, (chunk, src) in enumerate(similar, 1):
                    st.markdown(f"**{sim_idx}. ({src})**")
                    st.write(f"_{chunk[:300]}..._")

                st.markdown("### 💡 Yeni Araştırma Konu Önerileri")
                topics = suggest_topics_based_on_text(combined_text, api_key=api_key)
                st.success(topics)

                st.markdown("---")

# 🔁 Geçmiş Araştırmalarım
with tab3:
    st.subheader("🔁 Daha Önce Eklediğiniz Araştırmalar")
    search_term = st.text_input("📅 Geçmişte aradığınız bir konuyu yazın:")
    if search_term:
        with st.spinner("Geçmiş taranıyor..."):
            results = search_memory(search_term)
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                st.markdown(f"**{i+1}. {meta.get('source', 'Kaynak Yok')}**")
                st.info(doc[:500] + "...")

# 🧪 ArXiv Sekmesi
with tab4:
    st.subheader("🧪 ArXiv Preprint Arama")
    arxiv_query = st.text_input("🔍 ArXiv'te aramak istediğiniz konu:", "self-supervised learning")
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
                st.markdown(f"[🔗 ArXiv Linki]({paper['link']})")
                st.markdown("---")

# 📖 Makale Q&A Sekmesi
with tab5:
    st.subheader("📖 Yüklediğiniz makaleye soru sorun")

    uploaded_file = st.file_uploader("📌 PDF yükleyin", type=["pdf"])
    question = st.text_input("❓ Bu makaleyle ilgili ne öğrenmek istiyorsunuz?", "")

    if uploaded_file and question and st.button("🧠 Soruyu Yanıtla"):
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
                        st.success("✅ Yanıt:")
                        st.write(answer)
            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")

# 🧠 Hafızaya Dayalı Soru Sekmesi
with tab6:
    streamlit_memory_qa_tab(api_key)

# 📌 PDF'yi Milvus Hafızasına Ekle Sekmesi
with tab7:
    st.subheader("📌 PDF'yi Milvus Hafızasına Ekle")

    user_id = st.text_input("👤 Kullanıcı ID:", value="demo-user")
    uploaded_file = st.file_uploader("📎 PDF yükleyin", type=["pdf"], key="milvus_pdf")

    if uploaded_file and user_id and st.button("💾 Hafızaya Kaydet"):
        with st.spinner("PDF okunuyor ve embedding Milvus'a kaydediliyor..."):
            try:
                pdf_reader = PdfReader(uploaded_file)
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() or ""

                if len(full_text.strip()) < 100:
                    st.warning("Bu PDF'den yeterince metin çıkarılamadı.")
                else:
                    title = uploaded_file.name.replace(".pdf", "")
                    add_to_milvus(user_id=user_id, title=title, text=full_text, api_key=api_key)
                    st.success(f"✅ '{title}' başlıklı içerik hafızaya eklendi!")
            except Exception as e:
                st.error(f"Hata oluştu: {str(e)}")
