from app.milvus_engine import search_milvus
from openai import OpenAI

RAG_SYSTEM_MESSAGE = "Sen, sadece içeriklere dayanarak güvenilir ve akademik yanıtlar veren bir asistansın. Uydurma."
AAG_SYSTEM_MESSAGE = "Sen yaratıcı, açıklayıcı ve analoji temelli bir GPT asistansın. Kavramları benzetmelerle açıkla. Bilimsel ama sade örneklerle destekle."

# 🧮 RAG + AAG destekli yanıtlayıcı

def answer_question_with_memory(question: str, user_id: str, api_key: str, mode: str = "RAG", top_k: int = 3) -> str:
    """Kullanıcı tercihine göre RAG veya AAG yanıt döner."""
    top_titles = search_milvus(query=question, user_id=user_id, api_key=api_key, top_k=top_k)
    context = "\n".join([f"- {title}" for title, _ in top_titles])

    if mode == "RAG":
        system_msg = RAG_SYSTEM_MESSAGE
        user_prompt = f"""
Aşağıdaki makale başlıkları daha önceden kullanıcı tarafından eklenmiştir. Bu başlıklardaki içeriklere dayanarak soruyu yanıtla:

Makale başlıkları:
{context}

Soru:
{question}

Cevapla:
"""
    else:
        system_msg = AAG_SYSTEM_MESSAGE
        user_prompt = f"""
Kullanıcı aşağıdaki soruyu sordu. Soruyu, daha önce eklenmiş makale başlıklarına referans vererek benzetmelerle açıkla:

Makale başlıkları:
{context}

Soru:
{question}

Yaratıcı, sade ama bilimsel ve etkili bir cevap ver:
"""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()


# 🧬 Streamlit entegrasyonu için sekme kodu:

def streamlit_memory_qa_tab(api_key: str):
    import streamlit as st
    from app.rag_milvus import answer_question_with_memory

    st.subheader("🧠 Hafızaya Dayalı Soru-Cevap (Milvus + GPT-4o)")

    user_id = st.text_input("👤 Kullanıcı ID (size özgü bir ad girin):", value="demo-user")
    question = st.text_area("❓ Sormak istediğiniz soru:", height=100)

    # Kullanıcıya açıklamalı seçim sunalım
    response_mode = st.radio(
        "✍️ Yanıt tarzınızı seçin:",
        ["📚 Bilgiye Dayalı (RAG)", "🎨 Analojiyle Açıklayan (AAG)"],
        index=0
    )

    with st.expander("ℹ️ RAG ve AAG farkı nedir?"):
        st.markdown("""
**📚 Bilgiye Dayalı Yanıt (RAG):**
- Yalnızca daha önce eklediğiniz makalelerden içeriklere bakar.
- Akademik, güvenilir ve kısa yanıt verir.

**🎨 Analojiyle Açıklayan Yanıt (AAG):**
- Cevabı benzetmelerle açıklar.
- Konuyu sadeleştirerek örneklerle anlatır.
        """)

    if st.button("🚀 Yanıtla") and question and user_id:
        with st.spinner("🔍 En alakalı içerikler aranıyor ve GPT yanıtı getiriliyor..."):
            try:
                mode = "RAG" if "RAG" in response_mode else "AAG"
                result = answer_question_with_memory(
                    question=question,
                    user_id=user_id,
                    api_key=api_key,
                    mode=mode
                )
                st.success("✅ Yanıt")
                st.write(result)
            except Exception as e:
                st.error(f"❌ Hata: {str(e)}")

