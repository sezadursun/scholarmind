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
        model="gpt-4",
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

    st.subheader("\U0001f9e0 Hafızaya Dayalı Soru-Cevap (Milvus + GPT-4)")

    user_id = st.text_input("Kullanıcı ID (size özgü bir ad girin):", value="demo-user")
    question = st.text_area("Sormak istediğiniz soru:", height=100)
    response_mode = st.radio("Yanıt tarzını seçin:", ["RAG", "AAG"], index=0)

    if st.button("Yanıtla") and question and user_id:
        with st.spinner("Milvus ile en alakalı içerikler aranıyor ve GPT yanıtı getiriliyor..."):
            try:
                result = answer_question_with_memory(
                    question=question,
                    user_id=user_id,
                    api_key=api_key,
                    mode=response_mode
                )
                st.success("\u2705 Yanıt")
                st.write(result)
            except Exception as e:
                st.error(f"Hata: {str(e)}")
