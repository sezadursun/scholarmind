from openai import OpenAI
from typing import List, Dict
from app.prompts import SYSTEM_MESSAGE, SUMMARY_PROMPT_TEMPLATE

def summarize_paper(paper: Dict, api_key: str) -> str:
    """GPT ile bir makalenin kısa özetini oluşturur."""
    client = OpenAI(api_key=api_key)

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    prompt = SUMMARY_PROMPT_TEMPLATE.format(title=title, abstract=abstract)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Özetleme sırasında hata oluştu: {str(e)}"


def summarize_papers(papers: List[Dict], api_key: str) -> List[Dict]:
    """Birden fazla makaleyi özetler ve 'gpt_summary' ekler."""
    summarized = []
    for paper in papers:
        summary = summarize_paper(paper, api_key)
        paper["gpt_summary"] = summary
        summarized.append(paper)
    return summarized


def summarize_fulltext(full_text: str, api_key: str) -> str:
    """PDF gibi tam metin içerikleri akademik olarak özetler."""
    client = OpenAI(api_key=api_key)

    max_chunk = full_text[:4000]  # LLM token sınırı güvenliği için
    prompt = f"""
Aşağıda tam metni verilen bir akademik makale yer almaktadır. 
Lütfen bu metni, sade ve akademik bir dille 8-10 cümle ile özetle. Gereksiz tekrarları çıkar, temel fikri ver.

Tam Metin:
{max_chunk}

Özet:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Tam metin özetlenirken hata oluştu: {str(e)}"
