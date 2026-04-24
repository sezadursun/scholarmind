# app/paper_search.py

import time
import requests
from typing import Any, Dict, List, Optional

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = "paperId,title,abstract,authors,year,citationCount,url"


def search_papers(
    query: str,
    year: Optional[int] = 2020,
    limit: int = 5,
    retries: int = 3,
    timeout: int = 20,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:

    headers = {}

    # Semantic Scholar API key varsa böyle gönderilir.
    # OpenAI API key buraya verilmemeli.
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": limit,
        "fields": DEFAULT_FIELDS,
    }

    # Semantic Scholar year filtresi
    if year is not None:
        params["year"] = f"{year}-"

    for attempt in range(retries):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code == 200:
                data = response.json().get("data", [])
                return [_normalize_paper(paper) for paper in data]

            if response.status_code in [429, 500, 502, 503, 504]:
                wait_time = 2 ** attempt
                print(f"⚠️ Semantic Scholar geçici hata {response.status_code}. {wait_time} sn sonra tekrar deneniyor...")
                time.sleep(wait_time)
                continue

            print("❌ Semantic Scholar API Error:", response.status_code, response.text)
            return []

        except requests.RequestException as e:
            wait_time = 2 ** attempt
            print(f"❌ Request failed: {e}. {wait_time} sn sonra tekrar deneniyor...")
            time.sleep(wait_time)

    return []


def _normalize_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "paper_id": paper.get("paperId"),
        "title": paper.get("title") or "Başlık yok",
        "abstract": paper.get("abstract") or "Özet bulunamadı.",
        "authors": paper.get("authors") or [],
        "year": paper.get("year") or "Yıl bilgisi yok",
        "citationCount": paper.get("citationCount") or 0,
        "url": paper.get("url") or "#",
        "source": "semantic_scholar",
    }
