# app/paper_search.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
RELEVANCE_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = "paperId,title,abstract,authors,year,citationCount,url"


@dataclass
class PaperSearchError(RuntimeError):
    """Semantic Scholar arama hatasını kullanıcıya anlamlı biçimde taşır."""

    message: str
    status_code: Optional[int] = None
    response_text: str = ""

    def __str__(self) -> str:
        if self.status_code:
            return f"{self.message} (HTTP {self.status_code})"
        return self.message


def search_papers(
    query: str,
    year: Optional[int] = 2020,
    limit: int = 5,
    retries: int = 3,
    timeout: int = 25,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic Scholar'da akademik yayın arar.

    Önce önerilen bulk search endpoint'i kullanır. Geçici servis/rate-limit
    sorunlarında tekrar dener; bulk endpoint başarısız olursa relevance
    endpoint'ine düşer. Gerçek API hatalarını boş listeye çevirmek yerine
    PaperSearchError olarak yükseltir.
    """
    cleaned_query = " ".join((query or "").split())
    if not cleaned_query:
        raise ValueError("Arama konusu boş olamaz.")

    safe_limit = max(1, min(int(limit), 100))

    headers = {
        "Accept": "application/json",
        "User-Agent": "ScholarMind/1.0",
    }
    if api_key:
        headers["x-api-key"] = api_key.strip()

    params: Dict[str, Any] = {
        "query": cleaned_query,
        "limit": safe_limit,
        "fields": DEFAULT_FIELDS,
    }
    if year is not None:
        params["year"] = f"{int(year)}-"

    last_error: Optional[PaperSearchError] = None

    for endpoint in (BULK_SEARCH_URL, RELEVANCE_SEARCH_URL):
        try:
            payload = _request_with_retry(
                endpoint=endpoint,
                params=params,
                headers=headers,
                retries=retries,
                timeout=timeout,
            )
        except PaperSearchError as exc:
            last_error = exc
            # Bulk endpoint destek/servis sorunu yaşarsa relevance aramasını dene.
            continue

        papers = payload.get("data", [])
        if not isinstance(papers, list):
            raise PaperSearchError(
                "Semantic Scholar beklenmeyen bir yanıt döndürdü."
            )

        normalized = [_normalize_paper(paper) for paper in papers if isinstance(paper, dict)]
        return normalized[:safe_limit]

    if last_error is not None:
        raise last_error

    return []


def _request_with_retry(
    endpoint: str,
    params: Dict[str, Any],
    headers: Dict[str, str],
    retries: int,
    timeout: int,
) -> Dict[str, Any]:
    transient_statuses = {429, 500, 502, 503, 504}
    last_exception: Optional[Exception] = None

    for attempt in range(max(1, retries)):
        try:
            response = requests.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_exception = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise PaperSearchError(
                f"Semantic Scholar'a bağlanılamadı: {exc}"
            ) from exc

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise PaperSearchError(
                    "Semantic Scholar geçersiz JSON yanıtı döndürdü.",
                    status_code=response.status_code,
                    response_text=response.text[:500],
                ) from exc

        response_excerpt = (response.text or "")[:500]

        if response.status_code in transient_statuses and attempt < retries - 1:
            retry_after = response.headers.get("Retry-After")
            try:
                wait_time = float(retry_after) if retry_after else float(2 ** attempt)
            except ValueError:
                wait_time = float(2 ** attempt)
            time.sleep(min(wait_time, 10.0))
            continue

        if response.status_code == 429:
            raise PaperSearchError(
                "Semantic Scholar istek limiti aşıldı. Bir süre sonra tekrar deneyin "
                "veya SEMANTIC_SCHOLAR_API_KEY tanımlayın.",
                status_code=429,
                response_text=response_excerpt,
            )

        if response.status_code in {401, 403}:
            raise PaperSearchError(
                "Semantic Scholar API anahtarı geçersiz veya yetkisiz.",
                status_code=response.status_code,
                response_text=response_excerpt,
            )

        raise PaperSearchError(
            "Semantic Scholar araması başarısız oldu.",
            status_code=response.status_code,
            response_text=response_excerpt,
        )

    if last_exception:
        raise PaperSearchError(str(last_exception))

    raise PaperSearchError("Semantic Scholar araması tamamlanamadı.")


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
